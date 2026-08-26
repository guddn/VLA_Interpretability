"""Phase 1: 토큰 구간 인덱싱.

이 파일이 틀리면 그 뒤의 모든 숫자가 무의미해집니다. 그래서 전부 **실측 + 검증** 합니다.

멀티모달 시퀀스 레이아웃 (Prismatic/OpenVLA):

    [BOS] [visual ×N] [prompt 텍스트 토큰들...] [action ×7]
      0    1 .. N        N+1 .. N+T-1            N+T ..

- visual 토큰은 BOS **직후**(index 1)에 삽입된다고 알려져 있으나,
  이 모듈은 그것을 가정하지 않고 `probe_visual_span()` 으로 **실험적으로 확인**합니다.
- 언어 구간 L 은 템플릿 고정 문구를 제외한 **instruction 부분만** 잡습니다.
  ("In: What action should the robot take to" 는 모든 샘플에 동일하므로
   여기에 걸린 attention 은 지시 내용과 무관합니다.)
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import torch

# OpenVLA 공식 프롬프트 템플릿 (README 기준).
# !! 체크포인트마다 다를 수 있습니다. scripts/01_smoke_forward.py 가
#    실제 프롬프트를 출력해 주니 눈으로 대조하세요.
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def build_prompt(instruction: str) -> tuple[str, tuple[int, int]]:
    """프롬프트 문자열과, 그 안에서 instruction 이 차지하는 (start, end) 문자 오프셋."""
    instr = instruction.lower().strip().rstrip(".")
    prefix = PROMPT_TEMPLATE.split("{instruction}")[0]
    prompt = PROMPT_TEMPLATE.format(instruction=instr)
    start = len(prefix)
    return prompt, (start, start + len(instr))


@dataclasses.dataclass
class TokenSpans:
    """멀티모달 시퀀스에서 각 구간의 **절대 위치** 인덱스."""

    n_total_prompt: int          # action token 붙이기 전 길이
    n_visual: int
    visual: list[int]            # V
    language: list[int]          # L  (instruction 내용어만)
    template: list[int]          # 템플릿 고정 문구 (분석에서 보통 제외)
    sink: list[int]              # S  (BOS 등)
    action: list[int]            # A  (teacher-forced 로 붙인 action token)
    text_token_strings: list[str]  # 디버깅용

    def summary(self) -> str:
        return (
            f"total(prompt)={self.n_total_prompt}  "
            f"|V|={len(self.visual)}  |L|={len(self.language)}  "
            f"|template|={len(self.template)}  |S|={len(self.sink)}  |A|={len(self.action)}"
        )


# ---------------------------------------------------------------------
# 1. visual span 을 실험적으로 확인
# ---------------------------------------------------------------------
@torch.no_grad()
def probe_visual_span(vla, inputs_a, inputs_b) -> tuple[int, int]:
    """이미지만 다른 두 입력의 첫 층 hidden state 를 비교해 visual 구간을 찾습니다.

    이미지가 바뀌면 visual 토큰 위치의 임베딩만 달라져야 합니다.
    텍스트 토큰 위치가 같이 달라지면(= causal attention 때문에 뒤쪽은 달라짐)
    **앞쪽부터 연속으로 달라지는 첫 블록**이 visual 구간입니다.

    반환: (start, end) — end 는 exclusive
    """
    out_a = vla.model(**inputs_a, output_hidden_states=True, return_dict=True)
    out_b = vla.model(**inputs_b, output_hidden_states=True, return_dict=True)

    # embedding 층(hidden_states[0])은 attention 을 거치지 않았으므로
    # 정확히 visual 위치에서만 달라집니다.
    h_a = out_a.hidden_states[0][0].float()
    h_b = out_b.hidden_states[0][0].float()
    diff = (h_a - h_b).abs().sum(dim=-1)  # [T]
    changed = (diff > 1e-4).nonzero(as_tuple=True)[0].tolist()

    if not changed:
        raise RuntimeError(
            "두 이미지에 대해 embedding 이 전혀 다르지 않습니다. "
            "정말 다른 이미지를 넣었는지, pixel_values 가 반영되는지 확인하세요."
        )

    start, end = changed[0], changed[-1] + 1
    if len(changed) != end - start:
        raise RuntimeError(
            f"visual 구간이 연속적이지 않습니다: {changed[:5]} ... {changed[-5:]}. "
            "모델 구조가 예상과 다릅니다. 직접 확인하세요."
        )
    return start, end


# ---------------------------------------------------------------------
# 2. 텍스트 토큰 → 문자 오프셋
# ---------------------------------------------------------------------
def token_char_offsets(tokenizer, prompt: str, input_ids: Sequence[int]) -> list[tuple[int, int]]:
    """각 텍스트 토큰의 (문자 시작, 끝). fast tokenizer 가 아니면 수동 복원."""
    try:
        enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc["offset_mapping"]
        if len(offsets) == len(input_ids):
            return [tuple(o) for o in offsets]
    except Exception:  # noqa: BLE001
        pass

    # fallback: 토큰을 하나씩 디코드하며 누적 길이로 오프셋 복원
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tid in input_ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
        if piece == "" or piece in (tokenizer.bos_token, tokenizer.eos_token):
            offsets.append((cursor, cursor))
            continue
        idx = prompt.find(piece.strip(), cursor) if piece.strip() else -1
        if idx < 0:
            offsets.append((cursor, cursor))
        else:
            offsets.append((idx, idx + len(piece.strip())))
            cursor = idx + len(piece.strip())
    return offsets


# ---------------------------------------------------------------------
# 3. 전체 구간 만들기
# ---------------------------------------------------------------------
def build_spans(
    vla,
    prompt: str,
    instr_char_span: tuple[int, int],
    input_ids: torch.Tensor,
    visual_span: tuple[int, int],
    n_action_tokens: int = 7,
    sink_positions: Sequence[int] = (0,),
    instruction_only: bool = True,
) -> TokenSpans:
    """텍스트 토큰 인덱스를 멀티모달 절대 위치로 매핑하고 구간을 나눕니다.

    input_ids : [T_text]  (텍스트만, visual 삽입 전)
    visual_span: probe_visual_span() 결과 (start, end)
    """
    tokenizer = vla.tokenizer
    ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
    n_text = len(ids)
    v_start, v_end = visual_span
    n_visual = v_end - v_start

    # 텍스트 토큰 t 의 절대 위치
    def abs_pos(t: int) -> int:
        return t if t < v_start else t + n_visual

    offsets = token_char_offsets(tokenizer, prompt, ids)
    if len(offsets) != n_text:
        raise RuntimeError(f"offset 개수({len(offsets)}) != 텍스트 토큰 수({n_text})")

    i_start, i_end = instr_char_span
    language, template = [], []
    for t, (cs, ce) in enumerate(offsets):
        p = abs_pos(t)
        if p in set(sink_positions):
            continue
        if cs == ce:  # 특수 토큰 등
            template.append(p)
            continue
        # instruction 문자 구간과 겹치면 L
        if instruction_only:
            in_instr = (cs >= i_start) and (ce <= i_end)
        else:
            in_instr = True
        (language if in_instr else template).append(p)

    visual = list(range(v_start, v_end))
    n_total_prompt = n_text + n_visual
    action = list(range(n_total_prompt, n_total_prompt + n_action_tokens))

    if not language:
        raise RuntimeError(
            "언어 구간 L 이 비었습니다. instruction 문자 오프셋 매핑이 실패했습니다. "
            "PROMPT_TEMPLATE 이 실제 체크포인트의 템플릿과 같은지 확인하세요."
        )

    return TokenSpans(
        n_total_prompt=n_total_prompt,
        n_visual=n_visual,
        visual=visual,
        language=language,
        template=template,
        sink=[p for p in sink_positions if p < n_total_prompt],
        action=action,
        text_token_strings=[tokenizer.decode([i], skip_special_tokens=False) for i in ids],
    )


def pretty_print_spans(spans: TokenSpans, max_show: int = 40) -> str:
    """사람이 눈으로 검증하기 위한 출력. Phase 1 에서 반드시 확인하세요."""
    lines = [spans.summary(), ""]
    lset = set(spans.language)
    tset = set(spans.template)
    sset = set(spans.sink)
    v_start = spans.visual[0] if spans.visual else -1
    v_end = spans.visual[-1] + 1 if spans.visual else -1

    lines.append(f"  [{v_start:4d}..{v_end - 1:4d}]  V  (visual ×{spans.n_visual})")
    shown = 0
    for t, s in enumerate(spans.text_token_strings):
        p = t if t < v_start else t + spans.n_visual
        tag = "S" if p in sset else "L" if p in lset else "T" if p in tset else "?"
        lines.append(f"  [{p:4d}]        {tag}  {s!r}")
        shown += 1
        if shown >= max_show:
            lines.append("  ... (생략)")
            break
    return "\n".join(lines)
