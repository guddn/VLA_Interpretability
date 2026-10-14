"""[L2] 토큰 임베딩 치환 — 0층 입력의 특정 위치를 기준값(baseline) 벡터로 교체.

knockout([L3], `intervene.py`)과 무엇이 다른가
---------------------------------------------
knockout 은 **Q 행 → 대상 key** attention 만 막습니다. 다른 위치(예: L 뒤의 템플릿
토큰, 이미지 정보를 읽어 온 L 토큰)를 거쳐 우회하는 경로는 열려 있습니다
→ **직접 효과(direct effect)**.

치환은 그 위치의 **내용 자체**를 0층에서 바꿉니다. 이후 모든 층·모든 위치가 바뀐
내용을 보므로 우회 경로까지 함께 끊깁니다 → **총 효과(total effect)**.
위치(RoPE)와 시퀀스 길이·attention 구조는 그대로입니다.

왜 0층 pre-hook 인가
--------------------
OpenVLA(Prismatic) 는 `[BOS 임베딩 | projector(패치) | 나머지 텍스트 임베딩]` 을 이어 붙여
Llama 의 `inputs_embeds` 로 넘깁니다. Llama 는 위치 임베딩을 더하지 않으므로(RoPE 는
attention 안에서 적용) **0번 decoder layer 의 입력 hidden_states = 멀티모달 임베딩** 입니다.
여기에 hook 을 걸면 텍스트·이미지 위치를 같은 절대 위치(spans.*)로 다룰 수 있고,
Prismatic 내부 속성 이름(projector 등)에 의존하지 않습니다.

!! 실제 토큰 임베딩으로 치환하면 0층에서는 **문자열을 바꾼 것(04)·이미지를 바꾼 것(06)과
   수학적으로 같습니다.** 그래서 기준값은 실제 토큰이 아닌 **평균(mean)** 을 씁니다.
"""

from __future__ import annotations

import re
from typing import Sequence

import torch


class _StopForward(Exception):
    """기록만 하고 나머지 31개 층 계산을 건너뛰기 위한 내부 신호."""


# ---------------------------------------------------------------------
# 순수 함수 (GPU/모델 없이 테스트 가능)
# ---------------------------------------------------------------------
def replace_positions(hidden: torch.Tensor, positions: Sequence[int],
                      vectors: torch.Tensor) -> torch.Tensor:
    """hidden [B, T, D] 의 positions 행을 vectors [len(positions), D] 로 바꾼 **복사본**."""
    if hidden.dim() != 3:
        raise ValueError(f"hidden 은 [B,T,D] 여야 합니다. got {tuple(hidden.shape)}")
    pos = list(positions)
    if vectors.dim() != 2 or vectors.shape[0] != len(pos) or vectors.shape[1] != hidden.shape[-1]:
        raise ValueError(f"vectors shape {tuple(vectors.shape)} ≠ ({len(pos)}, {hidden.shape[-1]})")
    if pos and (min(pos) < 0 or max(pos) >= hidden.shape[1]):
        raise IndexError(f"치환 위치 {min(pos)}..{max(pos)} 가 시퀀스 길이 {hidden.shape[1]} 밖입니다. "
                         "KV cache 를 쓰는 generate 단계에 hook 이 걸린 건 아닌지 확인하세요.")
    out = hidden.clone()
    if pos:
        idx = torch.tensor(pos, dtype=torch.long, device=hidden.device)
        out[:, idx, :] = vectors.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
    return out


def rescale_to_norm(vectors: torch.Tensor, target_norm: float) -> torch.Tensor:
    """각 행의 방향은 유지하고 L2 크기를 target_norm 으로 맞춤 (mean_rescaled 기준값)."""
    v = vectors.float()
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / n * float(target_norm)


def text_index_to_pos(t: int, visual_span: tuple[int, int]) -> int:
    """텍스트 토큰 인덱스 t → 멀티모달 절대 위치 (token_index.build_spans 와 같은 규칙)."""
    v_start, v_end = visual_span
    return t if t < v_start else t + (v_end - v_start)


def check_alignment(layer0_input: torch.Tensor, text_embeds: torch.Tensor,
                    visual_span: tuple[int, int], atol: float = 1e-6) -> None:
    """G1: 0층 입력에서 **V 가 아닌 모든 위치**가 텍스트 임베딩과 순서대로 일치하는지.

    layer0_input : [T, D]       (T = n_text + n_visual)
    text_embeds  : [n_text, D]  embed_tokens(full_ids)
    텍스트 위치가 전부 맞으면 소거법으로 남은 n_visual 개가 정확히 visual_span 입니다.
    """
    v_start, v_end = visual_span
    n_vis = v_end - v_start
    T, n_text = layer0_input.shape[0], text_embeds.shape[0]
    if T != n_text + n_vis:
        raise RuntimeError(f"[G1] 0층 입력 길이 {T} ≠ 텍스트 {n_text} + 이미지 {n_vis}")
    pos = torch.tensor([text_index_to_pos(t, visual_span) for t in range(n_text)])
    diff = (layer0_input[pos].float() - text_embeds.float()).abs().amax(dim=-1)
    bad = (diff > atol).nonzero().flatten().tolist()
    if bad:
        raise RuntimeError(f"[G1] 텍스트 위치 {[int(pos[b]) for b in bad[:8]]} 의 0층 입력이 "
                           f"임베딩과 다릅니다 (최대 차이 {float(diff.max()):.3g}). "
                           "visual 삽입 위치 가정(BOS 뒤 256개)이 틀렸을 수 있습니다.")


def overlapping_positions(pos_char: dict[int, tuple[int, int]],
                          char_range: tuple[int, int]) -> list[int]:
    """{절대 위치: (문자 시작, 끝)} 중 char_range 와 겹치는 위치 (공백 제거 후 판정은 호출 쪽)."""
    a, b = char_range
    return sorted(p for p, (s, e) in pos_char.items() if s < b and e > a)


_SPATIAL_RE = re.compile(r"^pick up the black bowl (?P<loc>.+?) and place it on the plate$")


def spatial_location_span(instruction: str) -> tuple[int, int] | None:
    """LIBERO-spatial 지시문에서 task 를 구분하는 **위치구** 의 문자 구간 (지시문 기준).

    'pick up the black bowl [위치구] and place it on the plate' 형태가 아니면 None.
    """
    m = _SPATIAL_RE.match(instruction.lower().strip().rstrip("."))
    return (m.start("loc"), m.end("loc")) if m else None


# ---------------------------------------------------------------------
# hook
# ---------------------------------------------------------------------
def _get_hidden(args, kwargs):
    if args:
        return args[0], True
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"], False
    raise RuntimeError("0층 입력 hidden_states 를 찾지 못했습니다. LlamaDecoderLayer 시그니처를 확인하세요.")


class EmbeddingSubstitution:
    """0층 입력의 positions 를 vectors 로 교체. `teacher_forced_capture(attn_mask_fn=...)` 에 그대로 넣습니다."""

    def __init__(self, positions: Sequence[int], vectors: torch.Tensor, layer: int = 0):
        self.positions = list(positions)
        self.vectors = vectors.detach().float().cpu()
        if self.vectors.shape[0] != len(self.positions):
            raise ValueError("positions 와 vectors 개수가 다릅니다.")
        self.layer = layer
        self.n_calls = 0

    def _hook(self, module, args, kwargs):  # noqa: ANN001
        hidden, positional = _get_hidden(args, kwargs)
        new = replace_positions(hidden, self.positions, self.vectors)
        self.n_calls += 1
        if positional:
            return (new,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = new
        return args, kwargs

    def attach(self, vla) -> list:
        return [vla.decoder_layers[self.layer].register_forward_pre_hook(self._hook, with_kwargs=True)]

    def as_fn(self):
        return lambda vla: self.attach(vla)


class Layer0Recorder:
    """0층 입력 hidden_states[0] 을 [T, D] float32 cpu 로 기록. stop=True 면 기록 후 forward 중단."""

    def __init__(self, stop: bool = False):
        self.stop = stop
        self.hidden: torch.Tensor | None = None

    def _hook(self, module, args, kwargs):  # noqa: ANN001
        hidden, _ = _get_hidden(args, kwargs)
        self.hidden = hidden[0].detach().float().cpu()
        if self.stop:
            raise _StopForward
        return None

    def attach(self, vla) -> list:
        return [vla.decoder_layers[0].register_forward_pre_hook(self._hook, with_kwargs=True)]

    def as_fn(self):
        return lambda vla: self.attach(vla)


@torch.no_grad()
def record_layer0(vla, inputs) -> torch.Tensor:
    """프롬프트만으로 0층 입력을 기록 (나머지 층은 계산하지 않음). 반환 [T, D] float32 cpu."""
    rec = Layer0Recorder(stop=True)
    handles = rec.attach(vla)
    try:
        vla.model(**inputs, return_dict=True)
    except _StopForward:
        pass
    finally:
        for h in handles:
            h.remove()
    if rec.hidden is None:
        raise RuntimeError("0층 hook 이 호출되지 않았습니다.")
    return rec.hidden


# ---------------------------------------------------------------------
# 기준값 계산
# ---------------------------------------------------------------------
@torch.no_grad()
def language_token_ids(vla, instruction: str, visual_span, sink_positions=(0,),
                       instruction_only: bool = True) -> list[int]:
    """지시문의 L 구간 토큰 id (분석과 같은 spans 규칙)."""
    from . import token_index as TI
    prompt, span = TI.build_prompt(instruction)
    ids = vla.tokenizer(prompt, add_special_tokens=True)["input_ids"]
    ids = list(ids) + [TI.ACTION_PREFIX_TOKEN_ID]
    spans = TI.build_spans(vla, prompt, span, torch.tensor(ids), visual_span,
                           n_action_tokens=7, sink_positions=sink_positions,
                           instruction_only=instruction_only)
    n_vis = visual_span[1] - visual_span[0]
    return [ids[p if p < visual_span[0] else p - n_vis] for p in spans.language]


@torch.no_grad()
def language_mean(vla, instructions: Sequence[str], visual_span, **kw) -> tuple[torch.Tensor, list[int]]:
    """suite 지시문들의 L 토큰 임베딩 **빈도 가중 평균** [D] (= L 자리 임베딩의 기댓값)."""
    all_ids: list[int] = []
    for ins in instructions:
        all_ids += language_token_ids(vla, ins, visual_span, **kw)
    E = vla.language_model.get_input_embeddings().weight
    idx = torch.tensor(all_ids, dtype=torch.long, device=E.device)
    return E[idx].float().mean(dim=0).cpu(), all_ids
