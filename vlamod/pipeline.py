"""한 타임스텝을 통째로 분석하는 파이프라인.

관찰(비율) + 개입(KL) 을 한 번에 내서 tidy row 로 반환합니다.
H2 검정(비율 vs 인과기여 상관)이 이 row 들만 있으면 바로 됩니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import capture as cap
from . import intervene as IV
from . import metrics as M
from . import token_index as TI


def libero_env_action(action) -> np.ndarray:
    """OpenVLA 출력 → LIBERO `env.step()` 입력 규약으로 변환.

    !! 이걸 빼먹으면 **팔은 움직이는데 그리퍼가 반대로 작동**해서 성공률이 0% 가 됩니다.

    OpenVLA 는 LIBERO 데이터로 파인튜닝할 때 dataloader 가 그리퍼 부호를 뒤집어
    (0 = 닫기, 1 = 열기) 학습했습니다. 그래서 `predict_action` 의 그리퍼 출력은
    [0, 1] 범위이고, LIBERO env 는 [-1, +1] (−1 = 열기, +1 = 닫기) 를 기대합니다.

    OpenVLA 공식 평가 코드(run_libero_eval.py)가 하는 두 단계를 그대로 재현합니다:
        normalize_gripper_action(binarize=True):  [0,1] → [-1,+1] → sign
        invert_gripper_action:                    부호 반전

    | 모델 의도 | 모델 출력 | 변환 전 env 동작 | 변환 후 env 동작 |
    |-----------|-----------|------------------|------------------|
    | 열기      | ≈ 1       | +1 → **닫기** ❌ | −1 → 열기 ✅     |
    | 닫기      | ≈ 0       | ≈0 → 무동작 ❌   | +1 → 닫기 ✅     |

    앞 6차원(EE 델타 pose)은 건드리지 않습니다.
    """
    a = np.asarray(action, dtype=np.float64).copy()
    a[..., -1] = 2.0 * (a[..., -1] - 0.0) / (1.0 - 0.0) - 1.0   # [0,1] → [-1,+1]
    a[..., -1] = np.sign(a[..., -1])                             # binarize
    a[..., -1] = a[..., -1] * -1.0                               # invert
    return a


@torch.no_grad()
def policy_action(vla, image, instruction: str, unnorm_key: str,
                  env_convention: bool = True):
    """정책이 내는 **연속 action 7차원**. 환경을 한 스텝 진행시킬 때 씁니다.

    !! OpenVLA 의 `predict_action` 은 내부에서 input_ids 끝에 토큰 하나(id 29871,
       Llama 의 빈 토큰)를 덧붙입니다:

           input_ids = torch.cat((input_ids, torch.Tensor([29871]).long()...), dim=1)

       그런데 processor 가 준 `attention_mask` 는 **덧붙이기 전 길이**입니다.
       그대로 넘기면 멀티모달 마스크(256 + 34 = 290)와 실제 임베딩(256 + 35 = 291)이
       1 어긋나서 이렇게 죽습니다:

           RuntimeError: The size of tensor a (291) must match
                         the size of tensor b (290) at non-singleton dimension 3

       `attention_mask` 를 빼고 넘기면 generate 가 올바른 길이로 새로 만듭니다.
       (2026-10-06 부터는 `TI.prepare_inputs` 가 29871 과 attention_mask 를 같이 붙이므로
        predict_action 이 다시 붙이지 않습니다. attention_mask 제거는 안전장치로 남겨 둡니다.)
    """
    prompt, _ = TI.build_prompt(instruction)
    inputs = TI.prepare_inputs(vla, prompt, image)   # 29871 포함 — 정책과 같은 입력
    inputs = {k: v for k, v in inputs.items() if k != "attention_mask"}
    raw = vla.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    # env_convention=True(기본): LIBERO env 에 바로 넣을 수 있게 그리퍼 규약 변환.
    # 이 함수는 **env 를 진행시키는 용도로만** 쓰이므로 기본값이 True 가 맞습니다.
    # (attention 분석은 teacher_forced_capture 가 하고, 이 출력과 무관합니다)
    return libero_env_action(raw) if env_convention else np.asarray(raw)


@torch.no_grad()
def analyze_step(
    vla,
    image,
    instruction: str,
    sink_positions=(0,),
    instruction_only: bool = True,
    n_action: int = 7,
    visual_span: tuple[int, int] | None = None,
    do_intervene: bool = True,
    control_seed: int = 0,
    layers_subset=None,
) -> dict[str, Any]:
    """반환: 스칼라 dict (CSV 한 줄) + 'per_layer' 키에 층별 배열."""
    prompt, instr_span = TI.build_prompt(instruction)
    inputs = TI.prepare_inputs(vla, prompt, image)   # 29871 포함 — 정책과 같은 입력

    if visual_span is None:
        raise ValueError(
            "visual_span 을 넘겨주세요. 매 스텝 probe 하면 느립니다. "
            "01_smoke_forward.py 에서 한 번 실측한 값을 재사용하세요."
        )

    spans = TI.build_spans(
        vla,
        prompt=prompt,
        instr_char_span=instr_span,
        input_ids=inputs["input_ids"][0],
        visual_span=visual_span,
        n_action_tokens=n_action,
        sink_positions=sink_positions,
        instruction_only=instruction_only,
    )

    base = cap.generate_action_tokens(vla, inputs, n_action=n_action)
    c0 = cap.teacher_forced_capture(vla, inputs, base, spans.query)
    r = M.compute_ratios(c0.attn, c0.vnorm, spans.visual, spans.language, spans.sink)

    row: dict[str, Any] = {
        "instruction": instruction,
        "n_L": len(spans.language),
        "n_V": len(spans.visual),
        "uniform_baseline": r.uniform_baseline,
        "R_raw": float(r.r_raw.mean()),
        "R_norm": float(r.r_norm.mean()),
        "R_vnorm": float(r.r_vnorm.mean()),
        "mass_sink": float(r.mass_sink.mean()),
        "mass_other": float(r.mass_other.mean()),
        "action_bins": M.decode_actions(c0.action_logits).tolist(),
    }

    # 층별 평균 (head 평균) — 나중에 히트맵/층별 분석용
    per_layer = {
        "R_raw": r.r_raw.mean(dim=(1, 2)).tolist(),
        "R_norm": r.r_norm.mean(dim=(1, 2)).tolist(),
        "R_vnorm": r.r_vnorm.mean(dim=(1, 2)).tolist(),
    }
    # action step 별 (DoF 별 언어 의존이 다른지)
    row.update(
        {f"R_norm_step{k}": float(r.r_norm[:, :, k].mean()) for k in range(r.r_norm.shape[2])}
    )

    if do_intervene:
        ko_l = IV.knockout_language(spans, layers=layers_subset)
        ko_v = IV.knockout_vision(spans, layers=layers_subset)
        ko_c = IV.knockout_random_control(spans, seed=control_seed, layers=layers_subset)

        c_l = cap.teacher_forced_capture(vla, inputs, base, spans.query, attn_mask_fn=ko_l.as_fn())
        c_v = cap.teacher_forced_capture(vla, inputs, base, spans.query, attn_mask_fn=ko_v.as_fn())
        c_c = cap.teacher_forced_capture(vla, inputs, base, spans.query, attn_mask_fn=ko_c.as_fn())

        kl_l = float(M.action_kl(c0.action_logits, c_l.action_logits))
        kl_v = float(M.action_kl(c0.action_logits, c_v.action_logits))
        kl_c = float(M.action_kl(c0.action_logits, c_c.action_logits))

        row.update(
            {
                "KL_lang_knockout": kl_l,
                "KL_vis_knockout": kl_v,
                "KL_control_knockout": kl_c,
                "causal_ratio": M.causal_ratio(kl_l, kl_v),
                # 언어 knockout 이 '같은 개수 랜덤 visual 차단'보다 큰가.
                # 1보다 커야 언어가 특별하다고 말할 수 있습니다.
                "lang_vs_control": kl_l / kl_c if kl_c > 0 else float("nan"),
                # !! lang_vs_control 은 **분모가 0 에 가까워 매우 불안정**합니다.
                #    실측: KL_control 이 0.036~0.561 로 흔들리자 비율이 12~218 배로
                #    튀었습니다 (CV 100%). 비율 자체를 통계에 쓰면 안 됩니다.
                #    → 대조군을 '바닥값'으로 보고 **빼는** 쪽이 안정적입니다.
                "lang_minus_control": kl_l - kl_c,
                "vis_minus_control": kl_v - kl_c,
                # 대조군을 뺀 뒤의 인과 비율. causal_ratio 의 보정판.
                "causal_ratio_adj": M.causal_ratio(max(kl_l - kl_c, 0.0),
                                                   max(kl_v - kl_c, 0.0)),
                "argmax_changed_lang": int(
                    (M.decode_actions(c0.action_logits) != M.decode_actions(c_l.action_logits))
                    .sum()
                ),
                "argmax_changed_vis": int(
                    (M.decode_actions(c0.action_logits) != M.decode_actions(c_v.action_logits))
                    .sum()
                ),
            }
        )

    row["_per_layer"] = per_layer
    return row


@torch.no_grad()
def counterfactual_instruction(
    vla,
    image,
    instruction_a: str,
    instruction_b: str,
    visual_span: tuple[int, int],
    n_action: int = 7,
) -> dict[str, Any]:
    """ISS 식 개입: 같은 장면, 지시만 교체.

    0벡터/빈 문자열이 아니라 **같은 분포의 다른 지시문**으로 바꾸는 것이 핵심입니다.
    (빈 입력은 OOD 라서 '언어를 안 쓴다'가 아니라 '이상한 입력이라 망가졌다'가 됩니다)
    """
    outs = {}
    for tag, instr in (("a", instruction_a), ("b", instruction_b)):
        prompt, span = TI.build_prompt(instr)
        inputs = TI.prepare_inputs(vla, prompt, image)   # 29871 포함 — 정책과 같은 입력
        spans = TI.build_spans(
            vla, prompt, span, inputs["input_ids"][0], visual_span, n_action_tokens=n_action
        )
        ids = cap.generate_action_tokens(vla, inputs, n_action=n_action)
        c = cap.teacher_forced_capture(vla, inputs, ids, spans.query)
        outs[tag] = c

    kl = float(M.action_kl(outs["a"].action_logits, outs["b"].action_logits))
    same = int(
        (M.decode_actions(outs["a"].action_logits) == M.decode_actions(outs["b"].action_logits)).sum()
    )
    return {
        "instruction_a": instruction_a,
        "instruction_b": instruction_b,
        "KL_instruction_swap": kl,
        "n_identical_dof": same,   # 7 이면 지시를 완전히 무시한 것
    }
