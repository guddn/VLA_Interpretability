"""한 타임스텝을 통째로 분석하는 파이프라인.

관찰(비율) + 개입(KL) 을 한 번에 내서 tidy row 로 반환합니다.
H2 검정(비율 vs 인과기여 상관)이 이 row 들만 있으면 바로 됩니다.
"""

from __future__ import annotations

from typing import Any

import torch

from . import capture as cap
from . import intervene as IV
from . import metrics as M
from . import token_index as TI


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
    inputs = vla.processor(prompt, image).to(vla.device, dtype=vla.dtype)

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
    c0 = cap.teacher_forced_capture(vla, inputs, base, spans.action)
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

        c_l = cap.teacher_forced_capture(vla, inputs, base, spans.action, attn_mask_fn=ko_l.as_fn())
        c_v = cap.teacher_forced_capture(vla, inputs, base, spans.action, attn_mask_fn=ko_v.as_fn())
        c_c = cap.teacher_forced_capture(vla, inputs, base, spans.action, attn_mask_fn=ko_c.as_fn())

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
        inputs = vla.processor(prompt, image).to(vla.device, dtype=vla.dtype)
        spans = TI.build_spans(
            vla, prompt, span, inputs["input_ids"][0], visual_span, n_action_tokens=n_action
        )
        ids = cap.generate_action_tokens(vla, inputs, n_action=n_action)
        c = cap.teacher_forced_capture(vla, inputs, ids, spans.action)
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
