"""Phase 4 (H1): 비율이 '언어가 필요한 정도'에 따라 조절되는가?

조건 설계
---------
같은 장면·같은 타임스텝에서 지시문만 4가지로 바꿉니다:
  valid     : 원래 지시
  swapped   : 같은 suite 의 **다른 task 지시** (모순 지시)
  paraphrase: 동의어 치환 (의미 동일, 표면 다름)
  empty     : 빈 지시  ← 2510.13626 재현용. 단 OOD 이므로 해석 주의

측정
----
  R_norm / R_vnorm  : 조건에 따라 비율이 움직이는가
  KL_vs_valid       : 행동 분포가 실제로 바뀌는가
  n_identical_dof   : 7이면 지시를 완전히 무시

H1 예측: valid ↔ swapped 사이에서 **R 은 거의 안 움직이는데 KL 도 작다**
         → "비율이 demand 에 반응하지 않는다"

주의 (계획서 §1.2)
------------------
LIBERO 는 장면이 task 를 거의 결정합니다. 낮은 언어 의존이 모델 결함인지
벤치마크 특성인지 구분하려면 **한 장면에 후보가 둘 이상인 조건**이 필요합니다.
--ambiguous-tasks 로 그런 task id 를 직접 지정하세요 (없으면 결론을 약하게 쓸 것).

사용:
    python scripts/04_counterfactual.py --suite object --tasks 0 1 2 --episodes 2
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod import capture as cap  # noqa: E402
from vlamod import env_libero as EL  # noqa: E402
from vlamod import metrics as M  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.model_loader import load_openvla  # noqa: E402

PARAPHRASE = [
    ("pick up", "grasp"),
    ("place", "put"),
    ("put", "place"),
    ("move", "shift"),
    ("close", "shut"),
    ("open", "unclose"),
]


def paraphrase(instr: str) -> str:
    out = instr
    for a, b in PARAPHRASE:
        if a in out:
            return out.replace(a, b, 1)
    return "please " + out


@torch.no_grad()
def one_condition(vla, image, instruction, vspan, cfg):
    prompt, span = TI.build_prompt(instruction) if instruction.strip() else TI.build_prompt(" ")
    inputs = vla.processor(prompt, image).to(vla.device, dtype=vla.dtype)
    spans = TI.build_spans(
        vla, prompt, span, inputs["input_ids"][0], vspan,
        n_action_tokens=7,
        sink_positions=cfg["tokens"]["sink_positions"],
        instruction_only=cfg["tokens"]["instruction_only"],
    )
    ids = cap.generate_action_tokens(vla, inputs, n_action=7)
    c = cap.teacher_forced_capture(vla, inputs, ids, spans.action)
    r = M.compute_ratios(c.attn, c.vnorm, spans.visual, spans.language, spans.sink)
    return c, r, spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--suite", default="object")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 25, 40])
    ap.add_argument("--ambiguous-tasks", type=int, nargs="*", default=[],
                    help="장면에 후보 물체가 둘 이상인 task id (직접 확인해서 지정)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mcfg = cfg["model"]
    vla = load_openvla(path=mcfg["path"], device=mcfg["device"], dtype=mcfg["dtype"],
                       attn_implementation=mcfg["attn_implementation"])

    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(0)
    ia = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    ib = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    p0, _ = TI.build_prompt("dummy")
    vspan = TI.probe_visual_span(
        vla,
        vla.processor(p0, ia).to(vla.device, dtype=vla.dtype),
        vla.processor(p0, ib).to(vla.device, dtype=vla.dtype),
    )

    # 모든 task 의 지시문을 먼저 모아 둡니다 (swap 재료)
    tasks = {tid: EL.make_task(args.suite, tid) for tid in args.tasks}
    instrs = {tid: t.instruction for tid, t in tasks.items()}
    print("지시문 목록:")
    for tid, s in instrs.items():
        print(f"  [{tid}] {s!r}")

    rows = []
    for tid, task in tasks.items():
        swap_pool = [s for k, s in instrs.items() if k != tid]
        for ep in range(args.episodes):
            EL.reset_to(task, ep)
            obs = EL.step_noop(task, 10)
            step_ptr = 0
            for t in range(max(args.steps) + 1):
                if t in args.steps:
                    img = EL.obs_to_image(obs)
                    conds = {
                        "valid": instrs[tid],
                        "swapped": swap_pool[step_ptr % len(swap_pool)] if swap_pool else instrs[tid],
                        "paraphrase": paraphrase(instrs[tid]),
                        "empty": " ",
                    }
                    step_ptr += 1
                    base = None
                    for name, instr in conds.items():
                        c, r, _ = one_condition(vla, img, instr, vspan, cfg)
                        if name == "valid":
                            base = c
                        kl = float(M.action_kl(base.action_logits, c.action_logits))
                        same = int(
                            (M.decode_actions(base.action_logits)
                             == M.decode_actions(c.action_logits)).sum()
                        )
                        rows.append({
                            "suite": args.suite, "task_id": tid, "episode": ep, "t": t,
                            "condition": name, "instruction": instr,
                            "ambiguous": tid in args.ambiguous_tasks,
                            "R_raw": float(r.r_raw.mean()),
                            "R_norm": float(r.r_norm.mean()),
                            "R_vnorm": float(r.r_vnorm.mean()),
                            "uniform_baseline": r.uniform_baseline,
                            "KL_vs_valid": kl,
                            "n_identical_dof": same,
                        })
                action = vla.model.predict_action(
                    **vla.processor(TI.build_prompt(instrs[tid])[0], EL.obs_to_image(obs))
                    .to(vla.device, dtype=vla.dtype),
                    unnorm_key=mcfg["unnorm_key"], do_sample=False,
                )
                obs, _, done, _ = task.env.step(action.tolist())
                if done:
                    break
        task.env.close()

    df = pd.DataFrame(rows)
    out = f"outputs/counterfactual_{args.suite}.csv"
    df.to_csv(out, index=False)
    print(f"\n저장: {out}  ({len(df)} rows)")

    print("\n--- 조건별 요약 ---")
    print(df.groupby("condition")[["R_raw", "R_norm", "R_vnorm", "KL_vs_valid",
                                   "n_identical_dof"]].mean().round(4).to_string())

    print("\n--- H1 해석 ---")
    g = df.groupby("condition")["R_norm"].mean()
    if "valid" in g and "swapped" in g:
        d = abs(g["valid"] - g["swapped"])
        print(f"R_norm(valid) - R_norm(swapped) = {g['valid'] - g['swapped']:+.4f}")
        print("  |차이| 가 0 에 가까우면 → 비율이 지시 내용에 반응하지 않음 (H1 지지)")
        _ = d
    sw = df[df.condition == "swapped"]
    if len(sw):
        print(f"모순 지시에서 7 DoF 가 전부 그대로인 비율: "
              f"{(sw['n_identical_dof'] == 7).mean():.1%}")
        print("  높으면 → 지시를 바꿔도 같은 행동. 'linguistic blindness' 재현.")
    if args.ambiguous_tasks:
        print("\n--- 모호 장면 vs 비모호 장면 ---")
        print(df.groupby(["ambiguous", "condition"])[["R_norm", "KL_vs_valid"]]
              .mean().round(4).to_string())
        print("  모호 장면에서 R_norm 이 유의하게 커지지 않으면 → H1(비적응성) 지지")
    else:
        print("\n!! --ambiguous-tasks 를 지정하지 않았습니다.")
        print("   이 조건 없이는 'LIBERO 는 원래 언어가 필요 없다'는 반박을 막을 수 없습니다.")
    return 0


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    _ = itertools
    sys.exit(main())
