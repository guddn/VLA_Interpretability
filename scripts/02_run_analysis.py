"""Phase 2+3: LIBERO 롤아웃을 돌면서 매 스텝 비율 + knockout KL 을 기록.

출력: outputs/analysis_<suite>.csv  (tidy)
      outputs/perlayer_<suite>.npz  (층별 배열)

사용:
    python scripts/02_run_analysis.py --suite spatial --tasks 0 1 2 --episodes 3 --max-steps 40
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod.device import apply_env, apply_overrides, report_gpu  # noqa: E402
from vlamod import env_libero as EL  # noqa: E402
from vlamod import pipeline as P  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.model_loader import load_openvla  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5, help="몇 스텝마다 분석할지 (전부 하면 느림)")
    ap.add_argument("--no-intervene", action="store_true")
    ap.add_argument("--hf-home", default=None,
                    help="HuggingFace 캐시 루트. config 의 env.hf_home 을 덮어씀")
    ap.add_argument("--gpu", type=int, default=None, metavar="N",
                    help="사용할 GPU 번호 (예: --gpu 3)")
    ap.add_argument("--gpus", default=None, metavar="0,1",
                    help="여러 GPU 에 모델을 분할 로드 (예: --gpus 0,1). "
                         "한 장에 안 들어갈 때 씁니다. 수치는 단일 GPU 와 동일합니다.")
    ap.add_argument("--device", default=None,
                    help='--gpu 대신 문자열로 지정. "cuda:3" / "3" / "cpu"')
    ap.add_argument("--model", default=None, help="체크포인트 경로/HF repo. config 값을 덮어씀")
    ap.add_argument("--unnorm-key", default=None, help="action un-normalization key. config 값을 덮어씀")
    ap.add_argument("--tag", default=None, help="출력 파일명 접미사 (체크포인트 여러 개 비교 시)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    apply_env(cfg, args)          # ← transformers/libero import 전에 반드시 먼저
    mcfg = cfg["model"]
    apply_overrides(mcfg, args)
    torch.manual_seed(cfg["run"]["seed"])

    vla = load_openvla(
        path=mcfg["path"], device=mcfg["device"], dtype=mcfg["dtype"],
        attn_implementation=mcfg["attn_implementation"],
        gpus=mcfg.get("gpus"),
    )
    report_gpu(mcfg["device"], mcfg.get("gpus"))

    # visual span 은 한 번만 실측하고 재사용
    from PIL import Image
    rng = np.random.default_rng(0)
    ia = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    ib = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    p0, _ = TI.build_prompt("dummy instruction")
    vspan = TI.probe_visual_span(
        vla,
        vla.processor(p0, ia).to(vla.device, dtype=vla.dtype),
        vla.processor(p0, ib).to(vla.device, dtype=vla.dtype),
    )
    print(f"[visual span] {vspan}  n_visual={vspan[1] - vspan[0]}")

    rows, per_layer_acc = [], []
    for task_id in args.tasks:
        task = EL.make_task(args.suite, task_id)
        print(f"\n=== task {task_id}: {task.instruction!r}")
        for ep in range(args.episodes):
            EL.reset_to(task, ep)
            obs = EL.step_noop(task, 10)
            for t in range(args.max_steps):
                if t % args.stride == 0:
                    img = EL.obs_to_image(obs)
                    row = P.analyze_step(
                        vla, img, task.instruction,
                        sink_positions=cfg["tokens"]["sink_positions"],
                        instruction_only=cfg["tokens"]["instruction_only"],
                        visual_span=vspan,
                        do_intervene=not args.no_intervene,
                    )
                    pl = row.pop("_per_layer")
                    per_layer_acc.append(pl)
                    row.update({"suite": args.suite, "task_id": task_id, "episode": ep, "t": t})
                    rows.append(row)
                    print(
                        f"  t={t:3d} R_raw={row['R_raw']:.4f} R_norm={row['R_norm']:.4f} "
                        f"R_vn={row['R_vnorm']:.4f}"
                        + (
                            f" | KL_L={row['KL_lang_knockout']:.4f} "
                            f"KL_V={row['KL_vis_knockout']:.4f} "
                            f"KL_ctrl={row['KL_control_knockout']:.4f}"
                            if not args.no_intervene else ""
                        )
                    )
                # 정책이 낸 행동으로 환경을 진행 (분석 대상 궤적을 실제 정책이 만들도록)
                action = vla.model.predict_action(
                    **vla.processor(TI.build_prompt(task.instruction)[0], EL.obs_to_image(obs))
                    .to(vla.device, dtype=vla.dtype),
                    unnorm_key=mcfg["unnorm_key"],
                    do_sample=False,
                )
                obs, _, done, _ = task.env.step(action.tolist())
                if done:
                    break
        task.env.close()

    df = pd.DataFrame(rows)
    stem = f"{args.suite}{'_' + args.tag if args.tag else ''}"
    out_csv = f"outputs/analysis_{stem}.csv"
    df.to_csv(out_csv, index=False)
    np.savez(
        f"outputs/perlayer_{stem}.npz",
        R_raw=np.array([p["R_raw"] for p in per_layer_acc]),
        R_norm=np.array([p["R_norm"] for p in per_layer_acc]),
        R_vnorm=np.array([p["R_vnorm"] for p in per_layer_acc]),
    )
    print(f"\n저장: {out_csv}  ({len(df)} rows)")

    print("\n--- 요약 ---")
    cols = [c for c in ["R_raw", "R_norm", "R_vnorm", "uniform_baseline",
                        "KL_lang_knockout", "KL_vis_knockout", "KL_control_knockout",
                        "lang_vs_control", "causal_ratio"] if c in df.columns]
    print(df[cols].describe().T[["mean", "std", "min", "max"]])
    if "lang_vs_control" in df.columns:
        m = df["lang_vs_control"].mean()
        print(f"\n언어 knockout / 랜덤 대조군 knockout 비율 평균 = {m:.3f}")
        print("  1에 가까우면: 언어 토큰이 '아무 토큰이나 몇 개 지운 것'과 구별되지 않음")
    return 0


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    sys.exit(main())
