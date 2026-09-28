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
    ap.add_argument("--headroom-gb", type=float, default=None, metavar="G",
                    help="--gpus 분할 시 GPU 당 안전 마진(GB). 기본 1.5. 빠듯하면 0.8")
    ap.add_argument("--allow-cpu-offload", action="store_true",
                    help="GPU 에 다 못 올리면 일부 층을 CPU 로. **매우 느림** — 구조 검증용")
    ap.add_argument("--egl-device", type=int, default=None, metavar="E",
                    help="렌더링(EGL) 디바이스 번호. **CUDA 번호와 다른 체계**이고 "
                         "보통 0 하나뿐입니다. 미지정이면 자동 판정. "
                         "목록은 setup/07_egl_probe.py 로 확인하세요")
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
        headroom_gb=mcfg.get("headroom_gb", 1.5),
        cpu_offload=mcfg.get("cpu_offload", False),
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
            # !! 이 에피소드가 만든 행들의 위치를 기억해 두었다가, 끝난 뒤 성공 여부를
            #    소급해서 채웁니다. 성공률이 필요한 이유 두 가지:
            #      1) 이미지 전처리(상하/좌우 반전)가 학습과 어긋나면 성공률이 무너집니다.
            #         육안으로는 못 잡는 오류를 잡는 **유일한 경험적 검증**입니다.
            #      2) 나중에 "언어 의존 그룹 vs 비전 의존 그룹" 을 나눌 때
            #         성공/실패 라벨이 있어야 그룹 특성을 말할 수 있습니다.
            ep_row_start = len(rows)
            ep_done = False
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
                action = P.policy_action(
                    vla, EL.obs_to_image(obs), task.instruction, mcfg["unnorm_key"]
                )
                obs, _, done, _ = task.env.step(action.tolist())
                if done:
                    ep_done = True
                    break

            # --- 에피소드 종료: 성공 여부 판정 -------------------------
            # !! done=True 는 "성공" 과 같지 않습니다. horizon 도달로도 True 가 됩니다.
            #    LIBERO/robosuite 는 목표 술어를 직접 검사하는 check_success() 를 줍니다.
            #    그게 없으면 done 으로 폴백하되, 그 사실을 컬럼에 남깁니다.
            success, how = None, "unknown"
            for obj in (task.env, getattr(task.env, "env", None)):
                fn = getattr(obj, "check_success", None) or getattr(obj, "_check_success", None)
                if callable(fn):
                    try:
                        success, how = bool(fn()), "check_success"
                        break
                    except Exception:  # noqa: BLE001
                        pass
            if success is None:
                success, how = bool(ep_done), "done_flag(부정확)"
            for r in rows[ep_row_start:]:
                r["ep_success"] = success
                r["success_source"] = how
                r["ep_len"] = t + 1
            print(f"    └ ep {ep}: success={success} ({how}), {t + 1} steps")
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

    # ---- 성공률 — 전처리 정합성의 **유일한 경험적 검증** ----------------
    if "ep_success" in df.columns:
        per_ep = df.groupby(["task_id", "episode"])["ep_success"].first()
        sr = float(per_ep.mean())
        src = df["success_source"].iloc[0] if "success_source" in df.columns else "?"
        print(f"\n--- 태스크 성공률 ---")
        print(f"  {per_ep.sum():.0f} / {len(per_ep)} = {sr:.1%}   (판정: {src})")
        print("  ★ 이 값을 OpenVLA 논문의 LIBERO 수치와 대조하세요.")
        print("    크게 낮으면 이미지 전처리(상하/좌우 반전)가 학습과 어긋났을 수 있습니다.")
        print("    육안 확인으로는 좌우 반전을 잡을 수 없습니다 — 성공률이 유일한 단서입니다.")
        if sr == 0.0:
            print("  !! 성공률 0% 입니다. max_steps 가 너무 짧거나(기본 40),")
            print("     전처리가 어긋났거나, unnorm_key 가 체크포인트와 안 맞습니다.")
    return 0


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    sys.exit(main())
