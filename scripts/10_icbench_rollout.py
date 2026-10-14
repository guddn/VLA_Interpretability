"""10: ICBench 방식 모순 지시 롤아웃 (A단계 — 행동만, attention 분석 없음).

목적
----
"모순 지시를 받은 에피소드 초반의 내부 지표가, 그 에피소드가 **가짜 성공**(지시를 무시하고 원래 과제를 완수)으로
끝날지 **실제 실패**로 끝날지 예측하는가" 를 묻기 위한 행동 데이터를 만듭니다.
B단계(11, 예정)는 여기서 저장한 행동 시퀀스를 **시뮬레이터에서 재생**해 프레임을 복원한 뒤 분석합니다
(MuJoCo 는 같은 초기 상태 + 같은 행동이면 같은 궤적 → GPU 와 무관하게 프레임 재현).

변형: normal / V1 / V2 / V3 / V4 / V5 / R  (정의는 vlamod/icbench.py)
채점: 전부 **원래 bddl 성공 판정** (On bowl_1 plate_1). 그 외에 방해 그릇이 접시에 올라갔는지,
      각 그릇을 들어 올렸는지(높이 +3cm)를 따로 기록해 결과를 target / distractor / wrong_lift / abstain 으로 나눕니다.

!! 원 ICBench 지시문은 공개되지 않아 우리가 만든 'ICBench 방식' 지시문입니다. 논문 수치와 직접 비교 불가.
!! 성공 여부는 GPU 종류에 따라 달라질 수 있습니다(bf16). 같은 실험 묶음은 같은 GPU 로.

사용:
    python scripts/10_icbench_rollout.py --dry-run                       # 지시문 표만 출력 (모델 안 올림)
    python scripts/10_icbench_rollout.py --gpu 4 --tasks 0 --inits 10    # 파일럿
    python scripts/10_icbench_rollout.py --gpu 4 --tasks 0 1 2 3 4 --inits 20

출력: outputs/10_icbench_rollout/
  variants_<suite>.csv   task × 변형 지시문 표 (검토용)
  rollouts_<suite>.csv   한 행 = 한 롤아웃 (결과 범주, 스텝 수, 정상 궤적과 갈라진 시점 t_star)
  traj_<suite>.npz       롤아웃별 actions[T,7], eef[T,3], bowl_z[T,2]  (B단계 재생용)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod import icbench as IC  # noqa: E402
from vlamod import paths as OP  # noqa: E402

LIFT_M = 0.03          # 그릇을 '들었다' 고 볼 높이 상승 (m)
DIVERGE_M = 0.02       # 정상 궤적과 '갈라졌다' 고 볼 말단 위치 차이 (m)


def variants_table(tasks) -> pd.DataFrame:
    rows = []
    for tid in tasks:
        tgt, _ = IC.SPATIAL_SCENES[tid]
        vs = IC.spatial_variants(tid, IC.SPATIAL_PREFIX + tgt + IC.SPATIAL_SUFFIX)
        for k, s in vs.items():
            rows.append({"task_id": tid, "variant": k, "instruction": s,
                         "changed": " ".join(IC.changed_words(vs["normal"], s))})
    return pd.DataFrame(rows)


def rollout(vla, task, init, instruction, unnorm_key, max_steps, P, EL):
    EL.reset_to(task, init)
    obs = EL.step_noop(task, 10)
    z0 = np.array([IC.object_pos(task, IC.TARGET)[2], IC.object_pos(task, IC.DISTRACTOR)[2]])
    acts, eef, bz = [], [], []
    done, dis_on = False, False
    for t in range(max_steps):
        a = P.policy_action(vla, EL.obs_to_image(obs), instruction, unnorm_key)
        acts.append(np.asarray(a, dtype=np.float32))
        eef.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        obs, _, done, _ = task.env.step(a.tolist())
        bz.append([IC.object_pos(task, IC.TARGET)[2], IC.object_pos(task, IC.DISTRACTOR)[2]])
        if done:
            break
        if t % 10 == 9 and IC.on_plate(task, IC.DISTRACTOR):   # 방해 그릇을 올렸으면 더 볼 필요 없음
            dis_on = True
            break
    try:
        success = bool(task.env.check_success())
    except Exception:  # noqa: BLE001
        success = bool(done)
    dis_on = dis_on or IC.on_plate(task, IC.DISTRACTOR)
    bz = np.asarray(bz, dtype=np.float32)
    rise = (bz - z0[None, :]).max(axis=0) if len(bz) else np.zeros(2)
    return {
        "success": success, "distractor_on_plate": dis_on,
        "lifted_target": bool(rise[0] > LIFT_M), "lifted_distractor": bool(rise[1] > LIFT_M),
        "n_steps": len(acts),
    }, np.stack(acts), np.stack(eef), bz


def summarize(df: pd.DataFrame) -> None:
    print("\n--- 원래 판정 성공률(SR)과 LGS = SR(normal) − SR(변형) ---")
    sr = df.groupby("variant")["success"].mean().reindex(IC.VARIANTS).dropna()
    for v, s in sr.items():
        lgs = "" if v == "normal" else f"   LGS {100 * (sr['normal'] - s):+5.1f}"
        print(f"  {v:7s} SR {100 * s:5.1f}%{lgs}")

    ok = df[(df.variant == "normal") & df.success][["task_id", "init"]]
    sub = df.merge(ok, on=["task_id", "init"])
    sub = sub[sub.variant != "normal"]
    print(f"\n--- 정상 지시로 성공한 초기 상태 {len(ok)}개에서의 결과 분포 (가짜 성공 = target) ---")
    if len(sub):
        tab = pd.crosstab(sub["variant"], sub["outcome"]).reindex(IC.VARIANTS[1:]).dropna(how="all")
        print(tab.to_string())
        n_fail = int((sub["outcome"] != "target").sum())
        print(f"\n  실제 실패(target 이외) 총 {n_fail}개 / {len(sub)}개")
        if n_fail < 10:
            print("  !! 실제 실패가 10개 미만입니다. B단계의 '가짜 성공 vs 실제 실패' 비교는 표본이 부족합니다.")
        r = sub[sub.variant == "R"]
        if len(r):
            print(f"  R(방해 그릇을 가리키는 정상 명령)에서 지시를 따름(distractor): {(r.outcome == 'distractor').mean():.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--suite", default="spatial", choices=["spatial"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--inits", type=int, default=10, help="task 당 초기 상태 수 (LIBERO 는 50개 제공)")
    ap.add_argument("--init-offset", type=int, default=0, help="초기 상태 시작 번호 (나눠 돌릴 때)")
    ap.add_argument("--variants", nargs="+", default=IC.VARIANTS, choices=IC.VARIANTS)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--dry-run", action="store_true", help="지시문 표만 저장·출력하고 종료")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--gpu", type=int, default=None, metavar="N")
    ap.add_argument("--headroom-gb", type=float, default=None, metavar="G")
    ap.add_argument("--allow-cpu-offload", action="store_true")
    ap.add_argument("--egl-device", type=int, default=None, metavar="E")
    ap.add_argument("--gpus", default=None, metavar="0,1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    variants = ["normal"] + [v for v in args.variants if v != "normal"]   # t_star 계산에 normal 필수
    stem = f"{args.suite}{'_' + args.tag if args.tag else ''}"
    out_dir = OP.out_dir(OP.ICBENCH_ROLLOUT, args.tag)

    vt = variants_table(args.tasks)
    vt.to_csv(os.path.join(out_dir, f"variants_{stem}.csv"), index=False)
    print("--- 변형 지시문 ---")
    for tid, g in vt.groupby("task_id"):
        print(f"[task {tid}]")
        for _, r in g.iterrows():
            if r.variant in variants:
                print(f"  {r.variant:7s} {r.instruction}")
    if args.dry_run:
        print(f"\n저장: {os.path.join(out_dir, f'variants_{stem}.csv')}  (--dry-run: 롤아웃 안 함)")
        return 0

    import torch          # --dry-run 은 torch 없이도 돌도록 여기서 import
    import yaml
    from vlamod.device import apply_env, apply_overrides, report_gpu
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    apply_env(cfg, args)          # ← transformers/libero import 전에 반드시 먼저
    mcfg = cfg["model"]
    apply_overrides(mcfg, args)
    torch.manual_seed(cfg["run"]["seed"])
    from vlamod import env_libero as EL
    from vlamod import pipeline as P
    from vlamod.model_loader import load_openvla
    vla = load_openvla(path=mcfg["path"], device=mcfg["device"], dtype=mcfg["dtype"],
                       attn_implementation=mcfg["attn_implementation"],
                       gpus=mcfg.get("gpus"),
                       headroom_gb=mcfg.get("headroom_gb", 1.5),
                       cpu_offload=mcfg.get("cpu_offload", False))
    report_gpu(mcfg["device"], mcfg.get("gpus"))
    gpu_name = torch.cuda.get_device_name(vla.device) if torch.cuda.is_available() else "cpu"

    csv_path = os.path.join(out_dir, f"rollouts_{stem}.csv")
    npz_path = os.path.join(out_dir, f"traj_{stem}.npz")
    rows, traj = [], {}
    t_start = time.time()
    for tid in args.tasks:
        task = EL.make_task(args.suite, tid)
        vs = IC.spatial_variants(tid, task.instruction)      # 원문이 장면 표와 다르면 여기서 중단
        for init in range(args.init_offset, args.init_offset + args.inits):
            eef_normal = None
            for v in variants:
                res, acts, eef, bz = rollout(vla, task, init, vs[v], mcfg["unnorm_key"],
                                             args.max_steps, P, EL)
                key = f"{tid}_{init}_{v}"
                traj[f"{key}__actions"], traj[f"{key}__eef"], traj[f"{key}__bowl_z"] = acts, eef, bz
                if v == "normal":
                    eef_normal = eef
                outcome = IC.classify(res["success"], res["distractor_on_plate"],
                                      res["lifted_target"], res["lifted_distractor"])
                rows.append({
                    "suite": args.suite, "task_id": tid, "init": init, "variant": v,
                    "instruction": vs[v], "changed": " ".join(IC.changed_words(vs["normal"], vs[v])),
                    **res, "outcome": outcome,
                    "t_star": -1 if v == "normal" else IC.first_divergence(eef_normal, eef, DIVERGE_M),
                    "gpu": gpu_name,
                })
                print(f"  task {tid} init {init:2d} {v:7s} → {outcome:10s} ({res['n_steps']:3d} steps)"
                      + ("" if v == "normal" else f"  t*={rows[-1]['t_star']}"))
        task.env.close()
        # task 하나 끝날 때마다 저장 (중간에 죽어도 그때까지는 남음)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        np.savez_compressed(npz_path, **traj)
        print(f"  [저장] task {tid} 까지 {len(rows)}개 롤아웃, 경과 {(time.time() - t_start) / 60:.1f}분")

    df = pd.DataFrame(rows)
    print(f"\n저장: {csv_path}\n      {npz_path}")
    summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
