"""정책이 정상 작동하는가 — 전처리 변형을 나란히 돌려 **경험적으로** 판정.

    python scripts/08_policy_sanity.py --gpu 5 --tasks 0 1 --max-steps 220

왜 필요한가
-----------
LIBERO 성공률이 400 스텝에서도 0/5 였습니다. 스텝 부족이 아니므로
전처리 또는 unnorm_key 가 학습과 어긋났을 가능성이 큽니다.
후보를 하나씩 바꿔가며 **성공률과 행동 통계**로 고릅니다.

무엇을 찍는가 (성공률 0 이어도 판별 가능한 신호들)
--------------------------------------------------
  success        : 태스크 성공 여부
  gripper_closes : 그리퍼를 닫으려 시도한 스텝 수  ← 0 이면 집으려는 시도조차 없음
  ee_path        : end-effector 이동 거리 합       ← 0 에 가까우면 정지
  |action| 통계  : 평균/최대                        ← 포화(±1)면 제어가 깨진 것
  action 다양성  : 고유 action 개수                 ← 1~2 면 같은 행동 반복

**성공률이 전부 0 이어도, 이 보조 지표들의 차이로 어느 전처리가 맞는지 좁혀집니다.**
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod.device import apply_env, apply_overrides, report_gpu  # noqa: E402
from vlamod import env_libero as EL  # noqa: E402
from vlamod import pipeline as P  # noqa: E402
from vlamod import preprocess as PP  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.model_loader import load_openvla  # noqa: E402


@torch.no_grad()
def act(vla, img, instruction, unnorm_key):
    prompt, _ = TI.build_prompt(instruction)
    inputs = vla.processor(prompt, img).to(vla.device, dtype=vla.dtype)
    inputs = {k: v for k, v in inputs.items() if k != "attention_mask"}
    return vla.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)


def ee_pos(obs):
    for k in ("robot0_eef_pos", "robot0_eef_pos_xyz"):
        if k in obs:
            return np.asarray(obs[k], float)
    return None


def run_one(vla, suite, task_id, variant, unnorm_key, max_steps, gripper_fix, wait=10):
    task = EL.make_task(suite, task_id)
    EL.reset_to(task, 0)
    dummy = np.array([0.0] * 6 + [-1.0], dtype=np.float32)   # −1 = 열기 (LIBERO 규약)
    obs = None
    for _ in range(wait):
        obs, _, _, _ = task.env.step(dummy)

    raws, envs, path, prev = [], [], 0.0, ee_pos(obs)
    success, steps, done = False, 0, False
    for t in range(max_steps):
        img = PP.apply_variant(obs["agentview_image"], variant)
        raw = np.asarray(act(vla, img, task.instruction, unnorm_key), float)
        env_a = P.libero_env_action(raw) if gripper_fix else raw
        raws.append(raw); envs.append(env_a)
        obs, _, done, _ = task.env.step(env_a.tolist())
        steps = t + 1
        p = ee_pos(obs)
        if p is not None and prev is not None:
            path += float(np.linalg.norm(p - prev))
        prev = p
        if done:
            break
    fn = getattr(task.env, "check_success", None)
    try:
        success = bool(fn()) if callable(fn) else bool(done)
    except Exception:  # noqa: BLE001
        success = bool(done)
    task.env.close()

    R, E = np.array(raws), np.array(envs)
    g = R[:, 6]
    return dict(
        success=success, steps=steps,
        # env 가 실제로 받은 '닫기' 명령 수 (LIBERO: +1 = 닫기)
        grip_close=int((E[:, 6] > 0.5).sum()),
        ee_path=round(path, 4),
        act_absmean=round(float(np.abs(R[:, :6]).mean()), 4),
        act_absmax=round(float(np.abs(R[:, :6]).max()), 4),
        n_unique=int(len(np.unique(np.round(R, 4), axis=0))),
        # ★ 원시 그리퍼 출력 분포 — 규약 확인용
        g_min=round(float(g.min()), 3), g_max=round(float(g.max()), 3),
        g_near0=round(float((np.abs(g) < 0.1).mean()), 3),
        g_near1=round(float((np.abs(g - 1) < 0.1).mean()), 3),
        g_neg=round(float((g < -0.1).mean()), 3),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--variants", nargs="+", default=["current", "rot180_crop"],
                    help=f"가능: {list(PP.VARIANTS)}")
    ap.add_argument("--gripper-fix", choices=["on", "off", "both"], default="both",
                    help="OpenVLA 그리퍼 규약 변환. both(기본)=변환 전/후를 나란히 비교")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--egl-device", type=int, default=None)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--headroom-gb", type=float, default=None)
    ap.add_argument("--allow-cpu-offload", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    apply_env(cfg, args)
    mcfg = cfg["model"]
    apply_overrides(mcfg, args)
    vla = load_openvla(path=mcfg["path"], device=mcfg["device"], dtype=mcfg["dtype"],
                       attn_implementation=mcfg["attn_implementation"],
                       gpus=mcfg.get("gpus"),
                       headroom_gb=mcfg.get("headroom_gb", 1.5),
                       cpu_offload=mcfg.get("cpu_offload", False))
    report_gpu(mcfg["device"], mcfg.get("gpus"))

    # ---- unnorm_key 확인 — 여기서 틀리면 action 스케일이 통째로 어긋납니다
    print("\n" + "=" * 74)
    print("[A] norm_stats 키 목록  ← unnorm_key 가 이 안에 있어야 합니다")
    print("=" * 74)
    ns = getattr(vla.model, "norm_stats", None)
    if isinstance(ns, dict):
        for k in ns:
            mark = "  ← 지금 쓰는 값" if k == mcfg["unnorm_key"] else ""
            print(f"    {k}{mark}")
        if mcfg["unnorm_key"] not in ns:
            print(f"  ★ '{mcfg['unnorm_key']}' 가 목록에 없습니다!")
        cand = [k for k in ns if k != mcfg["unnorm_key"]]
        if cand:
            print(f"  참고: 다른 후보가 {len(cand)}개 있습니다. "
                  f"'_no_noops' 접미사 버전이 있으면 그쪽이 맞을 수 있습니다.")
    else:
        print("    norm_stats 를 못 찾았습니다.")

    print("\n" + "=" * 74)
    print("[B] 전처리 변형별 롤아웃")
    print("=" * 74)
    fixes = {"on": [True], "off": [False], "both": [False, True]}[args.gripper_fix]
    rows = []
    for v in args.variants:
        if v not in PP.VARIANTS:
            print(f"  알 수 없는 변형: {v}"); continue
        for gf in fixes:
            for tid in args.tasks:
                r = run_one(vla, args.suite, tid, v, mcfg["unnorm_key"],
                            args.max_steps, gripper_fix=gf)
                r.update(variant=v, task=tid, gripper_fix=gf)
                rows.append(r)
                tag = f"{v}+{'grip' if gf else 'raw '}"
                print(f"  [{tag:18s}] task {tid}  success={str(r['success']):5s}  "
                      f"steps={r['steps']:3d}  닫기명령={r['grip_close']:3d}  "
                      f"EE이동={r['ee_path']:6.3f}  |a|평균={r['act_absmean']:.3f}  "
                      f"고유={r['n_unique']}")
                print(f"  {'':20s} 원시 그리퍼: [{r['g_min']:+.3f}, {r['g_max']:+.3f}]  "
                      f"≈0: {r['g_near0']:.0%}  ≈1: {r['g_near1']:.0%}  음수: {r['g_neg']:.0%}")

    print("\n" + "=" * 74)
    print("[C] 판정")
    print("=" * 74)
    print("  ① 원시 그리퍼 분포부터 보세요 — 규약 확인")
    print("     ≈0 과 ≈1 에 몰려 있고 음수가 거의 없다 → [0,1] 규약 확정 → 변환이 필요합니다")
    print("     −1 ~ +1 에 퍼져 있다                   → 이미 env 규약. 변환하면 오히려 망가짐")
    print("  ② raw 와 grip 을 비교하세요")
    print("     grip 쪽만 success=True → 그리퍼 규약이 원인이었습니다")
    print("  ③ grip 끼리 이미지 변형을 비교하세요 → 이제서야 이미지 전처리 판정이 가능합니다")
    print("     (그리퍼가 고장 난 상태에서는 어떤 이미지 전처리도 성공할 수 없었습니다)")
    wins = [r for r in rows if r["success"]]
    if wins:
        print("\n  ★ 성공한 조합:")
        for r in wins:
            print(f"     {r['variant']} + {'grip' if r['gripper_fix'] else 'raw'}  (task {r['task']})")
    else:
        print("\n  성공 없음. 원시 그리퍼 분포와 닫기명령 수를 보고 판단하세요.")
    return 0


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    sys.exit(main())
