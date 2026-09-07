"""LIBERO 렌더링 실패를 **단계별로 쪼개서** 어디서 죽는지 찾는 진단 스크립트.

`04_verify.py` 의 렌더링 검사가 실패했을 때만 쓰세요.

    python setup/05_diag_render.py --gpu 0
    python setup/05_diag_render.py --gpu 0 --suite spatial --task-id 0

`env_libero.make_task()` 가 하는 일을 한 줄씩 풀어서, 각 단계마다
성공/실패와 **전체 트레이스백**을 찍습니다. 실패해도 다음 단계로 넘어가지 않고
그 자리에서 멈춥니다 (뒤 단계는 앞 단계 결과에 의존하므로).

특히 잘 나는 실패:
  - init_states 파일 없음      → LIBERO 저장소가 불완전 (git lfs / 서브셋 clone)
  - set_init_state 의 bare assert → init_states 배열 길이 ≠ 모델 nq+nv+1
                                    (bddl 과 init 파일이 서로 다른 task)
  - env.step 의 action 차원     → 컨트롤러 설정이 OSC_POSE 가 아님
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUITE_MAP = {"spatial": "libero_spatial", "object": "libero_object",
             "goal": "libero_goal", "long": "libero_10"}


def step(n: int, title: str):
    print(f"\n--- [{n}] {title} " + "-" * max(0, 50 - len(title)))


def die(e: BaseException) -> int:
    print("\n" + "!" * 66)
    traceback.print_exc()
    print("!" * 66)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--out", default="outputs/diag_render.png")
    args = ap.parse_args()

    try:
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    from vlamod.device import apply_env
    apply_env(cfg, args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    os.makedirs("outputs", exist_ok=True)
    print(f"[device] render(EGL)=cuda:{args.gpu}")

    env = None
    try:
        # ---------------------------------------------------------------
        step(1, "LIBERO 경로 설정 (~/.libero/config.yaml)")
        from libero.libero import get_libero_path
        for key in ("benchmark_root", "bddl_files", "init_states", "datasets", "assets"):
            try:
                p = get_libero_path(key)
                print(f"  {key:15s} = {p}   {'있음' if os.path.exists(p) else '★ 없음'}")
            except Exception as e:  # noqa: BLE001
                print(f"  {key:15s} = (조회 실패) {type(e).__name__}: {e}")

        # ---------------------------------------------------------------
        step(2, "benchmark / task 메타데이터")
        from libero.libero import benchmark
        suite_key = SUITE_MAP[args.suite]
        bench = benchmark.get_benchmark_dict()[suite_key]()
        task = bench.get_task(args.task_id)
        print(f"  suite          = {suite_key}")
        print(f"  n_tasks        = {bench.n_tasks}")
        for attr in ("name", "language", "problem", "problem_folder",
                     "bddl_file", "init_states_file"):
            print(f"  {attr:15s} = {getattr(task, attr, '(없음)')!r}")

        # ---------------------------------------------------------------
        step(3, "bddl 파일 존재 확인")
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        print(f"  {bddl}")
        print(f"  존재: {os.path.exists(bddl)}  "
              f"크기: {os.path.getsize(bddl) if os.path.exists(bddl) else '-'}")
        if not os.path.exists(bddl):
            print("  ★ bddl 파일이 없습니다. LIBERO 저장소가 불완전합니다.")
            return 1

        # ---------------------------------------------------------------
        step(4, "init_states 파일 로드  ← 여기가 자주 죽습니다")
        init_dir = get_libero_path("init_states")
        init_path = os.path.join(init_dir, task.problem_folder,
                                 getattr(task, "init_states_file", "?"))
        print(f"  {init_path}")
        print(f"  존재: {os.path.exists(init_path)}")
        init_states = bench.get_task_init_states(args.task_id)
        import numpy as np
        arr = np.asarray(init_states)
        print(f"  로드 성공: shape={arr.shape}  dtype={arr.dtype}")

        # ---------------------------------------------------------------
        step(5, "OffScreenRenderEnv 생성 (EGL 컨텍스트)")
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=args.resolution,
                                 camera_widths=args.resolution)
        print("  생성 성공")
        sim = getattr(getattr(env, "env", None), "sim", None)
        if sim is not None:
            nq, nv = sim.model.nq, sim.model.nv
            print(f"  모델 nq={nq} nv={nv}  →  init_state 기대 길이 = {nq + nv + 1}")
            print(f"  실제 init_state 길이 = {arr.shape[-1]}")
            if arr.shape[-1] != nq + nv + 1:
                print("  ★ 길이 불일치 — set_init_state 에서 bare assert 가 납니다.")
                print("    bddl 과 init_states 가 서로 다른 task 를 가리키고 있습니다.")

        # ---------------------------------------------------------------
        step(6, "env.seed(0)")
        env.seed(0)
        print("  OK")

        step(7, "env.reset()")
        env.reset()
        print("  OK")

        step(8, "env.set_init_state()")
        obs = env.set_init_state(init_states[0])
        print(f"  OK  obs keys = {sorted(obs.keys())[:8]} ...")

        step(9, "env.step() × 5 (물리 안정화)")
        dummy = np.array([0.0] * 6 + [-1.0], dtype=np.float32)
        print(f"  action_dim(env) = {getattr(env.env, 'action_dim', '?')}  "
              f"보내는 길이 = {len(dummy)}")
        for i in range(5):
            obs, _, _, _ = env.step(dummy)
        print("  OK")

        # ---------------------------------------------------------------
        step(10, "프레임 추출 + 저장")
        img = obs["agentview_image"]
        a = np.asarray(img)
        print(f"  raw shape={a.shape} dtype={a.dtype} std={a.std():.2f} "
              f"min={a.min()} max={a.max()}")
        if a.std() <= 1.0:
            print("  ★ 단색 프레임입니다 — 렌더링은 됐지만 장면이 안 그려졌습니다.")
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(a[::-1, ::-1]).astype(np.uint8)).save(args.out)
        print(f"  저장: {args.out}  (180도 회전 적용본)")
        raw_out = args.out.replace(".png", "_noflip.png")
        Image.fromarray(np.ascontiguousarray(a).astype(np.uint8)).save(raw_out)
        print(f"  저장: {raw_out}  (원본)")
        print("\n두 장을 열어 **로봇 팔이 똑바로 선 쪽**이 어느 것인지 확인하세요.")
        print("원본 쪽이 맞으면 env_libero.obs_to_image(flip=False) 로 바꿔야 합니다.")

    except Exception as e:  # noqa: BLE001
        return die(e)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    print("\n전부 통과했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
