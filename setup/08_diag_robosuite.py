"""robosuite / mujoco / LIBERO 버전 정합성을 진단합니다.

    python setup/08_diag_robosuite.py

이 증상 전용입니다:

    robosuite/robots/robot.py:161  setup_references
      self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x)
                                     for x in self.robot_joints]
    robosuite/utils/binding_utils.py:521  get_joint_qpos_addr
      assert joint_type in (mjJNT_HINGE, mjJNT_SLIDE)
    AssertionError            ← 메시지 없음

무엇을 의심하는가
-----------------
`joint_name2id(name)` 이 이름을 못 찾으면 MuJoCo 는 **-1** 을 돌려줍니다.
그러면 `jnt_type[-1]` = 모델의 **마지막** 관절을 읽게 되는데, LIBERO 장면의
마지막 관절은 대개 물체의 free joint 라서 HINGE/SLIDE 가 아니고 assert 가 터집니다.
즉 **로봇 관절 이름이 안 맞는다** 는 뜻이고, 보통 robosuite 버전 불일치입니다.

이 스크립트가 하는 일
---------------------
  [1] robosuite / mujoco / numpy 버전과 LIBERO 가 요구하는 핀을 대조
  [2] 순수 robosuite 환경(Lift/Panda)을 만들어 봄
      → 여기서도 죽으면 robosuite ↔ mujoco 문제 (LIBERO 무관)
      → 여기는 되는데 LIBERO 만 죽으면 LIBERO ↔ robosuite 문제
  [3] LIBERO 환경을 만들다 죽으면 **트레이스백의 지역변수를 뜯어서**
      기대한 관절 이름과 실제 모델의 관절 목록을 나란히 출력
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

SUITE_MAP = {"spatial": "libero_spatial", "object": "libero_object",
             "goal": "libero_goal", "long": "libero_10"}


def hdr(n, t):
    print(f"\n--- [{n}] {t} " + "-" * max(0, 52 - len(t)))


def jnt_type_name(v) -> str:
    import mujoco
    for k in ("mjJNT_FREE", "mjJNT_BALL", "mjJNT_SLIDE", "mjJNT_HINGE"):
        if int(v) == int(getattr(mujoco.mjtJoint, k)):
            return k
    return f"?({v})"


def main() -> int:  # noqa: C901
    # =============================================================== [1]
    hdr(1, "버전 대조")
    import numpy
    import mujoco
    import robosuite
    print(f"  robosuite = {getattr(robosuite, '__version__', '?')}")
    print(f"  mujoco    = {getattr(mujoco, '__version__', '?')}")
    print(f"  numpy     = {numpy.__version__}")
    print(f"  robosuite 경로 = {os.path.dirname(robosuite.__file__)}")

    # LIBERO 가 요구하는 핀을 직접 읽습니다 (제가 기억으로 단정하지 않기 위해)
    # robosuite 가 **선언한** mujoco 요구사항을 직접 읽습니다 (기억으로 단정 금지)
    try:
        from importlib.metadata import requires as _req
        pins = [r for r in (_req("robosuite") or []) if "mujoco" in r.lower()]
        print(f"\n  robosuite 가 요구하는 mujoco: {pins or '(선언 없음)'}")
    except Exception as e:  # noqa: BLE001
        print(f"\n  robosuite 요구사항 조회 실패: {e}")

    # !! 이 assert 가 왜 터지는지 직접 재현합니다.
    #    관절 이름/타입이 다 맞는데도 실패한다면 **enum 비교**가 깨진 것입니다.
    print("\n  enum 비교 동작 확인 (robosuite 의 assert 와 동일한 식):")
    hinge = mujoco.mjtJoint.mjJNT_HINGE
    slide = mujoco.mjtJoint.mjJNT_SLIDE
    v = numpy.int32(int(hinge))
    print(f"    mjJNT_HINGE = {hinge!r}  (int={int(hinge)})")
    print(f"    numpy.int32({int(hinge)}) == mjJNT_HINGE        → {v == hinge}")
    print(f"    numpy.int32({int(hinge)}) in (HINGE, SLIDE)     → {v in (hinge, slide)}")
    if not (v in (hinge, slide)):
        print("    ★ 여기가 False 입니다. 관절은 정상인데 **비교가 깨진** 것이고,")
        print("      robosuite 의 assert 는 절대 통과할 수 없습니다 → mujoco 버전 문제.")

    import libero
    lib_root = None
    paths = list(getattr(libero, "__path__", []))
    if paths:
        # libero.__path__[0] = <repo>/libero  → dirname 한 번이면 repo 루트
        lib_root = os.path.dirname(paths[0])
    print(f"\n  LIBERO 저장소 = {lib_root}")
    if lib_root:
        for fn in ("requirements.txt", "libero_requirements.txt", "setup.py",
                   "pyproject.toml"):
            p = os.path.join(lib_root, fn)
            if not os.path.exists(p):
                continue
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            hits = [ln.strip() for ln in txt.splitlines()
                    if "robosuite" in ln.lower() or "mujoco" in ln.lower()]
            if hits:
                print(f"    {fn}:")
                for h in hits:
                    print(f"      {h}")
        git_head = os.path.join(lib_root, ".git", "HEAD")
        if os.path.exists(git_head):
            print(f"    .git/HEAD: {open(git_head).read().strip()}")

    # =============================================================== [2]
    hdr(2, "순수 robosuite 환경 (Lift / Panda) — LIBERO 없이")
    plain_ok = False
    try:
        import robosuite as suite
        env = suite.make(
            env_name="Lift", robots="Panda",
            has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, ignore_done=True,
        )
        print("  ✓ 생성 성공 — robosuite ↔ mujoco 조합 자체는 정상입니다.")
        r = env.robots[0]
        print(f"    robot_joints = {list(getattr(r, 'robot_joints', []))}")
        plain_ok = True
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ 실패: {type(e).__name__}: {e}")
        print("    → LIBERO 와 무관합니다. robosuite/mujoco 버전 조합 문제입니다.")
        traceback.print_exc()

    # =============================================================== [3]
    hdr(3, "LIBERO 환경 — 실패하면 관절 목록을 뜯어봅니다")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bench.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=256, camera_widths=256)
        print("  ✓ 생성 성공 — 이 스크립트로는 재현되지 않았습니다.")
        env.close()
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ 실패: {type(e).__name__}: {e or '(메시지 없음)'}")

        # --- 트레이스백 프레임의 지역변수를 뜯습니다 --------------------
        want_name, robot, sim = None, None, None
        tb = e.__traceback__
        while tb is not None:
            f = tb.tb_frame
            loc = f.f_locals
            if f.f_code.co_name == "get_joint_qpos_addr" and "name" in loc:
                want_name = loc.get("name")
            if "self" in loc and hasattr(loc["self"], "robot_joints"):
                robot = loc["self"]
            if "self" in loc and hasattr(loc["self"], "sim") and sim is None:
                sim = getattr(loc["self"], "sim", None)
            tb = tb.tb_next

        print("\n  --- 실패 지점 분석 ---")
        if want_name is not None:
            print(f"  찾으려던 관절 이름: {want_name!r}")
        if robot is not None:
            names = list(getattr(robot, "robot_joints", []))
            print(f"  robot.robot_joints ({len(names)}개): {names}")
            sim = getattr(robot, "sim", sim)

        if sim is None:
            print("  sim 을 못 얻어 모델 관절 목록을 볼 수 없습니다.")
            print("  전체 트레이스백:")
            traceback.print_exc()
            return 1

        model = sim.model
        njnt = int(model.njnt)
        print(f"\n  모델의 실제 관절 {njnt}개:")
        actual = []
        for i in range(njnt):
            try:
                nm = model.joint_id2name(i)
            except Exception:  # noqa: BLE001
                nm = "(이름 조회 실패)"
            t = jnt_type_name(model.jnt_type[i])
            actual.append(nm)
            flag = "  ← 마지막(assert 가 읽은 것)" if i == njnt - 1 else ""
            print(f"    [{i:3d}] {str(nm):45s} {t}{flag}")

        if robot is not None:
            missing = [n for n in getattr(robot, "robot_joints", [])
                       if n not in actual]
            print(f"\n  ★ 모델에 없는 기대 관절: {missing if missing else '(없음)'}")
            if missing:
                print("    → 이름 불일치가 확인되었습니다. joint_name2id 가 -1 을 돌려주고")
                print("      jnt_type[-1] (마지막 free joint) 을 읽어 assert 가 터진 것입니다.")
                print("      원인은 robosuite 버전 불일치일 가능성이 큽니다.")
            else:
                print("    → 이름은 다 있습니다. 다른 원인입니다. 위 타입 목록을 보세요.")
        print("\n  전체 트레이스백:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
