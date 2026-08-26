"""LIBERO 래퍼 (얇게).

!! 검증 필요 지점 두 가지 — 처음 돌릴 때 반드시 눈으로 확인하세요.
   1) 이미지 방향: LIBERO 의 agentview_image 는 상하가 뒤집혀 나옵니다.
      OpenVLA 의 experiments/robot/libero/libero_utils.py 는 180도 회전
      (img[::-1, ::-1]) 을 적용합니다. --no-flip 으로 끄고 비교해 보세요.
      **방향이 틀리면 vision attention 분석 전체가 무의미해집니다.**
   2) task 별 language instruction 문자열이 어디서 나오는지
      (task.language 인지 task_description 인지) 버전마다 다릅니다.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

SUITE_MAP = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}


@dataclasses.dataclass
class LiberoTask:
    suite: str
    task_id: int
    instruction: str
    env: object
    init_states: object


def make_task(suite: str = "spatial", task_id: int = 0, resolution: int = 256) -> LiberoTask:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite_key = SUITE_MAP[suite]
    bench = benchmark.get_benchmark_dict()[suite_key]()
    task = bench.get_task(task_id)
    instruction = getattr(task, "language", None) or getattr(task, "task_description", None)
    if instruction is None:
        raise RuntimeError("task 에서 language instruction 을 못 찾았습니다. task 속성을 출력해 보세요.")

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        **{"bddl_file_name": bddl, "camera_heights": resolution, "camera_widths": resolution}
    )
    env.seed(0)
    init_states = bench.get_task_init_states(task_id)
    return LiberoTask(suite, task_id, instruction, env, init_states)


def reset_to(task: LiberoTask, episode: int = 0):
    task.env.reset()
    obs = task.env.set_init_state(task.init_states[episode % len(task.init_states)])
    return obs


def obs_to_image(obs, flip: bool = True):
    """agentview RGB → PIL.Image"""
    from PIL import Image

    img = obs["agentview_image"]
    if flip:
        img = img[::-1, ::-1]
    return Image.fromarray(np.ascontiguousarray(img).astype(np.uint8))


def step_noop(task: LiberoTask, n: int = 10):
    """LIBERO 는 시작 직후 물리가 안정될 때까지 몇 스텝 필요합니다."""
    dummy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    obs = None
    for _ in range(n):
        obs, _, _, _ = task.env.step(dummy)
    return obs
