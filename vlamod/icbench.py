"""ICBench 방식 모순 지시문 (LIBERO-Spatial) + 롤아웃 결과 판정.

원 논문: Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration
(arXiv 2603.06001) §3. **원 ICBench 지시문 목록은 공개되지 않았습니다** (2026-10-14 확인).
여기 있는 것은 논문의 분류(V1–V4)를 따라 우리가 만든 지시문이고, 논문 수치와 직접 비교할 수 없습니다.

장면 정보의 출처
----------------
LIBERO `bddl_files/libero_spatial/*.bddl` 의 (:init) — 10개 task 모두 같은 물체 7종
(akita_black_bowl ×2, cookies, glazed_rim_porcelain_ramekin, plate, wooden_cabinet, flat_stove)
을 다른 배치로 놓습니다. 성공 판정(:goal)은 전부 `(On akita_black_bowl_1 plate_1)` — **목표 그릇이 정해져 있어
다른 그릇을 접시에 올리면 실패**입니다. 그리고 모든 장면에서 **방해 그릇(bowl_2)이 다른 task 의 위치구 자리에** 있습니다.

변형 (모두 '원래 성공 판정'으로 채점 → 성공 = 가짜 성공)
-------------------------------------------------------
  normal  원문
  V1      물체 속성:   black bowl → blue bowl          (장면에 파란 물체 없음)
  V2      목표 속성:   on the plate → on the green plate (접시는 흰 바탕·붉은 테두리 — 'red/white' 는 모순이 아니라서 피함)
  V3      V1 + V2
  V4      공간 관계:   place it on the plate → place it under the plate
  V5      (추가) 위치구를 **이 장면에 그릇이 없는** 다른 task 의 위치구로 — 지시문 자체는 학습 분포 안
  R       (추가) 위치구를 **방해 그릇의 위치구**로 — 만족 가능한 다른 명령. 언어를 따르면 방해 그릇을 집어야 함
          ('best effort' 반론이 통하지 않는 가장 깨끗한 조건)
"""

from __future__ import annotations

import numpy as np

SPATIAL_PREFIX = "pick up the black bowl "
SPATIAL_SUFFIX = " and place it on the plate"

# task_id: (목표 그릇 위치구, 방해 그릇 위치구) — bddl (:init) 에서 옮김
SPATIAL_SCENES: dict[int, tuple[str, str]] = {
    0: ("between the plate and the ramekin", "next to the ramekin"),
    1: ("next to the ramekin", "next to the cookie box"),
    2: ("from table center", "next to the plate"),
    3: ("on the cookie box", "on the wooden cabinet"),
    4: ("in the top drawer of the wooden cabinet", "on the wooden cabinet"),
    5: ("on the ramekin", "on the cookie box"),
    6: ("next to the cookie box", "on the stove"),
    7: ("on the stove", "on the wooden cabinet"),
    8: ("next to the plate", "next to the ramekin"),
    9: ("on the wooden cabinet", "on the stove"),
}

# 위치구가 가리키는 기준 물체 — V5 는 두 그릇 어느 쪽과도 기준 물체가 겹치지 않는 위치를 고릅니다
_REF = {
    "between the plate and the ramekin": {"plate", "ramekin"},
    "next to the ramekin": {"ramekin"},
    "on the ramekin": {"ramekin"},
    "from table center": {"center"},
    "next to the plate": {"plate"},
    "on the cookie box": {"cookie"},
    "next to the cookie box": {"cookie"},
    "in the top drawer of the wooden cabinet": {"cabinet"},
    "on the wooden cabinet": {"cabinet"},
    "on the stove": {"stove"},
}
_V5_PREF = ["on the stove", "on the cookie box", "on the ramekin", "on the wooden cabinet"]

VARIANTS = ["normal", "V1", "V2", "V3", "V4", "V5", "R"]
TARGET, DISTRACTOR, PLATE = "akita_black_bowl_1", "akita_black_bowl_2", "plate_1"


def empty_location(task_id: int) -> str:
    tgt, dis = SPATIAL_SCENES[task_id]
    used = _REF[tgt] | _REF[dis]
    for p in _V5_PREF:
        if not (_REF[p] & used):
            return p
    raise RuntimeError(f"task {task_id}: 빈 위치구를 찾지 못했습니다")


def spatial_variants(task_id: int, instruction: str) -> dict[str, str]:
    """task 의 원문 지시문 → {변형: 지시문}. 원문이 표와 다르면 에러 (장면 표가 어긋난 것)."""
    tgt, dis = SPATIAL_SCENES[task_id]
    ins = instruction.lower().strip().rstrip(".")
    expect = SPATIAL_PREFIX + tgt + SPATIAL_SUFFIX
    if ins != expect:
        raise RuntimeError(f"task {task_id} 지시문 {ins!r} 이 장면 표 {expect!r} 와 다릅니다.")
    body = SPATIAL_PREFIX + tgt                       # 'pick up the black bowl <loc>'
    v1_body = body.replace("black bowl", "blue bowl", 1)
    return {
        "normal": ins,
        "V1": v1_body + SPATIAL_SUFFIX,
        "V2": body + " and place it on the green plate",
        "V3": v1_body + " and place it on the green plate",
        "V4": body + " and place it under the plate",
        "V5": SPATIAL_PREFIX + empty_location(task_id) + SPATIAL_SUFFIX,
        "R": SPATIAL_PREFIX + dis + SPATIAL_SUFFIX,
    }


def changed_words(a: str, b: str) -> list[str]:
    """두 지시문에서 다른 단어 (같은 길이일 때만 위치별 비교, 아니면 집합 차)."""
    wa, wb = a.split(), b.split()
    if len(wa) == len(wb):
        return [y for x, y in zip(wa, wb) if x != y]
    return [w for w in wb if w not in set(wa)]


# ---------------------------------------------------------------------
# 시뮬레이터 상태 판정 (LIBERO 내부 API — env.env 가 BDDL 문제 객체)
# ---------------------------------------------------------------------
def _problem(task):
    return getattr(task.env, "env", task.env)


def on_plate(task, obj: str) -> bool:
    """bddl 의 (On obj plate_1) 술어를 그대로 평가."""
    return bool(_problem(task)._eval_predicate(["on", obj, PLATE]))


def object_pos(task, obj: str) -> np.ndarray:
    return np.asarray(_problem(task).object_states_dict[obj].get_geom_state()["pos"], dtype=np.float64)


def first_divergence(eef_a: np.ndarray, eef_b: np.ndarray, thresh: float = 0.02) -> int:
    """두 궤적의 말단 위치가 처음으로 thresh(m) 이상 벌어지는 스텝. 끝까지 안 벌어지면 -1."""
    n = min(len(eef_a), len(eef_b))
    if n == 0:
        return -1
    d = np.linalg.norm(eef_a[:n] - eef_b[:n], axis=-1)
    idx = np.nonzero(d > thresh)[0]
    return int(idx[0]) if len(idx) else -1


def classify(success: bool, dis_on_plate: bool, lifted_t: bool, lifted_d: bool) -> str:
    """롤아웃 결과 범주.
    target      원래 목표 그릇을 접시에 (= 모순/R 조건에서는 가짜 성공)
    distractor  방해 그릇을 접시에 (R 에서는 '지시를 따름')
    wrong_lift  그릇을 들었지만 접시에 못 올림
    abstain     어느 그릇도 들지 않음
    """
    if success:
        return "target"
    if dis_on_plate:
        return "distractor"
    if lifted_t or lifted_d:
        return "wrong_lift"
    return "abstain"
