"""이미지 전처리 변형 — 학습 시 전처리와 맞추기 위한 후보들.

왜 이 파일이 필요한가
---------------------
LIBERO 롤아웃 성공률이 0% 로 나왔습니다 (400 스텝, 5 태스크).
OpenVLA 의 LIBERO 평가는 spatial suite 에 220 스텝을 쓰므로 스텝 부족이 아닙니다.
→ **전처리가 학습과 어긋났을 가능성**이 가장 큽니다.

가장 유력한 누락: **center crop**.
OpenVLA 는 LIBERO 체크포인트를 **이미지 증강(random crop)** 과 함께 파인튜닝했고,
평가 시에는 그 보정으로 **center crop 후 리사이즈**를 적용합니다
(`experiments/robot/openvla_utils.py` 의 `get_vla_action(..., center_crop=True)`).
crop_scale=0.9 → 각 변을 sqrt(0.9)≈0.9487 배로 중앙 크롭한 뒤 224 로 리사이즈.

!! 제 기억에 근거한 재현이라 **파라미터가 정확하다고 보장할 수 없습니다.**
   그래서 scripts/08_policy_sanity.py 가 변형들을 나란히 돌려 **경험적으로** 고릅니다.
"""

from __future__ import annotations

import math

from PIL import Image


def center_crop_resize(img: Image.Image, crop_scale: float = 0.9,
                       out: int = 224) -> Image.Image:
    """중앙을 sqrt(crop_scale) 비율로 잘라내고 out×out 으로 리사이즈."""
    w, h = img.size
    s = max(min(math.sqrt(crop_scale), 1.0), 0.05)
    nw, nh = int(round(w * s)), int(round(h * s))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((out, out), Image.BILINEAR)


# 이름 → (flip 방식, center crop 여부)
#   flip 은 env_libero.obs_to_image 가 아니라 여기서 직접 처리합니다.
VARIANTS = {
    "current":        dict(flip="rot180", crop=False),   # 지금 쓰는 것
    "rot180_crop":    dict(flip="rot180", crop=True),    # ★ 1순위 후보
    "vflip":          dict(flip="vflip",  crop=False),   # 상하만 반전
    "vflip_crop":     dict(flip="vflip",  crop=True),
    "none":           dict(flip="none",   crop=False),   # 원본 그대로
    "none_crop":      dict(flip="none",   crop=True),
}


def apply_variant(raw, name: str) -> Image.Image:
    """raw: obs['agentview_image'] (numpy HxWx3, 뒤집히지 않은 원본)"""
    import numpy as np
    spec = VARIANTS[name]
    a = np.asarray(raw)
    if spec["flip"] == "rot180":
        a = a[::-1, ::-1]
    elif spec["flip"] == "vflip":
        a = a[::-1]
    img = Image.fromarray(np.ascontiguousarray(a).astype("uint8"))
    if spec["crop"]:
        img = center_crop_resize(img)
    return img
