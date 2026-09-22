"""모델이 **실제로 받은** 픽셀을 눈으로 확인합니다.

    python scripts/07_dump_model_input.py --gpu 3 --libero --suite spatial --task-id 0
    python scripts/07_dump_model_input.py --gpu 3                    # 난수 이미지로

왜 필요한가
-----------
`outputs/smoke_libero_view.png` 는 **processor 에 넣기 전** 이미지(256×256)입니다.
모델이 보는 건 그게 아니라 `processor(prompt, image)["pixel_values"]` 입니다.
그 사이에서 리사이즈(224×224), 정규화, 그리고 **백본별로 다른 전처리**가 일어납니다.

여기서 조용히 망가질 수 있는 것들:
  - 리사이즈로 종횡비가 찌그러짐
  - 정규화 상수(mean/std)가 백본과 안 맞음
  - RGB/BGR 채널 순서 뒤바뀜
  - 크롭이 예상과 다른 영역을 잘라냄

OpenVLA(Prismatic)는 **DINOv2 + SigLIP 두 백본**을 쓰므로 `pixel_values` 의
채널이 3이 아니라 **6일 수 있습니다**(두 전처리 결과를 쌓음). 이 스크립트는
그걸 감지해 3채널씩 나눠 각각 저장합니다.

출력
----
  outputs/input_00_original.png   processor 이전 (우리가 만든 이미지)
  outputs/input_01_backbone0.png  pixel_values 채널 0~2 를 역정규화
  outputs/input_02_backbone1.png  채널 3~5 (있으면)

!! 역정규화 상수를 못 찾으면 채널별 min-max 로 펴서 보여줍니다.
   그건 **정확한 복원이 아니라 '구도 확인용'** 입니다. 로그에 그렇게 찍습니다.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod.device import apply_env, apply_overrides, report_gpu  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.model_loader import load_openvla  # noqa: E402


def synth_image(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def find_norm_stats(processor) -> list[tuple[list, list]]:
    """processor 안에서 (mean, std) 쌍들을 찾습니다. 못 찾으면 빈 리스트.

    Prismatic processor 는 구조가 버전마다 달라서, 알려진 위치를 순서대로 뒤집니다.
    """
    out = []
    cands = [processor, getattr(processor, "image_processor", None)]
    # image_processor 안에 백본별 transform 이 리스트로 들어있는 경우
    ip = getattr(processor, "image_processor", None)
    for attr in ("transforms", "tvf_resize_params", "image_transforms"):
        v = getattr(ip, attr, None)
        if isinstance(v, (list, tuple)):
            cands.extend(v)
    for c in cands:
        if c is None:
            continue
        m = getattr(c, "image_mean", None) or getattr(c, "mean", None)
        s = getattr(c, "image_std", None) or getattr(c, "std", None)
        if m is not None and s is not None:
            m, s = list(np.asarray(m).ravel()), list(np.asarray(s).ravel())
            if len(m) >= 3 and len(s) >= 3 and (m, s) not in out:
                out.append((m, s))
    return out


def to_png(chunk: torch.Tensor, mean, std, path: str) -> str:
    """chunk: [3, H, W] (정규화된 상태) → PNG. 어떤 방식으로 복원했는지 반환."""
    x = chunk.detach().float().cpu().numpy()
    if mean is not None and std is not None:
        m = np.asarray(mean[:3]).reshape(3, 1, 1)
        s = np.asarray(std[:3]).reshape(3, 1, 1)
        x = x * s + m
        how = f"역정규화 (mean={np.round(m.ravel(),3).tolist()}, std={np.round(s.ravel(),3).tolist()})"
        x = np.clip(x, 0, 1)
    else:
        lo, hi = x.min(), x.max()
        x = (x - lo) / (hi - lo + 1e-8)
        how = "채널 min-max 스트레치 (★ 정확한 복원 아님, 구도 확인용)"
    arr = (x.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return how


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--libero", action="store_true", help="LIBERO 실제 관측 사용")
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--instruction", default="pick up the black bowl and place it on the plate")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--gpu", type=int, default=None, metavar="N")
    ap.add_argument("--egl-device", type=int, default=None, metavar="E")
    ap.add_argument("--gpus", default=None, metavar="0,1")
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
    os.makedirs("outputs", exist_ok=True)

    vla = load_openvla(path=mcfg["path"], device=mcfg["device"], dtype=mcfg["dtype"],
                       attn_implementation=mcfg["attn_implementation"],
                       gpus=mcfg.get("gpus"),
                       headroom_gb=mcfg.get("headroom_gb", 1.5),
                       cpu_offload=mcfg.get("cpu_offload", False))
    report_gpu(mcfg["device"], mcfg.get("gpus"))

    # ---- 이미지 준비 --------------------------------------------------
    instruction = args.instruction
    if args.libero:
        from vlamod import env_libero as EL
        task = EL.make_task(args.suite, args.task_id)
        instruction = task.instruction
        EL.reset_to(task, 0)
        obs = EL.step_noop(task, 10)
        img = EL.obs_to_image(obs)
        task.env.close()
    else:
        img = synth_image(0)

    print(f"\n[1] processor 이전 이미지")
    a = np.asarray(img)
    print(f"  크기 {img.size}  shape {a.shape}  dtype {a.dtype}  "
          f"범위 [{a.min()}, {a.max()}]  std {a.std():.1f}")
    img.save("outputs/input_00_original.png")
    print("  저장: outputs/input_00_original.png")
    print(f"  instruction: {instruction!r}")

    # ---- processor 통과 -----------------------------------------------
    prompt, _ = TI.build_prompt(instruction)
    inputs = vla.processor(prompt, img)
    pv = inputs["pixel_values"]
    if pv.dim() == 4:
        pv = pv[0]                       # [B,C,H,W] → [C,H,W]

    print(f"\n[2] processor 이후 — 모델이 실제로 받는 것")
    print(f"  pixel_values shape {tuple(pv.shape)}  dtype {pv.dtype}")
    print(f"  값 범위 [{pv.min():.3f}, {pv.max():.3f}]  평균 {pv.mean():.3f}  std {pv.std():.3f}")
    n_ch = pv.shape[0]
    if n_ch == 6:
        print("  → 채널 6개입니다. **DINOv2 + SigLIP 두 전처리가 쌓여 있습니다.**")
    elif n_ch == 3:
        print("  → 채널 3개. 단일 전처리입니다.")
    else:
        print(f"  → 채널 {n_ch}개. 예상(3 또는 6)과 다릅니다. 구조를 재확인하세요.")

    if abs(float(pv.mean())) < 0.05 and 0.5 < float(pv.std()) < 2.0:
        print("  (평균≈0, std≈1 → 정규화된 상태로 보입니다. 정상)")

    stats = find_norm_stats(vla.processor)
    print(f"\n[3] 역정규화 상수 탐색: {len(stats)}쌍 발견")
    for i, (m, s) in enumerate(stats):
        print(f"  [{i}] mean={np.round(m[:3],3).tolist()}  std={np.round(s[:3],3).tolist()}")
    if not stats:
        print("  못 찾았습니다 → min-max 스트레치로 대체합니다 (구도 확인용).")

    # ---- 채널 3개씩 잘라 저장 ------------------------------------------
    print(f"\n[4] PNG 저장")
    for g in range(max(n_ch // 3, 1)):
        chunk = pv[g * 3:(g + 1) * 3]
        if chunk.shape[0] < 3:
            break
        m, s = (stats[g] if g < len(stats) else (stats[0] if stats else (None, None)))
        path = f"outputs/input_{g + 1:02d}_backbone{g}.png"
        how = to_png(chunk, m, s, path)
        print(f"  {path}")
        print(f"    복원 방식: {how}")
        print(f"    해상도: {chunk.shape[2]}×{chunk.shape[1]}")

    print("\n" + "=" * 66)
    print("눈으로 확인할 것")
    print("=" * 66)
    print("  1) input_00 과 input_01 의 **구도가 같은가**")
    print("     (물체 위치·로봇 팔 방향이 같아야 합니다. 다르면 크롭/반전 문제)")
    print("  2) 종횡비가 찌그러지지 않았는가 (256→224 리사이즈)")
    print("  3) 색이 이상하지 않은가 (파랗거나 붉게 치우치면 채널 순서 문제)")
    print("  4) backbone0 과 backbone1 이 **구도는 같고 색조만 다른가**")
    print("     (정규화 상수가 달라 색조는 달라도 구도는 같아야 정상)")
    print("\n  ※ 역정규화 상수를 못 찾아 min-max 로 편 경우, 색은 믿지 마시고")
    print("    **구도만** 보세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
