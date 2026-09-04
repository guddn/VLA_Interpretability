"""CLI 인자로 GPU / 체크포인트를 덮어쓰는 유틸.

모든 실행 스크립트가 공유합니다.

    python scripts/02_run_analysis.py --gpu 3
    python scripts/02_run_analysis.py --device cuda:3
    python scripts/02_run_analysis.py --device 3          # 숫자만 줘도 됨
    python scripts/02_run_analysis.py --device cpu        # 디버깅용 (매우 느림)

멀티 GPU 서버에서 주의할 점
---------------------------
1) **PyTorch 와 MuJoCo 는 GPU 를 따로 고릅니다.**
   모델을 cuda:3 에 올려도 LIBERO 의 EGL 렌더링은 기본적으로 0번 GPU 를 씁니다.
   `apply_overrides()` 가 `MUJOCO_EGL_DEVICE_ID` 를 같은 번호로 맞춰 줍니다.
   (MuJoCo 는 렌더 컨텍스트 생성 시점에 이 값을 읽으므로, `make_task()` 전에만
    설정되면 됩니다)
2) `CUDA_VISIBLE_DEVICES` 를 함께 쓰면 **번호가 재매핑**됩니다.
   `CUDA_VISIBLE_DEVICES=3` 상태에서는 그 GPU 가 `cuda:0` 이 됩니다.
   둘을 섞지 말고 한 가지 방식만 쓰세요. 이 모듈은 그 상황을 감지해 경고합니다.
"""

from __future__ import annotations

import os
import warnings


def normalize_device(value) -> str:
    """'3' / 3 / 'cuda:3' / 'cuda' / 'cpu' → 표준 device 문자열."""
    if value is None:
        raise ValueError("device 값이 None 입니다.")
    s = str(value).strip().lower()
    if s in ("cpu", "-1"):
        return "cpu"
    if s.isdigit():
        return f"cuda:{int(s)}"
    if s == "cuda":
        return "cuda:0"
    if s.startswith("cuda:"):
        idx = s.split(":", 1)[1]
        if not idx.isdigit():
            raise ValueError(f"device 형식이 잘못되었습니다: {value!r}")
        return f"cuda:{int(idx)}"
    raise ValueError(f"알 수 없는 device 값: {value!r}  (예: 0, cuda:0, cpu)")


def device_index(device: str) -> int | None:
    return None if device == "cpu" else int(device.split(":")[1])


def _check_visible_devices(device: str) -> None:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd or device == "cpu":
        return
    n_visible = len([x for x in cvd.split(",") if x.strip() != ""])
    idx = device_index(device)
    if idx is not None and idx >= n_visible:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={cvd} 로 {n_visible}개만 보이는데 {device} 를 요청했습니다.\n"
            f"  CUDA_VISIBLE_DEVICES 를 쓰면 번호가 0부터 다시 매겨집니다.\n"
            f"  → 둘 중 하나만 쓰세요: `CUDA_VISIBLE_DEVICES={cvd} ... --gpu 0`\n"
            f"    또는 CUDA_VISIBLE_DEVICES 를 해제하고 `--gpu <실제번호>`"
        )
    warnings.warn(
        f"CUDA_VISIBLE_DEVICES={cvd} 가 설정되어 있습니다. "
        f"{device} 는 그 목록 안에서의 상대 번호로 해석됩니다.",
        stacklevel=2,
    )


def apply_overrides(mcfg: dict, args) -> dict:
    """argparse 결과로 config 의 model 섹션을 덮어씁니다. mcfg 를 제자리 수정 후 반환.

    인식하는 인자 (없으면 조용히 건너뜀):
      --gpu / --device / --model / --unnorm-key
    """
    gpu = getattr(args, "gpu", None)
    dev = getattr(args, "device", None)

    if gpu is not None and dev is not None:
        raise SystemExit("--gpu 와 --device 는 같이 쓸 수 없습니다. 하나만 주세요.")

    chosen = gpu if gpu is not None else dev
    if chosen is not None:
        mcfg["device"] = normalize_device(chosen)

    mcfg["device"] = normalize_device(mcfg.get("device", "cuda:0"))
    _check_visible_devices(mcfg["device"])

    if getattr(args, "model", None):
        mcfg["path"] = args.model
    if getattr(args, "unnorm_key", None):
        mcfg["unnorm_key"] = args.unnorm_key

    # MuJoCo(EGL) 렌더링도 같은 GPU 로. LIBERO 환경 생성 전에만 설정되면 됩니다.
    idx = device_index(mcfg["device"])
    if idx is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(idx)

    print(f"[device] model={mcfg['device']}  render(EGL)={os.environ.get('MUJOCO_EGL_DEVICE_ID', '-')}")
    print(f"[model ] {mcfg['path']}  unnorm_key={mcfg.get('unnorm_key')}")
    return mcfg


def report_gpu(device: str) -> None:
    """실제로 그 GPU 가 잡혔는지 확인 출력. 모델 로드 직후에 부르세요."""
    import torch

    if device == "cpu":
        print("[gpu   ] cpu 모드 — 매우 느립니다. 디버깅용으로만 쓰세요.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 를 못 찾았습니다.")
    idx = device_index(device)
    n = torch.cuda.device_count()
    if idx >= n:
        raise RuntimeError(f"GPU {idx} 를 요청했지만 보이는 GPU 는 {n}개 (0~{n - 1}) 입니다.")
    free, total = torch.cuda.mem_get_info(idx)
    print(
        f"[gpu   ] cuda:{idx} {torch.cuda.get_device_name(idx)}  "
        f"사용가능 {free / 1e9:.1f}GB / 전체 {total / 1e9:.1f}GB"
    )
    if free < 18e9:
        warnings.warn(
            f"cuda:{idx} 의 여유 메모리가 {free / 1e9:.1f}GB 입니다. "
            "OpenVLA-7B(bf16) 는 약 16GB + attention 버퍼가 필요합니다.",
            stacklevel=2,
        )
