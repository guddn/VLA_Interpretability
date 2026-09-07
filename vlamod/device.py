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
import sys
import warnings


# =====================================================================
# 환경변수 — configs 의 env 섹션에서 읽습니다 (.bashrc 를 고칠 필요 없음)
# =====================================================================
def apply_env(cfg: dict, args=None) -> dict:
    """config 의 `env:` 섹션으로 프로세스 환경변수를 설정합니다.

    **반드시 transformers / huggingface_hub / libero 를 import 하기 전에** 부르세요.
    HF 캐시 경로는 huggingface_hub 가 import 되는 시점에 확정되므로,
    그 뒤에 설정하면 조용히 무시됩니다.

    우선순위: CLI 인자 > config > 기존 환경변수 > 시스템 기본

    반환: 실제로 적용된 값 dict (로그용)
    """
    ecfg = dict((cfg or {}).get("env") or {})
    applied: dict[str, str] = {}

    # --- HuggingFace 캐시 -------------------------------------------
    hf_home = getattr(args, "hf_home", None) or ecfg.get("hf_home")
    if hf_home:
        if "huggingface_hub" in sys.modules:
            warnings.warn(
                "huggingface_hub 가 이미 import 되었습니다. HF_HOME 설정이 반영되지 않을 수 "
                "있습니다. apply_env() 를 스크립트 맨 앞에서 호출하세요.",
                stacklevel=2,
            )
        root = os.path.abspath(os.path.expanduser(str(hf_home)))
        try:
            os.makedirs(os.path.join(root, "hub"), exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"hf_home 경로를 만들 수 없습니다: {root} ({e})") from e
        os.environ["HF_HOME"] = root
        os.environ["HF_HUB_CACHE"] = os.path.join(root, "hub")
        applied["HF_HOME"] = root

    # --- 렌더링 -----------------------------------------------------
    for key, envname in (("mujoco_gl", "MUJOCO_GL"),
                         ("pyopengl_platform", "PYOPENGL_PLATFORM")):
        val = ecfg.get(key)
        if val:
            os.environ[envname] = str(val)
            applied[envname] = str(val)

    # --- GPU 번호 체계 ------------------------------------------------
    # !! CUDA 의 기본 정렬은 CUDA_DEVICE_ORDER=FASTEST_FIRST 입니다.
    #    카드 모델이 섞인 서버에서는 torch 의 cuda:N 이 nvidia-smi / nvtop 의
    #    GPU N 과 **다른 물리 카드**를 가리킵니다.
    #    PCI_BUS_ID 로 두면 nvidia-smi 와 번호가 일치합니다.
    #    CUDA 런타임 초기화(첫 CUDA 호출) 전에만 설정되면 되므로,
    #    import torch 뒤라도 CUDA 를 아직 안 건드렸으면 유효합니다.
    order = ecfg.get("cuda_device_order", "PCI_BUS_ID")
    if order:
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_initialized():
                warnings.warn(
                    "CUDA 가 이미 초기화되어 CUDA_DEVICE_ORDER 변경이 반영되지 않습니다. "
                    "apply_env() 를 CUDA 호출 전에 부르세요.",
                    stacklevel=2,
                )
        except ImportError:
            pass
        os.environ["CUDA_DEVICE_ORDER"] = str(order)
        applied["CUDA_DEVICE_ORDER"] = str(order)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        # 스케줄러나 .bashrc 가 걸어둔 경우가 많습니다. 이게 있으면 번호가
        # **그 목록 안에서 0부터** 다시 매겨지므로 nvtop 번호와 어긋납니다.
        print(f"[env   ] !! CUDA_VISIBLE_DEVICES={cvd} 가 설정되어 있습니다. "
              f"GPU 번호는 이 목록 안의 상대 번호입니다 (cuda:0 = 실제 {cvd.split(',')[0]}번).")

    if applied:
        print("[env   ] " + "  ".join(f"{k}={v}" for k, v in applied.items()))
    return applied


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


def parse_gpus(value) -> list[int]:
    """'0,1' / '0 1' / [0,1] → [0, 1]"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = str(value).replace(" ", ",").split(",")
    out = []
    for it in items:
        it = str(it).strip()
        if it == "":
            continue
        if not it.isdigit():
            raise ValueError(f"--gpus 는 숫자 목록이어야 합니다: {value!r}")
        out.append(int(it))
    if len(set(out)) != len(out):
        raise ValueError(f"--gpus 에 중복이 있습니다: {value!r}")
    return out


def egl_devices() -> list | None:
    """EGL 디바이스 **핸들 목록**. 알 수 없으면 None.

    !! `OpenGL.EGL.eglQueryDevicesEXT` 는 PyOpenGL 버전에 따라 **없습니다.**
       (AttributeError: module 'OpenGL.EGL' has no attribute 'eglQueryDevicesEXT')
       확장 심볼은 서브모듈에 있고, 위치도 버전마다 다릅니다.
       그래서 알려진 경로를 순서대로 전부 시도합니다.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # 1) 확장 모듈 직접 import (가장 흔한 경로)
    try:
        from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT  # noqa: PLC0415
        return list(eglQueryDevicesEXT())
    except Exception:  # noqa: BLE001
        pass
    # 2) device_base 의 헬퍼
    try:
        from OpenGL.EGL.EXT.device_base import egl_get_devices  # noqa: PLC0415
        return list(egl_get_devices())
    except Exception:  # noqa: BLE001
        pass
    # 3) 최상위 네임스페이스 (신버전 PyOpenGL)
    try:
        from OpenGL import EGL  # noqa: PLC0415
        return list(EGL.eglQueryDevicesEXT())
    except Exception:  # noqa: BLE001
        pass
    # 4) ctypes 로 직접
    try:
        import ctypes  # noqa: PLC0415
        from OpenGL import EGL  # noqa: PLC0415
        n = EGL.EGLint()
        EGL.eglQueryDevicesEXT(0, None, ctypes.byref(n))
        buf = (EGL.EGLDeviceEXT * n.value)()
        EGL.eglQueryDevicesEXT(n.value, buf, ctypes.byref(n))
        return list(buf)[: n.value]
    except Exception:  # noqa: BLE001
        return None


def probe_egl_devices_via_robosuite() -> int | None:
    """최후의 수단 — robosuite 자신에게 물어봅니다.

    범위를 크게 벗어난 값을 주면 robosuite 가
      "must be an integer between 0 and N (inclusive)"
    라고 알려 줍니다. 그 N 을 파싱합니다.
    우리가 실제로 타는 코드 경로와 **정확히 같은** 답이라는 게 장점입니다.
    """
    import re  # noqa: PLC0415
    saved = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    try:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "9999"
        from robosuite.renderers.context.egl_context import (  # noqa: PLC0415
            create_initialized_egl_device_display,
        )
        create_initialized_egl_device_display(device_id=9999)
        return None
    except RuntimeError as e:
        m = re.search(r"between 0 and (\d+)", str(e))
        return int(m.group(1)) + 1 if m else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        if saved is None:
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        else:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = saved


def probe_egl_devices() -> int | None:
    """EGL 이 열거하는 디바이스 **개수**를 반환합니다. 알 수 없으면 None.

    !! 이건 CUDA 디바이스 수와 전혀 다를 수 있습니다.
       실제로 GPU 10장짜리 서버에서 EGL 은 1개만 열거하는 경우가 있습니다
       (드라이버/컨테이너 구성, 또는 NVIDIA EGL 대신 Mesa 가 잡힌 경우).
       그때 MUJOCO_EGL_DEVICE_ID 에 8 을 넣으면 robosuite 가 이렇게 죽습니다:

         RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be
         an integer between 0 and 0 (inclusive), got 8.

    실패해도 예외를 올리지 않습니다 — 모르면 None 을 주고 호출부가 0 으로 갑니다.
    """
    devs = egl_devices()
    if devs is not None:
        return len(devs)
    return probe_egl_devices_via_robosuite()


def resolve_egl_device(cuda_idx: int | None, explicit=None) -> tuple[int, str]:
    """렌더링에 쓸 EGL 디바이스 번호를 정합니다. (번호, 사유) 를 돌려줍니다.

    우선순위: explicit(--egl-device) > EGL 열거 결과에 근거한 추정 > 0
    """
    if explicit is not None:
        return int(explicit), "명시 지정(--egl-device)"
    n = probe_egl_devices()
    if n is None:
        return 0, "EGL 열거 실패 → 0 으로 폴백"
    if n <= 1:
        return 0, f"EGL 디바이스가 {n}개뿐 → 선택 불가"
    if cuda_idx is not None and cuda_idx < n:
        return cuda_idx, f"EGL {n}개 중 cuda 번호와 동일하게 추정 (미검증)"
    return 0, f"cuda:{cuda_idx} 는 EGL 범위(0~{n - 1}) 밖 → 0 으로 폴백"


def apply_overrides(mcfg: dict, args) -> dict:
    """argparse 결과로 config 의 model 섹션을 덮어씁니다. mcfg 를 제자리 수정 후 반환.

    인식하는 인자 (없으면 조용히 건너뜀):
      --gpu / --gpus / --device / --model / --unnorm-key
    """
    gpu = getattr(args, "gpu", None)
    dev = getattr(args, "device", None)
    gpus = parse_gpus(getattr(args, "gpus", None))

    n_given = sum(x is not None and x != [] for x in (gpu, dev, gpus or None))
    if n_given > 1:
        raise SystemExit("--gpu / --gpus / --device 중 하나만 쓰세요.")

    if gpus:
        if len(gpus) == 1:
            gpus, gpu = [], gpus[0]
        else:
            mcfg["gpus"] = gpus
            mcfg["device"] = f"cuda:{gpus[0]}"

    chosen = gpu if gpu is not None else dev
    if chosen is not None:
        mcfg["device"] = normalize_device(chosen)

    mcfg.setdefault("gpus", [])
    mcfg["device"] = normalize_device(mcfg.get("device", "cuda:0"))
    _check_visible_devices(mcfg["device"])

    if getattr(args, "model", None):
        mcfg["path"] = args.model
    if getattr(args, "unnorm_key", None):
        mcfg["unnorm_key"] = args.unnorm_key
    if getattr(args, "headroom_gb", None) is not None:
        mcfg["headroom_gb"] = float(args.headroom_gb)
    mcfg.setdefault("headroom_gb", 1.5)
    if getattr(args, "allow_cpu_offload", False):
        mcfg["cpu_offload"] = True
    mcfg.setdefault("cpu_offload", False)

    # --- MuJoCo(EGL) 렌더링 GPU --------------------------------------
    # !! CUDA 번호를 그대로 쓰면 안 됩니다. EGL 은 완전히 별개의 짧은 목록입니다.
    #    LIBERO 환경 생성 전에만 설정되면 됩니다.
    idx = device_index(mcfg["device"])
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    egl_explicit = getattr(args, "egl_device", None)
    if egl_explicit is None:
        egl_explicit = mcfg.get("egl_device")
    egl_id, why = resolve_egl_device(idx, egl_explicit)
    mcfg["egl_device"] = egl_id
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_id)

    shard = f"  shard={mcfg['gpus']}" if mcfg.get("gpus") else ""
    print(f"[device] model={mcfg['device']}{shard}  "
          f"render(EGL)=egl:{egl_id}  ({why})")
    print(f"[model ] {mcfg['path']}  unnorm_key={mcfg.get('unnorm_key')}")
    return mcfg


def report_gpu(device: str, gpus: list[int] | None = None) -> None:
    """실제로 그 GPU 가 잡혔는지 확인 출력. 모델 로드 직후에 부르세요."""
    import torch

    if gpus:
        total_free = 0.0
        for i in gpus:
            free, total = torch.cuda.mem_get_info(i)
            alloc = torch.cuda.memory_allocated(i)
            total_free += free
            print(f"[gpu   ] cuda:{i} {torch.cuda.get_device_name(i)}  "
                  f"이 프로세스 {alloc/1e9:.1f}GB / 남은 여유 {free/1e9:.1f}GB")
        print(f"[gpu   ] 합계 여유 {total_free/1e9:.1f}GB (분할 로드)")
        return

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
