"""EGL 이 실제로 무엇을 잡고 있는지 확정합니다.

    python setup/07_egl_probe.py

왜 필요한가
-----------
`MUJOCO_EGL_DEVICE_ID` 는 **CUDA 번호가 아닙니다.** EGL 은 자기만의 디바이스
목록을 갖고, 그 길이가 CUDA 디바이스 수와 다를 수 있습니다. 실제로 GPU 10장짜리
서버에서 EGL 이 1개만 열거하는 경우가 있습니다:

    RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be
    an integer between 0 and 0 (inclusive), got 8.

이 스크립트는 세 가지를 확인합니다.

  A) EGL 디바이스가 몇 개인가
  B) 각 EGL 디바이스가 **어느 CUDA GPU** 인가 (EGL_CUDA_DEVICE_NV 확장)
  C) 렌더러가 진짜 NVIDIA GPU 인가, 아니면 **소프트웨어(llvmpipe/swrast)** 인가
     ← 이게 소프트웨어면 롤아웃이 몇 배 느려집니다. 결과는 맞지만 시간이 갑니다.

확인되지 않는 항목은 "확인 불가" 로 적습니다. 추측해서 단정하지 않습니다.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

# EGL 확장 상수 (Khronos / NVIDIA 등록값)
EGL_CUDA_DEVICE_NV = 0x323A          # EGL_NV_device_cuda
EGL_DRM_DEVICE_FILE_EXT = 0x3233     # EGL_EXT_device_drm


def sh(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(실행 실패: {e})"


def main() -> int:  # noqa: C901
    print("=" * 74)
    print("A) 시스템 EGL 구성")
    print("=" * 74)

    libs = sh(["bash", "-c", "ldconfig -p | grep -i 'libEGL\\|libGLX\\|libOSMesa' || true"])
    print("  ldconfig | libEGL / libGLX / libOSMesa:")
    for line in (libs.splitlines() or ["  (없음)"]):
        print(f"    {line.strip()}")
    if "libEGL_nvidia" not in libs:
        print("\n    ★ libEGL_nvidia 가 없습니다 → **NVIDIA GPU 렌더링 불가**")
        print("      이 서버에는 CUDA(연산) 스택만 설치되어 있고 그래픽(OpenGL/EGL)")
        print("      스택이 빠져 있습니다. 드라이버를 --no-opengl-files 로 설치했거나,")
        print("      컨테이너에 NVIDIA_DRIVER_CAPABILITIES 가 compute 만 잡힌 경우입니다.")
        print("      → Mesa 소프트웨어 렌더링(llvmpipe)으로 동작합니다. 느리지만 결과는 같습니다.")
        print("      → GPU 렌더링을 원하면 관리자에게 요청해야 합니다 (README 참고).")

    vend_dir = "/usr/share/glvnd/egl_vendor.d"
    print(f"\n  {vend_dir}:")
    if os.path.isdir(vend_dir):
        for f in sorted(os.listdir(vend_dir)):
            print(f"    {f}")
    else:
        print("    (디렉토리 없음)")
    forced = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES")
    if forced:
        print(f"  __EGL_VENDOR_LIBRARY_FILENAMES={forced}  ← 벤더가 강제 지정됨")

    print(f"\n  MUJOCO_EGL_DEVICE_ID = {os.environ.get('MUJOCO_EGL_DEVICE_ID', '(설정 안 됨)')}")

    # ---------------------------------------------------------------- A2
    print("\n" + "-" * 74)
    print("A2) NVIDIA 그래픽 스택이 디스크에 존재하는가  ← root 없는 우회의 전제")
    print("-" * 74)

    print("  /dev/dri (DRM 렌더 노드):")
    if os.path.isdir("/dev/dri"):
        for f in sorted(os.listdir("/dev/dri")):
            print(f"    {f}")
    else:
        print("    (없음)  ← 그래픽용 디바이스 노드 자체가 노출되지 않았습니다")

    print("\n  커널 모듈:")
    lsmod = sh(["bash", "-c", "lsmod | grep -i nvidia || true"])
    for line in (lsmod.splitlines() or ["    (nvidia 모듈 안 보임)"]):
        print(f"    {line}")
    if "nvidia_drm" not in lsmod and "nvidia_modeset" not in lsmod:
        print("    ★ nvidia_drm / nvidia_modeset 이 없습니다 → 그래픽 기능 미탑재 커널 구성")

    print("\n  NVIDIA GL/EGL 라이브러리 파일 탐색:")
    found = sh(["bash", "-c",
                "ls -1 /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* "
                "/usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so* "
                "/usr/lib/x86_64-linux-gnu/libnvidia-glcore.so* "
                "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* 2>/dev/null || true"])
    have_nv_egl = False
    if found.strip():
        for line in found.splitlines():
            print(f"    {line}")
        have_nv_egl = "libEGL_nvidia" in found
    else:
        print("    (하나도 없음)")

    if have_nv_egl:
        print("\n  ✓ libEGL_nvidia 가 디스크에 있습니다! 벤더 JSON 만 없는 상태입니다.")
        print("    → **root 없이** 아래로 우회할 수 있습니다:")
        print("""
        mkdir -p ~/.config/egl_vendor.d
        cat > ~/.config/egl_vendor.d/10_nvidia.json <<'EOF'
        {"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
        EOF
        export __EGL_VENDOR_LIBRARY_FILENAMES=$HOME/.config/egl_vendor.d/10_nvidia.json
        python setup/07_egl_probe.py     # 디바이스 수가 늘어나는지 확인
        """)
    else:
        print("\n  ★ NVIDIA EGL 라이브러리가 디스크에 아예 없습니다.")
        print("    → root 없이 고칠 방법이 없습니다. 관리자 요청이 필요합니다.")
        print("    → 그동안은 Mesa 소프트웨어 렌더링으로 진행하시면 됩니다 (결과 동일).")

    # -----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("B) EGL 디바이스 열거")
    print("=" * 74)
    try:
        from OpenGL import EGL
    except Exception as e:  # noqa: BLE001
        print(f"  PyOpenGL EGL import 실패: {type(e).__name__}: {e}")
        return 1

    # !! eglQueryDevicesEXT 는 PyOpenGL 버전에 따라 OpenGL.EGL 최상위에 없습니다.
    #    device.py 의 헬퍼가 알려진 경로를 전부 시도합니다.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vlamod.device import egl_devices, probe_egl_devices_via_robosuite

    devices = egl_devices()
    if devices is None:
        print("  PyOpenGL 로는 열거하지 못했습니다. robosuite 에게 직접 물어봅니다...")
        n = probe_egl_devices_via_robosuite()
        if n is None:
            print("  ★ EGL 디바이스 열거에 완전히 실패했습니다.")
            print("    MUJOCO_EGL_DEVICE_ID 는 0 만 쓰세요.")
            return 1
        print(f"  robosuite 기준 EGL 디바이스 수 = {n}")
        print("  (핸들을 못 얻어 C 섹션의 상세 조회는 생략합니다)")
        return 0

    print(f"  EGL 디바이스 수 = {len(devices)}")
    print(f"  → MUJOCO_EGL_DEVICE_ID 로 쓸 수 있는 값은 0 ~ {len(devices) - 1} 뿐입니다.")

    try:
        import torch
        n_cuda = torch.cuda.device_count()
        print(f"  (참고) CUDA 디바이스 수 = {n_cuda}")
        if n_cuda != len(devices):
            print(f"  ★ 개수가 다릅니다 ({n_cuda} vs {len(devices)}). "
                  f"CUDA 번호를 EGL 에 그대로 쓰면 안 됩니다.")
    except Exception:  # noqa: BLE001
        n_cuda = None

    print("\n  각 EGL 디바이스가 어느 GPU 인지:")
    mapping = {}
    for i, dev in enumerate(devices):
        cuda_id, drm = "확인 불가", "확인 불가"
        try:
            v = EGL.EGLAttrib()
            if EGL.eglQueryDeviceAttribEXT(dev, EGL_CUDA_DEVICE_NV, ctypes.byref(v)):
                cuda_id = int(v.value)
                mapping[i] = cuda_id
        except Exception:  # noqa: BLE001
            pass
        try:
            s = EGL.eglQueryDeviceStringEXT(dev, EGL_DRM_DEVICE_FILE_EXT)
            if s:
                drm = s.decode() if isinstance(s, bytes) else str(s)
        except Exception:  # noqa: BLE001
            pass
        print(f"    egl:{i}  CUDA 디바이스={cuda_id}   DRM={drm}")

    if not mapping:
        print("    ※ EGL_CUDA_DEVICE_NV 조회가 안 됩니다. 드라이버가 이 확장을")
        print("      지원하지 않거나 NVIDIA EGL 이 아닐 수 있습니다.")

    # -----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("C) 실제 렌더러 확인 (컨텍스트를 만들어 GL_RENDERER 를 읽습니다)")
    print("=" * 74)
    target = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
    if target >= len(devices):
        print(f"  MUJOCO_EGL_DEVICE_ID={target} 가 범위를 벗어나 0 으로 시험합니다.")
        target = 0
    # !! EGL_PLATFORM_DEVICE_EXT / eglGetPlatformDisplayEXT 도 PyOpenGL 최상위에
    #    없을 수 있습니다. 확장 모듈 → 최상위 → 리터럴 순으로 폴백합니다.
    plat_const, get_disp = None, None
    try:
        from OpenGL.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT as _P
        plat_const = _P
    except Exception:  # noqa: BLE001
        plat_const = getattr(EGL, "EGL_PLATFORM_DEVICE_EXT", 0x313F)
    try:
        from OpenGL.EGL.EXT.platform_base import eglGetPlatformDisplayEXT as _G
        get_disp = _G
    except Exception:  # noqa: BLE001
        get_disp = getattr(EGL, "eglGetPlatformDisplayEXT", None)
    if get_disp is None:
        print("  eglGetPlatformDisplayEXT 를 못 찾았습니다.")
        print("  → 대신 05_diag_render.py 를 돌리세요. robosuite 가 컨텍스트를 만든 뒤")
        print("    GL_RENDERER 를 읽어 알려줍니다 (더 확실한 경로입니다).")
        return 0
    try:
        disp = get_disp(plat_const, devices[target], None)
        major, minor = EGL.EGLint(), EGL.EGLint()
        if not EGL.eglInitialize(disp, ctypes.byref(major), ctypes.byref(minor)):
            raise RuntimeError("eglInitialize 실패")
        print(f"  egl:{target} 초기화 성공  EGL {major.value}.{minor.value}")
        for name, const in (("EGL_VENDOR", EGL.EGL_VENDOR),
                            ("EGL_VERSION", EGL.EGL_VERSION)):
            try:
                s = EGL.eglQueryString(disp, const)
                print(f"    {name:12s} = {s.decode() if isinstance(s, bytes) else s}")
            except Exception:  # noqa: BLE001
                print(f"    {name:12s} = 확인 불가")

        # OpenGL 컨텍스트까지 만들어 GL_RENDERER 를 읽습니다
        try:
            EGL.eglBindAPI(EGL.EGL_OPENGL_API)
            cfg_attrs = [EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
                         EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
                         EGL.EGL_NONE]
            arr = (EGL.EGLint * len(cfg_attrs))(*cfg_attrs)
            cfg = (EGL.EGLConfig * 1)()
            ncfg = EGL.EGLint()
            EGL.eglChooseConfig(disp, arr, cfg, 1, ctypes.byref(ncfg))
            ctx = EGL.eglCreateContext(disp, cfg[0], EGL.EGL_NO_CONTEXT, None)
            EGL.eglMakeCurrent(disp, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, ctx)
            from OpenGL import GL
            for name, const in (("GL_VENDOR", GL.GL_VENDOR),
                                ("GL_RENDERER", GL.GL_RENDERER),
                                ("GL_VERSION", GL.GL_VERSION)):
                s = GL.glGetString(const)
                s = s.decode() if isinstance(s, bytes) else str(s)
                print(f"    {name:12s} = {s}")
                if name == "GL_RENDERER":
                    low = s.lower()
                    if any(k in low for k in ("llvmpipe", "softpipe", "swrast", "mesa")):
                        print("    ★ 소프트웨어 렌더러입니다 (CPU). 동작은 하지만 매우 느립니다.")
                        print("      Stage 3 롤아웃 시간이 몇 배로 늘어납니다.")
                    elif "nvidia" in low or "quadro" in low or "rtx" in low:
                        print("    ✓ NVIDIA GPU 렌더링입니다.")
        except Exception as e:  # noqa: BLE001
            print(f"    GL 컨텍스트 생성 실패: {type(e).__name__}: {e}")
            print("    (EGL 디스플레이는 열렸으니 robosuite 는 동작할 수 있습니다)")
    except Exception as e:  # noqa: BLE001
        print(f"  egl:{target} 초기화 실패: {type(e).__name__}: {e}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("결론")
    print("=" * 74)
    print(f"  --egl-device 에 쓸 수 있는 값: 0 ~ {len(devices) - 1}")
    if len(devices) == 1:
        print("  선택지가 없습니다. 0 을 쓰세요 (코드가 자동으로 그렇게 합니다).")
        print("  torch 는 --gpu 로 원하는 GPU 를 쓰고, 렌더링만 이 하나를 씁니다.")
        print("  → 모델 연산과 렌더링이 다른 카드에서 일어나도 결과는 동일합니다.")
    elif mapping:
        print("  EGL ↔ CUDA 매핑이 확인되었습니다:")
        for e, c in sorted(mapping.items()):
            print(f"    --egl-device {e}  →  cuda:{c}")
        print("  --gpu 와 같은 카드에서 렌더링하려면 위 표를 보고 지정하세요.")
    else:
        print("  매핑을 확인하지 못했습니다. --egl-device 를 바꿔가며 이 스크립트를")
        print("  다시 돌리고, 그때 다른 터미널에서 nvidia-smi 로 확인하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
