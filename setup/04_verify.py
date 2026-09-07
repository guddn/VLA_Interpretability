"""Stage 1 최종 검증 — 설치를 다시 하지 않고 **상태만** 점검합니다.

**A6000 서버(Linux)에서 실행하세요.** 노트북(Windows)의 역할은 `pytest` 뿐입니다.

    python setup/04_verify.py                 # 전체 (GPU 0)
    python setup/04_verify.py --gpu 4         # 특정 GPU 로
    python setup/04_verify.py --gpus 0,6      # 분할 로드용 GPU 들의 합계 여유를 검사
    python setup/04_verify.py --skip-render    # 렌더링 테스트 생략

--gpus 에 대해
--------------
이 스크립트는 **모델을 올리지 않습니다.** 따라서 `--gpus` 는 실제 분할 로드를
시험하지 않고, 지정한 GPU 들이 보이는지 + 합계 여유 메모리가 7B bf16 을
감당하는지만 확인합니다. 분할 로드 자체의 성공 여부는 `scripts/01_smoke_forward.py`
에서 처음 판명됩니다.

렌더링(EGL)은 나눠 쓸 수 없으므로 **첫 번째 GPU** 에 고정됩니다.

왜 이 파일이 따로 있는가
------------------------
`02_libero.sh` 의 검증은 `benchmark.get_benchmark_dict()` 까지만 봤는데,
**그건 파이썬 dict 를 읽는 것뿐이라 EGL 을 전혀 건드리지 않습니다.**
그게 통과해도 실제 렌더링은 죽을 수 있습니다.

여기서는 `OffScreenRenderEnv` 를 실제로 띄우고 **프레임을 한 장 뽑아
PNG 로 저장**합니다. 이게 통과하면 Stage 2 로 넘어가도 됩니다.

각 항목은 독립적으로 검사하고, 실패해도 계속 진행한 뒤 마지막에 요약합니다.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """데코레이터: 예외를 잡아 결과 목록에 기록."""
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                msg = fn(*a, **kw)
                RESULTS.append((name, True, msg or "OK"))
                print(f"  [ OK ] {name}: {msg or ''}")
                return True
            except Exception as e:  # noqa: BLE001
                # !! bare assert (메시지 없는 AssertionError) 가 라이브러리 내부에서
                #    나면 str(e) 가 빈 문자열이라 아무 정보도 안 남습니다.
                #    그래서 **항상** 마지막 프레임(파일:줄:함수)과 그 소스 줄을 붙입니다.
                detail = str(e).strip()
                where = ""
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    f = tb[-1]
                    where = f"  ← {os.path.basename(f.filename)}:{f.lineno} in {f.name}()"
                    if f.line:
                        where += f"\n           {f.line.strip()}"
                msg = f"{type(e).__name__}: {detail}" if detail else type(e).__name__
                RESULTS.append((name, False, msg + where))
                print(f"  [FAIL] {name}: {msg}{where}")
                if os.environ.get("VERIFY_TRACE"):
                    traceback.print_exc()
                return False
        return wrapped
    return deco


# ---------------------------------------------------------------- 개별 검사
@check("python")
def c_python():
    v = sys.version_info
    note = "" if v[:2] == (3, 10) else "  ← OpenVLA 는 3.10 권장 (치명적이진 않음)"
    return f"{v.major}.{v.minor}.{v.micro}{note}"


@check("PIP_CONSTRAINT")
def c_constraint():
    """이게 안 걸려 있으면 다음 pip 설치가 numpy 를 다시 2.x 로 올립니다.

    yaml(configs) 로는 해결되지 않습니다 — 그건 파이썬 프로세스 안의 설정이고
    pip 은 셸에서 따로 돌기 때문입니다. conda 의 activate.d 에 박는 것이 정답이고,
    setup/01_env.sh 가 자동으로 등록합니다.
    """
    p = os.environ.get("PIP_CONSTRAINT")
    assert p, (
        "설정되지 않았습니다. 다음 pip 설치가 numpy 를 2.x 로 올릴 수 있습니다.\n"
        "    영구 해결: bash setup/01_env.sh 를 한 번 더 돌리면 conda activate.d 에 등록합니다\n"
        "    임시 해결: export PIP_CONSTRAINT=$(pwd)/constraints.txt"
    )
    assert os.path.isfile(p), f"파일이 없습니다: {p}  (프로젝트를 옮기셨나요?)"
    return p


@check("numpy < 2")
def c_numpy():
    import numpy
    assert numpy.__version__.startswith("1."), (
        f"numpy {numpy.__version__} — torch 2.2.0 은 numpy 1.x 로 컴파일됨. "
        "조치: pip install 'numpy<2' --force-reinstall"
    )
    return numpy.__version__


@check("torch ↔ numpy 연동")
def c_torch_numpy():
    import torch
    _ = torch.randn(3).numpy()          # numpy 2 면 여기서 죽습니다
    return f"torch {torch.__version__}"


@check("CUDA 드라이버 ↔ torch 빌드")
def c_driver():
    """torch 의 CUDA 빌드가 드라이버보다 높으면 실행 시점에 죽습니다.

        RuntimeError: The NVIDIA driver on your system is too old (found version 11080)

    11080 = 드라이버가 지원하는 CUDA API 11.8 이라는 뜻입니다.
    torch cu121 은 드라이버 525 이상이 필요하므로, 구형 드라이버면 cu118 빌드를 써야 합니다.
    """
    import subprocess
    import torch

    built = torch.version.cuda or "?"
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        drv = "?"

    major = int(drv.split(".")[0]) if drv[:1].isdigit() else 0
    need = "cu121" if major >= 525 else "cu118"
    have = "cu121" if built.startswith("12") else "cu118" if built.startswith("11") else built

    assert have == need or (need == "cu121"), (
        f"드라이버 {drv} (CUDA {'12.x' if major >= 525 else '11.x'} 까지) 인데 "
        f"torch 는 {built} 빌드입니다.\n"
        f"    조치: pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 "
        f"--index-url https://download.pytorch.org/whl/{need} --force-reinstall"
    )
    return f"드라이버 {drv}, torch CUDA {built} ({have})"


WEIGHTS_GB = 15.0        # OpenVLA-7B bf16 가중치. model_loader.py 와 같은 값이어야 합니다


@check("CUDA")
def c_cuda(gpus: list[int], headroom_gb: float = 1.5):
    """gpus 가 1개면 단독 사용, 2개 이상이면 분할 로드를 가정하고 합계로 판정합니다."""
    import torch
    assert torch.cuda.is_available(), "CUDA 를 못 찾았습니다"
    n = torch.cuda.device_count()
    for g in gpus:
        assert g < n, f"GPU {g} 요청했으나 보이는 GPU 는 {n}개 (0~{n - 1})"

    lines, total_usable = [], 0.0
    for g in gpus:
        # get_device_name 은 CUDA 를 실제로 초기화합니다 — 드라이버가 낮으면 여기서 죽습니다
        name = torch.cuda.get_device_name(g)
        free, total = torch.cuda.mem_get_info(g)
        total_usable += max(free / 1e9 - headroom_gb, 0.0)
        lines.append(f"cuda:{g} {name} 여유 {free/1e9:.1f}/{total/1e9:.1f}GB")

    if len(gpus) == 1:
        free = torch.cuda.mem_get_info(gpus[0])[0] / 1e9
        # model_loader.load_openvla 의 단일 GPU 가드와 같은 기준
        assert free >= WEIGHTS_GB + 1.0, (
            f"cuda:{gpus[0]} 여유 {free:.1f}GB — 가중치 {WEIGHTS_GB:.0f}GB 를 못 올립니다.\n"
            f"    조치: 다른 GPU 를 쓰거나, --gpus 로 두 장에 나누세요"
        )
        note = "" if free >= WEIGHTS_GB + 3.0 else (
            "  ← 빠듯합니다 (attention 버퍼가 더 필요). 긴 롤아웃에서 OOM 가능"
        )
        return f"{n}장 보임 | " + lines[0] + note

    assert total_usable >= WEIGHTS_GB, (
        f"분할 사용가능 합계 {total_usable:.1f}GB — 가중치 {WEIGHTS_GB:.0f}GB 에 모자랍니다 "
        f"(headroom {headroom_gb}GB/장 차감 후).\n"
        f"    조치: --headroom-gb 0.8 로 낮추거나, 더 빈 GPU 를 쓰세요"
    )
    return (f"{n}장 보임 | " + " | ".join(lines)
            + f" | 분할 시 사용가능 합계 {total_usable:.1f}GB (가중치 {WEIGHTS_GB:.0f}GB)")


@check("opencv < 5")
def c_cv2():
    import cv2
    major = int(cv2.__version__.split(".")[0])
    assert major < 5, (
        f"opencv {cv2.__version__} — 5.x 는 numpy>=2 를 요구해 충돌합니다. "
        "조치: pip install 'opencv-python<5'"
    )
    return cv2.__version__


@check("transformers 4.40.x")
def c_transformers():
    import transformers
    assert transformers.__version__.startswith("4.40"), (
        f"{transformers.__version__} — 4.40.x 가 아니면 intervene.py 의 "
        "attention mask 훅이 깨집니다"
    )
    return transformers.__version__


@check("timm / tokenizers")
def c_misc():
    import timm, tokenizers
    return f"timm {timm.__version__}, tokenizers {tokenizers.__version__}"


@check("libero 패키지")
def c_libero():
    """LIBERO 최상위는 __init__.py 가 없는 **namespace package** 입니다.
    그래서 libero.__file__ 이 None 이고, dirname(None) 은 TypeError 를 냅니다.
    __path__ 로 폴백해야 합니다."""
    # !! LIBERO 디렉토리 안에서 import 하면 설치 여부와 무관하게 통과합니다(거짓 통과).
    #    검증 스크립트는 보통 프로젝트 루트에서 도니 문제없지만, 혹시를 대비해 경고합니다.
    import libero
    loc = getattr(libero, "__file__", None)
    if loc:
        return os.path.dirname(loc)
    paths = list(getattr(libero, "__path__", []))
    if paths:
        return f"{paths[0]}  (namespace package)"
    return "(namespace package, 경로 불명)"


@check("libero benchmark 목록")
def c_benchmark():
    from libero.libero import benchmark
    keys = list(benchmark.get_benchmark_dict().keys())
    assert "libero_spatial" in keys
    return ", ".join(keys)


@check("EGL 실제 렌더링 (★ 진짜 검사)")
def c_render(out_png: str):
    """OffScreenRenderEnv 를 띄워 프레임을 뽑습니다. 여기가 EGL 의 실제 관문입니다."""
    import numpy as np
    from PIL import Image
    from vlamod import env_libero as EL

    task = EL.make_task("spatial", 0)
    EL.reset_to(task, 0)
    obs = EL.step_noop(task, 5)
    img = EL.obs_to_image(obs)
    arr = np.asarray(img)
    task.env.close()

    assert arr.ndim == 3 and arr.shape[2] == 3, f"이상한 shape: {arr.shape}"
    assert arr.std() > 1.0, "프레임이 단색입니다 — 렌더링이 실패했을 수 있습니다"
    Image.fromarray(arr).save(out_png)
    return (f"{arr.shape[1]}x{arr.shape[0]}, std={arr.std():.1f}  →  {out_png}  "
            f"(**눈으로 로봇 팔 방향을 확인하세요**)")


@check("HF 캐시 / 체크포인트")
def c_hf(cfg: dict):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub = os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    st = os.statvfs(hub) if os.path.isdir(hub) else os.statvfs(os.path.expanduser("~"))
    free = st.f_bavail * st.f_frsize / 1e9

    want = cfg.get("model", {}).get("path", "")
    slug = "models--" + want.replace("/", "--")
    have = os.path.isdir(os.path.join(hub, slug))
    status = "체크포인트 있음" if have else f"체크포인트 없음 (setup/03 로 받으세요: {want})"
    return f"{hub}  여유 {free:.0f}GB  |  {status}"


# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gpu", type=int, default=None, metavar="N",
                    help="검사할 GPU 번호 (기본 0)")
    ap.add_argument("--gpus", default=None, metavar="0,6",
                    help="분할 로드에 쓸 GPU 목록. 합계 여유 메모리로 판정합니다. "
                         "EGL 렌더링은 첫 번째 GPU 에 붙습니다")
    ap.add_argument("--headroom-gb", type=float, default=1.5, metavar="G",
                    help="--gpus 판정 시 GPU 당 안전 마진(GB). 기본 1.5")
    ap.add_argument("--egl-device", type=int, default=None, metavar="E",
                    help="렌더링(EGL) 디바이스 번호. **CUDA 번호와 다른 체계**입니다. "
                         "미지정이면 자동 판정 (setup/07_egl_probe.py 로 목록 확인)")
    ap.add_argument("--hf-home", default=None,
                    help="HuggingFace 캐시 루트. config 의 env.hf_home 을 덮어씀")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--out", default="outputs/verify_render.png")
    ap.add_argument("--force", action="store_true",
                    help="Windows 등 비대상 환경에서도 강행")
    args = ap.parse_args()

    # --- 실행 위치 확인 -------------------------------------------------
    # 이 스크립트는 **A6000 서버(Linux)** 용입니다. 노트북에서는 CUDA/LIBERO 가
    # 없으므로 거의 전부 실패합니다. 노트북의 역할은 pytest 뿐입니다.
    if sys.platform.startswith("win"):
        print("!" * 66)
        print("!! Windows 에서 실행 중입니다.")
        print("!! 이 스크립트는 A6000 서버(Linux)에서 돌리는 것입니다.")
        print("!! 노트북에서 하실 것은 이것 하나뿐입니다:")
        print("!!     pytest -q tests/test_core.py")
        print("!!")
        print("!! 서버에서:")
        print("!!     ssh <서버> && cd <프로젝트> && conda activate vlamod")
        print("!!     python setup/04_verify.py --gpu <번호>")
        print("!" * 66)
        if not args.force:
            print("\n(그래도 강행하려면 --force 를 붙이세요)")
            return 2

    # --- config 읽기 ----------------------------------------------------
    try:
        import yaml
    except ModuleNotFoundError:
        print("!! pyyaml 이 없습니다.  pip install pyyaml")
        print("!! (설치 환경이 맞는지도 확인하세요 — conda activate vlamod)")
        return 1
    try:
        cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    except FileNotFoundError:
        print(f"!! config 를 못 찾았습니다: {args.config}")
        print("!! 프로젝트 루트에서 실행하고 있는지 확인하세요 (cd <프로젝트 루트>)")
        return 1

    # HF_HOME / MUJOCO_GL 을 config 에서 설정 (libero import 전에)
    from vlamod.device import apply_env, parse_gpus
    apply_env(cfg, args)

    # --- GPU 선택 ------------------------------------------------------
    gpu_list = parse_gpus(args.gpus)
    if gpu_list and args.gpu is not None:
        print("!! --gpu 와 --gpus 는 함께 쓸 수 없습니다.")
        return 1
    if not gpu_list:
        gpu_list = [0 if args.gpu is None else args.gpu]

    # !! MuJoCo(EGL) 는 PyTorch 와 **독립적으로** GPU 를 고르고, 디바이스 목록도
    #    완전히 별개입니다 (GPU 10장인데 EGL 은 1개만 열거하는 서버가 실제로 있습니다).
    #    CUDA 번호를 그대로 넣으면 robosuite 가 범위 초과로 죽습니다.
    from vlamod.device import probe_egl_devices, resolve_egl_device
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    n_egl = probe_egl_devices()
    egl_id, why = resolve_egl_device(gpu_list[0], args.egl_device)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_id)
    shard = f"  (분할 {gpu_list})" if len(gpu_list) > 1 else ""
    print(f"[device] 검사 대상 cuda:{gpu_list[0]}{shard}  "
          f"render=egl:{egl_id}  EGL 디바이스 수={n_egl}  ({why})")

    os.makedirs("outputs", exist_ok=True)

    print("\n=== Stage 1 검증 " + "=" * 48)
    c_python()
    c_constraint()
    c_numpy()
    c_torch_numpy()
    c_driver()
    c_cuda(gpu_list, args.headroom_gb)
    c_cv2()
    c_transformers()
    c_misc()
    ok_libero = c_libero()
    if ok_libero:
        c_benchmark()
        if not args.skip_render:
            c_render(args.out)
    c_hf(cfg)

    fails = [(n, m) for n, ok, m in RESULTS if not ok]
    print("\n" + "=" * 66)
    print(f"통과 {len(RESULTS) - len(fails)} / {len(RESULTS)}")
    if fails:
        print("\n실패 항목:")
        for n, m in fails:
            print(f"  - {n}: {m}")
        names = {n for n, _ in fails}
        print("\n조치 (순서대로):")
        if any("libero" in n for n in names):
            print("  1) LIBERO 설치:  cd ~/third_party/LIBERO && pip install -e .")
        if any("numpy" in n or "torch" in n or "opencv" in n for n in names):
            print("  2) 버전 되돌리기 (**반드시 다른 pip 설치를 모두 마친 뒤 마지막에**):")
            print("       pip install 'numpy<2' 'opencv-python<5' --force-reinstall")
            print("  3) 재발 방지 — pip 이 다시 못 올리게 제약을 걸어 두세요:")
            print("       export PIP_CONSTRAINT=$(pwd)/constraints.txt")
            print("     (이걸 걸어두면 이후 어떤 pip 설치도 numpy 를 2.x 로 못 올립니다)")
        if any("드라이버" in n for n in names):
            print("  0) torch CUDA 빌드를 드라이버에 맞추기 (위 메시지의 조치 명령 그대로)")
        if any("렌더링" in n for n in names):
            print("  4) EGL 확인:  ldconfig -p | grep libEGL")
        print("\n환경을 처음부터 다시 만들 필요는 없습니다. 위 순서면 복구됩니다.")
        print("\n자세한 스택은 VERIFY_TRACE=1 을 붙여 다시 돌리세요.")
        return 1

    flag = (f"--gpus {','.join(map(str, gpu_list))}" if len(gpu_list) > 1
            else f"--gpu {gpu_list[0]}")
    print("\n전부 통과했습니다. Stage 2 로 가세요:")
    print(f"  python scripts/01_smoke_forward.py {flag}")
    if len(gpu_list) > 1:
        print("  (분할 로드가 실제로 되는지는 여기서 처음 판명됩니다 — 미검증 경로입니다)")
    print(f"\n!! {args.out} 을 열어 로봇 팔이 똑바로 서 있는지 눈으로 확인하세요.")
    print("   뒤집혀 있으면 vlamod/env_libero.py 의 obs_to_image(flip=...) 를 바꿔야 합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
