"""Stage 1 최종 검증 — 설치를 다시 하지 않고 **상태만** 점검합니다.

**A6000 서버(Linux)에서 실행하세요.** 노트북(Windows)의 역할은 `pytest` 뿐입니다.

    python setup/04_verify.py                 # 전체
    python setup/04_verify.py --gpu 4         # 특정 GPU 로
    python setup/04_verify.py --skip-render    # 렌더링 테스트 생략

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
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
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


@check("CUDA")
def c_cuda(gpu: int):
    import torch
    assert torch.cuda.is_available(), "CUDA 를 못 찾았습니다"
    n = torch.cuda.device_count()
    assert gpu < n, f"GPU {gpu} 요청했으나 보이는 GPU 는 {n}개"
    free, total = torch.cuda.mem_get_info(gpu)
    note = "" if free > 18e9 else "  ← 여유 부족 경고 (7B bf16 은 ~16GB 필요)"
    return f"{n}장, cuda:{gpu} {torch.cuda.get_device_name(gpu)} 여유 {free/1e9:.1f}/{total/1e9:.1f}GB{note}"


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
    import libero
    return os.path.dirname(libero.__file__)


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
    ap.add_argument("--gpu", type=int, default=0)
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
    from vlamod.device import apply_env
    apply_env(cfg, args)

    os.makedirs("outputs", exist_ok=True)

    print("\n=== Stage 1 검증 " + "=" * 48)
    c_python()
    c_numpy()
    c_torch_numpy()
    c_cuda(args.gpu)
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
        if any("렌더링" in n for n in names):
            print("  4) EGL 확인:  ldconfig -p | grep libEGL")
        print("\n환경을 처음부터 다시 만들 필요는 없습니다. 위 순서면 복구됩니다.")
        print("\n자세한 스택은 VERIFY_TRACE=1 을 붙여 다시 돌리세요.")
        return 1

    print("\n전부 통과했습니다. Stage 2 로 가세요:")
    print(f"  python scripts/01_smoke_forward.py --gpu {args.gpu}")
    print(f"\n!! {args.out} 을 열어 로봇 팔이 똑바로 서 있는지 눈으로 확인하세요.")
    print("   뒤집혀 있으면 vlamod/env_libero.py 의 obs_to_image(flip=...) 를 바꿔야 합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
