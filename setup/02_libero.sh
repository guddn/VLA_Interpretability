#!/usr/bin/env bash
# ============================================================
# Phase 0-b: LIBERO 시뮬레이터 설치
# ============================================================
# 이 스크립트가 처리하는 함정 3가지:
#   (1) NGC pip 미러 DNS 실패로 설치가 멈춘 것처럼 보이는 문제
#   (2) headless EGL 렌더링 환경변수 (import 전에 있어야 함)
#   (3) LIBERO requirements 가 numpy 를 2.x 로 올려 torch 를 깨뜨리는 문제
# ============================================================
set -euo pipefail

# --- 설치 전 공통 설정 --------------------------------------------------
# NVIDIA NGC pip 미러가 /etc/pip.conf 에 박혀 있으면 패키지마다 DNS 5회 재시도가
# 걸려 설치가 멈춘 것처럼 보입니다. 환경변수가 설정파일보다 우선하므로 여기서 비웁니다.
# !! 빈 문자열("")은 pip 이 "설정 안 됨"으로 보고 /etc/pip.conf 값을 그대로 씁니다.
#    (PIP_RETRIES 는 먹히는데 이것만 안 먹는 이유입니다)
#    pypi.org 를 명시적으로 넣어 NGC 미러를 **덮어씁니다**.
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
export PIP_RETRIES=2
export PIP_TIMEOUT=10

# headless 렌더링. **라이브러리 import 전에** 있어야 하므로 스크립트 맨 위에 둡니다.
# (configs/default.yaml 로는 해결되지 않습니다 — 그건 파이썬 스크립트용 설정입니다)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# (3) pip 제약 — LIBERO requirements 가 numpy>=2 / opencv5 를 올리는 것을 원천 차단합니다.
#     이 스크립트 안의 모든 pip 명령에 자동 적용됩니다.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIP_CONSTRAINT="${REPO_ROOT}/constraints.txt"
echo "[pip] PIP_CONSTRAINT=${PIP_CONSTRAINT}"

# ------------------------------------------------------------------------

ENV_NAME="${ENV_NAME:-vlamod}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}" || {
  echo "  !! conda 환경 '${ENV_NAME}' 이 없습니다. setup/01_env.sh 를 먼저 실행하세요."
  exit 1
}
conda activate "${ENV_NAME}"
echo "  python: $(which python)"

echo "[1/4] LIBERO 클론 + 설치"
mkdir -p "${HOME}/third_party"
cd "${HOME}/third_party"
if [ ! -d LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO
pip install -e .

# !! 검증은 반드시 **LIBERO 디렉토리 밖에서** 해야 합니다.
#    안에서 하면 설치된 패키지가 아니라 현재 폴더의 libero/ 를 import 해서
#    설치가 실패해도 통과합니다(거짓 통과).
if ! (cd "${HOME}" && python -c "import libero" 2>/dev/null); then
  echo "  editable 설치가 밖에서 안 잡힙니다. compat 모드로 재시도합니다."
  echo "  (LIBERO 최상위는 namespace package 라 setuptools strict editable 과 충돌합니다)"
  pip install -e . --config-settings editable_mode=compat
fi
(cd "${HOME}" && python -c "import libero" 2>/dev/null) || {
  echo "  !! LIBERO 를 프로젝트 밖에서 import 할 수 없습니다."
  echo "  !! (EGL 문제가 아니라 설치 문제입니다)"
  echo "  !! 임시 우회:  export PYTHONPATH=\"${HOME}/third_party/LIBERO:\$PYTHONPATH\""
  exit 1
}
# LIBERO 최상위는 __init__.py 가 없는 namespace package 라 __file__ 이 None 입니다.
# os.path.dirname(None) 은 TypeError 를 내므로 __path__ 로 폴백합니다.
(cd "${HOME}" && python - <<'PYLOC'
import libero, os
loc = getattr(libero, "__file__", None)
if loc:
    print("  OK:", os.path.dirname(loc))
else:
    paths = list(getattr(libero, "__path__", []))
    print("  OK:", paths[0] if paths else "(namespace package, 경로 불명)")
PYLOC
)

echo "[2/4] OpenVLA 쪽 LIBERO 의존성"
cd "${HOME}/third_party/openvla"
if [ -f experiments/robot/libero/libero_requirements.txt ]; then
  pip install -r experiments/robot/libero/libero_requirements.txt
else
  echo "  !! libero_requirements.txt 를 못 찾았습니다."
  echo "  !! OpenVLA repo 구조가 바뀌었을 수 있습니다. 직접 확인하세요:"
  echo "  !! ls ${HOME}/third_party/openvla/experiments/robot/"
fi

echo "[3/4] 버전 충돌 되돌리기"
# LIBERO 의 requirements 가 numpy 를 2.x 로 올려버립니다.
# torch 2.2.0 은 numpy 1.x 에 대해 컴파일되어 있어 numpy 2 에서 크래시합니다.
# **반드시 모든 pip 설치가 끝난 뒤 마지막에** 되돌려야 합니다.
pip install "numpy<2" --force-reinstall -q
# LIBERO requirements 가 opencv-python 5.x 를 끌어오는데, 5.x 는 numpy>=2 를 요구해
# 위의 numpy 핀과 충돌합니다. robosuite 는 cv2 4.x 를 전제하므로 opencv 를 내립니다.
pip install "opencv-python<5" -q
echo "  numpy  = $(python -c 'import numpy; print(numpy.__version__)')"
echo "  opencv = $(python -c 'import cv2; print(cv2.__version__)')"

echo "[4/4] 검증"
# LIBERO 는 첫 import 시 데이터셋 경로를 input() 으로 물어볼 수 있으므로 stdin 을 막습니다.
python - <<'PYEOF' </dev/null
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy
print(f"  numpy      : {numpy.__version__}")
if not numpy.__version__.startswith("1."):
    print("  !! numpy 가 2.x 입니다. `pip install 'numpy<2' --force-reinstall` 을 다시 하세요.")
    sys.exit(1)

import torch
_ = torch.randn(3).numpy()          # numpy 2 면 여기서 죽습니다
print(f"  torch      : {torch.__version__}  (numpy 연동 OK)")

import cv2
print(f"  opencv     : {cv2.__version__}")
if int(cv2.__version__.split(".")[0]) >= 5:
    print("  !! opencv 5.x 는 numpy>=2 를 요구해 충돌합니다. robosuite 도 4.x 를 전제합니다.")
    print("  !! 조치:  pip install 'opencv-python<5'")
    sys.exit(1)

try:
    import libero
except ModuleNotFoundError:
    print("  !! LIBERO 패키지가 설치되지 않았습니다. **EGL 문제가 아닙니다.**")
    print("  !! 조치:  cd ~/third_party/LIBERO && pip install -e .")
    sys.exit(1)
print(f"  libero     : {os.path.dirname(libero.__file__)}")

try:
    from libero.libero import benchmark
    print("  benchmark  :", list(benchmark.get_benchmark_dict().keys()))
except Exception as e:                                  # noqa: BLE001
    print(f"  !! libero 는 설치됐으나 benchmark import 실패: {e!r}")
    print("  !! **이 경우가 EGL / MuJoCo 문제입니다.** 확인:")
    print("  !!   echo $MUJOCO_GL              (egl 이어야 함)")
    print("  !!   ldconfig -p | grep libEGL")
    sys.exit(1)
PYEOF

echo ""
echo "완료. 다음: python setup/03_download_ckpt.py --suite spatial"
echo ""
echo "참고: LIBERO 가 물어본 datasets 경로는 **시연 HDF5(학습용)** 저장소입니다."
echo "      저희는 사전학습 체크포인트로 rollout 만 하므로 받지 않아도 됩니다."
echo "      (필요한 bddl_files / init_files 는 저장소에 포함돼 있습니다)"
echo "!! 이 스크립트의 conda activate 는 부모 셸로 넘어가지 않습니다."
echo "!! python 명령을 직접 치기 전에 한 번 실행하세요:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "쉘 rc 파일에 아래를 추가해 두세요:"
echo '  export MUJOCO_GL=egl'
echo '  export PYOPENGL_PLATFORM=egl'
echo '  export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"'
