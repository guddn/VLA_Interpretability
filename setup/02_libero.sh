#!/usr/bin/env bash
# ============================================================
# Phase 0-b: LIBERO 시뮬레이터 설치
# ============================================================
# LIBERO는 robosuite/MuJoCo 기반입니다. headless 서버에서는
# EGL 렌더링이 필요하므로 아래 환경변수를 반드시 export 하세요:
#   export MUJOCO_GL=egl
#   export PYOPENGL_PLATFORM=egl
# ============================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-vlamod}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[1/3] LIBERO 클론 + 설치"
mkdir -p "${HOME}/third_party"
cd "${HOME}/third_party"
if [ ! -d LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO
pip install -e .

echo "[2/3] OpenVLA 쪽 LIBERO 의존성"
cd "${HOME}/third_party/openvla"
if [ -f experiments/robot/libero/libero_requirements.txt ]; then
  pip install -r experiments/robot/libero/libero_requirements.txt
else
  echo "  !! libero_requirements.txt 를 못 찾았습니다."
  echo "  !! OpenVLA repo 구조가 바뀌었을 수 있습니다. 직접 확인하세요:"
  echo "  !! ls ${HOME}/third_party/openvla/experiments/robot/"
fi

echo "[3/3] headless 렌더링 확인"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
try:
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    print("사용 가능한 benchmark suite:", list(bd.keys()))
except Exception as e:
    print("LIBERO import 실패:", repr(e))
    print("→ MUJOCO_GL=egl 를 export 했는지, EGL 드라이버가 있는지 확인하세요.")
PY

echo ""
echo "완료. 다음: python setup/03_download_ckpt.py --suite spatial"
echo "쉘 rc 파일에 아래 두 줄을 추가해 두세요:"
echo "  export MUJOCO_GL=egl"
echo "  export PYOPENGL_PLATFORM=egl"
