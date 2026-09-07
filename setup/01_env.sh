#!/usr/bin/env bash
# ============================================================
# Phase 0-a: conda 환경 + OpenVLA 설치
# 대상: NVIDIA GPU (24GB 이상 권장), CUDA 12.x
#
# 실행 순서:
#   bash setup/01_env.sh        ← 지금 이것. conda activate 를 미리 하지 마세요
#                                  (환경이 아직 없습니다). 스크립트가 스스로 만들고 씁니다.
#   bash setup/02_libero.sh     ← 내부에서 스스로 activate
#   conda activate vlamod       ← ★ 여기서 부모 셸에 적용. 서브셸의 activate 는 안 넘어옵니다
#   python setup/03_download_ckpt.py ...
#
# 재실행 안전: 환경이 이미 있으면 만들지 않고 재사용합니다.
# ============================================================
# 주의: OpenVLA는 transformers==4.40.1 에 강하게 묶여 있습니다.
#       버전을 올리면 attention mask 전달 규약이 바뀌어
#       vlamod/intervene.py 의 knockout 훅이 깨질 수 있습니다.
#       (자세한 내용은 vlamod/intervene.py 상단 주석 참고)
# ============================================================
set -euo pipefail

# --- 설치 전 공통 설정 --------------------------------------------------
# (1) NVIDIA NGC pip 미러가 /etc/pip.conf 에 박혀 있으면 패키지마다 DNS 5회 재시도가
#     걸려 설치가 멈춘 것처럼 보입니다. 환경변수가 설정파일보다 우선하므로 여기서 비웁니다.
# !! 빈 문자열("")은 pip 이 "설정 안 됨"으로 보고 /etc/pip.conf 값을 그대로 씁니다.
#    (PIP_RETRIES 는 먹히는데 이것만 안 먹는 이유입니다)
#    pypi.org 를 명시적으로 넣어 NGC 미러를 **덮어씁니다**.
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
export PIP_RETRIES=2
export PIP_TIMEOUT=10

# (2) headless 렌더링. **라이브러리 import 전에** 있어야 하므로 스크립트 맨 위에 둡니다.
#     (configs/default.yaml 로는 해결되지 않습니다 — 그건 파이썬 스크립트용 설정입니다)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# (3) pip 제약 — LIBERO requirements 가 numpy>=2 / opencv5 를 올리는 것을 원천 차단합니다.
#     이 스크립트 안의 모든 pip 명령에 자동 적용됩니다.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIP_CONSTRAINT="${REPO_ROOT}/constraints.txt"
echo "[pip] PIP_CONSTRAINT=${PIP_CONSTRAINT}"

# ------------------------------------------------------------------------

ENV_NAME="${ENV_NAME:-vlamod}"

echo "[1/5] conda 환경 준비: ${ENV_NAME}"
command -v conda >/dev/null 2>&1 || {
  echo "  !! conda 를 찾을 수 없습니다."
  echo "  !! (pip install conda 는 작동하지 않습니다 — Miniconda 설치 스크립트를 쓰세요)"
  echo "  !!   cd ~ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  echo "  !!   bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
  echo "  !!   \$HOME/miniconda3/bin/conda init bash && exec bash"
  exit 1
}
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# 재실행 안전: 이미 있으면 재사용 (conda create 는 기존 환경에 대해 실패합니다)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "  기존 환경 재사용: ${ENV_NAME}  (새로 만들려면 conda env remove -n ${ENV_NAME} 후 재실행)"
else
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"
echo "  python: $(which python)  ($(python --version 2>&1))"

echo "[2/5] PyTorch 설치 — 드라이버에 맞는 CUDA 빌드 자동 선택"
# torch 2.2.0+cu121 은 드라이버 525 이상을 요구합니다. 랩 서버는 구형 드라이버가
# 흔해서, 그대로 깔면 실행 시점에 이렇게 죽습니다:
#   RuntimeError: The NVIDIA driver on your system is too old (found version 11080)
# 그래서 드라이버 버전을 보고 cu121 / cu118 을 고릅니다. **torch 버전(2.2.0)은 동일**
# 하므로 numpy 1.x·transformers 4.40 호환은 그대로입니다.
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
DRV_MAJOR="${DRV%%.*}"
if [ -z "${DRV_MAJOR}" ]; then
  echo "  !! nvidia-smi 를 못 찾았습니다. GPU 없는 머신이면 cu118 로 진행합니다."
  CU="cu118"
elif [ "${DRV_MAJOR}" -ge 525 ] 2>/dev/null; then
  CU="cu121"
elif [ "${DRV_MAJOR}" -ge 450 ] 2>/dev/null; then
  CU="cu118"
else
  echo "  !! 드라이버 ${DRV} 는 너무 오래되었습니다 (450 이상 필요). 관리자에게 문의하세요."
  exit 1
fi
echo "  드라이버 ${DRV:-불명}  →  torch ${CU} 빌드 선택"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url "https://download.pytorch.org/whl/${CU}"

echo "[3/5] OpenVLA 클론 + 설치"
mkdir -p "${HOME}/third_party"
cd "${HOME}/third_party"
if [ ! -d openvla ]; then
  git clone https://github.com/openvla/openvla.git
fi
cd openvla
pip install -e .

echo "[4/5] 분석용 추가 패키지"
pip install \
    "transformers==4.40.1" \
    "tokenizers==0.19.1" \
    "timm==0.9.10" \
    "accelerate>=0.25.0" \
    "einops" \
    "numpy<2.0" \
    "pandas" \
    "scipy" \
    "matplotlib" \
    "seaborn" \
    "tqdm" \
    "pyyaml" \
    "pytest"

echo "[5/5] 확인"
python - <<'PY'
import torch, transformers
print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    # get_device_name 은 CUDA 를 실제로 초기화합니다.
    # 드라이버가 너무 낮으면 여기서 "driver is too old" 로 죽습니다.
    print("device     :", torch.cuda.get_device_name(0))
    print("vram (GB)  :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
print("transformers:", transformers.__version__)
assert transformers.__version__.startswith("4.40"), \
    "transformers 4.40.x 가 아닙니다. knockout 훅이 깨질 수 있습니다."
PY

# ---------------------------------------------------------------------------
# PIP_CONSTRAINT 를 conda 환경에 **영구히** 박습니다.
#
# 왜 여기인가:
#   - configs/default.yaml 로는 안 됩니다. 그건 파이썬 프로세스 안에서만 유효한데
#     pip 은 셸에서 따로 돕니다.
#   - ~/.bashrc 는 서버·프로젝트가 바뀌면 오염됩니다.
#   - conda 의 activate.d 는 `conda activate vlamod` 할 때만 켜지고
#     `conda deactivate` 하면 원래대로 돌아갑니다. 범위가 정확히 맞습니다.
# ---------------------------------------------------------------------------
ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys,os;print(sys.prefix)' 2>/dev/null || true)"
if [ -n "${ENV_PREFIX}" ] && [ -d "${ENV_PREFIX}" ]; then
  mkdir -p "${ENV_PREFIX}/etc/conda/activate.d" "${ENV_PREFIX}/etc/conda/deactivate.d"
  cat > "${ENV_PREFIX}/etc/conda/activate.d/vlamod_pip.sh" <<EOF
# vlamod: pip 이 numpy 를 2.x 로 올리는 것을 원천 차단합니다.
# setup/01_env.sh 가 자동 생성했습니다. 프로젝트를 옮기면 경로를 고치세요.
export VLAMOD_OLD_PIP_CONSTRAINT="\${PIP_CONSTRAINT:-}"
export PIP_CONSTRAINT="${REPO_ROOT}/constraints.txt"
EOF
  cat > "${ENV_PREFIX}/etc/conda/deactivate.d/vlamod_pip.sh" <<'EOF'
if [ -n "${VLAMOD_OLD_PIP_CONSTRAINT:-}" ]; then
  export PIP_CONSTRAINT="${VLAMOD_OLD_PIP_CONSTRAINT}"
else
  unset PIP_CONSTRAINT
fi
unset VLAMOD_OLD_PIP_CONSTRAINT
EOF
  echo "[pip] conda activate 시 PIP_CONSTRAINT 자동 설정하도록 등록했습니다:"
  echo "      ${ENV_PREFIX}/etc/conda/activate.d/vlamod_pip.sh"
  echo "      → 이제 export 를 매번 치지 않아도 됩니다 (다음 activate 부터 적용)"
else
  echo "[pip] !! conda 환경 경로를 못 찾아 activate.d 등록을 건너뜁니다."
  echo "        수동 pip 설치 전에는 export PIP_CONSTRAINT=... 를 직접 하세요."
fi

echo ""
echo "완료. 다음:"
echo "  bash setup/02_libero.sh"
echo ""
echo "!! 이 스크립트의 conda activate 는 **부모 셸로 넘어가지 않습니다** (서브셸)."
echo "!! 02 까지 끝난 뒤, python 명령을 직접 치기 전에 이걸 한 번 하세요:"
echo "     conda activate ${ENV_NAME}"
