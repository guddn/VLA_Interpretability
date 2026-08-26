#!/usr/bin/env bash
# ============================================================
# Phase 0-a: conda 환경 + OpenVLA 설치
# 대상: A6000 48GB, CUDA 12.x 가정
# ============================================================
# 주의: OpenVLA는 transformers==4.40.1 에 강하게 묶여 있습니다.
#       버전을 올리면 attention mask 전달 규약이 바뀌어
#       vlamod/intervene.py 의 knockout 훅이 깨질 수 있습니다.
#       (자세한 내용은 vlamod/intervene.py 상단 주석 참고)
# ============================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-vlamod}"

echo "[1/5] conda 환경 생성: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[2/5] PyTorch 설치 (CUDA 12.1 빌드)"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

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
    print("device     :", torch.cuda.get_device_name(0))
    print("vram (GB)  :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
print("transformers:", transformers.__version__)
assert transformers.__version__.startswith("4.40"), \
    "transformers 4.40.x 가 아닙니다. knockout 훅이 깨질 수 있습니다."
PY

echo ""
echo "완료. 다음: bash setup/02_libero.sh"
