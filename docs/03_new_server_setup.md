# 새 서버 세팅 가이드 (처음부터)

conda 도 없는 새 리눅스 서버에 이 프로젝트를 올리는 전체 순서입니다.
지금까지 실제로 겪은 함정이 모두 반영돼 있습니다.

**총 소요: 1.5~2시간 (대부분 다운로드 대기)**

---

## 0. 사전 조사 (5분) — 먼저 확인하고 시작하세요

```bash
hostname                                  # 서버 이름 (기록해 두세요)
python3 --version                         # 3.10 이면 좋고, 아니어도 conda 가 해결
which conda; ls -d ~/miniconda3 ~/anaconda3 /opt/conda 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1   # ★ CUDA 빌드 결정
df -h ~                                   # 홈 여유
ls -d ~/shared 2>/dev/null                # 큰 디스크가 어디 붙어 있는지
ldconfig -p | grep -E "libEGL|libGL\.so"  # EGL 드라이버 (없으면 conda 로 보충)
```

**여기서 정해야 할 두 가지:**

| 결정 | 기준 |
|---|---|
| **쓸 GPU 번호** | `memory.used` 가 낮은 **24GB 이상** 카드. 7B bf16 은 18~20GB 필요 |
| **체크포인트 저장 경로** | 여유 30GB 이상. 홈이 작으면 `~/shared/...` 같은 큰 디스크 |

**드라이버 버전이 torch 빌드를 결정합니다.**

| 드라이버 | torch 빌드 | 비고 |
|---|---|---|
| 525 이상 | `cu121` | 기본값 |
| 450~524 | **`cu118`** | 랩 서버에 흔합니다. cu121 을 깔면 `driver is too old` 로 죽습니다 |
| 450 미만 | — | 관리자에게 드라이버 업그레이드 요청 |

`setup/01_env.sh` 가 드라이버를 읽고 **자동으로 골라 줍니다.** torch 버전(2.2.0)은 같으므로
numpy·transformers 호환은 그대로입니다.

GPU 판정표:

| 카드 | 총 | 판정 |
|---|---|---|
| A6000 / A100 (48GB+) | 48GB | 여유 충분 |
| **A5000 · RTX 3090 (24GB)** | 24GB | **거의 비어 있으면 충분** |
| A4000 (16GB) | 16GB | **단독 불가** (가중치만 15GB) → `--gpus a,b` 로 두 장 |

---

## 1. Miniconda 설치 (5분)

`which conda` 가 비어 있을 때만. **`pip install conda` 는 하지 마세요** — 작동하는
conda 설치가 되지 않습니다.

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash
conda --version
```

`-b` 무인 설치, `-p` 설치 경로. **sudo 불필요**라 랩 서버에서도 됩니다.

> 이미 `~/miniconda3` 가 있는데 `which conda` 가 비었다면 초기화만 하면 됩니다:
> `~/miniconda3/bin/conda init bash && exec bash`

### venv 로 하고 싶다면

가능하지만 `setup/*.sh` 를 고쳐야 합니다. conda 를 쓰는 이유는 하나뿐입니다 —
**LIBERO 가 EGL/OpenGL 시스템 라이브러리를 필요로 하는데, sudo 없이 넣을 수 있는 건
conda 뿐**이기 때문입니다. `ldconfig -p | grep libEGL` 이 비어 있고 sudo 도 없다면
conda 를 쓰세요.

---

## 2. 코드 전송 (2분)

```bash
# (a) 로컬에서 rsync
rsync -av --exclude '.git' --exclude '.idea' --exclude '__pycache__' \
      ~/PycharmProjects/VLA_Interpretability/ user@새서버:~/VLA_Interpretability/

# (b) 또는 PyCharm: Tools → Deployment → Upload to...
# (c) 또는 git clone
```

```bash
ssh user@새서버
cd ~/VLA_Interpretability

# 전송 누락 확인 — 하나라도 없으면 나중에 ModuleNotFoundError 로 죽습니다
ls vlamod/device.py constraints.txt configs/default.yaml setup/04_verify.py
```

---

## 3. pip 제약 걸기 (30초) — ★ 설치 전에 반드시

**이 한 줄이 numpy 문제를 원천 차단합니다.**

```bash
export PIP_CONSTRAINT=$(pwd)/constraints.txt
echo "export PIP_CONSTRAINT=$(pwd)/constraints.txt" >> ~/.bashrc
```

왜 필요한가: LIBERO 의 requirements 가 `numpy>=2` 와 `opencv 5.x` 를 요구합니다.
그런데 torch 2.2.0 은 numpy 1.x 로 컴파일돼 있어 numpy 2 에서
`RuntimeError: Numpy is not available` 로 죽습니다. 설치할 때마다 되돌리는 대신
pip 이 아예 못 올리게 막습니다.

NGC pip 미러가 걸려 있는 서버라면 이것도 함께 (설치가 멈춘 것처럼 보이는 원인):

```bash
pip config debug | grep -i extra-index   # ngc.nvidia.com 이 보이면
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
echo 'export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"' >> ~/.bashrc
```

---

## 4. 환경 + OpenVLA (20분)

```bash
bash setup/01_env.sh
```

> **`conda activate vlamod` 를 먼저 하지 마세요.** 이 시점엔 환경이 아직 없습니다.
> 스크립트가 만들고 스스로 activate 합니다. 이미 환경이 있으면 재사용합니다(재실행 안전).

conda 환경 `vlamod` (python 3.10) 생성 → torch 2.2.0+cu121 → OpenVLA → 분석 패키지.
`transformers 4.40.x` 검사 assert 가 들어 있어 버전이 어긋나면 여기서 멈춥니다.

---

## 5. LIBERO (20분)

```bash
bash setup/02_libero.sh
```

4단계로 돕니다:

| 단계 | 내용 |
|---|---|
| `[1/4]` | LIBERO clone + `pip install -e .` + **import 검증** |
| `[2/4]` | OpenVLA 쪽 `libero_requirements.txt` |
| `[3/4]` | **numpy<2 / opencv<5 되돌리기** (모든 설치가 끝난 뒤 마지막에) |
| `[4/4]` | numpy → torch 연동 → cv2 → libero → benchmark 순서로 검증 |

**중간에 대화형 프롬프트가 나올 수 있습니다.** LIBERO 첫 import 때 데이터셋 경로를
묻습니다. `N` 을 눌러 기본값을 쓰시면 됩니다 — **저희는 시연 HDF5 를 안 씁니다.**
사전학습 체크포인트로 rollout 만 하므로 `bddl_files` / `init_files` 만 있으면 되고,
그 둘은 저장소에 포함돼 있습니다. (수십 GB 다운로드 불필요)

EGL 라이브러리가 없다면 여기서 보충:

```bash
conda install -c conda-forge mesalib libglu     # sudo 없이
```

---

## 5.5 ★ 부모 셸에 activate (10초) — 놓치기 쉬움

```bash
conda activate vlamod
which python                     # ~/miniconda3/envs/vlamod/bin/python 이어야 함
```

**스크립트는 서브셸에서 돕니다.** `01`/`02` 안의 `conda activate` 는 그 서브셸에서만
유효하고 **부모 셸(형우 님 터미널)로 넘어오지 않습니다.** 그래서 `bash setup/02_libero.sh`
가 끝나도 터미널은 여전히 `base` 입니다.

이걸 빠뜨리면 `python setup/03...` 이 **base 환경의 파이썬**으로 돌아
`ModuleNotFoundError: No module named 'yaml'` 같은 엉뚱한 에러가 납니다.

---

## 6. 경로 설정 (1분)

`configs/default.yaml` 의 `env:` 섹션을 **이 서버에 맞게** 고칩니다.
`.bashrc` 는 건드릴 필요 없습니다.

```yaml
env:
  hf_home: "~/shared/hdd_ext/nvme1/kimhyeongwoo"   # ← 0단계에서 정한 큰 디스크
  mujoco_gl: "egl"
  pyopengl_platform: "egl"
```

**그 경로가 이 서버에 없으면 반드시 바꾸세요.** 다운로드와 모델 로드가 같은 값을
읽으므로 여기 한 줄만 맞으면 됩니다.

> 서버가 여럿이면 복사해서 `configs/<서버명>.yaml` 로 만들고 `--config` 로 고르세요.
> 재현할 때 **config 파일명만 기록**하면 됩니다.

---

## 7. 체크포인트 (15~30분, 다운로드 대기)

```bash
conda activate vlamod
python setup/03_download_ckpt.py --suite spatial      # ~15GB
```

시작하면 저장 위치와 여유 공간을 먼저 찍고, 모자라면 받기 전에 멈춥니다.

여유가 넉넉하면 하나 더 받아 두세요. **체크포인트 2개**가 있어야
"그 모델에서만 나온 결과 아니냐"를 막을 수 있습니다.

```bash
python setup/03_download_ckpt.py --suite object       # +15GB
```

---

## 8. 검증 (2분) — ★ 여기가 관문

```bash
python setup/04_verify.py --gpu <0단계에서 정한 번호>
```

10개 항목을 독립적으로 검사하고, 실패해도 계속 진행한 뒤 **무엇을 다시 해야 하는지**
알려줍니다.

```
  [ OK ] numpy < 2: 1.26.4
  [ OK ] torch ↔ numpy 연동: torch 2.2.0+cu121
  [ OK ] CUDA: 10장, cuda:5 NVIDIA RTX A5000 여유 23.7/24.0GB
  [ OK ] opencv < 5: 4.14.0
  [ OK ] transformers 4.40.x: 4.40.1
  [ OK ] libero 패키지: /home/…/LIBERO/libero
  [ OK ] EGL 실제 렌더링 (★ 진짜 검사): 256x256, std=48.3 → outputs/verify_render.png
  [ OK ] HF 캐시 / 체크포인트: …/hub  여유 102GB  |  체크포인트 있음
통과 10 / 10
```

**`outputs/verify_render.png` 를 열어 로봇 팔이 똑바로 서 있는지 눈으로 확인하세요.**
LIBERO 이미지는 상하가 뒤집혀 나오는데, 방향이 틀리면 vision attention 분석 전체가
무의미해집니다. 뒤집혀 있으면 `vlamod/env_libero.py` 의 `obs_to_image(flip=...)` 를 바꿉니다.

---

## 9. Stage 2 로 (30분)

```bash
python scripts/01_smoke_forward.py --gpu <번호>
python scripts/01_smoke_forward.py --gpu <번호> --libero --suite spatial --task-id 0
```

`[4] 토큰 구간 분해` 출력에서 **L 태그가 instruction 내용어에만** 붙었는지
직접 확인하세요. 이건 자동화할 수 없습니다.

이후는 [`docs/02_runbook.md`](02_runbook.md) 를 따라가시면 됩니다.

---

## 요약 — 복사해서 쓰는 전체 순서

```bash
# 0. 조사
hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
which conda; df -h ~

# 1. conda (없을 때만)
cd ~ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash

# 2. 코드 (로컬에서 rsync 후 ssh)
cd ~/VLA_Interpretability
ls vlamod/device.py constraints.txt setup/04_verify.py

# 3. pip 제약 ★
export PIP_CONSTRAINT=$(pwd)/constraints.txt
echo "export PIP_CONSTRAINT=$(pwd)/constraints.txt" >> ~/.bashrc

# 4~5. 설치 (앞에서 activate 하지 말 것 — 스크립트가 알아서 함)
bash setup/01_env.sh
bash setup/02_libero.sh

# 5.5 ★ 부모 셸에 적용 (서브셸의 activate 는 안 넘어옴)
conda activate vlamod
which python

# 6. configs/default.yaml 의 env.hf_home 을 이 서버 경로로 수정

# 7~8. 체크포인트 + 검증
python setup/03_download_ckpt.py --suite spatial
python setup/04_verify.py --gpu <번호>

# 9. 시작
python scripts/01_smoke_forward.py --gpu <번호>
```

---

## 새 서버에서 특히 자주 걸리는 것

| 증상 | 원인 / 조치 |
|---|---|
| `pip install conda` 를 시도 | 작동하지 않습니다. Miniconda 설치 스크립트를 쓰세요 |
| 설치가 멈춘 것처럼 보임 | NGC pip 미러 DNS 재시도. `export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"` |
| `RuntimeError: Numpy is not available` | numpy 2.x. **3단계 `PIP_CONSTRAINT` 를 건너뛴 것** |
| `No module named 'libero'` | `cd ~/third_party/LIBERO && pip install -e .` (EGL 문제 아님) |
| `libero 는 되는데 benchmark 실패` | **이때가 EGL.** `conda install -c conda-forge mesalib libglu` |
| `ModuleNotFoundError: vlamod.device` | 코드 전송 누락. PyCharm Deployment 는 수동 업로드 |
| `No module named 'yaml'` 등 기본 패키지 없음 | **`conda activate vlamod` 를 안 한 것.** 5.5 단계 |
| `CondaValueError: prefix already exists` | 구버전 스크립트. 지금은 기존 환경을 재사용합니다 |
| `The NVIDIA driver ... is too old (found version 11080)` | **드라이버가 CUDA 11.8 까지인데 torch 는 cu121.** `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall` |
| LIBERO 폴더 밖에서만 `No module named 'libero'` | namespace package + strict editable 충돌. `cd ~/third_party/LIBERO && pip install -e . --config-settings editable_mode=compat` |
| `TypeError: expected str ... not NoneType` (`os.path.dirname`) | 구버전 스크립트 버그. LIBERO 는 namespace package 라 `__file__` 이 None 입니다. **설치는 성공한 것**이니 최신 `02_libero.sh` 로 재실행하세요 |
| `hf_home 경로를 만들 수 없습니다` | config 의 경로가 이 서버에 없음. 6단계 |
| CUDA OOM | 24GB 이상 빈 GPU 로 바꾸거나 `--gpus a,b` |
| 렌더링만 실패 | `render(EGL)` 번호가 `model` 과 다른지 확인 |
