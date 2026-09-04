# VLA_Interpretability

VLA(OpenVLA)에서 **action token 이 image token 과 language token 을 어떤 비율로 쓰는가**를
관찰(attention)과 인과 개입(knockout + KL) 두 축으로 측정하는 코드.

| 문서 | 내용 |
|---|---|
| [`docs/00_research_plan.md`](docs/00_research_plan.md) | 문제의식, 선행연구 지도, 예상되는 반박과 방어 |
| [`docs/01_module_reference.md`](docs/01_module_reference.md) | 각 파일의 목적과 설계 근거 |
| [`docs/02_runbook.md`](docs/02_runbook.md) | 단계별 실행 순서와 통과 기준(Gate) |

## 검증하는 가설

- **H1 (적응성)** — 언어/비전 비율이 *언어가 실제로 필요한 상황인지*에 따라 조절되는가?
- **H2 (지표 타당성)** — attention 비율(IVAR류)이 인과 기여(ΔKL)를 예측하는가?

두 가설 모두 **어느 쪽 결과가 나와도 발표거리**가 됩니다.

---

## 빠른 시작

```bash
# ── 노트북 (Windows) — 로직 검증만 ─────────────────────────────
pytest -q tests/test_core.py                  # 52 passed 나와야 정상

# ── 서버 (Linux + GPU) — 설치 ──────────────────────────────────
export PIP_CONSTRAINT=$(pwd)/constraints.txt  # ★ 먼저. 아래 함정 2 참고
bash setup/01_env.sh                          # conda env "vlamod" + torch + OpenVLA
bash setup/02_libero.sh                       # LIBERO + robosuite/MuJoCo
conda activate vlamod
python setup/03_download_ckpt.py --suite spatial     # ~15GB
python setup/04_verify.py --gpu 5             # ★ 10개 항목 상태 점검

# ── 서버 — 실행 ────────────────────────────────────────────────
python scripts/01_smoke_forward.py --gpu 5                  # ★ 관문
python scripts/01_smoke_forward.py --gpu 5 --libero --suite spatial --task-id 0

python scripts/02_run_analysis.py  --gpu 5 --suite spatial --tasks 0 1 2 --episodes 3
python scripts/03_correlation.py   --csv outputs/analysis_spatial.csv    # GPU 불필요
python scripts/04_counterfactual.py --gpu 5 --suite object --tasks 0 1 2 3
python scripts/05_plots.py         --suite spatial                      # GPU 불필요
```

**실행 위치**: 노트북은 코드 편집 + `pytest`, 서버는 나머지 전부.
`03` / `05` 는 CSV 만 있으면 노트북에서도 됩니다.

---

## 환경변수 — config 로 관리 (`.bashrc` 불필요)

`configs/default.yaml` 의 `env:` 섹션이 프로세스 환경변수를 설정합니다.
**경로가 바뀌면 이 한 줄만 고치면 됩니다.**

```yaml
env:
  hf_home: "~/shared/hdd_ext/nvme1/kimhyeongwoo"   # HF 캐시 (다운로드 + 로드 공용)
  mujoco_gl: "egl"                                  # headless 렌더링
  pyopengl_platform: "egl"
```

- `setup/03`, `setup/04`, `scripts/01·02·04` 가 **같은 값**을 읽습니다.
- 서버가 여러 대면 복사해서 `configs/<서버명>.yaml` 로 만들고 `--config` 로 고르세요.
  **재현할 때 config 파일명만 기록하면 됩니다.**
- 일회성 덮어쓰기: `--hf-home <경로>`

> **구현 주의:** HF 캐시 경로는 `huggingface_hub` 가 import 되는 시점에 확정됩니다.
> 그래서 `apply_env()` 를 스크립트 맨 앞(모델 로드 전)에서 호출합니다.
> 순서가 바뀌면 조용히 무시되므로, 이미 import 된 경우 경고를 냅니다.

---

## GPU 지정

모델을 올리는 **01 / 02 / 04 / setup-04** 만 아래 인자를 받습니다. **03 / 05 는 GPU 를 안 씁니다.**

| 인자 | 예시 | 설명 |
|---|---|---|
| `--gpu N` | `--gpu 5` | **가장 간단.** 5번 GPU 사용 |
| `--gpus` | `--gpus 0,6` | 한 장에 안 들어갈 때 여러 GPU 에 분할 로드 |
| `--device` | `--device cuda:5` · `--device 5` · `--device cpu` | 문자열 지정 |
| `--model` | `--model openvla/openvla-7b-finetuned-libero-object` | 체크포인트 교체 |
| `--unnorm-key` | `--unnorm-key libero_object` | action un-normalization key (02 / 04) |
| `--tag` | `--tag object_ckpt` | 출력 파일명 접미사 (02 / 04) |
| `--hf-home` | `--hf-home /data/hf` | HF 캐시 루트 |
| `--config` | `--config configs/lab.yaml` | 설정 파일 선택 |

`--gpu` / `--gpus` / `--device` 는 **셋 중 하나만** 쓸 수 있습니다.
생략하면 `configs/default.yaml` 의 `model.device` 를 씁니다.

로드 직후 아래가 출력됩니다. **요청한 번호가 그대로 찍히는지 매번 확인하세요.**

```
[env   ] HF_HOME=/home/…/kimhyeongwoo  MUJOCO_GL=egl
[device] model=cuda:5  render(EGL)=5
[model ] openvla/openvla-7b-finetuned-libero-spatial  unnorm_key=libero_spatial
[gpu   ] cuda:5 NVIDIA RTX A5000  사용가능 23.7GB / 전체 24.0GB
```

### 어느 GPU 를 골라야 하는가

OpenVLA-7B(≈7.5B 파라미터) bf16 은 **가중치만 15GB**, 활성값·KV 캐시·attention 버퍼까지
**18~20GB** 가 필요합니다.

| 카드 | 총 | 판정 |
|---|---|---|
| A6000 / A100 (48GB+) | 48GB | 여유 충분 |
| **A5000 · RTX 3090 (24GB)** | 24GB | **거의 비어 있으면 단일 GPU 로 충분** |
| A4000 (16GB) | 16GB | **단독 불가** (가중치 15GB) → `--gpus` 로 두 장 |

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

`memory.used` 가 낮은 24GB 카드를 고르세요. 20GB 미만이면 `report_gpu` 가 경고합니다.

### 멀티 GPU 분할 로드 (`--gpus`)

24GB 한 장이 안 나면 층을 나눠 올립니다.

```bash
python scripts/01_smoke_forward.py --gpus 0,6      # A4000 16GB × 2 = 32GB
```

- **수치는 단일 GPU 와 동일합니다.** 같은 연산을 배치만 나눠 하는 것이라
  attention 값도 action logit 도 바뀌지 않습니다. (양자화와 결정적으로 다른 점)
- 층 경계마다 GPU 간 전송이 생겨 **느려집니다.** 한 장에 들어가면 `--gpu` 를 쓰세요.
- 상한은 각 GPU 의 **현재 여유**에서 1.5GB 를 뺀 값으로 자동 계산합니다.
  고정값을 쓰면 남의 작업이 늘었을 때 OOM 이 나기 때문입니다.
- CPU 오프로드는 금지해 두었습니다 (`cpu: 0GiB`). 조용히 100배 느려지는 것을 막습니다.

### 멀티 GPU 서버의 함정 두 가지

**① PyTorch 와 MuJoCo 는 GPU 를 따로 고릅니다.** 모델을 `cuda:5` 에 올려도 LIBERO 의
EGL 렌더링은 기본적으로 0번을 씁니다. `--gpu` 를 쓰면 `MUJOCO_EGL_DEVICE_ID` 를
같은 번호로 맞춰 줍니다. 출력의 `render(EGL)=` 이 `model=` 과 같은지 확인하세요.

**② `CUDA_VISIBLE_DEVICES` 와 섞어 쓰지 마세요.** 섞으면 번호가 0부터 재매핑됩니다.
코드가 이 상황을 감지해 에러를 냅니다.

### 체크포인트를 병렬로 (불변성 확인용)

한 체크포인트 결과만으로는 "그 모델 얘기 아니냐"를 막을 수 없습니다.
**최소 2개**에서 같은 성질이 나오는지 확인해 두면, 나중에 실물(UR5e)로 넘어갈 때
전이를 주장하는 근거가 됩니다.

```bash
python scripts/02_run_analysis.py --gpu 1 --suite spatial \
       --model openvla/openvla-7b-finetuned-libero-spatial \
       --unnorm-key libero_spatial --tag spatial_ckpt &
python scripts/02_run_analysis.py --gpu 2 --suite object \
       --model openvla/openvla-7b-finetuned-libero-object \
       --unnorm-key libero_object --tag object_ckpt &
wait
```

---

## 구조

```
configs/default.yaml     env / model / tokens / metrics / intervene / run 설정
constraints.txt          pip 전역 제약 (numpy<2, opencv<5 …)
vlamod/
  device.py        apply_env(환경변수) · apply_overrides(GPU/모델) · report_gpu
  model_loader.py  OpenVLA 로딩(eager 강제, --gpus 분할), 백본 탐색, action bin id
  token_index.py   [BOS][visual][text][action] 구간 실측 + 육안 검증 출력
  capture.py       2-pass 캡처: generate → teacher-forced forward
  metrics.py       R_raw / R_norm / R_vnorm, action KL, causal_ratio
  intervene.py     pre-softmax attention knockout (language / vision / random control)
  pipeline.py      한 타임스텝 = 관찰 + 개입 → tidy row
  env_libero.py    LIBERO 얇은 래퍼
  viz.py           그림 (색맹 판별 검증 완료 3색)
setup/
  01_env.sh          conda + torch + OpenVLA
  02_libero.sh       LIBERO + 버전 충돌 되돌리기 + 검증
  03_download_ckpt.py 체크포인트 (config 의 env.hf_home 사용)
  04_verify.py       ★ 설치 없이 상태만 10개 항목 점검 (EGL 실렌더링 포함)
scripts/
  01_smoke_forward.py   Phase 0~1  구조 검증 7단계
  02_run_analysis.py    Phase 2~3  비율 + knockout KL 수집
  03_correlation.py     H2         상관 + 대조군 t-검정
  04_counterfactual.py  H1         지시 조건 4종 비교
  05_plots.py                      그림 4장
tests/test_core.py      GPU 없이 도는 단위 테스트 52개
```

### 산출물

```
02 ─▶ outputs/analysis_<suite>.csv       타임스텝별 tidy row
   ─▶ outputs/perlayer_<suite>.npz       층별 배열
04 ─▶ outputs/counterfactual_<suite>.csv
03 ─▶ outputs/correlation.csv
05 ─▶ outputs/fig_*.png
```

---

## 지표 정의

action query s, layer l, head h 에 대해:

| 지표 | 정의 | 용도 |
|---|---|---|
| `R_raw` | `M_L / (M_L + M_V)` | IVAR(2603.06001) 재현. **단독 주장 금지** |
| `R_norm` | `(M_L/\|L\|) / (M_L/\|L\| + M_V/\|V\|)` | 토큰 개수 비대칭 제거 |
| `R_vnorm` | `A(s,j)` 대신 `A(s,j)·‖v_j‖` | Kobayashi+2020. 실제 전달되는 벡터 크기 |
| `causal_ratio` | `KL_L / (KL_L + KL_V)` | 인과 기여 비율 (**주 증거**) |
| `lang_vs_control` | `KL_L / KL_random-visual` | **1에 가까우면 언어가 특별하지 않다는 뜻** |
| `uniform_baseline` | `\|L\|/(\|L\|+\|V\|)` | `R_raw` 의 기본값. 항상 같이 보고할 것 |

`|V|` ≈ 256, `|L|` ≈ 5~15 이므로 균등 분포만 가정해도 `R_raw ≈ 0.05` 가 나옵니다.
**`uniform_baseline` 보다 낮아야** "언어를 덜 본다"고 말할 수 있습니다.

---

## 함정 목록 (전부 실제로 겪었습니다)

1. **`attn_implementation="eager"` 필수.** flash_attention_2 / sdpa 는 `output_attentions=True` 를
   조용히 무시합니다. 에러 없이 `attentions=None` 이 나옵니다. `model_loader.py` 가 막아 둡니다.

2. **`PIP_CONSTRAINT` 를 걸어 두세요.** LIBERO 의 requirements 가 `numpy>=2` / `opencv 5.x` 를
   요구해서, pip 을 쓸 때마다 numpy 가 2.x 로 올라갑니다. torch 2.2.0 은 numpy 1.x 로
   컴파일돼 있어 그러면 `RuntimeError: Numpy is not available` 로 죽습니다.
   설치할 때마다 되돌리지 말고 원천 차단하세요:
   ```bash
   export PIP_CONSTRAINT=$(pwd)/constraints.txt
   ```

3. **NGC pip 미러.** `/etc/pip.conf` 에 `pypi.ngc.nvidia.com` 이 박혀 있으면 패키지마다
   DNS 5회 재시도가 걸려 설치가 **멈춘 것처럼** 보입니다. `export PIP_EXTRA_INDEX_URL=""`.

4. **transformers 4.40.x 고정.** 상위 버전은 attention mask 전달 규약이 바뀌어
   `intervene.py` 의 knockout 훅이 깨집니다.

5. **visual token 개수를 문서에서 베끼지 말 것.** `probe_visual_span()` 이 이미지를 바꿔가며
   embedding 차이로 **실측**합니다. 256이 아니면 경고가 뜹니다.

6. **템플릿 문구를 L 에서 제외할 것.** "In: What action should the robot take to" 는
   모든 샘플에 동일하므로 여기 걸린 attention 은 지시 내용과 무관합니다.

7. **BOS/attention sink 를 V 에 넣지 말 것.** 비전 비중이 인위적으로 부풀려집니다.

8. **knockout 은 pre-softmax 로.** softmax 뒤에 0으로 만들고 재정규화하는 것과 결과가 다릅니다.

9. **치환 baseline 은 빈 문자열이 아니라 다른 지시문으로.** 빈 입력은 OOD 라서
   "언어를 안 쓴다"가 아니라 "이상한 입력이라 망가졌다"가 됩니다.

10. **랜덤 대조군 없이 결론 내지 말 것.** 언어 토큰과 같은 개수의 visual 토큰을 무작위로
    차단한 것과 비교해야 "언어가 특별하다/아니다"를 말할 수 있습니다.

11. **`get_benchmark_dict()` 통과 ≠ 렌더링 OK.** 그건 파이썬 dict 를 읽을 뿐 EGL 을
    건드리지 않습니다. `setup/04_verify.py` 가 실제로 프레임을 뽑아 확인합니다.

12. **LIBERO 이미지 상하 반전.** `outputs/verify_render.png` 를 열어 **눈으로** 확인하세요.
    방향이 틀리면 vision attention 분석 전체가 무의미해집니다.

13. **LIBERO 는 원래 언어가 거의 필요 없는 벤치마크입니다.**
    `04_counterfactual.py --ambiguous-tasks` 로 후보 물체가 둘 이상인 task 를 지정하지 않으면
    "그건 벤치마크 탓"이라는 반박을 막을 수 없습니다.

---

## 문제가 생기면

```bash
python setup/04_verify.py --gpu <번호>     # 실패 항목과 조치를 알려줍니다
VERIFY_TRACE=1 python setup/04_verify.py   # 스택 트레이스까지
```

| 증상 | 원인 / 조치 |
|---|---|
| `RuntimeError: Numpy is not available` | numpy 2.x. `PIP_CONSTRAINT` 걸고 `pip install 'numpy<2' --force-reinstall` |
| `attentions 가 None` | eager 아님. `configs/*.yaml` 의 `attn_implementation` 확인 |
| `4D attention mask 를 기대했는데 ND` | transformers 버전. 4.40.x 로 |
| `ModuleNotFoundError: vlamod.device` | 서버 전송 누락. PyCharm Deployment 는 수동 업로드 필요 |
| `No module named 'libero'` | `cd ~/third_party/LIBERO && pip install -e .` (EGL 문제 아님) |
| `libero 는 되는데 benchmark import 실패` | **이때가 EGL 문제.** `ldconfig -p \| grep libEGL` |
| 설치가 멈춘 것처럼 보임 | NGC DNS 재시도. 함정 3 |
| `언어 구간 L 이 비었습니다` | `token_index.PROMPT_TEMPLATE` 이 체크포인트와 불일치 |
| `범위 검증: 실패` | `action_bin_token_ids` 의 오프셋 ±1 조정 |
| CUDA OOM | 24GB 이상 빈 GPU 로 바꾸거나 `--gpus a,b` |
| 모델은 뜨는데 렌더링만 실패 | `render(EGL)` 번호가 `model` 과 다른지 확인 |

---

## 미확인 사항 (직접 검증하세요)

- OpenVLA 체크포인트의 정확한 프롬프트 템플릿 (`token_index.PROMPT_TEMPLATE`)
- action bin ↔ vocab id 오프셋 (`model_loader.action_bin_token_ids`, `scripts/01` 이 검사)
- HF 체크포인트 repo 이름 (`setup/03_download_ckpt.py` 의 `CKPTS`)
- LIBERO task 객체의 instruction 속성명 (`task.language` vs `task.task_description`)
- transformers 4.40.1 에서 `LlamaDecoderLayer` 가 `attention_mask` 를 kwargs 로 받는지
- **`--gpus` 분할 로드가 실제로 되는지** — `trust_remote_code` 모델이 accelerate 의
  `device_map` 을 지원하지 않으면 실패합니다. 실패 시 명확한 에러를 냅니다.
