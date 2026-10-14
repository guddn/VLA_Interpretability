# VLA_Interpretability

VLA(OpenVLA-7B, LIBERO-Spatial)의 **언어 무시(Vision 편향)** 를 모델 내부에서 직접 확인하고 원인을 찾는 코드.
관찰(attention), 인과 개입(knockout · 임베딩 치환 + KL), 행동(ICBench 방식 모순 지시 롤아웃) 세 축으로 측정합니다.
실험은 **스크립트 번호(01–10)** 로 부릅니다.

| 문서 | 내용 |
|---|---|
| [`docs/00_research_plan.md`](docs/00_research_plan.md) | **연구 질문·가설·로드맵**, 선행연구 지도, 지금까지의 증거, 예상 반박과 방어 |
| [`docs/01_module_reference.md`](docs/01_module_reference.md) | `vlamod/` 13개 모듈 · 스크립트 01–10 · 테스트의 목적과 설계 근거 |
| [`docs/02_runbook.md`](docs/02_runbook.md) | 스크립트별 실행 명령과 통과 기준(Gate), 트러블슈팅 |
| [`docs/03_new_server_setup.md`](docs/03_new_server_setup.md) | **새 서버에 처음부터 세팅** (conda 부터) |
| [`docs/04_rendering_pipeline.md`](docs/04_rendering_pipeline.md) | 픽셀이 만들어져 visual token 이 되기까지 (패키지별 역할) |
| [`docs/05_interventions.md`](docs/05_interventions.md) | 인과 개입 설계 — **어디에 개입하는가**로 분류 |
| [`docs/06_metrics.md`](docs/06_metrics.md) | 지표 정의 — CSV 컬럼 사전, 수식, 알려진 구현 문제 |

## 연구 질문과 가설

> **VLA 의 언어 무시는 구조의 결함인가, 데이터 지름길의 결과인가 — 그리고 이를 내부 지표로 진단할 수 있는가.**

| 가설 | 내용 | 현재 (2026-10-14) |
|---|---|---|
| **H1 (비적응성)** | attention 분배는 언어가 필요한 상황인지와 무관하게 일정하다 | 04: 지시 내용이 바뀌어도 R_norm +0.004. 언어 필수 조건(task 7/9) 미측정 |
| **H2 (지표 타당성)** | attention 비율은 인과 기여·행동 결과를 잘 예측하지 못한다 | 09: R ↔ causal_ratio 가 개입 방식에 따라 부호 반전. 행동 예측은 11(예정) |
| **H3 (원인)** | 내용 수준 Vision 편향은 배치 → 과제 1:1 이라는 **데이터 지름길** 때문이다 | task 7/9(배치 동일·목표 반대)와 10 의 R 조건으로 판별 예정 |

진행 단계: **존재(10) → 측정(11) → 원인(02·09 의 task 7/9, 06 보완) → 개선(추론 시 개입)**. 단계별 관문은 `docs/00_research_plan.md` §4.

---
## setup 파일 기능

설치 (순서대로 1회)

- `01_env.sh`: conda env + torch(드라이버 맞춰 cu118/cu121) + OpenVLA 설치
- `02_libero.sh`: LIBERO 설치 + numpy/opencv 버전 되돌리기
- `03_download_ckpt.py`: 체크포인트 다운로드 (~15GB)
- `04_verify.py`: 설치 상태 13개 항목 점검 (EGL 실렌더링 포함)

진단 (`04_verify.py` 실패 시)

- `05_diag_render.py`: 렌더링 실패 지점을 10단계로 특정 + 속도 측정
- `06_gpu_map.py`: nvidia-smi / torch / EGL 의 GPU 번호 체계 대조
- `07_egl_probe.py`: EGL 디바이스 수·벤더, NVIDIA GL 스택 유무
- `08_diag_robosuite.py`: robosuite ↔ mujoco ↔ LIBERO 버전 정합성

## 빠른 시작

```bash
# ── 노트북 (Windows) — 로직 검증만 ─────────────────────────────
pytest -q tests/test_core.py                  # 93 passed 나와야 정상

# ── 서버 (Linux + GPU) — 설치 ──────────────────────────────────
export PIP_CONSTRAINT=$(pwd)/constraints.txt  # ★ 먼저. 아래 함정 2 참고
bash setup/01_env.sh                          # conda env "vlamod" + torch + OpenVLA
bash setup/02_libero.sh                       # LIBERO + robosuite/MuJoCo
conda activate vlamod
python setup/03_download_ckpt.py --suite spatial     # ~15GB
python setup/04_verify.py --gpu 4             # ★ 설치 상태 점검
python setup/04_verify.py --gpus 0,6          #   두 장에 나눠 쓸 예정이면 합계로 판정

# ── 서버 — 실행 ────────────────────────────────────────────────
python scripts/01_smoke_forward.py --gpu 4                  # ★ 관문 (Q 행 argmax 7/7 확인)
python scripts/01_smoke_forward.py --gpu 4 --libero --suite spatial --task-id 0
python scripts/08_policy_sanity.py --gpu 4 --tasks 0 --max-steps 220 --gripper-fix both   # ★ 성공 1회 먼저

python scripts/10_icbench_rollout.py --gpu 4 --tasks 0 7 9 --inits 10           # 1단계: 행동 편향의 존재 (파일럿)
python scripts/02_run_analysis.py  --gpu 4 --suite spatial --tasks 0 1 2 3 4 --episodes 3 --max-steps 220
python scripts/03_correlation.py                                       # GPU 불필요 (02 출력 자동 탐색)
python scripts/04_counterfactual.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 7 9 --ambiguous-tasks 7 9
python scripts/06_image_ablation.py --gpu 4 --suite spatial --tasks 0 1 2 3 4   # 04 와 같은 GPU 종류
python scripts/09_embedding_substitution.py --gpu 4 --suite spatial --tasks 0 1 2 3 4   # 02 와 같은 GPU 종류
python scripts/05_plots.py         --suite spatial                      # GPU 불필요
```

> **궤적은 GPU 종류에 따라 달라집니다** (bf16 반올림·커널 차이). 서로 비교할 실험은 같은 종류 GPU 로 돌리세요.
> 같은 GPU 재실행은 완전히 재현됩니다.

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

모델을 올리는 **01 · 02 · 04 · 06 · 07 · 08 · 09 · 10 · setup-04** 가 아래 인자를 받습니다. **03 · 05 · `10 --dry-run` 은 GPU 를 안 씁니다.**

| 인자 | 예시 | 설명 |
|---|---|---|
| `--gpu N` | `--gpu 5` | **가장 간단.** 5번 GPU 사용 |
| `--gpus` | `--gpus 0,6` | 한 장에 안 들어갈 때 여러 GPU 에 분할 로드 |
| `--device` | `--device cuda:5` · `--device 5` · `--device cpu` | 문자열 지정 |
| `--model` | `--model openvla/openvla-7b-finetuned-libero-object` | 체크포인트 교체 |
| `--unnorm-key` | `--unnorm-key libero_object` | action un-normalization key (02 / 04) |
| `--tag` | `--tag object_ckpt` | 출력 하위 폴더·파일명 접미사 (02 · 04 · 06 · 09 · 10) |
| `--hf-home` | `--hf-home /data/hf` | HF 캐시 루트 |
| `--config` | `--config configs/lab.yaml` | 설정 파일 선택 |

`--gpu` / `--gpus` / `--device` 는 **셋 중 하나만** 쓸 수 있습니다.
생략하면 `configs/default.yaml` 의 `model.device` 를 씁니다.

로드 직후 아래가 출력됩니다. **요청한 번호가 그대로 찍히는지 매번 확인하세요.**

```
[env   ] HF_HOME=/home/…/kimhyeongwoo  MUJOCO_GL=egl  PYOPENGL_PLATFORM=egl  CUDA_DEVICE_ORDER=PCI_BUS_ID
[device] model=cuda:4  render(EGL)=egl:0  (EGL 디바이스가 1개뿐 → 선택 불가)
[model ] openvla/openvla-7b-finetuned-libero-spatial  unnorm_key=libero_spatial
[gpu   ] cuda:4 NVIDIA RTX A4000  이 프로세스 15.1GB  |  남은 여유 1.6GB / 전체 16.9GB      ← 09 실제 로그
```

### 어느 GPU 를 골라야 하는가

OpenVLA-7B(≈7.5B 파라미터) bf16 은 **가중치만 15GB**, 활성값·KV 캐시·attention 버퍼까지
**18~20GB** 가 필요합니다.

| 카드 | 총 | 판정 |
|---|---|---|
| A6000 / A100 (48GB+) | 48GB | 여유 충분 |
| **A5000 · RTX 3090 (24GB)** | 24GB | **거의 비어 있으면 단일 GPU 로 충분** |
| A4000 (16GB) | 16GB | **단독 로드 가능하나 빠듯함** (실측 여유 1.6GB, 02·09 는 완주). 긴 롤아웃에서 OOM 위험 |

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
- CPU 오프로드는 기본 금지 (`cpu: 0GiB`). 조용히 100배 느려지는 것을 막습니다.
- 로드 전에 **사용 가능 합계 vs 가중치 15GB** 를 비교해, 모자라면 선택지를 안내하고 멈춥니다.

**메모리가 빠듯할 때 쓰는 두 인자:**

| 인자 | 효과 | 주의 |
|---|---|---|
| `--headroom-gb 0.8` | GPU 당 안전 마진을 1.5 → 0.8GB 로 축소 | OOM 위험 증가 |
| `--allow-cpu-offload` | 못 올린 층을 CPU 로 | **매우 느림.** 01 구조 검증용으로만 |

```bash
# 16GB 두 장(여유 7.7 + 10.1 = 17.8GB)에서 억지로 돌리는 예
python scripts/01_smoke_forward.py --gpus 0,6 --headroom-gb 0.8
python scripts/01_smoke_forward.py --gpus 0,6 --headroom-gb 0.8 --allow-cpu-offload
```

### 멀티 GPU 서버의 함정 네 가지

**① GPU 번호 체계가 세 가지입니다.** `nvidia-smi`/`nvtop` 은 PCI 순서, CUDA 의 기본은
`FASTEST_FIRST`(성능순 재정렬), EGL 은 또 다른 목록입니다. 카드 모델이 섞인 서버에서는
셋이 전부 어긋납니다. `configs/default.yaml` 의 `env.cuda_device_order: "PCI_BUS_ID"`
가 torch 번호를 nvtop 과 맞춰 줍니다.

```bash
python setup/06_gpu_map.py     # 세 체계를 나란히 비교하고 판정
```

**② EGL 디바이스 번호는 CUDA 번호가 아닙니다.** GPU 가 10장이어도 EGL 은 1개만
열거할 수 있습니다. CUDA 번호를 그대로 넣으면 이렇게 죽습니다:

```
RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be
an integer between 0 and 0 (inclusive), got 8.
```

`--gpu`(torch) 와 `--egl-device`(렌더링) 는 **별개 인자**입니다. 미지정이면 코드가
EGL 목록을 실측해 자동 결정하고, 사유를 출력합니다.

```bash
python setup/07_egl_probe.py   # EGL 디바이스 수 / 벤더 / GPU인지 소프트웨어인지
```

**③ NVIDIA 드라이버의 그래픽 스택이 없을 수 있습니다.** CUDA(연산)만 설치된 서버가
흔합니다. `ldconfig -p | grep libEGL` 에 `libEGL_nvidia` 가 없고
`/usr/share/glvnd/egl_vendor.d/` 에 `50_mesa.json` 만 있으면 그 경우입니다.

- **동작은 합니다.** Mesa 소프트웨어 렌더링(llvmpipe)으로 떨어집니다.
- 렌더가 수백 ms 로 느려지지만, OpenVLA 7B forward 가 스텝당 1~3초라 **병목은 아닙니다.**
- GPU 렌더링이 필요하면 관리자에게: 드라이버 설치 시 `--no-opengl-files` 를 빼거나
  (`libEGL_nvidia.so` + `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`),
  컨테이너면 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` 로 실행.

**④ `CUDA_VISIBLE_DEVICES` 와 섞어 쓰지 마세요.** 섞으면 번호가 0부터 재매핑됩니다.
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
  token_index.py   구간 실측 S|V×256|T|L|T|A×7, Q = A−1, 29871 덧붙이기(prepare_inputs) + 육안 검증 출력
  capture.py       2-pass 캡처: generate → teacher-forced forward (Q 행 검사)
  metrics.py       R_raw / R_norm / R_vnorm, 질량 분해, action KL, causal_ratio
  intervene.py     pre-softmax attention knockout (Q 행 → 대상: 직접 효과) + 무작위 대조군
  substitute.py    0층 임베딩 치환 (평균 기준값: 총 효과) + 정렬 검사·기록기 (09)
  pipeline.py      정책 행동(그리퍼 규약 변환) + 한 타임스텝 = 관찰 + 개입 → tidy row
  preprocess.py    이미지 변형 (08 전처리 후보, 06 hflip / patch_shuffle / noise)
  env_libero.py    LIBERO 얇은 래퍼 (+ suite 전체 지시문)
  icbench.py       ICBench 방식 모순 지시 7종 + 결과 판정 (10)
  paths.py         출력 경로 규약 outputs/<NN_스크립트명>/
  viz.py           그림 (색맹 판별 검증 완료 3색)
setup/
  01_env.sh          conda + torch + OpenVLA (드라이버 보고 cu118/cu121 자동 선택)
  02_libero.sh       LIBERO + 버전 충돌 되돌리기 + 검증
  03_download_ckpt.py 체크포인트 (config 의 env.hf_home 사용)
  04_verify.py       ★ 설치 없이 상태만 12개 항목 점검 (EGL 실렌더링 포함)
                       --gpu N / --gpus 0,6 / --egl-device E
  ── 아래는 04 가 실패했을 때만 쓰는 진단 도구 ──
  05_diag_render.py  렌더링 실패를 10단계로 쪼개 어디서 죽는지 특정.
                     GL_RENDERER(GPU/소프트웨어) 와 렌더 속도도 측정
  06_gpu_map.py      nvidia-smi / torch / EGL 세 번호 체계 대조·판정
  07_egl_probe.py    EGL 디바이스 수·벤더, NVIDIA GL 스택 존재 여부,
                     root 없는 우회 가능 여부
  08_diag_robosuite.py robosuite ↔ mujoco ↔ LIBERO 버전 정합성.
                     "메시지 없는 AssertionError" 전용
scripts/
  01_smoke_forward.py          ★ 관문  구조 검증 (구간·Q 행·캡처·지표)
  02_run_analysis.py                   롤아웃 + 매 스텝 R 3종 + knockout KL
  03_correlation.py            H2      상관 (t-검정 부분은 교체 예정)
  04_counterfactual.py         H1      지시 조건 4종 (개입 지점 L1)
  05_plots.py                          그림 4장
  06_image_ablation.py                 이미지 조건 5종 (L0, 04 의 이미지판)
  07_dump_model_input.py               모델이 실제로 받은 pixel_values 를 PNG 로 복원
  08_policy_sanity.py          ★ 선행  정책 성공 확인 (이미지 전처리 × 그리퍼 변환)
  09_embedding_substitution.py H2·원인 0층 평균 치환(총 효과) vs knockout(직접 효과), 같은 프레임 (L2)
  10_icbench_rollout.py        존재    ICBench 방식 모순 지시 롤아웃 (가짜 성공 vs 실제 실패)
tests/test_core.py      GPU 없이 도는 단위 테스트 93개
```

### 산출물

```
outputs/
  01_smoke_forward/      smoke_libero_view.png, smoke_capture.pt
  02_run_analysis/       analysis_<suite>.csv (타임스텝별 tidy row), perlayer_<suite>.npz (층별 배열)
    <tag>/               --tag 를 준 실행은 하위 폴더 (파일명에도 tag 유지)
  03_correlation/        correlation.csv
  04_counterfactual/     counterfactual_<suite>.csv
  05_plots/              fig_*_<suite>.png
  06_image_ablation/     image_ablation_<suite>.csv, examples/
  07_dump_model_input/   input_*.png
  09_embedding_substitution/  embed_sub_<suite>.csv, baselines_<suite>.pt, gates_<suite>.txt
  10_icbench_rollout/    variants_<suite>.csv, rollouts_<suite>.csv, traj_<suite>.npz
  setup_04_verify/       verify_render.png
  setup_05_diag_render/  diag_render.png
```

경로 규약은 `vlamod/paths.py` 한 곳에 있습니다. 03·05·06 은 다른 스크립트의 출력을 새 위치에서 먼저 찾고,
없으면 정리 전의 `outputs/<파일명>` 을 경고와 함께 읽습니다.

---

## 지표 정의

action query s, layer l, head h 에 대해:

| 지표 | 정의 | 용도 |
|---|---|---|
| `R_raw` | `M_L / (M_L + M_V)` | IVAR(2603.06001) 재현. **단독 주장 금지** |
| `R_norm` | `(M_L/\|L\|) / (M_L/\|L\| + M_V/\|V\|)` | 토큰 개수 비대칭 제거 |
| `R_vnorm` | `A(s,j)` 대신 `A(s,j)·‖v_j‖` | Kobayashi+2020. 실제 전달되는 벡터 크기 |
| `causal_ratio` | `KL_L / (KL_L + KL_V)` | 인과 기여 비율. ⚠ **개입 방식(knockout / 치환)에 따라 값과 추세가 달라짐** (09) |
| `lang_minus_control` | `KL_L − KL_random-visual` | 대조군을 뺀 언어 효과 (`lang_vs_control` 비율은 분모 ≈ 0 이라 불안정) |
| `uniform_baseline` | `\|L\|/(\|L\|+\|V\|)` | `R_raw` 의 기본값. 항상 같이 보고할 것 |

`|V|` = 256, `|L|` = 15–21 (spatial 실측)이므로 균등 분포만 가정해도 `R_raw ≈ 0.055–0.076` 이 나옵니다.
**`uniform_baseline` 보다 낮아야** "언어를 덜 본다"고 말할 수 있습니다. 실측 R_raw ≈ 0.42 (baseline 의 약 6배).
비율이 오른 것이 Text 질량 증가인지 Vision 질량 감소인지는 `mass_*` 로 따로 확인하세요. 전체 정의: `docs/06_metrics.md`.

---

## 확인 상태

확인됨 (실측):
- 프롬프트 템플릿 — L 태깅 육안 검증 + 정책 성공 (13/15)
- action bin ↔ vocab id — 01 의 범위 검사 통과, Q 행 argmax == 생성 토큰 7/7
- instruction 속성 (`task.language`), transformers 4.40.1 의 `attention_mask` kwarg (knockout 동작)
- 0층 hook 규약 (`LlamaDecoderLayer` 가 hidden_states 를 위치 인자로 받음) — 09 의 G1·G2 통과
- 그리퍼 규약 [0,1] → [−1,+1] 변환 필요 (08)

미확인:
- **`--gpus` 분할 로드** — `trust_remote_code` 모델이 accelerate `device_map` 을 지원하지 않으면 실패 (명확한 에러를 냄)
- 06 의 R 상승이 Text 질량 증가인지 Vision 질량 감소인지 (절대 질량 미기록)
- 10 의 모순 지시가 장면별로 진짜 모순인지 — task 0 이미지와 bddl 로만 확인, 나머지 장면은 `--dry-run` 표 검토 권장
