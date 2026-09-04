# 실행 런북 — 무엇을, 어떤 순서로, 어디까지 확인하고 넘어갈 것인가

작성일: 2026-08-25 (개정: 2026-09-04 — GPU 인자 반영)
전제: A6000 서버, Ubuntu, CUDA 12.x / 로컬은 C:\Users\hello\PycharmProjects\VLA_Interpretability

> 각 단계에 **통과 기준(Gate)** 이 있습니다. 기준을 못 넘으면 다음 단계로 가지 마세요.
> 앞 단계의 가정 위에서 뒤 단계가 돌기 때문에, 조용히 틀린 채로 진행하면 3일 뒤에 전부 다시 해야 합니다.

---

## 파이프라인 한눈에

```
Stage 0   노트북      pytest                     ← GPU 불필요. 로직 검증
   ↓
Stage 1   서버        setup/01,02,03             ← 환경 + LIBERO + 체크포인트
   ↓
Stage 2   서버 ★      scripts/01_smoke_forward   ← 구조 검증 7단계. 여기가 관문
   ↓
Stage 3   서버        scripts/02_run_analysis    ← 데이터 수집 (비율 + knockout KL)
   ↓
        ┌────────────┴────────────┐
        ▼                         ▼
Stage 4  scripts/03_correlation   Stage 5  scripts/04_counterfactual
         (H2: 지표 타당성)                  (H1: 비적응성)
        └────────────┬────────────┘
                     ▼
Stage 6   scripts/05_plots        ← 그림 4장
```

산출물 흐름:

```
02 ─▶ outputs/analysis_<suite>.csv     (타임스텝별 tidy row)
   ─▶ outputs/perlayer_<suite>.npz     (층별 배열)
04 ─▶ outputs/counterfactual_<suite>.csv
03 ─▶ outputs/correlation.csv
05 ─▶ outputs/fig_*.png
```

---

## Stage 0 — 노트북에서 로직 검증 (5분, GPU 불필요)

```bash
cd C:\Users\hello\PycharmProjects\VLA_Interpretability
pip install torch pytest          # CPU 빌드로 충분
pytest -q tests/test_core.py
```

**Gate 0**: `34 passed` 가 나와야 합니다.

여기서 검증되는 것: 지표 계산식, 질량 분해 합=1, KL 성질, knockout 마스킹 후 softmax 재정규화,
프롬프트 오프셋 매핑, **device 인자 파싱**. **모델 없이 잡을 수 있는 실수는 여기서 다 잡힙니다.**

실패하면 서버에 갈 필요가 없습니다. 실패한 테스트 이름을 그대로 알려주세요.

---

## Stage 1 — 서버 환경 구축 (1~2시간, 대부분 다운로드 대기)

```bash
# 저장소를 서버로 — 셋 중 하나
#  (a) PyCharm: Tools → Deployment → Upload to...   ← .idea/deployment.xml 이 있으면 이 방식
#  (b) rsync
rsync -av --exclude '.git' --exclude '.idea' --exclude '__pycache__' \
      ~/PycharmProjects/VLA_Interpretability/ user@server:~/VLA_Interpretability/
#  (c) git push 후 서버에서 clone

ssh user@server
cd ~/VLA_Interpretability

# 전송 누락 확인 — 이 파일이 없으면 01 스크립트가 ModuleNotFoundError 로 죽습니다
ls vlamod/device.py

bash setup/01_env.sh              # conda env "vlamod" + torch + OpenVLA
bash setup/02_libero.sh           # LIBERO + robosuite/MuJoCo
conda activate vlamod
# (MUJOCO_GL 등은 config 가 설정하므로 export 불필요. 설치 스크립트는 자체 export 함)

# 체크포인트 (~15GB). 저장 위치는 configs/default.yaml 의 env.hf_home 이 정합니다.
python setup/03_download_ckpt.py --suite spatial
```

> **환경변수는 `.bashrc` 가 아니라 `configs/default.yaml` 의 `env:` 섹션에서 관리합니다.**
> 다운로드 스크립트와 분석 스크립트가 같은 값을 읽으므로, 경로가 바뀌면 그 한 줄만 고치면 됩니다.
> 서버가 여러 대면 `configs/<서버명>.yaml` 로 복사해 두고 `--config` 로 고르세요 —
> 재현할 때 "어떤 config 로 돌렸는지"만 남기면 됩니다.
>
> ```yaml
> env:
>   hf_home: "~/shared/hdd_ext/nvme1/kimhyeongwoo"
>   mujoco_gl: "egl"
>   pyopengl_platform: "egl"
> ```

**Gate 1** — 아래 셋이 모두 통과해야 합니다.

```bash
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
python -c "import transformers; print(transformers.__version__)" # 4.40.x
python -c "from libero.libero import benchmark; print(list(benchmark.get_benchmark_dict().keys()))"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv   # 어느 GPU 가 비었는지
```

**여기서 쓸 GPU 번호를 정해 두세요.** 이후 모든 명령에 `--gpu N` 으로 넘깁니다.
`nvidia-smi` 의 `memory.used` 가 낮은 번호를 고르시면 됩니다. OpenVLA-7B(bf16) 는
가중치 약 16GB + attention 버퍼가 필요하므로 여유 20GB 이상을 권합니다.

`01_env.sh` 안에 transformers 4.40 검사 `assert` 가 들어 있어서, 버전이 다르면 설치 단계에서 멈춥니다.

### 여기서 자주 막히는 것

| 증상 | 원인 / 조치 |
|---|---|
| `libero` import 시 EGL 에러 | `MUJOCO_GL=egl` 미설정. export 후 재시도 |
| `libero_requirements.txt` 없음 | OpenVLA repo 구조 변경. `ls ~/third_party/openvla/experiments/robot/` 로 확인 |
| 체크포인트 404 | HF repo 이름 변경 가능. huggingface.co/openvla 에서 실제 이름 확인 후 `setup/03_download_ckpt.py` 의 `CKPTS` 수정 |
| conda 없음 | miniconda 설치 후 재시도. venv 로도 되지만 스크립트를 고쳐야 함 |

---

## GPU 지정 (Stage 2 이후 모든 명령에 공통)

모델을 올리는 **01 / 02 / 04** 만 아래 인자를 받습니다. **03 / 05 는 GPU 를 쓰지 않습니다.**

| 인자 | 예시 | 설명 |
|---|---|---|
| `--gpu N` | `--gpu 3` | 가장 간단. 3번 GPU 사용 |
| `--device` | `--device cuda:3` · `--device 3` · `--device cpu` | 문자열 지정. `--gpu` 와 동시 사용 불가 |
| `--model` | `--model openvla/openvla-7b-finetuned-libero-object` | 체크포인트 교체 |
| `--unnorm-key` | `--unnorm-key libero_object` | action un-normalization key (02 / 04) |
| `--tag` | `--tag object_ckpt` | 출력 파일명 접미사 (02 / 04) |

생략하면 `configs/default.yaml` 의 `model.device` 를 씁니다.

로드 직후 아래가 출력됩니다. **요청한 번호가 그대로 찍히는지 매번 확인하세요.**

```
[device] model=cuda:3  render(EGL)=3
[model ] openvla/openvla-7b-finetuned-libero-spatial  unnorm_key=libero_spatial
[gpu   ] cuda:3 NVIDIA RTX A6000  사용가능 47.2GB / 전체 51.5GB
```

### 멀티 GPU 서버의 함정 두 가지

**① PyTorch 와 MuJoCo 는 GPU 를 따로 고릅니다.**
모델을 `cuda:3` 에 올려도 LIBERO 의 EGL 렌더링은 기본적으로 0번을 씁니다.
0번이 남의 작업으로 꽉 차 있으면 **모델은 멀쩡한데 렌더링만 죽는** 에러가 납니다.
`--gpu` 를 쓰면 `MUJOCO_EGL_DEVICE_ID` 를 같은 번호로 맞춰 주므로 이 문제가 사라집니다.
출력의 `render(EGL)=` 이 `model=` 과 같은 번호인지 확인하세요.

**② `CUDA_VISIBLE_DEVICES` 와 섞어 쓰지 마세요.**
섞으면 번호가 0부터 재매핑됩니다 — `CUDA_VISIBLE_DEVICES=3` 상태의 `--gpu 3` 은
존재하지 않는 GPU 입니다. 코드가 이 상황을 감지해 에러를 냅니다. 한 방식만 쓰세요.

---

## Stage 2 ★ — 스모크 테스트 (30분, 이 프로젝트의 관문)

```bash
python scripts/01_smoke_forward.py --gpu 0                        # 합성 이미지, 구조만 검증
python scripts/01_smoke_forward.py --gpu 0 --libero --suite spatial --task-id 0
```

스크립트가 7단계를 순서대로 출력합니다. **각 단계의 통과 기준**은 이렇습니다.

| 단계 | 출력 | 통과 기준 | 실패 시 |
|---|---|---|---|
| `[1]` 모델 로딩 | layers, heads, vocab_size | layers=32, heads=32 (Llama-2-7B) | 백본 탐색 실패 → `find_language_model` 확인 |
| `[2]` 프롬프트 | 프롬프트 문자열 + instruction 문자 구간 | 구간이 지시문을 정확히 가리킴 | `token_index.PROMPT_TEMPLATE` 을 실제 체크포인트 템플릿으로 수정 |
| `[3]` visual span 실측 | `[1, 257) → n_visual = 256` | **start=1, n_visual=256 이면 이상적**. 다르면 경고가 뜨지만 진행은 됨 | 경고를 무시하지 말고 모델 구조를 확인. 숫자 자체는 코드가 알아서 씀 |
| `[4]` 구간 분해 | 토큰별 S/L/T 태그 | **★ 육안 검증. L 이 instruction 내용어에만** | `In`, `What`, `action` 에 L → 오프셋 매핑 실패 |
| `[5]` action token | 7개 id + bin 범위 | `범위 검증: 통과` | `action_bin_token_ids` 의 오프셋을 ±1 조정 |
| `[6]` 캡처 | attn/vnorm/logits shape | attn = `(32, 32, 7, T)`, **행 합 min/max 모두 1.0000** | 행 합이 1이 아니면 attention 캡처가 잘못된 것 |
| `[7]` 지표 | R_raw / R_norm / R_vnorm + baseline | 셋 다 `[0, 1]` 안, NaN 없음 | mass_sink 가 0.9 이상이면 sink 처리 재검토 |

**Gate 2**:
1. `[5] 범위 검증: 통과`
2. `[6] attention 행 합`이 1.0000
3. `[4]` 육안 검증 통과 — **이건 자동화 불가, 직접 보셔야 합니다**
4. `--libero` 실행 시 `outputs/smoke_libero_view.png` 를 열어 **로봇 팔이 똑바로 서 있는지** 확인
   (뒤집혀 있으면 `env_libero.obs_to_image(flip=False)` 로 바꾸고 다시)

### `[4]` 육안 검증 예시

```
  [   0]        S  '<s>'
  [   1..256]   V  (visual ×256)
  [ 257]        T  '▁In'
  [ 258]        T  ':'
  ...
  [ 268]        L  '▁pick'      ← 여기부터
  [ 269]        L  '▁up'
  [ 270]        L  '▁the'
  [ 271]        L  '▁black'
  [ 272]        L  '▁bowl'      ← 여기까지가 L
  [ 273]        T  '?'
  [ 274]        T  '▁Out'
```

이 모양이 아니면 뒤의 숫자는 전부 의미가 없습니다.

---

## Stage 3 — 데이터 수집 (1~3시간)

```bash
# 먼저 작게 돌려 시간을 재세요
python scripts/02_run_analysis.py --gpu 0 --suite spatial --tasks 0 --episodes 1 --max-steps 20 --stride 5

# 문제 없으면 본 실행
python scripts/02_run_analysis.py --gpu 0 --suite spatial --tasks 0 1 2 3 4 --episodes 3 --max-steps 40 --stride 5
```

### GPU 가 여러 장이면 체크포인트를 병렬로

한 체크포인트 결과만으로는 "그 모델 얘기 아니냐"를 막을 수 없습니다.
**최소 2개 체크포인트**에서 같은 성질이 나오는지 확인해 두면, 나중에 실물(UR5e)로
넘어갈 때 전이를 주장하는 근거가 됩니다. `--tag` 로 출력 파일명이 갈립니다.

```bash
python scripts/02_run_analysis.py --gpu 0 --suite spatial \
       --model openvla/openvla-7b-finetuned-libero-spatial \
       --unnorm-key libero_spatial --tag spatial_ckpt &
python scripts/02_run_analysis.py --gpu 1 --suite object \
       --model openvla/openvla-7b-finetuned-libero-object \
       --unnorm-key libero_object --tag object_ckpt &
wait
# → outputs/analysis_spatial_spatial_ckpt.csv, outputs/analysis_object_object_ckpt.csv
```

### 파라미터가 무엇을 정하는가

| 인자 | 의미 | 권장 |
|---|---|---|
| `--tasks` | 분석할 task id | 최소 5개 (일반화 주장의 최소선) |
| `--episodes` | task 당 에피소드 | 3 이상 (분산 보고용) |
| `--max-steps` | 에피소드 최대 길이 | 40 (LIBERO 성공 궤적 대부분 커버) |
| `--stride` | 몇 스텝마다 분석 | 5. 1로 하면 5배 느림 |
| `--no-intervene` | 개입 생략 | **쓰지 마세요.** 관찰만으로는 결론 못 냄 |

### 소요시간 추정 (측정값 아님 — 직접 재세요)

분석 1스텝 = generate 1회 + teacher-forced forward 4회(무개입 + knockout 3종).
A6000 bf16 + **eager attention**(flash 보다 느림) 기준 대략 **2~4초/스텝**으로 예상합니다.
위 본 실행 설정이면 분석 대상 스텝 = 5 task × 3 ep × 8 스텝 = 120개 → **10~30분** 수준.
여기에 환경을 굴리는 rollout forward 가 더해집니다.

> **정확한 숫자는 제가 모릅니다.** 실제 모델을 못 돌려봤습니다. 작은 설정으로 먼저 재보세요.

### 진행 중 화면에서 볼 것

```
  t=  0 R_raw=0.0421 R_norm=0.4832 R_vn=0.3910 | KL_L=0.0031 KL_V=2.8740 KL_ctrl=0.0028
```

- `KL_V >> KL_L` : 예상되는 결과. 비전이 인과적으로 지배적
- **`KL_L ≈ KL_ctrl`** : 언어 차단이 랜덤 토큰 차단과 구별 안 됨 → 가장 강한 형태의 "언어 무시" 증거
- `R_raw` 가 `uniform_baseline`(마지막 요약에 출력) 근처 : attention 이 언어에 특별히 안 쏠림

**Gate 3**: CSV 가 생기고, `R_*` 에 NaN 이 없고, `KL_*` 이 모두 유한한 양수.

---

## Stage 4 — H2 검정 (1분)

```bash
python scripts/03_correlation.py --csv outputs/analysis_spatial.csv   # GPU 불필요
```

**질문**: attention 비율이 인과 기여를 예측하는가?

| 결과 | 해석 | 논문에서의 쓸모 |
|---|---|---|
| `\|ρ\| < 0.3` | attention 비율은 인과 기여를 거의 예측 못 함 → **H2 지지** | 2605.00321 의 "attention 비판"을 정량 확인. IVAR 류 지표 전반에 경고 |
| `\|ρ\| > 0.6` | attention 비율이 쓸 만한 proxy → **H2 기각** | 기존 attention 연구를 **방어**해 주는 결과. 이것도 기여 |
| 0.3~0.6 | 조건부 | 층/head 별로 나눠 다시 |

같이 출력되는 **대조군 검정**이 더 중요할 수 있습니다:

```
[대조군 검정] lang_vs_control 평균=1.08, H0(=1) 대비 t=0.72, p=0.47
```

`p` 가 크면 → 언어 토큰 차단이 같은 개수의 랜덤 visual 토큰 차단과 **통계적으로 구별되지 않음**.
"VLA 가 언어를 안 쓴다"의 가장 엄밀한 형태입니다.

---

## Stage 5 — H1 검정 (1~2시간 + 사람 손 필요)

### 5-a. 먼저 모호 장면 task 를 골라야 합니다 ← **자동화 불가**

```bash
# 각 task 의 지시문과 첫 프레임을 뽑아 눈으로 확인
python scripts/01_smoke_forward.py --gpu 0 --libero --suite object --task-id 0
# outputs/smoke_libero_view.png 를 열어본다. task-id 를 바꿔가며 반복
```

**고르는 기준**: 지시문이 지목하는 물체와 **혼동될 수 있는 후보가 장면에 둘 이상** 있을 것.
예) "pick up the *black* bowl" 인데 장면에 흰 그릇도 있는 경우 → 색을 안 읽으면 틀림.
후보가 하나뿐이면 언어를 안 봐도 정답이라 **아무것도 측정되지 않습니다.**

`libero_object` suite 가 가장 후보가 많습니다. 3~5개를 고르세요.

### 5-b. 실행

```bash
python scripts/04_counterfactual.py --gpu 0 --suite object \
       --tasks 0 1 2 3 4 5 --episodes 2 \
       --ambiguous-tasks 1 3 5          # ← 5-a 에서 고른 id
```

### 4개 조건이 각각 무엇을 묻는가

| 조건 | 지시문 | 묻는 것 |
|---|---|---|
| `valid` | 원래 지시 | 기준선 |
| `paraphrase` | 동의어 치환 ("pick up"→"grasp") | 의미가 같으면 행동도 같아야 함. 다르면 **표면 형태에 과적합** |
| `swapped` | 같은 suite 의 **다른 task 지시** | 지시가 바뀌었는데 행동이 그대로면 **언어 무시** |
| `empty` | 빈 지시 | 2510.13626 재현. 단 **OOD 라 해석 주의** |

### 읽는 법

```
모순 지시에서 7 DoF 가 전부 그대로인 비율: 78.3%
```
→ 지시를 바꿔도 78%의 경우 완전히 같은 행동. linguistic blindness 재현.

```
R_norm(valid) - R_norm(swapped) = +0.0021
```
→ 비율이 지시 내용에 거의 반응 안 함. **H1 지지**.

```
--- 모호 장면 vs 비모호 장면 ---
ambiguous  condition    R_norm   KL_vs_valid
False      swapped      0.4831   0.0042
True       swapped      0.4855   0.0051
```
→ 모호 장면에서도 `R_norm` 이 안 올라감. **H1 의 핵심 증거**.
언어가 반드시 필요한 상황인데도 비율이 그대로라는 뜻입니다.

**Gate 5**: `--ambiguous-tasks` 를 **반드시** 지정할 것. 없으면 스크립트가 경고를 출력하고,
그 상태의 결과로는 "LIBERO 탓 아니냐"는 반박을 막을 수 없습니다.

---

## Stage 6 — 그림 (1분)

```bash
python scripts/05_plots.py --suite spatial     # GPU 불필요
python scripts/05_plots.py --suite object      # counterfactual 그림
```

| 파일 | 내용 | 발표에서의 역할 |
|---|---|---|
| `fig_layerwise_*.png` | 층별 R 3종 + uniform baseline | "어느 층에서 언어를 보는가" |
| `fig_h2_*.png` | R_norm vs causal_ratio 산점도 (ρ 표기) | **H2 핵심 슬라이드** |
| `fig_cf_ratio_*.png` | 조건별 R_norm 막대 | **H1 핵심 슬라이드** |
| `fig_cf_kl_*.png` | 조건별 KL 막대 | H1 보조 |

---

## 2주 스프린트 배치

| 일차 | 할 일 | 산출물 |
|---|---|---|
| 1 | Stage 0 + Stage 1 (+ 쓸 GPU 번호 확정) | 환경 완성 |
| 2~3 | **Stage 2** (여기서 대부분의 시간이 갑니다) | 구조 검증 통과 |
| 4 | Stage 3 소규모 → 시간 측정 → 규모 확정 | 파라미터 확정 |
| 5~6 | Stage 3 본 실행 | `analysis_spatial.csv` |
| 7 | Stage 4 | **H2 1차 결과** |
| 8~9 | Stage 5-a (모호 task 선정) | task id 목록 |
| 10~11 | Stage 5-b | `counterfactual_object.csv` |
| 12 | Stage 6 + 결과 정리 | 그림 4장 |
| 13~14 | 추가 suite / seed 확장, 분산 보고 | 재현성 확보 |

**14일 차의 목표 문장** — 이 한 줄이면 교수님 미팅에 충분합니다:

> "attention 비율은 X 였는데 인과 기여 비율은 Y 였고, 둘의 Spearman 상관은 ρ=Z 였습니다.
> 그리고 언어가 반드시 필요한 모호 장면에서도 비율이 유의하게 변하지 않았습니다."

---

## 알려진 비효율 (지금은 정확성 우선, 나중에 최적화)

1. **`02_run_analysis.py` 가 스텝마다 모델을 두 번 돌립니다.**
   분석용 캡처와, 환경을 굴리기 위한 `predict_action` 이 별개로 돕니다.
   정확성 문제는 아니지만 분석 스텝에서 약 2배 느립니다.
   → 느려서 문제가 되면 `analyze_step` 이 뱉은 `action_bins` 를 재사용하도록 고칠 수 있습니다.
   (다만 bin → 연속 action 역변환의 un-normalization 을 직접 해야 해서, 지금은 안전한 쪽을 택했습니다)
2. **`obs_to_image()` 가 한 스텝에 두 번 호출**됩니다. 비용은 작습니다.
3. **eager attention 은 flash 보다 느립니다.** 분석용이라 어쩔 수 없습니다.

---

## 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `attentions 가 None 입니다` | eager 아님 | `configs/default.yaml` 의 `attn_implementation` 확인 |
| `4D attention mask 를 기대했는데 ND` | transformers 버전 | 4.40.x 로 내리거나, 해당 버전의 `LlamaDecoderLayer.forward` 시그니처 확인 |
| `언어 구간 L 이 비었습니다` | 프롬프트 템플릿 불일치 | `token_index.PROMPT_TEMPLATE` 수정 |
| `visual 구간이 연속적이지 않습니다` | 모델 구조 가정 위반 | `probe_visual_span` 출력의 changed 인덱스를 직접 보기 |
| `action 위치가 시퀀스 길이를 벗어납니다` | visual 삽입 위치 가정 오류 | Stage 2 `[3]` 재확인 |
| `범위 검증: 실패` | action bin 오프셋 ±1 | `action_bin_token_ids` 의 `v - n_bins`, `v` 를 조정 |
| CUDA OOM | 48GB 면 안 날 텐데 | 다른 프로세스 확인. `nvidia-smi` |
| `n_identical_dof` 가 항상 7 | 정상일 수 있음 (언어 무시) | 단, `valid` 조건에서도 7이면 캡처가 잘못된 것 |
| 그림에 글자가 네모 | 한글 폰트 없음 | 축·제목은 이미 영문. 파일명에 한글이 있으면 그것만 바꾸기 |
| `ModuleNotFoundError: vlamod.device` | 서버 전송 누락 | `vlamod/device.py` 업로드. PyCharm Deployment 는 수동 업로드 필요 |
| `--gpu 와 --device 는 같이 쓸 수 없습니다` | 두 인자 동시 사용 | 하나만 주기 |
| `CUDA_VISIBLE_DEVICES=... 로 N개만 보이는데` | 환경변수와 `--gpu` 혼용 | 둘 중 하나만. 환경변수를 쓰면 `--gpu 0` |
| 모델은 뜨는데 LIBERO 렌더링만 실패 | EGL 이 다른 GPU 사용 | `[device]` 출력의 `render(EGL)` 이 `model` 과 같은지 확인 |
| `GPU N 를 요청했지만 보이는 GPU 는 M개` | 번호 범위 초과 | `nvidia-smi` 로 실제 번호 확인 |
| `사용가능 XGB` 경고 (18GB 미만) | 그 GPU 가 이미 사용 중 | 다른 번호로 `--gpu` 변경 |

---

## 결과가 "재미없게" 나오는 경우

이 연구는 **음성 결과도 결과입니다.** 아래 셋 중 무엇이 나와도 발표할 것이 있습니다.

| 나온 결과 | 이야기 |
|---|---|
| 언어를 거의 안 쓴다 (예상) | 선행연구 재현 + **비율이 demand 에 반응 안 함**(H1)이라는 더 구체적 진술 |
| 언어를 생각보다 많이 쓴다 | 선행연구와 불일치 → 어떤 조건에서 갈리는지가 새 질문. 2510.13626 vs 2601.04052 의 불일치와 연결 |
| attention 비율과 인과 기여가 무관 | **H2 지지.** IVAR 류 지표 기반 후속 연구 전체에 대한 경고 |
| attention 비율과 인과 기여가 잘 맞음 | H2 기각. attention 지표를 정당화 — 이것도 기여 |

가장 위험한 것은 결과가 아니라 **"그래서 뭐(so what)"에 답을 준비 안 하는 것**입니다.
계획서 §6 의 방어 문장을 미리 외워 두세요.
