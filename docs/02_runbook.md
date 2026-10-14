# 실행 런북 — 무엇을, 어떤 순서로, 어디까지 확인하고 넘어갈 것인가

작성 2026-08-25 · 개정 2026-09-04(GPU 인자) · **전면 개정 2026-10-14 (스크립트 번호 기준, 06–10 추가)**
전제: 랩 GPU 서버(Ubuntu) / 로컬은 `C:\Users\hello\PycharmProjects\VLA_Interpretability`

> 실험은 **스크립트 번호**로만 부릅니다 ("Stage" 표기 폐지).
> 각 스크립트에 **통과 기준(Gate)** 이 있습니다. 못 넘으면 다음으로 가지 마세요 — 앞 스크립트의 가정 위에서 뒤가 돕니다.
> 연구 질문과 단계(존재 → 측정 → 원인 → 개선)는 `docs/00_research_plan.md` §4.

---

## 파이프라인 한눈에

```
pytest (노트북, GPU 불필요) ─▶ setup/01–04 (서버 환경) ─▶ 01 구조 검증 ★관문 ─▶ 08 정책 성공 1회 ★
                                                                                      │
        ┌────────────────────────┬───────────────────────┬────────────────────────────┤
        ▼                        ▼                       ▼                            ▼
   02 롤아웃 + R + knockout   04 지시 조건            06 이미지 조건            09 치환 vs knockout
        │                        │                       │                            │
   03 상관(H2) · 05 그림 ◀───────┘                       │                            │
                                                         ▼                            ▼
                             10 ICBench 방식 롤아웃 (1단계: 편향 존재) ─▶ 11 가짜 성공 예측 (예정)
```

07 은 모델 입력 픽셀 확인용 (필요할 때만).

산출물:

```
outputs/
  01_smoke_forward/      smoke_libero_view.png, smoke_capture.pt
  02_run_analysis/       analysis_<suite>.csv, perlayer_<suite>.npz   (<tag>/ 하위 폴더)
  03_correlation/        correlation.csv
  04_counterfactual/     counterfactual_<suite>.csv
  05_plots/              fig_*_<suite>.png
  06_image_ablation/     image_ablation_<suite>.csv, examples/
  07_dump_model_input/   input_*.png
  09_embedding_substitution/  embed_sub_<suite>.csv, baselines_<suite>.pt, gates_<suite>.txt
  10_icbench_rollout/    variants_<suite>.csv, rollouts_<suite>.csv, traj_<suite>.npz
  setup_04_verify/  ·  setup_05_diag_render/
```

경로 규약은 `vlamod/paths.py`. 03·05·06 은 다른 스크립트 출력을 새 위치에서 먼저 찾고, 없으면 옛 `outputs/<파일명>` 을 경고와 함께 읽습니다.
**저장소 루트에서 실행하세요** (경로가 작업 디렉터리 기준).

---

## 0. 노트북에서 로직 검증 (5분, GPU 불필요)

```bash
cd C:\Users\hello\PycharmProjects\VLA_Interpretability
pip install torch pytest numpy pandas pillow pyyaml
pytest -q tests/test_core.py
```

**Gate**: `93 passed`. 지표 계산식, 질량 합 = 1, KL 성질, knockout 재정규화, 프롬프트 오프셋, 29871·Q 행, device 인자,
이미지 변형, 경로 규약, 치환 hook·정렬 검사, ICBench 변형 규칙이 검증됩니다. 실패하면 서버에 갈 필요가 없습니다.

---

## setup — 서버 환경 구축 (1~2시간, 대부분 다운로드 대기)

```bash
# 저장소를 서버로 — git push 후 서버에서 git pull (권장) / rsync / PyCharm Deployment
#  !! vlamod/ 를 스크립트와 **함께** 올리세요. 부분 동기화하면 AttributeError 가 납니다.
cd ~/private/test/VLA_Interpretability
export PIP_CONSTRAINT=$(pwd)/constraints.txt    # numpy<2 등 고정
bash setup/01_env.sh              # conda env "vlamod" + torch + OpenVLA
bash setup/02_libero.sh           # LIBERO + robosuite/MuJoCo
conda activate vlamod
python setup/03_download_ckpt.py --suite spatial    # ~15GB, 위치는 configs/default.yaml 의 env.hf_home
python setup/04_verify.py --gpu N                    # 상태 점검 (EGL 실렌더링 포함)
```

> **환경변수는 `.bashrc` 가 아니라 `configs/default.yaml` 의 `env:` 섹션**에서 관리합니다.
> 서버가 여러 대면 `configs/<서버명>.yaml` 로 복사해 `--config` 로 고르세요.

**Gate**: `setup/04_verify.py` 전 항목 통과, `outputs/setup_04_verify/verify_render.png` 에서 로봇 팔이 똑바로 서 있음.
실패하면 `05_diag_render.py`(렌더링) / `06_gpu_map.py`(GPU 번호) / `07_egl_probe.py`(EGL) / `08_diag_robosuite.py`(버전) 순으로 진단.

고정 버전: torch 2.2.0+cu118, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, numpy 1.26.4, robosuite 1.4.1, **mujoco 2.3.0**, python 3.10.

---

## GPU 지정 (모델을 올리는 모든 스크립트 공통)

GPU 사용: **01 · 02 · 04 · 06 · 07 · 08 · 09 · 10** 와 `setup/04`. **03 · 05 · `10 --dry-run` 은 GPU 불필요.**

| 인자 | 예시 | 설명 |
|---|---|---|
| `--gpu N` | `--gpu 4` | 가장 간단 |
| `--gpus` | `--gpus 0,6` | 여러 GPU 에 층 분할 로드 (**실서버 미검증**) |
| `--device` | `--device cuda:4` · `--device cpu` | 문자열 지정 |
| `--model` / `--unnorm-key` | `openvla/openvla-7b-finetuned-libero-object` / `libero_object` | 체크포인트 교체 |
| `--tag` | `--tag chk_gpu0` | 출력 하위 폴더 (02·04·06·09·10) |
| `--egl-device` | `--egl-device 0` | 렌더링 번호 (CUDA 번호와 **다른 체계**, 보통 0 하나) |

로드 직후 출력에서 **요청한 번호와 GPU 이름**을 확인하세요 (09 실제 로그):

```
[device] model=cuda:4  render(EGL)=egl:0  (EGL 디바이스가 1개뿐 → 선택 불가)
[gpu   ] cuda:4 NVIDIA RTX A4000  이 프로세스 15.1GB  |  남은 여유 1.6GB / 전체 16.9GB
```

- **궤적은 GPU 종류에 따라 달라집니다** (bf16 반올림·SM 수에 따른 커널 차이). 같은 GPU 재실행은 완전 재현.
  → 서로 비교할 실험(예: 04 ↔ 06, 02 ↔ 09)은 **같은 종류 GPU** 로. 실측: GPU 0(A4000) 13/15 vs GPU 5(A6000) 10/15.
- 실측 확인된 매핑: cuda:0 = A4000, cuda:4 = A4000, cuda:5 = A6000. 나머지는 `nvidia-smi --query-gpu=index,name --format=csv`.
- A4000(16GB) 단독 로드 가능하나 여유 약 1.6GB → 긴 롤아웃에서 OOM 위험 (경고가 뜸).
- `CUDA_VISIBLE_DEVICES` 와 `--gpu` 를 섞지 마세요 (번호 재매핑, 코드가 감지해 에러).

---

## 01 ★ — 구조 검증 (30분, 이 프로젝트의 관문)

```bash
python scripts/01_smoke_forward.py --gpu 4                        # 합성 이미지
python scripts/01_smoke_forward.py --gpu 4 --libero --suite spatial --task-id 0
```

| 단계 | 통과 기준 | 실패 시 |
|---|---|---|
| `[1]` 모델 로딩 | layers 32, heads 32 | `find_language_model` 확인 |
| `[2]` 프롬프트 | instruction 문자 구간이 지시문을 정확히 가리킴 | `token_index.PROMPT_TEMPLATE` |
| `[3]` visual span | `[1, 257)` → 256 | 경고를 무시하지 말고 구조 확인 |
| `[4]` 구간 분해 | **★ 육안: L 이 지시문 내용어에만**, 마지막 텍스트 토큰이 `▁`(29871), A·Q 범위 출력 | `In`·`What` 에 L → 오프셋 매핑 실패 |
| `[5]` action token | 7개 id, 범위 검증 통과 | `action_bin_token_ids` ±1 |
| `[6]` 캡처 | attn `(32, 32, 7, T)`, **Q 행 argmax == 생성 토큰 7/7**, 행 합 ≈ 1 (bf16 0.997–1.003) | 7/7 이 아니면 Q 위치가 틀린 것 |
| `[7]` 지표 | R 3종이 [0,1], NaN 없음 | mass_sink ≥ 0.9 면 sink 처리 재검토 |

**Gate**: 위 전부 + `--libero` 실행 시 `outputs/01_smoke_forward/smoke_libero_view.png` 에서 로봇 팔이 똑바로 서 있음.

`[4]` 실측 예 (spatial task 0, 시퀀스 298):

```
  [   0]        S  '<s>'
  [   1..256]   V  (visual ×256)
  [ 257..265]   T  '▁In' ':' '▁What' … '▁to'
  [ 266]        L  '▁pick'          ← 여기부터
  …
  [ 285]        L  '▁plate'         ← 여기까지 L (20개)
  [ 286..289]   T  '?' … '▁Out' ':'
  [ 290]        T  '▁'  (29871)     ← Q 시작
  A = 291–297,  Q = 290–296
```

---

## 08 ★ — 정책 성공 1회 확인

```bash
python scripts/08_policy_sanity.py --gpu 4 --tasks 0 --max-steps 220 --gripper-fix both
```

**Gate**: `current + grip` 조합이 성공. 실패하면 분석을 돌리지 마세요 — 성공률은 전처리·후처리 오류를 잡는 **유일한 경험적 검증**입니다
(그리퍼 규약 사례는 아래 트러블슈팅).

---

## 02 — 롤아웃 + R + knockout KL

```bash
python scripts/02_run_analysis.py --gpu 4 --suite spatial --tasks 0 --episodes 1 --max-steps 20 --stride 5   # 시간 측정
python scripts/02_run_analysis.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 --episodes 3 --max-steps 220 --stride 5
python scripts/02_run_analysis.py --gpu 4 --suite spatial --tasks 7 9 --episodes 3 --max-steps 220 --stride 5 --tag lang_required
```

| 인자 | 의미 | 권장 |
|---|---|---|
| `--tasks` | task id | 0–4 + **언어 필수 쌍 7·9** (원인 판별용) |
| `--episodes` | task 당 초기 상태 수 | 3 이상 |
| `--max-steps` | 에피소드 최대 길이 | **220** (OpenVLA LIBERO-spatial 평가 기준) |
| `--stride` | 몇 스텝마다 분석 | 5 |
| `--no-intervene` | 개입 생략 | 쓰지 마세요 |

- 분석 1스텝 = generate 1 + teacher-forced forward 4 (base + knockout 3) + 정책 행동 1. **소요 시간은 직접 재세요.**
- 끝에 **성공률**이 출력됩니다. 실측 13/15(A4000), 10/15(A6000).

**Gate**: CSV 생성, `R_*` NaN 없음, `KL_*` 유한, `KL_lang_knockout > KL_control_knockout` 가 거의 모든 행.
⚠ knockout 은 **직접 효과**만 잽니다 (`docs/05_interventions.md` [L3], 09 와 비교).

---

## 03 — H2 상관 (GPU 불필요)

```bash
python scripts/03_correlation.py      # 기본 입력 outputs/02_run_analysis/analysis_spatial.csv
python scripts/03_correlation.py --csv outputs/02_run_analysis/lang_required/analysis_spatial_lang_required.csv \
                                 --out outputs/03_correlation/correlation_lang_required.csv
```

- `|ρ| < 0.3` 이면 attention 비율은 인과를 거의 예측 못 함, `> 0.6` 이면 쓸 만한 대리지표.
- ⚠ 행은 에피소드 안에서 자기상관 → 출력 p 값은 과대. **에피소드 단위 bootstrap** 으로 다시 계산해야 함.
- ⚠ 마지막의 `lang_vs_control` t-검정은 분모 ≈ 0 문제와 자기상관 때문에 **쓰지 마세요** (교체 예정).
- ⚠ 비율 대 비율(R ↔ causal_ratio) 결론은 개입 방식에 따라 부호가 바뀜(09). **R ↔ KL_L** 을 함께 보고.

---

## 04 — 지시 조건 (H1 일부)

```bash
python scripts/04_counterfactual.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 7 9 --ambiguous-tasks 7 9
```

| 조건 | 지시문 | 묻는 것 |
|---|---|---|
| `valid` | 원문 | 기준 |
| `paraphrase` | "pick up" → "grasp" 등 | 의미가 같으면 행동도 같아야 함 |
| `swapped` | 같은 suite 의 다른 task 지시 | 바뀌었는데 행동이 그대로면 언어 무시 |
| `empty` | 빈 지시 | LIBERO-Plus 재현 (OOD, R 은 NaN) |

- 모든 조건이 valid 의 action 토큰으로 teacher-forcing (`--prefix-mode shared`) → 02 와 같은 규약.
- **`--ambiguous-tasks 7 9`**: spatial 에서 배치가 같고 목표만 반대인 쌍 — 이미지로 과제를 구분할 수 없어 **언어 필수**.
  예전 문서의 "object suite 에서 손으로 고르기"는 이것으로 대체.
- ⚠ swapped 는 장면마다 성격이 다름 (task 0: 방해 그릇을 가리키는 실행 가능한 명령 / task 1–4: 그릇이 없는 위치). 해석 시 구분.

**실측 (task 0–4, A4000)**: swapped KL 4.97 vs paraphrase 0.29 (30/30 장면), empty 8.31, **R_norm 변화 +0.004** —
행동은 지시 내용에 반응하는데 비율은 불변.

---

## 05 — 그림 (GPU 불필요)

```bash
python scripts/05_plots.py --suite spatial
```

| 파일 | 내용 |
|---|---|
| `fig_layerwise_*.png` | 층별 R 3종 + uniform baseline |
| `fig_h2_*.png` | R_norm vs causal_ratio 산점도 (knockout 기준) |
| `fig_cf_ratio_*.png` / `fig_cf_kl_*.png` | 04 조건별 R_norm / KL |

`--tag` 를 받지 않습니다 (기본 출력만 그림).

---

## 06 — 이미지 조건

```bash
python scripts/06_image_ablation.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 --episodes 2   # 04 와 같은 GPU 종류
```

- 조건: `valid` / `other_scene`(다음 task 의 같은 ep·t 장면) / `hflip` / `patch_shuffle`(32×32) / `noise`. 지시문은 항상 원문.
- 진행: 롤아웃으로 `--steps`(10 25 40) 시점 이미지 수집 → 조건별 분석. 예시 PNG(`examples/`)로 변형이 의도대로인지 육안 확인.
- 실측: other_scene 11.69, hflip 5.69, patch_shuffle 9.61, noise 10.13 / R_raw 0.39 → 0.475(patch_shuffle), 0.525(noise).
- ⚠ R_raw 상승은 Text 질량 증가인지 Vision 질량 감소인지 **구분 불가**(절대 질량 미기록). 보상 가설(이미지가 망가지면 KL_L 이 커지는가)은 보완 후 검정.
- ⚠ other_scene 은 로봇 자세도 함께 바뀜.

---

## 07 — 모델 입력 확인 (필요할 때만)

```bash
python scripts/07_dump_model_input.py --gpu 4 --libero --suite spatial --task-id 0
```

확인: `input_00`(processor 이전)과 `input_01/02`(백본별)의 구도 동일, 224×224 종횡비 유지, 색 치우침 없음, 채널 6(DINOv2 + SigLIP).
**좌우 반전은 못 잡습니다** → 성공률로 판단.

---

## 09 — 0층 평균 치환(총 효과) vs knockout(직접 효과)

```bash
python scripts/09_embedding_substitution.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 --episodes 3
#   --no-rescaled / --no-segments 로 forward 절약, --stride 20 으로 프레임 축소
```

- 한 프레임당 forward 최대 12회 (02 한 행의 약 3배). 약 185 프레임.
- **Gate (스크립트가 자동 판정)**: G1 정렬(매 프레임, 실패 시 중단) · G2 항등 치환 KL ≈ 0(실패 시 중단) · G3 재실행 KL(기록) · G4 L 치환 > 대조군 비율(50% 미만 경고).
- 해석 규칙(결과 전에 고정): `docs/05_interventions.md` [L2].
- 실측: 치환/knockout 비 L 1.17 [1.09, 1.23], **V 1.58** [1.46, 1.81] → knockout 은 이미지 의존을 과소평가.
  R_raw ↔ causal_ratio 상관이 knockout +0.35 / 치환 −0.17 로 **부호 반전**.

---

## 10 — ICBench 방식 모순 지시 롤아웃 (1단계: 편향의 존재)

```bash
python scripts/10_icbench_rollout.py --dry-run --tasks 0 1 2 3 4 7 9            # 지시문 표만 (GPU 불필요) → 검토
python scripts/10_icbench_rollout.py --gpu 4 --tasks 0 7 9 --inits 10            # 파일럿: 3 × 7변형 × 10 = 210 롤아웃
python scripts/10_icbench_rollout.py --gpu 4 --tasks 0 1 2 3 4 7 9 --inits 20    # 본 실행 (파일럿 후 결정)
```

- 변형: normal / V1 blue bowl / V2 green plate / V3 둘 다 / V4 under the plate / V5 그릇 없는 위치 / **R 방해 그릇 위치(실행 가능한 다른 명령)**.
- 채점은 전부 원래 bddl 성공 판정 → 변형에서 성공 = **가짜 성공**. 결과 범주 target / distractor / wrong_lift / abstain, 갈라진 스텝 `t_star`.
- task 하나 끝날 때마다 CSV·npz 저장 (중간에 죽어도 남음).
- **Gate**: 정상 지시로 성공한 초기 상태에서 **실제 실패(target 이외)가 10개 이상** — 미달이면 11 의 예측 비교는 표본 부족 경고가 뜹니다.
- 볼 것: R 에서 `distractor`(지시를 따름) 비율. **task 7·9 에서 따르는지**가 원인 판별(데이터 지름길 vs 구조)의 첫 데이터.

## 11 — (예정) 가짜 성공 예측

10 이 저장한 행동을 시뮬레이터에서 재생해 프레임 복원(MuJoCo 결정성 → GPU 무관) → 같은 프레임에 정상·모순 지시 →
R·KL·바뀐 단어 토큰 attention 으로 가짜 성공 vs 실제 실패 예측 (AUC, 에피소드 bootstrap). 재생이 원 궤적과 일치하는지부터 검사.

---

## 진행 순서 (연구 단계와 대응)

| 순서 | 할 일 | 단계 |
|---|---|---|
| 1 | pytest → setup → **01** → **08** | 전제 |
| 2 | **10 파일럿** (`--tasks 0 7 9`) | 1. 존재 |
| 3 | 02·09 를 `--tasks 7 9` 로 (task 0–4 결과와 비교) | 3. 원인 (판별) |
| 4 | 06 보완(절대 질량 + KL) → 보상 가설 | 3. 원인 |
| 5 | 11 구현·실행 | 2. 측정 |
| 6 | 층·단어 단위 분석 → 추론 시 개입 → 10 재측정 | 3·4 |

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
| `action 위치가 시퀀스 길이를 벗어납니다` | visual 삽입 위치 가정 오류 | 01 의 `[3]` 재확인 |
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
| 렌더링 중 `get_joint_qpos_addr` 에서 메시지 없는 `AssertionError` | mujoco 3.x 가 robosuite 1.4.1 과 비호환 (enum 비교 방식 변경) | `mujoco==2.3.0` 고정 (`constraints.txt`) |
| `The size of tensor a (291) must match ... (290)` | `predict_action` 이 입력 끝에 토큰 29871 을 덧붙이는데 `attention_mask` 는 붙이기 전 길이 | `attention_mask` 를 빼고 호출 (`pipeline.policy_action`) |
| **LIBERO 성공률 0%** (팔은 움직이는데 못 집음) | **그리퍼 규약 불일치.** 아래 사례 참고 | `pipeline.libero_env_action` 으로 변환 (`policy_action` 기본 적용). 확인: `scripts/08_policy_sanity.py` |
| 02 성공률이 이전 실행과 다름 (예: 13/15 → 10/15) | **GPU 종류가 다르면 궤적이 달라짐** (bf16 반올림·SM 수에 따른 cuBLAS 커널) | 비교할 실험은 같은 종류 GPU 로. 같은 GPU 재실행은 완전 재현됨 |
| `AttributeError: ... append_action_prefix` 등 | 스크립트만 새 버전, `vlamod/` 는 옛 버전 (부분 동기화) | `vlamod/` 전체를 함께 동기화 |
| 01 종료 시 `EGLError: EGL_NOT_INITIALIZED (eglDestroyContext)` | env 를 닫지 않고 종료 | `task.env.close()` (01 에 반영됨, 결과에는 무해) |
| 04 `언어 구간 L 이 비었습니다` (empty 조건) | 빈 지시문은 L 이 비는 게 정상 | `allow_empty_language` (반영됨, empty 의 R 은 NaN) |
| 09 `[G1] 텍스트 위치 ... 0층 입력이 임베딩과 다릅니다` | visual 삽입 위치 가정 위반 | 01 의 `[3]` visual span 재확인 |
| 09 `[G2] 원래 임베딩으로 '치환' 했는데 출력이 바뀌었습니다` | hook 구현 / transformers 버전 | 4.40.1 인지, `LlamaDecoderLayer` 가 hidden_states 를 위치 인자로 받는지 확인 |
| 10 `task N 지시문 ... 이 장면 표 ... 와 다릅니다` | LIBERO 버전의 지시문이 `icbench.SPATIAL_SCENES` 와 다름 | bddl 과 표를 대조해 수정 |

### 사례: LIBERO 성공률 0% — 그리퍼 규약 불일치 (2026-09-29 해결)

**증상.** 02 본 실행(5 task × 3 ep)과 `max_steps=400` 재시도 모두 성공률 0%.
팔은 물체 쪽으로 움직이지만 집지 못함. 에러 메시지는 없음.

**배제한 원인.** `unnorm_key` — `norm_stats` 키가 `libero_spatial` 하나뿐이라 맞음.
스텝 수 — 400 스텝에서도 0%.

**원인.** OpenVLA 와 LIBERO 의 그리퍼 규약이 반대입니다.

| | 범위 | 열기 | 닫기 |
|---|---|---|---|
| OpenVLA `predict_action` 출력 | [0, 1] | ≈ 1 | ≈ 0 |
| LIBERO `env.step()` 입력 | [−1, +1] | −1 | +1 |

변환 없이 넣으면 모델이 '닫기(≈0)'를 원할 때 env 는 ≈0 = **무동작**, '열기(≈1)'를 원할 때 **닫기**가 됩니다.
OpenVLA 공식 평가 코드(`run_libero_eval.py`)는 `normalize_gripper_action(binarize=True)` → `invert_gripper_action` 두 단계를 거치는데, 우리 코드에 이게 빠져 있었습니다.

**조치.** `vlamod/pipeline.py`

```python
def libero_env_action(action):
    a[..., -1] = 2.0 * a[..., -1] - 1.0   # [0,1] → [-1,+1]
    a[..., -1] = np.sign(a[..., -1])      # binarize
    a[..., -1] *= -1.0                    # invert
```

`policy_action(..., env_convention=True)` 가 기본으로 적용 → `02_run_analysis`, `04_counterfactual` 모두 반영.
attention 캡처(teacher-forced)는 이 변환과 무관 — **바뀌는 것은 궤적(=분석 대상 장면)뿐**입니다.

**검증.** `python scripts/08_policy_sanity.py --gpu 5 --tasks 0 --max-steps 220 --variants current rot180_crop --gripper-fix both`

| 조합 | success | steps | 닫기명령 | EE 이동 | 원시 그리퍼 ≈0 / ≈1 / 음수 |
|---|---|---|---|---|---|
| current + raw | ✗ | 220 | 17 | 0.539 | 92% / 8% / 0% |
| current + **grip** | **✓** | **77** | 40 | 0.736 | 52% / 48% / 0% |
| rot180_crop + raw | ✗ | 220 | 43 | 1.122 | 80% / 20% / 0% |
| rot180_crop + **grip** | **✓** | **95** | 51 | 0.773 | 54% / 46% / 0% |

- 원시 그리퍼가 0 과 1 에만 몰리고 음수 0% → [0,1] 규약 확정.
- 변환한 쪽만 성공 → 원인 확정.
- 이미지 전처리: `current`(180° 회전, crop 없음)와 `rot180_crop` 모두 성공. **task 1개·에피소드 1개라 둘의 우열은 판단 불가.** 기존 분석과 같은 `current` 를 유지합니다.

**기존 데이터에 미치는 영향.**
- `t = 0` 행: 정책 행동이 한 번도 적용되기 전 장면이므로 **그대로 유효**.
- `t > 0` 행: 고장 난 그리퍼가 만든 비정상 궤적 위의 장면 → **폐기 후 재수집**. 시간 추세·그룹 분석은 재수집 데이터로만 주장합니다.

**교훈.** 성공률은 전처리·후처리 오류를 잡는 유일한 경험적 검증입니다. 분석 파이프라인을 돌리기 전에 `08_policy_sanity.py` 로 성공 1회를 먼저 확인하세요.

---


## 결과 해석의 원칙

- **음성 결과도 결과입니다.** 편향이 없으면 "왜 OpenVLA 는 따르고 OFT·π0 는 무시하는가"로, attention 지표가 예측에 실패하면 "IVAR 류로는 진단 불가"로.
- **결과를 보고 지표를 고르지 마세요.** 개입 방식별(knockout·치환) 결과를 모두 보고하고, 해석 규칙은 실행 전에 `docs/05_interventions.md` 에 적습니다.
- **독립 단위는 에피소드입니다.** 행 단위 p 값은 과대 — 에피소드 bootstrap, 에피소드 내 상관을 함께.
- 가장 위험한 것은 결과가 아니라 **"그래서 뭐(so what)"** 에 답이 없는 것. `docs/00_research_plan.md` §7.
