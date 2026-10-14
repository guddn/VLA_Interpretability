# 연구 계획 — VLA 의 언어 무시(Vision 편향): 원인 진단

작성 2026-08-24 · **전면 개정 2026-10-14** (실험 결과 반영, 스크립트 번호 기준)
모델: OpenVLA-7B (`openvla/openvla-7b-finetuned-libero-spatial`) · 환경: LIBERO-Spatial

> 실험은 **스크립트 번호(01–11)** 로만 부릅니다. 예전 문서의 "Stage / Phase" 표기는 쓰지 않습니다.
> 결과 수치의 상세와 출처는 `docs/05_interventions.md`, `docs/06_metrics.md` 와 프로젝트 현황 문서에 있습니다.

---

## 0. 요약

**궁극 목표.** 행동 수준에서 보고된 **Vision 편향(언어 무시, linguistic blindness)** 을 모델 내부 동작으로 직접 확인하고,
그 **원인과 분포**(어느 층·어느 단어·어떤 조건)를 찾은 뒤, 원인에 맞는 개선 방안을 찾는다.

**연구 질문 (한 문장).**

> **VLA 의 언어 무시는 구조의 결함인가, 데이터 지름길의 결과인가 — 그리고 이를 행동 실험 없이 내부 지표로 진단할 수 있는가.**

**출발점에서 알아야 할 것 세 가지**

1. **행동 수준 증거는 이미 있습니다.** LIBERO-Plus(2510.13626), Stable Language Guidance(2601.04052), IGAR/ICBench(2603.06001).
   단, 모델마다 결과가 다릅니다(OpenVLA-OFT 는 빈 지시에도 버팀, π0 는 무너짐). **"VLA 는 언어를 무시한다"는 단일 명제가 아닙니다.**
2. **attention 비율 지표(IVAR)와 처방(IGAR)도 이미 있습니다.** 그런데 IGAR 는 원인을 attention sink(구조)로 **가정**하고
   바로 처방으로 갔습니다. **원인이 구조인지 데이터인지 판별한 연구는 조사 범위에서 찾지 못했습니다** → 차별점.
3. **IGAR 는 OpenVLA-7B 를 평가하지 않았습니다**(π0 / π0.5 / OpenVLA-OFT). 우리 수치 자체가 새 데이터입니다.

**지금까지 우리 데이터가 말하는 것 (잠정)**

- OpenVLA-7B 는 지시문이 **있어야** 작동한다 — 지시문을 지우면 7 DoF 중 평균 5개가 바뀜 (09, 02)
- 그러나 **무엇을 할지는 지시문 내용보다 이미지 내용으로 더 많이 정한다** — 장면 교체 KL 11.7 vs 지시문 교체 4.97 (06 vs 04, 28/30 장면)
- IGAR 식 "attention 이 Text 로 안 간다" 편향은 보이지 않는다 — 토큰당으로는 Text 를 훨씬 많이 봄 (R_norm 0.85)
- **attention 비율은 '얼마나 쓰는가'의 창이 아니다** — 지시 내용이 바뀌어도 비율 불변(04), 비율 대 인과 비율 상관이 개입 방식에 따라 부호 반전(09)

---

## 1. 문제의식 — 원래 직관과 정정

### 1.1 원래 직관

> "CLIP 선행연구에서 이미지와 language 의 사용 비율에 편향이 있었던 것 같다. VLA 에서도 그럴 것이다."

CLIP/VLM 계열에서 말하는 "모달리티 편향"은 세 가지 다른 현상이고, VLA 로 옮길 때 어느 것인지 명확히 해야 합니다.

| 현상 | 내용 | VLA 로의 이식 |
|---|---|---|
| **(A) Modality gap** (Liang+ NeurIPS 2022) | 이미지·텍스트 임베딩이 공유 공간에서 분리된 두 원뿔에 몰림 | **어려움.** 표현 기하학 얘기지 사용 비율 얘기가 아님 |
| **(B) Unimodal prior** (VQA-CP, Agrawal+ CVPR 2018) | 이미지를 안 보고 질문 통계로 답함 → 언어 편중 | **부호가 반대.** VLA 는 장면이 행동을 거의 결정 → 비전 편중 |
| **(C) 모달리티 기여도 측정** (MM-SHAP, ACL 2023) | 성능과 무관하게 각 모달리티의 기여분 측정 | **가장 직접적.** 이 연구가 하는 것의 원형 |

→ "CLIP 에서 비율 편향이 있었다"를 근거로 쓰려면 **(C)** 를 인용해야 합니다.

### 1.2 "적절한 비율"이라는 표현의 약점과 해결

- **규범적 기준이 없습니다.** attention 60:40 이 적절한가? 답이 없습니다.
- **토큰 개수가 비대칭입니다.** V 256개 vs L 15–21개(spatial 실측). 균등 분포만 가정해도 R_raw ≈ 0.07 — "비전이 93%"는 산수입니다.
- **LIBERO 의 장면이 과제를 결정합니다.** LIBERO-Spatial 은 장면마다 검은 그릇이 2개라 *어느 그릇인지*는 지시문이 정하지만,
  **task 마다 그릇 배치가 달라 배치만 보고도 과제를 맞힐 수 있습니다**(bddl 확인). 여기서 낮은 언어 의존을 보고
  "모델의 결함"이라고 하면 "데이터가 그런 것"이라고 반박됩니다. → **이 confound 가 원인 가설 A(데이터 지름길) 자체입니다.**

**해결.** 절대 비율을 버리고 **조건 간 비교**로 갑니다. 같은 모델에서 언어가 **필요한 조건 vs 필요 없는 조건**을 만들고
지표가 움직이는지 봅니다. 비교 대상이 자기 자신이라 규범 기준이 필요 없습니다.
LIBERO-Spatial 에는 그런 조건이 이미 있습니다 — **task 7 / 9** (§4).

---

## 2. 선행연구 지도

> 검증 표기: ✅ 원문 확인 / ⚠️ 요약만 확인, 정독 필요

### 2.1 "VLA 가 언어를 안 쓴다" — 행동 수준

| 논문 | 핵심 | 확인 |
|---|---|---|
| **LIBERO-Plus** (2510.13626) | 빈 지시에도 OpenVLA-OFT 성능 거의 유지, 목표 물체명을 바꿔도 원래 물체를 집음. "visual pattern matcher" | ✅ |
| **Stable Language Guidance** (2601.04052) | "modality collapse". π0: 94.15% → 빈 지시 25.20% → 80% 마스킹 7.80% | ✅ |
| **Restoring Linguistic Grounding / IGAR** (2603.06001) | ICBench(모순 지시 V1–V4)에서 SR 90%↑ = 가짜 성공. LGS = SR(정상) − SR(모순). OFT spatial LGS −0.2~5.2. **지시문 목록·코드 미공개**(2026-10-14 확인) | ✅ 전문 |
| RoboSemanticBench (2606.02277) | 의미 grounding 진단 벤치마크 | ⚠️ |

### 2.2 내부 지표·개입

| 논문 | 핵심 | 확인 |
|---|---|---|
| **IGAR** (2603.06001) | IVAR = Text attention / (Text + Vision). 원인을 **sink** 로 지목 → hidden-state spike 로 sink 탐지, text sink 를 0.6배로 줄여 재분배. 첫 16층에 개입 | ✅ |
| **VLA-Trace** (2605.30117) | attention knockout 으로 시각/텍스트 경로 분리 + CKA. OpenVLA 는 제어 정보가 양쪽에 분산 | ⚠️ |
| **Embodied Interpretability** (2605.00321) | ISS = distribution-matched baseline 치환 + KL. "attention 은 task 무관 영역에 뜬다"며 attention 계열 비판. **baseline 정의는 원문 미확인** | ⚠️ |
| **MM-SHAP** (2212.08158) | Shapley 기반 모달리티 기여도. VLA 이식 미발견 | ✅ |

### 2.3 방법론 (인용 안 하면 공격당함)

- Jain & Wallace (NAACL 2019) **Attention is not Explanation** / Wiegreffe & Pinter (EMNLP 2019) **…not not Explanation** — 둘 다 인용
- Kobayashi+ (EMNLP 2020) — `α·‖v‖` 가중 → 우리 `R_vnorm`
- Adebayo+ (NeurIPS 2018) Sanity Checks — 파라미터 randomization 대조군
- **Zhang & Nanda (ICLR 2024, 2309.16042)**, **Heimersheim & Nanda (2404.15255)** — activation patching 의 baseline 선택(Gaussian noise 불안정 등). 09 의 기준값 선택 근거
- Wang+ 2022 (IOI, mean ablation), Chan+ 2022 (causal scrubbing, resample ablation)

### 2.4 처방 쪽 (원인이 데이터일 때)

- **CAST** (2508.13446) — 반사실 라벨로 지시 따르기 개선. 원인이 데이터 지름길이면 처방은 이쪽 계열
- 인접: VLA-Pruner (2511.16449), Not All Redundant Tokens (2608.04483) — "attention 을 많이 먹는다 ≠ 정보를 많이 쓴다"

---

## 3. 연구 질문과 가설

| 질문 | 선행연구 | 상태 |
|---|---|---|
| VLA 가 언어를 무시하는가 (행동) | 답 있음 (모델마다 다름) | **OpenVLA-7B 미확인 → 10** |
| attention 비율은 얼마인가 | IVAR | 우리 측정: R_raw 0.42, R_norm 0.85 |
| 비율을 고치면 나아지는가 | IGAR | — |
| **비율이 언어 필요도에 따라 조절되는가 (H1)** | 비어 있음 | 04: 지시 내용 변화에 불변. **언어 필수 조건(7/9) 미측정** |
| **내부 지표가 실제 기여·행동을 예측하는가 (H2)** | 비어 있음 | 09: 비율 대 비율은 개입 의존(부호 반전). 행동 예측은 **11** |
| **언어 무시의 원인은 구조인가 데이터인가 (H3)** | 비어 있음 (IGAR 는 구조 가정) | **7/9 + 10 의 R 조건으로 판별 예정** |

**H1 (비적응성).** attention 분배는 언어가 실제로 필요한 상황인지와 무관하게 거의 일정하다.
- 근거(부분): 04 에서 지시 내용을 바꿔도 R_norm +0.004. 단 "필요도"를 바꾼 실험은 아직 아님
- 검정: task 7/9(언어 필수) vs task 0–4 에서 R·KL_L 비교

**H2 (지표 타당성).** attention 비율(IVAR 류)은 인과 기여나 행동 결과를 잘 예측하지 못한다.
- 근거: 09 — R_raw ↔ causal_ratio 가 knockout +0.35 / 치환 −0.17 (부호 반전). 두 방법 모두에서 버티는 건 R_raw ↔ KL_L(+0.30~+0.42)뿐
- 다음 검정: **가짜 성공 vs 실제 실패를 예측하는가** (11, AUC)

**H3 (원인).** OpenVLA 의 내용 수준 Vision 편향은 구조 결함이 아니라 **배치 → 과제 1:1 이라는 데이터 지름길**의 결과다.
- 예측: 지름길이 없는 task 7/9 에서는 지시를 따르고 언어 의존(KL_L, 위치구 attention)이 커진다
- 반증: 7/9 에서도 지시를 무시하면 구조적 원인(sink, 특정 층의 정보 흐름)을 찾아야 함

**보조 가설 — 보상(compensation).** 이미지를 망가뜨리면(06 noise·patch_shuffle) **언어의 인과 효과(KL_L)가 커진다**.
- 06 에서 R_raw 가 0.39 → 0.53 으로 올랐지만, 이는 Vision 질량 감소일 수도 있고 attention ≠ 사용이므로 인과 지표로 검정해야 함
- 참이면: 언어를 쓸 능력은 있고 평소 편향은 '선호' → "Text 를 무시한다"에 대한 강한 반론

---

## 4. 로드맵 — 단계마다 넘어가는 조건(관문)

| 단계 | 질문 | 스크립트 | 관문 |
|---|---|---|---|
| **1. 존재** | OpenVLA-7B 에 행동 수준 편향이 있는가 | **10** | 가짜 성공과 실제 실패가 **둘 다 충분히** 나옴 |
| **2. 측정** | 어떤 내부 지표가 "가짜 성공으로 끝날지"를 예측하는가 | **11** (예정) ← 02·09 의 지표 | 예측력 있는 지표 ≥ 1 (에피소드 bootstrap) |
| **3. 원인** | 구조인가 데이터인가 / 어느 층·단어인가 | 02·09 (task 7·9), 06 보완, 층·단어 단위 | 원인 후보가 하나로 좁혀짐 |
| **4. 개선** | 원인에 맞는 처방이 편향을 줄이는가 | 추론 시 개입 + 10 재측정 | LGS 개선, 정상 지시 성공률 유지 |

**관문에서 막힐 때**
- 1단계에서 편향이 안 나오면(R 조건을 대부분 따르면) → "OpenVLA 는 따르는데 OFT·π0 는 왜 무시하는가"로 전환
- 2단계에서 attention 지표가 예측에 실패해도 결과 → "IVAR 류로는 편향을 진단할 수 없다"
- 3단계에서 원인이 데이터면 → 구조 개선이 아니라 데이터·학습 처방 방향 제시 + 진단 지표로 마무리

### 판별 실험 — task 7 / 9

LIBERO-Spatial bddl 의 `(:init)` 확인 결과, 10개 task 는 같은 물체 7종을 서로 다른 배치로 놓고, **배치가 과제를 1:1 로 결정**합니다.
예외가 한 쌍 있습니다.

| task | 목표 그릇 | 방해 그릇 | 나머지 물체 |
|---|---|---|---|
| 7 | 스토브 위 | 캐비닛 위 | 동일 |
| 9 | 캐비닛 위 | 스토브 위 | 동일 |

이미지로는 과제를 구분할 수 없는 **유일한 쌍** → 언어 필수. task 7 장면 + task 9 지시문은 **학습 데이터에 실제로 있는 조합**이므로
"분포 밖이라 그렇다"·"best effort 였다" 반론이 통하지 않습니다.

| 7·9 결과 | 해석 | 개선 방향 |
|---|---|---|
| 지시를 따름 | 원인 = 데이터 지름길, 구조는 언어를 쓸 수 있음 | 데이터·학습 처방, 지름길 탐지 지표 |
| 지시를 무시 | 구조적 원인 | 원인 층·헤드를 표적으로 한 추론 시 개입 |

---

## 5. 스크립트 ↔ 질문

| # | 스크립트 | 답하는 질문 | 상태 |
|---|---|---|---|
| 01 | `01_smoke_forward.py` | 토큰 구간·Q 행·캡처가 맞는가 (모든 결과의 전제) | ✅ |
| 02 | `02_run_analysis.py` | 매 스텝 R 3종 + knockout(직접 효과) KL, 시간·그리퍼 단계 추세 | ✅ task 0–4 / 7·9 대기 |
| 03 | `03_correlation.py` | H2 (상관) | ✅ (t-검정 교체 필요) |
| 04 | `04_counterfactual.py` | 지시 내용을 바꾸면 비율·행동이 바뀌는가 (H1 일부) | ✅ → 10 으로 확장 |
| 05 | `05_plots.py` | 그림 | ✅ |
| 06 | `06_image_ablation.py` | 이미지를 바꾸면? / 보상 가설 | ✅ / 보상 검정은 보완 필요 |
| 07 | `07_dump_model_input.py` | 모델 입력 픽셀이 맞는가 | ✅ |
| 08 | `08_policy_sanity.py` | 정책이 성공하는가 (전처리·그리퍼) | ✅ |
| 09 | `09_embedding_substitution.py` | 총 효과 vs 직접 효과, H2 의 개입 의존성 | ✅ |
| 10 | `10_icbench_rollout.py` | **1단계: 행동 편향의 존재**, 7·9 판별 | ⬜ 파일럿 대기 |
| 11 | (예정) | **2단계: 가짜 성공 예측** — 행동 재생 → 같은 프레임에 정상·모순 지시 | ⬜ |

---

## 6. 지금까지의 증거로 본 "Vision 편향"

| 수준 | 근거 | 편향? |
|---|---|---|
| 존재 의존 (토큰을 없애면) | 09 치환: 지시문 KL 8.8 vs 이미지 11.3 | 약간 Vision 쪽, 둘 다 큼 |
| **내용 의존 (다른 것으로 바꾸면)** | 06 other_scene 11.7 vs 04 swapped 4.97 (28/30 장면, 중앙 2.4배) | **Vision 쪽** |
| attention 비율 | R_norm 0.85 | 편향 아님 |

- 06 other_scene 은 로봇 자세도 함께 바뀌므로 2.4배에 그 효과가 섞여 있음
- 04 swapped 는 task 0 = 실행 가능한 다른 명령, task 1–4 = 그릇 없는 위치로 섞여 있음
- 02 의 knockout 은 **직접 효과만** 잼(우회 경로가 V 쪽에 33개, L 쪽에 4개) → 이미지 의존 과소평가 (09 에서 확인)

---

## 7. 공격받을 지점과 방어

| 공격 | 방어 |
|---|---|
| "IVAR/IGAR 가 이미 했다" | 재현은 출발점. 기여는 **원인 판별(H3)** 과 **지표의 행동 예측력(H2)**. IGAR 는 OpenVLA-7B 미평가, 원인 가정만 함 |
| "attention 은 설명이 아니다" | attention 은 근거가 아니라 **검정 대상**. 주 증거는 인과(KL)와 행동. 실제로 attention 비율의 약점을 우리가 먼저 보임(04, 09) |
| "LIBERO 는 원래 언어가 필요 없다" | 맞음 — 그것이 가설 A. **task 7/9(언어 필수)** 로 직접 검정 |
| "V 256개라 비전이 이기는 게 당연" | R_norm·R_vnorm·uniform_baseline 함께 보고. 인과 지표가 주 증거 |
| "knockout 은 우회 경로가 열려 있다" | 맞음(직접 효과). 09 에서 치환(총 효과)과 같은 프레임에서 비교, 완전 knockout 추가 예정 |
| "평균 벡터 치환도 분포 밖" | 일부 맞음. 크기 보정(rescaled)·무작위 대조군으로 부분 방어, 한계로 명시 |
| "결과를 보고 지표를 골랐다" | 해석 규칙을 결과 전에 문서화(05_interventions), 개입 방식별 결과를 **모두** 보고 |
| "ICBench 수치와 다르다" | 원 지시문 미공개 → "ICBench 방식"으로 명시, 직접 비교하지 않음 |
| "V1(blue bowl)은 best effort 일 뿐" | 주 조건은 V4(공간 관계)와 R(실행 가능한 다른 명령) |
| "OpenVLA 하나로 일반화 못 한다" | 인정. 여력이 되면 OpenVLA-OFT 추가 (같은 Llama 백본, KL 정의 유지 가능 여부는 확인 필요) |
| "그래서 뭐(so what)" | "언어 무시가 데이터 탓이면 attention 을 고치는 처방(IGAR 류)은 엉뚱한 곳을 고치는 것" — 원인 판별이 처방 선택을 바꾼다 |

---

## 8. 개선 단계의 범위와 이후 확장

- **재학습 없는 추론 시 개입**까지가 현실적 범위. 7B 재학습은 현재 GPU(A4000 16GB, A6000 48GB)로 LoRA 도 부담이 클 가능성 (정확한 요구 사양 미확인)
- IGAR 와의 차별화: 찾아낸 위치(층·헤드·단어)에 **표적** 개입 / IGAR 의 sink 가정이 OpenVLA 에서 성립하지 않음을 보이기
- 원인이 데이터면 직접 구현 없이 **진단 지표 + 처방 방향**으로도 연구로 성립
- 이후: 층별 activation patching, SAE(2603.19183) 로 language-derived feature 분해, steering(2509.00328), MM-SHAP 의 VLA 이식

---

## 9. 신뢰도 표기

- ✅ 원문 확인: 2510.13626, 2601.04052, **2603.06001(전문)**, 2212.08158, Liang+ 2022
- ⚠️ 요약만: 2605.30117, 2605.00321(ISS baseline 정의 포함), 2606.02277, 2603.19183, 2508.13446
- ✅ 코드로 확인: visual token 256개(`[1, 257)`), 프롬프트 템플릿(L 태깅 육안 검증 + 정책 성공), action bin 범위, spatial bddl 장면 구성
- ⚠️ 미확인: 7B LoRA 메모리 요구량, OpenVLA-OFT 이식 난이도

## 참고문헌

- LIBERO-Plus — https://arxiv.org/abs/2510.13626
- Stable Language Guidance — https://arxiv.org/abs/2601.04052
- Restoring Linguistic Grounding (IVAR/IGAR/ICBench) — https://arxiv.org/abs/2603.06001 · https://ray-nh.github.io/igar/
- VLA-Trace — https://arxiv.org/abs/2605.30117
- Embodied Interpretability — https://arxiv.org/abs/2605.00321
- MM-SHAP — https://arxiv.org/abs/2212.08158
- Towards Best Practices of Activation Patching — https://arxiv.org/abs/2309.16042
- How to use and interpret activation patching — https://arxiv.org/abs/2404.15255
- CAST — https://arxiv.org/abs/2508.13446
- SAEs in VLA — https://arxiv.org/abs/2603.19183 · Steering VLA — https://arxiv.org/abs/2509.00328
- Sanity Checks for Saliency Maps — https://arxiv.org/abs/1810.03292
