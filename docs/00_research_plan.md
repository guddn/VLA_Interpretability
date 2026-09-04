# VLA Interpretability — 이미지/언어 토큰 기여 비율 검증 계획

작성일: 2026-08-24
목적: 선행연구 재현·검증 → 이후 독자 연구로 고도화
대상 질문: **"action token이 생성될 때 image token 정보와 language token 정보가 적절한 비율로 사용되는가?"**

---

## 0. 요약 (먼저 알아야 할 3가지)

1. **행동 수준에서는 이미 답이 나와 있습니다.** VLA는 언어를 거의 안 씁니다. LIBERO-Plus(2510.13626)는 OpenVLA-OFT에 **빈 지시(blank instruction)를 넣어도 성능이 거의 안 떨어지는데, 지시의 목표 물체를 바꾸면 원래 물체를 계속 집는다**는 것을 보였습니다. "visual pattern matcher"라는 표현까지 씁니다. 즉 "언어를 덜 쓴다"를 새로 발견하는 건 이미 늦었습니다.
2. **attention 비율 지표도 이미 있습니다.** `Restoring Linguistic Grounding in VLA`(2603.06001)가 **IVAR(Instruction Visual Attention Ratio)** = 텍스트 attention / (텍스트+비주얼 attention) 을 정의해 놓았습니다. 형우 님이 떠올린 지표와 사실상 같은 것입니다. **이걸 모르고 제안하면 리뷰에서 바로 맞습니다.**
3. **그래서 살아남는 질문은 "비율이 얼마인가"가 아니라 "비율이 상황에 맞게 조절되는가"입니다.** 이 각도가 아직 비어 있고, 형우 님의 mechanistic interpretability 배경과도 맞습니다. 아래 §3에서 이 재정의를 다룹니다.


---

## 1. 문제의식 정리 — 기존 직관과 정정

### 1.1 형우 님의 원래 직관

> "CLIP 선행연구에서 이미지와 language의 사용 비율에 편향이 있었던 것 같다. VLA에서도 그럴 것이다."

방향은 맞습니다. 다만 **CLIP에서 관찰된 '편향'은 세 가지 서로 다른 현상이고, VLA로 옮길 때 어느 것을 말하는지 명확히 해야 합니다.** 이걸 뭉뚱그리면 발표 때 첫 질문에서 무너집니다.

| CLIP/VLM 계열에서 말하는 "모달리티 편향" | 실제 내용 | VLA로의 이식 가능성 |
|---|---|---|
| **(A) Modality gap** (Liang et al., NeurIPS 2022, *Mind the Gap*) | 이미지 임베딩과 텍스트 임베딩이 공유 공간에서 **분리된 두 원뿔(cone)** 에 몰려 있음. 초기화 + contrastive loss의 기하학적 결과 | **직접 이식 어려움.** VLA는 contrastive가 아니라 next-token prediction. "두 임베딩 군집이 떨어져 있다"는 VLA의 action 생성과 직접 연결되지 않음 |
| **(B) Unimodal / language prior bias** (VQA-CP, Agrawal et al. CVPR 2018) | 모델이 이미지를 안 보고 질문의 통계만으로 답함 → **언어 편중** | **부호가 반대로 이식됨.** VLA는 오히려 **비전 편중**. VQA는 언어가 정답을 거의 결정했지만, VLA는 장면이 행동을 거의 결정 |
| **(C) 모달리티 기여도 측정 문제** (MM-SHAP, Parcalabescu & Frank, ACL 2023) | Shapley value로 **성능과 무관하게** 각 모달리티의 예측 기여분을 측정. "정확도가 높아도 한쪽 모달리티만 쓰고 있을 수 있다" | **가장 직접적으로 이식 가능.** 형우 님이 하려는 것이 바로 이것의 VLA판 |

**정정 필요 지점:** "CLIP에서 비율 편향이 있었다"를 근거로 쓰려면 **(C)** 를 인용해야 하고, (A) modality gap을 인용하면 "그건 표현 기하학 얘기지 사용 비율 얘기가 아니다"라고 반박당합니다.

### 1.2 "적절한 비율"이라는 표현의 최대 약점

**"적절한(appropriate)"의 기준이 무엇인가?** — 이것이 이 연구의 가장 큰 공격 지점입니다.

- attention 60:40이 적절한가? 90:10은 부적절한가? **규범적 기준(normative baseline)이 존재하지 않습니다.**
- 게다가 **토큰 개수가 애초에 비대칭**입니다. OpenVLA는 visual token 256개 vs language token 보통 5~15개. raw attention mass를 더해서 비교하면 **비전이 이기는 것이 수학적으로 거의 보장**됩니다. "vision이 90% 먹는다"는 발견이 아니라 산수입니다.
- LIBERO 같은 벤치마크는 **장면 자체가 task를 결정**합니다(한 장면에 목표 물체가 사실상 하나). 즉 **언어가 실제로 필요 없는 환경**입니다. 여기서 낮은 언어 의존을 관찰하고 "모델의 결함"이라고 주장하면, "그건 벤치마크가 그런 것이지 모델 탓이 아니다"로 반박됩니다. **이 confound가 이 연구의 1순위 위협입니다.**

**해결 방향:** 절대 비율을 버리고 **조건부 상대 비율**로 갑니다. 같은 모델·같은 장면에서 **언어가 반드시 필요한 조건과 필요 없는 조건**을 만들고, **비율이 조건에 따라 움직이는가**를 봅니다. 이러면 규범 기준이 필요 없습니다 — 비교 대상이 자기 자신이 되기 때문입니다.

---

## 2. 선행연구 지도 (이 질문에 직접 관련된 것만)

> 검증 표기: ✅ 원문 확인 / ⚠️ 검색 결과 요약만 확인, 원문 정독 필요 / ❓ 미확인

### 2.1 "VLA가 언어를 안 쓴다"는 행동 수준 증거

| 논문 | 핵심 | 확인 |
|---|---|---|
| **LIBERO-Plus: In-depth Robustness Analysis of VLA** (arXiv 2510.13626) | 7개 교란 축(물체 배치/카메라/로봇 초기상태/**언어**/조명/배경/센서노이즈). **빈 지시에도 OpenVLA-OFT 성능 거의 유지**(object suite). 목표 물체명을 바꿔도 원래 물체를 집음 → 성공률 0에 가깝게 붕괴. 결론: "visual pattern matcher" | ✅ |
| **Stable Language Guidance for VLA** (arXiv 2601.04052) | **"modality collapse"** — dense visual signal이 sparse linguistic signal을 압도. π₀는 원본 94.15% → 빈 지시 25.20% → 80% 마스킹 7.80%. 진단은 attention이 아니라 **언어 교란(destructive / obfuscated / OOD)** 기반 | ✅ |
| **RoboSemanticBench** (arXiv 2606.02277) | 의미 grounding 진단 전용 벤치마크 | ⚠️ |

> ⚠️ **주의:** 2510.13626과 2601.04052의 빈-지시 결과가 서로 다릅니다(OpenVLA-OFT는 거의 안 떨어짐 / π₀는 94→25로 붕괴). 모델·suite·프로토콜 차이입니다. **"VLA는 언어를 무시한다"를 단일 명제로 쓰면 안 됩니다.** 오히려 이 불일치 자체가 형우 님 연구의 진입점입니다 — *어떤 모델이, 어떤 조건에서, 얼마나* 가 아직 정리되지 않았습니다.

### 2.2 attention 비율 지표 (형우 님 아이디어의 직접 선행)

| 논문 | 핵심 | 확인 |
|---|---|---|
| **Restoring Linguistic Grounding in VLA via Train-Free Attention Recalibration** (arXiv 2603.06001) | **"linguistic blindness"** 진단. **IVAR = Σ_{j∈T} Ā(s,j) / Σ_{j∈V∪T} Ā(s,j)** — 정확히 "언어 attention 비율". 원인을 **attention sink**로 지목(action query가 시각적으로 salient한 토큰에 몰림). **LGS(Linguistic Grounding Score)** = 정상 지시 성공률 − 모순 지시 성공률. **ICBench**(LIBERO 기반, 모순 지시 4유형: 속성 치환/속성 증강/이중 교란/공간관계 치환). IGAR로 sink 토큰 0.6배 축소 후 재분배. π₀, π₀.₅, OpenVLA-OFT | ✅ **필독 1순위** |
| **VLA-Trace** (arXiv 2605.30117) | **attention knockout**으로 visual/textual 경로 분리 + CKA. π₀.₅는 불안정한 cross-modal fusion, OpenVLA는 visual·textual에 제어 정보 분산. 공통 결론: "시각 grounding은 좋고 fine-grained semantic following은 약하다" | ⚠️ |
| **Embodied Interpretability** (arXiv 2605.00321) | **ISS** = 입력 토큰을 distribution-matched baseline으로 치환 후 action 분포의 **KL divergence**. **"attention score는 주로 task-irrelevant 영역에 활성화된다"며 attention 계열을 명시 비판** | ⚠️ **반드시 읽고 방어 논리 준비** |
| **MM-SHAP** (arXiv 2212.08158, ACL 2023) | Shapley 기반 모달리티 기여도. **성능과 무관하게** 측정. 공식 구현 공개(Heidelberg-NLP/MM-SHAP) | ✅ **VLA 이식이 아직 안 보임 — 빈 곳** |

### 2.3 방법론 신뢰성 (인용 안 하면 공격당함)

- **Attention is not Explanation** (Jain & Wallace, NAACL 2019) — attention 가중치와 예측 기여가 어긋날 수 있음
- **Attention is not not Explanation** (Wiegreffe & Pinter, EMNLP 2019) — 위의 반론. 둘 다 인용해서 균형을 잡아야 함
- **Attention is not only a weight** (Kobayashi et al., EMNLP 2020) — attention weight 대신 **‖α·v‖ (value 벡터 norm 가중)** 을 써야 실제 기여에 가까움. **이 연구에서 반드시 채택할 보정**
- **Sanity Checks for Saliency Maps** (Adebayo et al., NeurIPS 2018) — 파라미터 randomization 대조군 필수
- **Attention rollout** (Abnar & Zuidema, ACL 2020) — 층을 가로지르는 누적 attention

### 2.4 인접 근거 (visual token 중복성)

- **VLA-Pruner** (2511.16449), **Not All Redundant Tokens Are Alike** (2608.04483) — visual token의 상당수가 지워도 되는 토큰. "vision이 attention을 많이 먹는다 ≠ vision 정보를 많이 쓴다"의 증거로 활용 가능

---

## 3. 재정의된 연구 질문 (차별화 지점)

선행연구가 답한 것과 안 답한 것을 나누면 이렇게 됩니다.

| 질문 | 상태 |
|---|---|
| VLA가 언어를 무시하는가? | **답 있음** (2510.13626, 2601.04052) |
| 언어 attention 비율은 얼마인가? | **답 있음** (IVAR, 2603.06001) |
| 비율을 고쳐서 성능을 올릴 수 있는가? | **답 있음** (IGAR, RSS+MCSI) |
| **비율이 입력 상황에 따라 조절되는가 (demand-adaptive인가)?** | **비어 있음** ← 여기 |
| **attention 비율이 실제 인과적 기여를 예측하는가?** | **비어 있음** ← 여기 |

### 핵심 가설 두 개

**H1 (비적응성 가설).** VLA의 언어/비전 attention 비율은 **언어가 실제로 필요한 상황인지와 거의 무관하게 일정하다.**
- 사람이라면 "빨간 컵 집어"는 컵이 둘일 때 언어를 더 봐야 합니다. 모델이 이 조절을 못 한다면, 이는 "언어를 덜 쓴다"보다 훨씬 구체적이고 mechanistic한 결함 진술입니다.
- 반증 가능: 비율이 조건에 따라 유의하게 움직이면 H1은 기각됩니다. **음성 결과도 논문이 됩니다.**

**H2 (지표 타당성 가설).** attention 비율(IVAR류)은 **인과적 개입 효과(ΔKL)를 잘 예측하지 못한다.**
- 2605.00321이 attention 계열을 비판한 것을 **정면으로 정량 검증**하는 실험입니다. 상관계수 하나로 끝납니다.
- H2가 참이면 IVAR 기반 후속 연구 전체에 대한 경고가 되고, 거짓이면 attention 지표를 정당화해 줍니다. **어느 쪽이든 결과가 나옵니다.**

---

## 4. 구현 절차

### Phase 0 — 환경 및 베이스라인 재현 (3~4일)

- 모델: **OpenVLA-7B (LIBERO fine-tuned 체크포인트)** 로 시작
  - 이유: autoregressive discrete action token → action logits가 명시적으로 존재하므로 KL 계산이 깔끔합니다. π₀/π₀.₅는 flow matching이라 gradient·확률 경로가 denoising step을 거쳐 난이도가 크게 올라갑니다.
  - 구조 확인 사항: DINOv2+SigLIP fused encoder → **visual token 256개**, Llama-2 7B backbone, 7-DoF를 각 256 bin으로 이산화해 **action token 7개**를 autoregressive 생성. (⚠️ 토큰 수는 체크포인트에서 직접 확인할 것)
- 시뮬레이터: **LIBERO** (Spatial / Object / Goal / Long). 성능 주장용이 아니라 **통제 환경**으로만 사용
- ⚠️ **실무 함정:** `flash_attention_2`로 로드하면 `output_attentions=True`가 무시됩니다. **`attn_implementation="eager"`** 로 로드해야 attention이 나옵니다. 속도는 느려지지만 분석용이므로 감수합니다.
- 산출물: 원 논문 성공률 재현 표 (±5%p 이내면 통과)

### Phase 1 — 토큰 인덱싱 (2일, 여기가 제일 지저분함)

prompt sequence에서 각 구간의 인덱스를 정확히 잘라야 이후 전부가 성립합니다.

```
[BOS] [visual tokens ×256] [prompt 텍스트] [instruction 텍스트] [suffix] [action tokens ×7]
                 ↑ V              ↑ 제외          ↑ L                        ↑ query
```

- **주의 1:** "In: What action should the robot take to {instruction}? Out:" 같은 템플릿의 **고정 문구는 L에서 제외**해야 합니다. 지시문의 내용어(content word)만 L로 잡아야 의미가 있습니다.
- **주의 2:** **BOS/sink 토큰은 별도 집합 S로 분리**합니다. 2603.06001이 attention sink를 원인으로 지목했으므로, sink를 V에 포함시키면 비전 비중이 인위적으로 부풀려집니다.
- 검증: 인덱스를 시각화해서 실제 이미지 패치/단어와 1:1 대응하는지 눈으로 확인

### Phase 2 — 관찰 지표 3종 (3일)

action token 생성 step *t*, layer *l*, head *h* 에 대해 (생성 중 캡처하면 query가 1토큰이라 메모리 부담 없음):

1. **Raw mass ratio (IVAR 재현)**
   `R_raw = Σ_{j∈L} A(j) / (Σ_{j∈L} A(j) + Σ_{j∈V} A(j))`
   → **선행연구 재현용. 단독으로 주장하면 안 됨** (토큰 수 비대칭)
2. **Per-token normalized ratio**
   `R_norm = (M_L/|L|) / (M_L/|L| + M_V/|V|)`
   → "토큰 하나당 얼마나 보는가". 개수 효과 제거
3. **Value-norm weighted ratio** (Kobayashi et al. 2020)
   `A(j)` 대신 `‖A(j)·v_j‖` 사용 → attention weight가 아니라 실제 전달되는 벡터 크기 기준

출력: **layer × head × action-step** 3차원 히트맵. "몇 층에서, 어느 head가, 7개 action token 중 몇 번째에서 언어를 보는가"

> **예상되는 결과와 그 함정:** R_raw는 0.05 근처(비전 압도)로 나올 가능성이 큽니다. 이건 발견이 아니라 산수라는 점을 스스로 먼저 말해야 합니다. R_norm에서 역전되는지가 진짜 관전 포인트입니다.

### Phase 3 — 인과 개입 (5일, 이 연구의 중심)

관찰만으로는 2605.00321에게 그대로 반박당합니다. 반드시 개입을 붙입니다.

| 개입 | 방법 | 측정 |
|---|---|---|
| **Language knockout** | action query → L 토큰 attention을 softmax 이전에 −inf로 마스킹 후 재정규화 | 7개 DoF 각 256-bin logit의 **KL(원본‖개입)** |
| **Vision knockout** | 동일하게 V에 적용 | 동일 |
| **ISS식 치환** | 언어 임베딩을 **같은 suite의 다른 지시문 임베딩**으로 치환 (0벡터 금지 — OOD가 되어 결과가 오염됨) | KL + action MSE |
| **Layer-wise knockout** | 특정 층 구간에서만 차단 | 어느 층에서 언어 정보가 실제로 쓰이는지 |

- 인과 기여도: `CE_L = KL(원본 ‖ L-knockout)`, `CE_V = KL(원본 ‖ V-knockout)` → **인과 비율 `CE_L/(CE_L+CE_V)`**
- **H2 검정:** 에피소드/타임스텝 단위로 `R_raw`(및 R_norm)와 `CE_L/(CE_L+CE_V)`의 **Spearman 상관**. 상관이 낮으면 → attention 비율 지표는 인과 기여의 proxy로 부적합. 이게 발표에서 제일 강한 슬라이드가 됩니다.

### Phase 4 — 반사실 설계 (H1 검정, 4일) ← **차별화의 핵심**

2×2 조건을 만듭니다. LIBERO 장면을 편집하거나 ICBench(2603.06001) 스타일로 지시를 조작합니다.

|  | **지시 정상** | **지시 모순/무의미** |
|---|---|---|
| **장면 비모호** (목표 물체 1개) | 언어 불필요 | 언어 불필요 |
| **장면 모호** (동종 물체 2개 이상, 색/위치로만 구분) | **언어 필수** | 언어 필수인데 틀림 |

- 측정: 네 조건에서 `R_norm`과 `CE_L/(CE_L+CE_V)` 비교
- **H1 예측:** 모호/비모호 간 차이가 통계적으로 유의하지 않다 → "비율이 demand에 반응하지 않는다"
- 이 설계가 §1.2의 **"LIBERO는 원래 언어가 필요 없다"는 confound를 정면으로 제거**합니다. 언어가 반드시 필요한 조건을 직접 만들었기 때문입니다.

### Phase 5 — 검증·대조군 (3일, 생략 금지)

| 대조군 | 목적 |
|---|---|
| **Randomized weights** (Adebayo et al.) | 지표가 학습된 모델에 실제로 의존하는지 |
| **Shuffled instruction** | 문법만 맞고 의미 없는 지시 |
| **Empty instruction** | 2510.13626 결과 재현 확인 |
| **Seed ×3 이상, task ×10 이상** | 분산 보고. 단일 rollout 그림 금지 |
| **행동 수준 대조** | attention 지표 변화가 실제 성공률 변화와 연결되는지 |

---

## 5. 최소 실행 계획 (2주 스프린트)

1. **1~2일차:** OpenVLA-7B + LIBERO 설치, eager attention으로 rollout 1회 성공
2. **3~4일차:** 토큰 인덱싱 확정 + 시각화 검증
3. **5~6일차:** R_raw / R_norm 계산, layer×head 히트맵 1장
4. **7~9일차:** language/vision knockout + KL. `CE` 비율 산출
5. **10~11일차:** R vs CE 상관 (H2 1차 결과)
6. **12~14일차:** 모호 장면 5개 수작업 제작 → Phase 4 축소판

**2주 끝의 결과물:** "attention 비율은 X인데 인과 기여 비율은 Y였고, 둘의 상관은 ρ=Z였다" — 이 문장 하나면 교수님 미팅에 충분합니다.

---

## 6. 이 연구가 공격받을 지점 (미리 준비할 방어)

| 공격 | 방어 |
|---|---|
| "IVAR(2603.06001)이 이미 했다" | "재현은 baseline이고, 기여는 **비율의 입력 적응성(H1)** 과 **비율 지표의 타당성 검정(H2)** 입니다. 두 논문 모두 안 다뤘습니다" |
| "attention은 설명이 아니다 (2605.00321)" | "그래서 attention을 **주장의 근거가 아니라 검정 대상**으로 놓았습니다. 인과 개입(KL)이 주 증거이고, attention은 그것과 비교되는 피검자입니다" |
| "LIBERO는 원래 언어가 필요 없다" | "Phase 4에서 **언어가 필수인 모호 장면**을 직접 구성했습니다" |
| "vision 토큰이 256개니 당연히 vision이 이긴다" | "R_norm(토큰당 정규화)과 value-norm 가중을 함께 보고합니다. raw는 재현용으로만 씁니다" |
| "OpenVLA 하나로 일반화 못 한다" | 정직하게 인정 + 여력 되면 **OpenVLA-OFT 또는 SmolVLA** 1개 추가. π₀ 계열은 flow matching이라 KL 정의부터 다시 해야 하므로 초기에는 피할 것 |
| "그래서 뭐 하자는 건가 (so what)" | **가장 위험한 질문.** 답: "비율이 조절되지 않는다면 데이터/아키텍처 어느 쪽을 고쳐야 하는지가 갈립니다. 또한 IVAR류 지표로 성능을 개선했다는 기존 결과들의 근거가 흔들립니다" — 이 답을 미리 문장으로 준비해 두세요 |

---

## 7. 이후 고도화 경로 (형우 님 관심사와의 연결)

- **SAE로의 확장:** attention 수준의 "비율"은 거칠습니다. residual stream을 SAE로 분해해 **language-derived feature vs vision-derived feature**를 세고, 그 비율을 재는 것이 다음 단계입니다. `SAEs Reveal Interpretable and Steerable Features in VLA`(2603.19183)가 π₀.₅/OpenVLA에서 레시피를 공개했습니다.
- **Steering으로의 확장:** 언어 기여를 인위적으로 키웠을 때 행동이 예측대로 바뀌는지 → `Mechanistic Interpretability for Steering VLA`(2509.00328, CoRL 2025)와 연결.
- **World model과의 연결:** 언어가 "무엇을"을, 비전이 "어떻게"를 담당한다면, world model의 predictive feature는 어느 쪽에서 오는가 — 프로젝트의 world model 트랙과 자연스럽게 이어지는 다리입니다.
- **MM-SHAP의 VLA 이식:** 검색 범위에서 VLA판을 찾지 못했습니다. Shapley 기반이라 attention 비판을 원천적으로 피해 갑니다. 계산비용이 병목이지만, action token 7개 × 모달리티 2개로 좁히면 감당 가능합니다.

---

## 8. 신뢰도 표기

- ✅ 원문 확인: 2510.13626(LIBERO-Plus), 2601.04052(Stable Language Guidance), 2603.06001(IVAR/IGAR), 2212.08158(MM-SHAP), Liang et al. 2022(modality gap)
- ⚠️ 요약만 확인, 원문 정독 필요: 2605.30117(VLA-Trace), 2605.00321(Embodied Interpretability), 2606.02277(RoboSemanticBench), 2603.19183(SAE)
- ⚠️ 검증 필요 사항: OpenVLA visual token 개수(256 가정), LIBERO fine-tuned 체크포인트의 정확한 프롬프트 템플릿 — 둘 다 **코드에서 직접 확인**할 것. 문서의 수치를 믿고 인덱싱하지 마세요.
- 2510.13626과 2601.04052의 빈-지시 결과 불일치는 **미해결**이며, 이 프로젝트에서 직접 확인할 항목입니다.

---

## 참고문헌

- LIBERO-Plus — https://arxiv.org/pdf/2510.13626
- Stable Language Guidance for VLA — https://arxiv.org/html/2601.04052
- Restoring Linguistic Grounding in VLA (IVAR/IGAR) — https://arxiv.org/html/2603.06001v1
- VLA-Trace — https://arxiv.org/abs/2605.30117
- Embodied Interpretability (ISS/NMR) — https://arxiv.org/html/2605.00321v1
- MM-SHAP — https://arxiv.org/abs/2212.08158 · 구현 https://github.com/Heidelberg-NLP/MM-SHAP
- Mind the Gap (modality gap) — https://github.com/Weixin-Liang/Modality-Gap
- SAEs Reveal Interpretable and Steerable Features in VLA — https://arxiv.org/html/2603.19183v1
- Mechanistic Interpretability for Steering VLA — https://arxiv.org/pdf/2509.00328
- VLA-Pruner — https://arxiv.org/html/2511.16449v1
- Sanity Checks for Saliency Maps — https://papers.neurips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf
