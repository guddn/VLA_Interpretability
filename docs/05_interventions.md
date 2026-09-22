# 인과 개입 설계 — **어디에 개입하는가**로 분류

관찰(attention 비율)만으로는 반박당합니다. Embodied Interpretability(2605.00321)의
"attention 은 task-irrelevant 영역에도 뜬다"가 정확히 그 비판입니다.
그래서 **개입해서 행동 분포가 실제로 바뀌는지**를 봅니다.

이 문서는 개입을 **파이프라인의 어느 지점을 건드리는가**로 분류합니다.
같은 "언어 제거"라도 어디서 제거하느냐에 따라 **주장할 수 있는 내용이 다릅니다.**

---

## 개입 지점 지도

```
픽셀                     [L0] 이미지 입력        ⬜ 미구현
  │  LIBERO → MuJoCo → EGL
  ▼
PIL.Image
  │  processor (리사이즈·정규화)
  ▼
pixel_values ──▶ DINOv2+SigLIP ──▶ projector ──┐
                                                ├─▶ 시퀀스 [S|V×256|T|L|A×7]
지시문 문자열 ─▶ tokenizer ────────────────────┘
  ▲                                              │
  └── [L1] 언어 입력       ✅ 04_counterfactual   │
                                                  ▼
                             [L2] 토큰 임베딩 치환  ⬜ 미구현
                                                  │
                                                  ▼
                        Llama-2-7B  32층 × 32헤드
                          각 층: softmax(QKᵀ/√d + mask)V
                                      ▲
                                      └── [L3] attention mask   ✅ intervene.py ★주력
                                          [L4] 층·헤드 선택      ⚠ 구현됐으나 미노출
                                                  │
                                                  ▼
                                      action logits [7, 256]
                                                  │
                                      [L5] 출력 비교 (측정, 개입 아님)
```

---

## [L0] 이미지 입력 — **⬜ 미구현**

**무엇을 바꾸나**: 모델에 넣기 전 픽셀 자체.

| 조건 | 내용 | 무엇을 잴 수 있나 |
|---|---|---|
| `random_image` | 난수 픽셀로 교체 | 시각 정보 전체의 기여 |
| `pixel_shuffle` | 픽셀 위치만 섞음 | **저수준 통계 유지, 공간 구조만 파괴** |
| `object_mask` | 목표 물체만 가림 | 특정 물체의 기여 |
| `noise` / `lighting` | 노이즈·조명 변화 | sim→real 강건성 |

**이 지점의 장점**: 모델 구조를 전혀 안 건드립니다. 아키텍처가 달라도 그대로 적용되므로
**나중에 다른 VLA 나 실물 로봇으로 옮길 때 그대로 재사용**됩니다.

**단점**: 입력 분포를 벗어납니다(OOD). "언어를 안 쓴다"가 아니라 "이상한 입력이라
망가졌다"일 수 있습니다. `pixel_shuffle` 이 그나마 나은 이유가 이것 — 색·밝기 분포는
그대로라 OOD 정도가 덜합니다.

> ### ★ 현재 설계의 가장 큰 구멍
> 언어 쪽은 [L1]에 조건이 4개인데 **이미지 쪽은 0개**입니다.
> "언어 조건은 정교하게 설계해 놓고 이미지는 왜 안 했나" 는 정당한 반박이고,
> 지금은 답이 없습니다. `scripts/06_image_ablation.py` 로 메울 예정.

---

## [L1] 언어 입력 — ✅ `scripts/04_counterfactual.py`

**무엇을 바꾸나**: 지시문 문자열. 같은 장면·같은 타임스텝에서 지시만 교체합니다.

| 조건 | 내용 | 역할 |
|---|---|---|
| `valid` | 원래 지시 | 기준 |
| **`swapped`** | 같은 suite 의 **다른 task 지시문** | **모순 지시. 핵심 조건** |
| `paraphrase` | 동의어 치환 (pick up→grasp 등) | 의미 동일, 표면 다름 |
| `empty` | 빈 지시 | 2510.13626 재현용. **OOD 라 단독 근거 금지** |

**측정**: `KL_vs_valid`, `n_identical_dof` (7이면 지시를 완전 무시), 조건별 `R_raw/R_norm/R_vnorm`

**왜 `swapped` 가 핵심인가**: `empty` 는 입력 분포 밖이라 "언어를 안 쓴다"와
"깨진 입력이라 이상해졌다"를 구분 못 합니다. `swapped` 는 **같은 분포의 다른 지시문**이라
그 혼동이 없습니다. ISS(2605.00321) 의 replacement 철학과 같습니다.

**H1 예측**: `valid` ↔ `swapped` 사이에서 **R 은 거의 안 움직이는데 KL 도 작다**
→ "비율이 언어 수요(demand)에 반응하지 않는다"

> ### 벤치마크 한계 — 반드시 대비할 것
> LIBERO 는 **장면이 task 를 거의 결정**합니다. 낮은 언어 의존이 나와도
> "모델 결함이 아니라 벤치마크가 언어를 필요로 하지 않기 때문" 이라고 반박됩니다.
> `--ambiguous-tasks` 로 **한 장면에 후보가 둘 이상인 task** 를 직접 지정해야
> 이 반박을 막을 수 있습니다. 지정 안 하면 스크립트가 경고를 냅니다.

---

## [L2] 토큰 임베딩 치환 — ⬜ 미구현

**무엇을 바꾸나**: 토큰이 이미 벡터가 된 뒤, 그 벡터를 다른 값으로 교체.

[L1]과 다른 점: [L1]은 토큰 **개수와 위치**까지 바뀌지만, [L2]는 **시퀀스 구조를 고정한 채
내용만** 바꿉니다. 그래서 위치 효과와 내용 효과를 분리할 수 있습니다.

`vlamod/pipeline.py` 의 `counterfactual_instruction()` 이 비슷한 일을 하지만,
실제로는 문자열을 바꿔 다시 토큰화하므로 **[L1]에 가깝습니다.** 그리고
`04_counterfactual.py` 는 이 함수를 쓰지 않고 자체 구현을 씁니다 — **중복입니다.**

---

## [L3] attention mask — ✅ `vlamod/intervene.py` ★ 현재 주력

**무엇을 바꾸나**: 입력은 그대로. action token 이 특정 토큰 집합을 **못 보게** 만듭니다.

### 구현

각 decoder layer 에 `forward_pre_hook(with_kwargs=True)` 를 걸어,
4D additive causal mask `[B, 1, Tq, Tk]` 의 (action query 행 × 대상 key 열) 교차점에
`-inf` 를 더합니다.

```
softmax( QKᵀ/√d + mask )      ← mask 에 -inf 를 더함
        ▲
        └─ 여기서 개입 (pre-softmax)
```

**왜 pre-softmax 인가**: softmax **이전**에 막으면 나머지 열이 자동으로 재정규화됩니다.
"그 토큰이 애초에 없었다면" 에 가깝습니다. softmax 이후에 0으로 만들고 다시 정규화하는
것과 **결과가 다릅니다.**

### 조건 3개

| # | 함수 | 차단 대상 | 개수 |
|---|---|---|---|
| 1 | `knockout_language` | L 전체 | 20 |
| 2 | `knockout_vision` | V 전체 | 256 |
| 3 | **`knockout_random_control`** | **무작위 visual 토큰 |L|개** | 20 |

### 3번이 이 설계의 핵심입니다

이게 없으면 다음 두 가지가 **구분되지 않습니다**:

- "언어를 지웠더니 행동이 안 변했다" → 언어를 안 쓴다
- "토큰 20개 지운 건 원래 아무 영향이 없다" → 측정 자체가 무의미

**같은 개수**를 지우는 대조군이 있어야 언어가 특별하다고 말할 수 있습니다.
`lang_vs_control = KL_L / KL_C` 가 **1 근처면 언어가 특별하지 않다**는 뜻입니다.

### 측정 지표

| 컬럼 | 정의 | 읽는 법 |
|---|---|---|
| `KL_lang_knockout` | KL(기준 ‖ 언어 차단) | 클수록 언어 의존 |
| `KL_vis_knockout` | KL(기준 ‖ 비전 차단) | 클수록 비전 의존 |
| `KL_control_knockout` | KL(기준 ‖ 랜덤 visual 차단) | **바닥값(floor)** |
| `causal_ratio` | KL_L / (KL_L + KL_V) | 인과 기여의 언어 몫 |
| **`lang_vs_control`** | KL_L / KL_C | **1 이하면 언어가 특별하지 않음** |
| `argmax_changed_lang/vis` | 7 DoF 중 argmax bin 이 바뀐 개수 | 분포가 아닌 **실제 행동** 변화 |

### 중요한 구현 세부 — 모든 조건이 같은 action token 으로 teacher-forcing

```python
base = generate_action_tokens(...)            # 기준에서 한 번만 생성
c0   = teacher_forced_capture(..., base, ...)                   # 기준
c_l  = teacher_forced_capture(..., base, ..., attn_mask_fn=ko_l)  # 같은 base
c_v  = teacher_forced_capture(..., base, ..., attn_mask_fn=ko_v)  # 같은 base
c_c  = teacher_forced_capture(..., base, ..., attn_mask_fn=ko_c)  # 같은 base
```

**같은 위치·같은 prefix 에서의 분포 변화만** 재기 때문에 KL 이 깨끗하게 해석됩니다.
prefix 가 달라지면 "분포가 바뀐 것"과 "앞 토큰이 달라져 흘러간 것"이 섞입니다.

> ### ⚠ [L1]과 규약이 다릅니다 — 결정 필요
> `04_counterfactual.py` 의 `one_condition()` 은 **조건마다 자기 action token 을
> 새로 생성**한 뒤 teacher-forcing 합니다. 즉 [L3]과 달리 prefix 가 조건마다 다릅니다.
>
> | | prefix | KL 이 재는 것 |
> |---|---|---|
> | [L3] knockout | **공통(base)** | 순수한 조건부 분포 변화 |
> | [L1] counterfactual | 조건별 | 분포 변화 + autoregressive drift |
>
> 둘 다 의미는 있지만 **같은 표에 나란히 놓으면 안 됩니다.**
> [L1]도 `valid` 의 토큰으로 통일하는 쪽을 권합니다 (수정 1줄 수준).

---

## [L4] 층·헤드 단위 — ⚠ 구현됐으나 **실행되지 않음**

`AttentionKnockout(..., layers=[10, 11, 12])` 로 **특정 층만** 차단할 수 있습니다.
`pipeline.analyze_step(layers_subset=...)` 까지 배선되어 있습니다.

**그런데 `scripts/02_run_analysis.py` 가 이 인자를 CLI 로 노출하지 않습니다.**
기본값 `None` = 전층 동시 차단. 즉 **지금은 층별 인과를 볼 수 없습니다.**

### 왜 이게 문제인가

| 축 | 층별 데이터 |
|---|---|
| 관찰 (`R`) | ✅ `outputs/perlayer_*.npz` 에 층별 저장 |
| 인과 (`ΔKL`) | ❌ 전층 통합값 하나뿐 |

**H2(attention 비율이 인과 기여를 예측하는가)를 층별로 검정할 수 없습니다.**
전체 평균끼리만 비교하면 표본이 사실상 1개씩이고, 32층×32헤드를 평균 내면
층별 분화(초기=시각, 후기=행동 결정 같은)가 뭉개집니다.

헤드 단위 knockout 은 아예 미구현입니다.

---

## [L5] 출력 비교 — 개입이 아니라 **측정**

action logits `[7, 256]` 두 개를 비교합니다.

| 함수 | 정의 | 성질 |
|---|---|---|
| `action_kl(P, Q)` | KL(P‖Q), 7 DoF 평균 | **비대칭**. 항상 (기준 ‖ 개입) 순서로 |
| `decode_actions` | argmax bin index | 분포가 아닌 실제 선택 |
| `causal_ratio(a, b)` | a/(a+b) | 0~1 정규화 |

### 아키텍처 이식성 한계

KL 은 **action 이 이산 bin 으로 토큰화된 모델에서만** 정의됩니다.
π0 같은 flow-matching / diffusion 정책은 연속 action 을 내므로 이 지표가 안 통합니다.
나중에 다른 VLA 로 확장하려면 **action 벡터의 L1/MSE** 를 병기해 두어야 합니다.
(현재 미구현)

---

## 분류 요약표

| 지점 | 무엇을 바꾸나 | 구현 | 조건 수 | 아키텍처 이식성 | OOD 위험 |
|---|---|---|---|---|---|
| [L0] 이미지 입력 | 픽셀 | ⬜ | 0 | **높음** | 중~높음 |
| [L1] 언어 입력 | 지시문 문자열 | ✅ | 4 | **높음** | 낮음(`swapped`)~높음(`empty`) |
| [L2] 임베딩 치환 | 토큰 벡터 | ⬜ | 0 | 중간 | 낮음 |
| [L3] attention mask | 정보 경로 | ✅ | 3 | 낮음(Transformer 전용) | **없음** |
| [L4] 층·헤드 | 개입 범위 | ⚠ 미노출 | 0 | 낮음 | 없음 |

**[L3]의 장점**: 입력이 분포 안에 그대로 있어 OOD 걱정이 없습니다. "이상한 입력이라
망가졌다"는 반박이 원천적으로 불가능합니다.

**[L3]의 단점**: Transformer attention 구조에 묶여 있습니다. 실물 로봇이나 다른
아키텍처로 못 옮깁니다. **그래서 [L0]/[L1]이 필요합니다** — 그쪽은 모델 내부를
안 건드리므로 어디든 적용됩니다.

---

## 알려진 문제 (해결 대기)

1. **[L0] 이미지 개입 0개** — 언어 4 vs 이미지 0 의 비대칭. 가장 큰 구멍
2. **[L4] CLI 미노출** — 층별 인과를 못 봄. H2 층별 검정 불가
3. **[L1]/[L3] teacher-forcing 규약 불일치** — 위 ⚠ 참고
4. **`configs/default.yaml` 의 `intervene:` 섹션이 죽어 있음**

   ```yaml
   intervene:
     targets: ["language", "vision"]
     replacement_baseline: "other_instruction"
   ```

   코드 어디서도 읽지 않습니다(grep 0건). **설정을 바꿔도 동작이 안 바뀝니다.**
   "config 에 적힌 대로 돌렸다"가 거짓이 되므로 재현성 관점에서 위험합니다.
   제거하거나 실제로 연결해야 합니다.
5. **연속 action 지표 부재** — KL 은 이산 bin 전용. π0 등으로 확장 불가

---

## 예상 반박과 방어

| 반박 | 방어 |
|---|---|
| "attention 을 막은 게 정보를 지운 것과 같나" | pre-softmax 마스킹이라 재정규화됨. 잔차 스트림 경유 우회 가능성은 **인정하고 명시**할 것 |
| "언어 20개 지운 효과가 작은 건 당연" | `knockout_random_control` 이 같은 개수를 지운 바닥값을 줌 |
| "빈 지시는 OOD 아닌가" | 맞음. 그래서 `swapped` 가 주 조건이고 `empty` 는 선행연구 재현용 |
| "LIBERO 는 원래 언어가 필요 없다" | `--ambiguous-tasks` 조건 필수. 없으면 결론을 약하게 쓸 것 |
| "전층을 한꺼번에 막은 결과 아닌가" | **현재 방어 불가.** [L4] 노출이 필요 |
| "이미지 쪽은 왜 안 했나" | **현재 방어 불가.** [L0] 구현이 필요 |
