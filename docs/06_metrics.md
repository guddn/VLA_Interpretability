# 06. 지표 정의

이 문서는 `02_run_analysis.py` / `04_counterfactual.py` 가 CSV 에 남기는 **모든 지표의 정확한 정의**입니다.
수식은 코드(`vlamod/metrics.py`, `capture.py`, `intervene.py`, `token_index.py`, `pipeline.py`)를 그대로 옮긴 것이며,
"의도"가 아니라 **"현재 구현이 실제로 계산하는 것"** 을 적었습니다. 둘이 다른 곳은 §7 에 모았습니다.

> ✅ **2026-10-06 수정**: §7-1(query 행 한 칸 어긋남), §7-2(29871 누락)를 고쳤습니다. 이 문서의 정의는 **수정 후 코드** 기준입니다.
> 2026-10-06 이전에 수집한 CSV(2026-09-29 Stage 3 포함)는 §7-1·§7-2 의 옛 동작으로 계산된 값입니다.

---

## 0. 한눈에 — CSV 컬럼 사전

`outputs/analysis_<suite>.csv` (한 행 = 한 타임스텝)

| 컬럼 | 종류 | 한 줄 정의 | 범위 | 비교 기준 |
|---|---|---|---|---|
| `R_raw` | 관찰 | V+L 로 간 attention 중 L 의 몫 | [0,1] | `uniform_baseline` |
| `R_norm` | 관찰 | 토큰 1개당 attention 으로 본 L 의 몫 | [0,1] | 0.5 |
| `R_vnorm` | 관찰 | attention × ‖value‖ 로 본 L 의 몫 | [0,1] | — |
| `uniform_baseline` | 관찰 | \|L\| / (\|L\|+\|V\|) — attention 이 균등할 때의 R_raw | — | — |
| `mass_sink` | 관찰 | sink(BOS) 로 간 attention 질량 | [0,1] | — |
| `mass_other` | 관찰 | V·L·sink 이외(템플릿 + action 토큰)로 간 질량 | [0,1] | — |
| `R_norm_step{k}` | 관찰 | DoF k 를 예측하는 query 행(Q 의 k 번째)만의 R_norm | [0,1] | 0.5 |
| `KL_lang_knockout` | 개입 | action→L attention 을 끊었을 때 action 분포 변화 | [0,∞) nats | `KL_control_knockout` |
| `KL_vis_knockout` | 개입 | action→V attention 을 끊었을 때 | [0,∞) nats | 〃 |
| `KL_control_knockout` | 개입 | action→(무작위 visual \|L\|개) 를 끊었을 때 | [0,∞) nats | 0 |
| `causal_ratio` | 개입 | KL_L / (KL_L + KL_V) | [0,1] | 0.5 |
| `causal_ratio_adj` | 개입 | 대조군을 뺀 뒤의 causal_ratio | [0,1] | 0.5 |
| `lang_minus_control` | 개입 | KL_L − KL_ctrl | ℝ | 0 |
| `vis_minus_control` | 개입 | KL_V − KL_ctrl | ℝ | 0 |
| `lang_vs_control` | 개입 | KL_L / KL_ctrl — **불안정, 통계에 쓰지 말 것** | (0,∞) | 1 |
| `argmax_changed_lang` | 개입 | 언어 knockout 으로 argmax bin 이 바뀐 DoF 수 | 0–7 | — |
| `argmax_changed_vis` | 개입 | 시각 knockout 으로 〃 | 〃 | — |
| `n_L`, `n_V` | 메타 | 언어/시각 토큰 개수 | — | — |
| `action_bins` | 메타 | 무개입 greedy action bin 7개 | 0–255 | — |
| `ep_success` | 에피소드 | `check_success()` 결과 (에피소드의 모든 행에 소급 기입) | bool | — |
| `success_source` | 에피소드 | 성공 판정 출처 (`check_success` / `done_flag(부정확)`) | — | — |
| `ep_len` | 에피소드 | 에피소드 실제 길이 (env 스텝) | — | — |
| `suite`, `task_id`, `episode`, `t` | 메타 | 식별자. `t` 는 에피소드 안의 env 스텝 | — | — |

`outputs/perlayer_<suite>.npz` — 층별 `R_raw`, `R_norm`, `R_vnorm` (헤드·query 평균, 행마다 길이 32 배열)

---

## 1. 토큰 구간

OpenVLA 입력 시퀀스를 다섯 집합(S·V·L·T·A)과 query 행 Q 로 나눕니다 (`token_index.build_spans`).

```
[BOS] [visual ×256] [In: What action should the robot take to] [지시문] [?\nOut:] [▁] [action ×7]
  S          V                        T                              L        T       T      A
                                                                            Q = ▁ 위치 + A 의 앞 6개  (= A − 1)
```

| 기호 | 이름 | 위치 (task 0 기준) | 개수 | 정의 |
|---|---|---|---|---|
| S | sink | 0 | 1 | `sink_positions` (기본 BOS 하나) |
| V | visual | [1, 257) | 256 | `probe_visual_span` 으로 실측 (서로 다른 이미지 두 장을 넣어 hidden 이 달라지는 구간) |
| L | language | 지시문 토큰 | 20 | 문자 오프셋이 지시문 구간과 **겹치는** 텍스트 토큰. 앞뒤 공백을 벗기고 판정 |
| T | template | 나머지 텍스트 + `▁` | 14 | 프롬프트 고정 문구 13개 + 끝에 붙인 빈 토큰 `▁`(id 29871) 1개. 모든 샘플에 동일 |
| A | action | [291, 298) | 7 | 프롬프트 뒤에 teacher-forcing 으로 붙인 action 토큰 **자기 자리** |
| Q | query | [290, 297) | 7 | action 토큰을 **예측하는 자리** = A − 1. 관찰·개입·logit 이 모두 이 행을 씀 |

- `▁`(29871): OpenVLA `predict_action` 이 학습 입력과 맞추려고 붙이는 토큰. processor 는 안 붙이므로 `TI.prepare_inputs` / `TI.append_action_prefix` 가 붙입니다. DoF 0 을 예측하는 query 가 바로 이 위치입니다.
- `instruction_only=True`(기본): L 은 지시문만. False 면 템플릿까지 L.
- \|L\| 은 지시문마다 다릅니다 (LIBERO-spatial task 0~4 에서 15~21). `uniform_baseline` 이 행마다 다른 이유입니다.
- 자기검증: \|L\| 이 지시문 단독 토큰화 개수와 2 이상 차이나면 경고.
- 희귀어는 subword 로 쪼개져 \|L\| 이 부풉니다 (예: `ramekin` → `r`/`ame`/`kin`).

---

## 2. 어떤 query 를 보는가

모든 관찰 지표의 query 는 **Q 의 7개 위치** (`spans.query`) 입니다.
autoregressive LM 에서 위치 p 의 출력이 p+1 번째 토큰을 예측하므로, Q 의 k 번째 행 = **DoF k 를 예측하는 행** 입니다.
attention 텐서 `attn[l, h, k, j]` = 층 l, 헤드 h 에서 **DoF k 를 예측하는 위치가 key j 에 준 attention**, shape `[32, 32, 7, T]`.
KL 에 쓰는 logit 도 **같은 Q 행**에서 가져옵니다 → 관찰과 개입이 같은 행을 봅니다.

캡처는 2-pass 입니다 (`capture.py`).

1. `generate_action_tokens`: greedy 로 action 토큰 7개 생성
2. `teacher_forced_capture`: [프롬프트 + 그 7개] 를 한 번에 forward → 모든 층·헤드 attention, value norm, logit 수집

> `capture.check_query_rows` 가 매 캡처마다 Q = [seq_len−8, seq_len−2] 인지 검증합니다. A 를 넘기면 에러가 납니다.

---

## 3. 관찰 지표 (attention)

### 3-1. 질량

한 칸 (l, h, k) 에서 집합 X 로 간 attention 합:

$$M_X(l,h,k) = \sum_{j \in X} A_{l,h}(k, j)$$

softmax 이므로 한 행의 합은 1: $M_S + M_V + M_L + M_{\text{other}} = 1$.

| 컬럼 | 식 (윗줄 = 32층 × 32헤드 × 7행 평균) |
|---|---|
| `mass_sink` | $\overline{M_S}$ |
| `mass_other` | $\overline{1 - M_V - M_L - M_S}$ — **템플릿 + 이전 action 토큰 + 자기 자신** |

### 3-2. R_raw — 원시 비율

$$R_{\text{raw}}(l,h,k) = \frac{M_L}{M_L + M_V}$$

- sink·template·action 으로 간 질량은 **분모에서 빠집니다.** "V+L 로 간 attention 중 L 의 몫" 이지 "전체 attention 중 L 의 몫" 이 아닙니다.
  - 예 (Stage 2, task 0): R_raw 0.386, V+L 질량 0.161 → 전체 attention 중 L 은 0.386 × 0.161 ≈ **6.2%**.
- **단독 주장 금지.** 토큰 수가 256 : 20 이라 attention 이 완전히 균등해도 R_raw = 20/276 = 0.072 입니다. 항상 `uniform_baseline` 과 같이 보고하세요.

### 3-3. uniform_baseline

$$\text{uniform\_baseline} = \frac{|L|}{|L| + |V|}$$

"토큰마다 attention 을 똑같이 줬다면" 의 R_raw.
**R_raw / baseline** 이 "언어 토큰이 개수 대비 몇 배 주목받는가" 입니다 (Stage 3 평균 기준 약 0.417 / 0.066 ≈ 6.4배).

### 3-4. R_norm — 토큰당 비율

$$R_{\text{norm}} = \frac{M_L/|L|}{M_L/|L| + M_V/|V|}$$

- 언어 토큰 1개의 평균 attention 과 시각 토큰 1개의 평균 attention 비교. **기준값 0.5** = 토큰당 동등.
- 0.845 → 언어 토큰 1개가 시각 토큰 1개보다 약 0.845/0.155 ≈ 5.5배 attention 을 받음.
- 약점: 시각 토큰은 대부분 배경이라 "평균 시각 토큰" 이 원래 낮습니다. R_norm 이 높다고 언어를 "더 쓴다" 는 뜻은 아닙니다.

### 3-5. R_vnorm — value-norm 가중 비율

$$R_{\text{vnorm}} = \frac{\sum_{j\in L} A(k,j)\,\lVert v_j \rVert}{\sum_{j\in L} A(k,j)\,\lVert v_j \rVert + \sum_{j\in V} A(k,j)\,\lVert v_j \rVert}$$

- $v_j$ = 층 l `v_proj` 출력을 헤드별로 자른 벡터의 L2 norm (`[H, T]`).
- 동기: attention 이 커도 value 가 작으면 실제로 옮기는 정보가 적음 (Kobayashi et al., 2020).
- 구현상 단순화: 원 논문은 $\lVert W_O v_j \rVert$ (출력 투영까지) 를 씁니다 → §7-4.
- 관측: R_vnorm ≈ R_raw − 0.02. 가중을 해도 그림이 거의 안 바뀝니다.

### 3-6. 평균 순서

CSV 의 R 값은 **칸별 비율의 평균** 입니다:

$$R = \frac{1}{32 \cdot 32 \cdot 7} \sum_{l,h,k} R(l,h,k)$$

"질량을 먼저 평균한 뒤 비율" 과 다릅니다. V+L 질량이 거의 0 인 헤드(sink 에 몰아주는 헤드)도 비율 하나로 동등하게 들어갑니다 → §7-6.

### 3-7. R_norm_step{k}, per-layer

- `R_norm_step{k}`: 층·헤드만 평균, query 는 Q 의 k 번째 행(= DoF k 를 예측하는 행) 하나. "DoF 별로 언어 의존이 다른가" 용도.
- `perlayer_*.npz`: 헤드·query 만 평균한 층별 32개 값.

---

## 4. 개입 지표 (attention knockout)

### 4-1. knockout 이 하는 일

`intervene.AttentionKnockout(query_idx, key_idx, layers)`

- 모든 decoder 층(기본)의 4D additive mask 에서 (query ∈ Q, key ∈ X) 칸을 `finfo.min`(≈ −∞) 으로 바꿉니다.
- **softmax 이전** 개입 → 나머지 key 들로 자동 재정규화. "action 쪽에서 그 토큰들이 안 보였다면" 에 가까움.
- 프롬프트 위치끼리의 attention, 그리고 **X 토큰 자체의 표현은 그대로** 입니다. 끊는 것은 A → X 경로뿐.
- `layers=None` → 32층 전부 동시 차단. 층별 knockout 은 구현돼 있지만 CLI 에 노출돼 있지 않습니다.

| 조건 | query | key |
|---|---|---|
| 언어 knockout | Q | L 전체 |
| 시각 knockout | Q | V 전체 (256개) |
| 대조군 knockout | Q | V 중 무작위 \|L\| 개 (`seed=0`) |

### 4-2. action_kl — 분포 변화량

DoF d 의 action 분포 = 그 DoF 를 예측하는 위치의 logit 중 **action bin 256개만 잘라** softmax 한 것 (vocab 전체가 아님; `action_logit_slice` 는 vocab 마지막 256개).

$$\mathrm{KL}_d = \sum_{b=1}^{256} p_d(b)\,\log\frac{p_d(b)}{q_d(b)}, \qquad \mathrm{KL} = \frac{1}{7}\sum_{d=0}^{6} \mathrm{KL}_d$$

- $p$ = 무개입, $q$ = knockout. 방향은 **KL(무개입 ‖ 개입)**, 단위 nats.
- 7 DoF **평균** (합이 아님).
- knockout 쪽도 **무개입 greedy action 토큰을 prefix 로 강제**(teacher forcing) 합니다. DoF d 의 KL 은 "앞 DoF 들이 원래대로 나왔다는 조건에서 d 의 분포가 얼마나 바뀌나" = 스텝별 직접 효과이며, 오류 누적은 반영하지 않습니다.
- 크기 감각: 무개입 분포가 한 bin $b^*$ 에 거의 몰려 있으면 $\mathrm{KL}_d \approx -\log q_d(b^*)$ 입니다. 따라서 **KL > ln 256 ≈ 5.55** 는 "개입 후 원래 bin 에 균등분포(1/256)보다도 낮은 확률을 준다" 는 뜻입니다.

### 4-3. causal_ratio

$$\text{causal\_ratio} = \frac{\mathrm{KL}_L}{\mathrm{KL}_L + \mathrm{KL}_V}$$

- 0.5 = 언어·시각 경로를 끊었을 때의 피해가 같음. R 과 **같은 축(0~1, L 의 몫)** 에 놓으려고 만든 값 → H2 는 "R 과 causal_ratio 가 같이 움직이는가".
- 약점: 두 knockout 의 크기가 다릅니다 (토큰 ~20개 vs 256개 차단). 0.5 가 "같은 정도로 쓴다" 는 뜻은 아닙니다. 토큰당 인과 기여를 원하면 개수 정규화가 필요합니다.

### 4-4. 대조군 파생 지표

| 컬럼 | 식 | 쓰임 |
|---|---|---|
| `lang_vs_control` | KL_L / KL_ctrl | **쓰지 말 것.** KL_ctrl 이 0.002~1.27 로 흔들려 비율이 4~3000배로 튐 |
| `lang_minus_control` | KL_L − KL_ctrl | 언어 차단 효과가 "아무 visual 토큰 \|L\|개 차단" 보다 얼마나 큰가 |
| `vis_minus_control` | KL_V − KL_ctrl | 시각 전체 차단이 무작위 소수 차단보다 얼마나 큰가 |
| `causal_ratio_adj` | max(KL_L−KL_c,0) / (max(KL_L−KL_c,0) + max(KL_V−KL_c,0)) | 대조군 바닥값을 뺀 causal_ratio. KL_c ≪ KL_L, KL_V 인 현재는 causal_ratio 와 거의 같음 |

- 대조군 토큰 집합은 `seed=0` 고정 → **같은 task 안에서는 모든 스텝이 같은 visual 위치를 차단** 합니다 (§7-5).

### 4-5. argmax_changed_lang / vis

$$\#\{d : \arg\max_b p_d(b) \ne \arg\max_b q_d(b)\}$$

분포가 아니라 **실제로 선택되는 bin** 이 바뀐 DoF 수. KL 이 커도 argmax 가 안 바뀔 수 있으니 같이 보세요.

---

## 5. H1 지표 (`04_counterfactual.py`)

`outputs/counterfactual_<suite>.csv` (한 행 = 스텝 × 지시 조건)

| 컬럼 | 정의 |
|---|---|
| `condition` | `valid`(원 지시) / `swapped`(다른 task 지시) / `paraphrase`(동의어 치환) / `empty`(공백 한 칸) |
| `R_raw`, `R_norm`, `R_vnorm`, `uniform_baseline` | §3 과 동일, 해당 지시문으로 계산 |
| `KL_vs_valid` | KL(valid ‖ 이 조건). 같은 장면에서 지시만 바꿨을 때의 분포 변화 |
| `n_identical_dof` | valid 와 argmax bin 이 같은 DoF 수. 7 이면 지시를 바꿔도 행동 동일 |
| `prefix_mode` | `shared`(기본): 모든 조건이 valid 의 action 토큰을 prefix 로 공유 / `own`: 각자 greedy |
| `ambiguous` | `--ambiguous-tasks` 로 지정한 task 인지 |

개입 위치별 분류는 `docs/05_interventions.md` 참고.

---

## 6. 통계 (`03_correlation.py`)

| 항목 | 내용 |
|---|---|
| H2 | (`R_raw`/`R_norm`/`R_vnorm`) × `causal_ratio` 의 Spearman ρ, Pearson r (n ≥ 8) |
| 해석 기준 (스크립트 출력) | \|ρ\| < 0.3 → attention 은 인과 기여를 거의 예측 못 함 (H2 지지) / \|ρ\| > 0.6 → 쓸 만한 proxy (H2 기각) / 사이 → 조건부 |
| 대조군 | `lang_vs_control` 의 1표본 t-검정 (H0: 평균 = 1) — §4-4 이유로 **`lang_minus_control` 의 Wilcoxon 부호순위 검정으로 교체 권장** |

> ⚠ **독립성 위반.** 한 에피소드 안의 행들은 5스텝 간격이라 강하게 자기상관합니다.
> 364행을 독립으로 보고 낸 p 값은 과대평가입니다. 독립 단위는 사실상 **에피소드 15개**.
> 에피소드 단위 bootstrap 또는 혼합효과 모형(task/episode 랜덤효과)으로 보고하세요.

---

## 7. 알려진 문제 — 의도 ≠ 구현

### 7-1. ✅ (해결 2026-10-06) query 행과 logit 행이 한 칸 어긋남

**수정 전 동작** (2026-10-06 이전 데이터에 해당):

| | 위치 | 의미 |
|---|---|---|
| attention 을 재는 행 / knockout 하는 query | A = [P, P+6] (P = 프롬프트 길이) | action 토큰 **자기 자리** |
| KL 에 쓰는 logit 행 (`capture.py`: `rows - 1`) | [P−1, P+5] | action 토큰을 **예측하는 자리** |

autoregressive LM 에서 위치 p 의 출력은 p+1 번째 토큰을 예측합니다. 따라서

- **DoF 0** 은 위치 P−1(프롬프트 마지막 토큰)이 예측하는데, 이 행은 attention 측정에도 knockout 에도 **포함되지 않습니다.** 인과 마스크 때문에 P−1 은 뒤쪽 위치의 knockout 영향을 받을 수 없으므로 **KL_0 = 0 (수치 오차 수준, 항상)**, `argmax_changed_*` 최대값은 **6**.
  → 보고된 KL = (DoF 1~6 의 KL 합) / 7.
- **A 의 마지막 행(P+6)** 은 action 뒤의 종료 토큰을 예측하는 자리인데 attention 평균에 1/7 비중으로 들어갑니다. 이 행의 knockout 은 어떤 KL 에도 반영되지 않습니다.
- `R_norm_step{k}` 는 DoF k 가 아니라 **DoF k+1 을 예측하는 행** 입니다.

**영향**: 관찰(R)과 개입(KL)이 **서로 다른 행 집합**을 보고 있어 H2 비교의 전제가 어긋납니다. causal_ratio 자체는 KL_L, KL_V 가 같은 방식으로 영향받아 변화가 작을 수 있지만, 수정 전에는 확신할 수 없습니다.
**수정**: `TokenSpans.query` (= A − 1) 를 추가하고 attention 캡처·knockout·logit 이 모두 Q 를 쓰도록 변경 (`token_index`, `capture`, `intervene`, `pipeline`, `01`, `04`). `capture.check_query_rows` 로 매번 검증, 테스트 2개 추가.
**옛 데이터 확인 방법**: 옛 코드로 `M.action_kl(p, q, reduction="none")` 을 찍으면 첫 원소가 0 (수치 오차 수준) 이어야 합니다.

### 7-2. ✅ (해결 2026-10-06) 분석 경로 프롬프트에 토큰 29871 이 없음

**수정 전 동작** (2026-10-06 이전 데이터에 해당):

OpenVLA `predict_action` 은 학습 때 입력과 맞추려고 프롬프트 끝에 빈 토큰(id 29871)을 붙입니다 (그래서 291 vs 290 에러가 났습니다).
**환경을 굴리는 정책(`policy_action`)은 29871 이 있고, 분석(`analyze_step` → `generate_action_tokens`)은 없습니다.**

- 분석하는 action 분포가 실제로 실행된 action 분포와 **다를 수 있습니다.**
- 분석 프롬프트가 학습 분포에서 한 토큰 벗어나 있습니다.

**수정**: `TI.prepare_inputs` (processor → device → `append_action_prefix`) 를 분석·정책·반사실 경로가 공통으로 사용. input_ids 와 attention_mask 를 **같이** 1칸 늘리므로 291/290 에러가 나지 않고, `predict_action` 도 이미 붙어 있으면 다시 붙이지 않습니다. 29871 위치는 T 로 태깅되고, DoF 0 을 예측하는 query 가 됩니다. 테스트 2개 추가.
**남은 확인**: 같은 관측에서 `generate_action_tokens` 의 토큰과 `predict_action` 이 고른 bin 이 같은지 (이제 같아야 정상).

### 7-3. mass_other 는 "템플릿" 이 아님

`mass_other` = 템플릿 + **이전 action 토큰 + 자기 자신**. 발표자료(2026-09-29)의 "template 40.5%" 는 정확히는 "템플릿 + action 토큰 40.5%" 입니다. 템플릿만 보려면 `spans.template` 질량을 별도 컬럼으로 추가해야 합니다.

### 7-4. R_vnorm 은 Kobayashi 식의 단순화

원 논문은 $\lVert \alpha_{kj} W_O v_j \rVert$ (출력 투영 포함), 우리는 $\alpha_{kj} \lVert v_j \rVert$. 헤드마다 $W_O$ 가 다르므로 헤드 간 비교에서 차이가 날 수 있습니다. "Kobayashi 지표" 가 아니라 "value-norm 가중" 이라고 쓰세요.

### 7-5. 대조군이 한 가지 배치뿐

`seed=0` 고정 → 한 task 안의 모든 스텝이 같은 visual 위치 \|L\| 개를 차단합니다. 그 위치가 우연히 배경이면 KL_ctrl 이 체계적으로 작게 나옵니다. 또 대조군은 **흩어진** visual 토큰이고 L 은 **연속된** 텍스트라 구조가 다릅니다. 시드 여러 개 + 연속 블록 대조군 + 템플릿 대조군 추가 권장.

### 7-6. 비율의 평균

§3-6. V+L 질량이 ~0 인 헤드도 비율 하나로 동등 기여 → 노이즈 헤드가 평균을 흔들 수 있습니다. 질량 가중 평균(질량을 먼저 합산한 뒤 비율)을 병행 보고하면 방어가 됩니다.

### 7-7. 전 층 동시 knockout

32층 전부에서 끊으므로 "어느 층에서 언어가 쓰이는가" 는 알 수 없고, 입력 분포에서 크게 벗어난 상태의 반응일 수 있습니다. 층 구간 knockout(`layers=`) 으로 보완.

---

## 8. 참고 값 — 수정 전 (Stage 3 재수집, 2026-09-29, 364행)

| 지표 | 평균 ± sd | 범위 |
|---|---|---|
| R_raw | 0.417 ± 0.042 | 0.309 – 0.509 |
| uniform_baseline | 0.066 ± 0.008 | 0.055 – 0.076 |
| R_norm | 0.845 ± 0.026 | 0.768 – 0.897 |
| R_vnorm | 0.396 ± 0.042 | 0.284 – 0.491 |
| KL_lang_knockout | 8.38 ± 2.99 | 1.16 – 19.25 |
| KL_vis_knockout | 7.95 ± 2.78 | 1.69 – 16.57 |
| KL_control_knockout | 0.223 ± 0.225 | 0.002 – 1.27 |
| causal_ratio | 0.511 ± 0.100 | 0.177 – 0.747 |

**§7-1, §7-2 수정 전 코드로 계산한 값입니다.** 수정 후 재수집 결과와의 비교표를 `07_results.md` 에 남길 것.
