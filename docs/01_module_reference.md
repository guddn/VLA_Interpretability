# 모듈 레퍼런스 — 각 파일이 왜 있고 무엇을 하는가

작성일: 2026-08-25
대상: `vlamod/` 8개 모듈 + `scripts/` 5개 + `tests/`

---

## 0. 전체 데이터 흐름

```
LIBERO 환경                    OpenVLA 모델
  env_libero.py  ──이미지──▶  model_loader.py  (eager 로드)
                                    │
                              token_index.py   ← [BOS][V×N][text][A×7] 구간을 실측
                                    │
                              capture.py       ← 2-pass: generate → teacher-forced forward
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
                metrics.py                  intervene.py
             (관찰: R_raw/R_norm/R_vnorm)   (개입: knockout → KL)
                      └─────────────┬─────────────┘
                                    ▼
                              pipeline.py      ← 한 타임스텝 = tidy row 1개
                                    ▼
                        scripts/02 → CSV → scripts/03(H2) · 04(H1) → viz.py
```

**핵심 설계 원칙 3가지** — 코드 전체가 이걸 따릅니다.

1. **하드코딩 금지, 실측.** "visual 토큰 256개" 같은 문서 속 숫자를 믿지 않고 매번 측정합니다.
2. **attention 은 증거가 아니라 검정 대상.** 주 증거는 인과 개입(KL)이고, attention 은 그것과 비교되는 피검자입니다.
3. **틀리면 조용히 틀리지 말고 즉시 죽을 것.** 가정이 깨지면 `assert` / `raise` 로 멈추게 했습니다.

---

## 1. `vlamod/model_loader.py` — 모델 로딩과 백본 접근

### 목적
OpenVLA 를 **attention 을 뽑을 수 있는 상태로** 올리고, 버전마다 다른 내부 구조를 안전하게 찾아냅니다.

### 핵심 요소

| 요소 | 하는 일 |
|---|---|
| `load_openvla()` | HF 에서 로드. `attn_implementation != "eager"` 면 **ValueError 로 거부** |
| `LoadedVLA` (dataclass) | model/processor/device 를 묶고, `decoder_layers`·`n_layers`·`n_heads`·`tokenizer` 를 속성으로 노출 |
| `find_language_model()` | Prismatic 래퍼 안의 `LlamaForCausalLM` 을 이름 후보 → `named_modules()` 순으로 탐색 |
| `action_bin_token_ids()` | action bin 256개에 대응하는 vocab id 구간 반환 |
| `action_logit_slice()` | 전체 vocab logits 에서 action bin 구간만 잘라냄 |

### 왜 이렇게 만들었나

- **`eager` 강제가 이 파일의 존재 이유입니다.** `flash_attention_2` / `sdpa` 로 로드하면
  `output_attentions=True` 가 **에러 없이 무시**되고 `attentions=None` 이 나옵니다.
  나중에 디버깅하면 원인을 찾기 어려우므로 로드 시점에 막습니다.
- 백본 탐색을 하드코딩하지 않은 이유: OpenVLA 는 `language_model` / `llm_backbone.llm` 등
  버전에 따라 경로가 다릅니다. 틀린 경로를 잡으면 엉뚱한 층에 훅을 겁니다.
- `action_logit_slice()` 가 중요한 이유: action token 은 7개 DoF 각각이 **256-bin 분류**입니다.
  전체 vocab(32000) 에 대해 KL 을 재면 무관한 토큰들이 분포를 희석시킵니다.
  256-bin 으로 좁혀야 "행동이 얼마나 바뀌었나"가 깨끗하게 나옵니다.

### 주의
`action_bin_token_ids()` 의 오프셋은 OpenVLA 의 `ActionTokenizer` 규약
(`token_id = vocab_size - bin`) 을 옮긴 것이라 **±1 어긋날 수 있습니다.**
`scripts/01` 이 실제 생성된 토큰이 이 범위 안에 드는지 검사합니다.

---

## 2. `vlamod/token_index.py` — 토큰 구간 인덱싱 ★ 가장 중요

### 목적
멀티모달 시퀀스 `[BOS][visual ×N][프롬프트 텍스트][action ×7]` 에서
**V(비전) / L(언어) / S(sink) / A(action)** 각 구간의 절대 위치를 확정합니다.

**이 파일이 틀리면 그 뒤의 모든 숫자가 무의미해집니다.**

### 핵심 요소

| 함수 | 하는 일 |
|---|---|
| `build_prompt()` | 프롬프트 문자열 + **instruction 이 차지하는 문자 오프셋** 반환 |
| `probe_visual_span()` | **이미지만 다른 두 입력의 embedding 층을 비교**해 visual 구간을 실측 |
| `token_char_offsets()` | 각 텍스트 토큰의 문자 오프셋 (fast tokenizer → 실패 시 수동 복원) |
| `build_spans()` | 텍스트 토큰 인덱스를 멀티모달 절대 위치로 매핑하고 4구간으로 분해 |
| `pretty_print_spans()` | **사람이 눈으로 검증하기 위한 출력** |

### `probe_visual_span()` 의 아이디어

이미지를 바꾸면 visual 토큰 위치의 임베딩만 달라져야 합니다.
`hidden_states[0]`(embedding 층)은 attention 을 아직 거치지 않았으므로
**정확히 visual 위치에서만** 차이가 납니다. 그 연속 블록이 visual 구간입니다.

연속적이지 않으면 `RuntimeError` — 구조 가정이 깨졌다는 신호입니다.

### 왜 instruction 만 L 로 잡는가

프롬프트는 `"In: What action should the robot take to {instruction}?\nOut:"` 입니다.
`"In: What action should the robot take to"` 는 **모든 샘플에 동일**하므로
여기 걸린 attention 은 지시 *내용*과 무관합니다.
이걸 L 에 포함시키면 "언어를 본다"가 "템플릿을 본다"로 오염됩니다.
→ `instruction_only=True`(기본값) 가 이걸 분리하고, 나머지는 `template` 구간으로 갑니다.

### 왜 sink 를 따로 빼는가

`Restoring Linguistic Grounding in VLA`(2603.06001)가 **attention sink** 를
linguistic blindness 의 원인으로 지목했습니다. BOS 등 sink 토큰이 attention 질량을 크게 먹는데,
이를 V 에 포함시키면 **비전 비중이 인위적으로 부풀려집니다.**
→ `sink_positions`(기본 `[0]`)를 별도 집합 S 로 분리하고, `mass_sink` 로 따로 보고합니다.

### 검증 방법
`pretty_print_spans()` 출력에서 **L 태그가 instruction 내용어에만** 붙었는지 눈으로 봅니다.
`In`, `What`, `action` 에 L 이 붙어 있으면 인덱싱이 틀린 것입니다.

---

## 3. `vlamod/capture.py` — attention / value / logit 캡처

### 목적
한 타임스텝에서 **모든 층·헤드의 attention**, **value 벡터 norm**, **action bin logits** 를 뽑습니다.

### 왜 2-pass 인가 (이 파일의 핵심 설계)

`generate(output_attentions=True)` 로 뽑으려 하면:
- step 마다 attention tuple 이 쪼개져 나오고,
- KV cache 때문에 step 0 은 `[B,H,T,T]`, 이후는 `[B,H,1,T]` 로 shape 이 다르고,
- transformers 버전에 따라 `None` 이 나오기도 합니다.

그래서:

```
pass 1: generate_action_tokens()      greedy 로 action token 7개 확보
pass 2: teacher_forced_capture()      [prompt + action] 전체를 한 번에 forward
```

greedy 이므로 두 pass 의 분포는 **동일합니다.** shape 도 `[L, H, 7, T]` 하나로 고정됩니다.

### 각 산출물

| 필드 | shape | 용도 |
|---|---|---|
| `attn` | `[L, H, 7, T]` | action query 행만 잘라 저장 (전체 T×T 를 들고 있지 않음) |
| `vnorm` | `[L, H, T]` | `v_proj` 출력에 forward hook → head 별 `‖v_j‖` |
| `action_logits` | `[7, 256]` | 위치 `p_k` 를 예측하는 logits 는 index `p_k − 1` |
| `action_ids` | `[7]` | 생성된 토큰 |

### value norm 을 왜 따로 뽑나
Kobayashi et al. (EMNLP 2020) — attention weight `α` 만으로는 실제 기여를 못 봅니다.
실제 전달되는 것은 `α·v` 이므로 `‖α·v‖ = α·‖v‖` 를 써야 합니다.
Llama 는 MHA(GQA 아님)이고 `v` 에는 rotary 가 안 걸리므로 `v_proj` 출력을 그대로 쓰면 됩니다.

### 개입 주입점
`attn_mask_fn` 인자에 `intervene.py` 의 훅을 넣으면
**같은 캡처 경로로 개입 조건도 측정**됩니다. 관찰과 개입의 코드 경로가 같아야 비교가 공정합니다.

---

## 4. `vlamod/metrics.py` — 지표 계산 (순수 함수, GPU 불필요)

### 목적
캡처된 텐서에서 **비율 3종 + KL** 을 계산합니다. 모델 의존성이 없어 단위 테스트가 가능합니다.

### 지표

| 지표 | 정의 | 이 지표가 필요한 이유 |
|---|---|---|
| `R_raw` | `M_L / (M_L + M_V)` | IVAR(2603.06001) **재현용**. 선행연구와 같은 축에서 비교하려면 필요 |
| `R_norm` | `(M_L/\|L\|) / (M_L/\|L\| + M_V/\|V\|)` | **토큰 개수 비대칭 제거** |
| `R_vnorm` | `A(s,j)` → `A(s,j)·‖v_j‖` | attention weight 가 아닌 **실제 전달량** 기준 |
| `uniform_baseline` | `\|L\|/(\|L\|+\|V\|)` | R_raw 의 **기본값**. 이것과 비교해야 의미가 생김 |
| `action_kl()` | `KL(P‖Q)` over 7×256 | 개입 전후 행동 분포 거리 |
| `causal_ratio()` | `KL_L/(KL_L+KL_V)` | **인과 기여 비율 — 주 증거** |

### `R_raw` 단독 사용을 왜 금지하는가

|V| ≈ 256, |L| ≈ 5~15 입니다. attention 은 행 합이 1이므로,
**완전 균등 분포만 가정해도** `R_raw ≈ 15/(15+256) ≈ 0.055` 가 나옵니다.
즉 "vision 이 94% 먹는다"는 발견이 아니라 **산수**입니다.
그래서 `uniform_baseline` 을 항상 같이 반환하고, README 에도 함께 보고하도록 적어 뒀습니다.

### `mass_*` 분해
`mass_v + mass_l + mass_sink + mass_other = 1` 이 성립합니다(단위 테스트로 검증).
`mass_sink` 가 크면 attention sink 가 지배 중이라는 뜻이고,
이 경우 `R_raw`/`R_norm` 은 남은 조각을 나눈 값이므로 해석에 주의해야 합니다.

---

## 5. `vlamod/intervene.py` — 인과 개입 (knockout)

### 목적
"attention 이 이만큼 걸려 있다"를 넘어 **"그 경로를 끊으면 행동이 실제로 바뀌는가"** 를 측정합니다.

### 왜 필요한가
`Embodied Interpretability`(2605.00321)가 "attention score 는 주로 task-irrelevant 영역에
활성화된다"며 attention 계열을 명시적으로 비판했습니다.
관찰만 제시하면 이 비판에 그대로 당합니다. **개입이 주 증거이고 attention 은 피검자입니다.**

### 구현

| 요소 | 하는 일 |
|---|---|
| `apply_block(mask, q, k)` | 4D additive mask 의 (q행, k열) 교차점을 `-inf` 로. **순수 함수 → 테스트 가능** |
| `AttentionKnockout` | 각 decoder layer 에 `forward_pre_hook(with_kwargs=True)` 를 걸어 mask 수정 |
| `knockout_language()` | action query → **언어 토큰** 경로 차단 |
| `knockout_vision()` | action query → **비전 토큰** 경로 차단 |
| `knockout_random_control()` | **언어와 같은 개수**의 visual 토큰을 무작위 차단 ← 대조군 |

### pre-softmax 마스킹인 이유
softmax **이전**에 `-inf` 를 더하면 나머지 열이 자동으로 재정규화됩니다.
softmax 이후에 0으로 만들고 다시 normalize 하는 것과 **수치가 다르고**,
전자가 "그 토큰이 애초에 없었다면"에 더 가깝습니다.

### `knockout_random_control` 이 이 코드에서 가장 중요한 대조군인 이유

"언어를 지웠더니 안 변했다"만으로는 부족합니다. 두 가설이 구분되지 않기 때문입니다:
- (a) 모델이 언어를 안 쓴다
- (b) 토큰 몇 개 지우는 건 원래 아무 영향이 없다

같은 개수의 visual 토큰을 무작위로 지운 것과 비교해야 (a)와 (b)가 갈립니다.
→ `lang_vs_control = KL_L / KL_random`. **1에 가까우면 언어가 특별하지 않다는 뜻**이고,
이게 오히려 논문에서 가장 강한 형태의 증거가 됩니다.

### 버전 의존성
transformers 4.40.x 는 4D additive causal mask 를 각 layer 에 `attention_mask=` kwarg 로 넘깁니다.
상위 버전은 규약이 바뀌어 훅이 깨집니다. 단 **조용히 틀리지 않고 즉시 에러**가 납니다
(`4D mask 를 기대했는데 ND 가 왔습니다`).

---

## 6. `vlamod/pipeline.py` — 한 타임스텝 = tidy row 1개

### 목적
관찰과 개입을 **한 함수 호출로 묶어** CSV 한 줄로 만듭니다. H2 검정이 이 row 들만으로 됩니다.

### `analyze_step()` 이 하는 일
1. 프롬프트 구성 → 구간 분해
2. 무개입 캡처 → `R_raw`/`R_norm`/`R_vnorm`, `mass_sink`, `mass_other`
3. 개입 3종(language / vision / random control) 캡처 → `KL_*`
4. 파생 지표: `causal_ratio`, `lang_vs_control`, `argmax_changed_*`
5. `_per_layer` 키에 층별 배열 (히트맵용)

`R_norm_step0` ~ `R_norm_step6` 도 넣었습니다 — **7개 DoF 중 어느 것이 언어에 더 의존하는지**
(예: gripper open/close 만 언어를 보는지) 확인용입니다.

### `counterfactual_instruction()`
같은 장면, 지시만 교체하는 ISS 식 개입.
**빈 문자열이 아니라 같은 분포의 다른 지시문**으로 바꿉니다 —
빈 입력은 OOD 라서 "언어를 안 쓴다"가 아니라 "이상한 입력이라 망가졌다"가 되기 때문입니다.
`n_identical_dof == 7` 이면 지시를 완전히 무시한 것입니다.

### `visual_span` 을 인자로 받는 이유
매 스텝 `probe_visual_span()` 을 돌리면 forward 2회가 추가로 듭니다.
한 번 실측한 값을 재사용하도록 **강제**(None 이면 ValueError)했습니다.

---

## 7. `vlamod/env_libero.py` — LIBERO 얇은 래퍼

### 목적
suite/task 를 열고, 초기 상태를 세팅하고, 관측을 PIL 이미지로 바꿉니다.

| 함수 | 하는 일 |
|---|---|
| `make_task()` | benchmark → task → bddl → `OffScreenRenderEnv` |
| `reset_to()` | 에피소드별 초기 상태 |
| `obs_to_image()` | `agentview_image` → PIL. **`flip=True` 가 기본** |
| `step_noop()` | 시작 직후 물리 안정화용 no-op 스텝 |

### 왜 얇게 만들었나
LIBERO 의 API 는 버전에 따라 달라지고, 여기서 추상화를 두껍게 하면
버전이 바뀔 때마다 고쳐야 할 코드가 늘어납니다. 필요한 4개 함수만 노출했습니다.

### 검증 필요 지점 2개
1. **이미지 상하 반전.** LIBERO 의 `agentview_image` 는 뒤집혀 나옵니다.
   방향이 틀리면 **vision attention 분석 전체가 무의미**해집니다.
   `scripts/01 --libero` 가 `outputs/smoke_libero_view.png` 를 저장하니 열어서 확인하세요.
2. **instruction 속성명.** `task.language` vs `task.task_description` — 버전마다 다릅니다.
   둘 다 없으면 `RuntimeError`.

---

## 8. `vlamod/viz.py` — 그림

### 목적
논문/발표에 그대로 쓸 수 있는 그림 3종.

| 함수 | 형태 | 용도 |
|---|---|---|
| `plot_layerwise()` | 3계열 라인 | 층별 비율 + `uniform_baseline` 기준선 |
| `plot_ratio_vs_causal()` | 산점도 | **H2 핵심 그림** — attention 비율 vs 인과 기여 |
| `plot_conditions()` | 막대 | **H1 핵심 그림** — 지시 조건 4종 비교 |

### 강제한 규칙
- 색은 **시리즈 이름에 고정**(dict). 시리즈가 빠져도 색이 재배정되지 않습니다.
- **이중 y축 금지.** 스케일이 다르면 그림을 나눕니다.
- 3계열까지만 색으로 구분.
- 2계열 이상이면 legend + **직접 라벨**을 함께 (색만으로 식별하지 않도록).
- 색상 3종은 색맹 판별 검증을 통과했습니다 (OKLab CVD ΔE 9.2, 정상시야 ΔE 24.0).
  색을 바꾸면 다시 검증하세요.
- 축·제목은 **영문**입니다. 서버에 한글 폰트가 없으면 네모로 깨지기 때문입니다.

---

## 9. `scripts/` — 실행 순서

| 스크립트 | Phase | 무엇을 확정하는가 |
|---|---|---|
| `01_smoke_forward.py` | 0~1 | **구조 검증 7단계.** eager 로드 / visual span 실측 / 구간 분해 육안 검증 / action token bin 범위 / attention 행 합 = 1 / 지표 계산 |
| `02_run_analysis.py` | 2~3 | 롤아웃 돌며 비율 + knockout KL 수집 → `analysis_*.csv`, `perlayer_*.npz` |
| `03_correlation.py` | H2 | Spearman/Pearson + **대조군 t-검정**(`lang_vs_control` 이 1과 다른가) |
| `04_counterfactual.py` | H1 | valid / paraphrase / swapped / empty 4조건 비교 |
| `05_plots.py` | — | 위 결과를 그림으로 |

### `01` 을 먼저 통과시켜야 하는 이유
02~04 는 전부 `01` 이 확인한 가정 위에서 돕니다.
특히 **`[4] 토큰 구간 분해` 출력의 육안 검증**은 자동화할 수 없습니다.

### `04` 의 미완성 부분
`--ambiguous-tasks` 에 **후보 물체가 둘 이상인 task id** 를 직접 지정해야 합니다.
지정하지 않으면 경고가 뜹니다. 이 조건 없이는
**"LIBERO 는 원래 언어가 필요 없는 벤치마크 아니냐"** 는 반박을 막을 수 없습니다.
시뮬레이터를 눈으로 보고 고르는 수밖에 없습니다.

---

## 10. `tests/test_core.py` — GPU 없이 도는 검증 14개

### 목적
모델 없이도 **계산 로직의 실수**를 잡습니다. 서버에 붙기 전에 노트북에서 돌릴 수 있습니다.

| 그룹 | 검증 내용 |
|---|---|
| metrics (5) | 균등 attention → `R_raw = \|L\|/(\|L\|+\|V\|)`, `R_norm = 0.5` / 전량 언어 → 1.0 / value norm 이 크면 `R_vnorm > R_raw` / **질량 분해 합 = 1** |
| KL (2) | `KL(p‖p) = 0`, 양수성, 비대칭성 |
| causal_ratio (1) | 경계값 |
| intervene (3) | 대상 셀만 `-inf` / 원본 불변(clone) / 범위 밖 무시 / **softmax 후 해당 열 0 + 나머지 재정규화** |
| token_index (3) | 프롬프트 오프셋이 instruction 을 정확히 가리키는지 / 언어·템플릿 분리 / 언어 구간이 비면 에러 |

### 왜 `apply_block` 을 순수 함수로 뺐나
훅 안에 로직이 들어 있으면 모델 없이 테스트할 수 없습니다.
마스크 수정 로직만 분리해서 **가짜 텐서로 검증**할 수 있게 했습니다.

---

## 11. 파일별 "여기가 틀리면 무엇이 무너지는가"

| 파일 | 틀렸을 때의 증상 | 방어 장치 |
|---|---|---|
| `model_loader` | `attentions=None`, 조용한 실패 | 로드 시 `eager` 강제 |
| `token_index` | 모든 숫자가 무의미 | `probe_visual_span` 실측 + `pretty_print_spans` 육안 검증 + L 비면 에러 |
| `capture` | shape 불일치, action 위치 벗어남 | `RuntimeError` + 행 합 = 1 검사 |
| `metrics` | 지표가 산수 결과일 뿐 | `uniform_baseline` 동반 보고 |
| `intervene` | 개입이 안 걸렸는데 걸린 줄 앎 | mask 차원 검사 + 랜덤 대조군 |
| `env_libero` | 이미지 상하 반전 → vision 분석 전부 무효 | 스모크 테스트가 PNG 저장 |
