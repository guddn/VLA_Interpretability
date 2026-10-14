# 모듈 레퍼런스 — 각 파일이 왜 있고 무엇을 하는가

작성 2026-08-25 · **전면 개정 2026-10-14** (Q 행·29871 수정, 그리퍼 변환, 치환·ICBench·경로 모듈 반영)
대상: `vlamod/` 13개 모듈 + `scripts/` 01–10 + `setup/` 01–08 + `tests/test_core.py`(93개)

> 실험은 스크립트 번호로만 부릅니다. 연구 질문과 각 스크립트가 맡은 역할은 `docs/00_research_plan.md` §5.

---

## 0. 전체 데이터 흐름

```
configs/default.yaml ─▶ device.py (apply_env → 환경변수 / apply_overrides → GPU·모델 인자)
                              │
LIBERO 환경                   ▼                              OpenVLA 모델
  env_libero.py ──이미지──▶ preprocess.py(06·08 변형) ──▶ model_loader.py (eager 강제, --gpus 분할)
  icbench.py (10 모순 지시·판정)                                │
                                                 token_index.py  ← 구간 실측 + 29871 덧붙이기
                                                               │      S|V×256|T|L|T|A×7,  Q = A − 1
                                                     capture.py  ← 2-pass: generate → teacher-forced forward
                                                               │      (attention·value norm·logit 을 Q 행에서)
                      ┌──────────────────────┬─────────────────┼──────────────────────┐
                      ▼                      ▼                 ▼                      ▼
                 metrics.py            intervene.py       substitute.py          pipeline.py
              (관찰: R 3종·질량,     (knockout: Q 행→대상   (0층 내용 치환:        (정책 행동 + 그리퍼 변환,
               KL, causal_ratio)      경로만 차단 = 직접)    우회 경로까지 = 총)     한 스텝 = tidy row)
                                                               │
                                    scripts/NN ──▶ outputs/NN_스크립트명/ (paths.py) ──▶ viz.py
```

**설계 원칙 4가지** — 코드 전체가 이걸 따릅니다.

1. **하드코딩 금지, 실측.** "visual 256개" 같은 문서 속 숫자를 믿지 않고 매번 측정합니다 (`probe_visual_span`).
2. **attention 은 증거가 아니라 검정 대상.** 주 증거는 인과 개입(KL)과 행동이고, attention 은 그것과 비교되는 피검자입니다.
3. **틀리면 조용히 틀리지 말고 즉시 죽을 것.** 가정이 깨지면 `raise` (Q 행 검사, 0층 정렬 검사, 지시문-장면 표 대조 등).
4. **분석 입력 = 정책 입력.** 분석 경로와 실제 롤아웃 경로가 같은 프롬프트·같은 토큰(29871 포함)을 씁니다 (`TI.prepare_inputs`).

---

## 1. `vlamod/device.py` — 환경변수와 GPU 인자

| 함수 | 하는 일 |
|---|---|
| `apply_env(cfg, args)` | config 의 `env:` 섹션으로 `HF_HOME`·`MUJOCO_GL`·`PYOPENGL_PLATFORM`·`CUDA_DEVICE_ORDER` 설정. **transformers/huggingface_hub/libero import 전에** 불러야 함 |
| `apply_overrides(mcfg, args)` | `--gpu` / `--gpus` / `--device` / `--model` / `--unnorm-key` / `--headroom-gb` / `--allow-cpu-offload` 반영. `--gpu` 와 `--device` 동시 사용, `CUDA_VISIBLE_DEVICES` 혼용을 감지해 에러 |
| `resolve_egl_device()` / `probe_egl_devices()` | 렌더링(EGL) 번호를 CUDA 번호와 **별개로** 결정. EGL 이 1개뿐이면 0 |
| `report_gpu()` | 로드 직후 GPU 이름·사용량 출력, 여유가 적으면 경고 |

**왜 필요한가.** HF 캐시 경로는 `huggingface_hub` import 시점에 확정되므로 순서가 바뀌면 조용히 무시됩니다.
GPU 번호는 nvidia-smi(PCI 순서) / CUDA 기본(FASTEST_FIRST) / EGL 세 체계가 다르고, `CUDA_DEVICE_ORDER=PCI_BUS_ID` 로 앞의 둘을 맞춥니다.

---

## 2. `vlamod/model_loader.py` — 모델 로딩과 백본 접근

| 요소 | 하는 일 |
|---|---|
| `load_openvla()` | HF 에서 로드. `attn_implementation != "eager"` 면 **ValueError**. `gpus` 가 2개 이상이면 층 분할 로드 (각 GPU 현재 여유 − headroom 으로 상한 계산, CPU 오프로드는 기본 금지) |
| `LoadedVLA` | model / processor / device / dtype 을 묶고 `language_model`·`decoder_layers`·`n_layers`·`n_heads`·`tokenizer` 노출 |
| `find_language_model()` | Prismatic 래퍼 안의 `LlamaForCausalLM` 탐색 (버전마다 경로가 다름) |
| `action_bin_token_ids()` / `action_logit_slice()` | action bin 256개 vocab id 구간 / 전체 logits 에서 그 구간만 |

- **eager 강제가 이 파일의 존재 이유.** flash/sdpa 로 로드하면 `output_attentions=True` 가 에러 없이 무시됩니다.
- **KL 을 256-bin 으로 좁히는 이유.** 전체 vocab(32000)으로 재면 무관한 토큰이 분포를 희석합니다.
- bin 오프셋은 OpenVLA `ActionTokenizer` 규약을 옮긴 것 — 01 이 생성 토큰이 범위 안에 드는지 검사합니다 (실측 통과).
- 메모리: 가중치 약 15GB. A4000(16GB) 단독 로드는 **됩니다**(09 실측: 프로세스 15.1GB, 여유 1.6GB) — 긴 롤아웃에서 OOM 위험이 있어 경고가 뜹니다.

---

## 3. `vlamod/token_index.py` — 토큰 구간 인덱싱 ★ 가장 중요

**이 파일이 틀리면 그 뒤의 모든 숫자가 무의미해집니다.**

| 요소 | 하는 일 |
|---|---|
| `PROMPT_TEMPLATE` | `"In: What action should the robot take to {instruction}?\nOut:"` |
| `build_prompt()` | 프롬프트 + **instruction 문자 구간**. 지시문은 소문자화·끝 마침표 제거 |
| `ACTION_PREFIX_TOKEN_ID = 29871` / `append_action_prefix()` | OpenVLA `predict_action` 이 입력 끝에 붙이는 빈 토큰 `▁`. **`input_ids` 와 `attention_mask` 를 함께** 늘림 (한쪽만 늘리면 291 vs 290 에러). 이미 있으면 그대로 |
| `prepare_inputs()` | processor → device → 29871 — **분석·정책 공통 입력** |
| `probe_visual_span()` | 이미지만 다른 두 입력의 embedding 층 차이로 visual 구간 실측 (연속이 아니면 에러) |
| `token_char_offsets()` | 토큰별 문자 오프셋. 덧붙인 29871 은 `(len, len)` 빈 구간으로 채움 |
| `build_spans()` | 텍스트 인덱스 → 멀티모달 절대 위치, 구간 분해. `allow_empty_language=True` 는 04 의 empty 조건 전용 |
| `pretty_print_spans()` | 육안 검증용 출력 (A 범위와 Q 범위 포함) |

**실측 구간 (spatial task 0)**

```
S 0 | V 1–256 | T 257–265 | L 266–285 (20개) | T 286–290 (290 = '▁' 29871) | A 291–297
Q = 290–296  (= A − 1, action 토큰을 "예측하는" 자리)          시퀀스 길이 298
```

- **Q 를 따로 두는 이유.** autoregressive LM 에서 위치 p 의 출력은 p+1 번째 토큰을 예측합니다.
  attention 측정·knockout·logit 은 **모두 Q 행**을 씁니다. 2026-10-06 이전에는 A 행을 써서 DoF 0 의 KL 이 항상 0이었습니다 (`docs/06_metrics.md` §7-1).
- **instruction 만 L 로 잡는 이유.** 템플릿 문구는 모든 샘플에 동일해 지시 *내용*과 무관합니다 (`instruction_only=True`).
- **sink(BOS) 를 따로 빼는 이유.** IGAR 가 sink 를 원인으로 지목했고, BOS 가 attention 의 약 42% 를 먹습니다. V 에 넣으면 비전 비중이 부풀려집니다.
- **앞 공백 문제.** fast tokenizer offset 은 앞 공백을 포함해서 `▁pick` 이 L 에서 빠지는 버그가 실제로 있었습니다 → 공백을 벗긴 실질 구간으로 겹침 판정. 지시문 단독 토큰화 개수와 ±1 넘게 다르면 경고.

---

## 4. `vlamod/capture.py` — attention / value / logit 캡처

**2-pass.** `generate(output_attentions=True)` 는 step 마다 shape 이 다르고 버전에 따라 None 이 나오므로:

```
pass 1: generate_action_tokens()      greedy 로 action token 7개
pass 2: teacher_forced_capture()      [prompt + action] 전체를 한 번에 forward
```

| 필드 (`Capture`) | shape | 내용 |
|---|---|---|
| `attn` | `[L, H, 7, T]` | **Q 행만** 잘라 저장 |
| `vnorm` | `[L, H, T]` | `v_proj` 출력 hook → head 별 ‖v_j‖ |
| `action_logits` | `[7, 256]` | `out.logits[0][Q]` 의 action bin 구간 |
| `action_ids` | `[7]` | 생성(또는 강제)한 action 토큰 |

- `check_query_rows()` — Q 행이 `[seq_len−8, …, seq_len−2]` 와 정확히 같은지 검사. A 를 넘기면 즉시 에러.
- `attn_mask_fn` — 개입 주입점. knockout(`intervene`)·치환(`substitute`)·기록기(`Layer0Recorder`) 모두 이 인자로 hook 을 붙입니다.
  **관찰과 개입의 코드 경로가 같아야** 비교가 공정합니다.
- 04·06·09 는 모든 조건에서 **같은 action 토큰(base)으로 teacher-forcing** (`prefix_mode=shared`) → 조건 간 KL 을 같은 규약으로 비교.

---

## 5. `vlamod/metrics.py` — 지표 계산 (순수 함수, GPU 불필요)

| 지표 | 정의 | 용도 |
|---|---|---|
| `R_raw` | `M_L / (M_L + M_V)` | IVAR 재현. **단독 주장 금지** |
| `R_norm` | `(M_L/\|L\|) / (M_L/\|L\| + M_V/\|V\|)` | 토큰 개수 비대칭 제거 |
| `R_vnorm` | `A·‖v‖` 로 가중한 R_raw | Kobayashi+ 2020 (단순화판, `06_metrics.md` §7-4) |
| `uniform_baseline` | `\|L\|/(\|L\|+\|V\|)` | R_raw 의 기본값 (spatial 실측 약 0.055–0.076) |
| `mass_v/l/sink/other` | 각 구간 질량, 합 = 1 | `mass_other` 는 **템플릿 + action 토큰** |
| `action_kl()` | KL(P‖Q), 7 DoF 평균 (`reduction="none"` 이면 DoF 별) | 개입 전후 행동 분포 거리 |
| `causal_ratio()` | `KL_L/(KL_L+KL_V)` | 인과 비율 — **개입 방식에 따라 값이 달라짐**(09) |
| `decode_actions()` | argmax bin | 실제 행동 변화(`argmax_changed_*`, `n_identical_dof`) |

- **비율 지표 해석의 함정** (실측으로 확인됨): R_raw 상승은 Text 질량 증가일 수도, Vision 질량 감소일 수도 있습니다 → `mass_l`·`mass_v` 를 함께 보세요.
  attention 비율 대 인과 비율 상관은 knockout +0.35 / 치환 −0.17 로 부호가 바뀌었습니다(09) — 비율 대 비율 주장은 피하고 **KL_L 과의 관계**로 보고합니다.

---

## 6. `vlamod/intervene.py` — attention knockout (직접 효과)

| 요소 | 하는 일 |
|---|---|
| `apply_block(mask, q, k)` | 4D additive mask 의 (q행, k열)을 −inf. 순수 함수 → 테스트 가능 |
| `AttentionKnockout(query, key, layers=None)` | 각 decoder layer 에 `forward_pre_hook(with_kwargs=True)` 로 mask 수정. `layers` 로 층 구간 지정 가능 (**CLI 미노출**) |
| `knockout_language` / `knockout_vision` | Q → L / Q → V 차단 |
| `knockout_random_control(seed=0)` | **\|L\| 개의 무작위 visual 토큰** 차단 — "토큰 몇 개 지우면 원래 바뀐다"를 걸러내는 대조군 |

- **pre-softmax 마스킹** → 나머지 열이 자동 재정규화. "그 토큰이 없었다면"에 가깝습니다.
- ⚠ **직접 효과만 잽니다.** Q 행만 막으므로 다른 위치를 거친 우회 경로가 열려 있습니다.
  L 차단은 템플릿 286–289(4개)를, V 차단은 T 257–265 / **L 266–285** / T 286–289(33개)를 거쳐 정보가 Q 로 갈 수 있습니다.
  → 이미지 의존을 과소평가 (09: 치환/knockout 비율 V 1.58 vs L 1.17). 총 효과는 `substitute.py` 로.
- `lang_vs_control = KL_L / KL_ctrl` 은 분모가 0 에 가까워 불안정 → `lang_minus_control`·`causal_ratio_adj` 를 쓰세요.
- transformers 4.40.x 의 4D mask kwarg 규약에 의존. 다르면 즉시 에러.

---

## 7. `vlamod/substitute.py` — 0층 임베딩 치환 (총 효과) — 09 용

OpenVLA 는 `[BOS | projector(패치) | 텍스트 임베딩]` 을 이어 Llama 의 `inputs_embeds` 로 넘기고, Llama 는 위치 임베딩을 더하지 않습니다(RoPE).
따라서 **0번 decoder layer 입력 = 멀티모달 임베딩** — 여기서 위치를 바꾸면 이후 모든 층·위치가 바뀐 내용을 봅니다.

| 요소 | 하는 일 |
|---|---|
| `replace_positions()` | hidden `[B,T,D]` 의 지정 위치를 벡터로 바꾼 **복사본** (dtype 유지, 범위 밖이면 에러) |
| `EmbeddingSubstitution(positions, vectors)` | 0층 pre-hook. 위치 인자/키워드 인자 둘 다 처리 |
| `Layer0Recorder(stop=False)` / `record_layer0()` | 0층 입력 기록. `stop=True` 면 기록 후 나머지 31층을 건너뜀 (V 평균 계산용) |
| `check_alignment()` | **G1**: 0층 입력의 V 가 아닌 모든 위치 == `embed_tokens(텍스트 id)` → 소거법으로 V 위치 확정 |
| `language_token_ids()` / `language_mean()` | suite 지시문들의 L 토큰 임베딩 **빈도 가중 평균** |
| `rescale_to_norm()` | 평균 방향 + 원래 평균 크기 (`mean_rescaled`) |
| `spatial_location_span()` / `overlapping_positions()` | spatial 지시문의 위치구 구간 → L_loc / L_shared |

- **실제 토큰으로 치환하지 않는 이유.** 0층에서 같은 길이의 다른 지시문 임베딩 = 04 swapped, 다른 장면 패치 = 06 other_scene 과 **수학적으로 같습니다**.
- 검증: transformers 4.40.1 실제 `LlamaDecoderLayer` 소형 모델로 항등 치환 logit 차이 0.0, hook = 수동 `inputs_embeds` 치환 확인. OpenVLA 실측 G1–G3 통과.
- ⚠ 평균 장면과 현재 장면의 거리가 클수록 V 치환 KL 이 커지는 교란 가능 (09 에서 시간과 ρ 0.57).

---

## 8. `vlamod/pipeline.py` — 정책 행동과 한 스텝 분석

| 함수 | 하는 일 |
|---|---|
| `libero_env_action()` | OpenVLA 그리퍼 [0,1](1=열기) → LIBERO [−1,+1](−1=열기): `2x−1` → sign → 부호 반전. **빠지면 성공률 0%** |
| `policy_action(env_convention=True)` | `prepare_inputs` 로 만든 입력으로 `predict_action` → 그리퍼 변환. env 진행 전용 |
| `analyze_step()` | 한 스텝: 구간 분해 → base 캡처(R 3종, 질량, DoF 별 R_norm) → knockout 3종 → KL·파생 지표 → tidy row + `_per_layer` |
| `counterfactual_instruction()` | 이름과 달리 문자열을 다시 토큰화하는 **[L1] 개입** — 04 는 자체 구현을 써서 **중복** (정리 대상) |

`analyze_step` 의 주요 컬럼: `R_raw/R_norm/R_vnorm`, `uniform_baseline`, `mass_sink/other`, `KL_{lang,vis,control}_knockout`,
`causal_ratio`, `lang_minus_control`, `vis_minus_control`, `causal_ratio_adj`, `argmax_changed_{lang,vis}`, `R_norm_step0..6`. 정의는 `docs/06_metrics.md`.

---

## 9. `vlamod/preprocess.py` — 이미지 변형

| 요소 | 쓰는 곳 | 내용 |
|---|---|---|
| `VARIANTS` / `apply_variant()` / `center_crop_resize()` | 08 | 상하/180° 회전 × center crop 조합. **crop 파라미터는 기억에 근거한 재현** → 08 이 경험적으로 비교 (결과: `current` 와 `rot180_crop` 모두 성공, n=1) |
| `hflip()` | 06 | 좌우 반전 |
| `patch_shuffle(block=32, seed)` | 06 | 타일을 섞음 (항등 순열 회피, 블록이 안 나눠떨어지면 ValueError) — 색 분포 유지, 공간 구조 파괴 |
| `noise_image(seed)` | 06 | 균등 난수 이미지 |

---

## 10. `vlamod/env_libero.py` — LIBERO 얇은 래퍼

| 함수 | 하는 일 |
|---|---|
| `make_task()` | benchmark → task → bddl → `OffScreenRenderEnv` |
| `suite_instructions()` | env 를 만들지 않고 suite 전체 지시문 (09 의 L 평균용) |
| `reset_to()` | 에피소드 번호 → LIBERO 초기 상태 (50개 순환) |
| `obs_to_image(flip=True)` | `agentview_image` → PIL. `flip=True` 는 **`[::-1, ::-1]` = 180° 회전** (OpenVLA LIBERO 평가와 동일) |
| `step_noop()` | 시작 직후 물리 안정화 no-op 10스텝 |

- 방향이 틀리면 vision 분석 전체가 무효 → 01 `--libero` 가 PNG 저장. **좌우는 눈으로 못 잡음** → 성공률이 근거 (13/15).
- instruction 속성: `task.language` 우선, 없으면 `task_description`.

---

## 11. `vlamod/icbench.py` — ICBench 방식 모순 지시 (10 용)

| 요소 | 하는 일 |
|---|---|
| `SPATIAL_SCENES` | task → (목표 그릇 위치구, 방해 그릇 위치구). **LIBERO bddl `(:init)` 에서 옮김** |
| `spatial_variants()` | 원문 → normal / V1 blue bowl / V2 green plate / V3 / V4 under the plate / V5 그릇 없는 위치 / R 방해 그릇 위치. 원문이 표와 다르면 에러 |
| `empty_location()` | 두 그릇과 기준 물체가 겹치지 않는 위치구 (V5) |
| `on_plate()` / `object_pos()` | LIBERO 내부 `_eval_predicate(["on", obj, "plate_1"])`·객체 위치 |
| `classify()` / `first_divergence()` | 결과 범주(target / distractor / wrong_lift / abstain) / 정상 궤적과 말단 위치 2cm 이상 벌어진 첫 스텝 |

- 원 ICBench 지시문은 **미공개** → 논문 분류를 따라 만든 것. 논문 수치와 직접 비교 불가.
- 색 선택: 라메킨이 흰 그릇처럼 보이고 접시 테두리가 붉어서 white·red 는 쓰지 않음.
- **task 7 / 9 는 배치가 같고 목표만 반대** — 언어 필수 쌍 (R 조건이 곧 서로의 원문).

---

## 12. `vlamod/paths.py` — 출력 경로 규약

`outputs/<NN_스크립트명>/[<tag>/]파일`. `out_dir()`·`out_path()` 로 쓰고, 다른 스크립트 출력은 `find_input()` 으로 읽습니다
(새 위치 → 없으면 정리 전 `outputs/<파일명>` 을 **경고와 함께**). 경로를 바꿀 때는 이 파일만 고칩니다.

## 13. `vlamod/viz.py` — 그림

`plot_layerwise()`(층별 R + baseline), `plot_ratio_vs_causal()`(H2 산점도), `plot_conditions()`(조건별 막대).
색은 시리즈 이름에 고정, 이중 y축 금지, 축·제목은 영문(서버 한글 폰트 없음), 3색은 색맹 판별 검증 완료.

---

## 14. `scripts/` — 번호별 역할

| # | 스크립트 | GPU | 무엇을 하는가 | 출력 (`outputs/…`) |
|---|---|---|---|---|
| 01 | `01_smoke_forward.py` | ✓ | 구조 검증: 로딩·visual span·구간 분해(육안)·action bin 범위·Q 행 argmax == 생성 토큰 7/7·행 합 1·지표 | `01_smoke_forward/` |
| 02 | `02_run_analysis.py` | ✓ | 롤아웃하며 stride 마다 `analyze_step` (R + knockout KL), 에피소드 성공 기록 | `02_run_analysis/` |
| 03 | `03_correlation.py` | — | R 3종 ↔ causal / KL_L 상관 (t-검정 부분은 부적절, 교체 예정) | `03_correlation/` |
| 04 | `04_counterfactual.py` | ✓ | 지시 조건 valid / paraphrase / swapped / empty | `04_counterfactual/` |
| 05 | `05_plots.py` | — | 02·04 결과 그림 4장 | `05_plots/` |
| 06 | `06_image_ablation.py` | ✓ | 이미지 조건 valid / other_scene / hflip / patch_shuffle / noise (2-pass: 프레임 수집 → 분석) | `06_image_ablation/` |
| 07 | `07_dump_model_input.py` | ✓ | `pixel_values` 를 백본별 PNG 로 복원 | `07_dump_model_input/` |
| 08 | `08_policy_sanity.py` | ✓ | 이미지 변형 × 그리퍼 변환 on/off 성공 비교 | (콘솔) |
| 09 | `09_embedding_substitution.py` | ✓ | 같은 프레임에서 knockout 3 + 평균 치환 3 + rescaled 2 + 위치구 구간 2, 통과 기준 G1–G4 | `09_embedding_substitution/` |
| 10 | `10_icbench_rollout.py` | ✓ | 모순 지시 7종 롤아웃 (행동만), `--dry-run` 은 지시문 표만 (GPU 불필요) | `10_icbench_rollout/` |
| 11 | (예정) | ✓ | 10 의 행동 재생 → 같은 프레임에 정상·모순 지시 → 가짜 성공 예측 | — |

공통 인자: `--gpu` / `--gpus` / `--device`, `--model`, `--unnorm-key`, `--hf-home`, `--config`, `--egl-device`. `--tag` 는 02·04·06·09·10.
**순서**: 01 통과 → 08 로 성공 1회 확인 → 나머지. 궤적이 GPU 종류에 따라 달라지므로 비교할 실험은 같은 종류 GPU 로.

## 15. `setup/`

| 파일 | 역할 |
|---|---|
| `01_env.sh` / `02_libero.sh` | conda env + torch(cu118/cu121) + OpenVLA / LIBERO + 버전 되돌리기 |
| `03_download_ckpt.py` | 체크포인트(~15GB), `env.hf_home` 사용 |
| `04_verify.py` | 설치 상태 점검 (EGL 실렌더링 포함) → `outputs/setup_04_verify/` |
| `05_diag_render.py` | 렌더링 실패 지점 단계별 특정 → `outputs/setup_05_diag_render/` |
| `06_gpu_map.py` / `07_egl_probe.py` / `08_diag_robosuite.py` | GPU 번호 체계 대조 / EGL 디바이스·벤더 / robosuite↔mujoco 정합성 |

## 16. `tests/test_core.py` — GPU 없이 도는 93개

| 그룹 | 검증 |
|---|---|
| metrics · KL · causal_ratio | 균등 attention → R_raw = baseline, R_norm = 0.5 / 질량 합 1 / KL 성질 |
| intervene | 대상 셀만 −inf, 원본 불변, softmax 후 재정규화 |
| token_index | 프롬프트 오프셋, L·템플릿 분리, 앞 공백 토큰 포함·`?` 제외, 빈 L 에러, **29871 덧붙이기·mask**, **Q 행 검사**, empty 허용 조건 |
| device · apply_env · multi-GPU | 인자 파싱·충돌 감지, EGL 규칙, HF_HOME 우선순위, `--gpus` 파싱, headroom |
| preprocess (06) | hflip 항등성, patch_shuffle 픽셀 보존·재현성·블록 검사, noise 결정성 |
| paths | 번호 폴더, tag 하위 폴더, 옛 위치 폴백 경고 |
| substitute (09) | 치환 위치·복사·dtype, rescale, **G1 정렬**, 위치구 분리, hook(위치/키워드 인자), 기록기 중단 |
| icbench (10) | 10개 task 변형 형식, V5 위치 규칙, R = 방해 그릇 위치구, 원문 불일치 에러, 결과 범주·갈라짐 시점 |

---

## 17. 파일별 "여기가 틀리면 무엇이 무너지는가"

| 파일 | 틀렸을 때의 증상 | 방어 장치 |
|---|---|---|
| `model_loader` | `attentions=None`, 조용한 실패 | 로드 시 eager 강제 |
| `token_index` | 모든 숫자가 무의미 | visual span 실측 + 육안 검증 + 빈 L 에러 + 토큰 수 대조 경고 |
| `capture` | 한 칸 밀린 행 (DoF 0 KL = 0) | `check_query_rows` + 01 의 argmax 7/7 검사 |
| `metrics` | 비율이 산수 결과일 뿐 | `uniform_baseline`·질량 동반 보고 |
| `intervene` | 개입이 안 걸렸는데 걸린 줄 앎 / 직접 효과를 총 효과로 오해 | mask 차원 검사 + 무작위 대조군 / 09 와 비교 |
| `substitute` | 엉뚱한 위치를 치환 | G1 정렬(매 프레임) + G2 항등 치환 KL ≈ 0 |
| `pipeline` | 성공률 0% (그리퍼 규약) / 분석 입력 ≠ 정책 입력 | `libero_env_action`, `prepare_inputs` 공용 |
| `env_libero` | 이미지 방향 오류 → vision 분석 무효 | 01 PNG + 성공률 |
| `icbench` | 장면과 안 맞는 '모순' 지시 | 원문-장면 표 대조 에러 + `--dry-run` 검토 |
