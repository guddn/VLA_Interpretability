# VLA_Interpretability

VLA(OpenVLA)에서 **action token 이 image token 과 language token 을 어떤 비율로 쓰는가**를
관찰(attention)과 인과 개입(knockout + KL) 두 축으로 측정하는 코드.

연구 계획 문서: [`docs/00_research_plan.md`](docs/00_research_plan.md)  ·  분석 패키지: `vlamod/`

## 검증하는 가설

- **H1 (비적응성)** — 언어/비전 비율이 *언어가 실제로 필요한 상황인지*와 무관하게 일정한가?
- **H2 (지표 타당성)** — attention 비율(IVAR류)이 인과 기여(ΔKL)를 예측하는가?

두 가설 모두 **어느 쪽 결과가 나와도 발표거리**가 됩니다.

## 빠른 시작

```bash
bash setup/01_env.sh                       # conda + torch + OpenVLA
bash setup/02_libero.sh                    # LIBERO (MUJOCO_GL=egl 필요)
python setup/03_download_ckpt.py --suite spatial

conda activate vlamod
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

pytest -q tests/test_core.py               # GPU 없이 돌아가는 로직 검증
python scripts/01_smoke_forward.py         # ★ 이걸 먼저 통과시킬 것
python scripts/01_smoke_forward.py --libero --suite spatial --task-id 0

python scripts/02_run_analysis.py --suite spatial --tasks 0 1 2 --episodes 3
python scripts/03_correlation.py --csv outputs/analysis_spatial.csv
python scripts/04_counterfactual.py --suite object --tasks 0 1 2 3
python scripts/05_plots.py --suite spatial
```

## 구조

```
vlamod/
  model_loader.py  OpenVLA 로딩(eager 강제), 백본 탐색, action bin id 구간
  token_index.py   [BOS][visual][text][action] 구간 실측 + 검증 출력
  capture.py       2-pass 캡처: generate → teacher-forced forward
  metrics.py       R_raw / R_norm / R_vnorm, action KL, causal_ratio
  intervene.py     pre-softmax attention knockout (language/vision/random control)
  pipeline.py      한 타임스텝 = 관찰 + 개입 → tidy row
  env_libero.py    LIBERO 얇은 래퍼
  viz.py           그림 (색맹 검증 완료 3색)
scripts/
  01_smoke_forward.py   Phase 0~1  구조 검증
  02_run_analysis.py    Phase 2~3  비율 + knockout KL 수집
  03_correlation.py     H2         상관 + 대조군 t-검정
  04_counterfactual.py  H1         지시 조건 4종 비교
  05_plots.py                      그림
tests/test_core.py      GPU 없이 도는 단위 테스트 (14개)
```

## 지표 정의

action query s, layer l, head h 에 대해:

| 지표 | 정의 | 용도 |
|---|---|---|
| `R_raw` | `M_L / (M_L + M_V)` | IVAR(2603.06001) 재현. **단독 주장 금지** |
| `R_norm` | `(M_L/\|L\|) / (M_L/\|L\| + M_V/\|V\|)` | 토큰 개수 비대칭 제거 |
| `R_vnorm` | `A(s,j)` 대신 `A(s,j)·‖v_j‖` | Kobayashi+2020. 실제 전달되는 벡터 크기 |
| `causal_ratio` | `KL_L / (KL_L + KL_V)` | 인과 기여 비율 (주 증거) |
| `lang_vs_control` | `KL_L / KL_random-visual` | **1에 가까우면 언어가 특별하지 않다는 뜻** |

`uniform_baseline = |L|/(|L|+|V|)` 을 항상 같이 보고하세요.
`R_raw` 는 이 값 근처가 기본값이므로, 그보다 낮아야 "언어를 덜 본다"고 말할 수 있습니다.

## 함정 목록 (전부 실제로 겪게 됩니다)

1. **`attn_implementation="eager"` 필수.** flash_attention_2 / sdpa 는 `output_attentions=True` 를
   조용히 무시합니다. 에러 없이 `attentions=None` 이 나옵니다. `model_loader.py` 가 막아 둡니다.
2. **transformers 4.40.x 고정.** 상위 버전은 attention mask 전달 규약이 바뀌어
   `intervene.py` 의 knockout 훅이 깨집니다.
3. **visual token 개수를 문서에서 베끼지 말 것.** `probe_visual_span()` 이 이미지를 바꿔가며
   embedding 차이로 **실측**합니다. 256이 아니면 경고가 뜹니다.
4. **템플릿 문구를 L 에서 제외할 것.** "In: What action should the robot take to" 는
   모든 샘플에 동일하므로 여기 걸린 attention 은 지시 내용과 무관합니다.
5. **BOS/attention sink 를 V 에 넣지 말 것.** 비전 비중이 인위적으로 부풀려집니다.
6. **knockout 은 pre-softmax 로.** softmax 뒤에 0으로 만들고 재정규화하는 것과 결과가 다릅니다.
7. **치환 baseline 은 빈 문자열이 아니라 다른 지시문으로.** 빈 입력은 OOD 라서
   "언어를 안 쓴다"가 아니라 "이상한 입력이라 망가졌다"가 됩니다.
8. **랜덤 대조군 없이 결론 내지 말 것.** 언어 토큰과 같은 개수의 visual 토큰을 무작위로
   차단한 것과 비교해야 "언어가 특별하다/아니다"를 말할 수 있습니다.
9. **LIBERO 이미지 상하 반전.** `env_libero.obs_to_image(flip=True)` 가 기본이지만
   `outputs/smoke_libero_view.png` 를 열어 **눈으로** 확인하세요.
10. **LIBERO 는 원래 언어가 거의 필요 없는 벤치마크입니다.**
    `04_counterfactual.py --ambiguous-tasks` 로 후보 물체가 둘 이상인 task 를 지정하지 않으면
    "그건 벤치마크 탓"이라는 반박을 막을 수 없습니다.

## 미확인 사항 (직접 검증하세요)

- OpenVLA 체크포인트의 정확한 프롬프트 템플릿 (`token_index.PROMPT_TEMPLATE`)
- action bin ↔ vocab id 오프셋 (`model_loader.action_bin_token_ids`, 01 스크립트가 검사)
- HF 체크포인트 repo 이름 (`setup/03_download_ckpt.py` 의 `CKPTS`)
- LIBERO task 객체의 instruction 속성명 (`task.language` vs `task.task_description`)
- transformers 4.40.1 에서 `LlamaDecoderLayer` 가 `attention_mask` 를 kwargs 로 받는지
  (`intervene.py`; 아니면 즉시 에러가 나므로 조용히 틀리진 않습니다)
