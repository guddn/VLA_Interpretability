# 인과 개입 설계 — **어디에 개입하는가**로 분류

관찰(attention 비율)만으로는 반박당합니다. Embodied Interpretability(2605.00321)의
"attention 은 task-irrelevant 영역에도 뜬다"가 정확히 그 비판입니다.
그래서 **개입해서 행동 분포가 실제로 바뀌는지**를 봅니다.

이 문서는 개입을 **파이프라인의 어느 지점을 건드리는가**로 분류합니다.
같은 "언어 제거"라도 어디서 제거하느냐에 따라 **주장할 수 있는 내용이 다릅니다.**

---

## 개입 지점 지도

```
픽셀                     [L0] 이미지 입력        ✅ 06_image_ablation
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
                             [L2] 토큰 임베딩 치환  ✅ 09_embedding_substitution
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

## [L0] 이미지 입력 — ✅ `scripts/06_image_ablation.py` (2026-10-14 추가)

**무엇을 바꾸나**: 모델에 넣기 직전의 이미지(`obs_to_image` 결과, 180° 회전 완료).
**지시문은 항상 원래 task 의 것**을 그대로 씁니다. [L1] 의 정확한 거울상입니다.

| | [L1] 04_counterfactual | [L0] 06_image_ablation |
|---|---|---|
| 고정 | 이미지 | 지시문 |
| 바꾸는 것 | 지시문 | 이미지 |
| 묻는 것 | 지시문 **내용**에 얼마나 의존하나 | 이미지 **내용**에 얼마나 의존하나 |

### 조건 5개

| 조건 | 넣는 이미지 | 무엇을 잴 수 있나 | [L1] 의 짝 | OOD 위험 |
|---|---|---|---|---|
| `valid` | 원래 이미지 | 기준 | `valid` | 없음 |
| **`other_scene`** | 같은 episode 번호·같은 시점 t 의 **다른 task 장면** | 장면 내용(물체 배치)에 얼마나 의존하나. **핵심 조건** | **`swapped`** | 낮음 (실제 장면) |
| `hflip` | 좌우 반전 | 왼쪽/오른쪽 공간 관계에 반응하나 | (엄밀한 짝 없음) | 중간 |
| `patch_shuffle` | 32×32 타일을 무작위로 섞음 | 색·밝기 분포는 유지, **공간 구조만 파괴** | — | 중간 |
| `noise` | 균등 난수 이미지 | 시각 정보 전체를 없앴을 때 | `empty` | **높음** |

- `other_scene` 짝은 `--tasks` 순서에서 다음 task (0←1, 1←2, …, 4←0).
- `patch_shuffle` / `noise` 의 난수 seed 는 장면마다 고정 (`task*10000 + ep*1000 + t`) → 재실행하면 같은 이미지.
- 예시 이미지(episode 0, 첫 시점)를 `outputs/06_image_ablation/examples/` 에 저장 → 변형이 의도대로 됐는지 육안 확인.

### 진행 방식 (2단계)

1. **롤아웃 패스**: 각 (task, episode) 를 정책으로 굴리며 `--steps`(기본 10, 25, 40) 시점의 이미지만 저장. 분석은 하지 않음.
2. **분석 패스**: 저장한 이미지로 조건별 캡처. 다른 task 의 같은 시점 이미지를 바로 꺼내 `other_scene` 을 만듦.

### 측정

[L1] 과 **같은 지표, 같은 규약**입니다.

- `KL_vs_valid`, `n_identical_dof`, `R_raw / R_norm / R_vnorm`, `mass_sink`
- teacher-forcing prefix 는 valid 의 action token 을 공유 (`--prefix-mode shared`) → 02 의 knockout, 04 와 같은 규약
- 출력: `outputs/06_image_ablation/image_ablation_<suite>.csv`. 04 결과가 있으면 실행 끝에 **"지시문 swapped KL vs 이미지 other_scene KL"** 을 나란히 출력

### 이 지점의 장점

모델 구조를 전혀 건드리지 않습니다. 아키텍처가 달라도 그대로 적용되므로
**다른 VLA 나 실물 로봇으로 옮길 때 그대로 재사용**됩니다 ([L1] 과 같은 장점).

### 해석할 때 주의할 점

1. **`other_scene` 은 두 가지가 동시에 바뀝니다** — 물체 배치 + 로봇 팔 자세(다른 궤적의 같은 시점).
   KL 이 커도 "배치 때문"인지 "자세 때문"인지 분리 불가.
   → 보완: **같은 task, 다른 시점 t** 의 이미지를 넣는 조건(자세만 다름)을 추가하면 분리 가능 (미구현).
2. **`hflip`·`noise` 는 학습 분포 밖일 수 있습니다** — [L1] 의 `empty` 처럼 단독 근거로 쓰지 말 것.
   `noise` 의 KL 이 크다는 것은 "이미지를 쓴다"가 아니라 "이상한 입력이라 망가졌다"일 수 있음.
3. **GPU 를 04 와 맞추세요** — 정책 궤적은 GPU 종류에 따라 달라집니다(bf16 반올림).
   04 와 같은 GPU 로 돌리면 두 스크립트의 장면이 (task, episode, t) 단위로 같아져
   **장면별로 짝지어** 지시문 개입과 이미지 개입을 비교할 수 있습니다. 다르면 평균끼리만 비교.
4. **`object_mask`(목표 물체만 가리기)는 아직 없습니다** — 물체 위치(segmentation)가 필요해서 보류.

### 기대되는 해석 틀

| 결과 | 의미 |
|---|---|
| `other_scene` KL ≫ `swapped` KL | 행동이 지시문보다 장면 내용에 훨씬 크게 의존 |
| `other_scene` KL ≈ `swapped` KL | 내용 수준에서 두 모달리티 의존이 비슷 ([L3] 의 causal_ratio ≈ 0.5 와 일치하는지 확인) |
| `patch_shuffle` KL ≪ `other_scene` KL | 저수준 통계(색·밝기)만으로도 상당 부분 결정 |
| `hflip` KL 이 큼 | 좌우 공간 관계에 민감 — LIBERO-Spatial 과제 특성과 일치 |
| 조건이 바뀌어도 R 이 거의 그대로 | [L1] 에 이어 **attention 비율이 내용 변화를 반영하지 못함** (H2) |

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

**H1 예측**: `valid` ↔ `swapped` 사이에서 **R 이 거의 안 움직인다** → "비율이 지시 내용·언어 수요에 반응하지 않는다"

**실측 (2026-10-13, A4000, task 0–4 × 2 ep × 3 시점 = 30장면)**: swapped KL 4.97 vs paraphrase 0.29 (30/30 장면, 중앙 19배), empty 8.31,
**R_norm 변화 valid − swapped ≈ +0.004** (대부분 \|L\| 길이 효과). → **행동은 지시 내용에 크게 반응하는데 비율은 불변** —
비율은 '얼마나 쓰는가' 의 창이 아님 (H2 쪽 증거이기도 함).
⚠ swapped 는 장면마다 성격이 다름: task 0 은 **방해 그릇을 가리키는 실행 가능한 명령**, task 1–4 는 **그릇이 없는 위치**. task 별 KL 3.8–6.8 로 뚜렷한 차이는 없음 (n=6).

> ### 벤치마크 한계 — 반드시 대비할 것
> LIBERO 는 **장면이 task 를 거의 결정**합니다. 낮은 언어 의존이 나와도
> "모델 결함이 아니라 벤치마크가 언어를 필요로 하지 않기 때문" 이라고 반박됩니다.
> `--ambiguous-tasks` 로 **이미지만으로 과제를 구분할 수 없는 task** 를 지정해야 이 반박을 막을 수 있습니다.
> LIBERO-Spatial 에서는 **task 7 / 9** 가 그 쌍입니다 (bddl 확인: 배치가 같고 목표 그릇만 반대 — 스토브 위 / 캐비닛 위).
> 나머지 task 는 배치가 과제를 1:1 로 결정 → 이것이 원인 가설 A(데이터 지름길) 자체 (`docs/00_research_plan.md` §3).


### [L1] 확장 — ICBench 방식 모순 지시 (`scripts/10_icbench_rollout.py`, A단계, 2026-10-14)

**질문**: 모순 지시를 받은 에피소드 초반의 내부 지표가, 그 에피소드가 **가짜 성공**(지시를 무시하고 원래 과제 완수)으로
끝날지 **실제 실패**로 끝날지 **예측하는가**. 가짜 성공만 따로 보면 "언어 영향이 작다"는 결론이 선택 기준 자체라 순환이므로,
반드시 실제 실패와 비교합니다.

- 근거 논문: 2603.06001 (IGAR) 의 ICBench. **지시문 목록 미공개** → 논문의 V1–V4 분류를 따라 직접 만듦 (`vlamod/icbench.py`)
- 장면 정보는 LIBERO bddl 에서 확인: spatial 10개 task 모두 같은 물체 7종, **검은 그릇 2개**, 성공 판정은 `(On akita_black_bowl_1 plate_1)`
  (다른 그릇을 올리면 실패). **모든 장면에서 방해 그릇이 다른 task 의 위치구 자리에 있음**
- 변형: `V1` blue bowl / `V2` green plate / `V3` 둘 다 / `V4` under the plate / `V5` 그릇이 없는 위치구 / **`R` 방해 그릇의 위치구**(만족 가능한 다른 명령)
  - "white"·"red" 는 쓰지 않음: 라메킨이 흰 그릇처럼 보이고 접시 테두리가 붉어 모순이 아닐 수 있음
  - `R` 은 학습 분포 안의 정상 명령이라 "best effort 였다" 반론이 통하지 않음 — 가장 깨끗한 조건
- 결과 범주: `target`(가짜 성공) / `distractor`(방해 그릇을 접시에) / `wrong_lift` / `abstain`, 정상 궤적과 갈라진 스텝 `t_star`
- **정상 지시로 성공한 초기 상태만** 비교에 씀 (원래 실력 부족으로 인한 실패와 분리)
- B단계(11, 예정): 저장한 행동을 시뮬레이터에서 재생해 프레임 복원 → 정상/변형 지시를 같은 프레임에 넣어 R·KL·바뀐 단어 토큰 attention 비교,
  가짜 성공 vs 실제 실패 예측력(AUC, 에피소드 bootstrap)

!! 실제 실패가 거의 안 나오면(OFT 는 spatial 에서 SR 92–98%) 예측 비교는 표본 부족 → 파일럿으로 먼저 확인.

**task 0 파일럿 결과와 2×2 보완 (2026-10-14).** V1–V4 는 가짜 성공 8–9/9(무시), V5 0/9·R 2/9(위치구는 따름).
그런데 무시된 단어(blue, green, under)는 모두 **spatial 학습 지시문에 없던 단어**, 따른 단어(stove, ramekin)는 **있던 단어**라
"낯선 단어라서 무시"와 "학습 중 고정된 슬롯이라서 무시"가 완전히 겹침 → 같은 슬롯에 반대쪽 단어를 넣은 조건을 추가:

| 슬롯 | spatial 학습에 있던 단어 | 없던 단어 |
|---|---|---|
| 고정 — 그릇 수식어 | `A_seen` wooden bowl | `V1` blue bowl |
| 고정 — 놓을 곳 | `T_seen` on the cookie box (**실행 가능**, `alt_target` 판정) | `V4` under / `V2` green plate |
| 변하는 — 위치구 | `V5` on the stove, `R` | `L_unseen` next to the sink (LIBERO 전체 0회) |

예측: 낯선 단어 가설이면 '없던 단어' 열만 무시, 고정 슬롯 가설이면 '고정' 행만 무시. 둘 다 일부 맞으면 칸마다 섞임 → 11 의 내부 분석이 판정.
실행: `--variants normal A_seen T_seen L_unseen --tag word2x2`, 요약: `--summarize <기존 CSV> <word2x2 CSV>` (normal 재현성도 검사).
⚠ "없던 단어"는 **spatial 미세조정 데이터 기준**(Llama·OXE 사전학습에는 있을 수 있음). wooden 은 'wooden cabinet' 문맥에서만 나옴.
토큰 분할은 서버에서 `--dry-run --show-tokens` 로 확인.

---

## [L2] 토큰 임베딩 치환 — ✅ `scripts/09_embedding_substitution.py` (2026-10-14 추가)

**무엇을 바꾸나**: 토큰이 벡터가 된 뒤(0층 입력), 지정한 위치의 벡터를 **기준값(baseline)** 으로 교체.
위치(RoPE)·시퀀스 길이·attention 구조는 그대로입니다. 구현: `vlamod/substitute.py`.

### 왜 필요한가 — knockout 은 "직접 효과"만 잽니다

[L3] knockout 이 막는 것은 **Q 행(290–296) → 대상 key** 뿐입니다. 다른 위치를 거쳐 가는 우회 경로는 열려 있습니다.

| knockout | 열려 있는 우회 경로 | 우회 위치 수 (task 0) |
|---|---|---|
| L 차단 | L → 템플릿 286–289 → Q | 4 |
| V 차단 | V → 템플릿 257–265 / **L 266–285** / 템플릿 286–289 → Q | **33** |

그래서 02 의 `causal_ratio` 는 **직접 효과끼리의 비교**이고, 우회 기회가 V 쪽에 8배 넘게 많습니다
(→ 이미지 기여 과소평가 가능성. 아래 "알려진 문제" 6번).
치환은 그 위치의 **내용 자체**를 0층에서 바꾸므로 이후 모든 층·모든 위치가 바뀐 내용을 봅니다 → **총 효과**.
09 는 같은 프레임에서 knockout 과 치환을 **둘 다** 계산해 행 단위로 비교합니다.

### 왜 0층 pre-hook 인가

OpenVLA(Prismatic)는 `[BOS 임베딩 | projector(패치) | 나머지 텍스트 임베딩]` 을 이어 Llama 의 `inputs_embeds` 로 넘기고,
Llama 는 위치 임베딩을 더하지 않습니다(RoPE 는 attention 안). 따라서 **0번 decoder layer 의 입력 = 멀티모달 임베딩**이고,
여기에 hook 을 걸면 텍스트·이미지를 같은 절대 위치(`spans.*`)로 다룰 수 있습니다. Prismatic 내부 속성 이름에 의존하지 않습니다.

### 기준값

| 기준값 | 정의 | 채택 |
|---|---|---|
| **mean** (주 조건) | **L**: suite 전체(spatial 10개) 지시문의 L 토큰 임베딩 **빈도 가중 평균** — "L 자리에 올 임베딩의 기댓값" (벡터 1개)<br>**V**: 수집한 모든 프레임의 0층 V 입력(= projector 출력)을 **패치 위치별** 평균 (256개) | ✅ |
| **mean_rescaled** | mean 의 방향, 크기는 치환되는 원래 임베딩들의 평균 L2 크기 | ✅ L·V |
| 실제 토큰(resample) | 같은 길이의 다른 지시문 / 다른 장면 | ❌ 0층에서는 **04 swapped·06 other_scene 과 수학적으로 동일** — 새 정보 없음 |
| zero | 0 벡터 | ❌ 분포 밖 (검증계획에서도 금지) |
| Gaussian noise | ROME 식 3σ | ❌ Zhang & Nanda(2309.16042) 가 불안정하다고 보고 |

mean_rescaled 를 넣은 이유: 평균 벡터는 원래 토큰보다 **크기가 작습니다**(평균하면 줄어듦). "크기가 작아서 분포 밖이 됐다"는 반박에 대비합니다.
실행 시 `‖mean‖ / 평균‖원래‖` 비율을 출력하고 CSV 에도 남깁니다(`norm_*`).

!! **mean 치환이 재는 것은 "언어 전체"가 아닙니다.** 평균 벡터에도 bowl·plate 같은 suite 공통 의미가 섞여 있으므로
"suite 평균에서 벗어나는, **이 지시문만의 정보**"를 잽니다. V 도 "평균 장면 대비 **이 장면만의 정보**"입니다.
knockout(경로 자체 차단)과 측정 대상이 다르다는 점을 결과 해석에 반드시 적으세요.

### 조건 (한 프레임당 forward 최대 12회)

| 조건 | 바꾸는 위치 | 바꾸는 값 |
|---|---|---|
| base | — | — |
| knockout L / V / ctrl | (02 와 동일) | Q 행 attention 차단 |
| `sub` L | `spans.language` (20) | L mean |
| `sub` V | `spans.visual` (256) | V mean (위치별) |
| `sub` ctrl | knockout 대조군과 **같은** 무작위 \|L\|개 V 위치 (seed 0) | 그 위치의 V mean |
| `sub_rescaled` L / V | 위와 같음 | mean_rescaled |
| `Lloc` / `Lshared` (spatial 전용) | L 중 **위치구** / 나머지 | L mean |

**구간 치환의 근거**: spatial 지시문은 모두 `pick up the black bowl [위치구] and place it on the plate` 입니다.
**task 를 구분하는 정보는 위치구에만** 있습니다. 위치구 치환이 크면 "어느 그릇인지" 정보를 실제로 씁니다.
반대로 나머지 치환이 지배적이면 지시문을 "형식"으로 쓰는 쪽입니다. `--no-segments` 로 끕니다.

프레임 수집: 06 처럼 롤아웃을 먼저 끝까지 돌려 프레임을 모은 뒤(V 평균에 전 프레임 필요), 분석합니다.
기본값 task 0–4 × 3 에피소드, `--stride 10`, `--max-steps 220` (약 200 프레임).
정책·규칙이 02 와 같으므로 **02 와 같은 GPU 로 돌리면 (task, ep, t) 프레임이 02 와 같아집니다.**

### 통과 기준 (gates) — 못 넘으면 결과를 해석하지 않음

| | 내용 | 실패 시 |
|---|---|---|
| **G1 정렬** | 매 프레임, 0층 입력의 **V 가 아닌 모든 위치** == `embed_tokens(텍스트 id)` (소거법으로 V 위치 확정) | 즉시 중단 |
| **G2 항등** | 앞 `--gate-frames`(3) 프레임, L·V 를 원래 0층 입력으로 "치환" → KL ≈ 0 (허용 max(1e-6, 2×G3)) | 즉시 중단 |
| **G3 바닥값** | 같은 프레임 base 재실행 KL (같은 GPU 재현성) | 기록 |
| **G4 대조군** | `KL_lang_sub > KL_control_sub` 인 행 비율 | 50% 미만이면 경고 |

결과는 `gates_<suite>.txt` 에 남습니다. hook 은 transformers 4.40.1 의 실제 `LlamaDecoderLayer` 로 소형 모델 통합 검사를 했습니다
(항등 치환 logit 차이 0.0, hook 치환 = 수동 `inputs_embeds` 치환, 치환 위치 이전 logit 불변, knockout 과 동시 부착 가능).

### 해석 규칙 — 결과를 보기 전에 고정

1. **`ratio_L_sub_ko`, `ratio_V_sub_ko`** (= KL_sub / KL_knockout, 행 단위) 의 중앙값 + **에피소드 cluster bootstrap 95% CI**
   - < 1: knockout 효과의 일부는 attention 재정규화로 인한 분포 밖 교란
   - \> 1: 우회 경로가 정보를 나름 (총 효과 > 직접 효과)
   - **V 쪽 비율이 L 쪽보다 뚜렷이 크면** 위 직접/총 효과 비대칭이 실제로 작동한다는 증거
2. **`causal_ratio_sub` vs `causal_ratio_ko`**: 평균 차이(CI) + 에피소드 내 Spearman ρ.
   크게 다르면 02 의 "언어:이미지 ≈ 반반" 은 **개입 방식에 의존하는 값**으로 격하해 보고
3. **구간 치환**: `KL_Lloc_sub` vs `KL_Lshared_sub` (토큰 수 7 대 12 내외라 크기 효과가 섞임 — 토큰 수 함께 보고)
4. **`R_raw_under_Lsub`**: 내용을 평균으로 바꿔도 attention 이 그 자리로 계속 가면(R_raw 유지) "attention 은 내용이 아니라 **자리**를 따라간다" — H2 보조 증거

수치 임계값(예: ρ ≥ 0.5)은 임의라서 두지 않았습니다. 방향과 CI 로만 판단합니다.

### 실측 결과 (2026-10-14, A4000, task 0–4 × 3 ep, stride 10 → 185행, 성공 13/15)

통과 기준 G1–G4 모두 통과 (G2·G3 KL = 0, G4 100% / 100%).

| | knockout (직접) | 치환 (총) | 치환 / knockout (행 중앙값, 95% CI) |
|---|---|---|---|
| L | 7.62 | 8.75 | 1.17 [1.09, 1.23] |
| V | 6.77 | **11.28** | **1.58** [1.46, 1.81] |
| 대조군 | 0.15 | 0.28 | — |
| causal_ratio | 0.522 | 0.453 | 차이 −0.08 [−0.11, −0.06] |

- 규칙 1: 둘 다 > 1 (우회 경로 기여), **V 쪽이 더 큼** → knockout 은 이미지 의존을 과소평가
- 규칙 2: 행 단위 일치도 낮음 (에피소드 내 ρ 중앙값 0.07). 시간 추세가 반대 (knockout +0.27 / 치환 −0.35)
  → **R_raw ↔ causal_ratio 상관이 knockout +0.35 [0.20, 0.47] / 치환 −0.17 [−0.32, −0.02] 로 부호 반전**.
  두 방법 모두에서 버티는 것은 **R_raw ↔ KL_L** (+0.42 / +0.30) 뿐 → 비율 대 비율 결론은 주장 불가
- 규칙 3: L_loc 5.65 vs L_shared 7.64, L_loc 가 큰 행 21%. 위치구 토큰 수가 task 별 3–9개, KL 비가산(합 13.3 > 전체 8.75) → 해석 보류
- 규칙 4: L 치환 시 R_raw 0.416 → 0.361 (94% 행에서 감소), V 치환 시 0.485 → attention 은 대부분 자리를 따라가고 내용에는 일부 반응
- mean_rescaled ≈ mean (L 8.68, V 10.87) → "평균 벡터가 작아서 OOD" 반박은 막힘
- ⚠ V 치환 KL 이 시간에 따라 커짐 (에피소드 내 ρ 0.57, knockout 은 0.09). 원래 V 크기도 시간과 ρ 0.48
  → '평균 장면과의 거리' 교란 가능. ‖V − V_mean‖ 통제 필요 (미구현)
- 제안(미구현): **완전 knockout** — L·V 뒤의 모든 위치가 L·V 를 못 보게 차단. 평균 벡터 없이 총 효과를 재므로 거리 교란이 없음

### 해석할 때 주의할 점 (공격 지점)

- mean 도 실제 토큰은 아닙니다. 특히 V 의 위치별 평균은 흐릿한 "평균 장면"입니다. 분포 밖 위험이 **줄었을 뿐 없어지지 않았습니다**. rescaled·G4 는 부분 방어입니다
- V 평균에 분석 대상 프레임 자신이 약 1/N(≈1/200) 섞입니다. 무시할 수준이지만 엄밀히는 누출입니다
- L 평균은 suite 10개 지시문 기준입니다. suite 목록을 못 읽으면 실행한 task 의 지시문만 쓰고 로그에 `!!` 로 표시합니다
- ISS(2605.00321)의 "distribution-matched baseline" 정확한 정의는 **원문 미확인**입니다. "ISS 재현"이 아니라 **"ISS 계열 mean ablation"** 으로 쓸 것
- 0층 치환만 구현했습니다. 층별 activation patching(같은 위치의 l층 잔차를 l층 평균으로)은 미구현
- `vlamod/pipeline.py` 의 `counterfactual_instruction()` 은 이름과 달리 문자열을 다시 토큰화하므로 [L1] 입니다 (04 와 중복, 정리 대상)

### 실행

```bash
python scripts/09_embedding_substitution.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 --episodes 3
#   --no-rescaled / --no-segments 로 forward 를 줄일 수 있음, --gate-frames 로 G2·G3 프레임 수 조정
```

출력 `outputs/09_embedding_substitution/`: `embed_sub_<suite>.csv`, `baselines_<suite>.pt`, `gates_<suite>.txt`. 컬럼 정의는 `docs/06_metrics.md` §5-2.

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
| 관찰 (`R`) | ✅ `outputs/02_run_analysis/perlayer_*.npz` 에 층별 저장 |
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
| [L0] 이미지 입력 | 픽셀 | ✅ | 5 | **높음** | 낮음(`other_scene`)~높음(`noise`) |
| [L1] 언어 입력 | 지시문 문자열 | ✅ | 4 | **높음** | 낮음(`swapped`)~높음(`empty`) |
| [L2] 임베딩 치환 | 토큰 벡터 (0층) | ✅ | 7 (+knockout 3 동시) | 중간 | 중간 (mean 은 실제 토큰 아님) |
| [L3] attention mask | 정보 경로 | ✅ | 3 | 낮음(Transformer 전용) | **없음** |
| [L4] 층·헤드 | 개입 범위 | ⚠ 미노출 | 0 | 낮음 | 없음 |

**[L3]의 장점**: 입력이 분포 안에 그대로 있어 OOD 걱정이 없습니다. "이상한 입력이라
망가졌다"는 반박이 원천적으로 불가능합니다.

**[L3]의 단점**: Transformer attention 구조에 묶여 있습니다. 실물 로봇이나 다른
아키텍처로 못 옮깁니다. **그래서 [L0]/[L1]이 필요합니다** (둘 다 구현 완료) — 그쪽은 모델 내부를
안 건드리므로 어디든 적용됩니다.

---

## 알려진 문제 (해결 대기)

1. ~~**[L0] 이미지 개입 0개**~~ → 2026-10-14 `06_image_ablation.py` 로 해결 (조건 5개). 남은 것: `other_scene` 의 배치/자세 분리, `object_mask`
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
6. **[L3] knockout 은 직접 효과만 잼 (2026-10-14 발견)** — Q 행만 차단하므로 우회 경로가 열려 있고,
   우회 위치가 L 4개 vs V 33개로 비대칭. 02 의 `causal_ratio` 는 이미지 기여를 과소평가했을 수 있음.
   → [L2] 09 로 확인 (2026-10-14): 치환/knockout 비 V 1.58 > L 1.17 — **비대칭이 실제로 작동**. 02 의 causal_ratio 는 개입 방식 의존

---

## 예상 반박과 방어

| 반박 | 방어 |
|---|---|
| "attention 을 막은 게 정보를 지운 것과 같나" | pre-softmax 마스킹이라 재정규화됨. 잔차 스트림 경유 우회 가능성은 **인정하고 명시**할 것 |
| "언어 20개 지운 효과가 작은 건 당연" | `knockout_random_control` 이 같은 개수를 지운 바닥값을 줌 |
| "빈 지시는 OOD 아닌가" | 맞음. 그래서 `swapped` 가 주 조건이고 `empty` 는 선행연구 재현용 |
| "LIBERO 는 원래 언어가 필요 없다" | 맞음 — 그것이 원인 가설 A. **task 7/9(언어 필수 쌍)** 와 10 의 R 조건으로 직접 검정 |
| "전층을 한꺼번에 막은 결과 아닌가" | **현재 방어 불가.** [L4] 노출이 필요 |
| "이미지 쪽은 왜 안 했나" | [L0] `06_image_ablation.py` 로 대응. `other_scene` 이 [L1] `swapped` 와 같은 단위의 짝 |
| "knockout 은 우회 경로가 열려 있지 않나" | 맞음 (직접 효과). [L2] 치환으로 총 효과를 같은 프레임에서 재고 비율을 보고 |
| "평균 벡터 치환도 분포 밖 아닌가" | 일부 맞음. `mean_rescaled`(크기 보정)·G4(무작위 위치 대조군)로 부분 방어, 한계로 명시 |
| "other_scene 은 로봇 자세까지 바뀌지 않나" | 맞음. 같은 task·다른 시점 조건으로 분리 필요 (미구현) — 결론에 명시할 것 |
