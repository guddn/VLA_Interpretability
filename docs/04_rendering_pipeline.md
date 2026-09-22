# 렌더링 파이프라인 — 픽셀 한 장이 만들어져 모델에 들어가기까지

`outputs/verify_render.png` 같은 이미지가 **어떤 패키지를 거쳐** 생기는지, 그리고
그게 OpenVLA 로 넘어가 **256개 visual token** 이 되기까지의 전 과정.

---

## 0. 가장 중요한 구분 — 두 스택은 분리되어 있습니다

```
[이미지를 만드는 쪽]                        [이미지를 쓰는 쪽]
LIBERO → robosuite → MuJoCo → EGL           OpenVLA (Prismatic VLM)
= 시뮬레이터                                 = 정책 모델
        │                                            ▲
        └──────── numpy array (256×256×3) ───────────┘
                   PIL.Image 로 전달
```

**OpenVLA 는 이미지를 만들지 않습니다.** 받아서 쓸 뿐입니다.
바꿔 말하면, 이 시뮬레이터 자리에 실물 카메라(UR5e + RealSense)를 끼워 넣어도
OpenVLA 쪽 코드는 한 줄도 안 바뀝니다. 나중에 실물로 넘어갈 때 이 경계가 그대로
이전 지점이 됩니다.

---

## 1. 패키지 역할표

### 이미지를 만드는 쪽

| 패키지 | 버전 | 역할 |
|---|---|---|
| `libero` | 소스 설치 | **무엇을** 그릴지: task 목록, 지시문, 물체 배치, 초기 상태 |
| `bddl` | 의존성 | task 정의 파일(`.bddl`)을 파싱 (BEHAVIOR 유래 선언 언어) |
| `robosuite` | **1.4.1** | 로봇/그리퍼/테이블/물체를 하나의 MJCF XML 로 **조립**. 컨트롤러, 관측 규격 |
| `mujoco` | **2.3.0** | 물리 엔진 + 렌더러. XML 컴파일, `mj_step`, `mjr_render` |
| `PyOpenGL` | — | 파이썬에서 EGL/OpenGL 함수를 호출하는 바인딩 |
| libglvnd + `libEGL_mesa` | 시스템 | 화면 없이 OpenGL 컨텍스트를 여는 계층 |
| `numpy` | **1.26.4** | 픽셀 배열 |
| `opencv-python` | 4.x | robosuite 내부 이미지 처리 |
| `gym` | 레거시 | robosuite 의존성. 경고만 뜨고 우리는 안 씀 |

### 이미지를 쓰는 쪽

| 패키지 | 버전 | 역할 |
|---|---|---|
| `torch` | 2.2.0+cu118 | 연산 |
| `transformers` | **4.40.1** | `AutoModelForVision2Seq` 로 OpenVLA 로드, attention 출력 |
| `timm` | 0.9.10 | DINOv2 / SigLIP 비전 백본 |
| `tokenizers` | 0.19.1 | Llama-2 토크나이저 |
| `Pillow` | — | numpy ↔ PIL 변환 |

**버전이 서로 물려 있는 지점**

- `mujoco` 3.x + `robosuite` 1.4.1 → 환경 생성 자체가 실패 (`get_joint_qpos_addr` assert)
- `numpy` 2.x + `torch` 2.2.0 → `RuntimeError: Numpy is not available`
- `transformers` 4.41+ → attention mask 전달 규약이 바뀌어 knockout 훅이 깨짐

---

## 2. 단계별 흐름

### [0] 환경변수 — 라이브러리 import 전에

```python
MUJOCO_GL=egl              # MuJoCo: 렌더 백엔드로 EGL 사용
PYOPENGL_PLATFORM=egl      # PyOpenGL: EGL 플랫폼 플러그인 로드
MUJOCO_EGL_DEVICE_ID=0     # EGL 디바이스 목록에서 몇 번째를 쓸지
```

`vlamod/device.py`의 `apply_env()` / `resolve_egl_device()` 가 설정합니다.
**import 시점에 읽히므로** 반드시 `import libero` 전이어야 합니다.

> `MUJOCO_EGL_DEVICE_ID` 는 **CUDA 번호가 아닙니다.** 이 서버에서 EGL 은
> 디바이스를 1개만 열거하므로 값은 0 뿐입니다.

### [1] 무엇을 그릴지 정하기 — `libero`

```python
bench = benchmark.get_benchmark_dict()["libero_spatial"]()
task  = bench.get_task(0)
#   task.language          지시문 문자열
#   task.bddl_file         장면 정의 파일 이름
#   task.init_states_file  초기 상태 파일 이름
```

`.bddl` 파일에는 어떤 물체가 어디 있고 목표가 무엇인지가 **선언적으로** 적혀 있습니다.
아직 3D 도, 픽셀도 없습니다.

### [2] MJCF XML 조립 — `libero` + `robosuite`

`OffScreenRenderEnv(bddl_file_name=...)` 안에서:

1. `bddl` 이 파일을 파싱 → 물체 목록 + 술어
2. LIBERO 가 자기 에셋(`libero/libero/assets/`: 그릇·접시·캐비닛의 XML + STL 메시 + 텍스처)에서 해당 물체를 꺼냄
3. robosuite 가 **아레나(테이블) + Panda 로봇 + 그리퍼 + 물체**를 하나의 MJCF XML 로 합침
4. 카메라(`agentview`, `robot0_eye_in_hand`)와 조명도 이 XML 안에 정의됨

즉 **로봇 모델은 robosuite 가, 물체는 LIBERO 가** 제공합니다.

### [3] 모델 컴파일 — `mujoco`

```python
mujoco.MjModel.from_xml_string(xml)   # → mjModel (구조·질량·관절)
mujoco.MjData(model)                  # → mjData  (현재 상태: qpos, qvel …)
```

여기서 관절 18개가 확정됩니다. `08_diag_robosuite.py` 가 뽑아준 그 목록입니다.

```
robot0_joint1..7          HINGE   ← Panda 7축
gripper0_finger_joint1,2  SLIDE   ← 그리퍼
akita_black_bowl_1/2, cookies_1, ramekin_1, plate_1   FREE  ← 자유낙하 물체
wooden_cabinet_1_top/middle/bottom_level              SLIDE ← 서랍
flat_stove_1_button                                   HINGE
```

**robosuite 1.4.1 이 이 관절들을 이름으로 찾는 단계에서 mujoco 3.x 와 깨졌습니다.**

### [4] EGL 컨텍스트 — `PyOpenGL` + libglvnd + Mesa

```
robosuite/renderers/context/egl_context.py
  eglQueryDevicesEXT()          → 디바이스 목록 (이 서버는 1개)
  eglGetPlatformDisplayEXT()    → 디스플레이
  eglInitialize / eglCreateContext / eglMakeCurrent
```

libglvnd 가 `/usr/share/glvnd/egl_vendor.d/*.json` 을 보고 구현체를 고릅니다.
이 서버엔 `50_mesa.json` 만 있어 **Mesa 소프트웨어 렌더러**가 잡힙니다.
결과 픽셀은 GPU 렌더링과 동일하고, 속도만 느립니다.

### [5] 초기 상태 복원 — `libero`

```python
init_states = bench.get_task_init_states(0)   # shape (50, 92)
env.set_init_state(init_states[0])
```

50개 에피소드 × 92차원(= `nq + nv + 1`). 이걸 `mjData` 에 통째로 써넣어
**매번 똑같은 장면에서 시작**하게 만듭니다. 재현성의 핵심입니다.

### [6] 물리 시뮬레이션 — `mujoco`

```python
for _ in range(5):
    obs, _, _, _ = env.step(dummy_action)   # 내부적으로 mj_step
```

물체를 테이블 위에 안정시킵니다. 이걸 안 하면 물체가 공중에 떠 있거나
튀는 순간이 찍힙니다.

### [7] 픽셀 만들기 — `mujoco` 렌더러

```
mjv_updateScene   현재 mjData → 장면 그래프(카메라·조명·기하)
mjr_render        OpenGL 로 오프스크린 프레임버퍼에 래스터화
mjr_readPixels    프레임버퍼 → numpy uint8 (256, 256, 3)
```

`obs["agentview_image"]` 가 이 배열입니다.

### [8] 방향 보정 — 우리 코드

```python
# vlamod/env_libero.py
img = obs["agentview_image"]
if flip:
    img = img[::-1, ::-1]        # 180도 회전
```

OpenGL 프레임버퍼는 **아래에서 위로** 쌓입니다. 그래서 그대로 보면 상하가 뒤집혀
있습니다. `img[::-1, ::-1]` 은 OpenVLA 공식 LIBERO 유틸이 쓰는 것과 **같은 연산**입니다.

> **미확인 사항**: `[::-1, ::-1]` 은 상하 + **좌우** 반전입니다. 상하만 필요한지
> 좌우까지 필요한지는 robosuite 의 `macros.IMAGE_CONVENTION` 설정에 달려 있고
> (기본 `"opengl"`), 실행 시 뜨는 `No private macro file found` 경고가 바로 이 파일
> 얘기입니다. **모델 학습 때의 전처리와 일치하기만 하면 됩니다.**
> 판정 방법은 하나뿐입니다 — **태스크 성공률**. 어긋나 있으면 성공률이 무너집니다.
> `libero_spatial` 은 `"between the plate and the ramekin"` 같은 공간 지시문이라
> 좌우가 뒤집히면 조용히 망가집니다. Stage 3 에서 성공률을 반드시 기록하세요.

### [9] 여기서 스택이 바뀝니다 — `transformers` + `timm`

```python
inputs = vla.processor(prompt, image)
```

PrismaticProcessor 내부:

1. 이미지를 **224×224** 로 리사이즈 + 정규화
2. **두 개의 비전 백본에 동시에** 넣음 (`timm`)
   - DINOv2 ViT-L/14
   - SigLIP ViT-SO400M/14
   - 224 ÷ 14 = 16 → **16×16 = 256 패치**
3. 두 백본의 특징을 **채널 방향으로 concat** (패치 수는 그대로 256)
4. projector MLP → Llama-2 임베딩 차원으로 사영
5. 텍스트 시퀀스의 **위치 1** 에 256개를 삽입

그래서 `01_smoke_forward.py [3]` 의 실측이 이렇게 나옵니다:

```
visual span = [1, 257)  →  n_visual = 256
[0] = <s>(BOS),  [1..256] = visual,  [257..] = 텍스트
```

**두 백본을 쓰는데 토큰이 512가 아니라 256인 이유**가 4번(채널 concat)입니다.
이건 우리 분석에 직접 영향을 줍니다 — 한 visual token 은 "DINOv2 패치"도
"SigLIP 패치"도 아닌 **둘이 합쳐진 것**이라, 어느 백본이 기여했는지는
현재 지표로 분리할 수 없습니다.

---

## 3. 우리가 겪은 고장을 이 그림에 얹으면

| 증상 | 어느 단계 | 원인 |
|---|---|---|
| `PYOPENGL_PLATFORM` 없어 멈춤 | [0] | GLX 를 시도하다 디스플레이 대기 |
| `MUJOCO_EGL_DEVICE_ID must be 0..0, got 8` | [4] | EGL 목록 길이 ≠ CUDA 장수 |
| EGL 디바이스 1개뿐 | [4] | `libEGL_nvidia` 미설치 → Mesa 폴백 |
| 메시지 없는 `AssertionError` | [3] | mujoco 3.x ↔ robosuite 1.4.1 |
| `Numpy is not available` | [7]·[9] | numpy 2.x ↔ torch 2.2.0 |
| 검은 프레임 (std≈0) | [7] | 컨텍스트는 생겼지만 렌더 실패 |
| 프레임 상하 반전 | [8] | flip 설정 오류 |
| `attentions=None` | [9] | eager 가 아닌 attention 구현 |

---

## 4. 한 줄 요약

**LIBERO 가 무엇을 그릴지 정하고, robosuite 가 조립하고, MuJoCo 가 물리와
래스터화를 하고, EGL 이 화면 없이 그걸 가능하게 하고, 그 결과 numpy 배열이
OpenVLA 로 넘어가 256개 visual token 이 됩니다.**
OpenVLA 는 이 중 마지막 한 칸만 담당합니다.
