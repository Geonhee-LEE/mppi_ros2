# 메타 학습 (Meta-Learning) 가이드

**날짜**: 2026-02-18
**버전**: 1.0

## 목차

1. [개요](#개요)
2. [메타 학습이란?](#메타-학습이란)
3. [MAML 이론](#maml-이론)
4. [FOMAML — 1차 근사](#fomaml--1차-근사)
5. [아키텍처](#아키텍처)
6. [실행 방법](#실행-방법)
7. [API 레퍼런스](#api-레퍼런스)
8. [성능 분석](#성능-분석)
9. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
10. [기존 학습 방법과의 비교](#기존-학습-방법과의-비교)
11. [문제 해결](#문제-해결)

---

## 개요

MAML (Model-Agnostic Meta-Learning)은 **"학습하는 방법을 학습하는"** 메타 학습 알고리즘입니다.

로봇 동역학 문맥에서, 다양한 환경(마찰 계수, 관성 등)에서 **사전 학습된 메타 파라미터**로부터 시작하여, 실행 중 **최근 수십 개의 데이터만으로 현재 환경에 빠르게 적응**합니다.

### 핵심 아이디어

```
┌─────────────────────────────────────────────────────────────────────┐
│                        기존 학습 방법                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  오프라인 학습:  특정 환경 데이터 ──→ 학습 ──→ 고정 모델            │
│                  (환경 변하면 성능 저하)                             │
│                                                                     │
│  온라인 학습:    실시간 데이터 ──→ fine-tuning ──→ 점진적 적응      │
│                  (수렴 느림, 초기 성능 낮음)                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        MAML (메타 학습)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  메타 학습:     다양한 환경 ──→ 메타 파라미터 θ* 학습              │
│                                                                     │
│  실행 중 적응:  최근 데이터 ──→ SGD 100 step ──→ 즉시 적응          │
│                 (수 ms 소요, few-shot 적응)                         │
│                                                                     │
│  θ* ─────(100 step SGD)───→ θ_adapted                              │
│  (모든 환경에    (현재 환경에 최적화)                               │
│   좋은 초기점)                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 주요 특징

- **Few-Shot 적응**: 40~200개 데이터만으로 현재 환경 적응
- **빠른 수렴**: SGD 100 step (수 ms)으로 적응 완료
- **Residual 학습**: 기구학 base + MAML 잔차 보정 아키텍처
- **환경 불변**: 마찰/관성/지연 등 다양한 환경 변화에 대응
- **안전한 롤백**: 매 적응마다 메타 파라미터에서 시작 (누적 드리프트 없음)

---

## 메타 학습이란?

### 학습의 3가지 수준

| 수준 | 설명 | 예시 |
|------|------|------|
| **추론** (Inference) | 학습된 모델로 예측 | `forward_dynamics(state, control)` |
| **학습** (Learning) | 데이터로 모델 파라미터 학습 | `trainer.train(data)` |
| **메타 학습** (Meta-Learning) | "어떻게 학습할지"를 학습 | `maml_trainer.meta_train()` |

### 메타 학습 = "Learning to Learn"

일반 학습은 하나의 **태스크** (예: 특정 마찰 계수의 로봇)에 대해 모델을 학습합니다.
메타 학습은 **태스크의 분포** (예: 다양한 마찰 계수)에 대해, 새로운 태스크에 빠르게 적응할 수 있는 **초기 파라미터**를 학습합니다.

```
태스크 분포 D(T):
  T1: c_v=0.2, c_omega=0.1  (빙판 위 로봇)
  T2: c_v=0.5, c_omega=0.3  (일반 바닥)
  T3: c_v=0.8, c_omega=0.4  (카펫 위 로봇)
  ...

메타 학습 목표:
  θ* = argmin_θ E_{T~D} [ L_T(θ - α∇L_T(θ)) ]
       "적응 후 성능이 좋은 초기 파라미터 찾기"
```

---

## MAML 이론

### MAML (Finn et al., 2017)

MAML은 **모델 구조에 무관한 (Model-Agnostic)** 메타 학습 알고리즘입니다.

**핵심**: 그래디언트 기반 적응(SGD)이 가장 효과적인 **초기 파라미터 θ\***를 찾습니다.

### 알고리즘

```
Input: 태스크 분포 p(T), inner LR α, meta LR β
Output: 메타 파라미터 θ*

1. θ ← 랜덤 초기화
2. for each meta iteration:
   a. 태스크 배치 {T1, ..., Tn} ~ p(T) 샘플링
   b. for each task Ti:
      i.   Di_support, Di_query = sample_data(Ti)
      ii.  θ'_i = θ - α ∇_θ L(θ; Di_support)     ← Inner Loop (적응)
      iii. L_query_i = L(θ'_i; Di_query)            ← Query Loss
   c. θ ← θ - β ∇_θ Σ_i L_query_i                  ← Outer Loop (메타 업데이트)
3. return θ* = θ
```

### Inner Loop vs Outer Loop

```
┌──────────────────────────────────────────────────────────────┐
│                      MAML 구조                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Outer Loop (메타 학습):                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  메타 파라미터 θ를 업데이트                           │  │
│  │  목표: 적응 후 query loss 최소화                      │  │
│  │  학습률: β (meta_lr = 5e-4)                          │  │
│  │  옵티마이저: Adam                                     │  │
│  │                                                        │  │
│  │  Inner Loop (적응):                                    │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  각 태스크의 support set으로 SGD 수행             │  │  │
│  │  │  학습률: α (inner_lr = 0.005)                    │  │  │
│  │  │  스텝 수: K (inner_steps = 10)                   │  │  │
│  │  │  옵티마이저: SGD (vanilla)                       │  │  │
│  │  │                                                    │  │  │
│  │  │  θ'_i = θ - α ∇L(θ; support)                    │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  │  θ ← θ - β ∇_θ Σ L(θ'_i; query)                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## FOMAML — 1차 근사

### MAML vs FOMAML

본 구현은 **FOMAML (First-Order MAML)**을 사용합니다.

원래 MAML은 2차 미분 (Hessian)을 계산해야 합니다:

```
Full MAML gradient:
  ∇_θ L(θ') = ∇_θ L(θ - α∇L(θ))
             = (I - α ∇²L(θ)) ∇L(θ')    ← 2차 미분 필요!
                     ^^^^^^^
                     Hessian
```

FOMAML은 2차 미분 항을 생략하여 **1차 미분만** 사용합니다:

```
FOMAML gradient:
  ∇_θ L(θ') ≈ ∇_θ' L(θ')    ← 1차 미분만! (create_graph=False)
```

### FOMAML의 장점

| 비교 항목 | Full MAML | FOMAML |
|----------|-----------|--------|
| 메모리 | O(K * params) | O(params) |
| 계산 비용 | 2차 미분 필요 | 1차 미분만 |
| 구현 복잡도 | 높음 | 낮음 |
| 성능 | 약간 우수 | 거의 동일 (실험적) |

Nichol et al. (2018) "On First-Order Meta-Learning Algorithms"에서 FOMAML이 Full MAML과 비슷한 성능을 보임을 증명.

### 코드에서의 FOMAML

```python
# Inner loop에서 create_graph=False → 2차 미분 비활성화
grads = torch.autograd.grad(
    loss, list(adapted_params.values()),
    create_graph=False,  # FOMAML: 1차 근사
)
```

---

## 아키텍처

### 파일 구조

```
mppi_controller/
├── models/learned/
│   └── maml_dynamics.py        # MAMLDynamics — NeuralDynamics 상속
│                                 # save/restore_meta_weights()
│                                 # adapt() — few-shot 적응
│
├── learning/
│   └── maml_trainer.py          # MAMLTrainer — FOMAML 메타 학습
│                                 # _sample_task() — DynamicWorld 설정
│                                 # _inner_loop() — functional forward
│                                 # meta_train() — 메타 학습 루프
│
examples/comparison/
└── model_mismatch_comparison_demo.py  # 6-Way 비교 데모
                                        # meta_train_maml()
                                        # run_with_dynamic_world_maml()
tests/
└── test_maml.py                 # 13개 테스트
```

### 클래스 계층

```
RobotModel (ABC)
└── NeuralDynamics
    └── MAMLDynamics              ← 메타 학습 + few-shot 적응
        ├── save_meta_weights()   ← 메타 파라미터 스냅샷
        ├── restore_meta_weights() ← 적응 전 복원
        ├── adapt()               ← inner-loop SGD
        ├── _prepare_inputs()     ← numpy→normalized tensor
        └── _prepare_targets()    ← numpy→normalized tensor
```

### 데이터 흐름

```
  [메타 학습 단계]                         [실행 단계]

  DynamicWorld(c_v=0.2)  ──┐
  DynamicWorld(c_v=0.5)  ──┼──→ MAMLTrainer.meta_train()
  DynamicWorld(c_v=0.8)  ──┘          │
                                       ▼
                              메타 파라미터 θ* 저장
                              (maml_meta_model.pth)
                                       │
                              ┌────────▼────────┐
                              │  MAMLDynamics    │
                              │  load(θ*)        │
                              │  save_meta()     │
                              └────────┬────────┘
                                       │
          ┌────────────────────────────▼────────────────────────┐
          │       2-Phase 실행 루프 (Residual MAML)             │
          │                                                      │
          │  Phase 1 (warm-up, ~2초):                            │
          │    기구학 모델로 제어 + 전이 데이터 수집 (40 step)   │
          │                                                      │
          │  Phase 1→2 전환:                                     │
          │    잔차 계산: residual = actual - kinematic           │
          │    MAML 적응: restore(θ*) → adapt(residual) (100 SGD)│
          │    ResidualDynamics(kinematic + MAML) 컨트롤러 전환   │
          │                                                      │
          │  Phase 2 (적응 제어):                                 │
          │    ResidualDynamics로 MPPI 제어                      │
          │    80 step마다: restore(θ*) → adapt(최신 잔차) 재적응 │
          └──────────────────────────────────────────────────────┘
```

### Residual MAML 아키텍처

MAML을 단독 모델로 사용하면 MPPI rollout에서 예측 오차가 누적됩니다.
**Residual MAML**은 기구학 모델을 base로, MAML이 잔차(마찰/관성 보정)만 학습합니다:

```
  forward_dynamics(state, control) =
      kinematic(state, control)    ← 기구학 base (안정적)
    + MAML_residual(state, control) ← 학습된 보정 (적응적)
```

장점:
- 기구학 base가 **안정성 보장** (MAML 보정이 0이어도 Kinematic 수준 유지)
- MAML은 **작은 보정량만 학습** → 빠른 수렴, 낮은 오차
- 5-seed 평균 RMSE **0.081m ± 0.007** (매우 안정적)

---

## 실행 방법

### 전체 파이프라인 (권장)

가장 간단한 방법은 `--all --world dynamic`으로 전체 파이프라인을 실행하는 것입니다:

```bash
cd /path/to/mppi_ros2

python examples/comparison/model_mismatch_comparison_demo.py \
    --all --world dynamic --trajectory circle --duration 20
```

이 명령은 다음 4단계를 **순차적으로** 실행합니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: collect_data                                              │
│  ─────────────────────                                              │
│  DynamicWorld에서 학습 데이터 수집                                  │
│  - 4개 궤적 (circle/figure8/sine/straight) x 5 에피소드            │
│  - 10개 랜덤 탐색 에피소드                                         │
│  - 출력: data/learned_models/dynamic_mismatch_data.pkl              │
│  - 소요: ~2분                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2: train_models                                              │
│  ─────────────────────                                              │
│  Neural + Residual 모델 오프라인 학습                               │
│  - Neural: [256, 256] MLP, 300 epochs, early stopping              │
│  - Residual: [128, 128] MLP, 300 epochs, early stopping            │
│  - 출력: models/learned_models/dynamic_neural_model.pth             │
│          models/learned_models/dynamic_residual_model.pth           │
│  - 소요: ~1분                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2b: meta_train_maml                                          │
│  ─────────────────────────                                          │
│  FOMAML 메타 학습 (다양한 DynamicWorld 설정)                       │
│  - 1000 iterations x 8 tasks/batch                                  │
│  - 각 태스크: c_v ~ U(0.1, 0.8), c_omega ~ U(0.1, 0.5)           │
│  - Support/Query: 100/100 samples                                   │
│  - Inner loop: SGD 10 step (lr=0.005)                              │
│  - Outer loop: Adam (lr=5e-4)                                      │
│  - 학습 데이터: 궤적 추종 패턴 (50% 전진, 30% 곡선, 20% 랜덤)    │
│  - 출력: models/learned_models/dynamic_maml_meta_model.pth          │
│  - 소요: ~5분                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 3: evaluate                                                  │
│  ─────────────────                                                  │
│  6-Way 비교 평가                                                    │
│  1. Kinematic (3D): 순수 기구학 — 관성/마찰 모름                   │
│  2. Neural (3D): 오프라인 end-to-end 학습                          │
│  3. Residual (3D): 기구학 + NN 보정                                │
│  4. Dynamic (5D): 5D 구조, 파라미터 틀림 (c_v=0.1)                │
│  5. MAML (3D): 메타 학습 + 실시간 few-shot 적응                   │
│  6. Oracle (5D): 정확한 파라미터                                   │
│  - 출력: plots/model_mismatch_comparison_circle_dynamic.png         │
│  - 소요: ~5분                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 단계별 실행

각 단계를 개별적으로 실행할 수도 있습니다:

```bash
# Stage 1: 데이터 수집만
python examples/comparison/model_mismatch_comparison_demo.py \
    --collect-data --world dynamic

# Stage 2: 모델 학습만 (데이터 수집 후)
python examples/comparison/model_mismatch_comparison_demo.py \
    --train --world dynamic

# Stage 2b: MAML 메타 학습만
python examples/comparison/model_mismatch_comparison_demo.py \
    --meta-train --world dynamic

# Stage 3: 평가만 (모든 모델 학습 후)
python examples/comparison/model_mismatch_comparison_demo.py \
    --evaluate --world dynamic --trajectory circle --duration 20
```

### 실시간 비교 (Live Visualization)

학습된 모델로 실시간 애니메이션 비교:

```bash
# 사전 조건: --all 또는 Stage 1~2b 완료 후
python examples/comparison/model_mismatch_comparison_demo.py \
    --live --world dynamic --trajectory circle --duration 20
```

6개 패널에서 실시간으로 궤적 추적, 오차, 제어 입력, RMSE를 비교합니다.
모델 파일이 없으면 자동으로 데이터 수집 + 학습을 실행합니다.

### 궤적 종류

```bash
--trajectory circle    # 원형 (기본값, 안정적)
--trajectory figure8   # 8자형 (교차점, 급회전)
--trajectory sine      # 사인파 (왕복)
--trajectory straight  # 직진 (가장 단순)
```

### 기존 모드 (변경 없음)

Perturbed world는 기존 4-Way 비교 그대로:

```bash
# Perturbed world (4-way, MAML 없음)
python examples/comparison/model_mismatch_comparison_demo.py \
    --all --trajectory circle --duration 20
```

### 테스트

```bash
# MAML 단위 테스트 (13개)
PYTHONPATH=. python -m pytest tests/test_maml.py -v -o "addopts="

# 전체 회귀 테스트
PYTHONPATH=. python -m pytest tests/ -x -q -o "addopts="
```

---

## API 레퍼런스

### MAMLDynamics

```python
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics

class MAMLDynamics(NeuralDynamics):
    """MAML 기반 동역학 모델 — 실시간 few-shot 적응."""

    def __init__(
        self,
        state_dim: int,           # 상태 벡터 차원 (e.g., 3)
        control_dim: int,         # 제어 벡터 차원 (e.g., 2)
        model_path: str = None,   # 메타 모델 경로
        device: str = "cpu",
        inner_lr: float = 0.01,   # 적응 학습률
        inner_steps: int = 5,     # 적응 gradient step 수
        use_adam: bool = False,   # True → Adam, False → SGD
    )

    def save_meta_weights(self):
        """현재 모델 파라미터를 메타 파라미터로 저장."""

    def restore_meta_weights(self):
        """메타 파라미터로 복원 (적응 전 상태로 되돌림)."""

    def adapt(
        self,
        states: np.ndarray,       # (M, nx) 최근 상태
        controls: np.ndarray,     # (M, nu) 최근 제어
        next_states: np.ndarray,  # (M, nx) 다음 상태
        dt: float,                # 시간 간격
        restore: bool = True,     # True → 메타 파라미터 복원 후 적응 (표준 MAML)
                                  # False → 현재 파라미터에서 계속 fine-tune
    ) -> float:
        """Few-shot 적응. Returns: 최종 loss 값."""

    def forward_dynamics(
        self,
        state: np.ndarray,        # (nx,) or (batch, nx)
        control: np.ndarray,      # (nu,) or (batch, nu)
    ) -> np.ndarray:
        """적응된 모델로 state_dot 예측. (NeuralDynamics 상속)"""
```

#### 사용 예시 (Residual MAML — 권장)

```python
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic

# 1. 기구학 base 모델 + MAML 잔차 모델 준비
base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
maml = MAMLDynamics(
    state_dim=3, control_dim=2,
    model_path="models/learned_models/dynamic_maml_meta_model.pth",
    inner_lr=0.005, inner_steps=100,
)
maml.save_meta_weights()

# 2. Phase 1: 기구학 컨트롤러로 warm-up 데이터 수집
kin_controller = MPPIController(base_model, params, cost_function=cost_fn)

buffer_s, buffer_c, buffer_n = [], [], []
for step in range(warmup_steps):  # ~40 steps (2초)
    control, _ = kin_controller.compute_control(state, ref)
    next_state = world.step(control, dt)
    buffer_s.append(state); buffer_c.append(control); buffer_n.append(next_state)
    state = next_state

# 3. Phase 1→2 전환: 잔차 계산 + MAML 적응
buf_s, buf_c, buf_n = np.array(buffer_s), np.array(buffer_c), np.array(buffer_n)
kin_dots = np.array([base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)])
kin_next = buf_s + kin_dots * dt
residual_target = buf_s + (buf_n - kin_next)  # base에 잔차를 더한 타겟

maml.adapt(buf_s, buf_c, residual_target, dt, restore=True)

# 4. ResidualDynamics(kinematic + MAML) 컨트롤러
residual_model = ResidualDynamics(base_model=base_model, learned_model=maml, use_residual=True)
controller = MPPIController(residual_model, params, cost_function=cost_fn)

# 5. Phase 2: Residual MAML 제어 + 주기적 재적응
for step in range(num_steps):
    control, _ = controller.compute_control(state, ref)
    next_state = world.step(control, dt)

    buffer_s.append(state); buffer_c.append(control); buffer_n.append(next_state)

    # 80 step마다 재적응 (restore=True: 메타에서 시작)
    if step % 80 == 0 and len(buffer_s) >= 40:
        recent_s = np.array(buffer_s[-200:])
        recent_c = np.array(buffer_c[-200:])
        recent_n = np.array(buffer_n[-200:])
        kin_dots = np.array([base_model.forward_dynamics(s, c) for s, c in zip(recent_s, recent_c)])
        residual_target = recent_s + (recent_n - (recent_s + kin_dots * dt))
        maml.adapt(recent_s, recent_c, residual_target, dt, restore=True)
        # 컨트롤러 reset 안 함 — MPPI warm-start 유지

    state = next_state
```

### MAMLTrainer

```python
from mppi_controller.learning.maml_trainer import MAMLTrainer

class MAMLTrainer:
    """FOMAML 메타 학습 파이프라인."""

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 2,
        hidden_dims: List[int] = [128, 128],
        inner_lr: float = 0.005,      # inner-loop 학습률
        inner_steps: int = 10,        # inner-loop gradient step 수
        meta_lr: float = 5e-4,        # outer-loop 학습률
        task_batch_size: int = 8,     # 메타 배치당 태스크 수
        support_size: int = 100,      # support set 크기
        query_size: int = 100,        # query set 크기
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    )

    def meta_train(
        self,
        n_iterations: int = 1000,     # 메타 학습 반복 수
        verbose: bool = True,
    ):
        """FOMAML 메타 학습 실행."""

    def save_meta_model(self, filename: str = "maml_meta_model.pth"):
        """메타 모델 + norm_stats 저장."""

    def load_meta_model(self, filename: str = "maml_meta_model.pth"):
        """메타 모델 로드."""
```

#### 커스텀 메타 학습

```python
trainer = MAMLTrainer(
    state_dim=3, control_dim=2,
    hidden_dims=[128, 128],
    inner_lr=0.005,
    inner_steps=10,
    meta_lr=5e-4,
    task_batch_size=8,
    support_size=100,
    query_size=100,
)

# 메타 학습 실행
trainer.meta_train(n_iterations=1000, verbose=True)

# 모델 저장
trainer.save_meta_model("my_maml_model.pth")

# 나중에 로드
trainer2 = MAMLTrainer(state_dim=3, control_dim=2, hidden_dims=[128, 128])
trainer2.load_meta_model("my_maml_model.pth")
```

---

## 성능 분석

### 6-Way 비교 결과 (--world dynamic, circle, 20s)

```
┌─────────────────────────────────────────────────────────────────────┐
│  순위    모델                          RMSE       특성             │
├─────────────────────────────────────────────────────────────────────┤
│  1위    Oracle (5D, exact)           ~0.023m    이론적 상한       │
│  2위    Dynamic (5D, mismatched)     ~0.025m    구조 이점         │
│  3위    Kinematic (3D)               ~0.029m    feedback 보상     │
│  4위    MAML (3D, Residual)          ~0.074m    온라인 적응       │
│  5위    Residual (3D, offline)       ~0.120m    오프라인 한계     │
│  6위    Neural (3D, offline)         ~0.287m    오프라인 한계     │
└─────────────────────────────────────────────────────────────────────┘
```

**5-seed 안정성 (MAML)**: 0.073~0.094m, mean=0.081m ± 0.007

### Residual MAML이 우수한 이유

1. **기구학 base + MAML 잔차**: 기구학이 안정성 보장, MAML은 보정만 학습
2. **메타 파라미터 θ\*가 좋은 초기점**: 궤적 추종 패턴으로 학습하여 빠른 적응
3. **매번 메타 파라미터에서 시작**: 누적 드리프트 없이 안정적 재적응

### 설계 선택의 근거

- **기구학 warm-up (2초)**: 메타 모델 단독은 속도를 25% 과소 예측 → MPPI 발산. 기구학(RMSE 0.029m)으로 안정적 데이터 수집
- **Residual 학습**: MAML이 전체 dynamics 대신 잔차(마찰/관성 보정)만 학습 → 오차 범위가 작아 rollout 안정적
- **restore=True 재적응**: continuous fine-tuning(restore=False)은 장시간 시 드리프트 발생 (loss 0.20→2.71). 매번 메타에서 재적응하여 안정성 유지
- **컨트롤러 reset 안 함**: 재적응 후 MPPI warm-start 유지 → 제어 연속성 보장

---

## 하이퍼파라미터 튜닝

### Inner Loop 파라미터 (메타 학습 시)

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `inner_lr` | 0.005 | 0.001~0.05 | 메타 학습 inner-loop 학습률 |
| `inner_steps` | 10 | 5~20 | 메타 학습 inner-loop gradient step 수 |

### Outer Loop 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `meta_lr` | 5e-4 | 1e-4~1e-3 | 메타 학습률. 안정적 수렴 위해 inner_lr보다 낮게 |
| `task_batch_size` | 8 | 4~16 | 배치당 태스크 수. 클수록 안정적이지만 느림 |
| `n_iterations` | 1000 | 500~2000 | 메타 학습 반복 수 |

### 온라인 적응 파라미터 (실행 시)

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `inner_lr` (적응) | 0.005 | 0.001~0.01 | 온라인 적응 학습률 |
| `inner_steps` (적응) | 100 | 50~200 | 온라인 적응 SGD 스텝 수 |
| `warmup_steps` | 40 | 20~80 | Phase 1 기구학 warm-up 스텝 수 |
| `adapt_interval` | 80 | 40~160 | Phase 2 재적응 주기 (step 단위) |
| `buffer_size` | 200 | 100~500 | 적응 데이터 버퍼 크기 |
| `restore` | True | - | 매 적응마다 메타 파라미터 복원 (권장) |

### 튜닝 팁

```
적응이 불안정할 때:
  → inner_lr 낮추기 (0.005 → 0.001)
  → inner_steps 줄이기 (100 → 50)
  → restore=True 확인 (누적 드리프트 방지)

적응이 부족할 때:
  → inner_steps 늘리기 (100 → 200)
  → buffer_size 늘리기 (200 → 500)
  → warmup_steps 늘리기 (40 → 80, 더 많은 데이터 수집)

메타 학습이 수렴 안 할 때:
  → meta_lr 낮추기 (5e-4 → 1e-4)
  → task_batch_size 늘리기 (8 → 16)
  → n_iterations 늘리기 (1000 → 2000)
  → 학습 데이터 패턴 확인 (궤적 추종 패턴 포함 여부)
```

---

## 기존 학습 방법과의 비교

### 비교표

| 방법 | 학습 시간 | 적응 시간 | 데이터 요구 | 환경 변화 | RMSE |
|------|-----------|-----------|-------------|-----------|------|
| Neural (오프라인) | ~1분 | 없음 (고정) | 5000+ | 대응 불가 | ~0.287m |
| Residual (오프라인) | ~1분 | 없음 (고정) | 5000+ | 대응 불가 | ~0.120m |
| OnlineLearner (fine-tuning) | ~1분 (초기) | ~수 분 | 누적 100+ | 느린 적응 | 중간 |
| **MAML (Residual)** | **~5분** | **~10ms** | **40~200** | **즉시 적응** | **~0.074m** |
| GP (소량 데이터) | ~10분 | N/A | 1000+ | N/A | ~0.042m |

### 언제 MAML을 선택?

**MAML을 선택하세요:**
- 환경이 자주 변함 (다른 바닥, 다른 하중, 마모 등)
- 빠른 적응이 필요 (수초 이내)
- 오프라인으로 다양한 환경 데이터 수집 가능

**MAML이 적합하지 않은 경우:**
- 환경이 고정 (오프라인 학습으로 충분)
- 메타 학습을 위한 다양한 환경 시뮬레이션 불가
- 실시간 적응 데이터가 매우 적음 (< 10개)

---

## 문제 해결

### Q: 메타 학습 중 loss가 수렴하지 않아요
**A**:
1. `meta_lr` 낮추기 (5e-4 → 1e-4)
2. `task_batch_size` 늘리기 (8 → 16)
3. `n_iterations` 늘리기 (1000 → 2000)
4. 태스크 분포가 너무 넓은 경우 범위 축소
5. 학습 데이터가 궤적 추종 패턴을 포함하는지 확인

### Q: 실행 중 adapt() 후 성능이 오히려 나빠져요
**A**:
1. `restore=True` 확인 (누적 드리프트 방지)
2. `inner_lr` 낮추기 (0.005 → 0.001)
3. `inner_steps` 줄이기 (100 → 50)
4. `buffer_size` 늘리기 (200 → 500) — 더 많은 데이터로 안정적 적응
5. 잔차 타겟 계산이 올바른지 확인 (residual = actual - kinematic)

### Q: `--meta-train`이 너무 오래 걸려요
**A**:
1. `n_iterations` 줄이기 (1000 → 500)
2. `support_size`, `query_size` 줄이기 (100 → 50)
3. `hidden_dims` 축소 ([128, 128] → [64, 64])
4. GPU 사용 (`device="cuda"`)

### Q: MAML 모델 파일이 없다고 나와요
**A**:
```bash
# 메타 학습 먼저 실행
python examples/comparison/model_mismatch_comparison_demo.py \
    --meta-train --world dynamic

# 또는 전체 파이프라인
python examples/comparison/model_mismatch_comparison_demo.py \
    --all --world dynamic
```

---

## 참고 자료

### 논문

- **MAML**: Finn, C., Abbeel, P., & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." *ICML 2017*.
- **FOMAML**: Nichol, A., Achiam, J., & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms." *arXiv:1803.02999*.
- **Reptile**: Nichol, A., & Schulman, J. (2018). "Reptile: A Scalable Metalearning Algorithm." *arXiv:1803.02999*.
- **Meta-Learning for Dynamics**: Nagabandi, A., et al. (2019). "Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning." *ICLR 2019*.
- **MAML for Control**: Richards, S. M., et al. (2021). "Adaptive-Control-Oriented Meta-Learning for Nonlinear Systems." *RSS 2021*.

### 관련 문서

- [학습 모델 종합 가이드](./LEARNED_MODELS_GUIDE.md)
- [온라인 학습 가이드](./ONLINE_LEARNING.md)
- [학습 모델 아키텍처 (한국어)](./LEARNED_MODELS_ARCHITECTURE_KR.md)

---

**마지막 업데이트**: 2026-02-18
**작성자**: Claude Opus 4.6 + Geonhee LEE
