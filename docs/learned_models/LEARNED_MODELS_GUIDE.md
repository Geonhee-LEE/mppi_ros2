# MPPI ROS2 학습 모델 종합 가이드

## 목차

1. [개요](#개요)
2. [학습 모델 타입](#학습-모델-타입)
3. [데이터 수집 및 전처리](#데이터-수집-및-전처리)
4. [모델별 학습 방법](#모델별-학습-방법)
5. [성능 비교 및 선택 가이드](#성능-비교-및-선택-가이드)
6. [온라인 학습 및 적응](#온라인-학습-및-적응)
7. [MPPI 통합](#mppi-통합)
8. [베스트 프랙티스](#베스트-프랙티스)
9. [API 레퍼런스](#api-레퍼런스)

---

## 개요

MPPI ROS2는 **3가지 학습 기반 동역학 모델**을 지원합니다:

1. **Neural Dynamics**: 심층 신경망 기반 end-to-end 학습
2. **Gaussian Process Dynamics**: 베이지안 비모수 모델, 불확실성 정량화
3. **Residual Dynamics**: 물리 모델 + 학습 보정 (하이브리드)

### 핵심 특징

- **통일된 인터페이스**: 모든 학습 모델이 `RobotModel` 베이스 클래스 구현
- **MPPI 호환성**: 기구학/동역학 모델과 동일한 방식으로 사용 가능
- **온라인 학습**: 실시간 데이터 수집 및 모델 업데이트
- **불확실성 정량화**: GP 기반 신뢰 구간 제공
- **Sim-to-Real 전이**: 도메인 적응 지원

---

## 학습 모델 타입

### 1. Neural Dynamics

**특징**:
- PyTorch 기반 Multi-Layer Perceptron (MLP)
- End-to-end 학습: (state, control) → state_dot
- 빠른 추론 속도 (~0.1ms)
- 대규모 데이터에 적합

**장점**:
- 높은 표현력 (복잡한 비선형 동역학)
- GPU 가속 가능
- 학습 데이터가 충분하면 높은 정확도

**단점**:
- 불확실성 정량화 어려움 (별도 앙상블 필요)
- 데이터 효율성 낮음 (GP 대비)
- 외삽 영역에서 신뢰도 낮음

**사용 사례**:
- 고속 제어 (~100Hz)
- 충분한 학습 데이터 확보 가능
- 불확실성보다 정확도 우선

### 2. Gaussian Process Dynamics

**특징**:
- GPyTorch 기반 베이지안 비모수 모델
- RBF/Matern 커널 + ARD (Automatic Relevance Determination)
- 평균 + 표준편차 예측 (불확실성)
- Exact GP (소규모) / Sparse GP (대규모)

**장점**:
- 불확실성 정량화 (1σ: 68%, 2σ: 95% 신뢰구간)
- 데이터 효율성 높음 (소량 데이터로 학습)
- 외삽 영역에서 불확실성 자동 증가
- Feature importance (ARD lengthscales)

**단점**:
- 추론 속도 느림 (Exact GP: O(N²), Sparse GP: O(NM))
- 대규모 데이터 처리 어려움 (N > 10,000)
- GPU 가속 제한적

**사용 사례**:
- 안전이 중요한 응용 (불확실성 고려)
- 데이터 수집 비용 높음
- 저속 제어 (~10Hz)
- 능동 학습 (불확실성 기반 데이터 수집)

### 3. Residual Dynamics

**특징**:
- 물리 모델 + 학습 보정의 하이브리드
- f_total = f_physics + f_learned
- f_learned는 Neural 또는 GP

**장점**:
- 물리 모델의 구조적 지식 활용
- 학습 부담 감소 (잔차만 학습)
- 외삽 시 물리 모델로 fallback
- 데이터 효율성 높음

**단점**:
- 물리 모델이 필요 (모델링 비용)
- 물리 모델 오차가 크면 효과 감소

**사용 사례**:
- 물리 모델이 존재하지만 불완전함
- 마찰, 공기저항 등 모델링 어려운 항
- Sim-to-Real 전이 (시뮬레이터 모델 보정)

---

## 데이터 수집 및 전처리

### DataCollector

실제 로봇 또는 시뮬레이터에서 동역학 데이터를 수집합니다.

```python
from mppi_controller.learning.data_collector import DataCollector

collector = DataCollector(
    state_dim=3,      # [x, y, θ]
    control_dim=2,    # [v, ω]
)

# Episode 시작
collector.start_episode()

# 제어 루프
for t in range(num_steps):
    state = get_current_state()
    control = controller.compute_control(state, ref_trajectory)

    # 제어 적용
    apply_control(control)
    time.sleep(dt)

    next_state = get_current_state()

    # 샘플 추가
    collector.add_sample(state, control, next_state, dt)

# Episode 종료
collector.end_episode()

# 데이터 추출
data = collector.get_data()
# Returns: {"states": (N, nx), "controls": (N, nu),
#           "next_states": (N, nx), "state_dots": (N, nx), "dt": (N,)}
```

### DynamicsDataset

학습을 위한 데이터 전처리 및 분할.

```python
from mppi_controller.learning.data_collector import DynamicsDataset

dataset = DynamicsDataset(
    data=data,
    train_ratio=0.8,     # 80% 학습, 20% 검증
    normalize=True,      # 자동 정규화
)

train_inputs, train_targets = dataset.get_train_data()
val_inputs, val_targets = dataset.get_val_data()

# 정규화 통계 (모델 저장 시 필요)
norm_stats = dataset.get_normalization_stats()
# Returns: {"state_mean", "state_std", "control_mean", "control_std",
#           "state_dot_mean", "state_dot_std"}
```

**정규화 중요성**:
- 신경망: 학습 안정성 및 수렴 속도 향상
- GP: 커널 최적화 효율성

---

## 모델별 학습 방법

### Neural Dynamics 학습

#### 1. 데이터 수집

```python
# 다양한 궤적에서 데이터 수집 (최소 5000 샘플 권장)
collector = DataCollector(state_dim=3, control_dim=2)

for episode in range(50):
    collector.start_episode()

    # 랜덤 레퍼런스 궤적 생성
    ref_fn = generate_random_trajectory()

    # 데이터 수집 (30초)
    collect_episode(ref_fn, duration=30.0, dt=0.05)

    collector.end_episode()

data = collector.get_data()
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
```

#### 2. 모델 학습

```python
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

trainer = NeuralNetworkTrainer(
    state_dim=3,
    control_dim=2,
    hidden_dims=[128, 128, 128],  # 3-layer MLP
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 학습 실행
history = trainer.train(
    train_inputs=dataset.train_inputs,
    train_targets=dataset.train_targets,
    val_inputs=dataset.val_inputs,
    val_targets=dataset.val_targets,
    norm_stats=dataset.get_normalization_stats(),
    epochs=200,
    batch_size=256,
    early_stopping_patience=20,
    verbose=True,
)

# 모델 저장
trainer.save_model("neural_model.pth")
```

#### 3. 모델 로드 및 사용

```python
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics

model = NeuralDynamics(
    state_dim=3,
    control_dim=2,
    model_path="neural_model.pth",
    device="cpu",
)

# MPPI와 통합
from mppi_controller.controllers.mppi.base_mppi import MPPIController

controller = MPPIController(model=model, params=params)
```

### Gaussian Process 학습

#### 1. 데이터 수집

```python
# GP는 데이터 효율적 (1000-2000 샘플로 충분)
collector = DataCollector(state_dim=3, control_dim=2)

for episode in range(10):
    collector.start_episode()
    collect_episode(ref_fn, duration=30.0, dt=0.05)
    collector.end_episode()

data = collector.get_data()
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
```

#### 2. 모델 학습

```python
from mppi_controller.learning.gaussian_process_trainer import GaussianProcessTrainer

trainer = GaussianProcessTrainer(
    state_dim=3,
    control_dim=2,
    kernel_type="rbf",        # "rbf" or "matern"
    use_ard=True,             # Automatic Relevance Determination
    use_sparse_gp=False,      # Exact GP (소규모 데이터)
    num_inducing_points=500,  # Sparse GP 사용 시
)

# 학습 실행
history = trainer.train(
    train_inputs=dataset.train_inputs,
    train_targets=dataset.train_targets,
    val_inputs=dataset.val_inputs,
    val_targets=dataset.val_targets,
    norm_stats=dataset.get_normalization_stats(),
    num_iterations=100,
    verbose=True,
)

# 모델 저장
trainer.save_model("gp_model.pth")
```

#### 3. 모델 로드 및 사용

```python
from mppi_controller.models.learned.gaussian_process_dynamics import GaussianProcessDynamics

model = GaussianProcessDynamics(
    state_dim=3,
    control_dim=2,
    model_path="gp_model.pth",
)

# 불확실성 포함 예측
state = np.array([0.0, 0.0, 0.0])
control = np.array([1.0, 0.5])

mean, std = model.predict_with_uncertainty(state, control)
print(f"State_dot: {mean} ± {std}")

# Feature importance 분석
lengthscales = model.get_lengthscales()
print(f"Lengthscales: {lengthscales}")  # 작을수록 중요한 feature
```

### Residual Dynamics 학습

#### 1. 데이터 수집 (실제 시스템)

```python
# 물리 모델과 실제 시스템 차이 수집
collector = DataCollector(state_dim=3, control_dim=2)

for episode in range(20):
    collector.start_episode()
    collect_episode_from_real_robot(duration=30.0)
    collector.end_episode()

data = collector.get_data()
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
```

#### 2. Residual 학습

```python
from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

# 물리 모델
physics_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

# 물리 모델 예측
physics_predictions = []
for state, control in zip(dataset.train_inputs[:, :3], dataset.train_inputs[:, 3:]):
    physics_dot = physics_model.forward_dynamics(state, control)
    physics_predictions.append(physics_dot)

physics_predictions = np.array(physics_predictions)

# Residual 타겟 계산
residual_targets = dataset.train_targets - physics_predictions

# Residual 학습 (Neural Network)
residual_trainer = NeuralNetworkTrainer(
    state_dim=3,
    control_dim=2,
    hidden_dims=[64, 64],  # Residual은 작은 네트워크로 충분
)

history = residual_trainer.train(
    train_inputs=dataset.train_inputs,
    train_targets=residual_targets,
    val_inputs=dataset.val_inputs,
    val_targets=dataset.val_targets - physics_predictions_val,
    norm_stats=dataset.get_normalization_stats(),
    epochs=100,
)

residual_trainer.save_model("residual_model.pth")
```

#### 3. Residual Dynamics 사용

```python
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

# Residual 함수 정의
def residual_fn(state, control):
    return residual_trainer.predict(state, control, denormalize=True)

# Residual Dynamics 생성
residual_model = ResidualDynamics(
    base_model=physics_model,
    residual_fn=residual_fn,
)

# MPPI와 통합
controller = MPPIController(model=residual_model, params=params)
```

---

## 성능 비교 및 선택 가이드

### 데이터 효율성

| 모델 타입 | 필요 데이터 | 학습 시간 | 추론 속도 |
|----------|------------|----------|----------|
| Neural | 5,000+ 샘플 | ~5분 (GPU) | 0.1ms |
| GP (Exact) | 1,000-2,000 샘플 | ~10분 | 10ms |
| GP (Sparse) | 5,000+ 샘플 | ~15분 | 5ms |
| Residual | 2,000-3,000 샘플 | ~3분 | 0.2ms |

### 정확도 비교 (Differential Drive 예시)

```
┌─────────────────────────────────────────────────────────┐
│          학습 데이터: 5,000 샘플 (50 episodes)         │
├─────────────────────────────────────────────────────────┤
│ Model Type    │  RMSE (m)  │  Max Error (m)  │ 2σ Cov │
├───────────────┼────────────┼─────────────────┼────────┤
│ Physics Only  │   0.150    │      0.450      │   N/A  │
│ Neural        │   0.035    │      0.120      │   N/A  │
│ GP (Exact)    │   0.042    │      0.135      │  96.2% │
│ GP (Sparse)   │   0.048    │      0.145      │  94.8% │
│ Residual+NN   │   0.028    │      0.095      │   N/A  │
│ Residual+GP   │   0.032    │      0.105      │  97.1% │
└───────────────┴────────────┴─────────────────┴────────┘

(2σ Cov: 2-sigma coverage, 이론적으로 95% 목표)
```

### 선택 가이드

#### Neural Dynamics를 선택하세요:
- ✅ 대량의 학습 데이터 확보 가능
- ✅ 고속 제어 필요 (>50Hz)
- ✅ GPU 가속 가능
- ✅ 불확실성보다 정확도 우선
- ❌ 데이터 수집 비용 높음
- ❌ 안전 크리티컬 응용

#### Gaussian Process를 선택하세요:
- ✅ 데이터 수집 비용 높음 (소량 데이터)
- ✅ 불확실성 정량화 필요
- ✅ 안전 크리티컬 응용
- ✅ 능동 학습 (불확실성 기반)
- ❌ 실시간 고속 제어 (>30Hz)
- ❌ 대규모 데이터 (>10,000)

#### Residual Dynamics를 선택하세요:
- ✅ 물리 모델 존재 (불완전함)
- ✅ Sim-to-Real 전이
- ✅ 데이터 효율성 + 정확도 동시 필요
- ✅ 외삽 영역에서 안정성 필요
- ❌ 물리 모델 없음
- ❌ 물리 모델이 매우 정확함 (학습 불필요)

---

## 온라인 학습 및 적응

자세한 내용은 [ONLINE_LEARNING.md](./ONLINE_LEARNING.md)를 참조하세요.

### 온라인 학습 개요

실시간으로 데이터를 수집하고 모델을 업데이트하여 도메인 변화에 적응합니다.

```python
from mppi_controller.learning.online_learner import OnlineLearner

# 사전 학습된 모델 및 트레이너
model = NeuralDynamics(model_path="neural_model.pth")
trainer = NeuralNetworkTrainer(...)

# 온라인 학습 관리자
online_learner = OnlineLearner(
    model=model,
    trainer=trainer,
    buffer_size=1000,          # 순환 버퍼 크기
    min_samples_for_update=100,
    update_interval=500,       # 500 샘플마다 재학습
)

# 제어 루프
for t in range(num_steps):
    state = get_state()
    control = controller.compute_control(state, ref_trajectory)

    apply_control(control)
    next_state = get_state()

    # 온라인 데이터 추가 (자동 재학습 트리거)
    online_learner.add_sample(state, control, next_state, dt)

# 성능 요약
summary = online_learner.get_performance_summary()
print(f"Updates: {summary['num_updates']}")
print(f"Improvement: {summary['adaptation_improvement']:.2f}%")
```

### Sim-to-Real 전이

시뮬레이터에서 학습한 모델을 실제 로봇에 적응:

1. **시뮬레이터에서 사전 학습**
   ```python
   # 시뮬레이터 데이터로 초기 모델 학습
   train_in_simulator(model, trainer, num_samples=10000)
   ```

2. **실제 로봇에서 온라인 적응**
   ```python
   # 실제 로봇 제어 + 온라인 학습
   online_learner = OnlineLearner(model, trainer)

   for episode in range(100):
       run_real_robot_episode(online_learner)
   ```

3. **적응 모니터링**
   ```python
   online_learner.monitor_adaptation(test_data)
   # Improvement: 45.3% (초기 RMSE: 0.250 → 현재: 0.137)
   ```

---

## MPPI 통합

### 기본 통합

모든 학습 모델은 `RobotModel` 인터페이스를 구현하므로 MPPI와 즉시 통합 가능:

```python
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams

# 모델 선택
model = NeuralDynamics(model_path="neural_model.pth")
# 또는
# model = GaussianProcessDynamics(model_path="gp_model.pth")
# 또는
# model = ResidualDynamics(base_model=physics_model, residual_fn=residual_fn)

# MPPI 파라미터
params = MPPIParams(
    num_samples=1024,
    horizon=30,
    dt=0.05,
    lambda_=1.0,
    sigma_u=np.array([0.5, 0.3]),
)

# MPPI 컨트롤러 생성
controller = MPPIController(model=model, params=params)

# 제어 계산
control, info = controller.compute_control(state, reference_trajectory)
```

### GP 불확실성 활용 (Risk-Aware MPPI)

GP의 불확실성을 MPPI 비용 함수에 통합:

```python
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController

# GP 모델
gp_model = GaussianProcessDynamics(model_path="gp_model.pth")

# Risk-Aware MPPI (CVaR)
controller = RiskAwareMPPIController(
    model=gp_model,
    params=params,
    alpha=0.1,  # CVaR 신뢰수준 (하위 10%)
)

# 제어 계산 (불확실성 높은 샘플 페널티)
control, info = controller.compute_control(state, reference_trajectory)
```

### Tube-MPPI와 학습 모델

Tube-MPPI는 모델 불확실성에 강건합니다:

```python
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController

# 학습 모델 (불완전할 수 있음)
learned_model = NeuralDynamics(model_path="neural_model.pth")

# Tube-MPPI (외란 강건 제어)
controller = TubeMPPIController(
    model=learned_model,
    params=params,
    k_fb=np.array([2.0, 1.5]),  # 피드백 게인
)

# 제어 계산 (외란 보상)
control, info = controller.compute_control(state, reference_trajectory)
```

---

## 베스트 프랙티스

### 데이터 수집

1. **다양한 궤적**: 직선, 곡선, 급회전, 정지 등 다양한 시나리오
2. **상태 공간 커버리지**: 전체 동작 범위 탐색
3. **노이즈 주입**: 실제 환경의 불확실성 반영
4. **에피소드 기반**: 긴 연속 궤적보다 짧은 에피소드 여러 개

### 모델 학습

1. **정규화 필수**: 입력/출력 정규화로 학습 안정성 향상
2. **Train/Val 분할**: 80/20 비율로 과적합 방지
3. **Early Stopping**: 검증 손실 모니터링
4. **하이퍼파라미터 튜닝**: 학습률, 레이어 크기, 커널 타입 등

### 모델 검증

1. **RMSE/Max Error**: 기본 정확도 메트릭
2. **불확실성 보정**: GP는 2σ coverage 95% 목표
3. **외삽 테스트**: 학습 범위 밖 데이터로 테스트
4. **실제 제어 테스트**: 시뮬레이터/실제 로봇에서 추적 성능 확인

### 온라인 학습

1. **점진적 적응**: 작은 학습률로 천천히 업데이트
2. **재앙적 망각 방지**: 버퍼에 이전 데이터 유지
3. **성능 모니터링**: 적응 개선도 추적
4. **안전 메커니즘**: 성능 저하 시 이전 모델로 복귀

---

## API 레퍼런스

### NeuralDynamics

```python
class NeuralDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: str,
        device: str = "cpu",
    )

    def forward_dynamics(
        self,
        state: np.ndarray,    # (nx,) or (batch, nx)
        control: np.ndarray,  # (nu,) or (batch, nu)
    ) -> np.ndarray:          # (nx,) or (batch, nx)
        """신경망 예측"""
```

### GaussianProcessDynamics

```python
class GaussianProcessDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: str,
    )

    def forward_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """GP 평균 예측"""

    def predict_with_uncertainty(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(평균, 표준편차) 반환"""

    def get_lengthscales(self) -> List[np.ndarray]:
        """Feature importance 반환"""
```

### ResidualDynamics

```python
class ResidualDynamics(RobotModel):
    def __init__(
        self,
        base_model: RobotModel,
        residual_fn: Optional[Callable] = None,
        uncertainty_fn: Optional[Callable] = None,
    )

    def forward_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """f_physics + f_learned"""
```

### OnlineLearner

```python
class OnlineLearner:
    def __init__(
        self,
        model: RobotModel,
        trainer: Union[NeuralNetworkTrainer, GaussianProcessTrainer],
        buffer_size: int = 1000,
        min_samples_for_update: int = 100,
        update_interval: int = 500,
    )

    def add_sample(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
        dt: float,
    ):
        """샘플 추가 (자동 재학습 트리거)"""

    def update_model(self, num_epochs: int = 10):
        """모델 fine-tuning"""

    def monitor_adaptation(self, test_data: Dict):
        """적응 성능 모니터링"""
```

---

## 데모 실행

### Neural Dynamics 학습

```bash
python examples/learned/neural_dynamics_learning_demo.py \
    --num_episodes 50 \
    --trajectory circle \
    --plot
```

### GP vs Neural 비교

```bash
python examples/learned/gp_vs_neural_comparison_demo.py \
    --num_episodes 20 \
    --data_fraction 0.5 \
    --plot
```

### 온라인 학습

```bash
python examples/learned/online_learning_demo.py \
    --duration 60.0 \
    --noise_std 0.05 \
    --plot
```

---

## 참고 자료

### 논문
- Nagabandi et al. (2018) - "Neural Network Dynamics for Model-Based RL"
- Deisenroth et al. (2015) - "Gaussian Processes for Data-Efficient Learning"
- Hewing et al. (2020) - "Learning-Based MPC: A Review"
- Cheng et al. (2019) - "End-to-End Safe RL with GP"

### 라이브러리
- PyTorch: https://pytorch.org
- GPyTorch: https://gpytorch.ai
- Scikit-learn: https://scikit-learn.org

---

## 문제 해결

### Q: 신경망 학습이 수렴하지 않아요
**A**:
1. 정규화 확인 (`normalize=True`)
2. 학습률 낮추기 (1e-4 ~ 1e-3)
3. 배치 크기 조정 (64, 128, 256)
4. 레이어 크기 축소 (과적합 가능성)

### Q: GP 학습이 너무 느려요
**A**:
1. Sparse GP 사용 (`use_sparse_gp=True`)
2. Inducing points 줄이기 (500 → 300)
3. 데이터 샘플 줄이기 (GP는 2000 샘플로 충분)

### Q: 온라인 학습 중 성능이 떨어져요
**A**:
1. 학습률 낮추기 (fine-tuning epochs 5 → 3)
2. 버퍼 크기 늘리기 (재앙적 망각 방지)
3. 업데이트 주기 늘리기 (500 → 1000)

### Q: 외삽 영역에서 모델이 이상해요
**A**:
1. Residual Dynamics 사용 (물리 모델 fallback)
2. GP 사용 (불확실성 자동 증가)
3. 데이터 수집 범위 확장

---

**마지막 업데이트**: 2025-02-07
**작성자**: Claude Sonnet 4.5 + Geonhee LEE
