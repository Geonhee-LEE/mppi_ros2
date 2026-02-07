# MPPI ROS2 - Model Predictive Path Integral Control

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-43%20Passing-brightgreen)](tests/)

완전한 MPPI (Model Predictive Path Integral) 제어 라이브러리로, 9가지 SOTA 변형과 3가지 로봇 모델 타입을 지원합니다.

## 🎯 주요 특징

### 9가지 MPPI 변형 구현 ✅

```
1. Vanilla MPPI          - 기본 구현 (Williams et al., 2016)
2. Tube-MPPI             - 외란 강건성 (Williams et al., 2018)
3. Log-MPPI              - 수치 안정성 (log-space softmax)
4. Tsallis-MPPI          - 탐색/집중 조절 (Yin et al., 2021)
5. Risk-Aware MPPI       - CVaR 안전성 (Yin et al., 2023)
6. Smooth MPPI           - 제어 부드러움 (Kim et al., 2021)
7. SVMPC                 - 샘플 다양성 (Lambert et al., 2020)
8. Spline-MPPI           - 메모리 효율 (Bhardwaj et al., 2024)
9. SVG-MPPI              - Guide Particle (Kondo et al., 2024)
```

### 3가지 로봇 모델 타입

- **Kinematic Model**: 속도 제어 기반 (v, ω)
- **Dynamic Model**: 가속도 제어, 질량/관성/마찰 고려 (a, α)
- **Learned Model**: 데이터 기반 학습 동역학
  - **Neural Dynamics**: PyTorch MLP 기반 end-to-end 학습
  - **Gaussian Process**: 불확실성 정량화
  - **Residual Dynamics**: 물리 모델 + 학습 보정

### 성능 벤치마크

| 변형 | RMSE | Solve Time | 특징 |
|------|------|------------|------|
| **SVG-MPPI** 🏆 | **0.0054m** | 234ms | 최고 정확도 |
| **Vanilla** | 0.0079m | **5.03ms** | 최고 속도 |
| **Spline** | 0.0181m | 42ms | 메모리 -73% |
| **SVMPC** | 0.0092m | 1515ms | 샘플 품질 |

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/mppi_ros2.git
cd mppi_ros2

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치
pip install -e .
```

### 기본 사용 예제

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.utils.trajectory import create_trajectory_function, generate_reference_trajectory

# 1. 모델 생성
model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

# 2. MPPI 파라미터 설정
params = MPPIParams(
    N=30,           # 예측 호라이즌
    dt=0.05,        # 시간 간격
    K=1024,         # 샘플 수
    lambda_=1.0,    # 온도 파라미터
    sigma=np.array([0.5, 0.5]),  # 노이즈 표준편차
    Q=np.array([10.0, 10.0, 1.0]),  # 상태 추적 가중치
    R=np.array([0.1, 0.1]),  # 제어 노력 가중치
)

# 3. 컨트롤러 생성
controller = MPPIController(model, params)

# 4. 시뮬레이터 설정
simulator = Simulator(model, controller, params.dt)

# 5. 레퍼런스 궤적
trajectory_fn = create_trajectory_function('circle')

def reference_fn(t):
    return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

# 6. 시뮬레이션 실행
initial_state = trajectory_fn(0.0)
simulator.reset(initial_state)
history = simulator.run(reference_fn, duration=15.0)

print(f"Position RMSE: {compute_metrics(history)['position_rmse']:.4f}m")
```

### 다른 MPPI 변형 사용

```python
# SVG-MPPI (최고 정확도)
from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController
from mppi_controller.controllers.mppi.mppi_params import SVGMPPIParams

params = SVGMPPIParams(
    N=30, dt=0.05, K=1024,
    svg_num_guide_particles=32,
    svgd_num_iterations=3,
)
controller = SVGMPPIController(model, params)

# Tube-MPPI (외란 강건성)
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.mppi_params import TubeMPPIParams

params = TubeMPPIParams(
    N=30, dt=0.05, K=1024,
    tube_enabled=True,
    K_fb=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
)
controller = TubeMPPIController(model, params)

# Spline-MPPI (메모리 효율)
from mppi_controller.controllers.mppi.spline_mppi import SplineMPPIController
from mppi_controller.controllers.mppi.mppi_params import SplineMPPIParams

params = SplineMPPIParams(
    N=30, dt=0.05, K=1024,
    spline_num_knots=8,
    spline_degree=3,
)
controller = SplineMPPIController(model, params)
```

### 학습 모델 사용

```python
# 1. 데이터 수집 및 학습
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

# 데이터 수집 (시뮬레이션)
collector = DataCollector(state_dim=3, control_dim=2)
# ... 데이터 수집 ...
collector.save("training_data.pkl")

# 데이터셋 준비
data = collector.get_data()
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)

# 신경망 학습
trainer = NeuralNetworkTrainer(state_dim=3, control_dim=2)
train_inputs, train_targets = dataset.get_train_data()
val_inputs, val_targets = dataset.get_val_data()
trainer.train(train_inputs, train_targets, val_inputs, val_targets,
              dataset.get_normalization_stats(), epochs=100)
trainer.save_model("my_model.pth")

# 2. 학습된 모델 사용
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics

neural_model = NeuralDynamics(
    state_dim=3,
    control_dim=2,
    model_path="models/learned_models/my_model.pth"
)
controller = MPPIController(neural_model, params)

# 3. Residual Learning (물리 + 학습)
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

residual_model = ResidualDynamics(
    base_model=kinematic_model,  # 물리 모델
    residual_fn=lambda s, u: neural_model.forward_dynamics(s, u) - kinematic_model.forward_dynamics(s, u)
)
controller = MPPIController(residual_model, params)

# 4. 온라인 학습 (실시간 모델 적응)
from mppi_controller.learning.online_learner import OnlineLearner

# 온라인 학습 관리자 생성
online_learner = OnlineLearner(
    model=neural_model,
    trainer=trainer,
    buffer_size=1000,
    min_samples_for_update=100,
    update_interval=500,  # 500 샘플마다 모델 업데이트
)

# 제어 루프에서 실시간 데이터 수집 및 학습
for t in range(num_steps):
    state = get_state()
    control = controller.compute_control(state, ref_trajectory)

    apply_control(control)
    next_state = get_state()

    # 자동으로 데이터 수집 및 모델 업데이트
    online_learner.add_sample(state, control, next_state, dt)

# 적응 성능 확인
summary = online_learner.get_performance_summary()
print(f"모델 업데이트 횟수: {summary['num_updates']}")
print(f"성능 개선도: {summary['adaptation_improvement']:.2f}%")
```

## 📊 예제 실행

### 기본 데모

```bash
# Vanilla MPPI (원형 궤적)
python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory circle

# 다른 궤적 타입
python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory figure8
python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory sine
```

### 모델별 비교

```bash
# 기구학 vs 동역학
python examples/comparison/kinematic_vs_dynamic_demo.py --trajectory circle --duration 20

# 물리 모델 vs 학습 모델
python examples/comparison/physics_vs_learned_demo.py --trajectory circle --duration 20
```

### MPPI 변형 비교

```bash
# Smooth MPPI 모델 비교
python examples/comparison/smooth_mppi_models_comparison.py --trajectory circle --duration 15

# SVMPC 모델 비교
python examples/comparison/svmpc_models_comparison.py --trajectory circle --duration 15

# Spline MPPI 모델 비교
python examples/comparison/spline_mppi_models_comparison.py --trajectory circle --knots 8

# SVG-MPPI 모델 비교
python examples/comparison/svg_mppi_models_comparison.py --trajectory circle --guides 32
```

### 전체 벤치마크

```bash
# 9개 변형 종합 비교
python examples/mppi_all_variants_benchmark.py --trajectory circle --duration 15
```

### 학습 모델 데모

```bash
# Neural Network 학습 파이프라인
python examples/learned/neural_dynamics_learning_demo.py --all

# Gaussian Process 학습 파이프라인
python examples/learned/gp_vs_neural_comparison_demo.py --all

# GP vs Neural Network 비교 (전체)
python examples/learned/gp_vs_neural_comparison_demo.py \
    --collect-data --train --evaluate

# 데이터 효율성 테스트 (20% 데이터만 사용)
python examples/learned/gp_vs_neural_comparison_demo.py \
    --all --data-fraction 0.2

# 다른 궤적으로 평가
python examples/learned/neural_dynamics_learning_demo.py --evaluate --trajectory figure8

# 온라인 학습 데모 (Sim-to-Real 적응)
python examples/learned/online_learning_demo.py --duration 60.0 --plot
```

## 🤖 ROS2 통합

### ROS2 빌드 및 실행

```bash
# ROS2 워크스페이스 생성 (처음만)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ln -s ~/mppi_ros2 .

# 빌드
cd ~/ros2_ws
colcon build --packages-select mppi_ros2

# 소스
source install/setup.bash
```

### 시뮬레이션 실행

```bash
# 전체 시스템 실행 (RVIZ 포함)
ros2 launch mppi_ros2 mppi_sim.launch.py

# RVIZ 없이 실행
ros2 launch mppi_ros2 mppi_sim.launch.py use_rviz:=false

# 다른 컨트롤러 타입 선택
ros2 launch mppi_ros2 mppi_sim.launch.py controller_type:=svg

# 다른 궤적 타입 선택
ros2 launch mppi_ros2 mppi_sim.launch.py trajectory_type:=figure8

# 동역학 모델 사용
ros2 launch mppi_ros2 mppi_sim.launch.py model_type:=dynamic
```

### 노드 개별 실행

```bash
# 시뮬레이션 로봇
ros2 run mppi_ros2 simple_robot_simulator

# MPPI 컨트롤러
ros2 run mppi_ros2 mppi_controller_node

# 레퍼런스 궤적 퍼블리셔
ros2 run mppi_ros2 trajectory_publisher

# MPPI 시각화
ros2 run mppi_ros2 mppi_visualizer_node
```

### ROS2 토픽

```bash
# 토픽 목록
ros2 topic list

# 주요 토픽:
# - /odom (nav_msgs/Odometry): 로봇 위치 및 속도
# - /cmd_vel (geometry_msgs/Twist): 제어 명령
# - /reference_path (nav_msgs/Path): 레퍼런스 경로
# - /mppi/visualization (visualization_msgs/MarkerArray): RVIZ 시각화

# 토픽 확인
ros2 topic echo /cmd_vel
ros2 topic echo /odom
```

### ROS2 파라미터 조정

```bash
# 파라미터 목록
ros2 param list /mppi_controller

# 파라미터 변경
ros2 param set /mppi_controller lambda_ 2.0
ros2 param set /mppi_controller K 2048

# 파라미터 저장
ros2 param dump /mppi_controller > my_params.yaml

# 파라미터 로드
ros2 run mppi_ros2 mppi_controller_node --ros-args --params-file my_params.yaml
```

### MPPI 변형 선택

| Controller Type | 설명 | 추천 사용 |
|----------------|------|----------|
| `vanilla` | 기본 MPPI | 일반 추적 |
| `tube` | Tube-MPPI | 외란 환경 |
| `log` | Log-MPPI | 수치 안정성 |
| `tsallis` | Tsallis-MPPI | 탐색/집중 조절 |
| `risk_aware` | Risk-Aware | 안전 중시 |
| `smooth` | Smooth MPPI | 제어 부드러움 |
| `svmpc` | SVMPC | 샘플 품질 |
| `spline` | Spline-MPPI | 메모리 효율 |
| `svg` | SVG-MPPI | 고정밀 추적 |

### 레퍼런스 궤적 타입

| Trajectory Type | 설명 |
|-----------------|------|
| `circle` | 원형 궤적 |
| `figure8` | 8자 궤적 |
| `sine` | 사인파 궤적 |
| `lemniscate` | ∞ 모양 궤적 |
| `straight` | 직선 궤적 |

### 설정 파일 수정

MPPI 컨트롤러 설정: `configs/mppi_controller.yaml`
```yaml
mppi_controller:
  ros__parameters:
    controller_type: vanilla
    N: 30
    K: 1024
    lambda_: 1.0
    # ... 기타 파라미터
```

궤적 설정: `configs/trajectory.yaml`
```yaml
trajectory_publisher:
  ros__parameters:
    trajectory_type: circle
    radius: 5.0
    frequency: 0.1
```

## 📁 프로젝트 구조

```
mppi_ros2/
├── mppi_controller/
│   ├── models/                    # 로봇 동역학 모델
│   │   ├── base_model.py          # 추상 베이스 클래스
│   │   ├── kinematic/             # 기구학 모델
│   │   ├── dynamic/               # 동역학 모델
│   │   └── learned/               # 학습 모델
│   │
│   ├── controllers/mppi/          # MPPI 컨트롤러
│   │   ├── base_mppi.py           # Vanilla MPPI
│   │   ├── tube_mppi.py           # Tube-MPPI
│   │   ├── log_mppi.py            # Log-MPPI
│   │   ├── tsallis_mppi.py        # Tsallis-MPPI
│   │   ├── risk_aware_mppi.py     # Risk-Aware MPPI
│   │   ├── smooth_mppi.py         # Smooth MPPI
│   │   ├── stein_variational_mppi.py  # SVMPC
│   │   ├── spline_mppi.py         # Spline-MPPI
│   │   ├── svg_mppi.py            # SVG-MPPI
│   │   ├── mppi_params.py         # 파라미터 클래스
│   │   ├── dynamics_wrapper.py    # 배치 동역학
│   │   ├── cost_functions.py      # 비용 함수
│   │   └── sampling.py            # 노이즈 샘플러
│   │
│   ├── simulation/                # 시뮬레이션 도구
│   │   ├── simulator.py           # 시뮬레이터
│   │   ├── visualizer.py          # 시각화
│   │   └── metrics.py             # 메트릭 계산
│   │
│   └── utils/                     # 유틸리티
│       ├── trajectory.py          # 궤적 생성
│       └── stein_variational.py   # SVGD 유틸리티
│
├── tests/                         # 유닛 테스트 (43개)
├── examples/                      # 예제 스크립트
├── docs/                          # 문서
└── configs/                       # 설정 파일
```

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 변형 테스트
pytest tests/test_svg_mppi.py -v
pytest tests/test_spline_mppi.py -v
pytest tests/test_stein_variational_mppi.py -v
```

**테스트 현황**: 43개 테스트 전부 통과 ✅

## 📈 성능 비교

### 정확도 vs 속도

```
                SVG-MPPI ●
                   ↑
정확도             │      ● SVMPC
                   │   ● Smooth
                   │● Vanilla, Tube, Log
                   │  ● Risk-Aware
                   │    ● Tsallis
                   │       ● Spline
                   └────────────────→ 속도
           느림 (1500ms)      빠름 (5ms)
```

### 메모리 효율성

- **Vanilla MPPI**: 61,440 elements
- **Spline-MPPI**: 16,384 elements (**-73.3%**)

### SVGD 복잡도

- **SVMPC**: O(K²) = 1,048,576 operations
- **SVG-MPPI**: O(G²) = 1,024 operations (**-99.9%**)

## 🎯 사용 시나리오 추천

| 시나리오 | 추천 변형 | 이유 |
|---------|----------|------|
| 실시간 제어 | Vanilla, Tube, Log | ~5ms 초고속 |
| 외란 환경 | Tube-MPPI | 명목+피드백 강건성 |
| 고정밀 추적 | SVG-MPPI | 0.0054m 최고 정확도 |
| 메모리 제약 | Spline-MPPI | 73% 메모리 감소 |
| 안전 중시 | Risk-Aware | CVaR 보수적 제어 |
| 탐색 필요 | Tsallis-MPPI | q 파라미터 조절 |
| 수치 안정성 | Log-MPPI | NaN/Inf 방지 |
| 제어 부드러움 | Smooth MPPI | Input-lifting |

## 📚 문서

### 프로젝트 문서
- [PRD (Product Requirements Document)](docs/mppi/PRD.md)
- [Implementation Status](docs/mppi/IMPLEMENTATION_STATUS.md)
- [CLAUDE Development Guide](CLAUDE.md)
- [TODO List](TODO.md)

### 학습 모델 가이드
- [학습 모델 종합 가이드](docs/learned_models/LEARNED_MODELS_GUIDE.md)
- [온라인 학습 가이드](docs/learned_models/ONLINE_LEARNING.md)

## 🔬 참고 논문

### Vanilla MPPI
- Williams et al. (2016) - "Aggressive Driving with MPPI"
- Williams et al. (2017) - "Information Theoretic MPC"

### M2 고도화
- Williams et al. (2018) - "Robust Sampling Based MPPI" (Tube-MPPI)

### M3 SOTA 변형
- Yin et al. (2021) - "Tsallis Entropy for MPPI"
- Yin et al. (2023) - "Risk-Aware MPPI"
- Lambert et al. (2020) - "Stein Variational MPC"

### M3.5 확장 변형
- Kim et al. (2021) - "Smooth MPPI"
- Bhardwaj et al. (2024) - "Spline-MPPI"
- Kondo et al. (2024) - "SVG-MPPI"

## 🛠️ 개발 로드맵

### ✅ 완료 (M1-M3.5)
- [x] 9가지 MPPI 변형 구현
- [x] 3가지 로봇 모델 타입
- [x] 종합 벤치마크 도구
- [x] 43개 유닛 테스트

### 🚧 진행 중 (M4)
- [ ] ROS2 통합
- [ ] nav2 Controller 플러그인
- [ ] RVIZ 실시간 시각화

### 📅 계획 중 (M5)
- [ ] C++ 포팅 (실시간 성능)
- [ ] GPU 가속 (CuPy/JAX)
- [ ] 추가 로봇 모델 (Swerve, Ackermann)

## 🤝 기여

이슈 및 PR을 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👨‍💻 개발자

**Geonhee Lee**
- GitHub: [@Geonhee-LEE](https://github.com/Geonhee-LEE)

**With assistance from:**
- Claude Sonnet 4.5 (Anthropic)

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들을 참고하여 개발되었습니다:

- [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi) - PyTorch GPU MPPI
- [mppi_playground](https://github.com/kohonda/mppi_playground) - MPPI 벤치마크
- [toy_claude_project](https://github.com/Geonhee-LEE/toy_claude_project) - 9가지 MPPI 변형

## 📞 연락

질문이나 제안이 있으시면 이슈를 열어주세요!

---

**Made with ❤️ using Claude Code**
