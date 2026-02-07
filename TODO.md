# MPPI ROS2 - TODO

프로젝트 개발 작업 목록입니다. Claude가 순차적으로 처리합니다.

---

## 🔴 High Priority (P0) - Phase 1: 기구학 모델 (M1)

### 기본 구조

- [x] #001 프로젝트 기본 구조 설정 ✓ 2026-02-07
  * Python 패키지 구조 생성 (mppi_controller/models, controllers, simulation)
  * pyproject.toml, requirements.txt 작성
  * 기본 디렉토리 생성 (tests, examples, configs, docs)

### 모델 추상화 계층

- [x] #002 추상 베이스 클래스 (base_model.py) ✓ 2026-02-07
  * RobotModel ABC 정의
  * 통일된 인터페이스: state_dim, control_dim, model_type
  * forward_dynamics(state, control) → state_dot
  * step(state, control, dt) - RK4 적분 기본 구현
  * get_control_bounds() → (lower, upper)
  * 벡터화 지원 (batch, nx) 형태

- [x] #003 Differential Drive 기구학 모델 ✓ 2026-02-07
  * models/kinematic/differential_drive_kinematic.py
  * 상태: [x, y, θ] (3차원)
  * 제어: [v, ω] (2차원) - 선속도, 각속도
  * forward_dynamics: dx/dt = v*cos(θ), dy/dt = v*sin(θ), dθ/dt = ω
  * model_type = "kinematic"

### MPPI 컨트롤러 인프라

- [x] #004 MPPIParams 데이터클래스 ✓ 2026-02-07
  * mppi_params.py - 파라미터 데이터클래스
  * N (horizon), dt, K (samples), lambda, sigma, Q, R, Qf
  * 기본값 설정 (N=30, dt=0.05, K=1024)

- [x] #005 BatchDynamicsWrapper (모든 모델 지원) ✓ 2026-02-07
  * dynamics_wrapper.py - 배치 동역학 래퍼
  * __init__(model: RobotModel, dt: float)
  * rollout(initial_state, controls) → trajectories (K, N+1, nx)
  * RK4 벡터화 (K개 샘플 병렬 전파)
  * **모델 타입 무관** - RobotModel 인터페이스만 준수하면 됨

- [x] #006 비용 함수 모듈 (기본) ✓ 2026-02-07
  * cost_functions.py - 비용 함수 클래스
  * StateTrackingCost (Q 가중치)
  * TerminalCost (Qf 가중치)
  * ControlEffortCost (R 가중치)
  * CompositeMPPICost (비용 합성)
  * ObstacleCost (장애물 회피) - M2 준비 완료
  * ControlRateCost (제어 변화율) - M2 준비 완료

- [x] #007 Gaussian 노이즈 샘플러 ✓ 2026-02-07
  * sampling.py - GaussianSampler 클래스
  * sample(U, K) → (K, N, nu) 노이즈
  * Σ = σ² I 대각 공분산
  * ColoredNoiseSampler (OU 프로세스) - M2 준비 완료
  * RectifiedGaussianSampler (pytorch_mppi 스타일) - Phase 4 준비 완료

- [x] #008 Vanilla MPPI 컨트롤러 구현 ✓ 2026-02-07
  * base_mppi.py - MPPIController 클래스
  * compute_control(state, ref) → (control, info)
  * info dict: sample_trajectories, sample_weights, best_trajectory, temperature, ess

### 시뮬레이션 인프라

- [x] #009 간이 시뮬레이터 (Simulator 클래스) ✓ 2026-02-07
  * simulation/simulator.py
  * __init__(model, controller, dt, process_noise_std)
  * reset(initial_state)
  * step(reference_trajectory) → step_info
  * run(reference_trajectory_fn, duration, realtime) → history
  * 메트릭 수집 (time, state, control, reference, solve_time, info)
  * 외란 주입 기능

- [x] #010 시각화 도구 (Visualizer 클래스) ✓ 2026-02-07
  * simulation/visualizer.py
  * plot_results(history, metrics) - 6패널 정적 플롯
    - XY 궤적, 위치 오차, 제어 입력, 각도 오차, 계산 시간, 메트릭 요약
  * animate_live(simulator, reference_trajectory_fn, duration) - 실시간 애니메이션
  * export_gif() - GIF 파일 생성 (Phase 4 예정)

- [x] #011 메트릭 계산 (metrics.py) ✓ 2026-02-07
  * simulation/metrics.py
  * compute_metrics(history) → dict
    - position_rmse, max_position_error
    - heading_rmse
    - control_rate (제어 변화율)
    - mean_solve_time, max_solve_time

### 테스트 및 검증

- [ ] #012 기구학 모델 유닛 테스트 (Phase 1 후속 작업)
  * tests/models/test_kinematic_models.py
  * forward_dynamics 검증
  * step() RK4 적분 검증
  * 벡터화 (batch) 지원 테스트

- [ ] #013 MPPI 유닛 테스트 (Phase 1 후속 작업)
  * tests/test_mppi.py - 기본 기능 테스트
  * tests/test_dynamics_wrapper.py - 동역학 래퍼 테스트
  * tests/test_cost_functions.py - 비용 함수 테스트
  * tests/test_sampling.py - 샘플링 테스트
  * tests/test_simulator.py - 시뮬레이터 테스트

- [x] #014 원형 궤적 추적 데모 ✓ 2026-02-07
  * examples/kinematic/mppi_differential_drive_kinematic_demo.py
  * 원형 경로 생성 유틸리티 (utils/trajectory.py)
  * **검증 결과**: Circle RMSE = 0.0060m ✓ (목표: < 0.2m)
  * **검증 결과**: Solve Time = 4.96ms ✓ (목표: < 100ms)
  * --trajectory {circle,figure8,sine} 선택 구현
  * --live 실시간 시뮬레이션 모드 구현
  * 6패널 정적 플롯 + ASCII 메트릭 요약 구현

---

## 🟠 Medium Priority (P1) - Phase 2: 동역학 모델 (M2)

### 동역학 모델 구현

- [ ] #101 Differential Drive 동역학 모델
  * models/dynamic/differential_drive_dynamic.py
  * 상태: [x, y, θ, v, ω] (5차원)
  * 제어: [a, α] (2차원) - 선가속도, 각가속도
  * forward_dynamics: 마찰/관성 고려
    - dx/dt = v*cos(θ), dy/dt = v*sin(θ), dθ/dt = ω
    - dv/dt = a - c_v*v, dω/dt = α - c_ω*ω
  * model_type = "dynamic"
  * 파라미터: mass, inertia, c_v, c_omega

- [ ] #102 동역학 모델 유닛 테스트
  * tests/models/test_dynamic_models.py
  * forward_dynamics 검증
  * 에너지 보존 검증 (c_v=0, c_omega=0 설정)
  * 벡터화 (batch) 지원 테스트

- [ ] #103 동역학 모델 원형 궤적 데모
  * examples/dynamic/mppi_differential_drive_dynamic_demo.py
  * --trajectory circle --duration 30
  * RMSE 측정 및 메트릭 수집

### 모델 비교

- [ ] #104 기구학 vs 동역학 비교 데모
  * examples/comparison/kinematic_vs_dynamic_demo.py
  * 동일한 레퍼런스 궤적으로 시뮬레이션
  * ASCII 메트릭 비교 테이블
  * 두 궤적 플롯 (동일 참조)
  * --live 실시간 비교

### M2 고급 기능

- [ ] #105 ControlRateCost 비용 함수
  * cost_functions.py에 ControlRateCost 추가
  * Δu 제어 변화율 비용
  * R_rate 가중치 파라미터

- [ ] #106 Adaptive Temperature
  * adaptive_temperature.py - AdaptiveTemperature 클래스
  * ESS (Effective Sample Size) 기반 λ 자동 튜닝
  * 목표 ESS 비율 유지

- [ ] #107 Colored Noise 샘플링
  * sampling.py에 ColoredNoiseSampler 추가
  * OU 프로세스 기반 시간 상관 노이즈
  * theta (reversion rate) 파라미터

- [ ] #108 Obstacle 비용 함수
  * cost_functions.py에 ObstacleCost 추가
  * 원형 장애물 회피
  * safety_margin 파라미터

- [ ] #109 Vanilla vs M2 비교 데모
  * examples/mppi_vanilla_vs_m2_demo.py
  * ControlRate, AdaptiveTemp, ColoredNoise 효과 비교
  * --live 실시간 비교

### Tube-MPPI (외란 강건성)

- [ ] #110 AncillaryController 구현
  * ancillary_controller.py - body frame 피드백
  * world → body 오차 변환
  * K_fb 피드백 게인

- [ ] #111 Tube-MPPI 컨트롤러
  * tube_mppi.py - TubeMPPIController
  * MPPIController 상속
  * 명목 상태 전파 + 피드백 보정
  * tube_enabled 플래그 (False → Vanilla 동작)

- [ ] #112 TubeAwareCost 비용 함수
  * cost_functions.py에 TubeAwareCost 추가
  * 장애물 safety_margin + tube_margin 확장

- [ ] #113 Vanilla vs Tube 비교 데모
  * examples/mppi_vanilla_vs_tube_demo.py
  * --noise 외란 강도 조절
  * --live 실시간 비교

---

## 🟠 Medium Priority (P1) - Phase 3: 학습 모델 (M3)

### 학습 모델 구현

- [ ] #201 ResidualDynamics 모델
  * models/learned/residual_dynamics.py
  * f_total(x, u) = f_physics(x, u) + f_learned(x, u)
  * __init__(base_model, residual_fn, uncertainty_fn)
  * model_type = "learned"
  * get_uncertainty(state, control) - GP 불확실성 (선택적)

- [ ] #202 NeuralDynamics 스켈레톤
  * models/learned/neural_dynamics.py
  * PyTorch 기반 신경망 동역학 (스켈레톤 구현)
  * 향후 학습 파이프라인 연동 준비

- [ ] #203 GaussianProcessDynamics 스켈레톤
  * models/learned/gaussian_process_dynamics.py
  * GP 기반 불확실성 정량화 (스켈레톤 구현)
  * 향후 GPytorch 연동 준비

### 학습 모델 테스트

- [ ] #204 학습 모델 유닛 테스트
  * tests/models/test_learned_models.py
  * ResidualDynamics 동등성 검증 (residual=0)
  * 더미 residual 효과 검증

- [ ] #205 Residual 동역학 데모
  * examples/learned/mppi_residual_dynamics_demo.py
  * 더미 residual_fn으로 RMSE 개선 확인
  * --live 실시간 시뮬레이션

### 모델 비교

- [ ] #206 Physics vs Learned 비교 데모
  * examples/comparison/physics_vs_learned_demo.py
  * 기구학, 동역학, Residual 동역학 3-way 비교
  * ASCII 메트릭 비교 테이블

- [ ] #207 모델 타입별 벤치마크 도구
  * examples/comparison/model_type_benchmark.py
  * Kinematic, Dynamic, Learned 동시 비교
  * --trajectory {circle,figure8,sine} 선택
  * ASCII 요약 테이블 + 차트

---

## 🟢 Low Priority (P2) - Phase 4: pytorch_mppi 개선

### 함수 기반 인터페이스

- [ ] #301 FunctionDynamicsWrapper (함수 기반)
  * dynamics_wrapper.py에 FunctionDynamicsWrapper 추가
  * __init__(dynamics_fn, dt)
  * dynamics_fn: (K, nx), (K, nu) → (K, nx)
  * pytorch_mppi 스타일 인터페이스

- [ ] #302 MPPIController 하이브리드 인터페이스
  * base_mppi.py 수정
  * __init__(model=None, dynamics_fn=None, ...)
  * 클래스 방식 (model) 또는 함수 방식 (dynamics_fn) 선택

- [ ] #303 정류 가우시안 샘플링
  * sampling.py에 use_rectified 파라미터 추가
  * 제약 위반 샘플 재샘플링 (pytorch_mppi 스타일)
  * 기존 클리핑 대비 성능 비교

- [ ] #304 함수 기반 인터페이스 테스트
  * tests/test_function_dynamics.py
  * 더미 신경망 함수 테스트
  * 클래스 방식과 동등성 검증

### 분석 도구

- [ ] #305 오프라인 분석 도구 (TrajectoryAnalyzer)
  * simulation/trajectory_analyzer.py
  * 궤적 히스토리 분석
  * 주파수 분석, 제어 스펙트럼 등

- [ ] #306 GIF export 기능
  * visualizer.py에 export_gif() 추가
  * mppi_playground 참고
  * --export-gif 플래그

---

## 🟢 Low Priority (P2) - MPPI 변형 (M3 SOTA)

### Log/Tsallis/Risk-Aware MPPI

- [ ] #401 Log-MPPI 컨트롤러
  * log_mppi.py - LogMPPIController
  * log-space softmax 가중치 (참조 구현)
  * Vanilla와 동등성 검증

- [ ] #402 Tsallis-MPPI 컨트롤러
  * tsallis_mppi.py - TsallisMPPIController
  * q-exponential 가중치
  * utils.py에 q_exponential, q_logarithm 추가
  * tsallis_q 파라미터 (기본 1.0)

- [ ] #403 Tsallis q 파라미터 비교 데모
  * examples/tsallis_mppi_demo.py
  * q=0.5, 1.0, 1.2, 1.5 비교
  * --live 실시간 시뮬레이션

- [ ] #404 Risk-Aware MPPI 컨트롤러
  * risk_aware_mppi.py - RiskAwareMPPIController
  * CVaR 가중치 절단 (alpha)
  * cvar_alpha 파라미터 (기본 1.0)

- [ ] #405 Risk-Aware 장애물 회피 데모
  * examples/risk_aware_mppi_demo.py
  * alpha별 회피 전략 비교
  * --live 실시간 시뮬레이션

### Stein Variational MPPI

- [ ] #406 Stein Variational MPPI 컨트롤러
  * stein_variational_mppi.py - SteinVariationalMPPIController
  * SVGD 기반 샘플 다양성
  * utils.py에 rbf_kernel, median_bandwidth 추가
  * svgd_num_iterations 파라미터

- [ ] #407 SVMPC iteration 비교 데모
  * examples/stein_variational_mppi_demo.py
  * SVGD iteration 수별 성능 비교
  * --live 실시간 시뮬레이션

### Smooth/Spline/SVG MPPI (M3.5 확장 변형)

- [ ] #408 Smooth MPPI 컨트롤러
  * smooth_mppi.py - SmoothMPPIController
  * Δu input-lifting 구조
  * cumsum 복원 로직
  * jerk cost (ΔΔu 페널티)

- [ ] #409 Smooth MPPI jerk weight 비교 데모
  * examples/smooth_mppi_demo.py
  * Vanilla vs SMPPI 제어 변화율 비교
  * --live 실시간 시뮬레이션

- [ ] #410 Spline-MPPI 컨트롤러
  * spline_mppi.py - SplineMPPIController
  * P개 knot 노이즈 → B-spline basis 보간
  * utils.py에 _bspline_basis (de Boor 재귀)
  * spline_num_knots, spline_degree 파라미터

- [ ] #411 Spline-MPPI knot 수 비교 데모
  * examples/spline_mppi_demo.py
  * P=4 vs P=8 비교
  * --live 실시간 시뮬레이션

- [ ] #412 SVG-MPPI 컨트롤러
  * svg_mppi.py - SVGMPPIController
  * G guide particle SVGD + follower resampling
  * svg_num_guide_particles, svg_guide_step_size 파라미터
  * SVMPC 상속 (고속화)

- [ ] #413 SVG-MPPI 장애물 회피 데모
  * examples/svg_mppi_demo.py
  * SVG vs SVMPC 비교 (계산 속도, 성능)
  * --live 실시간 시뮬레이션

### MPPI 전체 벤치마크

- [ ] #414 MPPI 전체 변형 벤치마크 도구
  * examples/mppi_all_variants_benchmark.py
  * 9종 변형 동시 비교
  * --trajectory {circle,figure8,sine} 선택
  * ASCII 요약 테이블 + 6패널 차트

---

## 📚 Documentation (P2)

- [ ] #501 MODEL_TYPES 문서 작성
  * docs/mppi/MODEL_TYPES.md
  * 3가지 모델 타입 설명 (Kinematic/Dynamic/Learned)
  * 인터페이스 사용법 및 예제

- [ ] #502 PRD 문서 업데이트
  * docs/mppi/PRD.md 업데이트
  * 모델 분류 체계 반영
  * 아키텍처 다이어그램 추가

- [ ] #503 MPPI 기술 가이드
  * docs/mppi/MPPI_GUIDE.md
  * 알고리즘 상세 설명
  * 논문 참조 및 수식

- [ ] #504 API 문서 자동 생성
  * Sphinx 설정
  * docstring 작성 규칙
  * 자동 빌드 스크립트

- [ ] #505 튜토리얼 작성
  * docs/tutorials/getting_started.md
  * docs/tutorials/custom_model.md - 커스텀 모델 작성법
  * docs/tutorials/custom_cost_function.md
  * docs/tutorials/tuning_guide.md

- [ ] #506 README 작성
  * 프로젝트 소개
  * Quick Start
  * 예제 실행 방법
  * 모델 타입 선택 가이드

---

## 🚀 ROS2 Integration (P1)

- [ ] #601 ROS2 패키지 구조
  * package.xml, setup.py 작성
  * colcon 빌드 설정
  * launch 파일

- [ ] #602 ROS2 기본 노드 구현
  * ros2/mppi_node.py - MPPI ROS2 wrapper
  * geometry_msgs/Twist 퍼블리시
  * nav_msgs/Odometry 서브스크라이브
  * nav_msgs/Path 레퍼런스 서브스크라이브
  * 모든 모델 타입 지원 (파라미터로 선택)

- [ ] #603 RVIZ 시각화 마커
  * ros2/mppi_rviz_visualizer.py
  * 샘플 궤적 (MarkerArray, 투명도)
  * 가중 평균 궤적 (Path, 시안)
  * 비용 히트맵 (Marker)
  * Tube 경계 (Tube-MPPI)

- [ ] #604 nav2 Controller 플러그인 (Python prototype)
  * nav2 호환 인터페이스
  * ComputeVelocityCommands 구현
  * 파라미터 서버 연동

- [ ] #605 동적 장애물 회피
  * sensor_msgs/LaserScan 처리
  * 실시간 장애물 맵 업데이트
  * ObstacleCost 동적 연동

- [ ] #606 실시간 경로 재계획
  * RealtimeReplanner 클래스
  * 충돌 위험 감지
  * 웨이포인트 재생성

- [ ] #607 ROS2 통합 테스트
  * launch 기반 통합 테스트
  * Gazebo 시뮬레이션 연동
  * 실제 로봇 테스트

---

## ⚡ Performance Optimization (P1)

- [ ] #701 GPU 가속 (CuPy/JAX)
  * rollout 병렬화
  * cost 계산 병렬화
  * K=4096+ 대규모 샘플 지원

- [ ] #702 SVMPC GPU 가속
  * pairwise kernel CUDA 가속
  * O(K²D) 연산 최적화

- [ ] #703 C++ MPPI 코어 변환
  * Python → C++ 포팅
  * Eigen 기반 배치 처리
  * pybind11 바인딩

- [ ] #704 C++ nav2 Controller 플러그인
  * nav2_core::Controller 상속
  * 실시간 성능 검증 (< 100ms)
  * 파라미터 YAML 설정

- [ ] #705 성능 프로파일링
  * cProfile 분석
  * 병목 지점 식별
  * 최적화 적용

---

## 🧪 Additional Robot Models (P2)

- [ ] #801 Swerve Drive 모델
  * models/kinematic/swerve_drive_kinematic.py
  * models/dynamic/swerve_drive_dynamic.py
  * 4륜 독립 조향

- [ ] #802 Non-coaxial Swerve 모델
  * models/kinematic/non_coaxial_swerve_kinematic.py
  * models/dynamic/non_coaxial_swerve_dynamic.py
  * 비동축 스워브

- [ ] #803 Ackermann 조향 모델
  * models/kinematic/ackermann_kinematic.py
  * models/dynamic/ackermann_dynamic.py
  * 자동차형 로봇

- [ ] #804 Omnidirectional 로봇 모델
  * models/kinematic/omnidirectional_kinematic.py
  * models/dynamic/omnidirectional_dynamic.py
  * Mecanum/Omni wheel

---

## 🐛 Bug Fixes (P2)

- [ ] #901 각도 정규화 엣지 케이스
  * ±π 경계 처리
  * 각도 차이 계산 안정화

- [ ] #902 고속 주행 오버슈트 개선
  * 제어 게인 튜닝
  * 예측 호라이즌 조정

- [ ] #903 수치 안정성 검증
  * NaN/Inf 체크 로직
  * log-space 연산 안정화

---

## 🐳 DevOps (P2)

- [ ] #951 Docker 컨테이너화
  * Dockerfile 작성
  * docker-compose.yml
  * 재현 가능한 환경

- [ ] #952 CI/CD 파이프라인
  * GitHub Actions
  * 자동 테스트
  * 자동 배포

- [ ] #953 Claude Issue Watcher 설치
  * .claude/scripts/issue-watcher.sh
  * systemd service 설정
  * 자동 PR 생성

- [ ] #954 TODO Worker 스크립트
  * .claude/scripts/todo-worker.sh
  * claude-todo-worker, claude-todo-task, claude-todo-all
  * ~/.local/bin/ 설치

---

## ✅ Completed

### 2026-02-07

#### Phase 1 (M1) - 기구학 모델 및 Vanilla MPPI ✓

- [x] #000 프로젝트 저장소 초기화
  * Git 저장소 생성
  * .claude 디렉토리 설정
  * CLAUDE.md, TODO.md 작성

- [x] #001 프로젝트 기본 구조 설정
  * Python 패키지 구조 생성
  * pyproject.toml, requirements.txt 작성
  * 30개 파일 생성, 4273줄 코드

- [x] #002-#011 Phase 1 핵심 구현
  * 추상 베이스 클래스 (RobotModel)
  * Differential Drive 기구학 모델
  * MPPI 컨트롤러 전체 인프라
  * 시뮬레이션 도구 (Simulator, Visualizer, Metrics)

- [x] #014 원형 궤적 데모 및 검증
  * Position RMSE: 0.0060m ✓ (목표: < 0.2m, **33배 우수**)
  * Solve Time: 4.96ms ✓ (목표: < 100ms, **20배 빠름**)
  * 커밋: `ede08f8` - feat: Phase 1 (M1) 완료

---

## 사용 방법

### 다음 작업 하나 처리
```bash
claude-todo-worker
```

### 특정 작업 처리
```bash
claude-todo-task "#001"
```

### 모든 작업 연속 처리
```bash
claude-todo-all
```

---

## 우선순위 기준

- **P0 (High)**: Phase 1 - 기구학 모델 및 시뮬레이션 인프라 (M1)
- **P1 (Medium)**: Phase 2, 3 - 동역학/학습 모델, ROS2 통합 (M2/M3/M4)
- **P2 (Low)**: Phase 4, MPPI 변형, 문서화, DevOps (추가 기능)

---

## Phase별 예상 타임라인

```
Phase 1 (M1) - 기구학 모델 및 시뮬레이션: 2주 (P0)
Phase 2 (M2) - 동역학 모델: 2주 (P1)
Phase 3 (M3) - 학습 모델: 1-2주 (P1)
Phase 4 - pytorch_mppi 개선: 1주 (P2, 선택적)

총 6-7주 예상
```

---

## 핵심 설계 원칙

1. **통일된 인터페이스**: RobotModel 추상 클래스로 모든 타입 (Kinematic/Dynamic/Learned) 통합
2. **점진적 확장**: Phase 1(기구학) → Phase 2(동역학) → Phase 3(학습)
3. **하위 호환성**: BatchDynamicsWrapper는 모든 모델 타입 지원, MPPI 수정 불필요
4. **벡터화 강제**: NumPy broadcasting, GPU 준비
5. **함수/클래스 하이브리드**: pytorch_mppi 유연성 + 타입 안정성

---

## 작업 규칙

1. 각 작업은 독립적인 기능 단위
2. 작업 완료 시 테스트 필수 (pytest)
3. PR 생성 및 리뷰 후 머지
4. TODO.md 업데이트는 자동으로 처리됨
5. ASCII 플로우로 진행 상황 시각화
