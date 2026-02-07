# MPPI ROS2 - Claude 개발 가이드

## 개발자 정보 및 선호사항

- **개발 주제**: ROS2 및 모바일 로봇 MPPI 제어
- **언어**: 한국어 사용
- **자동 승인**: 코드 수정 자동 승인, 최종 변경 부분만 요약
- **시각화**: ASCII 아트로 진행상황 및 플로우 표현
- **GitHub 관리**: Issue, PR 자동 생성/관리

## 프로젝트 개요

MPPI (Model Predictive Path Integral) 기반 ROS2 모바일 로봇 제어 시스템

### 핵심 목표
1. **MPPI 컨트롤러 구현** - 샘플링 기반 최적 제어
2. **ROS2 통합** - nav2 플러그인 및 실시간 제어
3. **다양한 로봇 모델 지원** - Differential Drive, Swerve Drive 등
4. **실시간 성능** - C++/GPU 가속 최적화

## 프로젝트 구조

```
mppi_ros2/
├── mppi_controller/              # MPPI 컨트롤러 패키지
│   ├── models/                   # 로봇 동역학 모델
│   │   ├── differential_drive/   # 차동 구동 (v, omega)
│   │   ├── swerve_drive/         # 스워브 구동
│   │   └── non_coaxial_swerve/   # 비동축 스워브
│   ├── controllers/
│   │   ├── mppi/                 # MPPI 알고리즘
│   │   │   ├── base_mppi.py      # Vanilla MPPI
│   │   │   ├── tube_mppi.py      # Tube-MPPI
│   │   │   ├── log_mppi.py       # Log-MPPI
│   │   │   ├── tsallis_mppi.py   # Tsallis-MPPI
│   │   │   ├── risk_aware_mppi.py # CVaR MPPI
│   │   │   ├── stein_variational_mppi.py # SVMPC
│   │   │   ├── smooth_mppi.py    # Smooth MPPI
│   │   │   ├── spline_mppi.py    # Spline-MPPI
│   │   │   ├── svg_mppi.py       # SVG-MPPI
│   │   │   ├── cost_functions.py # 비용 함수
│   │   │   ├── sampling.py       # 노이즈 샘플러
│   │   │   ├── dynamics_wrapper.py # 배치 동역학
│   │   │   └── mppi_params.py    # 파라미터
│   │   └── mpc/                  # MPC (비교용)
│   ├── ros2/                     # ROS2 노드
│   │   ├── mppi_node.py          # MPPI ROS2 노드
│   │   └── mppi_rviz_visualizer.py # RVIZ 시각화
│   ├── simulation/               # 시뮬레이터
│   └── utils/                    # 유틸리티
├── docs/                         # 문서
│   ├── mppi/
│   │   ├── PRD.md                # 제품 요구사항
│   │   └── MPPI_GUIDE.md         # 기술 가이드
│   └── api/                      # API 문서
├── tests/                        # 테스트
├── examples/                     # 예제 및 데모
├── configs/                      # 설정 파일
├── .claude/                      # Claude 설정
│   ├── scripts/                  # 자동화 스크립트
│   │   ├── issue-watcher.sh      # GitHub Issue Watcher
│   │   └── todo-worker.sh        # TODO Worker
│   └── memory/                   # Claude 메모리
├── CLAUDE.md                     # 본 파일
├── TODO.md                       # 작업 목록
└── README.md                     # 프로젝트 README
```

## 개발 가이드라인

### 코드 품질

1. **정확성 우선**
   - 모르는 내용은 절대 지어내지 말고 "해당 정보는 제공된 자료나 제 지식범위를 벗어납니다" 명시
   - 추측 필요시 "추측입니다" 선언

2. **근거 기반 응답**
   - 답변 전 근거 목록 작성
   - 근거의 신뢰성 자체 평가
   - 결론만 따로 요약
   - 근거가 약하면 "정확하지 않을 수 있다" 명시

3. **출처 명시**
   - "~에 근거하면", "일반적으로 ~로 알려져 있다" 스타일
   - 구체적인 연도, 수치, 인명, 지명 → "정확도: 높음/중간/낮음" 표시

4. **질문 검증**
   - 질문이 이상하거나 부정확하면 초안만 작성 후 구체적 질문 요청
   - 항상 비판적 검토, 정확도 향상

### 개발 워크플로우

#### MPPI 알고리즘 계층 구조

```
MPPIController (base_mppi.py) — Vanilla MPPI
├── _compute_weights()         ← 서브클래스 오버라이드 포인트
│
├── TubeMPPIController         ── 외란 강건성
│   └── AncillaryController    ── body frame 피드백
│
├── LogMPPIController          ── log-space softmax
│
├── TsallisMPPIController      ── q-exponential 가중치
│
├── RiskAwareMPPIController    ── CVaR 가중치 절단
│
├── SteinVariationalMPPIController ── SVGD 샘플 다양성
│
├── SmoothMPPIController       ── Δu input-lifting
│
├── SplineMPPIController       ── B-spline 보간
│
└── SVGMPPIController          ── Guide particle SVGD
```

### 인터페이스 규칙

- **모든 컨트롤러**: `compute_control(state, reference_trajectory) -> (control, info)` 시그니처 준수
- **MPPI info dict**: sample_trajectories, sample_weights, best_trajectory, temperature, ess 등
- **Tube-MPPI 추가 info**: nominal_state, feedback_correction, tube_width, tube_boundary

### 구현 우선순위

#### 🔴 High Priority (P0)
- MPPI 핵심 알고리즘 구현
- ROS2 기본 통합
- 성능 검증 (벤치마크)

#### 🟠 Medium Priority (P1)
- GPU 가속
- 추가 MPPI 변형
- 고급 기능 (장애물 회피, 재계획 등)

#### 🟢 Low Priority (P2)
- 문서화
- 웹 대시보드
- 추가 로봇 모델

## Claude 자동화 도구

### 1. GitHub Issue Watcher

```
┌─────────────────────────────────────────────────────────────┐
│                      동작 플로우                            │
├─────────────────────────────────────────────────────────────┤
│  📱 모바일에서 이슈 등록 + 'claude' 라벨                    │
│         ↓                                                   │
│  💻 로컬 머신이 이슈 감지 (30초 폴링)                       │
│         ↓                                                   │
│  🤖 Claude Code 자동 구현                                   │
│         ↓                                                   │
│  📤 자동 커밋 & PR 생성                                     │
│         ↓                                                   │
│  📱 모바일로 알림 (이슈 댓글)                               │
└─────────────────────────────────────────────────────────────┘
```

#### 설치
```bash
cd .claude/scripts
./install-watcher.sh
```

#### 사용
```bash
# 서비스 시작
systemctl --user start claude-watcher

# 상태 확인
systemctl --user status claude-watcher

# 로그 확인
journalctl --user -u claude-watcher -f
```

### 2. TODO Worker

파일 기반 작업 관리 시스템

```bash
# 다음 작업 처리
claude-todo-worker

# 특정 작업 처리
claude-todo-task "#101"

# 모든 작업 연속 처리
claude-todo-all
```

## 마일스톤 로드맵

```
M1: Vanilla MPPI (기본 구현)
├── MPPIParams 데이터클래스
├── BatchDynamicsWrapper
├── 비용 함수 (StateTracking, Obstacle)
├── Gaussian 샘플링
├── MPPI 컨트롤러
└── 기본 테스트

M2: 고도화
├── Colored Noise 샘플링
├── Tube-MPPI
├── Adaptive Temperature
├── ControlRateCost
└── GPU 가속

M3: SOTA 변형
├── Log-MPPI
├── Tsallis-MPPI
├── Risk-Aware MPPI
└── Stein Variational MPPI

M3.5: 확장 변형
├── Smooth MPPI
├── Spline-MPPI
└── SVG-MPPI

M4: ROS2 통합
├── nav2 플러그인
├── 실제 로봇 인터페이스
├── 파라미터 서버
└── RVIZ 시각화

M5: C++ 포팅
├── C++ MPPI 코어
├── Eigen 기반 배치 처리
├── nav2 Controller 플러그인
└── 실시간 성능 검증
```

## 참고 문서

### Claude Code 관련
- [Skills 가이드](https://code.claude.com/docs/ko/skills)
- [Sub-agents 가이드](https://code.claude.com/docs/ko/sub-agents)

### MPPI 관련 논문
- Williams et al. (2016) - "Aggressive Driving with MPPI"
- Williams et al. (2018) - "Robust Sampling Based MPPI" (Tube-MPPI)
- Yin et al. (2021) - "Tsallis Entropy for MPPI"
- Yin et al. (2023) - "Risk-Aware MPPI"
- Lambert et al. (2020) - "Stein Variational Model Predictive Control"
- Kim et al. (2021) - "Smooth MPPI"
- Bhardwaj et al. (2024) - "Spline-MPPI"
- Kondo et al. (2024) - "SVG-MPPI"

## 테스트 및 검증

### 테스트 실행
```bash
# 전체 테스트
pytest tests/ -v

# MPPI 테스트만
pytest tests/test_mppi*.py -v

# 특정 테스트
pytest tests/test_mppi.py::test_circle_tracking -v
```

### 성능 기준
- **위치 추적 RMSE**: < 0.2m (원형 궤적)
- **계산 시간**: < 100ms (K=1024, N=30)
- **실시간성**: 10Hz 제어 주기 유지

## 시각화

### RVIZ 마커
- 샘플 궤적 (투명도 기반)
- 가중 평균 궤적 (시안)
- 비용 히트맵
- Tube 경계 (Tube-MPPI)
- 장애물 영역
- 목표 경로

### 데모 실행
```bash
# Vanilla MPPI
python examples/mppi_basic_demo.py --trajectory circle --live

# MPPI 변형 비교
python examples/mppi_all_variants_benchmark.py --live --trajectory figure8

# Tube-MPPI vs Vanilla
python examples/mppi_vanilla_vs_tube_demo.py --live --noise 1.0
```

## 커밋 및 PR 규칙

### 커밋 메시지 형식
```
{type}: {subject}

{body}

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Types:**
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `docs`: 문서 수정
- `perf`: 성능 개선

### PR 생성
```bash
# 자동 PR 생성 (Claude)
# - 브랜치명: feature/issue-{번호} 또는 feature/{기능명}
# - 제목: 간결하게 (< 70자)
# - 본문: ## Summary, ## Test plan 포함
```

## 디버깅 팁

### MPPI 디버깅
1. `info` dict 확인: sample_weights, ess, temperature
2. 샘플 궤적 시각화: RVIZ에서 분포 확인
3. 비용 함수 값 로깅: 각 비용 컴포넌트 개별 확인
4. 수치 안정성: NaN/Inf 체크

### ROS2 디버깅
```bash
# 노드 상태 확인
ros2 node list
ros2 node info /mppi_controller

# 토픽 확인
ros2 topic list
ros2 topic echo /mppi_controller/control

# 파라미터 확인
ros2 param list /mppi_controller
ros2 param get /mppi_controller lambda_
```

## 성능 프로파일링

### Python 프로파일링
```python
import cProfile
cProfile.run('controller.compute_control(state, ref)', 'profile.prof')
```

### GPU 가속 체크리스트
- [ ] rollout 벡터화 (NumPy/CuPy/JAX)
- [ ] cost 병렬 계산
- [ ] SVGD 커널 연산 CUDA 가속
- [ ] 메모리 프리페칭

## 라이센스

MIT License
