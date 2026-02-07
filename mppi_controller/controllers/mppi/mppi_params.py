"""
MPPI 파라미터 데이터클래스

모든 MPPI 컨트롤러가 사용하는 파라미터 정의.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class MPPIParams:
    """
    MPPI 컨트롤러 파라미터

    Attributes:
        N: 예측 호라이즌 (타임스텝)
        dt: 타임스텝 간격 (초)
        K: 샘플 궤적 수
        lambda_: 온도 파라미터 (작을수록 최적 궤적에 집중)
        sigma: 제어 노이즈 표준편차 (nu,) 또는 스칼라
        Q: 상태 추적 비용 가중치 (nx,) 또는 (nx, nx)
        R: 제어 노력 비용 가중치 (nu,) 또는 (nu, nu)
        Qf: 터미널 상태 비용 가중치 (nx,) 또는 (nx, nx)
        u_min: 제어 입력 하한 (nu,) - None이면 모델의 제약 사용
        u_max: 제어 입력 상한 (nu,) - None이면 모델의 제약 사용
        device: 'cpu' 또는 'cuda' (GPU 가속용)
    """

    # 기본 파라미터
    N: int = 30  # 호라이즌
    dt: float = 0.05  # 50ms
    K: int = 1024  # 샘플 수

    # 온도 및 노이즈
    lambda_: float = 1.0  # 온도 파라미터
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))  # 제어 노이즈

    # 비용 함수 가중치
    Q: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 1.0])
    )  # [x, y, θ] 가중치
    R: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.1])
    )  # [v, ω] 가중치
    Qf: Optional[np.ndarray] = None  # 터미널 비용 (None이면 Q 사용)

    # 제어 제약 (None이면 모델 제약 사용)
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None

    # 디바이스 설정
    device: str = "cpu"  # 'cpu' or 'cuda'

    def __post_init__(self):
        """파라미터 검증 및 자동 설정"""
        # sigma를 ndarray로 변환
        if not isinstance(self.sigma, np.ndarray):
            self.sigma = np.array([self.sigma])

        # Q, R, Qf를 ndarray로 변환
        if not isinstance(self.Q, np.ndarray):
            self.Q = np.array(self.Q)
        if not isinstance(self.R, np.ndarray):
            self.R = np.array(self.R)

        # Qf가 None이면 Q와 동일하게 설정
        if self.Qf is None:
            self.Qf = self.Q.copy()
        elif not isinstance(self.Qf, np.ndarray):
            self.Qf = np.array(self.Qf)

        # u_min, u_max를 ndarray로 변환
        if self.u_min is not None and not isinstance(self.u_min, np.ndarray):
            self.u_min = np.array(self.u_min)
        if self.u_max is not None and not isinstance(self.u_max, np.ndarray):
            self.u_max = np.array(self.u_max)

        # 파라미터 검증
        assert self.N > 0, "N must be positive"
        assert self.dt > 0, "dt must be positive"
        assert self.K > 0, "K must be positive"
        assert self.lambda_ > 0, "lambda_ must be positive"
        assert np.all(self.sigma > 0), "sigma must be positive"
        assert np.all(self.Q >= 0), "Q must be non-negative"
        assert np.all(self.R >= 0), "R must be non-negative"
        assert np.all(self.Qf >= 0), "Qf must be non-negative"

        if self.u_min is not None and self.u_max is not None:
            assert np.all(self.u_min < self.u_max), "u_min must be less than u_max"

    def get_control_bounds(self):
        """제어 제약 반환 (있을 경우)"""
        if self.u_min is not None and self.u_max is not None:
            return (self.u_min, self.u_max)
        return None

    def __repr__(self) -> str:
        return (
            f"MPPIParams("
            f"N={self.N}, dt={self.dt}, K={self.K}, "
            f"lambda_={self.lambda_:.2f}, "
            f"device={self.device})"
        )


@dataclass
class TubeMPPIParams(MPPIParams):
    """
    Tube-MPPI 전용 추가 파라미터

    Attributes:
        tube_enabled: Tube-MPPI 활성화 (False면 Vanilla MPPI)
        K_fb: 피드백 게인 행렬 (nu, nx)
        tube_margin: Tube 마진 (m)
    """

    tube_enabled: bool = True
    K_fb: Optional[np.ndarray] = None  # 피드백 게인
    tube_margin: float = 0.1  # Tube 마진 (m)

    def __post_init__(self):
        super().__post_init__()
        if self.K_fb is not None and not isinstance(self.K_fb, np.ndarray):
            self.K_fb = np.array(self.K_fb)


@dataclass
class LogMPPIParams(MPPIParams):
    """
    Log-MPPI 전용 추가 파라미터

    Attributes:
        use_baseline: Baseline 적용 (최소 비용으로 정규화)
    """

    use_baseline: bool = True

    def __post_init__(self):
        super().__post_init__()


@dataclass
class TsallisMPPIParams(MPPIParams):
    """
    Tsallis-MPPI 전용 추가 파라미터

    Attributes:
        tsallis_q: Tsallis 엔트로피 파라미터 (1.0이면 Vanilla MPPI)
    """

    tsallis_q: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.tsallis_q > 0, "tsallis_q must be positive"


@dataclass
class RiskAwareMPPIParams(MPPIParams):
    """
    Risk-Aware MPPI 전용 추가 파라미터

    Attributes:
        cvar_alpha: CVaR 위험 파라미터 (0~1, 1이면 Vanilla MPPI)
    """

    cvar_alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.cvar_alpha <= 1, "cvar_alpha must be in (0, 1]"


@dataclass
class SteinVariationalMPPIParams(MPPIParams):
    """
    Stein Variational MPPI 전용 추가 파라미터

    Attributes:
        svgd_num_iterations: SVGD 반복 횟수
        svgd_step_size: SVGD 스텝 크기
    """

    svgd_num_iterations: int = 10
    svgd_step_size: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        assert self.svgd_num_iterations > 0, "svgd_num_iterations must be positive"
        assert self.svgd_step_size > 0, "svgd_step_size must be positive"


@dataclass
class SmoothMPPIParams(MPPIParams):
    """
    Smooth MPPI 전용 추가 파라미터

    Attributes:
        jerk_weight: Jerk 비용 가중치 (ΔΔu 페널티)
    """

    jerk_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.jerk_weight >= 0, "jerk_weight must be non-negative"


@dataclass
class SplineMPPIParams(MPPIParams):
    """
    Spline-MPPI 전용 추가 파라미터

    Attributes:
        spline_num_knots: B-spline knot 개수
        spline_degree: B-spline 차수
    """

    spline_num_knots: int = 8
    spline_degree: int = 3

    def __post_init__(self):
        super().__post_init__()
        assert self.spline_num_knots > 0, "spline_num_knots must be positive"
        assert self.spline_degree > 0, "spline_degree must be positive"
        assert (
            self.spline_num_knots > self.spline_degree
        ), "spline_num_knots must be greater than spline_degree"


@dataclass
class SVGMPPIParams(SteinVariationalMPPIParams):
    """
    SVG-MPPI 전용 추가 파라미터

    Attributes:
        svg_num_guide_particles: Guide particle 개수
        svg_guide_step_size: Guide particle 스텝 크기
    """

    svg_num_guide_particles: int = 10
    svg_guide_step_size: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.svg_num_guide_particles > 0
        ), "svg_num_guide_particles must be positive"
        assert self.svg_guide_step_size > 0, "svg_guide_step_size must be positive"


@dataclass
class CBFMPPIParams(MPPIParams):
    """
    CBF-MPPI 전용 추가 파라미터

    Attributes:
        cbf_obstacles: 장애물 리스트 [(x, y, radius), ...]
        cbf_weight: CBF 위반 비용 가중치
        cbf_alpha: Class-K function 파라미터 (0 < alpha <= 1)
        cbf_safety_margin: 추가 안전 마진 (m)
        cbf_use_safety_filter: QP 안전 필터 사용 여부
    """

    cbf_obstacles: List[tuple] = field(default_factory=list)
    cbf_weight: float = 1000.0
    cbf_alpha: float = 0.1
    cbf_safety_margin: float = 0.1
    cbf_use_safety_filter: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.cbf_alpha <= 1.0, "cbf_alpha must be in (0, 1]"
        assert self.cbf_weight >= 0, "cbf_weight must be non-negative"
        assert self.cbf_safety_margin >= 0, "cbf_safety_margin must be non-negative"


@dataclass
class ShieldMPPIParams(CBFMPPIParams):
    """
    Shield-MPPI 전용 추가 파라미터

    Rollout 중 매 timestep마다 CBF 제약을 해석적으로 적용하여
    모든 K개 샘플 궤적이 안전하도록 보장.

    Attributes:
        shield_enabled: Shield 기능 활성화 (False면 CBF-MPPI 폴백)
        shield_cbf_alpha: Shield용 CBF alpha (None이면 cbf_alpha 사용)
    """

    shield_enabled: bool = True
    shield_cbf_alpha: Optional[float] = None  # None이면 cbf_alpha 사용

    def __post_init__(self):
        super().__post_init__()
        if self.shield_cbf_alpha is not None:
            assert 0 < self.shield_cbf_alpha <= 1.0, \
                "shield_cbf_alpha must be in (0, 1]"
