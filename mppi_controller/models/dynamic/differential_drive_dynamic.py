"""
Differential Drive 동역학 모델

상태: [x, y, θ, v, ω] (5차원)
제어: [a, α] (2차원) - 선가속도, 각가속도

마찰 및 관성 고려.
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class DifferentialDriveDynamic(RobotModel):
    """
    Differential Drive 동역학 모델

    차동 구동 로봇의 동역학 모델. 질량, 관성, 마찰을 고려.

    상태:
        x (m): 월드 프레임 x 위치
        y (m): 월드 프레임 y 위치
        θ (rad): 월드 프레임 heading 각도
        v (m/s): 선속도 (로봇 body frame x 방향)
        ω (rad/s): 각속도 (로봇 body frame z 축 회전)

    제어:
        a (m/s²): 선가속도
        α (rad/s²): 각가속도

    동역학 (마찰/관성 고려):
        dx/dt = v * cos(θ)
        dy/dt = v * sin(θ)
        dθ/dt = ω
        dv/dt = a - c_v * v          # 선형 마찰
        dω/dt = α - c_ω * ω          # 각 마찰

    물리 파라미터:
        mass: 로봇 질량 (kg)
        inertia: 로봇 관성 모멘트 (kg·m²)
        c_v: 선형 마찰 계수 (1/s)
        c_omega: 각 마찰 계수 (1/s)

    Args:
        mass: 로봇 질량 (kg)
        inertia: 관성 모멘트 (kg·m²)
        c_v: 선형 마찰 계수 (1/s)
        c_omega: 각 마찰 계수 (1/s)
        a_max: 최대 선가속도 (m/s²)
        alpha_max: 최대 각가속도 (rad/s²)
        v_max: 최대 선속도 (m/s)
        omega_max: 최대 각속도 (rad/s)
    """

    def __init__(
        self,
        mass: float = 10.0,
        inertia: float = 1.0,
        c_v: float = 0.1,
        c_omega: float = 0.1,
        a_max: float = 2.0,
        alpha_max: float = 2.0,
        v_max: float = 2.0,
        omega_max: float = 2.0,
    ):
        self.mass = mass
        self.inertia = inertia
        self.c_v = c_v
        self.c_omega = c_omega
        self.a_max = a_max
        self.alpha_max = alpha_max
        self.v_max = v_max
        self.omega_max = omega_max

        # 제어 제약 (가속도)
        self._control_lower = np.array([-a_max, -alpha_max])
        self._control_upper = np.array([a_max, alpha_max])

    @property
    def state_dim(self) -> int:
        return 5  # [x, y, θ, v, ω]

    @property
    def control_dim(self) -> int:
        return 2  # [a, α]

    @property
    def model_type(self) -> str:
        return "dynamic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        동역학: dx/dt = f(x, u)

        Args:
            state: (5,) 또는 (batch, 5) - [x, y, θ, v, ω]
            control: (2,) 또는 (batch, 2) - [a, α]

        Returns:
            state_dot: (5,) 또는 (batch, 5) - [dx/dt, dy/dt, dθ/dt, dv/dt, dω/dt]
        """
        # 벡터화 지원: state[..., i]는 마지막 차원 인덱싱
        theta = state[..., 2]
        v = state[..., 3]
        omega = state[..., 4]

        a = control[..., 0]
        alpha = control[..., 1]

        # 위치 동역학 (기구학)
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega

        # 속도 동역학 (마찰 포함)
        v_dot = a - self.c_v * v
        omega_dot = alpha - self.c_omega * omega

        # 스택하여 반환
        return np.stack([x_dot, y_dot, theta_dot, v_dot, omega_dot], axis=-1)

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """제어 제약 반환 (가속도)"""
        return (self._control_lower, self._control_upper)

    def state_to_dict(self, state: np.ndarray) -> dict:
        """
        상태를 딕셔너리로 변환 (디버깅/시각화용)

        Args:
            state: (5,) - [x, y, θ, v, ω]

        Returns:
            {"x": x, "y": y, "theta": theta, "v": v, "omega": omega}
        """
        return {
            "x": state[0],
            "y": state[1],
            "theta": state[2],
            "v": state[3],
            "omega": state[4],
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        상태 정규화

        - θ를 [-π, π] 범위로
        - v, ω를 최대값으로 클리핑

        Args:
            state: (5,) 또는 (batch, 5)

        Returns:
            normalized_state: (5,) 또는 (batch, 5)
        """
        normalized = state.copy()

        # θ를 [-π, π] 범위로 정규화
        normalized[..., 2] = np.arctan2(
            np.sin(state[..., 2]), np.cos(state[..., 2])
        )

        # v를 [-v_max, v_max] 범위로 클리핑 (후진 허용)
        normalized[..., 3] = np.clip(state[..., 3], -self.v_max, self.v_max)

        # ω를 [-omega_max, omega_max] 범위로 클리핑
        normalized[..., 4] = np.clip(state[..., 4], -self.omega_max, self.omega_max)

        return normalized

    def compute_energy(self, state: np.ndarray) -> float:
        """
        운동 에너지 계산 (검증용)

        E = 0.5 * m * v² + 0.5 * I * ω²

        Args:
            state: (5,) - [x, y, θ, v, ω]

        Returns:
            energy: 운동 에너지 (J)
        """
        v = state[3]
        omega = state[4]

        linear_energy = 0.5 * self.mass * v**2
        angular_energy = 0.5 * self.inertia * omega**2

        return linear_energy + angular_energy

    def __repr__(self) -> str:
        return (
            f"DifferentialDriveDynamic("
            f"mass={self.mass}, inertia={self.inertia}, "
            f"c_v={self.c_v}, c_omega={self.c_omega}, "
            f"a_max={self.a_max}, alpha_max={self.alpha_max})"
        )
