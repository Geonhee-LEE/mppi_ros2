"""
Dynamic Kinematic Adapter — 5D MPPI 내부 모델

5D 상태 [x, y, θ, v, ω]와 velocity command [v_cmd, ω_cmd]를 사용하는
MPPI 컨트롤러용 모델. PD 제어 + 마찰 동역학을 표현하여
기구학(3D)으로는 포착할 수 없는 관성/마찰 효과를 인지한 제어 가능.

Usage:
    # 잘못된 파라미터 (컨트롤러가 추측)
    model = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1)

    # 정확한 파라미터 (Oracle)
    oracle = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)

    # MPPI에 사용
    controller = MPPIController(model, params_5d, cost_function=cost_5d)
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel


class DynamicKinematicAdapter(RobotModel):
    """
    5D MPPI 내부 모델: [x, y, θ, v, ω] + velocity command [v_cmd, ω_cmd].

    PD + friction → state_dot 계산.
    MPPI가 5D state를 롤아웃하면서 관성/마찰 동역학을 인지.

    Args:
        c_v: 선형 마찰 계수
        c_omega: 각속도 마찰 계수
        k_v: PD 선형 속도 게인
        k_omega: PD 각속도 게인
    """

    def __init__(self, c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0):
        self._c_v = c_v
        self._c_omega = c_omega
        self._k_v = k_v
        self._k_omega = k_omega

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(self, state, control):
        """
        state: (..., 5) = [x, y, θ, v, ω]
        control: (..., 2) = [v_cmd, ω_cmd]
        return: state_dot (..., 5)
        """
        theta = state[..., 2]
        v = state[..., 3]
        omega = state[..., 4]

        v_cmd = control[..., 0]
        omega_cmd = control[..., 1]

        # PD + friction
        a = self._k_v * (v_cmd - v) - self._c_v * v
        alpha = self._k_omega * (omega_cmd - omega) - self._c_omega * omega

        # state_dot = [v*cos(θ), v*sin(θ), ω, a, α]
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        state_dot = np.zeros_like(state)
        state_dot[..., 0] = dx
        state_dot[..., 1] = dy
        state_dot[..., 2] = dtheta
        state_dot[..., 3] = a
        state_dot[..., 4] = alpha

        return state_dot

    def get_control_bounds(self):
        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])
        return lower, upper

    def state_to_dict(self, state):
        return {
            "x": state[0], "y": state[1], "theta": state[2],
            "v": state[3], "omega": state[4],
        }

    def normalize_state(self, state):
        result = state.copy()
        result[..., 2] = np.arctan2(np.sin(state[..., 2]), np.cos(state[..., 2]))
        return result
