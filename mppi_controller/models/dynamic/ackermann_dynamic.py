"""
Ackermann Steering Dynamic Model (Bicycle Model with Friction)

State: [x, y, theta, v, delta] (5D)
Control: [a, phi] (2D) - acceleration, steering rate

Dynamics:
    dx/dt = v * cos(theta)
    dy/dt = v * sin(theta)
    dtheta/dt = v * tan(delta) / L
    dv/dt = a - c_v * v
    ddelta/dt = phi
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class AckermannDynamic(RobotModel):
    """
    Ackermann Steering Dynamic Model (Bicycle Approximation with Friction)

    Extends the kinematic bicycle model with velocity as a state variable
    and linear friction.

    State:
        x (m): world frame x position
        y (m): world frame y position
        theta (rad): heading angle
        v (m/s): longitudinal velocity
        delta (rad): front steering angle

    Control:
        a (m/s^2): longitudinal acceleration
        phi (rad/s): steering rate

    Args:
        wheelbase: distance between front and rear axle (m)
        mass: robot mass (kg)
        c_v: linear friction coefficient (1/s)
        v_max: maximum velocity (m/s)
        a_max: maximum acceleration (m/s^2)
        max_steer: maximum steering angle (rad)
        steer_rate_max: maximum steering rate (rad/s)
    """

    def __init__(
        self,
        wheelbase: float = 0.5,
        mass: float = 10.0,
        c_v: float = 0.1,
        v_max: float = 2.0,
        a_max: float = 2.0,
        max_steer: float = 0.5,
        steer_rate_max: float = 1.0,
    ):
        self.wheelbase = wheelbase
        self.mass = mass
        self.c_v = c_v
        self.v_max = v_max
        self.a_max = a_max
        self.max_steer = max_steer
        self.steer_rate_max = steer_rate_max

        # Control bounds: [a, phi]
        self._control_lower = np.array([-a_max, -steer_rate_max])
        self._control_upper = np.array([a_max, steer_rate_max])

    @property
    def state_dim(self) -> int:
        return 5  # [x, y, theta, v, delta]

    @property
    def control_dim(self) -> int:
        return 2  # [a, phi]

    @property
    def model_type(self) -> str:
        return "dynamic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Dynamic model: dx/dt = f(x, u)

        Args:
            state: (5,) or (batch, 5) - [x, y, theta, v, delta]
            control: (2,) or (batch, 2) - [a, phi]

        Returns:
            state_dot: (5,) or (batch, 5)
        """
        theta = state[..., 2]
        v = state[..., 3]
        delta = state[..., 4]
        a = control[..., 0]
        phi = control[..., 1]

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = v * np.tan(delta) / self.wheelbase
        v_dot = a - self.c_v * v
        delta_dot = phi

        return np.stack(
            [x_dot, y_dot, theta_dot, v_dot, delta_dot], axis=-1
        )

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return control bounds [a, phi]"""
        return (self._control_lower, self._control_upper)

    def state_to_dict(self, state: np.ndarray) -> dict:
        """Convert state to dict for debugging."""
        return {
            "x": state[0],
            "y": state[1],
            "theta": state[2],
            "v": state[3],
            "delta": state[4],
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state:
        - theta to [-pi, pi]
        - v clipped to [-v_max, v_max]
        - delta clipped to [-max_steer, max_steer]
        """
        normalized = state.copy()
        normalized[..., 2] = np.arctan2(
            np.sin(state[..., 2]), np.cos(state[..., 2])
        )
        normalized[..., 3] = np.clip(state[..., 3], -self.v_max, self.v_max)
        normalized[..., 4] = np.clip(
            state[..., 4], -self.max_steer, self.max_steer
        )
        return normalized

    def compute_energy(self, state: np.ndarray) -> float:
        """
        Kinetic energy: E = 0.5 * m * v^2

        Args:
            state: (5,) - [x, y, theta, v, delta]

        Returns:
            energy (J)
        """
        v = state[3]
        return 0.5 * self.mass * v**2

    def __repr__(self) -> str:
        return (
            f"AckermannDynamic("
            f"wheelbase={self.wheelbase}, mass={self.mass}, "
            f"c_v={self.c_v}, a_max={self.a_max})"
        )
