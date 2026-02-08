"""
Swerve Drive Dynamic Model (Omnidirectional with Friction)

State: [x, y, theta, vx, vy, omega] (6D)
Control: [ax, ay, alpha] (3D) - body-frame accelerations

Dynamics:
    dx/dt = vx * cos(theta) - vy * sin(theta)
    dy/dt = vx * sin(theta) + vy * cos(theta)
    dtheta/dt = omega
    dvx/dt = ax - c_v * vx
    dvy/dt = ay - c_v * vy
    domega/dt = alpha - c_omega * omega
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class SwerveDriveDynamic(RobotModel):
    """
    Swerve Drive Dynamic Model (Omnidirectional with Friction)

    Extends the swerve kinematic model with velocity states and friction.

    State:
        x (m): world frame x position
        y (m): world frame y position
        theta (rad): heading angle
        vx (m/s): body-frame x velocity (forward)
        vy (m/s): body-frame y velocity (lateral)
        omega (rad/s): angular velocity

    Control:
        ax (m/s^2): forward acceleration
        ay (m/s^2): lateral acceleration
        alpha (rad/s^2): angular acceleration

    Args:
        mass: robot mass (kg)
        inertia: moment of inertia (kg*m^2)
        c_v: linear friction coefficient (1/s)
        c_omega: angular friction coefficient (1/s)
        vx_max: maximum forward velocity (m/s)
        vy_max: maximum lateral velocity (m/s)
        omega_max: maximum angular velocity (rad/s)
        ax_max: maximum forward acceleration (m/s^2)
        ay_max: maximum lateral acceleration (m/s^2)
        alpha_max: maximum angular acceleration (rad/s^2)
    """

    def __init__(
        self,
        mass: float = 10.0,
        inertia: float = 1.0,
        c_v: float = 0.1,
        c_omega: float = 0.1,
        vx_max: float = 2.0,
        vy_max: float = 2.0,
        omega_max: float = 2.0,
        ax_max: float = 2.0,
        ay_max: float = 2.0,
        alpha_max: float = 2.0,
    ):
        self.mass = mass
        self.inertia = inertia
        self.c_v = c_v
        self.c_omega = c_omega
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.omega_max = omega_max
        self.ax_max = ax_max
        self.ay_max = ay_max
        self.alpha_max = alpha_max

        # Control bounds: [ax, ay, alpha]
        self._control_lower = np.array([-ax_max, -ay_max, -alpha_max])
        self._control_upper = np.array([ax_max, ay_max, alpha_max])

    @property
    def state_dim(self) -> int:
        return 6  # [x, y, theta, vx, vy, omega]

    @property
    def control_dim(self) -> int:
        return 3  # [ax, ay, alpha]

    @property
    def model_type(self) -> str:
        return "dynamic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Dynamic model: dx/dt = f(x, u)

        Args:
            state: (6,) or (batch, 6) - [x, y, theta, vx, vy, omega]
            control: (3,) or (batch, 3) - [ax, ay, alpha]

        Returns:
            state_dot: (6,) or (batch, 6)
        """
        theta = state[..., 2]
        vx = state[..., 3]
        vy = state[..., 4]
        omega = state[..., 5]
        ax = control[..., 0]
        ay = control[..., 1]
        alpha = control[..., 2]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Position dynamics (body-to-world rotation)
        x_dot = vx * cos_theta - vy * sin_theta
        y_dot = vx * sin_theta + vy * cos_theta
        theta_dot = omega

        # Velocity dynamics (with friction)
        vx_dot = ax - self.c_v * vx
        vy_dot = ay - self.c_v * vy
        omega_dot = alpha - self.c_omega * omega

        return np.stack(
            [x_dot, y_dot, theta_dot, vx_dot, vy_dot, omega_dot], axis=-1
        )

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return control bounds [ax, ay, alpha]"""
        return (self._control_lower, self._control_upper)

    def state_to_dict(self, state: np.ndarray) -> dict:
        """Convert state to dict for debugging."""
        return {
            "x": state[0],
            "y": state[1],
            "theta": state[2],
            "vx": state[3],
            "vy": state[4],
            "omega": state[5],
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state:
        - theta to [-pi, pi]
        - velocities clipped to bounds
        """
        normalized = state.copy()
        normalized[..., 2] = np.arctan2(
            np.sin(state[..., 2]), np.cos(state[..., 2])
        )
        normalized[..., 3] = np.clip(state[..., 3], -self.vx_max, self.vx_max)
        normalized[..., 4] = np.clip(state[..., 4], -self.vy_max, self.vy_max)
        normalized[..., 5] = np.clip(
            state[..., 5], -self.omega_max, self.omega_max
        )
        return normalized

    def compute_energy(self, state: np.ndarray) -> float:
        """
        Kinetic energy: E = 0.5*m*(vx^2+vy^2) + 0.5*I*omega^2

        Args:
            state: (6,) - [x, y, theta, vx, vy, omega]

        Returns:
            energy (J)
        """
        vx = state[3]
        vy = state[4]
        omega = state[5]
        return 0.5 * self.mass * (vx**2 + vy**2) + 0.5 * self.inertia * omega**2

    def __repr__(self) -> str:
        return (
            f"SwerveDriveDynamic("
            f"mass={self.mass}, inertia={self.inertia}, "
            f"c_v={self.c_v}, c_omega={self.c_omega})"
        )
