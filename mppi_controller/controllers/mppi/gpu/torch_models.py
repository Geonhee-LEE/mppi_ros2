"""
PyTorch GPU Models for Ackermann and Swerve Drive

Torch implementations matching the numpy RobotModel counterparts.
Used by TorchDynamicsWrapper for GPU-accelerated MPPI rollouts.
"""

import torch


class TorchAckermannKinematic:
    """
    Ackermann Kinematic (Bicycle Model) — GPU tensor ops

    State: [x, y, theta, delta]
    Control: [v, phi]
    """

    def __init__(self, wheelbase=0.5, device="cuda"):
        self.wheelbase = wheelbase
        self.device = torch.device(device)

    def forward_dynamics(self, state, control):
        theta = state[..., 2]
        delta = state[..., 3]
        v = control[..., 0]
        phi = control[..., 1]

        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = v * torch.tan(delta) / self.wheelbase
        delta_dot = phi

        return torch.stack([x_dot, y_dot, theta_dot, delta_dot], dim=-1)

    def step_rk4(self, state, control, dt):
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class TorchAckermannDynamic:
    """
    Ackermann Dynamic (Bicycle + Friction) — GPU tensor ops

    State: [x, y, theta, v, delta]
    Control: [a, phi]
    """

    def __init__(self, wheelbase=0.5, c_v=0.1, device="cuda"):
        self.wheelbase = wheelbase
        self.c_v = c_v
        self.device = torch.device(device)

    def forward_dynamics(self, state, control):
        theta = state[..., 2]
        v = state[..., 3]
        delta = state[..., 4]
        a = control[..., 0]
        phi = control[..., 1]

        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = v * torch.tan(delta) / self.wheelbase
        v_dot = a - self.c_v * v
        delta_dot = phi

        return torch.stack([x_dot, y_dot, theta_dot, v_dot, delta_dot], dim=-1)

    def step_rk4(self, state, control, dt):
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class TorchSwerveDriveKinematic:
    """
    Swerve Drive Kinematic (Omnidirectional) — GPU tensor ops

    State: [x, y, theta]
    Control: [vx, vy, omega]
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)

    def forward_dynamics(self, state, control):
        theta = state[..., 2]
        vx = control[..., 0]
        vy = control[..., 1]
        omega = control[..., 2]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x_dot = vx * cos_t - vy * sin_t
        y_dot = vx * sin_t + vy * cos_t
        theta_dot = omega

        return torch.stack([x_dot, y_dot, theta_dot], dim=-1)

    def step_rk4(self, state, control, dt):
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class TorchSwerveDriveDynamic:
    """
    Swerve Drive Dynamic (Omnidirectional + Friction) — GPU tensor ops

    State: [x, y, theta, vx, vy, omega]
    Control: [ax, ay, alpha]
    """

    def __init__(self, c_v=0.1, c_omega=0.1, device="cuda"):
        self.c_v = c_v
        self.c_omega = c_omega
        self.device = torch.device(device)

    def forward_dynamics(self, state, control):
        theta = state[..., 2]
        vx = state[..., 3]
        vy = state[..., 4]
        omega = state[..., 5]
        ax = control[..., 0]
        ay = control[..., 1]
        alpha = control[..., 2]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x_dot = vx * cos_t - vy * sin_t
        y_dot = vx * sin_t + vy * cos_t
        theta_dot = omega
        vx_dot = ax - self.c_v * vx
        vy_dot = ay - self.c_v * vy
        omega_dot = alpha - self.c_omega * omega

        return torch.stack(
            [x_dot, y_dot, theta_dot, vx_dot, vy_dot, omega_dot], dim=-1
        )

    def step_rk4(self, state, control, dt):
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
