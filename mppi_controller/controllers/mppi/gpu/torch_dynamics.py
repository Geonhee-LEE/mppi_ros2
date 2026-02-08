"""
PyTorch GPU 동역학 모듈

Differential Drive 기구학을 PyTorch 텐서로 구현하여
K개 샘플을 GPU에서 병렬 rollout.
"""

import torch


class TorchDiffDriveKinematic:
    """
    Differential Drive 기구학 — GPU 텐서 연산

    상태: [x, y, theta]
    제어: [v, omega]

    dx/dt = v * cos(theta)
    dy/dt = v * sin(theta)
    dtheta/dt = omega
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)

    def forward_dynamics(self, state, control):
        """
        연속 시간 동역학: dx/dt = f(x, u)

        Args:
            state: (..., 3) torch tensor [x, y, theta]
            control: (..., 2) torch tensor [v, omega]

        Returns:
            state_dot: (..., 3) torch tensor
        """
        theta = state[..., 2]
        v = control[..., 0]
        omega = control[..., 1]

        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = omega

        return torch.stack([x_dot, y_dot, theta_dot], dim=-1)

    def step_rk4(self, state, control, dt):
        """
        RK4 적분 — 배치

        Args:
            state: (K, 3) 현재 상태
            control: (K, 2) 제어 입력
            dt: float 시간 간격

        Returns:
            next_state: (K, 3) 다음 상태
        """
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class TorchDynamicsWrapper:
    """
    GPU rollout 래퍼: (K, N, nu) 제어 → (K, N+1, nx) 궤적

    각 timestep에서 K개 샘플을 GPU에서 병렬 처리.
    N은 30 정도로 작으므로 루프 오버헤드 무시 가능.
    """

    def __init__(self, torch_model, dt, device="cuda"):
        self.torch_model = torch_model
        self.dt = dt
        self.device = torch.device(device)

    def rollout(self, initial_state, controls):
        """
        GPU rollout

        Args:
            initial_state: (K, nx) torch tensor — 초기 상태 (이미 GPU에 있음)
            controls: (K, N, nu) torch tensor — 제어 시퀀스 (이미 GPU에 있음)

        Returns:
            trajectories: (K, N+1, nx) torch tensor (GPU에 유지)
        """
        K, N, _ = controls.shape
        nx = initial_state.shape[-1]

        trajectories = torch.empty(
            K, N + 1, nx, device=self.device, dtype=torch.float32
        )
        trajectories[:, 0, :] = initial_state

        for t in range(N):
            trajectories[:, t + 1, :] = self.torch_model.step_rk4(
                trajectories[:, t, :], controls[:, t, :], self.dt
            )

        return trajectories
