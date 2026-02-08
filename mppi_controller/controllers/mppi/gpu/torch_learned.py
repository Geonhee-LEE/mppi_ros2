"""
PyTorch GPU 학습 동역학 모듈

NeuralDynamics의 nn.Module을 직접 감싸서 torch 텐서 연산으로
numpy↔torch 변환 병목 제거. MPPI GPU 파이프라인에서 사용.
"""

import torch


class TorchNeuralDynamics:
    """
    NeuralDynamics GPU 래퍼 — nn.Module 직접 사용

    NeuralDynamics.forward_dynamics()는 매 호출마다 numpy→torch→numpy 변환이 발생하여
    MPPI에서 N=30 × RK4(4회) = 120회/control 호출 시 큰 오버헤드.

    이 클래스는 nn.Module을 직접 GPU에서 torch 텐서로 연산하여 변환 비용 제거.

    Args:
        neural_model: NeuralDynamics 인스턴스 (model + norm_stats 로드 완료)
        device: torch device 문자열
    """

    def __init__(self, neural_model, device="cuda"):
        self.device = torch.device(device)

        if neural_model.model is None:
            raise RuntimeError(
                "NeuralDynamics model not loaded. Load a model before GPU wrapping."
            )

        # nn.Module을 GPU로 이동
        self.nn_module = neural_model.model.to(self.device)
        self.nn_module.eval()

        # 정규화 통계를 GPU 텐서로 미리 변환
        ns = neural_model.norm_stats
        if ns is not None:
            self.state_mean = torch.tensor(
                ns["state_mean"], device=self.device, dtype=torch.float32
            )
            self.state_std = torch.tensor(
                ns["state_std"], device=self.device, dtype=torch.float32
            )
            self.control_mean = torch.tensor(
                ns["control_mean"], device=self.device, dtype=torch.float32
            )
            self.control_std = torch.tensor(
                ns["control_std"], device=self.device, dtype=torch.float32
            )
            self.dot_mean = torch.tensor(
                ns["state_dot_mean"], device=self.device, dtype=torch.float32
            )
            self.dot_std = torch.tensor(
                ns["state_dot_std"], device=self.device, dtype=torch.float32
            )
            self.has_norm = True
        else:
            self.has_norm = False

    def forward_dynamics(self, state, control):
        """
        연속 시간 동역학: dx/dt = NN(x, u)

        모든 연산이 GPU torch 텐서로 수행됨.

        Args:
            state: (..., nx) torch tensor
            control: (..., nu) torch tensor

        Returns:
            state_dot: (..., nx) torch tensor
        """
        # 정규화
        if self.has_norm:
            state_n = (state - self.state_mean) / self.state_std
            control_n = (control - self.control_mean) / self.control_std
        else:
            state_n = state
            control_n = control

        # [state, control] 결합
        inputs = torch.cat([state_n, control_n], dim=-1)

        # Forward pass (no_grad는 호출부에서 관리)
        output = self.nn_module(inputs)

        # 역정규화
        if self.has_norm:
            output = output * self.dot_std + self.dot_mean

        return output

    def step_rk4(self, state, control, dt):
        """
        RK4 적분 — 배치

        Args:
            state: (K, nx) 현재 상태
            control: (K, nu) 제어 입력
            dt: float 시간 간격

        Returns:
            next_state: (K, nx) 다음 상태
        """
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
