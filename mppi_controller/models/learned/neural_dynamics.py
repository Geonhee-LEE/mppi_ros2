"""
Neural Network 기반 동역학 모델 (스켈레톤)

향후 PyTorch로 신경망 학습 파이프라인 연동 예정.
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class NeuralDynamics(RobotModel):
    """
    신경망 기반 동역학 모델 (스켈레톤)

    순수 데이터 기반 동역학 학습:
        dx/dt = NN(x, u; θ)

    장점:
        - 복잡한 비선형 동역학 표현 가능
        - End-to-end 학습
        - 모델링 가정 불필요

    단점:
        - 데이터 요구량 높음
        - 외삽 불안정
        - 물리 법칙 보장 없음

    향후 구현 계획:
        - PyTorch MLP/RNN 모델
        - 학습 데이터 수집 파이프라인
        - 온라인 학습 (fine-tuning)
        - Physics-informed NN (PINN)

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        network_config: 신경망 설정 (hidden_dims, activation 등)
        model_path: 학습된 모델 경로 (체크포인트)
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        network_config: Optional[dict] = None,
        model_path: Optional[str] = None,
    ):
        self._state_dim = state_dim
        self._control_dim = control_dim
        self.network_config = network_config or {
            "hidden_dims": [64, 64],
            "activation": "relu",
        }
        self.model_path = model_path

        # TODO: PyTorch 모델 로드
        # if model_path is not None:
        #     self.model = torch.load(model_path)
        # else:
        #     self.model = self._build_network()

        print(
            "[NeuralDynamics] Skeleton implementation. "
            "Full PyTorch integration planned for future."
        )

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    @property
    def model_type(self) -> str:
        return "learned"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        신경망 forward pass: dx/dt = NN(x, u)

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            state_dot: (nx,) 또는 (batch, nx)
        """
        # TODO: PyTorch forward pass
        # with torch.no_grad():
        #     state_tensor = torch.from_numpy(state).float()
        #     control_tensor = torch.from_numpy(control).float()
        #     input_tensor = torch.cat([state_tensor, control_tensor], dim=-1)
        #     output_tensor = self.model(input_tensor)
        #     state_dot = output_tensor.numpy()

        # 현재: 더미 구현 (zero dynamics)
        if state.ndim == 1:
            return np.zeros(self._state_dim)
        else:
            batch_size = state.shape[0]
            return np.zeros((batch_size, self._state_dim))

    def _build_network(self):
        """신경망 아키텍처 생성 (TODO)"""
        # TODO: PyTorch nn.Sequential 또는 커스텀 모델
        pass

    def train(self, train_data: dict, val_data: dict, epochs: int = 100):
        """
        신경망 학습 (TODO)

        Args:
            train_data: {"states": (N, nx), "controls": (N, nu), "next_states": (N, nx)}
            val_data: 검증 데이터
            epochs: 에폭 수
        """
        # TODO: PyTorch training loop
        pass

    def save(self, path: str):
        """모델 저장 (TODO)"""
        # TODO: torch.save(self.model.state_dict(), path)
        pass

    def load(self, path: str):
        """모델 로드 (TODO)"""
        # TODO: self.model.load_state_dict(torch.load(path))
        pass

    def __repr__(self) -> str:
        return (
            f"NeuralDynamics("
            f"state_dim={self._state_dim}, "
            f"control_dim={self._control_dim}, "
            f"config={self.network_config})"
        )
