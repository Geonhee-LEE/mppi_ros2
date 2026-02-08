"""
MC-Dropout Bayesian Neural Network 동역학 모델

추론 시 dropout을 활성화하고 M회 forward pass로 불확실성 추정.
앙상블보다 경량 (단일 모델, M회 샘플링).
"""

import numpy as np
import torch
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple, Dict


class MCDropoutDynamics(RobotModel):
    """
    MC-Dropout Bayesian NN 동역학 모델

    단일 MLP에서 dropout을 추론 시에도 활성화하여
    M회 forward pass의 분산으로 불확실성 추정:

        dx/dt_m = NN(x, u; θ, mask_m)   m = 1..M
        mean = (1/M) Σ dx/dt_m
        std  = sqrt((1/(M-1)) Σ (dx/dt_m - mean)²)

    앙상블 대비 장점:
        - 단일 모델 (M배 파라미터 절약)
        - 학습 비용 1회 (앙상블은 M회)
        - dropout_rate가 불확실성 크기 조절

    단점:
        - dropout_rate > 0 필수 (학습 시 포함)
        - 앙상블보다 불확실성 추정 품질 낮을 수 있음

    사용 예시:
        model = MCDropoutDynamics(
            state_dim=3, control_dim=2,
            model_path="models/learned_models/best_model.pth",
            num_samples=20,
        )
        mean = model.forward_dynamics(state, control)
        mean, std = model.predict_with_uncertainty(state, control)

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        model_path: 학습된 모델 경로 (dropout_rate > 0 으로 학습된 체크포인트)
        num_samples: MC 샘플 수 (높을수록 정밀, 느림)
        device: 'cpu' or 'cuda'
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        num_samples: int = 20,
        device: str = "cpu",
    ):
        self._state_dim = state_dim
        self._control_dim = control_dim
        self.num_samples = num_samples
        self.device = torch.device(device)

        self.model = None
        self.norm_stats: Optional[Dict] = None
        self.dropout_rate = 0.0

        if model_path is not None:
            self._load_model(model_path)

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
        MC-Dropout 평균 예측: dx/dt = (1/M) Σ NN(x, u; mask_m)

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            state_dot: (nx,) 또는 (batch, nx) — MC 평균
        """
        mean, _ = self.predict_with_uncertainty(state, control)
        return mean

    def predict_with_uncertainty(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC-Dropout 예측 + 불확실성

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            mean: (nx,) 또는 (batch, nx) — MC 평균
            std: (nx,) 또는 (batch, nx) — MC 표준편차
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Normalize inputs
        if self.norm_stats is not None:
            state_n = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_n = (control - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_n = state
            control_n = control

        # Build input tensor
        if state.ndim == 1:
            inputs = np.concatenate([state_n, control_n])
            inputs_t = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)
        else:
            inputs = np.concatenate([state_n, control_n], axis=1)
            inputs_t = torch.FloatTensor(inputs).to(self.device)

        # MC-Dropout: keep model in train mode to enable dropout
        self.model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(inputs_t)
                predictions.append(pred)

        # Stack: (M, batch, nx)
        preds = torch.stack(predictions, dim=0)
        mean_t = preds.mean(dim=0)
        std_t = preds.std(dim=0) if self.num_samples > 1 else torch.zeros_like(mean_t)

        mean = mean_t.cpu().numpy()
        std = std_t.cpu().numpy()

        # Denormalize
        if self.norm_stats is not None:
            mean = mean * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]
            std = std * self.norm_stats["state_dot_std"]

        # Squeeze
        if state.ndim == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std

    def _load_model(self, model_path: str):
        """Load trained model (must have dropout_rate > 0)"""
        from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        config = checkpoint["config"]
        input_dim = config["state_dim"] + config["control_dim"]
        output_dim = config["state_dim"]
        hidden_dims = config["hidden_dims"]
        activation = config.get("activation", "relu")
        dropout_rate = config.get("dropout_rate", 0.0)

        self.dropout_rate = dropout_rate

        self.model = DynamicsMLPModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.norm_stats = checkpoint.get("norm_stats")

    def load_model(self, model_path: str):
        """Load model (public method)"""
        self._load_model(model_path)

    def get_model_info(self) -> Dict:
        if self.model is None:
            return {"loaded": False}

        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "loaded": True,
            "num_parameters": num_params,
            "num_samples": self.num_samples,
            "dropout_rate": self.dropout_rate,
            "device": str(self.device),
            "normalized": self.norm_stats is not None,
        }

    def __repr__(self) -> str:
        if self.model is not None:
            num_params = sum(p.numel() for p in self.model.parameters())
            return (
                f"MCDropoutDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"params={num_params:,}, "
                f"M={self.num_samples}, "
                f"dropout={self.dropout_rate}, "
                f"loaded=True)"
            )
        return (
            f"MCDropoutDynamics("
            f"state_dim={self._state_dim}, "
            f"control_dim={self._control_dim}, "
            f"loaded=False)"
        )
