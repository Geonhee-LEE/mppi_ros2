"""
앙상블 신경망 기반 동역학 모델

M개 MLP의 평균으로 예측, 분산으로 불확실성 정량화.
"""

import numpy as np
import torch
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple, Dict, List


class EnsembleNeuralDynamics(RobotModel):
    """
    앙상블 신경망 동역학 모델

    M개의 독립 학습된 MLP 앙상블:
        dx/dt = (1/M) Σ NN_m(x, u)    (평균)
        Var = (1/(M-1)) Σ (NN_m - mean)²  (분산 → 불확실성)

    장점:
        - NN에도 불확실성 정량화 가능
        - Risk-Aware MPPI 연동
        - GP보다 빠른 추론 (O(1) vs O(N²))

    사용 예시:
        # EnsembleTrainer로 학습 후 로드
        model = EnsembleNeuralDynamics(
            state_dim=3, control_dim=2,
            model_path="models/learned_models/ensemble.pth"
        )
        mean = model.forward_dynamics(state, control)
        mean, std = model.predict_with_uncertainty(state, control)
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self._state_dim = state_dim
        self._control_dim = control_dim
        self.device = torch.device(device)

        self.models: Optional[List[torch.nn.Module]] = None
        self.norm_stats: Optional[Dict] = None
        self.num_models = 0

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
        앙상블 평균: dx/dt = (1/M) Σ NN_m(x, u)

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            state_dot: (nx,) 또는 (batch, nx) - 앙상블 평균
        """
        mean, _ = self.predict_with_uncertainty(state, control)
        return mean

    def predict_with_uncertainty(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        앙상블 예측 + 불확실성

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            mean: (nx,) 또는 (batch, nx) - 앙상블 평균
            std: (nx,) 또는 (batch, nx) - 앙상블 표준편차
        """
        if self.models is None:
            raise RuntimeError("Models not loaded. Call load_model() first.")

        # Normalize
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

        # Forward pass through all ensemble members
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(inputs_t)
                predictions.append(pred)

        # Stack: (M, batch, nx)
        preds = torch.stack(predictions, dim=0)
        mean_t = preds.mean(dim=0)
        std_t = preds.std(dim=0) if self.num_models > 1 else torch.zeros_like(mean_t)

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
        """Load ensemble from checkpoint"""
        from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        config = checkpoint["config"]
        input_dim = config["state_dim"] + config["control_dim"]
        output_dim = config["state_dim"]
        hidden_dims = config["hidden_dims"]
        activation = config.get("activation", "relu")
        dropout_rate = config.get("dropout_rate", 0.0)

        models_state = checkpoint["models_state_dict"]
        self.num_models = len(models_state)
        self.models = []

        for state_dict in models_state:
            model = DynamicsMLPModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout_rate=dropout_rate,
            ).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)

        self.norm_stats = checkpoint.get("norm_stats")

    def load_model(self, model_path: str):
        """Load model (public method)"""
        self._load_model(model_path)

    def get_model_info(self) -> Dict:
        if self.models is None:
            return {"loaded": False}

        total_params = sum(
            sum(p.numel() for p in m.parameters()) for m in self.models
        )
        return {
            "loaded": True,
            "num_models": self.num_models,
            "num_parameters": total_params,
            "device": str(self.device),
            "normalized": self.norm_stats is not None,
        }

    def __repr__(self) -> str:
        if self.models is not None:
            total_params = sum(
                sum(p.numel() for p in m.parameters()) for m in self.models
            )
            return (
                f"EnsembleNeuralDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"M={self.num_models}, "
                f"params={total_params:,}, "
                f"loaded=True)"
            )
        return (
            f"EnsembleNeuralDynamics("
            f"state_dim={self._state_dim}, "
            f"control_dim={self._control_dim}, "
            f"loaded=False)"
        )
