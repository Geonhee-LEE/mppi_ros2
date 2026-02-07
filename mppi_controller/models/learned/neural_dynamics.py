"""
Neural Network 기반 동역학 모델

PyTorch 학습 모델 통합.
"""

import numpy as np
import torch
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple, Dict


class NeuralDynamics(RobotModel):
    """
    신경망 기반 동역학 모델

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

    사용 예시:
        # 학습된 모델 로드
        neural_model = NeuralDynamics(
            state_dim=3,
            control_dim=2,
            model_path="models/learned_models/best_model.pth"
        )

        # 추론
        state_dot = neural_model.forward_dynamics(state, control)

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        model_path: 학습된 모델 경로 (PyTorch checkpoint)
        device: 'cpu' or 'cuda'
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
        self.model_path = model_path

        # Load model if path provided
        if model_path is not None:
            self._load_model(model_path)
        else:
            self.model = None
            self.norm_stats = None
            print(
                "[NeuralDynamics] No model loaded. "
                "Call load_model() or use NeuralNetworkTrainer to train."
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
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.model.eval()

        # Normalize inputs
        if self.norm_stats is not None:
            state_norm = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_norm = (control - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_norm = state
            control_norm = control

        # Concatenate [state, control]
        if state.ndim == 1:
            inputs = np.concatenate([state_norm, control_norm])
            inputs_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)
        else:
            inputs = np.concatenate([state_norm, control_norm], axis=1)
            inputs_tensor = torch.FloatTensor(inputs).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs_tensor = self.model(inputs_tensor)
            outputs = outputs_tensor.cpu().numpy()

        # Denormalize outputs
        if self.norm_stats is not None:
            outputs = outputs * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]

        # Squeeze if single sample
        if state.ndim == 1:
            outputs = outputs.squeeze(0)

        return outputs

    def _load_model(self, model_path: str):
        """
        Load trained model

        Args:
            model_path: Path to PyTorch checkpoint
        """
        try:
            # Import here to avoid dependency if not using neural models
            from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel

            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract config
            config = checkpoint["config"]
            input_dim = config["state_dim"] + config["control_dim"]
            output_dim = config["state_dim"]
            hidden_dims = config["hidden_dims"]

            # Build model
            self.model = DynamicsMLPModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
            ).to(self.device)

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Load normalization stats
            self.norm_stats = checkpoint.get("norm_stats")

            print(f"[NeuralDynamics] Model loaded from {model_path}")
            print(f"  Input dim: {input_dim} (state={config['state_dim']}, control={config['control_dim']})")
            print(f"  Hidden dims: {hidden_dims}")
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Parameters: {num_params:,}")

        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch"
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def load_model(self, model_path: str):
        """Load model (public method)"""
        self.model_path = model_path
        self._load_model(model_path)

    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {"loaded": False}

        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "loaded": True,
            "input_dim": self.model.input_dim,
            "output_dim": self.model.output_dim,
            "hidden_dims": self.model.hidden_dims,
            "num_parameters": num_params,
            "device": str(self.device),
            "normalized": self.norm_stats is not None,
        }

    def __repr__(self) -> str:
        if self.model is not None:
            num_params = sum(p.numel() for p in self.model.parameters())
            return (
                f"NeuralDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"params={num_params:,}, "
                f"loaded=True)"
            )
        else:
            return (
                f"NeuralDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"loaded=False)"
            )
