"""
MAML (Model-Agnostic Meta-Learning) 기반 동역학 모델

FOMAML로 사전 학습된 메타 파라미터에서 시작하여,
실행 중 최근 데이터(few-shot)로 빠르게 적응하는 동역학 모델.

Usage:
    maml = MAMLDynamics(state_dim=3, control_dim=2,
                        model_path="maml_meta_model.pth",
                        inner_lr=0.01, inner_steps=5)
    maml.save_meta_weights()

    # 실행 중 적응
    loss = maml.adapt(states, controls, next_states, dt)

    # MPPI rollout에 사용
    state_dot = maml.forward_dynamics(state, control)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from mppi_controller.models.learned.neural_dynamics import NeuralDynamics


class MAMLDynamics(NeuralDynamics):
    """
    MAML 기반 동역학 모델 — 실시간 few-shot 적응.

    NeuralDynamics를 상속하며, 메타 파라미터 저장/복원 및
    inner-loop SGD 적응 기능을 추가.

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        model_path: 메타 학습된 모델 경로
        device: 'cpu' or 'cuda'
        inner_lr: 적응 학습률
        inner_steps: 적응 gradient step 수
        use_adam: True → 온라인 적응에 Adam 사용, False → SGD
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        use_adam: bool = False,
    ):
        super().__init__(state_dim, control_dim, model_path, device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.use_adam = use_adam
        self._meta_weights = None
        self._online_optimizer = None

    def save_meta_weights(self):
        """현재 모델 파라미터를 메타 파라미터로 저장."""
        if self.model is not None:
            self._meta_weights = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }

    def restore_meta_weights(self):
        """메타 파라미터로 복원 (적응 전 상태로 되돌림)."""
        if self._meta_weights is not None and self.model is not None:
            self.model.load_state_dict(self._meta_weights)

    def adapt(self, states, controls, next_states, dt, restore=True,
              sample_weights=None, temporal_decay=None):
        """
        Few-shot 적응: 최근 데이터로 inner-loop gradient descent.

        Args:
            states: (M, nx) 최근 상태
            controls: (M, nu) 최근 제어
            next_states: (M, nx) 다음 상태
            dt: float 시간 간격
            restore: True → 메타 파라미터 복원 후 적응 (표준 MAML)
                     False → 현재 파라미터에서 계속 fine-tune (온라인 연속 학습)
            sample_weights: (M,) 샘플별 가중치 (None이면 균등)
            temporal_decay: float, 시간 감쇠 비율 (예: 0.99 → 최근 데이터 강조)
                           None이면 사용 안함. sample_weights와 중첩 가능.

        Returns:
            float: 최종 loss 값
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if restore:
            self.restore_meta_weights()
        self.model.train()

        # target: state_dot = (next_state - state) / dt
        targets = (next_states - states) / dt

        # Angle wrapping for theta (index 2)
        if states.shape[1] >= 3:
            theta_diff = next_states[:, 2] - states[:, 2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            targets[:, 2] = theta_diff / dt

        inputs_t = self._prepare_inputs(states, controls)
        targets_t = self._prepare_targets(targets)

        # 가중치 계산
        M = states.shape[0]
        weights = np.ones(M, dtype=np.float32)

        if temporal_decay is not None:
            # Exponential weighting: 최근 데이터에 높은 가중치
            decay_weights = np.array([temporal_decay ** (M - 1 - i) for i in range(M)], dtype=np.float32)
            weights *= decay_weights

        if sample_weights is not None:
            weights *= sample_weights.astype(np.float32)

        # 정규화
        weights /= weights.sum()
        weights_t = torch.FloatTensor(weights).to(self.device)

        if restore or self._online_optimizer is None:
            if self.use_adam:
                self._online_optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.inner_lr
                )
            else:
                self._online_optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.inner_lr
                )

        use_weighted = temporal_decay is not None or sample_weights is not None

        loss_val = 0.0
        for _ in range(self.inner_steps):
            self._online_optimizer.zero_grad()
            pred = self.model(inputs_t)
            if use_weighted:
                # Weighted MSE loss
                per_sample_loss = ((pred - targets_t) ** 2).mean(dim=1)
                loss = (per_sample_loss * weights_t).sum()
            else:
                loss = F.mse_loss(pred, targets_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._online_optimizer.step()
            loss_val = loss.item()

        self.model.eval()
        return loss_val

    def _prepare_inputs(self, states, controls):
        """numpy → normalized tensor [state_norm, control_norm]."""
        if self.norm_stats is not None:
            state_norm = (states - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_norm = (controls - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_norm = states
            control_norm = controls

        inputs = np.concatenate([state_norm, control_norm], axis=1)
        return torch.FloatTensor(inputs).to(self.device)

    def _prepare_targets(self, state_dots):
        """numpy → normalized tensor."""
        if self.norm_stats is not None:
            targets_norm = (state_dots - self.norm_stats["state_dot_mean"]) / self.norm_stats["state_dot_std"]
        else:
            targets_norm = state_dots

        return torch.FloatTensor(targets_norm).to(self.device)

    def __repr__(self) -> str:
        if self.model is not None:
            num_params = sum(p.numel() for p in self.model.parameters())
            return (
                f"MAMLDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"params={num_params:,}, "
                f"inner_lr={self.inner_lr}, "
                f"inner_steps={self.inner_steps}, "
                f"meta_saved={self._meta_weights is not None})"
            )
        else:
            return (
                f"MAMLDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"loaded=False)"
            )
