"""
불확실성 인식 MPPI 비용 함수

모델 불확실성(GP/Ensemble std)에 비례하는 페널티를 부과하여
불확실한 영역에서 보수적 제어를 유도.
"""

import numpy as np
from typing import Optional, Callable
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class UncertaintyAwareCost(CostFunction):
    """
    불확실성 인식 비용 함수

    J_uncertainty = β × Σ_{t=0}^{N-1} ||σ(x_t, u_t)||²

    여기서 σ(x, u)는 모델의 예측 불확실성 (GP std 또는 앙상블 std).

    불확실성이 높은 영역에서 비용이 증가하여:
    1. 잘 학습된 영역으로 경로 유도
    2. Active exploration 방지 (안전 우선)
    3. Risk-Aware MPPI와 시너지

    사용 예시:
        # GP 모델의 불확실성 사용
        gp_model = GaussianProcessDynamics(...)
        unc_cost = UncertaintyAwareCost(
            uncertainty_fn=lambda states, controls: gp_model.predict_with_uncertainty(states, controls)[1],
            beta=10.0,
        )

        # Ensemble 모델 사용
        ensemble = EnsembleNeuralDynamics(...)
        unc_cost = UncertaintyAwareCost(
            uncertainty_fn=lambda s, c: ensemble.predict_with_uncertainty(s, c)[1],
            beta=5.0,
        )

    Args:
        uncertainty_fn: (states, controls) → std (batch, nx) 또는 (nx,)
        beta: 불확실성 페널티 가중치 (높을수록 보수적)
        reduce: 'sum' (차원 합), 'max' (최대 차원), 'mean' (차원 평균)
    """

    def __init__(
        self,
        uncertainty_fn: Callable,
        beta: float = 10.0,
        reduce: str = "sum",
    ):
        self.uncertainty_fn = uncertainty_fn
        self.beta = beta
        self.reduce = reduce

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        불확실성 비용 계산

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 (미사용)

        Returns:
            costs: (K,) 불확실성 페널티
        """
        K, N_plus_1, nx = trajectories.shape
        N = N_plus_1 - 1
        costs = np.zeros(K)

        for t in range(N):
            states_t = trajectories[:, t, :]  # (K, nx)
            controls_t = controls[:, t, :]    # (K, nu)

            # 불확실성 추정: (K, nx)
            std = self.uncertainty_fn(states_t, controls_t)

            # 차원 축소
            if self.reduce == "sum":
                uncertainty_penalty = np.sum(std ** 2, axis=-1)  # (K,)
            elif self.reduce == "max":
                uncertainty_penalty = np.max(std ** 2, axis=-1)
            elif self.reduce == "mean":
                uncertainty_penalty = np.mean(std ** 2, axis=-1)
            else:
                uncertainty_penalty = np.sum(std ** 2, axis=-1)

            costs += self.beta * uncertainty_penalty

        return costs
