"""
Control Barrier Function (CBF) 비용 함수

Discrete-time CBF 조건을 비용 페널티로 변환하여
MPPI 샘플링을 안전 영역으로 유도.
"""

import numpy as np
from typing import List, Dict
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class ControlBarrierCost(CostFunction):
    """
    Discrete-time CBF 비용 함수 (Approach A)

    Barrier function:
        h(x) = ||p - p_obs||^2 - (r + margin)^2

    CBF 조건 (discrete-time):
        h(x_{t+1}) - (1 - alpha) * h(x_t) >= 0

    위반 비용:
        cost = weight * sum_t max(0, -(h(x_{t+1}) - (1-alpha)*h(x_t)))

    h > 0 이면 안전, h <= 0 이면 장애물 내부.
    alpha가 클수록 보수적 (안전 마진 확대).

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        cbf_alpha: Class-K function 파라미터 (0 < alpha <= 1)
        cbf_weight: CBF 위반 비용 가중치
        safety_margin: 추가 안전 마진 (m)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        cbf_alpha: float = 0.1,
        cbf_weight: float = 1000.0,
        safety_margin: float = 0.1,
    ):
        self.obstacles = obstacles
        self.cbf_alpha = cbf_alpha
        self.cbf_weight = cbf_weight
        self.safety_margin = safety_margin

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        CBF 위반 비용 계산 (fully vectorized)

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            costs: (K,) 각 샘플의 CBF 비용
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin

            # 거리 제곱 (K, N+1)
            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            dist_sq = dx**2 + dy**2

            # Barrier value: h(x) = ||p - p_obs||^2 - r_eff^2
            h = dist_sq - effective_r**2  # (K, N+1)

            # Discrete CBF 조건: h(x_{t+1}) - (1-alpha)*h(x_t) >= 0
            # 위반 = max(0, -[h(x_{t+1}) - (1-alpha)*h(x_t)])
            cbf_condition = h[:, 1:] - (1.0 - self.cbf_alpha) * h[:, :-1]  # (K, N)
            violation = np.maximum(0.0, -cbf_condition)

            costs += self.cbf_weight * np.sum(violation, axis=1)

        return costs

    def get_barrier_info(self, trajectories: np.ndarray) -> Dict:
        """
        Barrier 정보 반환 (디버깅/시각화용)

        Args:
            trajectories: (K, N+1, nx) 또는 (N+1, nx) 궤적

        Returns:
            dict:
                - barrier_values: 각 장애물별 barrier 값
                - min_barrier: 전체 최소 barrier 값
                - is_safe: 모든 시간스텝에서 안전 여부
        """
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, :, :]

        positions = trajectories[:, :, :2]

        if len(self.obstacles) == 0:
            return {
                "barrier_values": np.array([]),
                "min_barrier": float('inf'),
                "is_safe": True,
            }

        all_barrier_values = []

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin
            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            dist_sq = dx**2 + dy**2
            h = dist_sq - effective_r**2
            all_barrier_values.append(h)

        # 모든 장애물에 대한 barrier 스택
        barrier_stack = np.array(all_barrier_values)  # (num_obs, K, N+1)
        min_barrier = np.min(barrier_stack)

        return {
            "barrier_values": barrier_stack,
            "min_barrier": float(min_barrier),
            "is_safe": bool(min_barrier > 0),
        }

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"ControlBarrierCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"alpha={self.cbf_alpha}, "
            f"weight={self.cbf_weight})"
        )
