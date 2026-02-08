"""
Superellipsoid 장애물 비용 함수

원형/구형 장애물 대신 superellipse(초타원) 형태의 장애물 지원.
벽, 차량, 가구 등 비원형 장애물을 더 정밀하게 모델링.

Superellipse 방정식:
    (|x-cx|/a)^n + (|y-cy|/b)^n = 1

    a, b: 반축 (semi-axis)
    n: 형상 파라미터
        n=2  → 일반 타원
        n→∞ → 직사각형
        n=1  → 다이아몬드
        n>2  → 모서리가 둥근 직사각형

Barrier function:
    h(x, y) = ((x-cx)/a)^n + ((y-cy)/b)^n - 1

    h > 0 → 장애물 외부 (안전)
    h = 0 → 장애물 경계
    h < 0 → 장애물 내부 (충돌)

회전 지원:
    [x', y'] = R(-θ) @ [x-cx, y-cy]
    R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]

Ref: Rimon & Koditschek (1992)
     "Exact Robot Navigation Using Artificial Potential Functions"
"""

import numpy as np
from typing import List, Dict
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class SuperellipsoidObstacle:
    """
    Superellipsoid 장애물 정의

    Args:
        cx, cy: 중심 좌표
        a, b: 반축 (semi-axis lengths)
        n: 형상 파라미터 (n=2: 타원, n>2: 둥근 직사각형)
        theta: 회전각 (rad, 기본 0)
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        a: float,
        b: float,
        n: float = 2.0,
        theta: float = 0.0,
    ):
        self.cx = cx
        self.cy = cy
        self.a = a
        self.b = b
        self.n = n
        self.theta = theta
        self._cos_theta = np.cos(-theta)
        self._sin_theta = np.sin(-theta)

    def barrier(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Barrier 값 계산 (벡터화)

        h = (|x'|/a)^n + (|y'|/b)^n - 1

        Args:
            x, y: 좌표 (임의 shape)

        Returns:
            h: 동일 shape, h>0 = 안전
        """
        # 회전 변환 (장애물 로컬 좌표)
        dx = x - self.cx
        dy = y - self.cy
        x_local = self._cos_theta * dx + self._sin_theta * dy
        y_local = -self._sin_theta * dx + self._cos_theta * dy

        # superellipse 값
        val = (np.abs(x_local) / self.a) ** self.n + \
              (np.abs(y_local) / self.b) ** self.n

        return val - 1.0

    def __repr__(self) -> str:
        return (
            f"SuperellipsoidObstacle(center=({self.cx:.1f}, {self.cy:.1f}), "
            f"axes=({self.a:.1f}, {self.b:.1f}), n={self.n:.1f}, "
            f"theta={np.degrees(self.theta):.0f}°)"
        )


class SuperellipsoidCost(CostFunction):
    """
    Superellipsoid 장애물 비용 함수

    Discrete-time CBF 조건으로 비용 계산:
        violation = max(0, -(h_{t+1} - (1-α)·h_t))

    Args:
        obstacles: SuperellipsoidObstacle 리스트
        cbf_alpha: CBF decay rate (0 < α ≤ 1)
        cbf_weight: CBF 위반 비용 가중치
        safety_margin: 추가 안전 마진 (장애물 확대)
    """

    def __init__(
        self,
        obstacles: List[SuperellipsoidObstacle] = None,
        cbf_alpha: float = 0.1,
        cbf_weight: float = 1000.0,
        safety_margin: float = 0.0,
    ):
        self.obstacles = obstacles or []
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
        Superellipsoid CBF 위반 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        if not self.obstacles:
            return costs

        x_pos = trajectories[:, :, 0]  # (K, N+1)
        y_pos = trajectories[:, :, 1]  # (K, N+1)

        for obs in self.obstacles:
            # 안전 마진 적용: 반축 확대
            obs_expanded = SuperellipsoidObstacle(
                obs.cx, obs.cy,
                obs.a + self.safety_margin,
                obs.b + self.safety_margin,
                obs.n, obs.theta,
            ) if self.safety_margin > 0 else obs

            h = obs_expanded.barrier(x_pos, y_pos)  # (K, N+1)

            # Discrete CBF: h_{t+1} - (1-α)·h_t ≥ 0
            cbf_condition = h[:, 1:] - (1.0 - self.cbf_alpha) * h[:, :-1]
            violation = np.maximum(0.0, -cbf_condition)

            costs += self.cbf_weight * np.sum(violation, axis=1)

        return costs

    def get_barrier_info(self, trajectory: np.ndarray) -> Dict:
        """
        Barrier 정보 반환

        Args:
            trajectory: (N+1, nx) 또는 (K, N+1, nx)

        Returns:
            dict: barrier_values, min_barrier, is_safe
        """
        if trajectory.ndim == 2:
            trajectory = trajectory[np.newaxis, :, :]

        if not self.obstacles:
            return {
                "barrier_values": np.array([]),
                "min_barrier": float("inf"),
                "is_safe": True,
            }

        x_pos = trajectory[:, :, 0]
        y_pos = trajectory[:, :, 1]

        all_h = []
        for obs in self.obstacles:
            obs_expanded = SuperellipsoidObstacle(
                obs.cx, obs.cy,
                obs.a + self.safety_margin,
                obs.b + self.safety_margin,
                obs.n, obs.theta,
            ) if self.safety_margin > 0 else obs

            h = obs_expanded.barrier(x_pos, y_pos)
            all_h.append(h)

        barrier_stack = np.array(all_h)
        min_barrier = float(np.min(barrier_stack))

        return {
            "barrier_values": barrier_stack,
            "min_barrier": min_barrier,
            "is_safe": min_barrier > 0,
        }

    def update_obstacles(self, obstacles: List[SuperellipsoidObstacle]):
        """장애물 업데이트"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"SuperellipsoidCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"alpha={self.cbf_alpha}, "
            f"weight={self.cbf_weight})"
        )
