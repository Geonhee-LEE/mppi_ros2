"""
PyTorch GPU 비용 함수 모듈

모든 비용을 하나의 fused 연산으로 GPU 커널 호출 최소화.
StateTracking + Terminal + ControlEffort + Obstacle을 통합 계산.
"""

import torch
import numpy as np
from typing import Optional, List


class TorchCompositeCost:
    """
    GPU 통합 비용 함수

    개별 CostFunction 호출 대신 하나의 fused 연산으로
    GPU 커널 호출 횟수를 최소화.

    지원 비용:
        - StateTrackingCost (Q)
        - TerminalCost (Qf)
        - ControlEffortCost (R)
        - ObstacleCost (선택)

    Args:
        Q: (nx,) 상태 추적 비용 가중치
        R: (nu,) 제어 노력 비용 가중치
        Qf: (nx,) 터미널 비용 가중치
        obstacles: [(x, y, radius), ...] 장애물 리스트
        obstacle_weight: 장애물 비용 가중치
        safety_margin: 안전 마진 (m)
        device: torch device
    """

    def __init__(
        self,
        Q,
        R,
        Qf,
        obstacles: Optional[List[tuple]] = None,
        obstacle_weight: float = 100.0,
        safety_margin: float = 0.2,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.Q = torch.tensor(np.asarray(Q), device=self.device, dtype=torch.float32)
        self.R = torch.tensor(np.asarray(R), device=self.device, dtype=torch.float32)
        self.Qf = torch.tensor(np.asarray(Qf), device=self.device, dtype=torch.float32)

        # 장애물 정보
        self.obstacle_weight = obstacle_weight
        self.safety_margin = safety_margin
        if obstacles and len(obstacles) > 0:
            self.obstacles = torch.tensor(
                obstacles, device=self.device, dtype=torch.float32
            )  # (M, 3) — [x, y, radius]
        else:
            self.obstacles = None

    def compute_cost(self, trajectories, controls, reference):
        """
        통합 비용 계산

        Args:
            trajectories: (K, N+1, nx) torch tensor — 샘플 궤적
            controls: (K, N, nu) torch tensor — 샘플 제어
            reference: (N+1, nx) torch tensor — 레퍼런스 궤적

        Returns:
            costs: (K,) torch tensor — 총 비용
        """
        # State tracking cost: (K, N, nx)
        errors = trajectories[:, :-1, :] - reference[:-1, :]
        state_cost = torch.sum(errors ** 2 * self.Q, dim=(1, 2))

        # Terminal cost: (K, nx)
        terminal_error = trajectories[:, -1, :] - reference[-1, :]
        terminal_cost = torch.sum(terminal_error ** 2 * self.Qf, dim=1)

        # Control effort cost: (K, N, nu)
        control_cost = torch.sum(controls ** 2 * self.R, dim=(1, 2))

        total = state_cost + terminal_cost + control_cost

        # Obstacle cost (선택)
        if self.obstacles is not None:
            total = total + self._obstacle_cost(trajectories)

        return total

    def _obstacle_cost(self, trajectories):
        """
        장애물 회피 비용 (벡터화)

        Args:
            trajectories: (K, N+1, nx)

        Returns:
            costs: (K,)
        """
        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        # 장애물 위치/반지름: (M, 3)
        obs_pos = self.obstacles[:, :2]  # (M, 2)
        obs_rad = self.obstacles[:, 2]  # (M,)

        # 브로드캐스트: (K, N+1, 1, 2) - (M, 2) → (K, N+1, M, 2)
        diff = positions.unsqueeze(2) - obs_pos  # (K, N+1, M, 2)
        distances = torch.norm(diff, dim=-1)  # (K, N+1, M)

        # 침투 깊이: (K, N+1, M)
        penetrations = (obs_rad + self.safety_margin) - distances

        # 침투한 경우만 비용 부과 (exponential)
        obstacle_costs = torch.where(
            penetrations > 0,
            torch.exp(penetrations * 5.0),
            torch.zeros_like(penetrations),
        )

        # (K, N+1, M) → (K,)
        return self.obstacle_weight * obstacle_costs.sum(dim=(1, 2))

    def update_obstacles(self, obstacles):
        """장애물 리스트 업데이트"""
        if obstacles and len(obstacles) > 0:
            self.obstacles = torch.tensor(
                obstacles, device=self.device, dtype=torch.float32
            )
        else:
            self.obstacles = None
