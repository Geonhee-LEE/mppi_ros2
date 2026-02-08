"""
Gatekeeper Safety Shielding

MPPI 제어를 적용하기 전에 백업 궤적의 안전성을 검증.
백업 궤적이 안전하면 MPPI 제어를 적용, 불안전하면 백업 제어로 대체.

핵심 아이디어:
  1. MPPI가 제어 u_mppi를 출력
  2. u_mppi 적용 후 예측 상태 x_next 계산
  3. x_next에서 백업 정책으로 생성한 궤적이 안전한가?
  4. 안전 → u_mppi 적용
  5. 불안전 → 현재 상태에서 백업 정책 적용

이 과정으로 무한 시간 안전성(infinite-time safety) 보장:
  - 시스템은 항상 안전한 백업 궤적으로 돌아갈 수 있는 상태를 유지
  - Gatekeeper가 열리는(MPPI 허용) 조건이 곧 forward invariance

Ref: Gurriet et al. (2020)
     "Scalable Safety-Critical Control of Robotic Systems"

Usage:
    gatekeeper = Gatekeeper(
        backup_controller=BrakeBackupController(),
        model=model, obstacles=obstacles,
        safety_margin=0.15, backup_horizon=30, dt=0.05,
    )
    u_safe, info = gatekeeper.filter(state, u_mppi)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from mppi_controller.controllers.mppi.backup_controller import (
    BackupController,
    BrakeBackupController,
)


class Gatekeeper:
    """
    Gatekeeper Safety Shield

    Args:
        backup_controller: 백업 컨트롤러
        model: RobotModel
        obstacles: [(x, y, radius), ...]
        safety_margin: 추가 안전 마진 (m)
        backup_horizon: 백업 궤적 길이 (timesteps)
        dt: 시간 간격
    """

    def __init__(
        self,
        backup_controller: Optional[BackupController] = None,
        model=None,
        obstacles: List[tuple] = None,
        safety_margin: float = 0.15,
        backup_horizon: int = 30,
        dt: float = 0.05,
    ):
        self.backup_controller = backup_controller or BrakeBackupController()
        self.model = model
        self.obstacles = obstacles or []
        self.safety_margin = safety_margin
        self.backup_horizon = backup_horizon
        self.dt = dt
        self.stats = []

    def filter(
        self,
        state: np.ndarray,
        u_mppi: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Gatekeeper 안전 필터

        Args:
            state: (nx,) 현재 상태
            u_mppi: (nu,) MPPI 출력 제어

        Returns:
            u_safe: (nu,) 안전한 제어
            info: dict - gatekeeper 정보
        """
        if self.model is None:
            return u_mppi.copy(), {"gate_open": True, "reason": "no_model"}

        if not self.obstacles:
            info = {
                "gate_open": True,
                "reason": "no_obstacles",
                "backup_min_barrier": float("inf"),
            }
            self.stats.append(info)
            return u_mppi.copy(), info

        # 1. u_mppi 적용 후 예측 상태
        x_next = self.model.step(state, u_mppi, self.dt)

        # 2. x_next에서 백업 궤적 생성
        backup_traj = self.backup_controller.generate_backup_trajectory(
            x_next, self.model, self.dt, self.backup_horizon, self.obstacles
        )

        # 3. 백업 궤적의 안전성 검증
        is_safe, min_barrier = self._check_trajectory_safety(backup_traj)

        if is_safe:
            # Gate open: MPPI 제어 허용
            info = {
                "gate_open": True,
                "reason": "backup_safe",
                "backup_min_barrier": min_barrier,
                "backup_trajectory": backup_traj,
            }
            self.stats.append(info)
            return u_mppi.copy(), info
        else:
            # Gate closed: 백업 제어 적용
            u_backup = self.backup_controller.compute_backup_control(
                state, self.obstacles
            )
            info = {
                "gate_open": False,
                "reason": "backup_unsafe",
                "backup_min_barrier": min_barrier,
                "u_mppi": u_mppi.copy(),
                "u_backup": u_backup.copy(),
                "backup_trajectory": backup_traj,
            }
            self.stats.append(info)
            return u_backup, info

    def _check_trajectory_safety(
        self, trajectory: np.ndarray
    ) -> Tuple[bool, float]:
        """
        궤적의 모든 상태가 안전한지 검증

        h(x) = dist² - r_eff² > 0 for all states and all obstacles

        Args:
            trajectory: (T+1, nx) 궤적

        Returns:
            is_safe: bool
            min_barrier: float - 최소 barrier 값
        """
        positions = trajectory[:, :2]  # (T+1, 2)
        min_barrier = float("inf")

        for obs in self.obstacles:
            obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
            effective_r = obs_r + self.safety_margin

            dx = positions[:, 0] - obs_x
            dy = positions[:, 1] - obs_y
            dist_sq = dx**2 + dy**2
            h = dist_sq - effective_r**2

            min_h = float(np.min(h))
            min_barrier = min(min_barrier, min_h)

        return min_barrier > 0, min_barrier

    def update_obstacles(self, obstacles: List[tuple]):
        """장애물 업데이트"""
        self.obstacles = obstacles

    def get_statistics(self) -> Dict:
        """Gatekeeper 통계"""
        if not self.stats:
            return {
                "total_steps": 0,
                "gate_open_rate": 1.0,
                "gate_closed_count": 0,
            }

        total = len(self.stats)
        closed = sum(1 for s in self.stats if not s["gate_open"])

        return {
            "total_steps": total,
            "gate_open_rate": (total - closed) / total,
            "gate_closed_count": closed,
            "gate_closed_rate": closed / total,
        }

    def reset(self):
        """통계 초기화"""
        self.stats = []

    def __repr__(self) -> str:
        return (
            f"Gatekeeper("
            f"backup={self.backup_controller.__class__.__name__}, "
            f"obstacles={len(self.obstacles)}, "
            f"horizon={self.backup_horizon})"
        )
