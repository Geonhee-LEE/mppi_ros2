"""
Backup Controllers for Gatekeeper Safety Shielding

안전이 보장되는 단순한 백업 정책.
Gatekeeper가 MPPI 제어를 검증할 때 사용:
  - 다음 상태에서 백업 정책 적용 시 무한 시간 안전한가?
  - 안전하면 MPPI 제어 적용, 불안전하면 백업 제어 적용.

백업 정책 종류:
  1. BrakeBackupController: 즉시 정지 (v=0, ω=0)
  2. TurnAndBrakeBackupController: 장애물 반대 방향 회전 후 정지

Ref: Gurriet et al. (2020)
     "Scalable Safety-Critical Control of Robotic Systems"
"""

import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod


class BackupController(ABC):
    """백업 컨트롤러 추상 클래스"""

    @abstractmethod
    def compute_backup_control(
        self, state: np.ndarray, obstacles: List[tuple]
    ) -> np.ndarray:
        """
        백업 제어 계산

        Args:
            state: (nx,) 현재 상태
            obstacles: [(x, y, r), ...] 장애물

        Returns:
            control: (nu,) 백업 제어
        """
        pass

    @abstractmethod
    def generate_backup_trajectory(
        self,
        state: np.ndarray,
        model,
        dt: float,
        horizon: int,
        obstacles: List[tuple],
    ) -> np.ndarray:
        """
        백업 궤적 생성

        Args:
            state: (nx,) 시작 상태
            model: RobotModel
            dt: 시간 간격
            horizon: 백업 궤적 길이
            obstacles: 장애물

        Returns:
            trajectory: (horizon+1, nx) 백업 궤적
        """
        pass


class BrakeBackupController(BackupController):
    """
    즉시 정지 백업 컨트롤러

    v = 0, ω = 0으로 즉시 정지.
    가장 단순하지만 급정지 시 안전하지 않을 수 있음.
    정적 환경에서 유효 (로봇이 멈추면 안전).
    """

    def compute_backup_control(
        self, state: np.ndarray, obstacles: List[tuple]
    ) -> np.ndarray:
        return np.array([0.0, 0.0])

    def generate_backup_trajectory(
        self,
        state: np.ndarray,
        model,
        dt: float,
        horizon: int,
        obstacles: List[tuple],
    ) -> np.ndarray:
        trajectory = np.zeros((horizon + 1, len(state)))
        trajectory[0] = state.copy()
        u_backup = np.array([0.0, 0.0])

        for t in range(horizon):
            trajectory[t + 1] = model.step(trajectory[t], u_backup, dt)

        return trajectory


class TurnAndBrakeBackupController(BackupController):
    """
    회전 후 정지 백업 컨트롤러

    1단계 (turn_steps): 가장 가까운 장애물 반대 방향으로 회전
    2단계: 정지 (v=0, ω=0)

    동적 환경에서도 유효: 로봇을 장애물에서 멀어지는 방향으로 돌림.

    Args:
        turn_speed: 회전 속도 (rad/s)
        turn_steps: 회전 지속 시간스텝
    """

    def __init__(self, turn_speed: float = 0.5, turn_steps: int = 5):
        self.turn_speed = turn_speed
        self.turn_steps = turn_steps

    def compute_backup_control(
        self, state: np.ndarray, obstacles: List[tuple]
    ) -> np.ndarray:
        if not obstacles:
            return np.array([0.0, 0.0])

        x, y, theta = state[0], state[1], state[2]

        # 가장 가까운 장애물 찾기
        min_dist = float("inf")
        closest_obs = None
        for obs in obstacles:
            dx = obs[0] - x
            dy = obs[1] - y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_obs = (dx, dy)

        if closest_obs is None:
            return np.array([0.0, 0.0])

        # 장애물 방향 각도
        obs_angle = np.arctan2(closest_obs[1], closest_obs[0])
        # 로봇 현재 방향과의 차이
        angle_diff = obs_angle - theta
        # [-π, π] 정규화
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # 장애물 반대 방향으로 회전
        # angle_diff ≈ 0 (정면)이면 기본 좌회전
        if abs(angle_diff) < 1e-6:
            omega = self.turn_speed
        else:
            omega = -np.sign(angle_diff) * self.turn_speed
        return np.array([0.0, omega])

    def generate_backup_trajectory(
        self,
        state: np.ndarray,
        model,
        dt: float,
        horizon: int,
        obstacles: List[tuple],
    ) -> np.ndarray:
        trajectory = np.zeros((horizon + 1, len(state)))
        trajectory[0] = state.copy()

        u_turn = self.compute_backup_control(state, obstacles)
        u_stop = np.array([0.0, 0.0])

        for t in range(horizon):
            u = u_turn if t < self.turn_steps else u_stop
            trajectory[t + 1] = model.step(trajectory[t], u, dt)

        return trajectory
