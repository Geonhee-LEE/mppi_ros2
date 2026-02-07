"""
배치 동역학 래퍼

RobotModel을 K개 샘플로 벡터화하여 MPPI 샘플링 지원.
모든 모델 타입 (kinematic/dynamic/learned)에서 작동.
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Tuple


class BatchDynamicsWrapper:
    """
    배치 동역학 래퍼

    RobotModel을 K개 샘플로 벡터화하여 병렬 rollout 지원.
    모든 모델 타입 (kinematic/dynamic/learned)에서 작동.

    Args:
        model: RobotModel 인스턴스
        dt: 타임스텝 간격 (초)

    Attributes:
        nx: 상태 차원
        nu: 제어 차원
    """

    def __init__(self, model: RobotModel, dt: float):
        self.model = model
        self.dt = dt
        self.nx = model.state_dim
        self.nu = model.control_dim

    def rollout(
        self, initial_state: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """
        K개 제어 시퀀스를 병렬로 rollout

        Args:
            initial_state: (nx,) 초기 상태
            controls: (K, N, nu) K개 샘플, N 스텝 제어 시퀀스

        Returns:
            trajectories: (K, N+1, nx) K개 샘플 궤적
                - trajectories[:, 0, :] = initial_state (브로드캐스트)
                - trajectories[:, t+1, :] = step(trajectories[:, t, :], controls[:, t, :])
        """
        K, N, _ = controls.shape

        # 궤적 저장 배열 초기화
        trajectories = np.zeros((K, N + 1, self.nx))

        # 초기 상태 설정 (브로드캐스트)
        trajectories[:, 0, :] = initial_state

        # 시간 스텝별 전파
        for t in range(N):
            state_t = trajectories[:, t, :]  # (K, nx)
            control_t = controls[:, t, :]  # (K, nu)

            # 배치로 한 스텝 전파 (RK4)
            trajectories[:, t + 1, :] = self.model.step(state_t, control_t, self.dt)

        return trajectories

    def single_rollout(
        self, initial_state: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """
        단일 제어 시퀀스 rollout (디버깅용)

        Args:
            initial_state: (nx,) 초기 상태
            controls: (N, nu) 제어 시퀀스

        Returns:
            trajectory: (N+1, nx) 궤적
        """
        N = controls.shape[0]
        trajectory = np.zeros((N + 1, self.nx))
        trajectory[0, :] = initial_state

        for t in range(N):
            trajectory[t + 1, :] = self.model.step(
                trajectory[t, :], controls[t, :], self.dt
            )

        return trajectory

    def forward_dynamics_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """
        배치 forward dynamics (디버깅용)

        Args:
            states: (K, nx)
            controls: (K, nu)

        Returns:
            state_dots: (K, nx)
        """
        return self.model.forward_dynamics(states, controls)

    def __repr__(self) -> str:
        return (
            f"BatchDynamicsWrapper("
            f"model={self.model.__class__.__name__}, "
            f"dt={self.dt}, nx={self.nx}, nu={self.nu})"
        )
