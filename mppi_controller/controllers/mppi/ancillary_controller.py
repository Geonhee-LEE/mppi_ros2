"""
Ancillary Controller (보조 컨트롤러)

Tube-MPPI를 위한 body frame 피드백 컨트롤러.
"""

import numpy as np
from typing import Optional


class AncillaryController:
    """
    Ancillary Controller (보조 컨트롤러)

    Tube-MPPI에서 명목 상태 주위의 오차를 보정하는 피드백 컨트롤러.

    동작 원리:
        1. World frame 오차: e_world = x - x_nominal
        2. Body frame 변환: e_body = R^T @ e_world
        3. 피드백 제어: u_fb = -K_fb @ e_body

    장점:
        - World frame 오차를 body frame으로 변환하여 방향성 고려
        - 선형 피드백으로 계산 효율적
        - 튜닝 가능한 게인

    Args:
        K_fb: 피드백 게인 행렬 (nu, nx)
            - Differential Drive Kinematic: (2, 3) - [v, ω] ← [x, y, θ]
            - Differential Drive Dynamic: (2, 5) - [a, α] ← [x, y, θ, v, ω]
        max_correction: 최대 피드백 보정값 (제어 입력 제한)
    """

    def __init__(
        self,
        K_fb: np.ndarray,
        max_correction: Optional[np.ndarray] = None,
    ):
        self.K_fb = K_fb
        self.max_correction = max_correction

        # 게인 검증
        assert K_fb.ndim == 2, "K_fb must be 2D array (nu, nx)"
        self.nu, self.nx = K_fb.shape

    def compute_feedback(
        self,
        state: np.ndarray,
        nominal_state: np.ndarray,
    ) -> np.ndarray:
        """
        피드백 제어 계산

        Args:
            state: (nx,) 현재 상태
            nominal_state: (nx,) 명목 상태

        Returns:
            feedback_control: (nu,) 피드백 제어
        """
        # 1. World frame 오차
        error_world = state - nominal_state

        # 2. Body frame으로 변환 (Differential Drive의 경우)
        # θ는 heading 각도 (state[2] 또는 nominal_state[2])
        if self.nx >= 3:
            theta = nominal_state[2]  # 명목 상태의 heading 사용
            error_body = self._world_to_body(error_world, theta)
        else:
            error_body = error_world

        # 3. 피드백 제어: u_fb = -K_fb @ e_body
        feedback_control = -self.K_fb @ error_body

        # 4. 최대 보정값 제한
        if self.max_correction is not None:
            feedback_control = np.clip(
                feedback_control, -self.max_correction, self.max_correction
            )

        return feedback_control

    def _world_to_body(self, error_world: np.ndarray, theta: float) -> np.ndarray:
        """
        World frame 오차를 Body frame으로 변환

        Args:
            error_world: (nx,) world frame 오차
            theta: heading 각도 (rad)

        Returns:
            error_body: (nx,) body frame 오차
        """
        # Rotation matrix: R(θ) = [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]]
        # Body frame: R^T @ [e_x, e_y]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        error_body = error_world.copy()

        # x, y 오차만 회전 (θ 오차는 그대로)
        e_x = error_world[0]
        e_y = error_world[1]

        error_body[0] = cos_theta * e_x + sin_theta * e_y  # longitudinal
        error_body[1] = -sin_theta * e_x + cos_theta * e_y  # lateral

        return error_body

    def set_gain(self, K_fb: np.ndarray):
        """피드백 게인 업데이트"""
        assert K_fb.shape == (self.nu, self.nx), f"K_fb shape mismatch: {K_fb.shape}"
        self.K_fb = K_fb

    def __repr__(self) -> str:
        return (
            f"AncillaryController("
            f"K_fb.shape={self.K_fb.shape}, "
            f"max_correction={self.max_correction})"
        )


def create_default_ancillary_controller(
    model_type: str,
    gain_scale: float = 1.0,
) -> AncillaryController:
    """
    모델 타입에 따른 기본 Ancillary Controller 생성

    Args:
        model_type: "kinematic" 또는 "dynamic"
        gain_scale: 게인 스케일 (튜닝용)

    Returns:
        AncillaryController 인스턴스
    """
    if model_type == "kinematic":
        # Differential Drive Kinematic: (2, 3)
        # [v, ω] ← [e_x, e_y, e_θ]
        K_fb = gain_scale * np.array(
            [
                [1.0, 0.0, 0.0],  # v ← e_x (longitudinal)
                [0.0, 2.0, 1.0],  # ω ← e_y, e_θ (lateral + heading)
            ]
        )
        max_correction = np.array([0.5, 0.5])  # [v, ω] 제한

    elif model_type == "dynamic":
        # Differential Drive Dynamic: (2, 5)
        # [a, α] ← [e_x, e_y, e_θ, e_v, e_ω]
        K_fb = gain_scale * np.array(
            [
                [0.5, 0.0, 0.0, 1.0, 0.0],  # a ← e_x, e_v
                [0.0, 1.0, 0.5, 0.0, 1.0],  # α ← e_y, e_θ, e_ω
            ]
        )
        max_correction = np.array([1.0, 1.0])  # [a, α] 제한

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return AncillaryController(K_fb, max_correction)
