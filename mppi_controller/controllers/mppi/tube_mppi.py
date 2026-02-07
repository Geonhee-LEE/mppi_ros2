"""
Tube-MPPI 컨트롤러

외란에 강건한 MPPI. 명목 상태와 실제 상태를 분리하여 관리.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import TubeMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.ancillary_controller import (
    AncillaryController,
    create_default_ancillary_controller,
)


class TubeMPPIController(MPPIController):
    """
    Tube-MPPI 컨트롤러

    외란에 강건한 MPPI. 명목 상태 주위에 "tube"를 형성하여
    실제 상태가 tube 내부에 머무르도록 보장.

    동작 원리:
        1. MPPI로 명목 제어 u_nominal 계산
        2. 명목 상태 x_nominal 전파 (외란 없음)
        3. Ancillary controller로 피드백 보정 u_fb 계산
        4. 최종 제어: u = u_nominal + u_fb

    장점:
        - 외란 강건성 (process noise, 모델링 오차)
        - 이론적 안정성 보장
        - 계산 효율성 (MPPI + 선형 피드백)

    참고 논문:
        Williams et al. (2018) - "Robust Sampling Based MPPI"

    Args:
        model: RobotModel 인스턴스
        params: TubeMPPIParams 파라미터
        ancillary_controller: AncillaryController (None이면 기본값 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: TubeMPPIParams,
        ancillary_controller: Optional[AncillaryController] = None,
    ):
        # 부모 클래스 초기화 (Vanilla MPPI)
        super().__init__(model, params)

        # Tube-MPPI 파라미터
        self.tube_params = params
        self.tube_enabled = params.tube_enabled

        # Ancillary controller 설정
        if ancillary_controller is None:
            self.ancillary_controller = create_default_ancillary_controller(
                model.model_type
            )
        else:
            self.ancillary_controller = ancillary_controller

        # 명목 상태 (외란 없는 상태)
        self.nominal_state = None

        # Tube 폭 추적 (디버깅용)
        self.tube_width_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Tube-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태 (실제, 외란 포함)
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
                - (Vanilla MPPI info) +
                - nominal_state: (nx,) 명목 상태
                - feedback_correction: (nu,) 피드백 보정
                - tube_width: float Tube 폭 (||x - x_nominal||)
                - tube_boundary: (nx,) Tube 경계 (시각화용)
        """
        if not self.tube_enabled:
            # Tube 비활성화 → Vanilla MPPI
            control, info = super().compute_control(state, reference_trajectory)
            # Tube 관련 정보 추가 (비활성화 상태)
            info.update({
                "nominal_state": state.copy(),
                "feedback_correction": np.zeros(self.model.control_dim),
                "tube_width": 0.0,
                "tube_margin": self.tube_params.tube_margin,
                "tube_enabled": False,
            })
            return control, info

        # 1. 명목 상태 초기화 (첫 호출)
        if self.nominal_state is None:
            self.nominal_state = state.copy()

        # 2. Vanilla MPPI로 명목 제어 계산
        # 명목 상태를 사용하여 MPPI 실행
        nominal_control, mppi_info = super().compute_control(
            self.nominal_state, reference_trajectory
        )

        # 3. Ancillary controller로 피드백 보정
        feedback_correction = self.ancillary_controller.compute_feedback(
            state, self.nominal_state
        )

        # 4. 최종 제어: u = u_nominal + u_fb
        control = nominal_control + feedback_correction

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            control = np.clip(control, self.u_min, self.u_max)

        # 5. 명목 상태 전파 (외란 없음)
        self.nominal_state = self.model.step(
            self.nominal_state, nominal_control, self.params.dt
        )
        self.nominal_state = self.model.normalize_state(self.nominal_state)

        # 6. Tube 폭 계산
        tube_width = np.linalg.norm(state - self.nominal_state)
        self.tube_width_history.append(tube_width)

        # 7. 정보 저장
        info = mppi_info.copy()
        info.update({
            "nominal_state": self.nominal_state.copy(),
            "feedback_correction": feedback_correction,
            "tube_width": tube_width,
            "tube_margin": self.tube_params.tube_margin,
            "tube_enabled": self.tube_enabled,
        })

        return control, info

    def reset(self):
        """명목 제어 시퀀스 및 명목 상태 초기화"""
        super().reset()
        self.nominal_state = None
        self.tube_width_history = []

    def get_tube_statistics(self) -> Dict:
        """
        Tube 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_tube_width: float 평균 Tube 폭
                - max_tube_width: float 최대 Tube 폭
                - tube_width_history: List[float] Tube 폭 히스토리
        """
        if len(self.tube_width_history) == 0:
            return {
                "mean_tube_width": 0.0,
                "max_tube_width": 0.0,
                "tube_width_history": [],
            }

        return {
            "mean_tube_width": np.mean(self.tube_width_history),
            "max_tube_width": np.max(self.tube_width_history),
            "tube_width_history": self.tube_width_history.copy(),
        }

    def set_tube_enabled(self, enabled: bool):
        """Tube-MPPI 활성화/비활성화"""
        self.tube_enabled = enabled
        if not enabled:
            # 비활성화 시 명목 상태 초기화
            self.nominal_state = None

    def __repr__(self) -> str:
        tube_status = "enabled" if self.tube_enabled else "disabled"
        return (
            f"TubeMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"tube={tube_status}, "
            f"params={self.params})"
        )
