"""
CBF-MPPI Controller

Control Barrier Function을 MPPI에 통합한 안전 제어기.

Approach A: CBF 비용 함수 (sampling-aligned)
Approach B: CBF QP 안전 필터 (optional, formal guarantee)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import CBFMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction,
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class CBFMPPIController(MPPIController):
    """
    CBF-MPPI Controller

    Control Barrier Function을 통합한 안전한 MPPI 제어기.

    Layer 1 (Approach A): ControlBarrierCost를 CompositeMPPICost에 추가.
        - 샘플링 단계에서 안전하지 않은 궤적에 높은 비용 부과.
        - 기존 MPPI 프레임워크와 자연스럽게 통합.

    Layer 2 (Approach B, optional): CBFSafetyFilter로 최종 제어 보정.
        - QP로 최소한의 수정으로 CBF 제약 보장.
        - 형식적 안전 보장 제공.

    _compute_weights()를 오버라이드하지 않음 - CBF는 비용 함수를 통해 통합.

    Args:
        model: RobotModel 인스턴스
        params: CBFMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 + CBF 비용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: CBFMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        # CBF 비용 함수 생성
        self.cbf_cost = ControlBarrierCost(
            obstacles=params.cbf_obstacles,
            cbf_alpha=params.cbf_alpha,
            cbf_weight=params.cbf_weight,
            safety_margin=params.cbf_safety_margin,
        )

        # 비용 함수 구성: 사용자 제공 or 기본 + CBF
        if cost_function is None:
            composite_cost = CompositeMPPICost([
                StateTrackingCost(params.Q),
                TerminalCost(params.Qf),
                ControlEffortCost(params.R),
                self.cbf_cost,
            ])
        else:
            # 사용자 비용 함수에 CBF 추가
            if isinstance(cost_function, CompositeMPPICost):
                cost_fns = cost_function.cost_functions + [self.cbf_cost]
                composite_cost = CompositeMPPICost(cost_fns)
            else:
                composite_cost = CompositeMPPICost([cost_function, self.cbf_cost])

        # 부모 클래스 초기화 (CBF 포함 비용 함수 전달)
        super().__init__(model, params, composite_cost, noise_sampler)

        # CBF 파라미터 저장
        self.cbf_params = params

        # 안전 필터 (optional, Approach B)
        self.safety_filter = None
        if params.cbf_use_safety_filter:
            self.safety_filter = CBFSafetyFilter(
                obstacles=params.cbf_obstacles,
                cbf_alpha=params.cbf_alpha,
                safety_margin=params.cbf_safety_margin,
            )

        # CBF 통계
        self.cbf_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        CBF-MPPI 제어 계산

        1. MPPI 최적 제어 계산 (CBF 비용 포함)
        2. (선택) QP 안전 필터로 사후 보정

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - MPPI info + CBF info
        """
        # Layer 1: MPPI 제어 계산 (CBF 비용 포함)
        control, info = super().compute_control(state, reference_trajectory)

        # Barrier 정보 수집
        best_traj = info["best_trajectory"]  # (N+1, nx)
        barrier_info = self.cbf_cost.get_barrier_info(best_traj)

        # Layer 2: 안전 필터 (optional)
        filter_info = {"filtered": False, "correction_norm": 0.0}
        if self.safety_filter is not None:
            control, filter_info = self.safety_filter.filter_control(
                state, control, self.u_min, self.u_max
            )

        # CBF 통계 저장
        cbf_stats = {
            "min_barrier": barrier_info["min_barrier"],
            "is_safe": barrier_info["is_safe"],
            "filtered": filter_info["filtered"],
            "correction_norm": filter_info.get("correction_norm", 0.0),
        }
        self.cbf_stats_history.append(cbf_stats)

        # info에 CBF 정보 추가
        info["barrier_values"] = barrier_info["barrier_values"]
        info["min_barrier"] = barrier_info["min_barrier"]
        info["is_safe"] = barrier_info["is_safe"]
        info["cbf_filtered"] = filter_info["filtered"]
        info["cbf_correction_norm"] = filter_info.get("correction_norm", 0.0)

        return control, info

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.cbf_cost.update_obstacles(obstacles)
        if self.safety_filter is not None:
            self.safety_filter.update_obstacles(obstacles)

    def get_cbf_statistics(self) -> Dict:
        """CBF 통계 반환"""
        if not self.cbf_stats_history:
            return {
                "mean_min_barrier": 0.0,
                "min_min_barrier": 0.0,
                "safety_rate": 0.0,
                "filter_rate": 0.0,
                "mean_correction_norm": 0.0,
            }

        min_barriers = [s["min_barrier"] for s in self.cbf_stats_history]
        safe_count = sum(1 for s in self.cbf_stats_history if s["is_safe"])
        filter_count = sum(1 for s in self.cbf_stats_history if s["filtered"])
        correction_norms = [s["correction_norm"] for s in self.cbf_stats_history]

        return {
            "mean_min_barrier": float(np.mean(min_barriers)),
            "min_min_barrier": float(np.min(min_barriers)),
            "safety_rate": safe_count / len(self.cbf_stats_history),
            "filter_rate": filter_count / len(self.cbf_stats_history),
            "mean_correction_norm": float(np.mean(correction_norms)),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.cbf_stats_history = []
        if self.safety_filter is not None:
            self.safety_filter.reset()

    def __repr__(self) -> str:
        return (
            f"CBFMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"obstacles={len(self.cbf_params.cbf_obstacles)}, "
            f"alpha={self.cbf_params.cbf_alpha}, "
            f"safety_filter={self.safety_filter is not None})"
        )
