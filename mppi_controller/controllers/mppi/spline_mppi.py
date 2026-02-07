"""
Spline MPPI Controller

B-spline 보간으로 메모리 효율적인 MPPI.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.interpolate import BSpline
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import SplineMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


class SplineMPPIController(MPPIController):
    """
    Spline MPPI Controller

    B-spline 보간을 사용하여 메모리 효율적인 제어를 생성하는 MPPI 변형.

    동작 원리:
        1. P개 제어 knot만 샘플링 (P << N)
           - 예: N=30, P=8 → 메모리 4배 감소
        2. B-spline 보간으로 N개 제어값 생성
           - Smooth 제어 자동 보장
        3. MPPI 가중치 계산
        4. Knot 업데이트 (N 대신 P)

    B-spline 특성:
        - Smooth: C^(k-1) 연속성 (k=degree)
        - Local control: Knot 변경 시 국소적 영향
        - Convex hull: 제어값이 knot 범위 내

    장점:
        - ✅ 메모리 효율: O(K*P) vs O(K*N)
        - ✅ 자동 제어 평활 (B-spline)
        - ✅ 샘플 복잡도 감소
        - ✅ 더 적은 노이즈 샘플링

    단점:
        - ⚠️ B-spline 평가 오버헤드
        - ⚠️ 급격한 제어 변화 어려움
        - ⚠️ Knot 수(P) 튜닝 필요

    적용 사례:
        - 메모리 제약 시스템
        - Long horizon MPC (N 큰 경우)
        - Smooth 제어 필수

    참고 논문:
        - Bhardwaj et al. (2024) - "Spline-MPPI"

    Args:
        model: RobotModel 인스턴스
        params: SplineMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: SplineMPPIParams):
        # 부모 클래스 초기화
        super().__init__(model, params)

        # Spline 파라미터
        self.spline_params = params
        self.P = params.spline_num_knots  # Knot 개수
        self.degree = params.spline_degree  # B-spline 차수

        # Knot 시퀀스 (P, nu)
        self.U_knots = np.zeros((self.P, self.model.control_dim))

        # Knot 위치 (시간 정규화)
        self.knot_positions = np.linspace(0, self.params.N - 1, self.P)

        # 통계 (디버깅용)
        self.spline_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Spline MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K

        # 1. **P개 knot 노이즈 샘플링** (핵심!)
        # Knot space에서 샘플링 (N 대신 P)
        knot_noise = np.random.normal(0, self.params.sigma, (K, self.P, self.model.control_dim))
        sampled_knots = self.U_knots + knot_noise  # (K, P, nu)

        # 제어 제약 클리핑 (knot level)
        if self.u_min is not None and self.u_max is not None:
            sampled_knots = np.clip(sampled_knots, self.u_min, self.u_max)

        # 2. **B-spline 보간으로 N개 제어값 생성**
        sampled_controls = np.zeros((K, self.params.N, self.model.control_dim))

        for k in range(K):
            for u_dim in range(self.model.control_dim):
                # B-spline 보간 (각 제어 차원별)
                sampled_controls[k, :, u_dim] = self._bspline_interpolate(
                    sampled_knots[k, :, u_dim], self.params.N
                )

        # 제어 제약 클리핑 (최종)
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 롤아웃
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 비용 계산
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. MPPI 가중치
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. ESS 계산
        ess = 1.0 / np.sum(weights**2)

        # 7. **Knot 업데이트** (가중 평균, P개만!)
        weighted_knot_noise = np.sum(weights[:, None, None] * knot_noise, axis=0)
        self.U_knots = self.U_knots + weighted_knot_noise

        # 8. U 복원 (최종 제어 시퀀스, 호환성)
        for u_dim in range(self.model.control_dim):
            self.U[:, u_dim] = self._bspline_interpolate(
                self.U_knots[:, u_dim], self.params.N
            )

        # 9. Receding horizon (Knot)
        # Shift knot positions
        self.U_knots = np.roll(self.U_knots, -1, axis=0)
        self.U_knots[-1, :] = 0.0

        # 10. Receding horizon (U)
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 11. 최적 제어
        optimal_control = self.U[0, :]

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            optimal_control = np.clip(optimal_control, self.u_min, self.u_max)

        # 12. Spline 통계 저장
        knot_variance = np.var(sampled_knots, axis=0).mean()
        spline_stats = {
            "num_knots": self.P,
            "degree": self.degree,
            "knot_variance": knot_variance,
        }
        self.spline_stats_history.append(spline_stats)

        # 13. 정보 저장
        info = {
            "sample_trajectories": sample_trajectories,
            "sample_controls": sampled_controls,
            "sample_knots": sampled_knots,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[np.argmax(weights)],
            "best_cost": np.min(costs),
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "spline_stats": spline_stats,
        }

        return optimal_control, info

    def _bspline_interpolate(
        self, knot_values: np.ndarray, num_points: int
    ) -> np.ndarray:
        """
        B-spline 보간

        Args:
            knot_values: (P,) Knot 제어값
            num_points: 생성할 점 개수 (N)

        Returns:
            interpolated: (num_points,) 보간된 제어값
        """
        P = len(knot_values)
        k = self.degree

        # Knot vector 생성 (clamped)
        # knot_vector 길이 = P + k + 1
        # 시작과 끝에 k+1개 반복 (clamped B-spline)
        internal_knots = P - k - 1
        if internal_knots < 0:
            # P가 너무 작으면 uniform knot vector 사용
            knot_vector = np.linspace(0, 1, P + k + 1)
        else:
            knot_vector = np.concatenate(
                [
                    np.zeros(k + 1),  # 시작 clamp (k+1개)
                    np.linspace(0, 1, internal_knots + 2)[1:-1],  # 내부 knot
                    np.ones(k + 1),  # 끝 clamp (k+1개)
                ]
            )

        # B-spline 생성
        spl = BSpline(knot_vector, knot_values, k)

        # 균등 간격으로 평가
        t_eval = np.linspace(0, 1, num_points)
        interpolated = spl(t_eval)

        return interpolated

    def get_spline_statistics(self) -> Dict:
        """
        Spline 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_knot_variance: float 평균 knot 분산
                - spline_stats_history: List[dict] 통계 히스토리
        """
        if len(self.spline_stats_history) == 0:
            return {
                "mean_knot_variance": 0.0,
                "spline_stats_history": [],
            }

        knot_variances = [s["knot_variance"] for s in self.spline_stats_history]

        return {
            "mean_knot_variance": np.mean(knot_variances),
            "spline_stats_history": self.spline_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 Knot 초기화"""
        super().reset()
        self.U_knots = np.zeros((self.P, self.model.control_dim))
        self.spline_stats_history = []

    def __repr__(self) -> str:
        return (
            f"SplineMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"knots={self.P}, degree={self.degree}, "
            f"params={self.params})"
        )
