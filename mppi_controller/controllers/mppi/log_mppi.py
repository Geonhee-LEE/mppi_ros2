"""
Log-MPPI Controller

수치 안정성을 위한 log-space softmax 가중치 계산.
"""

import numpy as np
from typing import Dict, Tuple
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import LogMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


class LogMPPIController(MPPIController):
    """
    Log-MPPI Controller

    수치 안정성을 위해 log-space에서 softmax 가중치를 계산하는 MPPI 변형.

    동작 원리:
        1. 비용 계산: S_k (k=1,...,K)
        2. Log-space 가중치:
           log w_k = -S_k/λ - log_sum_exp(-S/λ)
        3. Exp-space 변환: w_k = exp(log w_k)
        4. 가중 평균 제어: u* = Σ w_k * u_k

    장점:
        - exp overflow/underflow 방지
        - 높은 비용 차이에서도 안정적
        - Vanilla MPPI와 동일한 결과 (수치 정밀도 향상)

    단점:
        - 약간의 추가 계산 (log_sum_exp)
        - 매우 큰 K에서는 메모리 사용량 증가 가능

    참고:
        - Log-sum-exp trick: log Σ exp(x_i) = max(x) + log Σ exp(x_i - max(x))

    Args:
        model: RobotModel 인스턴스
        params: LogMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: LogMPPIParams):
        # 부모 클래스 초기화
        super().__init__(model, params)

        # Log-MPPI 파라미터
        self.log_params = params
        self.use_baseline = params.use_baseline

        # 통계 (디버깅용)
        self.log_weight_stats_history = []

    def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Log-space softmax 가중치 계산

        Args:
            costs: (K,) 샘플 비용
            lambda_: 온도 파라미터 (사용하지 않지만 시그니처 일관성)

        Returns:
            weights: (K,) 정규화된 가중치
        """
        K = len(costs)

        # 1. Baseline 적용 (선택적)
        if self.use_baseline:
            # Baseline: 최소 비용
            baseline = np.min(costs)
            costs_shifted = costs - baseline
        else:
            costs_shifted = costs
            baseline = 0.0

        # 2. Log-space 가중치 계산
        # log w_k = -S_k/λ (self.params.lambda_ 사용)
        log_weights_unnorm = -costs_shifted / self.params.lambda_

        # 3. Log-sum-exp trick으로 정규화
        # log Z = log Σ exp(-S_k/λ)
        log_Z = self._log_sum_exp(log_weights_unnorm)

        # 4. 정규화된 log 가중치
        log_weights = log_weights_unnorm - log_Z

        # 5. Exp-space 변환
        weights = np.exp(log_weights)

        # 6. 수치 검증 (sum = 1)
        weights = weights / np.sum(weights)  # 추가 정규화 (수치 오차 보정)

        # 7. ESS 계산
        ess = 1.0 / np.sum(weights**2)

        # 8. 통계 저장
        stats = {
            "log_weights": log_weights.copy(),
            "log_Z": log_Z,
            "baseline": baseline,
            "ess": ess,
            "ess_ratio": ess / K,
            "max_log_weight": np.max(log_weights),
            "min_log_weight": np.min(log_weights),
        }

        self.log_weight_stats_history.append(stats)

        return weights

    def _log_sum_exp(self, log_values: np.ndarray) -> float:
        """
        Log-sum-exp trick으로 수치 안정적 계산

        Args:
            log_values: (K,) log-space 값

        Returns:
            log_sum: log Σ exp(log_values)
        """
        # log Σ exp(x_i) = max(x) + log Σ exp(x_i - max(x))
        max_log = np.max(log_values)
        log_sum = max_log + np.log(np.sum(np.exp(log_values - max_log)))
        return log_sum

    def get_log_weight_statistics(self) -> Dict:
        """
        Log 가중치 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_log_Z: float 평균 log partition
                - mean_ess_ratio: float 평균 ESS 비율
                - log_weight_stats_history: List[dict] 통계 히스토리
        """
        if len(self.log_weight_stats_history) == 0:
            return {
                "mean_log_Z": 0.0,
                "mean_ess_ratio": 0.0,
                "log_weight_stats_history": [],
            }

        log_Zs = [s["log_Z"] for s in self.log_weight_stats_history]
        ess_ratios = [s["ess_ratio"] for s in self.log_weight_stats_history]

        return {
            "mean_log_Z": np.mean(log_Zs),
            "mean_ess_ratio": np.mean(ess_ratios),
            "log_weight_stats_history": self.log_weight_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.log_weight_stats_history = []

    def __repr__(self) -> str:
        baseline_status = "enabled" if self.use_baseline else "disabled"
        return (
            f"LogMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"baseline={baseline_status}, "
            f"params={self.params})"
        )
