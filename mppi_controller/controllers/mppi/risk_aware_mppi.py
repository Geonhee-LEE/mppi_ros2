"""
Risk-Aware MPPI Controller

CVaR (Conditional Value at Risk) 기반 위험 회피 제어.
"""

import numpy as np
from typing import Dict
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import RiskAwareMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


class RiskAwareMPPIController(MPPIController):
    """
    Risk-Aware MPPI Controller

    CVaR (Conditional Value at Risk)을 사용하여 위험 회피 제어를 수행하는 MPPI 변형.

    동작 원리:
        1. 비용 계산: S_k (k=1,...,K)
        2. 비용을 오름차순 정렬
        3. 상위 α*100% 샘플만 선택 (CVaR set)
           - α=1.0: 모든 샘플 (Vanilla MPPI)
           - α=0.9: 상위 90% (하위 10% 제외)
           - α=0.5: 상위 50% (중앙값 기준)
        4. CVaR set의 softmax 가중치 계산
        5. 가중 평균 제어: u* = Σ w_k * u_k

    CVaR 해석:
        - α=1.0: Risk-neutral (위험 중립, Vanilla MPPI)
        - α=0.9: Mildly risk-averse (약한 위험 회피)
        - α=0.5: Highly risk-averse (강한 위험 회피)
        - α=0.1: Extremely risk-averse (극도 위험 회피)

    장점:
        - ✅ 최악 경우 비용 최소화 (tail risk 제거)
        - ✅ 이상치 샘플 무시 (robustness)
        - ✅ 안전성 중요 환경에 적합
        - ✅ α 하나로 위험 회피 강도 조절

    단점:
        - 과도한 위험 회피 시 보수적 제어 (성능 저하)
        - 일부 샘플 정보 손실

    적용 사례:
        - 장애물 회피 (충돌 위험 최소화)
        - 불확실성 높은 환경
        - 안전성 중요 시스템 (의료, 자율주행)

    참고 논문:
        - Yin et al. (2023) - "Risk-Aware MPPI"

    Args:
        model: RobotModel 인스턴스
        params: RiskAwareMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: RiskAwareMPPIParams):
        # 부모 클래스 초기화
        super().__init__(model, params)

        # Risk-Aware 파라미터
        self.risk_params = params
        self.cvar_alpha = params.cvar_alpha

        # 통계 (디버깅용)
        self.risk_stats_history = []

    def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
        """
        CVaR 기반 가중치 계산

        Args:
            costs: (K,) 샘플 비용
            lambda_: 온도 파라미터

        Returns:
            weights: (K,) 정규화된 가중치
        """
        K = len(costs)
        alpha = self.cvar_alpha

        # 1. 비용을 오름차순 정렬 (낮은 비용이 좋음)
        sorted_indices = np.argsort(costs)

        # 2. CVaR cutoff (상위 α*100% 샘플 선택)
        # alpha=1.0이면 모든 샘플, alpha=0.5면 상위 50%
        cvar_count = max(1, int(K * alpha))
        cvar_indices = sorted_indices[:cvar_count]

        # 3. CVaR set의 비용
        cvar_costs = costs[cvar_indices]

        # 4. Baseline 적용 (최소 비용으로 정규화)
        baseline = np.min(cvar_costs)
        cvar_costs_shifted = cvar_costs - baseline

        # 5. CVaR set의 softmax 가중치
        exp_costs = np.exp(-cvar_costs_shifted / lambda_)
        cvar_weights_unnormalized = exp_costs

        # 6. 전체 샘플에 대한 가중치 (CVaR 외부는 0)
        weights = np.zeros(K)
        weights[cvar_indices] = cvar_weights_unnormalized / np.sum(
            cvar_weights_unnormalized
        )

        # 7. VaR (Value at Risk) 계산
        # VaR = CVaR cutoff 지점의 비용
        var_index = sorted_indices[cvar_count - 1]
        var_cost = costs[var_index]

        # 8. CVaR (Conditional Value at Risk) 계산
        # CVaR = CVaR set의 평균 비용
        cvar_cost = np.mean(cvar_costs)

        # 9. ESS 계산
        ess = 1.0 / np.sum(weights**2)

        # 10. 통계 저장
        stats = {
            "cvar_alpha": alpha,
            "cvar_count": cvar_count,
            "var_cost": var_cost,
            "cvar_cost": cvar_cost,
            "baseline": baseline,
            "ess": ess,
            "ess_ratio": ess / K,
            "num_zero_weights": np.sum(weights == 0),
            "num_cvar_samples": cvar_count,
        }

        self.risk_stats_history.append(stats)

        return weights

    def get_risk_statistics(self) -> Dict:
        """
        Risk-Aware 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_var_cost: float 평균 VaR
                - mean_cvar_cost: float 평균 CVaR
                - mean_ess_ratio: float 평균 ESS 비율
                - mean_zero_weights: float 평균 0 가중치 개수
                - risk_stats_history: List[dict] 통계 히스토리
        """
        if len(self.risk_stats_history) == 0:
            return {
                "mean_var_cost": 0.0,
                "mean_cvar_cost": 0.0,
                "mean_ess_ratio": 0.0,
                "mean_zero_weights": 0.0,
                "risk_stats_history": [],
            }

        var_costs = [s["var_cost"] for s in self.risk_stats_history]
        cvar_costs = [s["cvar_cost"] for s in self.risk_stats_history]
        ess_ratios = [s["ess_ratio"] for s in self.risk_stats_history]
        zero_weights = [s["num_zero_weights"] for s in self.risk_stats_history]

        return {
            "mean_var_cost": np.mean(var_costs),
            "mean_cvar_cost": np.mean(cvar_costs),
            "mean_ess_ratio": np.mean(ess_ratios),
            "mean_zero_weights": np.mean(zero_weights),
            "risk_stats_history": self.risk_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.risk_stats_history = []

    def __repr__(self) -> str:
        return (
            f"RiskAwareMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"alpha={self.cvar_alpha:.2f}, "
            f"params={self.params})"
        )
