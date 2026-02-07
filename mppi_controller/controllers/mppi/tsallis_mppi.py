"""
Tsallis-MPPI Controller

Tsallis 엔트로피 기반 가중치 계산으로 탐색-활용 균형 조절.
"""

import numpy as np
from typing import Dict
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import TsallisMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


class TsallisMPPIController(MPPIController):
    """
    Tsallis-MPPI Controller

    Tsallis 엔트로피를 사용하여 탐색-활용 균형을 조절하는 MPPI 변형.

    동작 원리:
        1. 비용 계산: S_k (k=1,...,K)
        2. Tsallis 가중치:
           w_k ∝ [1 - (1-q) * S_k / λ]_+^(1/(1-q))
        3. 정규화: w_k = w_k / Σ w_k
        4. 가중 평균 제어: u* = Σ w_k * u_k

    Tsallis 파라미터 q:
        - q < 1: Heavy-tailed 분포 → 탐색 강화 (샘플 다양성)
        - q = 1: Shannon 엔트로피 → Vanilla MPPI
        - q > 1: Light-tailed 분포 → 활용 강화 (최적 집중)

    수학적 배경:
        - Tsallis 엔트로피: S_q = (1 - Σ p_i^q) / (q-1)
        - q-exponential: exp_q(x) = [1 + (1-q)x]_+^(1/(1-q))
        - q→1 극한에서 Shannon 엔트로피로 수렴

    장점:
        - 탐색-활용 트레이드오프 명시적 제어
        - q 하나로 탐색 강도 조절
        - Vanilla MPPI를 특수 케이스로 포함

    단점:
        - q 튜닝 필요 (문제 의존적)
        - q > 1일 때 가중치 절단 발생 가능

    참고 논문:
        - Yin et al. (2021) - "Tsallis Entropy for MPPI"

    Args:
        model: RobotModel 인스턴스
        params: TsallisMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: TsallisMPPIParams):
        # 부모 클래스 초기화
        super().__init__(model, params)

        # Tsallis 파라미터
        self.tsallis_params = params
        self.tsallis_q = params.tsallis_q

        # 통계 (디버깅용)
        self.tsallis_stats_history = []

    def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Tsallis q-exponential 가중치 계산

        Args:
            costs: (K,) 샘플 비용
            lambda_: 온도 파라미터

        Returns:
            weights: (K,) 정규화된 가중치
        """
        K = len(costs)
        q = self.tsallis_q

        # 1. Baseline 적용 (최소 비용으로 정규화)
        baseline = np.min(costs)
        costs_shifted = costs - baseline

        # 2. Tsallis q-exponential 가중치
        if np.isclose(q, 1.0, atol=1e-6):
            # q=1: Vanilla MPPI (Shannon entropy)
            unnormalized_weights = np.exp(-costs_shifted / lambda_)
        else:
            # q≠1: Tsallis entropy
            # w_k = [1 - (1-q) * S_k / λ]_+^(1/(1-q))
            exponent = -costs_shifted / lambda_
            argument = 1.0 + (1.0 - q) * exponent

            # 양수 부분만 (절단)
            argument = np.maximum(argument, 0.0)

            # q-exponential
            power = 1.0 / (1.0 - q)
            unnormalized_weights = np.power(argument, power)

        # 3. 정규화
        weights_sum = np.sum(unnormalized_weights)
        if weights_sum > 0:
            weights = unnormalized_weights / weights_sum
        else:
            # 모든 가중치가 0인 경우 (극단적 상황)
            weights = np.ones(K) / K

        # 4. ESS 계산
        ess = 1.0 / np.sum(weights**2)

        # 5. 통계 저장
        stats = {
            "tsallis_q": q,
            "baseline": baseline,
            "ess": ess,
            "ess_ratio": ess / K,
            "weights_sum": weights_sum,
            "num_zero_weights": np.sum(weights == 0),
            "max_weight": np.max(weights),
            "min_weight": np.min(weights[weights > 0])
            if np.any(weights > 0)
            else 0.0,
        }

        self.tsallis_stats_history.append(stats)

        return weights

    def get_tsallis_statistics(self) -> Dict:
        """
        Tsallis 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_ess_ratio: float 평균 ESS 비율
                - mean_zero_weights: float 평균 0 가중치 개수
                - tsallis_stats_history: List[dict] 통계 히스토리
        """
        if len(self.tsallis_stats_history) == 0:
            return {
                "mean_ess_ratio": 0.0,
                "mean_zero_weights": 0.0,
                "tsallis_stats_history": [],
            }

        ess_ratios = [s["ess_ratio"] for s in self.tsallis_stats_history]
        zero_weights = [s["num_zero_weights"] for s in self.tsallis_stats_history]

        return {
            "mean_ess_ratio": np.mean(ess_ratios),
            "mean_zero_weights": np.mean(zero_weights),
            "tsallis_stats_history": self.tsallis_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.tsallis_stats_history = []

    def __repr__(self) -> str:
        return (
            f"TsallisMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"q={self.tsallis_q:.2f}, "
            f"params={self.params})"
        )
