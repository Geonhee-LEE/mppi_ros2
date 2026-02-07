"""
Adaptive Temperature (적응적 온도 조절)

ESS (Effective Sample Size) 기반 λ 자동 튜닝.
"""

import numpy as np
from typing import Optional


class AdaptiveTemperature:
    """
    Adaptive Temperature (적응적 온도 조절)

    ESS (Effective Sample Size)를 목표값으로 유지하도록 λ를 자동 조정.

    동작 원리:
        - ESS = 1 / Σ w_k² (가중치 집중도 측정)
        - ESS가 낮으면 (소수 샘플에 집중) → λ 증가 (탐색 강화)
        - ESS가 높으면 (균등 분포) → λ 감소 (활용 강화)

    장점:
        - 자동 탐색-활용 균형
        - 수동 튜닝 불필요
        - 환경 변화에 적응

    참고:
        mppi_playground의 adaptive temperature 구현 참고

    Args:
        initial_lambda: 초기 온도 파라미터
        target_ess_ratio: 목표 ESS 비율 (0~1, 1이면 균등 분포)
        adaptation_rate: 적응 속도 (0~1)
        lambda_min: 최소 온도
        lambda_max: 최대 온도
    """

    def __init__(
        self,
        initial_lambda: float = 1.0,
        target_ess_ratio: float = 0.5,
        adaptation_rate: float = 0.1,
        lambda_min: float = 0.1,
        lambda_max: float = 100.0,
    ):
        self.lambda_ = initial_lambda
        self.target_ess_ratio = target_ess_ratio
        self.adaptation_rate = adaptation_rate
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # 히스토리 (디버깅용)
        self.lambda_history = [initial_lambda]
        self.ess_history = []

    def update(self, weights: np.ndarray, K: int) -> float:
        """
        ESS 기반 λ 업데이트

        Args:
            weights: (K,) 정규화된 가중치
            K: 샘플 개수

        Returns:
            updated_lambda: 업데이트된 온도 파라미터
        """
        # 1. ESS 계산
        ess = 1.0 / np.sum(weights**2)
        ess_ratio = ess / K

        # 2. ESS 비율과 목표 비교
        ess_error = ess_ratio - self.target_ess_ratio

        # 3. λ 업데이트 (proportional control)
        # ESS가 목표보다 낮으면 (집중) → λ 증가 (탐색)
        # ESS가 목표보다 높으면 (분산) → λ 감소 (활용)
        delta_lambda = -self.adaptation_rate * ess_error * self.lambda_

        self.lambda_ = self.lambda_ + delta_lambda

        # 4. λ 제한
        self.lambda_ = np.clip(self.lambda_, self.lambda_min, self.lambda_max)

        # 5. 히스토리 저장
        self.lambda_history.append(self.lambda_)
        self.ess_history.append(ess)

        return self.lambda_

    def get_lambda(self) -> float:
        """현재 λ 반환"""
        return self.lambda_

    def reset(self, initial_lambda: Optional[float] = None):
        """λ 초기화"""
        if initial_lambda is not None:
            self.lambda_ = initial_lambda
        self.lambda_history = [self.lambda_]
        self.ess_history = []

    def get_statistics(self) -> dict:
        """통계 반환 (디버깅용)"""
        if len(self.ess_history) == 0:
            return {
                "mean_lambda": self.lambda_,
                "mean_ess": 0.0,
                "lambda_history": self.lambda_history.copy(),
                "ess_history": [],
            }

        return {
            "mean_lambda": np.mean(self.lambda_history),
            "mean_ess": np.mean(self.ess_history),
            "lambda_history": self.lambda_history.copy(),
            "ess_history": self.ess_history.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveTemperature("
            f"lambda_={self.lambda_:.2f}, "
            f"target_ess_ratio={self.target_ess_ratio:.2f})"
        )
