"""
Residual Dynamics 모델

물리 기반 모델 + 학습된 보정 = 총 동역학

f_total(x, u) = f_physics(x, u) + f_learned(x, u)
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple, Callable


class ResidualDynamics(RobotModel):
    """
    Residual Dynamics 모델

    물리 기반 모델의 부정확성을 학습된 residual로 보정.

    총 동역학:
        f_total(x, u) = f_physics(x, u) + f_learned(x, u)

    장점:
        - 물리 법칙 보존 (safety)
        - 모델링 오차 보정 (accuracy)
        - 데이터 효율성 (residual만 학습)
        - 외삽 안정성 (physics 기반)

    사용 예시:
        1. 기구학 모델 + 실제 슬립 보정
        2. 동역학 모델 + 마찰 보정
        3. 저차 모델 + 고차 효과 보정

    Args:
        base_model: 물리 기반 RobotModel (기구학 또는 동역학)
        residual_fn: (state, control) → residual_state_dot
            학습된 residual 함수 (신경망, GP 등)
            None이면 base_model과 동일 (디버깅용)
        uncertainty_fn: (state, control) → uncertainty
            불확실성 정량화 함수 (GP 등)
            None이면 불확실성 정보 없음
        use_residual: True면 residual 활성화, False면 base_model만
    """

    def __init__(
        self,
        base_model: RobotModel,
        residual_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        uncertainty_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        use_residual: bool = True,
        learned_model: Optional[RobotModel] = None,
    ):
        self.base_model = base_model
        self.use_residual = use_residual

        # learned_model이 주어지면 자동으로 residual_fn/uncertainty_fn 연결
        if learned_model is not None:
            self.learned_model = learned_model
            self.residual_fn = learned_model.forward_dynamics
            # GP 모델이면 uncertainty_fn 자동 연결
            if hasattr(learned_model, 'predict_with_uncertainty'):
                self.uncertainty_fn = lambda s, u: learned_model.predict_with_uncertainty(s, u)[1]
            else:
                self.uncertainty_fn = uncertainty_fn
        else:
            self.learned_model = None
            self.residual_fn = residual_fn
            self.uncertainty_fn = uncertainty_fn

        # 통계 (디버깅용)
        self.stats = {
            "residual_mean": None,
            "residual_std": None,
            "num_calls": 0,
        }

    @property
    def state_dim(self) -> int:
        return self.base_model.state_dim

    @property
    def control_dim(self) -> int:
        return self.base_model.control_dim

    @property
    def model_type(self) -> str:
        return "learned"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        총 동역학: f_total = f_physics + f_learned

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            state_dot: (nx,) 또는 (batch, nx)
        """
        # 1. 물리 기반 동역학
        physics_dot = self.base_model.forward_dynamics(state, control)

        # 2. Residual 추가 (있을 경우)
        if self.use_residual and self.residual_fn is not None:
            residual_dot = self.residual_fn(state, control)

            # 통계 업데이트 (디버깅)
            self._update_stats(residual_dot)

            total_dot = physics_dot + residual_dot
        else:
            total_dot = physics_dot

        return total_dot

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """제어 제약 (base_model과 동일)"""
        return self.base_model.get_control_bounds()

    def state_to_dict(self, state: np.ndarray) -> dict:
        """상태 딕셔너리 (base_model 위임)"""
        return self.base_model.state_to_dict(state)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """상태 정규화 (base_model 위임)"""
        return self.base_model.normalize_state(state)

    def get_uncertainty(
        self, state: np.ndarray, control: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        불확실성 정량화 (GP 등)

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            uncertainty: (nx,) 또는 (batch, nx) - 표준편차
            None if uncertainty_fn is None
        """
        if self.uncertainty_fn is not None:
            return self.uncertainty_fn(state, control)
        return None

    def get_residual_contribution(
        self, state: np.ndarray, control: np.ndarray
    ) -> dict:
        """
        Residual 기여도 분석 (디버깅/해석용)

        Args:
            state: (nx,)
            control: (nu,)

        Returns:
            dict:
                - physics_dot: (nx,) 물리 기반 동역학
                - residual_dot: (nx,) 학습된 보정
                - total_dot: (nx,) 총 동역학
                - residual_ratio: (nx,) residual / total (비율)
        """
        physics_dot = self.base_model.forward_dynamics(state, control)

        if self.use_residual and self.residual_fn is not None:
            residual_dot = self.residual_fn(state, control)
            total_dot = physics_dot + residual_dot

            # Residual 기여도 비율
            residual_ratio = np.zeros_like(residual_dot)
            nonzero = np.abs(total_dot) > 1e-6
            residual_ratio[nonzero] = (
                residual_dot[nonzero] / total_dot[nonzero]
            )
        else:
            residual_dot = np.zeros_like(physics_dot)
            total_dot = physics_dot
            residual_ratio = np.zeros_like(physics_dot)

        return {
            "physics_dot": physics_dot,
            "residual_dot": residual_dot,
            "total_dot": total_dot,
            "residual_ratio": residual_ratio,
        }

    def _update_stats(self, residual_dot: np.ndarray):
        """통계 업데이트 (디버깅용)"""
        # 배치인 경우 평균
        if residual_dot.ndim > 1:
            residual_dot = np.mean(residual_dot, axis=0)

        self.stats["num_calls"] += 1

        # Running mean/std (Welford's algorithm)
        if self.stats["residual_mean"] is None:
            self.stats["residual_mean"] = residual_dot.copy()
            self.stats["residual_std"] = np.zeros_like(residual_dot)
        else:
            delta = residual_dot - self.stats["residual_mean"]
            self.stats["residual_mean"] += delta / self.stats["num_calls"]
            delta2 = residual_dot - self.stats["residual_mean"]
            self.stats["residual_std"] += delta * delta2

    def get_stats(self) -> dict:
        """통계 반환 (디버깅용)"""
        if self.stats["num_calls"] > 1:
            std = np.sqrt(self.stats["residual_std"] / (self.stats["num_calls"] - 1))
        else:
            std = self.stats["residual_std"]

        return {
            "residual_mean": self.stats["residual_mean"],
            "residual_std": std,
            "num_calls": self.stats["num_calls"],
        }

    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            "residual_mean": None,
            "residual_std": None,
            "num_calls": 0,
        }

    def __repr__(self) -> str:
        residual_status = "enabled" if self.use_residual else "disabled"
        return (
            f"ResidualDynamics("
            f"base={self.base_model.__class__.__name__}, "
            f"residual={residual_status})"
        )


# ========== 더미 Residual 함수 예시 ==========

def create_constant_residual(residual_value: np.ndarray) -> Callable:
    """
    상수 residual 함수 생성 (테스트용)

    Args:
        residual_value: (nx,) 상수 보정값

    Returns:
        residual_fn: (state, control) → residual_value
    """
    def residual_fn(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        # 배치 지원
        if state.ndim == 1:
            return residual_value.copy()
        else:
            batch_size = state.shape[0]
            return np.tile(residual_value, (batch_size, 1))

    return residual_fn


def create_state_dependent_residual(
    correction_gain: np.ndarray
) -> Callable:
    """
    상태 의존 residual 함수 (예: 슬립 보정)

    residual = gain * state

    Args:
        correction_gain: (nx, nx) 보정 게인 행렬

    Returns:
        residual_fn: (state, control) → residual
    """
    def residual_fn(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        # state @ correction_gain.T
        if state.ndim == 1:
            return state @ correction_gain.T
        else:
            return state @ correction_gain.T

    return residual_fn


def create_control_dependent_residual(
    correction_matrix: np.ndarray
) -> Callable:
    """
    제어 의존 residual 함수 (예: 액추에이터 bias)

    residual = correction_matrix @ control

    Args:
        correction_matrix: (nx, nu) 보정 행렬

    Returns:
        residual_fn: (state, control) → residual
    """
    def residual_fn(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        # control @ correction_matrix.T
        if control.ndim == 1:
            return control @ correction_matrix.T
        else:
            return control @ correction_matrix.T

    return residual_fn


def create_sine_residual(
    amplitude: np.ndarray, frequency: float, phase: float = 0.0
) -> Callable:
    """
    주기적 residual 함수 (예: 주기적 외란)

    residual = amplitude * sin(2π * frequency * t + phase)

    Args:
        amplitude: (nx,) 진폭
        frequency: 주파수 (Hz)
        phase: 위상 (rad)

    Returns:
        residual_fn: (state, control) → residual

    Note:
        시간 t는 state[0] / v_nominal로 추정 (근사)
    """
    import time as time_module

    t_start = time_module.time()

    def residual_fn(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        t = time_module.time() - t_start
        residual_value = amplitude * np.sin(2 * np.pi * frequency * t + phase)

        # 배치 지원
        if state.ndim == 1:
            return residual_value
        else:
            batch_size = state.shape[0]
            return np.tile(residual_value, (batch_size, 1))

    return residual_fn
