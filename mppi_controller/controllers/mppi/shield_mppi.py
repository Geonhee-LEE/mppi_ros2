"""
Shield-MPPI Controller

Rollout 중 매 timestep마다 CBF 제약을 해석적으로 적용하여
모든 K개 샘플 궤적이 안전하도록 보장.

Approach C: Shielded Rollout — per-step CBF enforcement during rollout.

기존 CBF-MPPI:
  rollout(raw controls) → unsafe trajectories → CBF cost penalty → softmax

Shield-MPPI:
  shielded_rollout(raw controls → per-step CBF clip → safe controls)
  → ALL safe trajectories → cost → softmax

Reference: Yin et al. (2023) "Shield Model Predictive Path Integral"
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import ShieldMPPIParams
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class ShieldMPPIController(CBFMPPIController):
    """
    Shield-MPPI Controller

    CBF-MPPI를 확장하여 rollout 중 매 timestep마다 해석적 CBF 제약을 적용.
    모든 K개 샘플 궤적이 안전한 상태를 보장.

    핵심 차이점:
    - CBF-MPPI: 안전하지 않은 궤적에 비용 페널티만 부과
    - Shield-MPPI: rollout 중 제어를 클리핑하여 모든 궤적이 안전

    해석적 CBF (Diff Drive Closed-Form):
        h(x) = (x-xo)² + (y-yo)² - r_eff²
        Lg_h = [2(x-xo)·cos(θ) + 2(y-yo)·sin(θ), 0]

        CBF 제약: Lg_h[0]·v + α·h ≥ 0
        Lg_h[0] < 0 (장애물 접근): v ≤ α·h / |Lg_h[0]|
        Lg_h[0] ≥ 0 (이탈): 무조건 만족
        ω: 항상 자유 (Lg_h[1] = 0)

    가중 업데이트: shielded noise (shielded_controls - U) 사용으로 편향 방지.

    Args:
        model: RobotModel 인스턴스
        params: ShieldMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 + CBF 비용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: ShieldMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)

        self.shield_params = params
        self._shield_enabled = params.shield_enabled
        self._shield_alpha = (
            params.shield_cbf_alpha
            if params.shield_cbf_alpha is not None
            else params.cbf_alpha
        )

        # Shield 통계
        self.shield_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Shield-MPPI 제어 계산 (전체 오버라이드)

        1. 노이즈 샘플링
        2. Shielded rollout (per-step CBF enforcement)
        3. 안전한 궤적에 대해 비용 계산
        4. 표준 softmax 가중치 계산
        5. Shielded noise로 제어 업데이트
        6. (선택) QP 출력 필터

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - MPPI info + CBF info + Shield info
        """
        if not self._shield_enabled:
            return super().compute_control(state, reference_trajectory)

        K = self.params.K
        N = self.params.N

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. Shielded rollout
        sample_trajectories, shielded_controls, shield_info = (
            self._shielded_rollout(state, sampled_controls)
        )

        # 4. 비용 계산 (안전한 궤적 기반)
        costs = self.cost_function.compute_cost(
            sample_trajectories, shielded_controls, reference_trajectory
        )

        # 5. MPPI 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. Shielded noise로 제어 업데이트 (편향 방지)
        shielded_noise = shielded_controls - self.U  # (K, N, nu)
        weighted_noise = np.sum(
            weights[:, None, None] * shielded_noise, axis=0
        )  # (N, nu)
        self.U = self.U + weighted_noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 7. Receding horizon 시프트
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 8. 최적 제어
        optimal_control = self.U[0, :]

        # 9. Layer 2: 안전 필터 (optional)
        filter_info = {"filtered": False, "correction_norm": 0.0}
        if self.safety_filter is not None:
            optimal_control, filter_info = self.safety_filter.filter_control(
                state, optimal_control, self.u_min, self.u_max
            )

        # 10. Barrier info 수집
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)
        best_traj = sample_trajectories[best_idx]
        barrier_info = self.cbf_cost.get_barrier_info(best_traj)

        # Shield 통계 저장
        shield_stats = {
            "intervention_rate": shield_info["intervention_rate"],
            "mean_vel_reduction": shield_info["mean_vel_reduction"],
            "total_interventions": shield_info["total_interventions"],
            "total_steps": shield_info["total_steps"],
        }
        self.shield_stats_history.append(shield_stats)

        # CBF 통계 저장
        cbf_stats = {
            "min_barrier": barrier_info["min_barrier"],
            "is_safe": barrier_info["is_safe"],
            "filtered": filter_info["filtered"],
            "correction_norm": filter_info.get("correction_norm", 0.0),
        }
        self.cbf_stats_history.append(cbf_stats)

        # info 구성
        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": best_traj,
            "best_cost": costs[best_idx],
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            # CBF info
            "barrier_values": barrier_info["barrier_values"],
            "min_barrier": barrier_info["min_barrier"],
            "is_safe": barrier_info["is_safe"],
            "cbf_filtered": filter_info["filtered"],
            "cbf_correction_norm": filter_info.get("correction_norm", 0.0),
            # Shield info
            "shield_intervention_rate": shield_info["intervention_rate"],
            "shield_mean_vel_reduction": shield_info["mean_vel_reduction"],
            "shielded_controls": shielded_controls,
        }
        self.last_info = info

        return optimal_control, info

    def _shielded_rollout(
        self, initial_state: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Shielded rollout: 매 timestep마다 CBF 제약 적용

        Args:
            initial_state: (nx,) 초기 상태
            controls: (K, N, nu) 원본 제어 시퀀스

        Returns:
            trajectories: (K, N+1, nx) 안전한 궤적
            shielded_controls: (K, N, nu) CBF 적용된 제어
            info: dict - shield 통계
        """
        K, N, nu = controls.shape
        nx = self.model.state_dim

        trajectories = np.zeros((K, N + 1, nx))
        shielded_controls = controls.copy()
        trajectories[:, 0, :] = initial_state

        total_interventions = 0
        total_vel_reduction = 0.0
        total_steps = K * N

        for t in range(N):
            states_t = trajectories[:, t, :]  # (K, nx)
            controls_t = controls[:, t, :]  # (K, nu)

            # CBF shield 적용
            safe_controls_t, intervened, vel_reduction = (
                self._cbf_shield_batch(states_t, controls_t)
            )

            shielded_controls[:, t, :] = safe_controls_t
            total_interventions += np.sum(intervened)
            total_vel_reduction += np.sum(vel_reduction)

            # 안전한 제어로 다음 상태 전파
            trajectories[:, t + 1, :] = self.model.step(
                states_t, safe_controls_t, self.params.dt
            )

        num_intervened = total_interventions
        info = {
            "intervention_rate": float(num_intervened / total_steps) if total_steps > 0 else 0.0,
            "mean_vel_reduction": float(total_vel_reduction / max(num_intervened, 1)),
            "total_interventions": int(num_intervened),
            "total_steps": total_steps,
        }

        return trajectories, shielded_controls, info

    def _cbf_shield_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        해석적 CBF shield (완전 벡터화)

        Differential drive closed-form:
            h(x) = (x-xo)² + (y-yo)² - r_eff²
            Lg_h[0] = 2(x-xo)·cos(θ) + 2(y-yo)·sin(θ)

            Lg_h[0] < 0 (접근): v_ceiling = α·h / |Lg_h[0]|
            Lg_h[0] ≥ 0 (이탈): 무조건 만족
            ω: 항상 자유

        Args:
            states: (K, nx) 현재 상태 [x, y, θ]
            controls: (K, nu) 원본 제어 [v, ω]

        Returns:
            safe_controls: (K, nu) CBF 적용된 제어
            intervened: (K,) bool, 개입 여부
            vel_reduction: (K,) 속도 감소량 (원본 - 클리핑)
        """
        K = states.shape[0]
        safe_controls = controls.copy()
        v_original = controls[:, 0]  # (K,)

        x = states[:, 0]  # (K,)
        y = states[:, 1]  # (K,)
        theta = states[:, 2]  # (K,)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        obstacles = self.cbf_params.cbf_obstacles
        safety_margin = self.cbf_params.cbf_safety_margin
        alpha = self._shield_alpha

        # 모든 장애물에 대해 v_ceiling 계산 후 최보수적(최소) 적용
        v_ceiling = np.full(K, np.inf)

        for obs_x, obs_y, obs_r in obstacles:
            effective_r = obs_r + safety_margin

            dx = x - obs_x  # (K,)
            dy = y - obs_y  # (K,)

            # Barrier value
            h = dx**2 + dy**2 - effective_r**2  # (K,)

            # Lg_h[0] = 2*dx*cos(θ) + 2*dy*sin(θ)
            Lg_h_v = 2.0 * dx * cos_theta + 2.0 * dy * sin_theta  # (K,)

            # CBF 제약: Lg_h_v * v + α * h ≥ 0
            # 접근 시 (Lg_h_v < 0): v ≤ α * h / |Lg_h_v|
            # 이탈 시 (Lg_h_v ≥ 0): 무조건 만족 → ceiling = inf

            # Lg_h_v < 0인 경우만 제약 적용
            approaching = Lg_h_v < -1e-10  # 수치 안정성

            # v_ceiling_obs = α * h / |Lg_h_v| (접근 시만)
            # h < 0이면 ceiling도 음수 → 후진만 허용
            v_ceiling_obs = np.where(
                approaching,
                alpha * h / np.maximum(np.abs(Lg_h_v), 1e-10),
                np.inf,
            )

            v_ceiling = np.minimum(v_ceiling, v_ceiling_obs)

        # 속도 클리핑: v_safe = min(v_original, v_ceiling)
        # v_ceiling이 음수일 수 있음 (장애물 내부에서 후진 필요)
        v_safe = np.minimum(v_original, v_ceiling)

        # 제어 제약 클리핑 적용
        if self.u_min is not None:
            v_safe = np.maximum(v_safe, self.u_min[0])

        safe_controls[:, 0] = v_safe

        # 개입 여부 및 속도 감소량
        intervened = v_safe < v_original - 1e-10
        vel_reduction = np.where(intervened, v_original - v_safe, 0.0)

        return safe_controls, intervened, vel_reduction

    def get_shield_statistics(self) -> Dict:
        """Shield 통계 반환"""
        if not self.shield_stats_history:
            return {
                "mean_intervention_rate": 0.0,
                "max_intervention_rate": 0.0,
                "mean_vel_reduction": 0.0,
                "total_interventions": 0,
                "num_steps": 0,
            }

        rates = [s["intervention_rate"] for s in self.shield_stats_history]
        reductions = [s["mean_vel_reduction"] for s in self.shield_stats_history]
        total_int = sum(s["total_interventions"] for s in self.shield_stats_history)

        return {
            "mean_intervention_rate": float(np.mean(rates)),
            "max_intervention_rate": float(np.max(rates)),
            "mean_vel_reduction": float(np.mean(reductions)),
            "total_interventions": total_int,
            "num_steps": len(self.shield_stats_history),
        }

    def update_obstacles(self, obstacles: List[tuple]):
        """
        동적 장애물 업데이트 (Shield 내부 참조도 갱신)

        부모 CBFMPPIController.update_obstacles()는 cbf_cost와 safety_filter를
        업데이트하지만, Shield의 _cbf_shield_batch()는 self.cbf_params.cbf_obstacles를
        직접 참조하므로 여기서도 갱신해야 함.
        """
        super().update_obstacles(obstacles)
        self.cbf_params.cbf_obstacles = obstacles

    def set_shield_enabled(self, enabled: bool):
        """Shield 활성화/비활성화"""
        self._shield_enabled = enabled

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.shield_stats_history = []

    def __repr__(self) -> str:
        return (
            f"ShieldMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"obstacles={len(self.cbf_params.cbf_obstacles)}, "
            f"shield_enabled={self._shield_enabled}, "
            f"alpha={self._shield_alpha})"
        )
