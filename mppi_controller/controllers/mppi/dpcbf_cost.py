"""
Dynamic Parabolic CBF (DPCBF) 비용 함수

Line-of-Sight (LoS) 좌표계에서 포물선(parabolic) 안전 경계를 정의.
접근 방향에 따라 적응적 안전 영역:
  - 정면 충돌 방향(β≈0): 가장 넓은 안전 영역
  - 측면 통과(β≈±π): 최소 안전 영역

기존 원형 CBF 대비 장점:
  - 측면 통과 시 불필요한 회피 기동 감소
  - 접근 속도가 높을수록 안전 영역 자동 확대
  - LoS 좌표계로 방향별 안전 마진 세밀 조절

Ref: Kim et al. (2026, ICRA)
     "Dynamic Parabolic Control Barrier Function for
      Safe Navigation in Dynamic Environments"

LoS 좌표:
    r   = ||p_robot - p_obs||           (거리)
    β   = atan2(p_rel_y, p_rel_x) - ψ  (LoS 편향각, ψ는 접근 방향)

포물선 안전 경계:
    r_safe(β) = R_eff + a · exp(-β²/(2σ²))
    h = r - r_safe(β)
    h > 0 → 안전, h ≤ 0 → 위험

계수 a는 접근 속도에 비례하여 적응:
    a = a_base + a_vel · max(0, -v_approach)
    v_approach = <v_rel, p_rel_hat> (접근 속도, 음수=접근)
"""

import numpy as np
from typing import List, Tuple, Dict
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class DynamicParabolicCBFCost(CostFunction):
    """
    Dynamic Parabolic CBF (DPCBF) 비용 함수

    LoS 좌표계에서 Gaussian-shaped 안전 경계를 적용.
    접근 방향/속도에 따라 적응적으로 안전 영역 조절.

    장애물: (x, y, radius, vx, vy) 5-tuple.

    Args:
        obstacles: [(x, y, r, vx, vy), ...] 동적 장애물
        cbf_weight: CBF 위반 비용 가중치
        safety_margin: 기본 안전 마진 (m)
        a_base: 포물선 기본 계수 (m)
        a_vel: 접근 속도 비례 계수 (s)
        sigma_beta: 방향 감쇠 폭 (rad)
        dt: 시간 간격 (로봇 속도 계산용)
    """

    def __init__(
        self,
        obstacles: List[Tuple[float, ...]] = None,
        cbf_weight: float = 1000.0,
        safety_margin: float = 0.1,
        a_base: float = 0.3,
        a_vel: float = 0.5,
        sigma_beta: float = 0.8,
        dt: float = 0.05,
    ):
        self.obstacles = obstacles or []
        self.cbf_weight = cbf_weight
        self.safety_margin = safety_margin
        self.a_base = a_base
        self.a_vel = a_vel
        self.sigma_beta = sigma_beta
        self.dt = dt

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        DPCBF 위반 비용 계산

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스

        Returns:
            costs: (K,)
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        if not self.obstacles:
            return costs

        # 로봇 위치: (K, N+1, 2)
        pos = trajectories[:, :, :2]

        # 로봇 속도 (유한 차분): (K, N, 2)
        robot_vel = (pos[:, 1:, :] - pos[:, :-1, :]) / self.dt

        for obs in self.obstacles:
            if len(obs) >= 5:
                ox, oy, obs_r, ovx, ovy = obs[0], obs[1], obs[2], obs[3], obs[4]
            else:
                ox, oy, obs_r = obs[0], obs[1], obs[2]
                ovx, ovy = 0.0, 0.0

            effective_r = obs_r + self.safety_margin

            # ── LoS 좌표 변환 ──

            # 상대 위치: (K, N, 2) — robot - obstacle
            p_rel_x = pos[:, :-1, 0] - ox  # (K, N)
            p_rel_y = pos[:, :-1, 1] - oy  # (K, N)

            # 거리 r = ||p_rel||
            r = np.sqrt(p_rel_x**2 + p_rel_y**2)  # (K, N)
            r_safe_denom = np.maximum(r, 1e-6)

            # 상대 속도: v_robot - v_obs
            v_rel_x = robot_vel[:, :, 0] - ovx  # (K, N)
            v_rel_y = robot_vel[:, :, 1] - ovy  # (K, N)

            # 접근 방향 (LoS 방향): p_rel의 단위 벡터
            p_hat_x = p_rel_x / r_safe_denom
            p_hat_y = p_rel_y / r_safe_denom

            # 접근 속도 (음수 = 접근 중)
            # v_approach = <v_rel, -p_hat> (장애물 방향 속도 성분)
            v_approach = -(v_rel_x * p_hat_x + v_rel_y * p_hat_y)  # (K, N)

            # LoS 편향각 β: 상대 속도 방향과 LoS 방향의 각도 차이
            # β = 0이면 정면 충돌 경로, |β|가 크면 측면 통과
            v_rel_norm = np.sqrt(v_rel_x**2 + v_rel_y**2 + 1e-10)

            # 상대속도 방향
            v_hat_x = v_rel_x / v_rel_norm
            v_hat_y = v_rel_y / v_rel_norm

            # β = LoS 방향과 상대속도 방향의 각도 차이
            # cos(β) = <-v_hat, p_hat> (접근 방향일 때 cos(β) > 0)
            cos_beta = -(v_hat_x * p_hat_x + v_hat_y * p_hat_y)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            beta = np.arccos(cos_beta)  # (K, N), [0, π]

            # ── 적응적 포물선 안전 경계 ──

            # 접근 속도에 비례하여 a 증가
            approaching_speed = np.maximum(0.0, v_approach)
            a = self.a_base + self.a_vel * approaching_speed  # (K, N)

            # Gaussian-shaped 경계: r_safe(β) = R_eff + a · exp(-β²/(2σ²))
            gaussian_bump = a * np.exp(
                -beta**2 / (2.0 * self.sigma_beta**2)
            )  # (K, N)
            r_safe = effective_r + gaussian_bump  # (K, N)

            # ── Barrier 계산 ──

            # h = r - r_safe(β)
            h = r - r_safe  # (K, N)

            # h < 0이면 포물선 안전 경계 내부 → 위반
            violation = np.maximum(0.0, -h)
            costs += self.cbf_weight * np.sum(violation, axis=1)

        return costs

    def get_barrier_info(self, trajectory: np.ndarray) -> Dict:
        """
        Barrier 정보 반환 (디버깅/시각화용)

        Args:
            trajectory: (N+1, nx) 또는 (K, N+1, nx)

        Returns:
            dict: barrier_values, min_barrier, is_safe, safety_boundaries
        """
        if trajectory.ndim == 2:
            trajectory = trajectory[np.newaxis, :, :]

        if not self.obstacles:
            return {
                "barrier_values": np.array([]),
                "min_barrier": float("inf"),
                "is_safe": True,
            }

        pos = trajectory[:, :, :2]
        robot_vel = (pos[:, 1:, :] - pos[:, :-1, :]) / self.dt

        all_h = []
        for obs in self.obstacles:
            if len(obs) >= 5:
                ox, oy, obs_r, ovx, ovy = obs[0], obs[1], obs[2], obs[3], obs[4]
            else:
                ox, oy, obs_r = obs[0], obs[1], obs[2]
                ovx, ovy = 0.0, 0.0

            effective_r = obs_r + self.safety_margin

            p_rel_x = pos[:, :-1, 0] - ox
            p_rel_y = pos[:, :-1, 1] - oy
            r = np.sqrt(p_rel_x**2 + p_rel_y**2)
            r_safe_denom = np.maximum(r, 1e-6)

            v_rel_x = robot_vel[:, :, 0] - ovx
            v_rel_y = robot_vel[:, :, 1] - ovy

            p_hat_x = p_rel_x / r_safe_denom
            p_hat_y = p_rel_y / r_safe_denom

            v_approach = -(v_rel_x * p_hat_x + v_rel_y * p_hat_y)

            v_rel_norm = np.sqrt(v_rel_x**2 + v_rel_y**2 + 1e-10)
            v_hat_x = v_rel_x / v_rel_norm
            v_hat_y = v_rel_y / v_rel_norm

            cos_beta = -(v_hat_x * p_hat_x + v_hat_y * p_hat_y)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            beta = np.arccos(cos_beta)

            approaching_speed = np.maximum(0.0, v_approach)
            a = self.a_base + self.a_vel * approaching_speed
            gaussian_bump = a * np.exp(-beta**2 / (2.0 * self.sigma_beta**2))
            r_safe = effective_r + gaussian_bump

            h = r - r_safe
            all_h.append(h)

        barrier_stack = np.array(all_h)
        min_barrier = float(np.min(barrier_stack))

        return {
            "barrier_values": barrier_stack,
            "min_barrier": min_barrier,
            "is_safe": min_barrier > 0,
        }

    def update_obstacles(self, obstacles: List[Tuple[float, ...]]):
        """동적 장애물 업데이트 — (x, y, r, vx, vy) 형태"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"DynamicParabolicCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"a_base={self.a_base}, a_vel={self.a_vel}, "
            f"weight={self.cbf_weight})"
        )
