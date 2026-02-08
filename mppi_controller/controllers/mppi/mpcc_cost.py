"""
MPCC (Model Predictive Contouring Control) Cost Function

경로 추종을 Contouring error(수직) + Lag error(수평) + Progress(진행도)로 분리.
속도 조절이 자연스럽고, 곡선 경로 추종에서 기존 tracking cost보다 우수.

핵심 수학:
  경로 파라미터화: p_ref(θ) (호장 θ ∈ [0, L])
  로봇 위치 p = (x, y) → 최근점 θ* 투영

  ψ_path = 경로 방향 (heading at θ*)
  e_c = -(x-x_ref)·sin(ψ) + (y-y_ref)·cos(ψ)   # contouring (수직)
  e_l =  (x-x_ref)·cos(ψ) + (y-y_ref)·sin(ψ)   # lag (수평)

  J_mpcc = Σ_t [Q_c·e_c² + Q_l·e_l²] - Q_θ·(θ*_T - θ*_0) + Q_h·Σ heading_err²

Ref: Liniger et al. (2015) "Optimization-based Autonomous Racing"
"""

import numpy as np
from typing import Optional, Dict
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class PathParameterization:
    """
    경로 파라미터화 — 웨이포인트 연결 선분으로 호장(arc-length) 파라미터화.

    waypoints를 선분으로 연결하고, 임의 (x,y) 점의 최근점 투영(θ*)과
    경로 방향(ψ_path)을 벡터화하여 계산.

    Args:
        waypoints: (M, 2) 또는 (M, 3) — 경로 웨이포인트 (x, y[, θ])
    """

    def __init__(self, waypoints: np.ndarray):
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[0] < 2:
            raise ValueError("waypoints must be (M, 2+) with M >= 2")
        self._xy = waypoints[:, :2].copy()  # (M, 2)
        M = self._xy.shape[0]

        # 각 선분: p[i] → p[i+1]
        diffs = np.diff(self._xy, axis=0)  # (M-1, 2)
        self._seg_lengths = np.linalg.norm(diffs, axis=1)  # (M-1,)
        # 누적 호장
        self._cum_lengths = np.concatenate(
            [[0.0], np.cumsum(self._seg_lengths)]
        )  # (M,)
        self._total_length = self._cum_lengths[-1]

        # 단위 방향 벡터 및 heading
        safe_len = np.maximum(self._seg_lengths, 1e-12)
        self._seg_dirs = diffs / safe_len[:, None]  # (M-1, 2)
        self._seg_headings = np.arctan2(
            self._seg_dirs[:, 1], self._seg_dirs[:, 0]
        )  # (M-1,)

    @property
    def total_length(self) -> float:
        return float(self._total_length)

    def project(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple:
        """
        (x, y) 점들을 경로에 투영.

        Args:
            x: (*shape,) x 좌표
            y: (*shape,) y 좌표

        Returns:
            theta_star: (*shape,) 호장 파라미터
            psi_path: (*shape,) 경로 heading
            closest: (*shape, 2) 최근점 좌표
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        orig_shape = x.shape
        xf = x.ravel()  # (P,)
        yf = y.ravel()
        P = xf.size
        n_seg = len(self._seg_lengths)

        # 모든 점 × 모든 선분 조합 (P, n_seg)
        # 각 점에서 각 선분 시작까지의 벡터
        px = xf[:, None] - self._xy[:-1, 0][None, :]  # (P, n_seg)
        py = yf[:, None] - self._xy[:-1, 1][None, :]

        # 선분 방향에 대한 투영 t ∈ [0, seg_len]
        t_proj = px * self._seg_dirs[:, 0] + py * self._seg_dirs[:, 1]  # (P, n_seg)
        t_proj = np.clip(t_proj, 0.0, self._seg_lengths[None, :])

        # 최근점 좌표
        cx = self._xy[:-1, 0] + t_proj * self._seg_dirs[:, 0]  # (P, n_seg)
        cy = self._xy[:-1, 1] + t_proj * self._seg_dirs[:, 1]

        # 거리 제곱
        dist_sq = (xf[:, None] - cx) ** 2 + (yf[:, None] - cy) ** 2  # (P, n_seg)

        # 최소 거리 선분 인덱스
        best_seg = np.argmin(dist_sq, axis=1)  # (P,)
        idx = np.arange(P)

        theta_star = self._cum_lengths[best_seg] + t_proj[idx, best_seg]
        psi_path = self._seg_headings[best_seg]
        closest = np.stack(
            [cx[idx, best_seg], cy[idx, best_seg]], axis=-1
        )  # (P, 2)

        return (
            theta_star.reshape(orig_shape),
            psi_path.reshape(orig_shape),
            closest.reshape(orig_shape + (2,)),
        )

    def get_point(self, theta: np.ndarray) -> np.ndarray:
        """
        호장 θ에서 경로 점 반환.

        Args:
            theta: (*shape,) 호장 파라미터

        Returns:
            points: (*shape, 2)
        """
        theta = np.asarray(theta, dtype=np.float64)
        orig_shape = theta.shape
        tf = np.clip(theta.ravel(), 0.0, self._total_length)  # (P,)

        # 각 점이 어느 선분에 속하는지
        seg_idx = np.searchsorted(self._cum_lengths[1:], tf, side="right")
        seg_idx = np.clip(seg_idx, 0, len(self._seg_lengths) - 1)

        local_t = tf - self._cum_lengths[seg_idx]
        pts_x = self._xy[seg_idx, 0] + local_t * self._seg_dirs[seg_idx, 0]
        pts_y = self._xy[seg_idx, 1] + local_t * self._seg_dirs[seg_idx, 1]

        return np.stack([pts_x, pts_y], axis=-1).reshape(orig_shape + (2,))


class MPCCCost(CostFunction):
    """
    MPCC Cost Function

    경로 추종을 contouring(수직) + lag(수평) + progress(진행도)로 분리.

    J = Σ_t [Q_c·e_c² + Q_l·e_l²] - Q_θ·(θ*_T - θ*_0) + Q_h·Σ heading_err²

    Args:
        reference_path: (M, 2+) 경로 웨이포인트, 또는 None (나중에 set_reference_path)
        Q_c: contouring error 가중치
        Q_l: lag error 가중치
        Q_theta: progress 보상 가중치
        Q_heading: heading error 가중치
    """

    def __init__(
        self,
        reference_path: Optional[np.ndarray] = None,
        Q_c: float = 50.0,
        Q_l: float = 10.0,
        Q_theta: float = 5.0,
        Q_heading: float = 1.0,
    ):
        self.Q_c = Q_c
        self.Q_l = Q_l
        self.Q_theta = Q_theta
        self.Q_heading = Q_heading
        self._path: Optional[PathParameterization] = None

        if reference_path is not None:
            self.set_reference_path(reference_path)

    def set_reference_path(self, waypoints: np.ndarray):
        """경로 설정/업데이트"""
        self._path = PathParameterization(waypoints)

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        MPCC 비용 계산.

        reference_trajectory가 전달될 때:
          - reference_path가 미설정이면 reference_trajectory[:, :2]를 경로로 사용.

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        K, N_plus_1, nx = trajectories.shape

        # 경로 미설정 시 reference_trajectory를 경로로 사용
        if self._path is None:
            if reference_trajectory is not None:
                self.set_reference_path(reference_trajectory[:, :2])
            else:
                return np.zeros(K)

        # 로봇 x, y 좌표 — (K, N+1)
        rx = trajectories[:, :, 0]
        ry = trajectories[:, :, 1]

        # 경로 투영 — (K, N+1)
        theta_star, psi_path, _ = self._path.project(rx, ry)

        # 오차 벡터 (K, N+1)
        dx = rx - self._path.get_point(theta_star)[..., 0]  # 이미 closest와 동일
        dy = ry - self._path.get_point(theta_star)[..., 1]

        # contouring & lag errors
        cos_psi = np.cos(psi_path)
        sin_psi = np.sin(psi_path)
        e_c = -dx * sin_psi + dy * cos_psi  # (K, N+1) contouring
        e_l = dx * cos_psi + dy * sin_psi   # (K, N+1) lag

        # 비용 항
        cost_contouring = self.Q_c * np.sum(e_c[:, :-1] ** 2, axis=1)  # (K,)
        cost_lag = self.Q_l * np.sum(e_l[:, :-1] ** 2, axis=1)         # (K,)

        # progress 보상 (음수 비용 = 보상)
        progress = theta_star[:, -1] - theta_star[:, 0]  # (K,)
        cost_progress = -self.Q_theta * progress

        # heading 비용
        cost_heading = np.zeros(K)
        if nx >= 3 and self.Q_heading > 0:
            robot_theta = trajectories[:, :-1, 2]  # (K, N)
            heading_err = robot_theta - psi_path[:, :-1]
            # [-π, π] 정규화
            heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
            cost_heading = self.Q_heading * np.sum(heading_err ** 2, axis=1)

        return cost_contouring + cost_lag + cost_progress + cost_heading

    def get_contouring_info(self, trajectory: np.ndarray) -> Dict:
        """
        단일 궤적에 대한 상세 MPCC 정보.

        Args:
            trajectory: (N+1, nx) 단일 궤적

        Returns:
            dict with contouring_errors, lag_errors, progress, theta_star, etc.
        """
        if self._path is None:
            return {
                "contouring_errors": np.array([]),
                "lag_errors": np.array([]),
                "progress": 0.0,
                "theta_star": np.array([]),
                "mean_contouring_error": 0.0,
                "mean_lag_error": 0.0,
            }

        rx = trajectory[:, 0]
        ry = trajectory[:, 1]

        theta_star, psi_path, closest = self._path.project(rx, ry)

        dx = rx - closest[:, 0]
        dy = ry - closest[:, 1]

        cos_psi = np.cos(psi_path)
        sin_psi = np.sin(psi_path)
        e_c = -dx * sin_psi + dy * cos_psi
        e_l = dx * cos_psi + dy * sin_psi

        progress = theta_star[-1] - theta_star[0]

        return {
            "contouring_errors": e_c,
            "lag_errors": e_l,
            "progress": float(progress),
            "theta_star": theta_star,
            "mean_contouring_error": float(np.mean(np.abs(e_c))),
            "mean_lag_error": float(np.mean(np.abs(e_l))),
        }

    def __repr__(self) -> str:
        return (
            f"MPCCCost(Q_c={self.Q_c}, Q_l={self.Q_l}, "
            f"Q_θ={self.Q_theta}, Q_h={self.Q_heading})"
        )
