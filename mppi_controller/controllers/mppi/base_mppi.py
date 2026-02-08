"""
Vanilla MPPI 컨트롤러

Model Predictive Path Integral (MPPI) 기본 구현.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction,
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi.sampling import GaussianSampler, NoiseSampler


class MPPIController:
    """
    Vanilla MPPI 컨트롤러

    Model Predictive Path Integral 기본 구현.
    정보 이론 기반 샘플링 최적 제어.

    알고리즘:
        1. 명목 제어 시퀀스 U 주위에서 K개 노이즈 샘플링
        2. 각 샘플 제어로 궤적 rollout
        3. 궤적 비용 계산
        4. Softmax 가중치 계산: w_k = exp(-cost_k / λ) / Z
        5. 가중 평균으로 제어 업데이트: U = Σ w_k (U + ε_k)

    Args:
        model: RobotModel 인스턴스 (또는 dynamics_fn)
        params: MPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: MPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        self.model = model
        self.params = params

        # 동역학 래퍼 (배치 rollout)
        self.dynamics_wrapper = BatchDynamicsWrapper(model, params.dt)

        # 비용 함수 설정
        if cost_function is None:
            # 기본 비용 함수: StateTracking + Terminal + ControlEffort
            self.cost_function = CompositeMPPICost(
                [
                    StateTrackingCost(params.Q),
                    TerminalCost(params.Qf),
                    ControlEffortCost(params.R),
                ]
            )
        else:
            self.cost_function = cost_function

        # 노이즈 샘플러 설정
        if noise_sampler is None:
            self.noise_sampler = GaussianSampler(params.sigma)
        else:
            self.noise_sampler = noise_sampler

        # 제어 제약
        control_bounds = params.get_control_bounds()
        if control_bounds is None:
            control_bounds = model.get_control_bounds()

        if control_bounds is not None:
            self.u_min, self.u_max = control_bounds
        else:
            self.u_min, self.u_max = None, None

        # 명목 제어 시퀀스 초기화 (N, nu)
        self.U = np.zeros((params.N, model.control_dim))

        # 메트릭 저장
        self.last_info = {}

        # GPU 가속 설정
        self._use_gpu = (params.device == "cuda") and self._check_gpu_available()
        if self._use_gpu:
            from mppi_controller.controllers.mppi.gpu import (
                TorchDynamicsWrapper,
                TorchCompositeCost,
                TorchGaussianSampler,
                get_torch_model,
            )
            import torch
            torch_model = get_torch_model(model, params.device)
            self._gpu_dynamics = TorchDynamicsWrapper(
                torch_model, params.dt, params.device
            )
            self._gpu_cost = TorchCompositeCost(
                params.Q, params.R, params.Qf, device=params.device
            )
            self._gpu_sampler = TorchGaussianSampler(params.sigma, params.device)
            self._torch = torch

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
                - sample_trajectories: (K, N+1, nx)
                - sample_weights: (K,)
                - best_trajectory: (N+1, nx)
                - temperature: float
                - ess: float (Effective Sample Size)
                - num_samples: int
        """
        if self._use_gpu:
            return self._compute_control_gpu(state, reference_trajectory)

        K = self.params.K
        N = self.params.N

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise  # 브로드캐스트

        # 제어 제약 클리핑 (safety)
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 샘플 궤적 rollout (K, N+1, nx)
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 비용 계산 (K,)
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. MPPI 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. 가중 평균으로 제어 업데이트
        # U_new = Σ w_k (U + ε_k) = U + Σ w_k ε_k
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)
        self.U = self.U + weighted_noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 7. 다음 스텝을 위한 시프트 (receding horizon)
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0  # 마지막 제어는 0으로

        # 8. 최적 제어 반환 (첫 번째 제어)
        optimal_control = self.U[0, :]

        # 9. 정보 저장
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": costs[best_idx],
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
        }
        self.last_info = info

        return optimal_control, info

    def _compute_control_gpu(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """GPU 가속 제어 계산 — 전체 파이프라인을 GPU에서 실행"""
        torch = self._torch
        K, N = self.params.K, self.params.N
        device = self.params.device

        # 1. GPU 노이즈 생성 (K, N, nu) — GPU에서 직접 생성
        noise_t = self._gpu_sampler.sample(self.U, K)

        # 2. GPU 제어 시퀀스
        U_t = torch.tensor(self.U, device=device, dtype=torch.float32)
        controls_t = U_t + noise_t

        if self.u_min is not None:
            u_min_t = torch.tensor(self.u_min, device=device, dtype=torch.float32)
            u_max_t = torch.tensor(self.u_max, device=device, dtype=torch.float32)
            controls_t = torch.clamp(controls_t, u_min_t, u_max_t)

        # 3. GPU rollout (K, N+1, nx)
        state_t = torch.tensor(state, device=device, dtype=torch.float32)
        state_t = state_t.unsqueeze(0).expand(K, -1)  # (K, nx)
        trajectories_t = self._gpu_dynamics.rollout(state_t, controls_t)

        # 4. GPU 비용 (K,)
        ref_t = torch.tensor(
            reference_trajectory, device=device, dtype=torch.float32
        )
        costs_t = self._gpu_cost.compute_cost(trajectories_t, controls_t, ref_t)

        # 5. GPU 가중치 (softmax)
        min_cost = torch.min(costs_t)
        exp_costs = torch.exp(-(costs_t - min_cost) / self.params.lambda_)
        weights_t = exp_costs / torch.sum(exp_costs)

        # 6. GPU 가중 평균 업데이트
        weighted_noise = torch.sum(
            weights_t[:, None, None] * noise_t, dim=0
        )  # (N, nu)
        U_new = U_t + weighted_noise

        if self.u_min is not None:
            U_new = torch.clamp(U_new, u_min_t, u_max_t)

        # 7. Receding horizon shift
        U_new = torch.roll(U_new, -1, dims=0)
        U_new[-1, :] = 0.0

        # 8. CPU로 변환하여 저장
        self.U = U_new.cpu().numpy()
        optimal_control = self.U[0, :]

        # 9. Info (numpy로 변환)
        costs_np = costs_t.detach().cpu().numpy()
        weights_np = weights_t.detach().cpu().numpy()
        trajectories_np = trajectories_t.detach().cpu().numpy()
        best_idx = int(torch.argmin(costs_t).item())

        ess = float(1.0 / (weights_np ** 2).sum())

        info = {
            "sample_trajectories": trajectories_np,
            "sample_weights": weights_np,
            "best_trajectory": trajectories_np[best_idx],
            "best_cost": float(costs_np[best_idx]),
            "mean_cost": float(costs_np.mean()),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
        }
        self.last_info = info

        return optimal_control, info

    @staticmethod
    def _check_gpu_available():
        """CUDA GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
        """
        MPPI 가중치 계산 (softmax)

        w_k = exp(-cost_k / λ) / Σ exp(-cost_k / λ)

        Args:
            costs: (K,) 각 샘플의 비용
            lambda_: 온도 파라미터

        Returns:
            weights: (K,) 정규화된 가중치
        """
        # 수치 안정성을 위한 log-space 연산
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / lambda_)
        weights = exp_costs / np.sum(exp_costs)

        return weights

    def _compute_ess(self, weights: np.ndarray) -> float:
        """
        Effective Sample Size (ESS) 계산

        ESS = 1 / Σ w_k²

        가중치가 균등하면 ESS ≈ K, 한 샘플에 집중하면 ESS ≈ 1.

        Args:
            weights: (K,) 정규화된 가중치

        Returns:
            ess: Effective Sample Size
        """
        return 1.0 / np.sum(weights**2)

    def reset(self):
        """명목 제어 시퀀스 초기화"""
        self.U = np.zeros((self.params.N, self.model.control_dim))

    def set_control_sequence(self, U: np.ndarray):
        """명목 제어 시퀀스 설정 (warm start)"""
        assert U.shape == (self.params.N, self.model.control_dim)
        self.U = U.copy()

    def __repr__(self) -> str:
        return (
            f"MPPIController("
            f"model={self.model.__class__.__name__}, "
            f"params={self.params})"
        )
