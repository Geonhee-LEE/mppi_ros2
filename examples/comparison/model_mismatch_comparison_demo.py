#!/usr/bin/env python3
"""
Model Mismatch Comparison Demo

시뮬레이터와 컨트롤러의 모델이 다를 때 (model mismatch) 학습 모델의 진가를 입증하는 데모.

두 가지 세계 모드를 지원:

  --world perturbed (기본값): 섭동된 기구학 세계 → 4-Way 비교
    1. Kinematic (Mismatched): 순수 기구학 — 미스매치로 큰 오차
    2. Neural (End-to-End): 학습된 NN — 전체 동역학 근사
    3. Residual (Hybrid): 기구학 + NN 보정 — 물리+데이터 융합
    4. Oracle (Perfect): 섭동 동역학 — 이론적 상한

  --world dynamic: DifferentialDriveDynamic(5D) 세계 → 6-Way 비교
    1. Kinematic (3D): 관성/마찰 모름
    2. Neural (3D): 데이터에서 전체 학습
    3. Residual (3D): 물리+데이터 융합
    4. Dynamic (5D): 구조 알지만 파라미터 틀림 (c_v=0.1 vs 실제 0.5)
    5. MAML (3D): FOMAML 메타 학습 + 실시간 few-shot 적응
    6. Oracle (5D): 정확한 파라미터

Usage:
    # Perturbed world (기본, 4-way)
    python model_mismatch_comparison_demo.py --all
    python model_mismatch_comparison_demo.py --live --trajectory circle --duration 20

    # Dynamic world (6-way)
    python model_mismatch_comparison_demo.py --all --world dynamic --trajectory circle --duration 20
    python model_mismatch_comparison_demo.py --live --world dynamic --trajectory circle --duration 20
"""

import numpy as np
import argparse
import sys
import os
import time
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.models.base_model import RobotModel
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
from mppi_controller.learning.maml_trainer import MAMLTrainer
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction,
    CompositeMPPICost,
    ControlEffortCost,
)
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
from mppi_controller.simulation.metrics import (
    compute_metrics,
    print_metrics,
    compare_metrics,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

# ==================== 경로 설정 ====================
DATA_DIR = "data/learned_models"
MODEL_DIR = "models/learned_models"
PLOT_DIR = "plots"
DATA_FILE = "mismatch_data.pkl"
NEURAL_MODEL_FILE = "mismatch_neural_model.pth"
RESIDUAL_MODEL_FILE = "mismatch_residual_model.pth"

# Dynamic world 전용 파일
DYNAMIC_DATA_FILE = "dynamic_mismatch_data.pkl"
DYNAMIC_NEURAL_MODEL_FILE = "dynamic_neural_model.pth"
DYNAMIC_RESIDUAL_MODEL_FILE = "dynamic_residual_model.pth"
DYNAMIC_MAML_MODEL_FILE = "dynamic_maml_meta_model.pth"

# ==================== 섭동 파라미터 ====================
# 강한 미스매치: 실제 로봇의 바닥 마찰 + 모터 바이어스 + 비대칭 마찰
FRICTION_FACTOR = 0.55        # 위치 변화량 * 0.55 (45% 속도 감쇠)
ACTUATOR_BIAS = np.array([0.12, -0.05])  # [v, omega] 바이어스
HEADING_FRICTION = 0.80       # 각속도 감쇠 (20% 회전 마찰)
PROCESS_NOISE_STD = np.array([0.005, 0.005, 0.002])  # 평가 시 작은 노이즈


# ==================== 섭동된 현실 세계 ====================

def perturbed_step(state, control, base_model, dt, add_noise=True):
    """
    섭동된 "현실 세계" step 함수.

    실제 로봇에서 발생하는 모델 미스매치를 시뮬레이션:
    1. 액추에이터 바이어스: [+0.12, -0.05] on [v, omega]
    2. 마찰: 위치 변화량 * 0.55 (45% 선속도 감쇠)
    3. 회전 마찰: 각속도 변화량 * 0.80 (20% 회전 감쇠)
    4. 프로세스 노이즈 (선택적)
    """
    # 1. 액추에이터 바이어스 적용
    biased_control = control + ACTUATOR_BIAS

    # 2. 기구학 적분 (RK4)
    nominal_next = base_model.step(state, biased_control, dt)

    # 3. 마찰: 위치/각도 변화량 감쇠
    delta = nominal_next - state
    delta[:2] *= FRICTION_FACTOR   # x, y 변화량 감쇠
    delta[2] *= HEADING_FRICTION   # theta 변화량 감쇠
    next_state = state + delta

    # 4. 프로세스 노이즈
    if add_noise:
        noise = np.random.normal(0.0, PROCESS_NOISE_STD)
        next_state += noise

    # 5. 각도 정규화
    next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))

    return next_state


class PerturbedDiffDriveModel(RobotModel):
    """
    섭동된 Differential Drive 모델 (Oracle).

    MPPI 롤아웃에서 perturbed_step과 동일한 섭동 적용 (노이즈 제외).
    컨트롤러가 "현실 세계"의 동역학을 완벽히 알고 있는 이상적 상한.

    step()을 직접 오버라이드하여 perturbed_step()과 정확히 동일한 계산 수행.
    """

    def __init__(self, v_max=1.0, omega_max=1.0):
        self.base_model = DifferentialDriveKinematic(v_max=v_max, omega_max=omega_max)

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(self, state, control):
        """섭동 동역학 (배치 지원) — MPPI의 배치 rollout에서 사용."""
        # 액추에이터 바이어스
        if control.ndim == 1:
            biased_control = control + ACTUATOR_BIAS
        else:
            biased_control = control + ACTUATOR_BIAS[np.newaxis, :]

        # 기구학 state_dot
        state_dot = self.base_model.forward_dynamics(state, biased_control)

        # 마찰: state_dot 감쇠
        result = state_dot.copy()
        result[..., :2] *= FRICTION_FACTOR
        result[..., 2] *= HEADING_FRICTION

        return result

    def step(self, state, control, dt):
        """
        perturbed_step과 동일한 적분 (노이즈 제외, 배치 지원).

        perturbed_step은 RK4 적분 후 delta에 마찰을 곱하므로,
        Oracle도 동일한 순서를 따라야 한다.
        """
        # 액추에이터 바이어스
        if control.ndim == 1:
            biased_control = control + ACTUATOR_BIAS
        else:
            biased_control = control + ACTUATOR_BIAS[np.newaxis, :]

        # 기구학 RK4 적분 (base_model의 step)
        nominal_next = self.base_model.step(state, biased_control, dt)

        # 마찰: delta에 적용
        delta = nominal_next - state
        delta[..., :2] *= FRICTION_FACTOR
        delta[..., 2] *= HEADING_FRICTION

        return state + delta

    def get_control_bounds(self):
        return self.base_model.get_control_bounds()

    def state_to_dict(self, state):
        return self.base_model.state_to_dict(state)

    def normalize_state(self, state):
        return self.base_model.normalize_state(state)


# ==================== Dynamic World (5D 현실 세계) ====================

# DifferentialDriveDynamic의 파라미터: [x,y,θ,v,ω], 제어=[a,α]
# 기구학 컨트롤러는 [v_cmd, ω_cmd]를 출력하므로 PD 변환 필요.
DYNAMIC_WORLD_PARAMS = {
    "c_v": 0.5,          # 선형 마찰 계수
    "c_omega": 0.3,      # 각속도 마찰 계수
    "k_v": 5.0,          # PD 선형 속도 게인
    "k_omega": 5.0,      # PD 각속도 게인
    "process_noise_std": np.array([0.005, 0.005, 0.002, 0.01, 0.005]),
}


class DynamicWorld:
    """
    5D DifferentialDriveDynamic을 "현실 세계"로 감싸는 래퍼.

    velocity command [v_cmd, ω_cmd] → PD control → [a, α] → 5D step.
    외부에는 3D observation [x, y, θ]만 반환.
    """

    def __init__(self, c_v=0.5, c_omega=0.3, k_v=5.0, k_omega=5.0,
                 process_noise_std=np.array([0.005, 0.005, 0.002, 0.01, 0.005])):
        self.dynamic_model = DifferentialDriveDynamic(
            c_v=c_v, c_omega=c_omega,
            a_max=5.0, alpha_max=5.0,    # PD 출력 범위 충분히 넓게
            v_max=2.0, omega_max=2.0,
        )
        self.k_v = k_v
        self.k_omega = k_omega
        self.process_noise_std = process_noise_std
        self.state_5d = np.zeros(5)  # [x, y, θ, v, ω]

    def reset(self, state_3d):
        """3D 상태로 리셋 → 5D [x, y, θ, 0, 0]."""
        self.state_5d = np.array([state_3d[0], state_3d[1], state_3d[2], 0.0, 0.0])
        return self.get_observation()

    def step(self, velocity_cmd, dt, add_noise=True):
        """
        velocity_cmd = [v_cmd, ω_cmd] → PD → [a, α] → 5D step → 3D obs.
        """
        v_cmd, omega_cmd = velocity_cmd[0], velocity_cmd[1]
        v_cur, omega_cur = self.state_5d[3], self.state_5d[4]

        # PD control: 가속도 = 게인 * (목표 - 현재)
        a = self.k_v * (v_cmd - v_cur)
        alpha = self.k_omega * (omega_cmd - omega_cur)

        accel_cmd = np.array([a, alpha])
        next_state = self.dynamic_model.step(self.state_5d, accel_cmd, dt)

        # 프로세스 노이즈
        if add_noise:
            noise = np.random.normal(0.0, self.process_noise_std)
            next_state += noise

        # 각도 정규화
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))

        self.state_5d = next_state
        return self.get_observation()

    def get_observation(self):
        """3D observation [x, y, θ]."""
        return self.state_5d[:3].copy()

    def get_full_state(self):
        """5D full state [x, y, θ, v, ω]."""
        return self.state_5d.copy()


class DynamicKinematicAdapter(RobotModel):
    """
    5D MPPI 내부 모델: [x,y,θ,v,ω] + velocity command [v_cmd, ω_cmd].

    PD + friction → state_dot 계산.
    MPPI가 5D state를 롤아웃하면서 관성/마찰 동역학을 인지.
    """

    def __init__(self, c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0):
        self._c_v = c_v
        self._c_omega = c_omega
        self._k_v = k_v
        self._k_omega = k_omega

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(self, state, control):
        """
        state: (..., 5) = [x, y, θ, v, ω]
        control: (..., 2) = [v_cmd, ω_cmd]
        return: state_dot (..., 5)
        """
        theta = state[..., 2]
        v = state[..., 3]
        omega = state[..., 4]

        v_cmd = control[..., 0]
        omega_cmd = control[..., 1]

        # PD + friction
        a = self._k_v * (v_cmd - v) - self._c_v * v
        alpha = self._k_omega * (omega_cmd - omega) - self._c_omega * omega

        # state_dot = [v*cos(θ), v*sin(θ), ω, a, α]
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        state_dot = np.zeros_like(state)
        state_dot[..., 0] = dx
        state_dot[..., 1] = dy
        state_dot[..., 2] = dtheta
        state_dot[..., 3] = a
        state_dot[..., 4] = alpha

        return state_dot

    def get_control_bounds(self):
        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])
        return lower, upper

    def state_to_dict(self, state):
        return {"x": state[0], "y": state[1], "theta": state[2],
                "v": state[3], "omega": state[4]}

    def normalize_state(self, state):
        result = state.copy()
        result[..., 2] = np.arctan2(np.sin(state[..., 2]), np.cos(state[..., 2]))
        return result


# ==================== Angle-Aware 비용 함수 ====================
# circle_trajectory의 heading이 [-π, π]를 넘어 무한히 증가하므로,
# MPPI rollout (normalized heading)과의 차이에서 2π 오차가 발생.
# heading error를 arctan2로 래핑하여 올바른 비용 계산.

class AngleAwareTrackingCost(CostFunction):
    """StateTrackingCost + heading angle wrapping."""

    def __init__(self, Q, angle_indices=(2,)):
        self.Q = Q
        self.is_diagonal = Q.ndim == 1
        self.angle_indices = angle_indices

    def compute_cost(self, trajectories, controls, reference_trajectory):
        errors = trajectories[:, :-1, :] - reference_trajectory[:-1, :]
        for idx in self.angle_indices:
            errors[:, :, idx] = np.arctan2(
                np.sin(errors[:, :, idx]), np.cos(errors[:, :, idx])
            )
        if self.is_diagonal:
            return np.sum(errors**2 * self.Q, axis=(1, 2))
        weighted = np.einsum("ktn,nm->ktm", errors, self.Q)
        return np.sum(weighted * errors, axis=(1, 2))


class AngleAwareTerminalCost(CostFunction):
    """TerminalCost + heading angle wrapping."""

    def __init__(self, Qf, angle_indices=(2,)):
        self.Qf = Qf
        self.is_diagonal = Qf.ndim == 1
        self.angle_indices = angle_indices

    def compute_cost(self, trajectories, controls, reference_trajectory):
        errors = trajectories[:, -1, :] - reference_trajectory[-1, :]
        for idx in self.angle_indices:
            errors[:, idx] = np.arctan2(
                np.sin(errors[:, idx]), np.cos(errors[:, idx])
            )
        if self.is_diagonal:
            return np.sum(errors**2 * self.Qf, axis=1)
        weighted = errors @ self.Qf
        return np.sum(weighted * errors, axis=1)


# ==================== MPPI 파라미터 ====================

def create_mppi_params():
    return MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )


def create_angle_aware_cost(params):
    """Angle-aware 비용 함수 생성 (heading wrapping 적용)."""
    return CompositeMPPICost([
        AngleAwareTrackingCost(params.Q),
        AngleAwareTerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ])


# ==================== 5D MPPI 파라미터 & 유틸리티 ====================

def create_5d_mppi_params():
    """5D state [x,y,θ,v,ω] 용 MPPI 파라미터."""
    return MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),
    )


def create_5d_angle_aware_cost(params):
    """5D용 Angle-aware 비용 함수 (θ=index 2)."""
    return CompositeMPPICost([
        AngleAwareTrackingCost(params.Q, angle_indices=(2,)),
        AngleAwareTerminalCost(params.Qf, angle_indices=(2,)),
        ControlEffortCost(params.R),
    ])


def make_5d_reference(ref_3d):
    """
    3D reference (N+1, 3) → 5D reference (N+1, 5).

    v_ref: 연속 위치 차이로 추정, ω_ref: heading 차이로 추정.
    """
    N_plus_1 = ref_3d.shape[0]
    ref_5d = np.zeros((N_plus_1, 5))
    ref_5d[:, :3] = ref_3d

    if N_plus_1 > 1:
        # v_ref: 위치 변화량 / dt
        dx = np.diff(ref_3d[:, 0])
        dy = np.diff(ref_3d[:, 1])
        v_ref = np.sqrt(dx**2 + dy**2) / 0.05  # dt=0.05
        ref_5d[:-1, 3] = v_ref
        ref_5d[-1, 3] = v_ref[-1] if len(v_ref) > 0 else 0.0

        # ω_ref: heading 변화량 / dt
        dtheta = np.diff(ref_3d[:, 2])
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        omega_ref = dtheta / 0.05
        ref_5d[:-1, 4] = omega_ref
        ref_5d[-1, 4] = omega_ref[-1] if len(omega_ref) > 0 else 0.0

    return ref_5d


# ==================== 커스텀 시뮬레이션 루프 ====================

def run_with_perturbed_world(controller, base_model, params, trajectory_fn, duration, seed):
    """
    섭동된 세계에서 컨트롤러를 실행하는 커스텀 시뮬 루프.

    Simulator 클래스 대신 직접 루프를 돌려서 perturbed_step() 사용.
    """
    np.random.seed(seed)
    controller.reset()

    state = trajectory_fn(0.0).copy()
    t = 0.0
    dt = params.dt
    num_steps = int(duration / dt)

    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "solve_time": [],
    }

    for _ in range(num_steps):
        ref_traj = generate_reference_trajectory(trajectory_fn, t, params.N, dt)

        t_start = time.time()
        control, info = controller.compute_control(state, ref_traj)
        solve_time = time.time() - t_start

        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_traj[0].copy())
        history["solve_time"].append(solve_time)

        state = perturbed_step(state, control, base_model, dt, add_noise=True)
        t += dt

    for key in history:
        history[key] = np.array(history[key])

    return history


def run_with_dynamic_world(controller, world, params, trajectory_fn, duration, seed, is_5d=False):
    """
    DynamicWorld에서 컨트롤러를 실행하는 커스텀 시뮬 루프.

    3D 컨트롤러(Kinematic/Neural/Residual): 3D state, 3D ref.
    5D 컨트롤러(Dynamic/Oracle): 5D state (world.get_full_state()), 5D ref.
    메트릭은 3D(x,y,θ) 기준으로 통일.
    """
    np.random.seed(seed)
    controller.reset()

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = np.array([*state_3d, 0.0, 0.0]) if is_5d else state_3d.copy()

    t = 0.0
    dt = params.dt
    num_steps = int(duration / dt)

    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "solve_time": [],
    }

    for _ in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params.N, dt)
        ref = make_5d_reference(ref_3d) if is_5d else ref_3d

        t_start = time.time()
        control, info = controller.compute_control(state, ref)
        solve_time = time.time() - t_start

        # 메트릭 기록 (항상 3D)
        history["time"].append(t)
        history["state"].append(state[:3].copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        # DynamicWorld step (velocity command → PD → 5D → 3D obs)
        obs_3d = world.step(control, dt, add_noise=True)

        if is_5d:
            state = world.get_full_state()
        else:
            state = obs_3d.copy()

        t += dt

    for key in history:
        history[key] = np.array(history[key])

    return history


# ==================== Stage 1: 데이터 수집 ====================

def collect_data(args):
    """섭동된/동적 세계에서 학습 데이터 수집."""
    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")

    print("\n" + "=" * 80)
    title = "Stage 1: Data Collection (Dynamic World)" if use_dynamic else "Stage 1: Data Collection (Perturbed World)"
    print(title.center(80))
    print("=" * 80)

    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = create_mppi_params()
    controller = MPPIController(base_model, params)

    collector = DataCollector(state_dim=3, control_dim=2, save_dir=DATA_DIR)

    # Dynamic world 인스턴스 (필요시)
    if use_dynamic:
        dyn_world = DynamicWorld(**DYNAMIC_WORLD_PARAMS)

    # 데이터 수집: 노이즈 OFF로 깨끗한 데이터 확보
    trajectories = ["circle", "figure8", "sine", "straight"]
    episodes_per_traj = 5
    episode_duration = 20.0  # seconds

    for traj_type in trajectories:
        trajectory_fn = create_trajectory_function(traj_type)

        for ep in range(episodes_per_traj):
            print(f"  Collecting: {traj_type} episode {ep+1}/{episodes_per_traj}")
            np.random.seed(ep * 100 + hash(traj_type) % 1000)

            state = trajectory_fn(0.0).copy()
            controller.reset()
            if use_dynamic:
                dyn_world.reset(state)
            t = 0.0
            dt = params.dt
            num_steps = int(episode_duration / dt)

            for _ in range(num_steps):
                ref_traj = generate_reference_trajectory(trajectory_fn, t, params.N, dt)
                control, _ = controller.compute_control(state, ref_traj)

                if use_dynamic:
                    obs_3d = dyn_world.step(control, dt, add_noise=False)
                    next_state = obs_3d
                else:
                    next_state = perturbed_step(state, control, base_model, dt, add_noise=False)

                collector.add_sample(state, control, next_state, dt)

                state = next_state
                t += dt

            collector.end_episode()

    # 랜덤 탐색 에피소드 추가 (다양한 제어 입력 커버)
    print("  Collecting: random exploration episodes")
    for ep in range(10):
        np.random.seed(9000 + ep)
        state = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(-np.pi, np.pi),
        ])
        if use_dynamic:
            dyn_world.reset(state)
        dt = params.dt
        for _ in range(200):
            control = np.array([
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
            ])
            if use_dynamic:
                obs_3d = dyn_world.step(control, dt, add_noise=False)
                next_state = obs_3d
            else:
                next_state = perturbed_step(state, control, base_model, dt, add_noise=False)
            collector.add_sample(state, control, next_state, dt)
            state = next_state
        collector.end_episode()

    # 통계 출력
    stats = collector.get_statistics()
    print(f"\n  Collected {stats['num_samples']} samples across {stats['num_episodes']} episodes")
    print(f"  State mean: {stats['state_mean']}")
    print(f"  State std:  {stats['state_std']}")

    data_file = DYNAMIC_DATA_FILE if use_dynamic else DATA_FILE
    collector.save(data_file)
    print(f"  Saved to: {os.path.join(DATA_DIR, data_file)}")


# ==================== Stage 2: 모델 학습 ====================

def train_models(args):
    """Neural 및 Residual 모델 학습."""
    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")

    print("\n" + "=" * 80)
    print("Stage 2: Model Training".center(80))
    print("=" * 80)

    # 데이터/모델 파일 선택
    data_file = DYNAMIC_DATA_FILE if use_dynamic else DATA_FILE
    neural_file = DYNAMIC_NEURAL_MODEL_FILE if use_dynamic else NEURAL_MODEL_FILE
    residual_file = DYNAMIC_RESIDUAL_MODEL_FILE if use_dynamic else RESIDUAL_MODEL_FILE

    # 데이터 로드
    data_path = os.path.join(DATA_DIR, data_file)
    if not os.path.exists(data_path):
        print(f"  Error: Data file not found: {data_path}")
        print("  Run with --collect-data first.")
        return

    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)

    print(f"  Loaded {len(raw_data['states'])} samples")

    # 각도 래핑으로 인한 state_dot 보정
    # (next_theta - theta)/dt 에서 angle wrapping으로 +-2pi 스파이크 발생 방지
    states = raw_data["states"]
    next_states = raw_data["next_states"]
    dt_arr = raw_data["dt"]
    corrected_dots = (next_states - states) / dt_arr[:, np.newaxis]
    # theta 차이만 angle-aware로 재계산
    theta_diff = next_states[:, 2] - states[:, 2]
    theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
    corrected_dots[:, 2] = theta_diff / dt_arr
    raw_data["state_dots"] = corrected_dots

    # ---- Neural Model (End-to-End) ----
    print("\n  Training Neural Model (End-to-End)...")
    neural_dataset = DynamicsDataset(raw_data, train_ratio=0.8, normalize=True)
    train_inputs, train_targets = neural_dataset.get_train_data()
    val_inputs, val_targets = neural_dataset.get_val_data()
    norm_stats = neural_dataset.get_normalization_stats()

    neural_trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[256, 256],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=1e-3,
        device="cpu",
        save_dir=MODEL_DIR,
    )

    neural_trainer.train(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        norm_stats=norm_stats,
        epochs=300,
        batch_size=128,
        early_stopping_patience=40,
        verbose=True,
    )
    neural_trainer.save_model(neural_file)

    # ---- Residual Model (보정량만 학습) ----
    print("\n  Training Residual Model (Correction Only)...")

    # residual_dot = actual_dot - kinematic_dot 계산
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    states = raw_data["states"]
    controls = raw_data["controls"]
    actual_dots = raw_data["state_dots"]

    # 기구학 예측 (벡터화)
    kinematic_dots = base_model.forward_dynamics(states, controls)

    # Residual = actual - kinematic
    residual_dots = actual_dots - kinematic_dots

    print(f"  Residual stats:")
    print(f"    mean: {np.mean(residual_dots, axis=0)}")
    print(f"    std:  {np.std(residual_dots, axis=0)}")

    # Residual 데이터셋 구성
    residual_data = {
        "states": states,
        "controls": controls,
        "state_dots": residual_dots,
        "next_states": raw_data["next_states"],
        "dt": raw_data["dt"],
    }

    residual_dataset = DynamicsDataset(residual_data, train_ratio=0.8, normalize=True)
    res_train_inputs, res_train_targets = residual_dataset.get_train_data()
    res_val_inputs, res_val_targets = residual_dataset.get_val_data()
    res_norm_stats = residual_dataset.get_normalization_stats()

    residual_trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[128, 128],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=1e-3,
        device="cpu",
        save_dir=MODEL_DIR,
    )

    residual_trainer.train(
        train_inputs=res_train_inputs,
        train_targets=res_train_targets,
        val_inputs=res_val_inputs,
        val_targets=res_val_targets,
        norm_stats=res_norm_stats,
        epochs=300,
        batch_size=128,
        early_stopping_patience=40,
        verbose=True,
    )
    residual_trainer.save_model(residual_file)

    print("\n  Models saved:")
    print(f"    Neural:   {os.path.join(MODEL_DIR, neural_file)}")
    print(f"    Residual: {os.path.join(MODEL_DIR, residual_file)}")


# ==================== Stage 2b: MAML 메타 학습 ====================

def meta_train_maml(args):
    """MAML 메타 학습 (다양한 DynamicWorld 설정으로 학습)."""
    print("\n" + "=" * 80)
    print("Stage 2b: MAML Meta-Training".center(80))
    print("=" * 80)

    trainer = MAMLTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[128, 128],
        inner_lr=0.005, inner_steps=10,
        meta_lr=5e-4,
        task_batch_size=8,
        support_size=100,
        query_size=100,
        device="cpu",
        save_dir=MODEL_DIR,
    )
    trainer.meta_train(n_iterations=1000, verbose=True)
    trainer.save_meta_model(DYNAMIC_MAML_MODEL_FILE)


def run_with_dynamic_world_maml(_, maml_model, world, params,
                                 trajectory_fn, duration, seed,
                                 warmup_steps=40, adapt_interval=80,
                                 buffer_size=200):
    """
    MAML 컨트롤러 전용 시뮬 루프 — ResidualDynamics + MAML 온라인 적응.

    Phase 1 (warm-up, ~2초): 기구학 모델로 제어 + 데이터 수집
    Phase 2 (adapted): ResidualDynamics(kinematic + MAML residual)로 전환,
                        주기적으로 MAML residual을 메타에서 재적응

    핵심: 기구학이 base dynamics 제공 → 안정성 보장,
          MAML은 잔차(마찰/관성 보정)만 학습 → 빠른 적응.
    """
    from collections import deque

    np.random.seed(seed)

    # 기구학 모델 (base)
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    cost_fn = create_angle_aware_cost(params)

    # Phase 1: 기구학 컨트롤러
    kin_controller = MPPIController(base_model, params, cost_function=cost_fn)

    # Phase 2: ResidualDynamics(kinematic + MAML)
    residual_model = ResidualDynamics(
        base_model=base_model,
        learned_model=maml_model,
        use_residual=True,
    )
    maml_residual_controller = MPPIController(residual_model, params, cost_function=cost_fn)

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = state_3d.copy()

    t = 0.0
    dt = params.dt
    num_steps = int(duration / dt)

    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "solve_time": [],
    }

    buffer_states = deque(maxlen=buffer_size)
    buffer_controls = deque(maxlen=buffer_size)
    buffer_next = deque(maxlen=buffer_size)
    adapted = False

    for step_i in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params.N, dt)

        t_start = time.time()
        if not adapted:
            control, info = kin_controller.compute_control(state, ref_3d)
        else:
            control, info = maml_residual_controller.compute_control(state, ref_3d)
        solve_time = time.time() - t_start

        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        obs_3d = world.step(control, dt, add_noise=True)

        buffer_states.append(state.copy())
        buffer_controls.append(control.copy())
        buffer_next.append(obs_3d.copy())

        # Phase 1→2 전환: 잔차 학습
        if not adapted and step_i >= warmup_steps:
            buf_s = np.array(buffer_states)
            buf_c = np.array(buffer_controls)
            buf_n = np.array(buffer_next)

            # 잔차 target: actual_next - kinematic_next
            kin_dots = np.array([
                base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)
            ])
            kin_next = buf_s + kin_dots * dt
            # MAML target: (residual_next - state) / dt = residual_dot
            # residual_next = actual_next - kin_dot * dt  (state cancels)
            # We need target state such that (target - state)/dt = residual_dot
            residual_target = buf_s + (buf_n - kin_next)

            maml_model.adapt(buf_s, buf_c, residual_target, dt, restore=True)
            maml_residual_controller.reset()
            adapted = True

        # Phase 2: 주기적 잔차 재적응
        elif adapted and step_i % adapt_interval == 0:
            buf_s = np.array(buffer_states)
            buf_c = np.array(buffer_controls)
            buf_n = np.array(buffer_next)
            kin_dots = np.array([
                base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)
            ])
            kin_next = buf_s + kin_dots * dt
            residual_target = buf_s + (buf_n - kin_next)
            maml_model.adapt(buf_s, buf_c, residual_target, dt, restore=True)

        state = obs_3d.copy()
        t += dt

    for key in history:
        history[key] = np.array(history[key])

    return history


# ==================== Stage 3: 평가 ====================

def evaluate(args):
    """4-Way (perturbed) 또는 5-Way (dynamic) 비교 평가."""
    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")

    n_controllers = 6 if use_dynamic else 4
    print("\n" + "=" * 80)
    print(f"Stage 3: Evaluation ({n_controllers}-Way Comparison, {world_type})".center(80))
    print("=" * 80)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Seed:       {args.seed}")
    print(f"  World:      {world_type}")
    print("=" * 80)

    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params_3d = create_mppi_params()
    trajectory_fn = create_trajectory_function(args.trajectory)

    # 모델 파일 존재 확인
    neural_file = DYNAMIC_NEURAL_MODEL_FILE if use_dynamic else NEURAL_MODEL_FILE
    residual_file = DYNAMIC_RESIDUAL_MODEL_FILE if use_dynamic else RESIDUAL_MODEL_FILE
    neural_path = os.path.join(MODEL_DIR, neural_file)
    residual_path = os.path.join(MODEL_DIR, residual_file)

    if not os.path.exists(neural_path) or not os.path.exists(residual_path):
        print(f"\n  Error: Model files not found.")
        print(f"    Neural:   {neural_path} {'(found)' if os.path.exists(neural_path) else '(MISSING)'}")
        print(f"    Residual: {residual_path} {'(found)' if os.path.exists(residual_path) else '(MISSING)'}")
        print("  Run with --collect-data --train first.")
        return

    cost_fn_3d = create_angle_aware_cost(params_3d)

    all_histories = {}
    all_metrics = {}

    if use_dynamic:
        # ---- Dynamic World: 5-Way 비교 ----
        params_5d = create_5d_mppi_params()
        cost_fn_5d = create_5d_angle_aware_cost(params_5d)

        # 1. Kinematic (3D, 관성/마찰 모름)
        print(f"\n  [1/{n_controllers}] Kinematic (Mismatched, 3D)...")
        kin_controller = MPPIController(base_model, params_3d, cost_function=cost_fn_3d)
        world_kin = DynamicWorld(**DYNAMIC_WORLD_PARAMS)
        all_histories["Kinematic"] = run_with_dynamic_world(
            kin_controller, world_kin, params_3d, trajectory_fn, args.duration, args.seed, is_5d=False
        )
        all_metrics["Kinematic"] = compute_metrics(all_histories["Kinematic"])
        print(f"        RMSE: {all_metrics['Kinematic']['position_rmse']:.4f}m")

        # 2. Neural (3D, 학습)
        print(f"  [2/{n_controllers}] Neural (End-to-End, 3D)...")
        neural_model = NeuralDynamics(state_dim=3, control_dim=2, model_path=neural_path)
        neural_controller = MPPIController(neural_model, params_3d, cost_function=cost_fn_3d)
        world_neural = DynamicWorld(**DYNAMIC_WORLD_PARAMS)
        all_histories["Neural"] = run_with_dynamic_world(
            neural_controller, world_neural, params_3d, trajectory_fn, args.duration, args.seed, is_5d=False
        )
        all_metrics["Neural"] = compute_metrics(all_histories["Neural"])
        print(f"        RMSE: {all_metrics['Neural']['position_rmse']:.4f}m")

        # 3. Residual (3D, 물리+NN)
        print(f"  [3/{n_controllers}] Residual (Hybrid, 3D)...")
        residual_nn = NeuralDynamics(state_dim=3, control_dim=2, model_path=residual_path)
        residual_model = ResidualDynamics(
            base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
            learned_model=residual_nn,
        )
        residual_controller = MPPIController(residual_model, params_3d, cost_function=cost_fn_3d)
        world_residual = DynamicWorld(**DYNAMIC_WORLD_PARAMS)
        all_histories["Residual"] = run_with_dynamic_world(
            residual_controller, world_residual, params_3d, trajectory_fn, args.duration, args.seed, is_5d=False
        )
        all_metrics["Residual"] = compute_metrics(all_histories["Residual"])
        print(f"        RMSE: {all_metrics['Residual']['position_rmse']:.4f}m")

        # 4. Dynamic (5D, 구조 알지만 파라미터 틀림)
        print(f"  [4/{n_controllers}] Dynamic (5D, mismatched params: c_v=0.1)...")
        dynamic_model = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
        dynamic_controller = MPPIController(dynamic_model, params_5d, cost_function=cost_fn_5d)
        world_dynamic = DynamicWorld(**DYNAMIC_WORLD_PARAMS)
        all_histories["Dynamic"] = run_with_dynamic_world(
            dynamic_controller, world_dynamic, params_5d, trajectory_fn, args.duration, args.seed, is_5d=True
        )
        all_metrics["Dynamic"] = compute_metrics(all_histories["Dynamic"])
        print(f"        RMSE: {all_metrics['Dynamic']['position_rmse']:.4f}m")

        # 5. MAML (3D, Residual MAML 적응)
        maml_path = os.path.join(MODEL_DIR, DYNAMIC_MAML_MODEL_FILE)
        if os.path.exists(maml_path):
            print(f"  [5/{n_controllers}] MAML (Residual-Adaptive, 3D)...")
            maml_model = MAMLDynamics(
                state_dim=3, control_dim=2,
                model_path=maml_path, inner_lr=0.005, inner_steps=100,
            )
            maml_model.save_meta_weights()
            world_maml = DynamicWorld(**DYNAMIC_WORLD_PARAMS)
            all_histories["MAML"] = run_with_dynamic_world_maml(
                None, maml_model, world_maml, params_3d,
                trajectory_fn, args.duration, args.seed,
            )
            all_metrics["MAML"] = compute_metrics(all_histories["MAML"])
            print(f"        RMSE: {all_metrics['MAML']['position_rmse']:.4f}m")
        else:
            print(f"  [5/{n_controllers}] MAML — SKIPPED (no meta model, run --meta-train)")

        # 6. Oracle (5D, 정확한 파라미터)
        print(f"  [6/{n_controllers}] Oracle (5D, exact params: c_v=0.5)...")
        oracle_model = DynamicKinematicAdapter(
            c_v=DYNAMIC_WORLD_PARAMS["c_v"],
            c_omega=DYNAMIC_WORLD_PARAMS["c_omega"],
            k_v=DYNAMIC_WORLD_PARAMS["k_v"],
            k_omega=DYNAMIC_WORLD_PARAMS["k_omega"],
        )
        oracle_controller = MPPIController(oracle_model, params_5d, cost_function=cost_fn_5d)
        world_oracle = DynamicWorld(**DYNAMIC_WORLD_PARAMS)
        all_histories["Oracle"] = run_with_dynamic_world(
            oracle_controller, world_oracle, params_5d, trajectory_fn, args.duration, args.seed, is_5d=True
        )
        all_metrics["Oracle"] = compute_metrics(all_histories["Oracle"])
        print(f"        RMSE: {all_metrics['Oracle']['position_rmse']:.4f}m")

        label_order = ["Kinematic", "Neural", "Residual", "Dynamic"]
        if "MAML" in all_histories:
            label_order.append("MAML")
        label_order.append("Oracle")

    else:
        # ---- Perturbed World: 4-Way 비교 (기존 로직) ----

        # 1. Kinematic (Mismatched)
        print(f"\n  [1/{n_controllers}] Kinematic (Mismatched)...")
        kin_controller = MPPIController(base_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Kinematic"] = run_with_perturbed_world(
            kin_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Kinematic"] = compute_metrics(all_histories["Kinematic"])
        print(f"        RMSE: {all_metrics['Kinematic']['position_rmse']:.4f}m")

        # 2. Neural (End-to-End)
        print(f"  [2/{n_controllers}] Neural (End-to-End)...")
        neural_model = NeuralDynamics(state_dim=3, control_dim=2, model_path=neural_path)
        neural_controller = MPPIController(neural_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Neural"] = run_with_perturbed_world(
            neural_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Neural"] = compute_metrics(all_histories["Neural"])
        print(f"        RMSE: {all_metrics['Neural']['position_rmse']:.4f}m")

        # 3. Residual (Hybrid)
        print(f"  [3/{n_controllers}] Residual (Hybrid)...")
        residual_nn = NeuralDynamics(state_dim=3, control_dim=2, model_path=residual_path)
        residual_model = ResidualDynamics(
            base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
            learned_model=residual_nn,
        )
        residual_controller = MPPIController(residual_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Residual"] = run_with_perturbed_world(
            residual_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Residual"] = compute_metrics(all_histories["Residual"])
        print(f"        RMSE: {all_metrics['Residual']['position_rmse']:.4f}m")

        # 4. Oracle (Perfect)
        print(f"  [4/{n_controllers}] Oracle (Perfect)...")
        oracle_model = PerturbedDiffDriveModel(v_max=1.0, omega_max=1.0)
        oracle_controller = MPPIController(oracle_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Oracle"] = run_with_perturbed_world(
            oracle_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Oracle"] = compute_metrics(all_histories["Oracle"])
        print(f"        RMSE: {all_metrics['Oracle']['position_rmse']:.4f}m")

        label_order = ["Kinematic", "Neural", "Residual", "Oracle"]

    # ---- 비교 출력 ----
    for label in label_order:
        print_metrics(all_metrics[label], title=label)

    metrics_list = [all_metrics[l] for l in label_order]
    compare_metrics(metrics_list, label_order, title=f"Model Mismatch Comparison ({world_type})")

    # ---- 순서 검증 ----
    print("\n  RMSE ranking:")
    sorted_labels = sorted(label_order, key=lambda l: all_metrics[l]["position_rmse"])
    for i, label in enumerate(sorted_labels):
        rmse = all_metrics[label]["position_rmse"]
        marker = " (best)" if i == 0 else ""
        print(f"    {i+1}. {label}: {rmse:.4f}m{marker}")

    if use_dynamic:
        print("\n  Expected: Oracle < MAML ~ Dynamic < Kinematic < Residual << Neural")
    else:
        print("\n  Expected: Oracle < Residual <= Neural << Kinematic")

    # ---- 시각화 ----
    visualize_general(
        args, params_3d, label_order, all_histories, all_metrics, world_type
    )


# ==================== Stage 4: 시각화 ====================

COLORS = {
    "Kinematic": "#e74c3c",   # red
    "Neural": "#3498db",       # blue
    "Residual": "#2ecc71",     # green
    "Dynamic": "#e67e22",      # orange
    "MAML": "#00bcd4",         # cyan
    "Oracle": "#9b59b6",       # purple
    "Reference": "#7f8c8d",    # gray
}


def visualize_general(args, params, label_order, all_histories, all_metrics, world_type):
    """N-Way 비교 시각화 (4-Way perturbed / 5-Way dynamic)."""
    import matplotlib.pyplot as plt

    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    if world_type == "dynamic":
        subtitle = f"(Dynamic World: c_v={DYNAMIC_WORLD_PARAMS['c_v']}, c_omega={DYNAMIC_WORLD_PARAMS['c_omega']})"
    else:
        subtitle = f"(Perturbed: friction={FRICTION_FACTOR}, bias={ACTUATOR_BIAS})"

    fig.suptitle(
        f"Model Mismatch Comparison - {args.trajectory.capitalize()} {subtitle}",
        fontsize=16, fontweight="bold",
    )

    first_label = label_order[0]

    # 1. XY 궤적 비교
    ax = axes[0, 0]
    refs = all_histories[first_label]["reference"]
    ax.plot(refs[:, 0], refs[:, 1], "--", color=COLORS["Reference"],
            label="Reference", linewidth=2, alpha=0.7)
    for name in label_order:
        states = all_histories[name]["state"]
        ax.plot(states[:, 0], states[:, 1], "-", color=COLORS[name],
                label=name, linewidth=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 위치 오차 시계열
    ax = axes[0, 1]
    for name in label_order:
        hist = all_histories[name]
        states = hist["state"]
        ref = hist["reference"]
        errors = np.linalg.norm(states[:, :2] - ref[:, :2], axis=1)
        ax.plot(hist["time"], errors, "-", color=COLORS[name], label=name, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. RMSE 바 차트
    ax = axes[0, 2]
    rmses = [all_metrics[n]["position_rmse"] for n in label_order]
    bar_colors = [COLORS[n] for n in label_order]
    bars = ax.bar(label_order, rmses, color=bar_colors, alpha=0.8)
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f"{rmse:.4f}m", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    if len(label_order) >= 5:
        ax.tick_params(axis="x", labelsize=8)

    # 4. 제어 입력 비교 (v only)
    ax = axes[1, 0]
    for name in label_order:
        hist = all_histories[name]
        controls = hist["control"]
        ax.plot(hist["time"], controls[:, 0], "-", color=COLORS[name],
                label=f"{name} v", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("Control Input (Linear Velocity)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. 계산 시간 비교
    ax = axes[1, 1]
    for name in label_order:
        hist = all_histories[name]
        solve_times_ms = hist["solve_time"] * 1000
        ax.plot(hist["time"], solve_times_ms, "-", color=COLORS[name],
                label=name, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 요약 텍스트
    ax = axes[1, 2]
    ax.axis("off")

    n_way = len(label_order)
    lines = [
        f"  {n_way}-Way Model Mismatch Comparison ({world_type})",
        f"  {'=' * 48}",
        "",
    ]

    if world_type == "dynamic":
        lines += [
            f"  Dynamic World Parameters:",
            f"    c_v={DYNAMIC_WORLD_PARAMS['c_v']}, c_omega={DYNAMIC_WORLD_PARAMS['c_omega']}",
            f"    k_v={DYNAMIC_WORLD_PARAMS['k_v']}, k_omega={DYNAMIC_WORLD_PARAMS['k_omega']}",
            f"    noise={DYNAMIC_WORLD_PARAMS['process_noise_std']}",
        ]
    else:
        lines += [
            f"  Perturbation Parameters:",
            f"    Friction factor:    {FRICTION_FACTOR}",
            f"    Heading friction:   {HEADING_FRICTION}",
            f"    Actuator bias:      {ACTUATOR_BIAS}",
            f"    Process noise:      {PROCESS_NOISE_STD}",
        ]

    lines += ["", "  Results (Position RMSE):"]
    for label in label_order:
        rmse = all_metrics[label]["position_rmse"]
        solve_ms = all_metrics[label]["mean_solve_time"]
        lines.append(f"    {label:20s}: {rmse:.4f} m  ({solve_ms:.1f} ms)")

    lines += [
        "",
        f"  Config:",
        f"    Trajectory: {args.trajectory}",
        f"    Duration:   {args.duration}s",
        f"    Samples:    K={params.K}, N={params.N}",
    ]

    summary_text = "\n".join(lines)
    ax.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    suffix = f"_{world_type}" if world_type == "dynamic" else ""
    save_path = os.path.join(PLOT_DIR, f"model_mismatch_comparison_{args.trajectory}{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}")
    plt.close()


# ==================== Live 실시간 비교 ====================

def run_live_comparison(args):
    """
    N-Way 실시간 비교 애니메이션.

    FuncAnimation으로 프레임마다 컨트롤러들을 전진시키며
    XY 궤적, 위치 오차, 제어 입력, RMSE 바 차트를 실시간 갱신.

    --world perturbed: 4-Way (Kinematic/Neural/Residual/Oracle)
    --world dynamic:   6-Way (Kinematic/Neural/Residual/Dynamic/MAML/Oracle)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from collections import deque

    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")
    n_way = 6 if use_dynamic else 4

    print("\n" + "=" * 80)
    print(f"Live Comparison ({n_way}-Way, {world_type}, Real-Time)".center(80))
    print("=" * 80)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Seed:       {args.seed}")
    print(f"  World:      {world_type}")
    print("=" * 80)

    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params_3d = create_mppi_params()
    trajectory_fn = create_trajectory_function(args.trajectory)

    # 모델 파일 존재 확인 — 없으면 자동으로 데이터 수집 + 학습
    neural_file = DYNAMIC_NEURAL_MODEL_FILE if use_dynamic else NEURAL_MODEL_FILE
    residual_file = DYNAMIC_RESIDUAL_MODEL_FILE if use_dynamic else RESIDUAL_MODEL_FILE
    neural_path = os.path.join(MODEL_DIR, neural_file)
    residual_path = os.path.join(MODEL_DIR, residual_file)

    if not os.path.exists(neural_path) or not os.path.exists(residual_path):
        print(f"\n  Model files not found — auto-running collect-data + train...")
        collect_data(args)
        train_models(args)

    # 공통 모델 생성
    neural_model = NeuralDynamics(state_dim=3, control_dim=2, model_path=neural_path)
    residual_nn = NeuralDynamics(state_dim=3, control_dim=2, model_path=residual_path)
    residual_model = ResidualDynamics(
        base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
        learned_model=residual_nn,
    )
    cost_fn_3d = create_angle_aware_cost(params_3d)

    # 컨트롤러 + 메타데이터
    controllers = {}
    is_5d_map = {}
    params_map = {}

    controllers["Kinematic"] = MPPIController(base_model, params_3d, cost_function=cost_fn_3d)
    controllers["Neural"] = MPPIController(neural_model, params_3d, cost_function=cost_fn_3d)
    controllers["Residual"] = MPPIController(residual_model, params_3d, cost_function=cost_fn_3d)
    is_5d_map["Kinematic"] = False
    is_5d_map["Neural"] = False
    is_5d_map["Residual"] = False
    params_map["Kinematic"] = params_3d
    params_map["Neural"] = params_3d
    params_map["Residual"] = params_3d

    if use_dynamic:
        params_5d = create_5d_mppi_params()
        cost_fn_5d = create_5d_angle_aware_cost(params_5d)

        dynamic_model = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
        controllers["Dynamic"] = MPPIController(dynamic_model, params_5d, cost_function=cost_fn_5d)
        is_5d_map["Dynamic"] = True
        params_map["Dynamic"] = params_5d

        # MAML (3D, few-shot 적응)
        maml_path = os.path.join(MODEL_DIR, DYNAMIC_MAML_MODEL_FILE)
        maml_model_live = None
        if not os.path.exists(maml_path):
            print(f"\n  MAML meta model not found — auto-running meta-train...")
            meta_train_maml(args)
        if os.path.exists(maml_path):
            maml_model_live = MAMLDynamics(
                state_dim=3, control_dim=2,
                model_path=maml_path, inner_lr=0.01, inner_steps=5,
            )
            maml_model_live.save_meta_weights()
            controllers["MAML"] = MPPIController(maml_model_live, params_3d, cost_function=cost_fn_3d)
            is_5d_map["MAML"] = False
            params_map["MAML"] = params_3d

        oracle_model = DynamicKinematicAdapter(
            c_v=DYNAMIC_WORLD_PARAMS["c_v"],
            c_omega=DYNAMIC_WORLD_PARAMS["c_omega"],
            k_v=DYNAMIC_WORLD_PARAMS["k_v"],
            k_omega=DYNAMIC_WORLD_PARAMS["k_omega"],
        )
        controllers["Oracle"] = MPPIController(oracle_model, params_5d, cost_function=cost_fn_5d)
        is_5d_map["Oracle"] = True
        params_map["Oracle"] = params_5d
    else:
        oracle_model = PerturbedDiffDriveModel(v_max=1.0, omega_max=1.0)
        controllers["Oracle"] = MPPIController(oracle_model, params_3d, cost_function=cost_fn_3d)
        is_5d_map["Oracle"] = False
        params_map["Oracle"] = params_3d

    dt = params_3d.dt
    num_steps = int(args.duration / dt)
    names = list(controllers.keys())

    # 초기 상태 (모두 동일)
    init_state_3d = trajectory_fn(0.0)[:3].copy()
    np.random.seed(args.seed)

    # 각 컨트롤러별 상태/히스토리
    states = {}
    worlds = {}  # dynamic world 인스턴스 (dynamic 모드)
    for name in names:
        if is_5d_map[name]:
            states[name] = np.array([*init_state_3d, 0.0, 0.0])
        else:
            states[name] = init_state_3d.copy()
        if use_dynamic:
            worlds[name] = DynamicWorld(**DYNAMIC_WORLD_PARAMS)
            worlds[name].reset(init_state_3d)

    for ctrl in controllers.values():
        ctrl.reset()

    # MAML adaptation buffers
    maml_buffer_states = deque(maxlen=50)
    maml_buffer_controls = deque(maxlen=50)
    maml_buffer_next = deque(maxlen=50)

    data = {
        n: {"xy": [], "times": [], "errors": [], "controls_v": [], "solve_times": []}
        for n in names
    }
    noise_seeds = np.random.randint(0, 100000, size=num_steps)

    # ---- Figure 구성: 2x2 패널 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    if use_dynamic:
        subtitle = f"(Dynamic: c_v={DYNAMIC_WORLD_PARAMS['c_v']})"
    else:
        subtitle = f"(Perturbed: friction={FRICTION_FACTOR})"
    fig.suptitle(
        f"Model Mismatch Live - {args.trajectory.capitalize()} {subtitle}",
        fontsize=14, fontweight="bold",
    )

    ax_xy = axes[0, 0]
    ax_err = axes[0, 1]
    ax_ctrl = axes[1, 0]
    ax_bar = axes[1, 1]

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectory")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    ref_times = np.arange(0, args.duration, dt)
    ref_pts = np.array([trajectory_fn(t) for t in ref_times])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "--", color=COLORS["Reference"],
               label="Reference", linewidth=2, alpha=0.5)

    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Position Tracking Error")
    ax_err.grid(True, alpha=0.3)

    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.set_ylabel("Linear Velocity (m/s)")
    ax_ctrl.set_title("Control Input (v)")
    ax_ctrl.grid(True, alpha=0.3)

    ax_bar.set_ylabel("Position RMSE (m)")
    ax_bar.set_title("Running RMSE")
    ax_bar.grid(True, axis="y", alpha=0.3)

    lines_xy = {}
    dots_xy = {}
    lines_err = {}
    lines_ctrl = {}
    for name in names:
        c = COLORS[name]
        lines_xy[name], = ax_xy.plot([], [], color=c, linewidth=2, label=name)
        dots_xy[name], = ax_xy.plot([], [], "o", color=c, markersize=7)
        lines_err[name], = ax_err.plot([], [], color=c, linewidth=2, label=name)
        lines_ctrl[name], = ax_ctrl.plot([], [], color=c, linewidth=1.5, label=name, alpha=0.8)

    ax_xy.legend(fontsize=8)
    ax_err.legend(fontsize=8)
    ax_ctrl.legend(fontsize=8)

    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=11, family="monospace")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def init():
        for name in names:
            lines_xy[name].set_data([], [])
            dots_xy[name].set_data([], [])
            lines_err[name].set_data([], [])
            lines_ctrl[name].set_data([], [])
        return []

    def update(frame):
        if frame >= num_steps:
            return []

        t_current = frame * dt

        for name in names:
            p = params_map[name]
            ref_3d = generate_reference_trajectory(trajectory_fn, t_current, p.N, dt)
            ref = make_5d_reference(ref_3d) if is_5d_map[name] else ref_3d

            # MAML: 적응 버퍼 충분하면 제어 전에 adapt
            if name == "MAML" and maml_model_live is not None:
                if frame > 0 and frame % 10 == 0 and len(maml_buffer_states) >= 20:
                    maml_model_live.adapt(
                        np.array(maml_buffer_states),
                        np.array(maml_buffer_controls),
                        np.array(maml_buffer_next), dt
                    )

            t_start = time.time()
            control, info = controllers[name].compute_control(states[name], ref)
            solve_time = time.time() - t_start

            # 3D 위치로 오차 계산
            state_3d = states[name][:3]
            ref_pt = ref_3d[0, :2]
            error = float(np.linalg.norm(state_3d[:2] - ref_pt))

            data[name]["xy"].append(state_3d[:2].copy())
            data[name]["times"].append(t_current)
            data[name]["errors"].append(error)
            data[name]["controls_v"].append(float(control[0]))
            data[name]["solve_times"].append(solve_time * 1000)

            # Step
            np.random.seed(noise_seeds[frame])
            if use_dynamic:
                obs_3d = worlds[name].step(control, dt, add_noise=True)
                if is_5d_map[name]:
                    states[name] = worlds[name].get_full_state()
                else:
                    states[name] = obs_3d.copy()

                # MAML: 전이 데이터 버퍼에 기록
                if name == "MAML" and maml_model_live is not None:
                    maml_buffer_states.append(state_3d.copy())
                    maml_buffer_controls.append(control.copy())
                    maml_buffer_next.append(obs_3d.copy())
            else:
                next_state = perturbed_step(states[name], control, base_model, dt, add_noise=True)
                states[name] = next_state

        # 플롯 업데이트
        for name in names:
            xy = np.array(data[name]["xy"])
            times = np.array(data[name]["times"])
            lines_xy[name].set_data(xy[:, 0], xy[:, 1])
            dots_xy[name].set_data([xy[-1, 0]], [xy[-1, 1]])
            lines_err[name].set_data(times, data[name]["errors"])
            lines_ctrl[name].set_data(times, data[name]["controls_v"])

        all_pts = []
        for name in names:
            if data[name]["xy"]:
                all_pts.extend(data[name]["xy"])
        if all_pts:
            all_pts_arr = np.array(all_pts)
            margin = 1.5
            ax_xy.set_xlim(all_pts_arr[:, 0].min() - margin, all_pts_arr[:, 0].max() + margin)
            ax_xy.set_ylim(all_pts_arr[:, 1].min() - margin, all_pts_arr[:, 1].max() + margin)

        for ax in [ax_err, ax_ctrl]:
            ax.relim()
            ax.autoscale_view()

        if frame % 20 == 0 or frame == num_steps - 1:
            ax_bar.clear()
            ax_bar.set_ylabel("Position RMSE (m)")
            ax_bar.set_title("Running RMSE")
            ax_bar.grid(True, axis="y", alpha=0.3)
            rmses = []
            bar_names = []
            bar_colors = []
            for name in names:
                errs = data[name]["errors"]
                if errs:
                    rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
                    rmses.append(rmse)
                    bar_names.append(name)
                    bar_colors.append(COLORS[name])
            if rmses:
                bars = ax_bar.bar(bar_names, rmses, color=bar_colors, alpha=0.8)
                for bar, rmse in zip(bars, rmses):
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{rmse:.4f}",
                        ha="center", va="bottom", fontsize=9,
                    )

        time_text.set_text(f"t = {t_current:.1f}s / {args.duration:.0f}s")
        return []

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=num_steps, interval=1, blit=False, repeat=False,
    )
    plt.show()
    print("\nLive comparison finished.")


# ==================== main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Model Mismatch Comparison Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perturbed world (기본, 4-way)
  python model_mismatch_comparison_demo.py --all
  python model_mismatch_comparison_demo.py --live --trajectory circle --duration 20

  # Dynamic world (6-way: Kinematic/Neural/Residual/Dynamic/MAML/Oracle)
  python model_mismatch_comparison_demo.py --all --world dynamic --trajectory circle --duration 20
  python model_mismatch_comparison_demo.py --live --world dynamic --trajectory circle --duration 20
        """,
    )
    parser.add_argument("--collect-data", action="store_true", help="Stage 1: Collect training data")
    parser.add_argument("--train", action="store_true", help="Stage 2: Train neural/residual models")
    parser.add_argument("--evaluate", action="store_true", help="Stage 3: Run evaluation")
    parser.add_argument("--meta-train", action="store_true", help="Stage 2b: MAML meta-training (dynamic world only)")
    parser.add_argument("--live", action="store_true", help="Live real-time comparison visualization")
    parser.add_argument("--all", action="store_true", help="Run all stages (collect-data + train + evaluate)")
    parser.add_argument(
        "--world", type=str, default="perturbed",
        choices=["perturbed", "dynamic"],
        help="World type: perturbed (4-way) or dynamic (6-way with DifferentialDriveDynamic) (default: perturbed)",
    )
    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=["circle", "figure8", "sine", "straight"],
        help="Reference trajectory type (default: circle)",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Evaluation duration in seconds (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # --all이면 전체 파이프라인 실행
    if args.all:
        args.collect_data = True
        args.train = True
        args.evaluate = True
        if args.world == "dynamic":
            args.meta_train = True

    # 아무 옵션도 없으면 도움말 출력
    if not (args.collect_data or args.train or args.evaluate or args.live or args.meta_train):
        parser.print_help()
        return

    n_way = 6 if args.world == "dynamic" else 4
    print("\n" + "#" * 80)
    print("#" + f"Model Mismatch Comparison Demo ({n_way}-Way, {args.world})".center(78) + "#")
    print("#" * 80)
    if args.world == "dynamic":
        print(f"\n  Dynamic world: DifferentialDriveDynamic (5D, inertia+friction)")
        print(f"  6-way comparison: Kinematic / Neural / Residual / Dynamic / MAML / Oracle")
    else:
        print(f"\n  Perturbed world: kinematic + friction, bias, noise")
        print(f"  4-way comparison: Kinematic / Neural / Residual / Oracle")
    print()

    if args.collect_data:
        collect_data(args)

    if args.train:
        train_models(args)

    if args.meta_train:
        meta_train_maml(args)

    if args.evaluate:
        evaluate(args)

    if args.live:
        run_live_comparison(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
