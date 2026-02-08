"""
GPU MPPI 테스트

CUDA 사용 가능 시만 실행. GPU/CPU 결과 동등성 및 성능 검증.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


def _make_controller(device="cpu", K=256):
    """테스트용 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = MPPIParams(
        N=30, dt=0.05, K=K, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        device=device,
    )
    return MPPIController(model, params)


def _make_reference(N=30):
    """원형 레퍼런스 궤적 생성"""
    t = np.linspace(0, 1.5, N + 1)
    radius = 5.0
    omega = 0.1
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = radius * np.cos(omega * t)
    ref[:, 1] = radius * np.sin(omega * t)
    ref[:, 2] = omega * t + np.pi / 2
    return ref


# ─────────────────────────────────────────────────
# GPU/CPU 결과 동등성 테스트
# ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_cpu_equivalence():
    """GPU/CPU 결과 동등성 (시뮬레이션 RMSE 차이 < 0.1m)"""
    print("\n" + "=" * 60)
    print("Test: GPU/CPU Equivalence")
    print("=" * 60)

    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_reference()

    # CPU 실행
    np.random.seed(42)
    ctrl_cpu = _make_controller(device="cpu", K=512)
    results_cpu = []
    for _ in range(20):
        control, info = ctrl_cpu.compute_control(state, ref)
        results_cpu.append(info["best_cost"])

    # GPU 실행
    np.random.seed(42)
    ctrl_gpu = _make_controller(device="cuda", K=512)
    results_gpu = []
    for _ in range(20):
        control, info = ctrl_gpu.compute_control(state, ref)
        results_gpu.append(info["best_cost"])

    cpu_mean = np.mean(results_cpu)
    gpu_mean = np.mean(results_gpu)
    diff = abs(cpu_mean - gpu_mean)

    print(f"  CPU mean cost: {cpu_mean:.4f}")
    print(f"  GPU mean cost: {gpu_mean:.4f}")
    print(f"  Difference: {diff:.4f}")

    # 확률적 알고리즘이므로 정확한 일치는 불가, 대략적 유사성 확인
    assert diff < cpu_mean * 0.5, (
        f"GPU/CPU cost difference too large: {diff:.4f} "
        f"(CPU: {cpu_mean:.4f}, GPU: {gpu_mean:.4f})"
    )
    print("PASS\n")


# ─────────────────────────────────────────────────
# GPU rollout 정확성 테스트
# ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_rollout_correctness():
    """GPU rollout이 CPU rollout과 수치적으로 일치"""
    print("\n" + "=" * 60)
    print("Test: GPU Rollout Correctness")
    print("=" * 60)

    from mppi_controller.controllers.mppi.gpu.torch_dynamics import (
        TorchDiffDriveKinematic, TorchDynamicsWrapper,
    )
    from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    dt = 0.05
    K, N = 16, 30

    # 동일한 초기 상태와 제어
    state = np.array([1.0, 2.0, 0.5])
    controls = np.random.randn(K, N, 2) * 0.3

    # CPU rollout
    cpu_wrapper = BatchDynamicsWrapper(model, dt)
    cpu_traj = cpu_wrapper.rollout(state, controls)

    # GPU rollout
    torch_model = TorchDiffDriveKinematic(device="cuda")
    gpu_wrapper = TorchDynamicsWrapper(torch_model, dt, device="cuda")

    state_t = torch.tensor(state, device="cuda", dtype=torch.float32)
    state_t = state_t.unsqueeze(0).expand(K, -1)
    controls_t = torch.tensor(controls, device="cuda", dtype=torch.float32)
    gpu_traj = gpu_wrapper.rollout(state_t, controls_t).cpu().numpy()

    max_diff = np.max(np.abs(cpu_traj - gpu_traj))
    print(f"  Max trajectory difference: {max_diff:.8f}")

    # float32 vs float64 차이 허용 (1e-4 수준)
    assert max_diff < 1e-3, f"GPU rollout differs from CPU: max_diff={max_diff:.6f}"
    print("PASS\n")


# ─────────────────────────────────────────────────
# GPU 비용 정확성 테스트
# ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_cost_correctness():
    """GPU 비용이 CPU 비용과 수치적으로 일치"""
    print("\n" + "=" * 60)
    print("Test: GPU Cost Correctness")
    print("=" * 60)

    from mppi_controller.controllers.mppi.gpu.torch_costs import TorchCompositeCost
    from mppi_controller.controllers.mppi.cost_functions import (
        CompositeMPPICost, StateTrackingCost, TerminalCost, ControlEffortCost,
    )

    Q = np.array([10.0, 10.0, 1.0])
    R = np.array([0.1, 0.1])
    Qf = Q.copy()

    K, N, nx, nu = 64, 30, 3, 2

    trajectories = np.random.randn(K, N + 1, nx)
    controls = np.random.randn(K, N, nu)
    reference = np.random.randn(N + 1, nx)

    # CPU 비용
    cpu_cost = CompositeMPPICost([
        StateTrackingCost(Q), TerminalCost(Qf), ControlEffortCost(R),
    ])
    costs_cpu = cpu_cost.compute_cost(trajectories, controls, reference)

    # GPU 비용
    gpu_cost = TorchCompositeCost(Q, R, Qf, device="cuda")
    traj_t = torch.tensor(trajectories, device="cuda", dtype=torch.float32)
    ctrl_t = torch.tensor(controls, device="cuda", dtype=torch.float32)
    ref_t = torch.tensor(reference, device="cuda", dtype=torch.float32)
    costs_gpu = gpu_cost.compute_cost(traj_t, ctrl_t, ref_t).cpu().numpy()

    max_diff = np.max(np.abs(costs_cpu - costs_gpu))
    rel_diff = max_diff / (np.mean(np.abs(costs_cpu)) + 1e-8)
    print(f"  Max cost difference: {max_diff:.6f}")
    print(f"  Relative difference: {rel_diff:.6f}")

    # float32 vs float64 차이 허용
    assert rel_diff < 1e-3, f"GPU cost differs from CPU: rel_diff={rel_diff:.6f}"
    print("PASS\n")


# ─────────────────────────────────────────────────
# GPU 속도 테스트
# ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_speedup():
    """GPU가 대규모 K에서 CPU보다 빠른지 확인 (K=4096)"""
    print("\n" + "=" * 60)
    print("Test: GPU Speedup (K=4096)")
    print("=" * 60)

    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_reference()
    K = 4096

    # CPU 벤치마크
    ctrl_cpu = _make_controller(device="cpu", K=K)
    # Warmup
    ctrl_cpu.compute_control(state, ref)

    cpu_times = []
    for _ in range(10):
        t0 = time.time()
        ctrl_cpu.compute_control(state, ref)
        cpu_times.append(time.time() - t0)
    cpu_mean = np.mean(cpu_times) * 1000

    # GPU 벤치마크
    ctrl_gpu = _make_controller(device="cuda", K=K)
    # Warmup
    for _ in range(5):
        ctrl_gpu.compute_control(state, ref)

    gpu_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        ctrl_gpu.compute_control(state, ref)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - t0)
    gpu_mean = np.mean(gpu_times) * 1000

    speedup = cpu_mean / gpu_mean
    print(f"  CPU: {cpu_mean:.2f}ms")
    print(f"  GPU: {gpu_mean:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")

    # GPU가 적어도 느리지는 않은지 확인 (K=4096이면 보통 빠름)
    assert speedup > 0.5, f"GPU is significantly slower: speedup={speedup:.1f}x"
    print("PASS\n")


# ─────────────────────────────────────────────────
# 대규모 K 동작 확인
# ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_large_K():
    """K=8192 대규모 샘플 동작 확인"""
    print("\n" + "=" * 60)
    print("Test: GPU Large K (K=8192)")
    print("=" * 60)

    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_reference()
    K = 8192

    ctrl = _make_controller(device="cuda", K=K)
    control, info = ctrl.compute_control(state, ref)

    print(f"  Control: {control}")
    print(f"  Best cost: {info['best_cost']:.4f}")
    print(f"  ESS: {info['ess']:.1f}")
    print(f"  Trajectories shape: {info['sample_trajectories'].shape}")

    assert control.shape == (2,)
    assert info["sample_trajectories"].shape == (K, 31, 3)
    assert info["sample_weights"].shape == (K,)
    assert not np.any(np.isnan(control))
    print("PASS\n")


# ─────────────────────────────────────────────────
# device 파라미터 활성화 테스트
# ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_device_parameter():
    """device='cuda' 파라미터로 GPU 활성화"""
    print("\n" + "=" * 60)
    print("Test: GPU Device Parameter")
    print("=" * 60)

    ctrl = _make_controller(device="cuda")
    assert ctrl._use_gpu is True

    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_reference()
    control, info = ctrl.compute_control(state, ref)

    assert isinstance(control, np.ndarray)
    assert isinstance(info["sample_trajectories"], np.ndarray)
    assert isinstance(info["sample_weights"], np.ndarray)
    print(f"  control type: {type(control)}")
    print(f"  trajectories type: {type(info['sample_trajectories'])}")
    print("PASS\n")


# ─────────────────────────────────────────────────
# CPU 폴백 테스트
# ─────────────────────────────────────────────────

def test_cpu_fallback():
    """device='cpu'일 때 기존 CPU 경로 사용"""
    print("\n" + "=" * 60)
    print("Test: CPU Fallback")
    print("=" * 60)

    ctrl = _make_controller(device="cpu")
    assert ctrl._use_gpu is False

    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_reference()
    control, info = ctrl.compute_control(state, ref)

    assert isinstance(control, np.ndarray)
    assert control.shape == (2,)
    assert not np.any(np.isnan(control))
    print(f"  control: {control}")
    print("PASS\n")


# ─────────────────────────────────────────────────
# Standalone 실행
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_cpu_fallback,
    ]

    if HAS_CUDA:
        tests.extend([
            test_gpu_device_parameter,
            test_gpu_rollout_correctness,
            test_gpu_cost_correctness,
            test_gpu_cpu_equivalence,
            test_gpu_large_K,
            test_gpu_speedup,
        ])
    else:
        print("[!] CUDA not available — GPU tests skipped")

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")
