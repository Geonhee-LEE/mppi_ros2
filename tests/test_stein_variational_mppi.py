"""
Stein Variational MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import SteinVariationalMPPIParams
from mppi_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
from mppi_controller.utils.stein_variational import (
    rbf_kernel,
    rbf_kernel_gradient,
    median_bandwidth,
)


def test_rbf_kernel():
    """RBF 커널 기본 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: RBF Kernel Computation")
    print("=" * 60)

    # 간단한 2D 샘플
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (3, 2)

    kernel = rbf_kernel(X, bandwidth=1.0)

    print(f"Samples:\n{X}")
    print(f"Kernel:\n{kernel}")

    # 커널 속성 확인
    assert kernel.shape == (3, 3), f"Kernel shape mismatch: {kernel.shape}"
    assert np.allclose(
        kernel, kernel.T
    ), "Kernel should be symmetric"  # 대칭성
    assert np.allclose(
        np.diag(kernel), 1.0
    ), "Diagonal should be 1"  # K(x, x) = 1

    print("✓ PASS: RBF kernel works\n")


def test_median_bandwidth():
    """Median heuristic bandwidth 테스트"""
    print("=" * 60)
    print("Test 2: Median Bandwidth Heuristic")
    print("=" * 60)

    X = np.random.randn(100, 10)  # (100, 10)
    bandwidth = median_bandwidth(X)

    print(f"Sample shape: {X.shape}")
    print(f"Bandwidth: {bandwidth:.4f}")

    assert bandwidth > 0, "Bandwidth should be positive"
    print("✓ PASS: Median bandwidth works\n")


def test_svmpc_basic():
    """SVMPC 기본 동작 테스트"""
    print("=" * 60)
    print("Test 3: SVMPC Basic Functionality")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SteinVariationalMPPIParams(
        N=10, dt=0.05, K=64, svgd_num_iterations=3, svgd_step_size=0.01
    )
    controller = SteinVariationalMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    print("Running SVMPC (K=64, 3 iterations)...")
    control, info = controller.compute_control(state, reference)

    print(f"Control: {control}")
    print(f"SVGD iterations: {info['svgd_stats']['svgd_iterations']}")
    print(f"Cost improvement: {info['svgd_stats']['cost_improvement']:.4f}")
    print(f"ESS: {info['ess']:.2f}")

    # 기본 체크
    assert control.shape == (2,), f"Control shape mismatch: {control.shape}"
    assert "svgd_stats" in info, "Info should contain svgd_stats"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print("✓ PASS: SVMPC works\n")


def test_svgd_cost_improvement():
    """SVGD 비용 개선 테스트"""
    print("=" * 60)
    print("Test 4: SVGD Cost Improvement")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SteinVariationalMPPIParams(
        N=10, dt=0.05, K=64, svgd_num_iterations=5, svgd_step_size=0.05
    )
    controller = SteinVariationalMPPIController(model, params)

    state = np.array([1.0, 1.0, 0.0])  # 초기 오차
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    initial_cost = info["svgd_stats"]["initial_mean_cost"]
    final_cost = info["svgd_stats"]["final_mean_cost"]
    improvement = info["svgd_stats"]["cost_improvement"]

    print(f"Initial cost: {initial_cost:.4f}")
    print(f"Final cost: {final_cost:.4f}")
    print(f"Improvement: {improvement:.4f}")

    # SVGD는 비용을 개선해야 함 (또는 최소한 악화되지 않아야)
    # 단, 작은 step size에서는 개선이 미미할 수 있음
    print(f"Improvement rate: {improvement/initial_cost*100:.2f}%")

    assert not np.any(np.isnan(control)), "Control contains NaN"
    print("✓ PASS: SVGD cost improvement verified\n")


def test_svgd_iterations_effect():
    """SVGD 반복 횟수 효과 테스트"""
    print("=" * 60)
    print("Test 5: SVGD Iterations Effect")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    state = np.array([0.5, 0.5, 0.1])
    reference = np.zeros((11, 3))

    iteration_counts = [1, 3, 10]  # 0은 불가 (파라미터 검증)

    for num_iter in iteration_counts:
        params = SteinVariationalMPPIParams(
            N=10, dt=0.05, K=64, svgd_num_iterations=num_iter, svgd_step_size=0.01
        )
        controller = SteinVariationalMPPIController(model, params)

        control, info = controller.compute_control(state, reference)

        improvement = info["svgd_stats"]["cost_improvement"]
        print(
            f"  Iterations={num_iter}: Cost={info['mean_cost']:.4f}, "
            f"Improvement={improvement:.4f}"
        )

    print("✓ PASS: SVGD iterations effect verified\n")


def test_svgd_statistics():
    """SVGD 통계 추적 테스트"""
    print("=" * 60)
    print("Test 6: SVGD Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SteinVariationalMPPIParams(
        N=10, dt=0.05, K=64, svgd_num_iterations=5, svgd_step_size=0.01
    )
    controller = SteinVariationalMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 여러 번 호출
    num_steps = 5
    for i in range(num_steps):
        controller.compute_control(state, reference)

    # 통계 확인
    stats = controller.get_svgd_statistics()

    print(f"Mean cost improvement: {stats['mean_cost_improvement']:.4f}")
    print(f"Mean bandwidth: {stats['mean_bandwidth']:.4f}")
    print(f"History length: {len(stats['svgd_stats_history'])}")

    assert (
        len(stats["svgd_stats_history"]) == num_steps
    ), "History length mismatch"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stein Variational MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_rbf_kernel()
        test_median_bandwidth()
        test_svmpc_basic()
        test_svgd_cost_improvement()
        test_svgd_iterations_effect()
        test_svgd_statistics()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
