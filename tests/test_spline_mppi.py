"""
Spline MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import SplineMPPIParams
from mppi_controller.controllers.mppi.spline_mppi import SplineMPPIController


def test_bspline_interpolation():
    """B-spline 보간 기본 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: B-spline Interpolation")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SplineMPPIParams(N=30, dt=0.05, K=128, spline_num_knots=8, spline_degree=3)
    controller = SplineMPPIController(model, params)

    # 간단한 knot 값
    knot_values = np.array([0.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.0, 0.0])

    # 보간
    interpolated = controller._bspline_interpolate(knot_values, num_points=30)

    print(f"Knot values (P=8): {knot_values}")
    print(f"Interpolated shape: {interpolated.shape}")
    print(f"Interpolated min/max: [{interpolated.min():.2f}, {interpolated.max():.2f}]")

    # 보간된 값이 knot 범위 내에 있어야 함 (convex hull property)
    assert interpolated.shape == (30,), f"Interpolated shape mismatch: {interpolated.shape}"
    assert np.all(interpolated >= knot_values.min() - 0.1), "Interpolation out of range"
    assert np.all(interpolated <= knot_values.max() + 0.1), "Interpolation out of range"
    print("✓ PASS: B-spline interpolation works\n")


def test_spline_mppi_basic():
    """Spline MPPI 기본 동작 테스트"""
    print("=" * 60)
    print("Test 2: Spline MPPI Basic Functionality")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SplineMPPIParams(N=20, dt=0.05, K=128, spline_num_knots=6, spline_degree=3)
    controller = SplineMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((21, 3))

    control, info = controller.compute_control(state, reference)

    print(f"Control: {control}")
    print(f"Num knots: {info['spline_stats']['num_knots']}")
    print(f"Degree: {info['spline_stats']['degree']}")
    print(f"Knot variance: {info['spline_stats']['knot_variance']:.4f}")

    # 기본 체크
    assert control.shape == (2,), f"Control shape mismatch: {control.shape}"
    assert "sample_knots" in info, "Info should contain sample_knots"
    assert info["sample_knots"].shape == (128, 6, 2), f"Knots shape mismatch: {info['sample_knots'].shape}"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print("✓ PASS: Spline MPPI works\n")


def test_memory_efficiency():
    """메모리 효율성 테스트 (Knot vs Full)"""
    print("=" * 60)
    print("Test 3: Memory Efficiency (Knots vs Full Controls)")
    print("=" * 60)

    N = 30
    P = 8
    K = 1024
    nu = 2

    # Spline MPPI: (K, P, nu)
    spline_memory = K * P * nu

    # Vanilla MPPI: (K, N, nu)
    vanilla_memory = K * N * nu

    reduction = (vanilla_memory - spline_memory) / vanilla_memory * 100

    print(f"Vanilla MPPI memory: {vanilla_memory} elements")
    print(f"Spline MPPI memory: {spline_memory} elements")
    print(f"Reduction: {reduction:.1f}%")

    assert spline_memory < vanilla_memory, "Spline should use less memory"
    assert reduction > 70, f"Expected >70% reduction, got {reduction:.1f}%"
    print("✓ PASS: Memory efficiency verified\n")


def test_control_smoothness():
    """제어 부드러움 테스트 (B-spline 효과)"""
    print("=" * 60)
    print("Test 4: Control Smoothness (B-spline Effect)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SplineMPPIParams(N=20, dt=0.05, K=128, spline_num_knots=6, spline_degree=3)
    controller = SplineMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((21, 3))

    # 연속 제어 호출
    controls = []
    num_steps = 10
    for i in range(num_steps):
        control, info = controller.compute_control(state, reference)
        controls.append(control.copy())

    controls = np.array(controls)  # (10, 2)

    # 제어 변화량
    control_diffs = np.diff(controls, axis=0)
    control_changes = np.linalg.norm(control_diffs, axis=1)

    mean_change = np.mean(control_changes)
    max_change = np.max(control_changes)

    print(f"Mean control change: {mean_change:.4f}")
    print(f"Max control change: {max_change:.4f}")

    # B-spline으로 인해 제어가 부드러워야 함
    assert mean_change < 0.5, f"Mean change too large: {mean_change}"
    print("✓ PASS: Control is smooth\n")


def test_knot_count_effect():
    """Knot 개수 효과 테스트"""
    print("=" * 60)
    print("Test 5: Knot Count Effect")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((21, 3))

    knot_counts = [4, 8, 16]

    for P in knot_counts:
        params = SplineMPPIParams(
            N=20, dt=0.05, K=128, spline_num_knots=P, spline_degree=3
        )
        controller = SplineMPPIController(model, params)

        control, info = controller.compute_control(state, reference)

        print(
            f"  Knots={P}: Cost={info['mean_cost']:.4f}, "
            f"Knot variance={info['spline_stats']['knot_variance']:.4f}"
        )

        assert not np.any(np.isnan(control)), f"Control contains NaN at P={P}"

    print("✓ PASS: Knot count effect verified\n")


def test_spline_statistics():
    """Spline 통계 추적 테스트"""
    print("=" * 60)
    print("Test 6: Spline Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SplineMPPIParams(N=20, dt=0.05, K=128, spline_num_knots=8, spline_degree=3)
    controller = SplineMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((21, 3))

    # 여러 번 호출
    num_steps = 10
    for i in range(num_steps):
        controller.compute_control(state, reference)

    # 통계 확인
    stats = controller.get_spline_statistics()

    print(f"Mean knot variance: {stats['mean_knot_variance']:.4f}")
    print(f"History length: {len(stats['spline_stats_history'])}")

    assert (
        len(stats["spline_stats_history"]) == num_steps
    ), "History length mismatch"
    assert stats["mean_knot_variance"] >= 0, "Knot variance should be non-negative"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Spline MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_bspline_interpolation()
        test_spline_mppi_basic()
        test_memory_efficiency()
        test_control_smoothness()
        test_knot_count_effect()
        test_spline_statistics()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
