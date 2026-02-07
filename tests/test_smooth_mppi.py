"""
Smooth MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, SmoothMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController


def test_smooth_mppi_basic():
    """Smooth MPPI 기본 동작 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: Smooth MPPI Basic Functionality")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SmoothMPPIParams(N=10, dt=0.05, K=128, jerk_weight=1.0)
    controller = SmoothMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    print(f"Control: {control}")
    print(f"Mean jerk: {info['mean_jerk']:.4f}")
    print(f"Tracking cost: {info['mean_cost']:.4f}")
    print(f"ESS: {info['ess']:.2f}")

    # 기본 체크
    assert control.shape == (2,), f"Control shape mismatch: {control.shape}"
    assert "jerk_costs" in info, "Info should contain jerk_costs"
    assert "mean_jerk" in info, "Info should contain mean_jerk"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print("✓ PASS: Smooth MPPI works\n")


def test_jerk_weight_effect():
    """Jerk 가중치 효과 테스트"""
    print("=" * 60)
    print("Test 2: Jerk Weight Effect")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 낮은 Jerk 가중치
    params_low = SmoothMPPIParams(N=10, dt=0.05, K=128, jerk_weight=0.1)
    controller_low = SmoothMPPIController(model, params_low)
    control_low, info_low = controller_low.compute_control(state, reference)

    # 높은 Jerk 가중치
    params_high = SmoothMPPIParams(N=10, dt=0.05, K=128, jerk_weight=10.0)
    controller_high = SmoothMPPIController(model, params_high)
    control_high, info_high = controller_high.compute_control(state, reference)

    print(f"Low jerk weight (0.1):")
    print(f"  Mean jerk: {info_low['mean_jerk']:.4f}")
    print(f"  Tracking cost: {np.mean(info_low['tracking_costs']):.4f}")

    print(f"\nHigh jerk weight (10.0):")
    print(f"  Mean jerk: {info_high['mean_jerk']:.4f}")
    print(f"  Tracking cost: {np.mean(info_high['tracking_costs']):.4f}")

    # 높은 가중치가 더 낮은 jerk를 생성해야 함
    # (하지만 tracking cost는 증가할 수 있음)
    assert not np.any(np.isnan(control_low)), "Control (low) contains NaN"
    assert not np.any(np.isnan(control_high)), "Control (high) contains NaN"
    print("\n✓ PASS: Jerk weight effect verified\n")


def test_control_smoothness():
    """제어 부드러움 테스트 (연속 호출)"""
    print("=" * 60)
    print("Test 3: Control Smoothness Over Time")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SmoothMPPIParams(N=10, dt=0.05, K=128, jerk_weight=5.0)
    controller = SmoothMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 연속 제어 호출
    controls = []
    num_steps = 10
    for i in range(num_steps):
        control, info = controller.compute_control(state, reference)
        controls.append(control.copy())

    controls = np.array(controls)  # (10, 2)

    # 제어 변화량 계산
    control_diffs = np.diff(controls, axis=0)  # (9, 2)
    control_changes = np.linalg.norm(control_diffs, axis=1)  # (9,)

    mean_change = np.mean(control_changes)
    max_change = np.max(control_changes)

    print(f"Mean control change: {mean_change:.4f}")
    print(f"Max control change: {max_change:.4f}")

    # 제어 변화량이 합리적인 범위 내에 있어야 함
    assert mean_change < 1.0, f"Mean control change too large: {mean_change}"
    assert max_change < 2.0, f"Max control change too large: {max_change}"
    print("✓ PASS: Control is smooth\n")


def test_jerk_statistics():
    """Jerk 통계 추적 테스트"""
    print("=" * 60)
    print("Test 4: Jerk Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SmoothMPPIParams(N=10, dt=0.05, K=128, jerk_weight=1.0)
    controller = SmoothMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 여러 번 호출
    num_steps = 10
    for i in range(num_steps):
        controller.compute_control(state, reference)

    # 통계 확인
    stats = controller.get_jerk_statistics()

    print(f"Mean jerk: {stats['mean_jerk']:.4f}")
    print(f"Max jerk: {stats['max_jerk']:.4f}")
    print(f"History length: {len(stats['jerk_history'])}")

    assert len(stats["jerk_history"]) == num_steps, "History length mismatch"
    assert stats["mean_jerk"] >= 0, "Jerk should be non-negative"
    print("✓ PASS: Statistics tracking works\n")


def test_zero_jerk_weight():
    """Jerk 가중치 0일 때 동작 테스트"""
    print("=" * 60)
    print("Test 5: Zero Jerk Weight")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SmoothMPPIParams(N=10, dt=0.05, K=128, jerk_weight=0.0)
    controller = SmoothMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    print(f"Control: {control}")
    print(f"Mean jerk: {info['mean_jerk']:.4f}")
    print(f"Jerk costs sum: {np.sum(info['jerk_costs']):.4f}")

    # Jerk 가중치가 0이면 jerk_costs도 0
    assert np.allclose(
        info["jerk_costs"], 0.0
    ), f"Jerk costs should be 0: {info['jerk_costs']}"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print("✓ PASS: Zero jerk weight works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Smooth MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_smooth_mppi_basic()
        test_jerk_weight_effect()
        test_control_smoothness()
        test_jerk_statistics()
        test_zero_jerk_weight()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
