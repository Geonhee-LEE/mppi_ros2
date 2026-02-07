"""
Residual Dynamics 모델 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.learned.residual_dynamics import (
    ResidualDynamics,
    create_constant_residual,
)


def test_residual_dynamics_equivalence():
    """Residual이 None일 때 베이스 모델과 동일한지 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: Residual=None → Base Model Equivalence")
    print("=" * 60)

    base_model = DifferentialDriveKinematic()
    residual_model = ResidualDynamics(base_model, residual_fn=None)

    state = np.array([1.0, 2.0, 0.5])
    control = np.array([0.5, 0.1])

    base_dot = base_model.forward_dynamics(state, control)
    residual_dot = residual_model.forward_dynamics(state, control)

    difference = np.linalg.norm(base_dot - residual_dot)

    print(f"Base dot:     {base_dot}")
    print(f"Residual dot: {residual_dot}")
    print(f"Difference:   {difference:.6e}")

    assert difference < 1e-10, f"Difference too large: {difference}"
    print("✓ PASS: Residual=None matches base model\n")


def test_constant_residual():
    """상수 residual 효과 테스트"""
    print("=" * 60)
    print("Test 2: Constant Residual Effect")
    print("=" * 60)

    base_model = DifferentialDriveKinematic()
    residual_value = np.array([0.1, 0.2, 0.05])
    residual_fn = create_constant_residual(residual_value)
    residual_model = ResidualDynamics(base_model, residual_fn=residual_fn)

    state = np.array([1.0, 2.0, 0.5])
    control = np.array([0.5, 0.1])

    base_dot = base_model.forward_dynamics(state, control)
    residual_dot = residual_model.forward_dynamics(state, control)
    expected_dot = base_dot + residual_value

    difference = np.linalg.norm(residual_dot - expected_dot)

    print(f"Base dot:     {base_dot}")
    print(f"Residual val: {residual_value}")
    print(f"Expected dot: {expected_dot}")
    print(f"Actual dot:   {residual_dot}")
    print(f"Difference:   {difference:.6e}")

    assert difference < 1e-10, f"Difference too large: {difference}"
    print("✓ PASS: Constant residual correctly added\n")


def test_batch_residual():
    """배치 처리 테스트"""
    print("=" * 60)
    print("Test 3: Batch Processing")
    print("=" * 60)

    base_model = DifferentialDriveKinematic()
    residual_value = np.array([0.1, 0.2, 0.05])
    residual_fn = create_constant_residual(residual_value)
    residual_model = ResidualDynamics(base_model, residual_fn=residual_fn)

    batch_size = 10
    states = np.random.randn(batch_size, 3)
    controls = np.random.randn(batch_size, 2)

    residual_dots = residual_model.forward_dynamics(states, controls)

    print(f"Batch size: {batch_size}")
    print(f"Output shape: {residual_dots.shape}")
    print(f"Expected: ({batch_size}, 3)")

    assert residual_dots.shape == (batch_size, 3), f"Shape mismatch: {residual_dots.shape}"
    print("✓ PASS: Batch processing works correctly\n")


def test_residual_contribution():
    """Residual 기여도 분석 테스트"""
    print("=" * 60)
    print("Test 4: Residual Contribution Analysis")
    print("=" * 60)

    base_model = DifferentialDriveKinematic()
    residual_value = np.array([0.05, 0.05, 0.0])
    residual_fn = create_constant_residual(residual_value)
    residual_model = ResidualDynamics(base_model, residual_fn=residual_fn)

    state = np.array([1.0, 2.0, 0.5])
    control = np.array([0.5, 0.1])

    contribution = residual_model.get_residual_contribution(state, control)

    print(f"Physics dot:   {contribution['physics_dot']}")
    print(f"Residual dot:  {contribution['residual_dot']}")
    print(f"Total dot:     {contribution['total_dot']}")
    print(f"Residual %:    {contribution['residual_ratio'] * 100}")

    # 검증
    expected_total = contribution['physics_dot'] + contribution['residual_dot']
    difference = np.linalg.norm(contribution['total_dot'] - expected_total)

    assert difference < 1e-10, f"Total != Physics + Residual: {difference}"
    print("✓ PASS: Contribution analysis correct\n")


def test_stats_tracking():
    """통계 추적 테스트"""
    print("=" * 60)
    print("Test 5: Statistics Tracking")
    print("=" * 60)

    base_model = DifferentialDriveKinematic()
    residual_value = np.array([0.1, 0.2, 0.05])
    residual_fn = create_constant_residual(residual_value)
    residual_model = ResidualDynamics(base_model, residual_fn=residual_fn)

    # 여러 번 호출
    num_calls = 100
    for i in range(num_calls):
        state = np.random.randn(3)
        control = np.random.randn(2)
        residual_model.forward_dynamics(state, control)

    stats = residual_model.get_stats()

    print(f"Num calls: {stats['num_calls']}")
    print(f"Mean residual: {stats['residual_mean']}")
    print(f"Std residual: {stats['residual_std']}")

    # 상수 residual이므로 평균이 residual_value와 같아야 함
    mean_difference = np.linalg.norm(stats['residual_mean'] - residual_value)

    print(f"Expected mean: {residual_value}")
    print(f"Difference: {mean_difference:.6e}")

    assert stats['num_calls'] == num_calls, "Num calls mismatch"
    assert mean_difference < 0.01, f"Mean too different: {mean_difference}"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Residual Dynamics Model Tests".center(60))
    print("=" * 60)

    try:
        test_residual_dynamics_equivalence()
        test_constant_residual()
        test_batch_residual()
        test_residual_contribution()
        test_stats_tracking()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
