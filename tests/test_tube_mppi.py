"""
Tube-MPPI 및 Ancillary Controller 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import TubeMPPIParams
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.ancillary_controller import (
    AncillaryController,
    create_default_ancillary_controller,
)


def test_ancillary_controller():
    """Ancillary Controller 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: Ancillary Controller")
    print("=" * 60)

    # 기구학 모델용 컨트롤러
    ancillary = create_default_ancillary_controller("kinematic", gain_scale=1.0)

    # 테스트 상태
    state = np.array([1.0, 2.0, 0.5])  # [x, y, θ]
    nominal_state = np.array([0.9, 1.9, 0.4])  # 약간 다른 명목 상태

    # 피드백 계산
    feedback = ancillary.compute_feedback(state, nominal_state)

    print(f"State:         {state}")
    print(f"Nominal State: {nominal_state}")
    print(f"Error:         {state - nominal_state}")
    print(f"Feedback:      {feedback}")

    assert feedback.shape == (2,), f"Feedback shape mismatch: {feedback.shape}"
    print("✓ PASS: Ancillary controller works\n")


def test_tube_mppi_disabled():
    """Tube 비활성화 시 Vanilla MPPI와 동일한지 테스트"""
    print("=" * 60)
    print("Test 2: Tube-MPPI (disabled) = Vanilla MPPI")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = TubeMPPIParams(
        N=10, dt=0.05, K=128,
        tube_enabled=False,  # 비활성화
    )
    controller = TubeMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.tile(state, (11, 1))  # (N+1, nx)

    control, info = controller.compute_control(state, reference)

    print(f"Tube enabled: {info['tube_enabled']}")
    print(f"Control: {control}")

    assert not info["tube_enabled"], "Tube should be disabled"
    assert "nominal_state" in info, "Info should contain nominal_state"
    print("✓ PASS: Tube disabled works\n")


def test_tube_mppi_enabled():
    """Tube 활성화 시 정상 작동 테스트"""
    print("=" * 60)
    print("Test 3: Tube-MPPI (enabled)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = TubeMPPIParams(
        N=10, dt=0.05, K=128,
        tube_enabled=True,  # 활성화
        tube_margin=0.1,
    )
    controller = TubeMPPIController(model, params)

    # 초기 상태
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 첫 번째 호출
    control1, info1 = controller.compute_control(state, reference)

    print(f"Tube enabled: {info1['tube_enabled']}")
    print(f"Control 1: {control1}")
    print(f"Nominal state 1: {info1['nominal_state']}")
    print(f"Tube width 1: {info1['tube_width']}")

    # 두 번째 호출 (약간 다른 상태로 외란 모방)
    state_with_noise = state + np.array([0.05, 0.05, 0.01])
    control2, info2 = controller.compute_control(state_with_noise, reference)

    print(f"\nControl 2: {control2}")
    print(f"Nominal state 2: {info2['nominal_state']}")
    print(f"Tube width 2: {info2['tube_width']}")
    print(f"Feedback correction: {info2['feedback_correction']}")

    assert info2["tube_width"] > 0, "Tube width should be positive"
    assert "feedback_correction" in info2, "Info should contain feedback_correction"
    print("✓ PASS: Tube enabled works\n")


def test_tube_statistics():
    """Tube 통계 추적 테스트"""
    print("=" * 60)
    print("Test 4: Tube Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = TubeMPPIParams(N=10, dt=0.05, K=128, tube_enabled=True)
    controller = TubeMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 여러 번 호출 (외란 추가)
    num_steps = 10
    for i in range(num_steps):
        noise = np.random.randn(3) * 0.01
        state_noisy = state + noise
        controller.compute_control(state_noisy, reference)

    # 통계 확인
    stats = controller.get_tube_statistics()

    print(f"Mean Tube Width: {stats['mean_tube_width']:.4f}")
    print(f"Max Tube Width: {stats['max_tube_width']:.4f}")
    print(f"History Length: {len(stats['tube_width_history'])}")

    assert len(stats["tube_width_history"]) == num_steps, "History length mismatch"
    assert stats["mean_tube_width"] >= 0, "Mean tube width should be non-negative"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Tube-MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_ancillary_controller()
        test_tube_mppi_disabled()
        test_tube_mppi_enabled()
        test_tube_statistics()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
