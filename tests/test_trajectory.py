"""
궤적 생성 유틸리티 (trajectory.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.utils.trajectory import (
    circle_trajectory,
    figure_eight_trajectory,
    sine_wave_trajectory,
    straight_line_trajectory,
    generate_reference_trajectory,
    create_trajectory_function,
)


# ── Tests ──────────────────────────────────────────────────


def test_circle_shape():
    print("\n" + "=" * 60)
    print("Test: circle_trajectory returns (3,) = [x, y, theta]")
    print("=" * 60)

    state = circle_trajectory(0.0)
    assert state.shape == (3,), f"shape: {state.shape}"
    state2 = circle_trajectory(5.0)
    assert state2.shape == (3,), f"shape at t=5: {state2.shape}"
    print("PASS")


def test_circle_radius():
    print("\n" + "=" * 60)
    print("Test: circle sqrt(x^2 + y^2) ~ radius")
    print("=" * 60)

    radius = 3.0
    for t in [0, 1, 2, 5, 10]:
        state = circle_trajectory(float(t), radius=radius)
        r = np.sqrt(state[0] ** 2 + state[1] ** 2)
        assert abs(r - radius) < 1e-10, f"t={t}: r={r}, expected={radius}"
    print("PASS")


def test_figure8_shape():
    print("\n" + "=" * 60)
    print("Test: figure_eight_trajectory returns (3,)")
    print("=" * 60)

    state = figure_eight_trajectory(0.0)
    assert state.shape == (3,), f"shape: {state.shape}"
    state2 = figure_eight_trajectory(5.0)
    assert state2.shape == (3,), f"shape at t=5: {state2.shape}"
    # Check no NaN
    assert not np.any(np.isnan(state)), f"NaN at t=0: {state}"
    assert not np.any(np.isnan(state2)), f"NaN at t=5: {state2}"
    print("PASS")


def test_sine_shape():
    print("\n" + "=" * 60)
    print("Test: sine_wave_trajectory returns (3,)")
    print("=" * 60)

    state = sine_wave_trajectory(0.0)
    assert state.shape == (3,), f"shape: {state.shape}"
    state2 = sine_wave_trajectory(3.0)
    assert state2.shape == (3,), f"shape at t=3: {state2.shape}"
    print("PASS")


def test_straight_line():
    print("\n" + "=" * 60)
    print("Test: straight_line x = v*t, y = 0 (heading=0)")
    print("=" * 60)

    v = 2.0
    for t in [0, 1, 3, 5]:
        state = straight_line_trajectory(float(t), velocity=v, heading=0.0)
        assert abs(state[0] - v * t) < 1e-10, f"x={state[0]}, expected={v*t}"
        assert abs(state[1]) < 1e-10, f"y={state[1]}, expected=0"
        assert abs(state[2]) < 1e-10, f"theta={state[2]}, expected=0"
    print("PASS")


def test_generate_reference_shape():
    print("\n" + "=" * 60)
    print("Test: generate_reference_trajectory returns (N+1, 3)")
    print("=" * 60)

    N = 20
    dt = 0.05
    ref = generate_reference_trajectory(circle_trajectory, 0.0, N, dt)

    assert ref.shape == (N + 1, 3), f"shape: {ref.shape}"
    # First point should match circle_trajectory(0.0)
    expected_first = circle_trajectory(0.0)
    assert np.allclose(ref[0], expected_first), \
        f"first: {ref[0]} != {expected_first}"
    print("PASS")


def test_create_trajectory_function():
    print("\n" + "=" * 60)
    print("Test: create_trajectory_function for all types")
    print("=" * 60)

    for ttype in ["circle", "figure8", "sine", "straight"]:
        fn = create_trajectory_function(ttype)
        state = fn(1.0)
        assert state.shape == (3,), f"{ttype}: shape={state.shape}"
        assert not np.any(np.isnan(state)), f"{ttype}: NaN"
        print(f"  {ttype}: {state}")
    print("PASS")


def test_unknown_trajectory_type():
    print("\n" + "=" * 60)
    print("Test: unknown type -> ValueError")
    print("=" * 60)

    try:
        create_trajectory_function("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  caught: {e}")
    print("PASS")


def test_heading_consistency():
    print("\n" + "=" * 60)
    print("Test: heading ~ atan2(dy, dx) for circle trajectory")
    print("=" * 60)

    dt_check = 0.001
    for t in [0.0, 1.0, 3.0, 5.0]:
        s0 = circle_trajectory(t)
        s1 = circle_trajectory(t + dt_check)
        dx = s1[0] - s0[0]
        dy = s1[1] - s0[1]
        numerical_heading = np.arctan2(dy, dx)

        # Compare with reported heading
        heading = s0[2]
        diff = np.arctan2(np.sin(heading - numerical_heading),
                          np.cos(heading - numerical_heading))
        assert abs(diff) < 0.05, \
            f"t={t}: heading={heading:.4f}, numerical={numerical_heading:.4f}, diff={diff:.4f}"

    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Trajectory Utils Unit Tests")
    print("=" * 60)

    tests = [
        test_circle_shape,
        test_circle_radius,
        test_figure8_shape,
        test_sine_shape,
        test_straight_line,
        test_generate_reference_shape,
        test_create_trajectory_function,
        test_unknown_trajectory_type,
        test_heading_consistency,
    ]

    try:
        for t in tests:
            t()
        print(f"\n{'=' * 60}")
        print(f"  All {len(tests)} Tests Passed!")
        print(f"{'=' * 60}")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
