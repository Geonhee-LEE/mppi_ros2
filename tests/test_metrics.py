"""
메트릭 (metrics.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.simulation.metrics import compute_metrics, angle_difference


def _make_history(T=50, nx=3, nu=2, noise_scale=0.1):
    """헬퍼: 기본 히스토리 생성"""
    ref = np.zeros((T, nx))
    ref[:, 0] = np.linspace(0, 5, T)
    ref[:, 1] = np.sin(np.linspace(0, 2 * np.pi, T))

    state = ref + np.random.randn(T, nx) * noise_scale
    control = np.random.randn(T, nu) * 0.5
    solve_time = np.random.uniform(0.001, 0.01, T)

    return {
        "state": state,
        "control": control,
        "reference": ref,
        "solve_time": solve_time,
    }


# ── Tests ──────────────────────────────────────────────────


def test_compute_metrics_keys():
    print("\n" + "=" * 60)
    print("Test: compute_metrics returns required keys")
    print("=" * 60)

    history = _make_history()
    metrics = compute_metrics(history)

    required = [
        "position_rmse", "max_position_error",
        "heading_rmse", "max_heading_error",
        "control_rate", "max_control_rate",
        "mean_solve_time", "max_solve_time", "std_solve_time",
    ]
    for key in required:
        assert key in metrics, f"missing key: {key}"
    print(f"  keys: {list(metrics.keys())}")
    print("PASS")


def test_position_rmse_zero():
    print("\n" + "=" * 60)
    print("Test: perfect tracking -> position_rmse = 0")
    print("=" * 60)

    T, nx, nu = 30, 3, 2
    ref = np.random.randn(T, nx)
    history = {
        "state": ref.copy(),
        "control": np.zeros((T, nu)),
        "reference": ref,
        "solve_time": np.ones(T) * 0.001,
    }

    metrics = compute_metrics(history)
    assert abs(metrics["position_rmse"]) < 1e-10, \
        f"rmse: {metrics['position_rmse']}"
    assert abs(metrics["max_position_error"]) < 1e-10
    print("PASS")


def test_position_rmse_positive():
    print("\n" + "=" * 60)
    print("Test: with error -> position_rmse > 0")
    print("=" * 60)

    history = _make_history(noise_scale=0.5)
    metrics = compute_metrics(history)

    assert metrics["position_rmse"] > 0, f"rmse: {metrics['position_rmse']}"
    assert metrics["max_position_error"] >= metrics["position_rmse"], \
        "max should be >= rmse"
    print(f"  rmse={metrics['position_rmse']:.4f}, max={metrics['max_position_error']:.4f}")
    print("PASS")


def test_heading_rmse():
    print("\n" + "=" * 60)
    print("Test: heading_rmse with angle normalization")
    print("=" * 60)

    T, nx, nu = 30, 3, 2
    ref = np.zeros((T, nx))
    state = np.zeros((T, nx))
    # Small heading error
    state[:, 2] = 0.1

    history = {
        "state": state,
        "control": np.zeros((T, nu)),
        "reference": ref,
        "solve_time": np.ones(T) * 0.001,
    }

    metrics = compute_metrics(history)
    assert abs(metrics["heading_rmse"] - 0.1) < 1e-6, \
        f"heading_rmse: {metrics['heading_rmse']}"
    print(f"  heading_rmse={metrics['heading_rmse']:.6f}")
    print("PASS")


def test_angle_difference():
    print("\n" + "=" * 60)
    print("Test: angle_difference wraps around pi boundary")
    print("=" * 60)

    # Basic case
    diff = angle_difference(np.array([0.1]), np.array([0.0]))
    assert abs(diff[0] - 0.1) < 1e-10, f"basic: {diff[0]}"

    # Wrap-around: 3.0 - (-3.0) should NOT be 6.0 but ~0.28
    a1 = np.array([3.0])
    a2 = np.array([-3.0])
    diff = angle_difference(a1, a2)
    expected = 3.0 - (-3.0)  # 6.0 -> wrapped
    wrapped = np.arctan2(np.sin(expected), np.cos(expected))
    assert abs(diff[0] - wrapped) < 1e-10, f"wrap: {diff[0]} != {wrapped}"

    # pi boundary: pi - (-pi) = 2pi -> wrapped to 0
    diff_pi = angle_difference(np.array([np.pi]), np.array([-np.pi]))
    assert abs(diff_pi[0]) < 1e-10, f"pi boundary: {diff_pi[0]}"

    print(f"  basic=0.1->0.1, wrap=6.0->{wrapped:.4f}, pi_boundary={diff_pi[0]:.2e}")
    print("PASS")


def test_control_rate():
    print("\n" + "=" * 60)
    print("Test: control_rate from diff(controls)")
    print("=" * 60)

    T, nx, nu = 30, 3, 2
    # Constant control -> rate = 0
    history_const = {
        "state": np.zeros((T, nx)),
        "control": np.ones((T, nu)) * 0.5,
        "reference": np.zeros((T, nx)),
        "solve_time": np.ones(T) * 0.001,
    }
    metrics_const = compute_metrics(history_const)
    assert abs(metrics_const["control_rate"]) < 1e-10, \
        f"const rate: {metrics_const['control_rate']}"

    # Varying control -> rate > 0
    history_vary = {
        "state": np.zeros((T, nx)),
        "control": np.random.randn(T, nu),
        "reference": np.zeros((T, nx)),
        "solve_time": np.ones(T) * 0.001,
    }
    metrics_vary = compute_metrics(history_vary)
    assert metrics_vary["control_rate"] > 0, \
        f"vary rate: {metrics_vary['control_rate']}"
    print(f"  const_rate={metrics_const['control_rate']:.2e}, vary_rate={metrics_vary['control_rate']:.4f}")
    print("PASS")


def test_solve_time_ms():
    print("\n" + "=" * 60)
    print("Test: solve_time conversion sec -> ms")
    print("=" * 60)

    T, nx, nu = 10, 3, 2
    solve_sec = 0.005  # 5ms
    history = {
        "state": np.zeros((T, nx)),
        "control": np.zeros((T, nu)),
        "reference": np.zeros((T, nx)),
        "solve_time": np.ones(T) * solve_sec,
    }

    metrics = compute_metrics(history)
    assert abs(metrics["mean_solve_time"] - 5.0) < 1e-6, \
        f"mean_solve_time: {metrics['mean_solve_time']}"
    assert abs(metrics["max_solve_time"] - 5.0) < 1e-6
    assert abs(metrics["std_solve_time"]) < 1e-6  # all same
    print(f"  mean={metrics['mean_solve_time']:.2f}ms")
    print("PASS")


def test_single_step_history():
    print("\n" + "=" * 60)
    print("Test: T=1 -> control_rate = 0")
    print("=" * 60)

    history = {
        "state": np.array([[1.0, 2.0, 0.5]]),
        "control": np.array([[0.5, 0.3]]),
        "reference": np.array([[1.0, 2.0, 0.5]]),
        "solve_time": np.array([0.005]),
    }

    metrics = compute_metrics(history)
    assert metrics["control_rate"] == 0.0, \
        f"control_rate: {metrics['control_rate']}"
    assert metrics["max_control_rate"] == 0.0
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Metrics Unit Tests")
    print("=" * 60)

    tests = [
        test_compute_metrics_keys,
        test_position_rmse_zero,
        test_position_rmse_positive,
        test_heading_rmse,
        test_angle_difference,
        test_control_rate,
        test_solve_time_ms,
        test_single_step_history,
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
