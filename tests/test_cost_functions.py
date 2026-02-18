"""
비용 함수 (cost_functions.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ControlRateCost,
    ObstacleCost,
    CompositeMPPICost,
)

# ── Fixtures ───────────────────────────────────────────────

K, N, nx, nu = 32, 10, 3, 2


def _random_data():
    traj = np.random.randn(K, N + 1, nx)
    ctrl = np.random.randn(K, N, nu)
    ref = np.random.randn(N + 1, nx)
    return traj, ctrl, ref


# ── Tests ──────────────────────────────────────────────────


def test_state_tracking_cost_shape():
    print("\n" + "=" * 60)
    print("Test: StateTrackingCost shape (K,), non-negative")
    print("=" * 60)

    Q = np.array([10.0, 10.0, 1.0])
    cost_fn = StateTrackingCost(Q)
    traj, ctrl, ref = _random_data()

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,), f"shape: {costs.shape}"
    assert np.all(costs >= 0), f"negative cost: {costs.min()}"
    print(f"  shape={costs.shape}, min={costs.min():.4f}, max={costs.max():.4f}")
    print("PASS")


def test_state_tracking_cost_zero():
    print("\n" + "=" * 60)
    print("Test: StateTrackingCost = 0 when traj == ref")
    print("=" * 60)

    Q = np.array([10.0, 10.0, 1.0])
    cost_fn = StateTrackingCost(Q)

    ref = np.random.randn(N + 1, nx)
    traj = np.tile(ref, (K, 1, 1))  # (K, N+1, nx) == ref
    ctrl = np.zeros((K, N, nu))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.allclose(costs, 0.0, atol=1e-10), f"cost not zero: {costs}"
    print(f"  max_cost={costs.max():.2e}")
    print("PASS")


def test_state_tracking_excludes_terminal():
    print("\n" + "=" * 60)
    print("Test: StateTrackingCost excludes terminal state")
    print("=" * 60)

    Q = np.array([10.0, 10.0, 1.0])
    cost_fn = StateTrackingCost(Q)

    ref = np.zeros((N + 1, nx))
    traj = np.zeros((K, N + 1, nx))
    # Only terminal state deviates
    traj[:, -1, :] = 100.0
    ctrl = np.zeros((K, N, nu))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    # Terminal state should NOT contribute
    assert np.allclose(costs, 0.0, atol=1e-10), f"terminal leaked: {costs}"
    print("PASS")


def test_terminal_cost_shape():
    print("\n" + "=" * 60)
    print("Test: TerminalCost shape (K,), non-negative")
    print("=" * 60)

    Qf = np.array([10.0, 10.0, 1.0])
    cost_fn = TerminalCost(Qf)
    traj, ctrl, ref = _random_data()

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,), f"shape: {costs.shape}"
    assert np.all(costs >= 0), f"negative cost: {costs.min()}"
    print("PASS")


def test_terminal_cost_only_terminal():
    print("\n" + "=" * 60)
    print("Test: TerminalCost uses only terminal state")
    print("=" * 60)

    Qf = np.array([1.0, 1.0, 1.0])
    cost_fn = TerminalCost(Qf)

    ref = np.zeros((N + 1, nx))
    traj = np.zeros((K, N + 1, nx))
    # Only non-terminal states deviate
    traj[:, :-1, :] = 100.0
    ctrl = np.zeros((K, N, nu))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.allclose(costs, 0.0, atol=1e-10), f"non-terminal leaked: {costs}"
    print("PASS")


def test_control_effort_cost_shape():
    print("\n" + "=" * 60)
    print("Test: ControlEffortCost shape (K,), non-negative")
    print("=" * 60)

    R = np.array([0.1, 0.1])
    cost_fn = ControlEffortCost(R)
    traj, ctrl, ref = _random_data()

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,), f"shape: {costs.shape}"
    assert np.all(costs >= 0), f"negative cost: {costs.min()}"
    print("PASS")


def test_control_effort_cost_zero():
    print("\n" + "=" * 60)
    print("Test: ControlEffortCost = 0 when u = 0")
    print("=" * 60)

    R = np.array([0.1, 0.1])
    cost_fn = ControlEffortCost(R)

    traj = np.zeros((K, N + 1, nx))
    ctrl = np.zeros((K, N, nu))
    ref = np.zeros((N + 1, nx))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.allclose(costs, 0.0), f"cost not zero: {costs}"
    print("PASS")


def test_control_rate_cost():
    print("\n" + "=" * 60)
    print("Test: ControlRateCost: du=0 -> 0, du!=0 -> >0")
    print("=" * 60)

    R_rate = np.array([1.0, 1.0])
    cost_fn = ControlRateCost(R_rate)

    traj = np.zeros((K, N + 1, nx))
    ref = np.zeros((N + 1, nx))

    # Constant control -> du = 0
    ctrl_const = np.ones((K, N, nu)) * 0.5
    costs_const = cost_fn.compute_cost(traj, ctrl_const, ref)
    assert np.allclose(costs_const, 0.0), f"const control cost: {costs_const}"

    # Varying control -> du != 0
    ctrl_vary = np.random.randn(K, N, nu)
    costs_vary = cost_fn.compute_cost(traj, ctrl_vary, ref)
    assert np.all(costs_vary > 0), f"varying control cost should be >0"

    print(f"  const_cost={costs_const.max():.2e}, vary_cost_mean={costs_vary.mean():.4f}")
    print("PASS")


def test_obstacle_cost_no_collision():
    print("\n" + "=" * 60)
    print("Test: ObstacleCost = ~0 when far from obstacle")
    print("=" * 60)

    obstacles = [(10.0, 10.0, 0.5)]
    cost_fn = ObstacleCost(obstacles, safety_margin=0.2, cost_weight=100.0)

    # Trajectory near origin, obstacle at (10, 10)
    traj = np.zeros((K, N + 1, nx))
    traj[:, :, 0] = np.linspace(0, 1, N + 1)
    ctrl = np.zeros((K, N, nu))
    ref = np.zeros((N + 1, nx))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.allclose(costs, 0.0, atol=1e-10), f"unexpected cost: {costs}"
    print("PASS")


def test_obstacle_cost_collision():
    print("\n" + "=" * 60)
    print("Test: ObstacleCost >> 0 when inside obstacle")
    print("=" * 60)

    obstacles = [(0.5, 0.0, 1.0)]
    cost_fn = ObstacleCost(obstacles, safety_margin=0.2, cost_weight=100.0)

    traj = np.zeros((K, N + 1, nx))
    traj[:, :, 0] = np.linspace(0, 1, N + 1)  # passes through obstacle center
    ctrl = np.zeros((K, N, nu))
    ref = np.zeros((N + 1, nx))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.all(costs > 0), f"collision not penalized: {costs}"
    print(f"  mean_cost={costs.mean():.2f}")
    print("PASS")


def test_obstacle_cost_empty():
    print("\n" + "=" * 60)
    print("Test: ObstacleCost = 0 with empty obstacle list")
    print("=" * 60)

    cost_fn = ObstacleCost(obstacles=[], safety_margin=0.2, cost_weight=100.0)
    traj, ctrl, ref = _random_data()

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.allclose(costs, 0.0), f"empty obstacles cost: {costs}"
    print("PASS")


def test_composite_cost_sum():
    print("\n" + "=" * 60)
    print("Test: CompositeMPPICost == sum of individual costs")
    print("=" * 60)

    Q = np.array([10.0, 10.0, 1.0])
    Qf = np.array([10.0, 10.0, 1.0])
    R = np.array([0.1, 0.1])

    c_state = StateTrackingCost(Q)
    c_term = TerminalCost(Qf)
    c_ctrl = ControlEffortCost(R)
    composite = CompositeMPPICost([c_state, c_term, c_ctrl])

    traj, ctrl, ref = _random_data()

    total = composite.compute_cost(traj, ctrl, ref)
    individual_sum = (
        c_state.compute_cost(traj, ctrl, ref)
        + c_term.compute_cost(traj, ctrl, ref)
        + c_ctrl.compute_cost(traj, ctrl, ref)
    )

    assert np.allclose(total, individual_sum, atol=1e-10), \
        f"diff={np.max(np.abs(total - individual_sum))}"
    print("PASS")


def test_diagonal_vs_full_matrix():
    print("\n" + "=" * 60)
    print("Test: diagonal Q == diag(Q) full matrix")
    print("=" * 60)

    Q_diag = np.array([10.0, 10.0, 1.0])
    Q_full = np.diag(Q_diag)

    c_diag = StateTrackingCost(Q_diag)
    c_full = StateTrackingCost(Q_full)

    traj, ctrl, ref = _random_data()

    costs_diag = c_diag.compute_cost(traj, ctrl, ref)
    costs_full = c_full.compute_cost(traj, ctrl, ref)

    assert np.allclose(costs_diag, costs_full, atol=1e-10), \
        f"diff={np.max(np.abs(costs_diag - costs_full))}"
    print("PASS")


def test_cost_function_abc():
    print("\n" + "=" * 60)
    print("Test: CostFunction ABC -> cannot instantiate")
    print("=" * 60)

    try:
        CostFunction()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Cost Functions Unit Tests")
    print("=" * 60)

    tests = [
        test_state_tracking_cost_shape,
        test_state_tracking_cost_zero,
        test_state_tracking_excludes_terminal,
        test_terminal_cost_shape,
        test_terminal_cost_only_terminal,
        test_control_effort_cost_shape,
        test_control_effort_cost_zero,
        test_control_rate_cost,
        test_obstacle_cost_no_collision,
        test_obstacle_cost_collision,
        test_obstacle_cost_empty,
        test_composite_cost_sum,
        test_diagonal_vs_full_matrix,
        test_cost_function_abc,
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
