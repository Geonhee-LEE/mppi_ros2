"""
시뮬레이터 (simulator.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory

N_horizon = 10
dt = 0.05


def _make_sim(process_noise_std=None):
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = MPPIParams(K=32, N=N_horizon, dt=dt, sigma=np.array([0.5, 0.5]))
    ctrl = MPPIController(model, params)
    return Simulator(model, ctrl, dt, process_noise_std=process_noise_std)


def _make_ref_fn():
    def ref_fn(t):
        return generate_reference_trajectory(circle_trajectory, t, N_horizon, dt)
    return ref_fn


# ── Tests ──────────────────────────────────────────────────


def test_reset():
    print("\n" + "=" * 60)
    print("Test: reset() initializes state and clears history")
    print("=" * 60)

    sim = _make_sim()
    init = np.array([1.0, 2.0, 0.5])
    sim.reset(init)

    assert np.allclose(sim.state, init), f"state: {sim.state}"
    assert sim.t == 0.0, f"t: {sim.t}"
    assert all(len(v) == 0 for v in sim.history.values()), "history not empty"
    print("PASS")


def test_step_returns_dict():
    print("\n" + "=" * 60)
    print("Test: step() returns dict with required keys")
    print("=" * 60)

    sim = _make_sim()
    sim.reset(np.array([5.0, 0.0, np.pi / 2]))
    ref = generate_reference_trajectory(circle_trajectory, 0.0, N_horizon, dt)

    result = sim.step(ref)
    assert isinstance(result, dict), f"type: {type(result)}"
    for key in ["state", "control", "solve_time", "info"]:
        assert key in result, f"missing key: {key}"
    assert result["state"].shape == (3,), f"state shape: {result['state'].shape}"
    assert result["control"].shape == (2,), f"control shape: {result['control'].shape}"
    assert result["solve_time"] > 0, f"solve_time: {result['solve_time']}"
    print("PASS")


def test_history_recording():
    print("\n" + "=" * 60)
    print("Test: N steps -> history length N")
    print("=" * 60)

    sim = _make_sim()
    sim.reset(np.array([5.0, 0.0, np.pi / 2]))

    n_steps = 5
    for i in range(n_steps):
        ref = generate_reference_trajectory(circle_trajectory, sim.t, N_horizon, dt)
        sim.step(ref)

    assert len(sim.history["state"]) == n_steps, \
        f"state history len: {len(sim.history['state'])}"
    assert len(sim.history["control"]) == n_steps
    assert len(sim.history["solve_time"]) == n_steps
    print("PASS")


def test_run_duration():
    print("\n" + "=" * 60)
    print("Test: run(duration) -> correct number of steps")
    print("=" * 60)

    sim = _make_sim()
    sim.reset(np.array([5.0, 0.0, np.pi / 2]))

    duration = 0.5  # seconds
    expected_steps = int(duration / dt)  # 10
    ref_fn = _make_ref_fn()
    history = sim.run(ref_fn, duration)

    assert history["state"].shape[0] == expected_steps, \
        f"steps: {history['state'].shape[0]}, expected: {expected_steps}"
    print(f"  duration={duration}s, dt={dt}, steps={expected_steps}")
    print("PASS")


def test_get_history_shapes():
    print("\n" + "=" * 60)
    print("Test: get_history() returns np.ndarray with correct shapes")
    print("=" * 60)

    sim = _make_sim()
    sim.reset(np.array([5.0, 0.0, np.pi / 2]))

    n_steps = 8
    for i in range(n_steps):
        ref = generate_reference_trajectory(circle_trajectory, sim.t, N_horizon, dt)
        sim.step(ref)

    history = sim.get_history()
    assert history["state"].shape == (n_steps, 3), f"state: {history['state'].shape}"
    assert history["control"].shape == (n_steps, 2), f"control: {history['control'].shape}"
    assert history["time"].shape == (n_steps,), f"time: {history['time'].shape}"
    assert history["solve_time"].shape == (n_steps,), f"solve_time: {history['solve_time'].shape}"
    assert history["reference"].shape == (n_steps, 3), f"reference: {history['reference'].shape}"
    print("PASS")


def test_process_noise():
    print("\n" + "=" * 60)
    print("Test: process noise causes trajectory variation")
    print("=" * 60)

    np.random.seed(42)

    # Run without noise
    sim_clean = _make_sim(process_noise_std=None)
    sim_clean.reset(np.array([5.0, 0.0, np.pi / 2]))
    ref_fn = _make_ref_fn()
    hist_clean = sim_clean.run(ref_fn, 0.25)

    # Run with noise
    np.random.seed(42)
    sim_noisy = _make_sim(process_noise_std=np.array([0.1, 0.1, 0.05]))
    sim_noisy.reset(np.array([5.0, 0.0, np.pi / 2]))
    hist_noisy = sim_noisy.run(ref_fn, 0.25)

    diff = np.max(np.abs(hist_clean["state"] - hist_noisy["state"]))
    assert diff > 1e-6, f"noise had no effect: diff={diff}"
    print(f"  max state diff = {diff:.6f}")
    print("PASS")


def test_time_increment():
    print("\n" + "=" * 60)
    print("Test: each step increments t by dt")
    print("=" * 60)

    sim = _make_sim()
    sim.reset(np.array([5.0, 0.0, np.pi / 2]))

    for i in range(5):
        ref = generate_reference_trajectory(circle_trajectory, sim.t, N_horizon, dt)
        sim.step(ref)
        expected_t = (i + 1) * dt
        assert abs(sim.t - expected_t) < 1e-10, \
            f"t={sim.t}, expected={expected_t}"

    print(f"  final t={sim.t:.4f}")
    print("PASS")


def test_state_normalization():
    print("\n" + "=" * 60)
    print("Test: DiffDrive angle normalization (wrap to [-pi, pi])")
    print("=" * 60)

    sim = _make_sim()
    # Start near pi boundary
    sim.reset(np.array([0.0, 0.0, np.pi - 0.01]))

    for _ in range(20):
        ref = generate_reference_trajectory(circle_trajectory, sim.t, N_horizon, dt)
        sim.step(ref)

    # Theta should be normalized to [-pi, pi]
    theta = sim.state[2]
    assert -np.pi - 1e-6 <= theta <= np.pi + 1e-6, \
        f"theta out of range: {theta}"
    print(f"  final theta={theta:.4f}")
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Simulator Unit Tests")
    print("=" * 60)

    tests = [
        test_reset,
        test_step_returns_dict,
        test_history_recording,
        test_run_duration,
        test_get_history_shapes,
        test_process_noise,
        test_time_increment,
        test_state_normalization,
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
