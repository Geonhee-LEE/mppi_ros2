"""
배치 동역학 래퍼 (dynamics_wrapper.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper

K, N = 32, 10
dt = 0.05


def _make_wrapper():
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    return BatchDynamicsWrapper(model, dt)


# ── Tests ──────────────────────────────────────────────────


def test_rollout_shape():
    print("\n" + "=" * 60)
    print("Test: rollout output shape (K, N+1, nx)")
    print("=" * 60)

    wrapper = _make_wrapper()
    state = np.array([0.0, 0.0, 0.0])
    controls = np.random.randn(K, N, 2) * 0.3

    traj = wrapper.rollout(state, controls)
    assert traj.shape == (K, N + 1, 3), f"shape: {traj.shape}"
    print(f"  shape={traj.shape}")
    print("PASS")


def test_initial_state_broadcast():
    print("\n" + "=" * 60)
    print("Test: traj[:, 0, :] == initial_state for all K")
    print("=" * 60)

    wrapper = _make_wrapper()
    state = np.array([1.0, 2.0, 0.5])
    controls = np.random.randn(K, N, 2) * 0.3

    traj = wrapper.rollout(state, controls)
    for k in range(K):
        assert np.allclose(traj[k, 0, :], state), \
            f"sample {k}: {traj[k, 0, :]} != {state}"
    print("PASS")


def test_single_rollout_shape():
    print("\n" + "=" * 60)
    print("Test: single_rollout output shape (N+1, nx)")
    print("=" * 60)

    wrapper = _make_wrapper()
    state = np.array([0.0, 0.0, 0.0])
    controls = np.random.randn(N, 2) * 0.3

    traj = wrapper.single_rollout(state, controls)
    assert traj.shape == (N + 1, 3), f"shape: {traj.shape}"
    assert np.allclose(traj[0], state), f"initial: {traj[0]}"
    print("PASS")


def test_zero_control():
    print("\n" + "=" * 60)
    print("Test: u=0 -> DiffDrive stays put")
    print("=" * 60)

    wrapper = _make_wrapper()
    state = np.array([1.0, 2.0, 0.5])
    controls = np.zeros((K, N, 2))

    traj = wrapper.rollout(state, controls)
    for k in range(K):
        for t in range(N + 1):
            assert np.allclose(traj[k, t, :], state, atol=1e-10), \
                f"moved with zero control at k={k}, t={t}: {traj[k, t, :]}"
    print("PASS")


def test_forward_dynamics_batch():
    print("\n" + "=" * 60)
    print("Test: forward_dynamics_batch (K, nx) output")
    print("=" * 60)

    wrapper = _make_wrapper()
    states = np.random.randn(K, 3)
    controls = np.random.randn(K, 2)

    state_dots = wrapper.forward_dynamics_batch(states, controls)
    assert state_dots.shape == (K, 3), f"shape: {state_dots.shape}"

    # dx/dt = v*cos(theta), so check first element
    expected_xdot = controls[:, 0] * np.cos(states[:, 2])
    assert np.allclose(state_dots[:, 0], expected_xdot, atol=1e-10), "xdot mismatch"
    print("PASS")


def test_different_models():
    print("\n" + "=" * 60)
    print("Test: wrapper works with Ackermann model")
    print("=" * 60)

    try:
        from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
        model = AckermannKinematic()
        wrapper = BatchDynamicsWrapper(model, dt)

        state = np.zeros(model.state_dim)
        controls = np.random.randn(K, N, model.control_dim) * 0.1

        traj = wrapper.rollout(state, controls)
        assert traj.shape == (K, N + 1, model.state_dim), f"shape: {traj.shape}"
        print(f"  Ackermann traj shape: {traj.shape}")
    except ImportError:
        print("  (Ackermann model not available, skipping)")
    print("PASS")


def test_consistency():
    print("\n" + "=" * 60)
    print("Test: single_rollout == rollout[0] for same control")
    print("=" * 60)

    wrapper = _make_wrapper()
    state = np.array([0.0, 0.0, 0.0])
    single_ctrl = np.random.randn(N, 2) * 0.3

    single_traj = wrapper.single_rollout(state, single_ctrl)

    # Batch rollout with K=1
    batch_ctrl = single_ctrl[np.newaxis, :, :]  # (1, N, 2)
    batch_traj = wrapper.rollout(state, batch_ctrl)

    assert np.allclose(single_traj, batch_traj[0], atol=1e-10), \
        f"max diff: {np.max(np.abs(single_traj - batch_traj[0]))}"
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BatchDynamicsWrapper Unit Tests")
    print("=" * 60)

    tests = [
        test_rollout_shape,
        test_initial_state_broadcast,
        test_single_rollout_shape,
        test_zero_control,
        test_forward_dynamics_batch,
        test_different_models,
        test_consistency,
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
