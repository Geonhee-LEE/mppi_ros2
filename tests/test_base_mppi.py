"""
Vanilla MPPI (base_mppi.py) 유닛 테스트
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
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi.sampling import GaussianSampler
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


def _make_controller(**kwargs):
    """헬퍼: 기본 MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(K=64, N=10, dt=0.05, lambda_=1.0,
                    sigma=np.array([0.5, 0.5]),
                    Q=np.array([10.0, 10.0, 1.0]),
                    R=np.array([0.1, 0.1]))
    defaults.update(kwargs)
    params = MPPIParams(**defaults)
    return MPPIController(model, params)


def _make_ref(N=10, dt=0.05):
    """헬퍼: 레퍼런스 궤적 생성"""
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# ── Tests ──────────────────────────────────────────────────


def test_compute_control_shape():
    print("\n" + "=" * 60)
    print("Test: compute_control 반환 shape")
    print("=" * 60)

    ctrl = _make_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"control shape: {control.shape}"
    assert isinstance(info, dict), "info must be dict"
    print("  control shape:", control.shape, "  info keys:", list(info.keys()))
    print("PASS")


def test_weights_sum_to_one():
    print("\n" + "=" * 60)
    print("Test: _compute_weights() sum == 1.0")
    print("=" * 60)

    ctrl = _make_controller()
    costs = np.random.uniform(0, 100, size=(64,))
    weights = ctrl._compute_weights(costs, ctrl.params.lambda_)

    assert weights.shape == (64,), f"weights shape: {weights.shape}"
    assert abs(np.sum(weights) - 1.0) < 1e-6, f"sum={np.sum(weights)}"
    assert np.all(weights >= 0), "weights must be non-negative"
    print(f"  sum={np.sum(weights):.10f}, min={weights.min():.6e}, max={weights.max():.6e}")
    print("PASS")


def test_ess_range():
    print("\n" + "=" * 60)
    print("Test: ESS in [1, K]")
    print("=" * 60)

    ctrl = _make_controller(K=128)
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()
    _, info = ctrl.compute_control(state, ref)

    ess = info["ess"]
    assert 1.0 <= ess <= 128.0, f"ESS={ess} out of [1, 128]"
    print(f"  ESS={ess:.2f}")
    print("PASS")


def test_control_constraints():
    print("\n" + "=" * 60)
    print("Test: control in [u_min, u_max]")
    print("=" * 60)

    ctrl = _make_controller(
        u_min=np.array([-0.5, -0.5]),
        u_max=np.array([0.5, 0.5]),
        sigma=np.array([2.0, 2.0]),
    )
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for _ in range(5):
        control, _ = ctrl.compute_control(state, ref)
        assert np.all(control >= -0.5 - 1e-8), f"below u_min: {control}"
        assert np.all(control <= 0.5 + 1e-8), f"above u_max: {control}"

    print(f"  last control: {control}")
    print("PASS")


def test_receding_horizon_shift():
    print("\n" + "=" * 60)
    print("Test: receding horizon (U shift + last=0)")
    print("=" * 60)

    ctrl = _make_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()
    ctrl.compute_control(state, ref)

    # After compute_control, U has been shifted: last row should be zero
    assert np.allclose(ctrl.U[-1, :], 0.0), f"last U row: {ctrl.U[-1, :]}"
    print(f"  U[-1] = {ctrl.U[-1, :]}")
    print("PASS")


def test_reset():
    print("\n" + "=" * 60)
    print("Test: reset() -> U = zeros")
    print("=" * 60)

    ctrl = _make_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()
    ctrl.compute_control(state, ref)

    ctrl.reset()
    assert np.allclose(ctrl.U, 0.0), "U not zeroed after reset"
    assert ctrl.U.shape == (ctrl.params.N, 2), f"U shape: {ctrl.U.shape}"
    print("PASS")


def test_set_control_sequence():
    print("\n" + "=" * 60)
    print("Test: set_control_sequence (warm start)")
    print("=" * 60)

    ctrl = _make_controller()
    U_warm = np.ones((ctrl.params.N, 2)) * 0.3
    ctrl.set_control_sequence(U_warm)

    assert np.allclose(ctrl.U, 0.3), "warm start not applied"
    # Verify it's a copy (mutation safety)
    U_warm[0, 0] = 999.0
    assert ctrl.U[0, 0] != 999.0, "set_control_sequence must copy"
    print("PASS")


def test_default_cost_function():
    print("\n" + "=" * 60)
    print("Test: cost_function=None -> auto-create CompositeMPPICost")
    print("=" * 60)

    ctrl = _make_controller()
    assert isinstance(ctrl.cost_function, CompositeMPPICost), \
        f"type: {type(ctrl.cost_function)}"
    print(f"  cost_function: {ctrl.cost_function}")
    print("PASS")


def test_default_sampler():
    print("\n" + "=" * 60)
    print("Test: noise_sampler=None -> GaussianSampler")
    print("=" * 60)

    ctrl = _make_controller()
    assert isinstance(ctrl.noise_sampler, GaussianSampler), \
        f"type: {type(ctrl.noise_sampler)}"
    print(f"  noise_sampler: {ctrl.noise_sampler}")
    print("PASS")


def test_numerical_stability():
    print("\n" + "=" * 60)
    print("Test: large costs -> no NaN/Inf")
    print("=" * 60)

    ctrl = _make_controller()
    # Very large costs
    costs = np.array([1e10, 1e12, 1e8, 1e15] * 16)
    weights = ctrl._compute_weights(costs, ctrl.params.lambda_)

    assert not np.any(np.isnan(weights)), "NaN in weights"
    assert not np.any(np.isinf(weights)), "Inf in weights"
    assert abs(np.sum(weights) - 1.0) < 1e-6, f"sum={np.sum(weights)}"
    print(f"  max_cost=1e15, weights sum={np.sum(weights):.10f}")
    print("PASS")


def test_info_dict_keys():
    print("\n" + "=" * 60)
    print("Test: info dict has required keys")
    print("=" * 60)

    ctrl = _make_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()
    _, info = ctrl.compute_control(state, ref)

    required_keys = [
        "sample_trajectories", "sample_weights", "best_trajectory",
        "best_cost", "mean_cost", "temperature", "ess", "num_samples",
    ]
    for key in required_keys:
        assert key in info, f"missing key: {key}"

    assert info["sample_trajectories"].shape == (64, 11, 3), \
        f"trajectories shape: {info['sample_trajectories'].shape}"
    assert info["sample_weights"].shape == (64,), \
        f"weights shape: {info['sample_weights'].shape}"
    assert info["best_trajectory"].shape == (11, 3), \
        f"best_trajectory shape: {info['best_trajectory'].shape}"
    assert info["num_samples"] == 64
    print(f"  keys: {list(info.keys())}")
    print("PASS")


def test_zero_noise():
    print("\n" + "=" * 60)
    print("Test: sigma~0 -> deterministic control")
    print("=" * 60)

    ctrl = _make_controller(sigma=np.array([1e-12, 1e-12]), K=32)
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    c1, _ = ctrl.compute_control(state, ref)
    ctrl.reset()
    c2, _ = ctrl.compute_control(state, ref)

    # With essentially zero noise, both calls should yield near-identical controls
    # (both will be near zero since U starts at zero and noise is zero)
    assert np.allclose(c1, c2, atol=1e-6), f"c1={c1}, c2={c2}"
    print(f"  c1={c1}, c2={c2}")
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Vanilla MPPI (base_mppi) Unit Tests")
    print("=" * 60)

    tests = [
        test_compute_control_shape,
        test_weights_sum_to_one,
        test_ess_range,
        test_control_constraints,
        test_receding_horizon_shift,
        test_reset,
        test_set_control_sequence,
        test_default_cost_function,
        test_default_sampler,
        test_numerical_stability,
        test_info_dict_keys,
        test_zero_noise,
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
