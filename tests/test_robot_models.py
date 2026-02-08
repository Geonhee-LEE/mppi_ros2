"""
Robot Model Tests

Tests for Ackermann Kinematic/Dynamic and Swerve Drive Kinematic/Dynamic models.
Covers: dimensions, dynamics equations, batch processing, RK4 integration,
normalize_state, control bounds, MPPI controller integration, and simulation.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.models.dynamic.ackermann_dynamic import AckermannDynamic
from mppi_controller.models.kinematic.swerve_drive_kinematic import SwerveDriveKinematic
from mppi_controller.models.dynamic.swerve_drive_dynamic import SwerveDriveDynamic
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ═══════════════════════════════════════════════════════
#  Ackermann Kinematic Tests
# ═══════════════════════════════════════════════════════

class TestAckermannKinematic:
    def setup_method(self):
        self.model = AckermannKinematic(
            wheelbase=0.5, v_max=1.0, max_steer=0.5, steer_rate_max=1.0
        )

    def test_dimensions(self):
        assert self.model.state_dim == 4
        assert self.model.control_dim == 2

    def test_model_type(self):
        assert self.model.model_type == "kinematic"

    def test_is_robot_model(self):
        assert isinstance(self.model, RobotModel)

    def test_forward_dynamics_straight(self):
        """v=1, delta=0 -> pure forward motion"""
        state = np.array([0.0, 0.0, 0.0, 0.0])  # theta=0, delta=0
        control = np.array([1.0, 0.0])  # v=1, phi=0
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_forward_dynamics_turning(self):
        """v=1, delta=0.3 -> turning"""
        state = np.array([0.0, 0.0, 0.0, 0.3])
        control = np.array([1.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        assert dot[0] == pytest.approx(1.0, abs=1e-10)  # x_dot = cos(0)
        assert dot[1] == pytest.approx(0.0, abs=1e-10)  # y_dot = sin(0)
        expected_theta_dot = np.tan(0.3) / 0.5
        assert dot[2] == pytest.approx(expected_theta_dot, abs=1e-10)
        assert dot[3] == pytest.approx(0.0, abs=1e-10)

    def test_forward_dynamics_steering_rate(self):
        """phi != 0 changes delta"""
        state = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.5])  # phi=0.5
        dot = self.model.forward_dynamics(state, control)
        assert dot[3] == pytest.approx(0.5, abs=1e-10)

    def test_batch_dynamics(self):
        """Batch (K, 4) input"""
        K = 100
        states = np.random.randn(K, 4) * 0.1
        controls = np.random.randn(K, 2) * 0.1
        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 4)

    def test_rk4_step(self):
        """RK4 integration moves the state"""
        state = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0])
        dt = 0.05
        next_state = self.model.step(state, control, dt)
        assert next_state.shape == (4,)
        assert next_state[0] > 0  # moved forward

    def test_normalize_state_theta(self):
        """theta wraps to [-pi, pi]"""
        state = np.array([0.0, 0.0, 4.0, 0.0])
        norm = self.model.normalize_state(state)
        assert -np.pi <= norm[2] <= np.pi

    def test_normalize_state_delta_clip(self):
        """delta clipped to [-max_steer, max_steer]"""
        state = np.array([0.0, 0.0, 0.0, 1.0])  # delta=1.0 > max_steer=0.5
        norm = self.model.normalize_state(state)
        assert norm[3] == pytest.approx(0.5, abs=1e-10)

    def test_normalize_state_batch(self):
        states = np.array([[0.0, 0.0, 4.0, 1.0], [0.0, 0.0, -4.0, -1.0]])
        norm = self.model.normalize_state(states)
        assert norm.shape == (2, 4)
        assert all(-np.pi <= norm[i, 2] <= np.pi for i in range(2))
        assert all(abs(norm[i, 3]) <= 0.5 + 1e-10 for i in range(2))

    def test_control_bounds(self):
        lb, ub = self.model.get_control_bounds()
        np.testing.assert_allclose(lb, [-1.0, -1.0])
        np.testing.assert_allclose(ub, [1.0, 1.0])

    def test_state_to_dict(self):
        state = np.array([1.0, 2.0, 0.5, 0.1])
        d = self.model.state_to_dict(state)
        assert d["x"] == 1.0
        assert d["delta"] == 0.1

    def test_repr(self):
        r = repr(self.model)
        assert "AckermannKinematic" in r
        assert "wheelbase" in r

    def test_xy_at_indices_0_1(self):
        """x, y always at state[0], state[1] for cost function compatibility"""
        state = np.array([3.0, 4.0, 0.1, 0.2])
        assert state[0] == 3.0
        assert state[1] == 4.0


# ═══════════════════════════════════════════════════════
#  Ackermann Dynamic Tests
# ═══════════════════════════════════════════════════════

class TestAckermannDynamic:
    def setup_method(self):
        self.model = AckermannDynamic(
            wheelbase=0.5, mass=10.0, c_v=0.1,
            v_max=2.0, a_max=2.0, max_steer=0.5, steer_rate_max=1.0,
        )

    def test_dimensions(self):
        assert self.model.state_dim == 5
        assert self.model.control_dim == 2

    def test_model_type(self):
        assert self.model.model_type == "dynamic"

    def test_is_robot_model(self):
        assert isinstance(self.model, RobotModel)

    def test_forward_dynamics_zero(self):
        """Zero state, zero control -> zero derivatives"""
        state = np.zeros(5)
        control = np.zeros(2)
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, np.zeros(5), atol=1e-10)

    def test_forward_dynamics_acceleration(self):
        """Acceleration with friction"""
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # v=1
        control = np.array([2.0, 0.0])  # a=2
        dot = self.model.forward_dynamics(state, control)
        # v_dot = a - c_v * v = 2.0 - 0.1*1.0 = 1.9
        assert dot[3] == pytest.approx(1.9, abs=1e-10)

    def test_forward_dynamics_friction_deceleration(self):
        """No acceleration, friction slows velocity"""
        state = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
        control = np.zeros(2)
        dot = self.model.forward_dynamics(state, control)
        assert dot[3] < 0  # friction decelerates

    def test_batch_dynamics(self):
        K = 100
        states = np.random.randn(K, 5) * 0.1
        controls = np.random.randn(K, 2) * 0.1
        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 5)

    def test_rk4_step(self):
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0])
        dt = 0.05
        next_state = self.model.step(state, control, dt)
        assert next_state.shape == (5,)
        assert next_state[3] > 0  # velocity increased

    def test_normalize_state(self):
        state = np.array([0.0, 0.0, 4.0, 5.0, 1.0])
        norm = self.model.normalize_state(state)
        assert -np.pi <= norm[2] <= np.pi
        assert abs(norm[3]) <= 2.0 + 1e-10
        assert abs(norm[4]) <= 0.5 + 1e-10

    def test_control_bounds(self):
        lb, ub = self.model.get_control_bounds()
        np.testing.assert_allclose(lb, [-2.0, -1.0])
        np.testing.assert_allclose(ub, [2.0, 1.0])

    def test_compute_energy(self):
        state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
        energy = self.model.compute_energy(state)
        assert energy == pytest.approx(0.5 * 10.0 * 4.0)

    def test_state_to_dict(self):
        state = np.array([1.0, 2.0, 0.5, 0.3, 0.1])
        d = self.model.state_to_dict(state)
        assert d["v"] == 0.3
        assert d["delta"] == 0.1


# ═══════════════════════════════════════════════════════
#  Swerve Drive Kinematic Tests
# ═══════════════════════════════════════════════════════

class TestSwerveDriveKinematic:
    def setup_method(self):
        self.model = SwerveDriveKinematic(
            vx_max=1.0, vy_max=1.0, omega_max=1.0
        )

    def test_dimensions(self):
        assert self.model.state_dim == 3
        assert self.model.control_dim == 3

    def test_model_type(self):
        assert self.model.model_type == "kinematic"

    def test_is_robot_model(self):
        assert isinstance(self.model, RobotModel)

    def test_forward_dynamics_forward(self):
        """vx=1, theta=0 -> move in +x"""
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [1.0, 0.0, 0.0], atol=1e-10)

    def test_forward_dynamics_lateral(self):
        """vy=1, theta=0 -> move in +y (body lateral)"""
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([0.0, 1.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [0.0, 1.0, 0.0], atol=1e-10)

    def test_forward_dynamics_rotated(self):
        """vx=1, theta=pi/2 -> move in +y (world)"""
        state = np.array([0.0, 0.0, np.pi / 2])
        control = np.array([1.0, 0.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [0.0, 1.0, 0.0], atol=1e-10)

    def test_forward_dynamics_rotation(self):
        """omega=1 -> pure rotation"""
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 1.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [0.0, 0.0, 1.0], atol=1e-10)

    def test_forward_dynamics_diagonal(self):
        """vx=1, vy=1, theta=0 -> move diagonally"""
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([1.0, 1.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [1.0, 1.0, 0.0], atol=1e-10)

    def test_batch_dynamics(self):
        K = 100
        states = np.random.randn(K, 3)
        controls = np.random.randn(K, 3)
        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 3)

    def test_rk4_step(self):
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.0])
        next_state = self.model.step(state, control, 0.05)
        assert next_state[0] > 0

    def test_normalize_state(self):
        state = np.array([0.0, 0.0, 4.0])
        norm = self.model.normalize_state(state)
        assert -np.pi <= norm[2] <= np.pi

    def test_control_bounds(self):
        lb, ub = self.model.get_control_bounds()
        np.testing.assert_allclose(lb, [-1.0, -1.0, -1.0])
        np.testing.assert_allclose(ub, [1.0, 1.0, 1.0])

    def test_state_to_dict(self):
        state = np.array([1.0, 2.0, 0.5])
        d = self.model.state_to_dict(state)
        assert d["x"] == 1.0
        assert d["theta"] == 0.5


# ═══════════════════════════════════════════════════════
#  Swerve Drive Dynamic Tests
# ═══════════════════════════════════════════════════════

class TestSwerveDriveDynamic:
    def setup_method(self):
        self.model = SwerveDriveDynamic(
            mass=10.0, inertia=1.0, c_v=0.1, c_omega=0.1,
            vx_max=2.0, vy_max=2.0, omega_max=2.0,
            ax_max=2.0, ay_max=2.0, alpha_max=2.0,
        )

    def test_dimensions(self):
        assert self.model.state_dim == 6
        assert self.model.control_dim == 3

    def test_model_type(self):
        assert self.model.model_type == "dynamic"

    def test_is_robot_model(self):
        assert isinstance(self.model, RobotModel)

    def test_forward_dynamics_zero(self):
        state = np.zeros(6)
        control = np.zeros(3)
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, np.zeros(6), atol=1e-10)

    def test_forward_dynamics_acceleration(self):
        """ax=1 with vx=1 -> vx_dot = 1 - 0.1*1 = 0.9"""
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        assert dot[3] == pytest.approx(0.9, abs=1e-10)
        assert dot[0] == pytest.approx(1.0, abs=1e-10)  # x_dot = vx*cos(0)

    def test_forward_dynamics_lateral_friction(self):
        """vy nonzero -> friction on vy"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
        control = np.zeros(3)
        dot = self.model.forward_dynamics(state, control)
        assert dot[4] == pytest.approx(-0.2, abs=1e-10)  # -c_v * vy

    def test_forward_dynamics_angular_friction(self):
        """omega nonzero -> friction on omega"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
        control = np.zeros(3)
        dot = self.model.forward_dynamics(state, control)
        assert dot[5] == pytest.approx(-0.3, abs=1e-10)  # -c_omega * omega

    def test_batch_dynamics(self):
        K = 100
        states = np.random.randn(K, 6) * 0.1
        controls = np.random.randn(K, 3) * 0.1
        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 6)

    def test_rk4_step(self):
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.0])
        next_state = self.model.step(state, control, 0.05)
        assert next_state[3] > 0  # velocity increased

    def test_normalize_state(self):
        state = np.array([0.0, 0.0, 4.0, 5.0, -5.0, 5.0])
        norm = self.model.normalize_state(state)
        assert -np.pi <= norm[2] <= np.pi
        assert abs(norm[3]) <= 2.0 + 1e-10
        assert abs(norm[4]) <= 2.0 + 1e-10
        assert abs(norm[5]) <= 2.0 + 1e-10

    def test_control_bounds(self):
        lb, ub = self.model.get_control_bounds()
        np.testing.assert_allclose(lb, [-2.0, -2.0, -2.0])
        np.testing.assert_allclose(ub, [2.0, 2.0, 2.0])

    def test_compute_energy(self):
        state = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
        energy = self.model.compute_energy(state)
        # 0.5*10*(1+1) + 0.5*1*4 = 10 + 2 = 12
        assert energy == pytest.approx(12.0, abs=1e-10)

    def test_state_to_dict(self):
        state = np.array([1.0, 2.0, 0.5, 0.3, 0.4, 0.5])
        d = self.model.state_to_dict(state)
        assert d["vx"] == 0.3
        assert d["omega"] == 0.5


# ═══════════════════════════════════════════════════════
#  BatchDynamicsWrapper Integration
# ═══════════════════════════════════════════════════════

class TestBatchDynamicsWrapper:
    @pytest.mark.parametrize("model_cls,state_dim,ctrl_dim,kwargs", [
        (AckermannKinematic, 4, 2, {}),
        (AckermannDynamic, 5, 2, {}),
        (SwerveDriveKinematic, 3, 3, {}),
        (SwerveDriveDynamic, 6, 3, {}),
    ])
    def test_wrapper_rollout(self, model_cls, state_dim, ctrl_dim, kwargs):
        """BatchDynamicsWrapper produces correct trajectory shape"""
        model = model_cls(**kwargs)
        wrapper = BatchDynamicsWrapper(model, dt=0.05)
        K, N = 64, 20
        initial_state = np.zeros(state_dim)
        controls = np.random.randn(K, N, ctrl_dim) * 0.1
        trajectories = wrapper.rollout(initial_state, controls)
        assert trajectories.shape == (K, N + 1, state_dim)
        # First state matches initial
        np.testing.assert_allclose(
            trajectories[:, 0, :], np.tile(initial_state, (K, 1))
        )


# ═══════════════════════════════════════════════════════
#  MPPI Controller Integration
# ═══════════════════════════════════════════════════════

class TestMPPIIntegration:
    def _make_ref(self, N, nx):
        """Create a simple circle reference with correct state dimension."""
        trajectory_fn = create_trajectory_function("circle")
        ref_3d = generate_reference_trajectory(trajectory_fn, 0.0, N, 0.05)
        if nx == 3:
            return ref_3d
        # Pad extra dims with zeros
        ref = np.zeros((N + 1, nx))
        ref[:, :3] = ref_3d
        return ref

    def test_ackermann_kinematic_mppi(self):
        model = AckermannKinematic(wheelbase=0.5, v_max=1.0, max_steer=0.5)
        params = MPPIParams(
            N=20, dt=0.05, K=128, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
        )
        controller = MPPIController(model, params)
        state = np.array([5.0, 0.0, np.pi / 2, 0.0])
        ref = self._make_ref(20, 4)
        control, info = controller.compute_control(state, ref)
        assert control.shape == (2,)
        assert "sample_trajectories" in info

    def test_ackermann_dynamic_mppi(self):
        model = AckermannDynamic(wheelbase=0.5, a_max=2.0)
        params = MPPIParams(
            N=20, dt=0.05, K=128, lambda_=1.0,
            sigma=np.array([1.0, 0.5]),
            Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
            R=np.array([0.1, 0.1]),
        )
        controller = MPPIController(model, params)
        state = np.array([5.0, 0.0, np.pi / 2, 0.0, 0.0])
        ref = self._make_ref(20, 5)
        control, info = controller.compute_control(state, ref)
        assert control.shape == (2,)

    def test_swerve_kinematic_mppi(self):
        model = SwerveDriveKinematic(vx_max=1.0, vy_max=1.0, omega_max=1.0)
        params = MPPIParams(
            N=20, dt=0.05, K=128, lambda_=1.0,
            sigma=np.array([0.5, 0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1, 0.1]),
        )
        controller = MPPIController(model, params)
        state = np.array([5.0, 0.0, np.pi / 2])
        ref = self._make_ref(20, 3)
        control, info = controller.compute_control(state, ref)
        assert control.shape == (3,)

    def test_swerve_dynamic_mppi(self):
        model = SwerveDriveDynamic()
        params = MPPIParams(
            N=20, dt=0.05, K=128, lambda_=1.0,
            sigma=np.array([1.0, 1.0, 1.0]),
            Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1, 0.1]),
            R=np.array([0.1, 0.1, 0.1]),
        )
        controller = MPPIController(model, params)
        state = np.array([5.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])
        ref = self._make_ref(20, 6)
        control, info = controller.compute_control(state, ref)
        assert control.shape == (3,)


# ═══════════════════════════════════════════════════════
#  Simulation Integration
# ═══════════════════════════════════════════════════════

class TestSimulation:
    def _run_simulation(self, model, params, initial_state, ref_fn, duration=5.0):
        controller = MPPIController(model, params)
        sim = Simulator(model, controller, params.dt)
        sim.reset(initial_state)
        history = sim.run(ref_fn, duration, realtime=False)
        return history

    def test_ackermann_kinematic_simulation(self):
        model = AckermannKinematic(wheelbase=0.5, v_max=1.0)
        params = MPPIParams(
            N=20, dt=0.05, K=256, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
        )
        traj_fn = create_trajectory_function("circle")

        def ref_fn(t):
            ref_3d = generate_reference_trajectory(traj_fn, t, 20, 0.05)
            ref = np.zeros((21, 4))
            ref[:, :3] = ref_3d
            return ref

        initial = np.array([5.0, 0.0, np.pi / 2, 0.0])
        history = self._run_simulation(model, params, initial, ref_fn, duration=3.0)
        assert len(history["time"]) > 10
        # Check no NaN
        assert not np.any(np.isnan(history["state"]))

    def test_swerve_kinematic_simulation(self):
        model = SwerveDriveKinematic(vx_max=1.0, vy_max=1.0, omega_max=1.0)
        params = MPPIParams(
            N=20, dt=0.05, K=256, lambda_=1.0,
            sigma=np.array([0.5, 0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1, 0.1]),
        )
        traj_fn = create_trajectory_function("circle")

        def ref_fn(t):
            return generate_reference_trajectory(traj_fn, t, 20, 0.05)

        initial = np.array([5.0, 0.0, np.pi / 2])
        history = self._run_simulation(model, params, initial, ref_fn, duration=3.0)
        assert len(history["time"]) > 10
        assert not np.any(np.isnan(history["state"]))


# ═══════════════════════════════════════════════════════
#  GPU Torch Model Tests (CPU fallback)
# ═══════════════════════════════════════════════════════

class TestGPUTorchModels:
    """Test torch model factory with CPU (CUDA optional)."""

    def test_get_torch_model_ackermann_kinematic(self):
        from mppi_controller.controllers.mppi.gpu import get_torch_model
        model = AckermannKinematic(wheelbase=0.5)
        torch_model = get_torch_model(model, device="cpu")
        assert torch_model.wheelbase == 0.5

    def test_get_torch_model_ackermann_dynamic(self):
        from mppi_controller.controllers.mppi.gpu import get_torch_model
        model = AckermannDynamic(wheelbase=0.5, c_v=0.2)
        torch_model = get_torch_model(model, device="cpu")
        assert torch_model.wheelbase == 0.5
        assert torch_model.c_v == 0.2

    def test_get_torch_model_swerve_kinematic(self):
        from mppi_controller.controllers.mppi.gpu import get_torch_model
        model = SwerveDriveKinematic()
        torch_model = get_torch_model(model, device="cpu")
        assert torch_model is not None

    def test_get_torch_model_swerve_dynamic(self):
        from mppi_controller.controllers.mppi.gpu import get_torch_model
        model = SwerveDriveDynamic(c_v=0.2, c_omega=0.3)
        torch_model = get_torch_model(model, device="cpu")
        assert torch_model.c_v == 0.2
        assert torch_model.c_omega == 0.3

    def test_torch_ackermann_kinematic_forward(self):
        import torch
        from mppi_controller.controllers.mppi.gpu.torch_models import TorchAckermannKinematic
        tm = TorchAckermannKinematic(wheelbase=0.5, device="cpu")
        state = torch.zeros(10, 4)
        control = torch.ones(10, 2)
        dot = tm.forward_dynamics(state, control)
        assert dot.shape == (10, 4)

    def test_torch_swerve_dynamic_rk4(self):
        import torch
        from mppi_controller.controllers.mppi.gpu.torch_models import TorchSwerveDriveDynamic
        tm = TorchSwerveDriveDynamic(device="cpu")
        state = torch.zeros(10, 6)
        control = torch.ones(10, 3) * 0.1
        next_state = tm.step_rk4(state, control, 0.05)
        assert next_state.shape == (10, 6)
        assert torch.all(next_state[:, 3] > 0)  # vx increased


# ═══════════════════════════════════════════════════════
#  Standalone runner
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    test_classes = [
        TestAckermannKinematic,
        TestAckermannDynamic,
        TestSwerveDriveKinematic,
        TestSwerveDriveDynamic,
        TestBatchDynamicsWrapper,
        TestMPPIIntegration,
        TestSimulation,
        TestGPUTorchModels,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for attr in dir(instance):
            if not attr.startswith("test_"):
                continue
            if hasattr(instance, "setup_method"):
                instance.setup_method()
            try:
                method = getattr(instance, attr)
                # Skip parametrized tests in standalone mode
                if hasattr(method, "pytestmark"):
                    continue
                method()
                passed += 1
                print(f"  PASS: {cls.__name__}.{attr}")
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{attr}", str(e)))
                print(f"  FAIL: {cls.__name__}.{attr} -> {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("Failed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
