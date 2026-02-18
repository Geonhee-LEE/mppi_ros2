#!/usr/bin/env python3
"""
S2: Dynamic Bouncing Obstacles

원형 궤적을 추적하면서 4-6개의 반사 장애물을 회피.
CBF (standard) vs C3BF (velocity-aware collision cone) 비교.

Usage:
    python dynamic_bouncing.py
    python dynamic_bouncing.py --live
    python dynamic_bouncing.py --no-plot
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    CBFMPPIParams,
    ShieldMPPIParams,
)
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.c3bf_cost import CollisionConeCBFCost
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.dynamic_obstacle import DynamicObstacle, BouncingMotion
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


class DynamicBouncingEnv(SimulationEnvironment):
    """반사 장애물 시나리오"""

    def __init__(self, n_obstacles: int = 5, seed: int = 42):
        config = EnvironmentConfig(
            name=f"S2: Dynamic Bouncing ({n_obstacles} obstacles)",
            duration=15.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            Qf=np.array([20.0, 20.0, 2.0]),
            seed=seed,
        )
        super().__init__(config)

        rng = np.random.RandomState(seed)
        self._dynamic_obstacles = []
        bounds = (-7.0, -7.0, 7.0, 7.0)

        for _ in range(n_obstacles):
            start = (rng.uniform(-5, 5), rng.uniform(-5, 5))
            speed = rng.uniform(0.3, 0.8)
            angle = rng.uniform(0, 2 * np.pi)
            vel = (speed * np.cos(angle), speed * np.sin(angle))
            radius = rng.uniform(0.3, 0.6)
            motion = BouncingMotion(start, vel, bounds)
            self._dynamic_obstacles.append(DynamicObstacle(motion, radius))

        self._trajectory_fn = create_trajectory_function("circle", radius=5.0)

        # 컨트롤러 참조 (on_step에서 업데이트용)
        self._controllers = {}

    def get_initial_state(self):
        return self._trajectory_fn(0.0)

    def get_obstacles(self, t=0.0):
        return [obs.as_3tuple(t) for obs in self._dynamic_obstacles]

    def get_obstacles_5tuple(self, t=0.0):
        return [obs.as_5tuple(t) for obs in self._dynamic_obstacles]

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        traj_fn = self._trajectory_fn

        def ref_fn(t):
            return generate_reference_trajectory(traj_fn, t, N, dt)
        return ref_fn

    def get_controller_configs(self):
        c = self.config
        common = dict(
            N=c.N, dt=c.dt, K=c.K, lambda_=c.lambda_,
            sigma=c.sigma, Q=c.Q, R=c.R, Qf=c.Qf,
        )
        initial_obs = self.get_obstacles(0.0)
        cbf_kw = dict(
            cbf_obstacles=initial_obs,
            cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        )

        configs = []

        # CBF-MPPI (standard)
        m1 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p1 = CBFMPPIParams(**common, **cbf_kw)
        c1 = CBFMPPIController(m1, p1)
        self._controllers["CBF-MPPI"] = c1
        configs.append(ControllerConfig("CBF-MPPI", c1, m1, "#1f77b4"))

        # C3BF (velocity-aware collision cone)
        m2 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        c3bf_cost = CollisionConeCBFCost(
            obstacles=self.get_obstacles_5tuple(0.0),
            cbf_alpha=0.3,
            cbf_weight=1000.0,
            safety_margin=0.1,
            dt=c.dt,
        )
        base_costs = [
            StateTrackingCost(c.Q),
            TerminalCost(c.Qf if c.Qf is not None else c.Q),
            ControlEffortCost(c.R),
            c3bf_cost,
        ]
        composite_cost = CompositeMPPICost(base_costs)
        p2 = CBFMPPIParams(**common, **cbf_kw)
        c2 = CBFMPPIController(m2, p2, cost_function=composite_cost)
        self._controllers["C3BF-MPPI"] = c2
        self._c3bf_cost = c3bf_cost
        configs.append(ControllerConfig("C3BF-MPPI", c2, m2, "#ff7f0e"))

        # Shield-MPPI
        m3 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p3 = ShieldMPPIParams(**common, **cbf_kw, shield_enabled=True)
        c3 = ShieldMPPIController(m3, p3)
        self._controllers["Shield-MPPI"] = c3
        configs.append(ControllerConfig("Shield-MPPI", c3, m3, "#2ca02c"))

        return configs

    def on_step(self, t, states, controls, infos):
        """매 스텝: 동적 장애물 업데이트"""
        current_obs_3 = self.get_obstacles(t)
        current_obs_5 = self.get_obstacles_5tuple(t)

        for name, ctrl in self._controllers.items():
            if hasattr(ctrl, "update_obstacles"):
                ctrl.update_obstacles(current_obs_3)

        if hasattr(self, "_c3bf_cost"):
            self._c3bf_cost.obstacles = current_obs_5

    def draw_environment(self, ax, t=0.0):
        import matplotlib.pyplot as plt
        for obs in self._dynamic_obstacles:
            px, py = obs.get_position(t)
            circle = plt.Circle((px, py), obs.radius, color="red", alpha=0.3)
            ax.add_patch(circle)
            vx, vy = obs.get_velocity(t)
            ax.arrow(px, py, vx * 0.5, vy * 0.5,
                     head_width=0.1, head_length=0.05, fc="red", ec="red", alpha=0.5)


def run_scenario(live=False, no_plot=False, seed=42, n_obstacles=5):
    env = DynamicBouncingEnv(n_obstacles=n_obstacles, seed=seed)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    initial_state = env.get_initial_state()
    duration = env.config.duration
    dt = env.config.dt

    if live:
        simulators = {}
        ref_fns = {}
        colors = {}
        for cc in configs:
            sim = Simulator(cc.model, cc.controller, dt)
            sim.reset(initial_state)
            simulators[cc.name] = sim
            ref_fns[cc.name] = ref_fn
            colors[cc.name] = cc.color

        viz = EnvVisualizer(env)
        viz.run_and_animate(simulators, ref_fns, duration,
                            controller_colors=colors)
    else:
        if no_plot:
            matplotlib.use("Agg")

        histories = {}
        all_metrics = {}
        all_env_metrics = {}
        colors = {}

        for cc in configs:
            print(f"  Running {cc.name}...")
            sim = Simulator(cc.model, cc.controller, dt)
            sim.reset(initial_state)

            num_steps = int(duration / dt)
            for step in range(num_steps):
                t = sim.t
                env.on_step(t, {cc.name: sim.state}, {}, {})
                ref_traj = ref_fn(t)
                sim.step(ref_traj)

            history = sim.get_history()
            histories[cc.name] = history
            all_metrics[cc.name] = compute_metrics(history)

            def make_obs_fn(env_ref):
                return lambda t: env_ref.get_obstacles(t)

            all_env_metrics[cc.name] = compute_env_metrics(
                history, obstacles_fn=make_obs_fn(env),
            )
            colors[cc.name] = cc.color

        for name in histories:
            print_metrics(all_metrics[name], title=name)
        print_env_comparison(all_env_metrics, title=env.name)

        if not no_plot:
            viz = EnvVisualizer(env)
            fig = viz.run_and_plot(
                histories, all_metrics, all_env_metrics,
                controller_colors=colors,
                save_path="plots/s2_dynamic_bouncing.png",
            )
            plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S2: Dynamic Bouncing Obstacles")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-obstacles", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("S2: Dynamic Bouncing Obstacles".center(70))
    print("CBF vs C3BF vs Shield-MPPI".center(70))
    print("=" * 70 + "\n")

    run_scenario(live=args.live, no_plot=args.no_plot,
                 seed=args.seed, n_obstacles=args.n_obstacles)


if __name__ == "__main__":
    main()
