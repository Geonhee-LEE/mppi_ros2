#!/usr/bin/env python3
"""
S3: Chasing Evader

목표 지점으로 이동하면서 추적자(predator)를 회피.
Shield-MPPI vs DPCBF vs Gatekeeper 3-way 비교.

Usage:
    python chasing_evading.py
    python chasing_evading.py --live
    python chasing_evading.py --no-plot
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
    MPPIParams,
    CBFMPPIParams,
    ShieldMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.dpcbf_cost import DynamicParabolicCBFCost
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import generate_reference_trajectory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.dynamic_obstacle import DynamicObstacle, ChasingMotion
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


class ChasingEvadingEnv(SimulationEnvironment):
    """추적자 회피 시나리오"""

    def __init__(self, seed: int = 42):
        config = EnvironmentConfig(
            name="S3: Chasing Evader",
            duration=15.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            Qf=np.array([30.0, 30.0, 3.0]),
            seed=seed,
        )
        super().__init__(config)

        # 추적자 장애물 (2개)
        self._chasers = [
            DynamicObstacle(
                ChasingMotion(start=(-4.0, 3.0), speed=0.35),
                radius=0.5,
            ),
            DynamicObstacle(
                ChasingMotion(start=(3.0, -4.0), speed=0.3),
                radius=0.4,
            ),
        ]

        # 목표 경로: 직선 (0,0) → (8,0) → (8,6) → (0,6)
        self._goal_waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([8.0, 0.0, 0.0]),
            np.array([8.0, 6.0, np.pi / 2]),
            np.array([0.0, 6.0, np.pi]),
        ]

        self._controllers = {}

    def get_initial_state(self):
        return np.array([0.0, 0.0, 0.0])

    def get_obstacles(self, t=0.0):
        return [c.as_3tuple(t) for c in self._chasers]

    def get_obstacles_5tuple(self, t=0.0):
        return [c.as_5tuple(t) for c in self._chasers]

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        waypoints = self._goal_waypoints
        total = len(waypoints)

        # 속도 기반 웨이포인트 이동
        speed = 0.6
        seg_dists = []
        for i in range(total - 1):
            d = np.linalg.norm(waypoints[i + 1][:2] - waypoints[i][:2])
            seg_dists.append(d)
        cum_dists = np.concatenate([[0.0], np.cumsum(seg_dists)])
        total_dist = cum_dists[-1]

        def point_at_dist(s):
            s = np.clip(s, 0, total_dist)
            idx = np.searchsorted(cum_dists[1:], s, side="right")
            idx = min(idx, len(seg_dists) - 1)
            local = s - cum_dists[idx]
            seg_len = seg_dists[idx]
            t_ratio = local / max(seg_len, 1e-6)
            p = waypoints[idx] + t_ratio * (waypoints[idx + 1] - waypoints[idx])
            return p

        def ref_fn(t):
            ref = np.zeros((N + 1, 3))
            base_dist = speed * t
            for i in range(N + 1):
                s = base_dist + speed * i * dt
                ref[i] = point_at_dist(s)
            return ref

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
            cbf_weight=1500.0, cbf_alpha=0.3, cbf_safety_margin=0.15,
        )

        configs = []

        # Shield-MPPI
        m1 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.5)
        p1 = ShieldMPPIParams(**common, **cbf_kw, shield_enabled=True)
        c1 = ShieldMPPIController(m1, p1)
        self._controllers["Shield-MPPI"] = c1
        configs.append(ControllerConfig("Shield-MPPI", c1, m1, "#1f77b4"))

        # DPCBF-MPPI
        m2 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.5)
        dpcbf_cost = DynamicParabolicCBFCost(
            obstacles=self.get_obstacles_5tuple(0.0),
            cbf_weight=1500.0,
            safety_margin=0.15,
            dt=c.dt,
        )
        base_costs = [
            StateTrackingCost(c.Q),
            TerminalCost(c.Qf if c.Qf is not None else c.Q),
            ControlEffortCost(c.R),
            dpcbf_cost,
        ]
        composite = CompositeMPPICost(base_costs)
        p2 = CBFMPPIParams(**common, **cbf_kw)
        c2 = CBFMPPIController(m2, p2, cost_function=composite)
        self._controllers["DPCBF-MPPI"] = c2
        self._dpcbf_cost = dpcbf_cost
        configs.append(ControllerConfig("DPCBF-MPPI", c2, m2, "#ff7f0e"))

        # CBF-MPPI (baseline)
        m3 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.5)
        p3 = CBFMPPIParams(**common, **cbf_kw)
        c3 = CBFMPPIController(m3, p3)
        self._controllers["CBF-MPPI"] = c3
        configs.append(ControllerConfig("CBF-MPPI", c3, m3, "#2ca02c"))

        return configs

    def on_step(self, t, states, controls, infos):
        """추적자 목표 업데이트"""
        # 가장 가까운 로봇 위치를 추적 목표로 설정
        robot_positions = []
        for name, state in states.items():
            robot_positions.append(state[:2])

        if robot_positions:
            avg_pos = np.mean(robot_positions, axis=0)
            for chaser in self._chasers:
                chaser.motion.set_target(avg_pos[0], avg_pos[1])

        # 장애물 업데이트
        current_obs_3 = self.get_obstacles(t)
        current_obs_5 = self.get_obstacles_5tuple(t)

        for name, ctrl in self._controllers.items():
            if hasattr(ctrl, "update_obstacles"):
                ctrl.update_obstacles(current_obs_3)

        if hasattr(self, "_dpcbf_cost"):
            self._dpcbf_cost.obstacles = current_obs_5

    def draw_environment(self, ax, t=0.0):
        import matplotlib.pyplot as plt
        for chaser in self._chasers:
            px, py = chaser.get_position(t)
            circle = plt.Circle((px, py), chaser.radius, color="darkred", alpha=0.4)
            ax.add_patch(circle)
            ax.plot(px, py, "r*", markersize=12)

        # 목표 웨이포인트
        for i, wp in enumerate(self._goal_waypoints):
            ax.plot(wp[0], wp[1], "g^", markersize=10, alpha=0.7)
            ax.annotate(f"WP{i}", (wp[0], wp[1]), fontsize=7, ha="center")


def run_scenario(live=False, no_plot=False, seed=42):
    env = ChasingEvadingEnv(seed=seed)
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
            # 추적자 리셋
            for chaser in env._chasers:
                chaser.motion.reset(
                    (chaser.motion._positions[0][0], chaser.motion._positions[0][1])
                )

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
            all_env_metrics[cc.name] = compute_env_metrics(
                history,
                obstacles_fn=lambda t: env.get_obstacles(t),
                goal=np.array([0.0, 6.0]),
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
                save_path="plots/s3_chasing_evading.png",
            )
            plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S3: Chasing Evader")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("S3: Chasing Evader".center(70))
    print("Shield-MPPI vs DPCBF vs CBF-MPPI".center(70))
    print("=" * 70 + "\n")

    run_scenario(live=args.live, no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
