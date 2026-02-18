#!/usr/bin/env python3
"""
S1: Static Obstacle Field

정적 장애물 필드를 통과하는 시나리오.
Vanilla MPPI vs CBF-MPPI vs Shield-MPPI 3-way 비교.

Usage:
    python static_obstacle_field.py
    python static_obstacle_field.py --live
    python static_obstacle_field.py --layout slalom
    python static_obstacle_field.py --no-plot
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
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.obstacle_field import generate_random_field, generate_slalom, generate_funnel
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


class StaticObstacleFieldEnv(SimulationEnvironment):
    """정적 장애물 필드 시나리오"""

    def __init__(self, layout: str = "random", seed: int = 42):
        config = EnvironmentConfig(
            name=f"S1: Static Obstacle Field ({layout})",
            duration=20.0,
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
        self.layout = layout
        self._obstacles = self._generate_obstacles(seed)
        self._trajectory_fn = create_trajectory_function("circle", radius=5.0)

    def _generate_obstacles(self, seed):
        if self.layout == "random":
            return generate_random_field(
                n=15,
                x_range=(-6.0, 6.0),
                y_range=(-6.0, 6.0),
                radius_range=(0.3, 0.6),
                exclusion_zones=[
                    (5.0, 0.0, 1.5),   # start region
                    (-5.0, 0.0, 1.5),
                    (0.0, 5.0, 1.5),
                    (0.0, -5.0, 1.5),
                ],
                seed=seed,
            )
        elif self.layout == "slalom":
            obs, _ = generate_slalom(
                n_gates=6,
                spacing=2.0,
                gate_width=1.8,
                pillar_radius=0.3,
                start_x=-5.0,
                start_y=0.0,
                direction=0.0,
                alternating_offset=0.8,
            )
            return obs
        elif self.layout == "dense":
            return generate_random_field(
                n=25,
                x_range=(-6.0, 6.0),
                y_range=(-6.0, 6.0),
                radius_range=(0.2, 0.4),
                exclusion_zones=[
                    (5.0, 0.0, 1.2),
                    (-5.0, 0.0, 1.2),
                    (0.0, 5.0, 1.2),
                    (0.0, -5.0, 1.2),
                ],
                min_spacing=0.2,
                seed=seed,
            )
        return []

    def get_initial_state(self):
        return self._trajectory_fn(0.0)

    def get_obstacles(self, t=0.0):
        return self._obstacles

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
        cbf_kw = dict(
            cbf_obstacles=self._obstacles,
            cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        )

        configs = []
        # Vanilla
        m1 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p1 = MPPIParams(**common)
        c1 = MPPIController(m1, p1)
        configs.append(ControllerConfig("Vanilla", c1, m1, "#1f77b4"))

        # CBF-MPPI
        m2 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p2 = CBFMPPIParams(**common, **cbf_kw)
        c2 = CBFMPPIController(m2, p2)
        configs.append(ControllerConfig("CBF-MPPI", c2, m2, "#ff7f0e"))

        # Shield-MPPI
        m3 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p3 = ShieldMPPIParams(**common, **cbf_kw, shield_enabled=True)
        c3 = ShieldMPPIController(m3, p3)
        configs.append(ControllerConfig("Shield-MPPI", c3, m3, "#2ca02c"))

        return configs

    def draw_environment(self, ax, t=0.0):
        super().draw_environment(ax, t)
        # 궤적 경로 표시
        ts = np.linspace(0, 2 * np.pi / 0.1, 300)
        pts = np.array([self._trajectory_fn(t_) for t_ in ts])
        ax.plot(pts[:, 0], pts[:, 1], "k--", alpha=0.2, linewidth=1)


def run_scenario(layout="random", live=False, no_plot=False, seed=42):
    env = StaticObstacleFieldEnv(layout=layout, seed=seed)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    initial_state = env.get_initial_state()
    duration = env.config.duration

    if live:
        # 실시간 애니메이션
        simulators = {}
        ref_fns = {}
        colors = {}
        for cc in configs:
            sim = Simulator(cc.model, cc.controller, env.config.dt)
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

        obstacles = env.get_obstacles()
        obstacles_fn = lambda t: obstacles

        for cc in configs:
            print(f"  Running {cc.name}...")
            sim = Simulator(cc.model, cc.controller, env.config.dt)
            sim.reset(initial_state)
            history = sim.run(ref_fn, duration)
            histories[cc.name] = history
            all_metrics[cc.name] = compute_metrics(history)
            all_env_metrics[cc.name] = compute_env_metrics(
                history, obstacles_fn=obstacles_fn
            )
            colors[cc.name] = cc.color

        # 결과 출력
        for name in histories:
            print_metrics(all_metrics[name], title=name)
        print_env_comparison(all_env_metrics, title=env.name)

        if not no_plot:
            viz = EnvVisualizer(env)
            fig = viz.run_and_plot(
                histories, all_metrics, all_env_metrics,
                controller_colors=colors,
                save_path=f"plots/s1_static_obstacle_{layout}.png",
            )
            plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S1: Static Obstacle Field")
    parser.add_argument("--layout", choices=["random", "slalom", "dense"],
                        default="random")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print(f"S1: Static Obstacle Field ({args.layout})".center(70))
    print("Vanilla MPPI vs CBF-MPPI vs Shield-MPPI".center(70))
    print("=" * 70 + "\n")

    run_scenario(layout=args.layout, live=args.live,
                 no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
