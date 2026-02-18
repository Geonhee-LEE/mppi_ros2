#!/usr/bin/env python3
"""
S9: Narrow Corridor

좁은 통로 + 90도 회전을 통과하는 시나리오.
Shield-MPPI vs CBF-MPPI vs Optimal-Decay CBF 3-way 비교.

Usage:
    python narrow_corridor.py
    python narrow_corridor.py --live
    python narrow_corridor.py --no-plot
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
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import generate_reference_trajectory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.obstacle_field import generate_corridor, generate_funnel
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


class NarrowCorridorEnv(SimulationEnvironment):
    """좁은 통로 + 90도 회전 시나리오"""

    def __init__(self, seed: int = 42):
        config = EnvironmentConfig(
            name="S9: Narrow Corridor + 90-deg Turns",
            duration=25.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([15.0, 15.0, 2.0]),
            R=np.array([0.05, 0.05]),
            Qf=np.array([30.0, 30.0, 4.0]),
            seed=seed,
        )
        super().__init__(config)

        # 복도 경로: L자형 + 좁은 구간
        self._path_points = [
            (0.0, 0.0),
            (4.0, 0.0),    # 직선
            (4.0, 4.0),    # 90도 우회전
            (8.0, 4.0),    # 직선
            (8.0, 0.0),    # 90도 우회전
            (12.0, 0.0),   # 직선
        ]

        # 벽 장애물 생성
        corridor_width = 1.2
        funnel_obstacles = generate_funnel(
            start_width=2.0, end_width=0.9,
            length=2.5,
            start_x=5.5, start_y=4.0,
            direction=0.0,
        )
        corridor_obstacles = generate_corridor(
            self._path_points,
            width=corridor_width,
            thickness=0.12,
            spacing=0.18,
        )

        self._obstacles = corridor_obstacles + funnel_obstacles

        # 레퍼런스: 경로 중심선을 따라 이동
        self._waypoints = np.array(self._path_points + [(12.0, 0.0)])

    def get_initial_state(self):
        return np.array([0.0, 0.0, 0.0])

    def get_obstacles(self, t=0.0):
        return self._obstacles

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        waypoints = self._waypoints

        # 시간 기반 웨이포인트 보간
        total_dist = 0.0
        seg_dists = []
        for i in range(len(waypoints) - 1):
            d = np.linalg.norm(waypoints[i + 1, :2] - waypoints[i, :2])
            seg_dists.append(d)
            total_dist += d

        speed = 0.5  # 느린 속도 (좁은 통로)
        cum_dists = np.concatenate([[0.0], np.cumsum(seg_dists)])

        def point_at_dist(s):
            s = np.clip(s, 0, total_dist)
            seg_idx = np.searchsorted(cum_dists[1:], s, side="right")
            seg_idx = min(seg_idx, len(seg_dists) - 1)
            local_s = s - cum_dists[seg_idx]
            seg_len = seg_dists[seg_idx]
            t_ratio = local_s / max(seg_len, 1e-6)
            p = waypoints[seg_idx] + t_ratio * (waypoints[seg_idx + 1] - waypoints[seg_idx])
            # heading
            direction = waypoints[seg_idx + 1, :2] - waypoints[seg_idx, :2]
            heading = np.arctan2(direction[1], direction[0])
            return np.array([p[0], p[1], heading])

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
        cbf_kw = dict(
            cbf_obstacles=self._obstacles,
            cbf_weight=1500.0, cbf_alpha=0.2, cbf_safety_margin=0.08,
        )

        configs = []

        # CBF-MPPI
        m1 = DifferentialDriveKinematic(v_max=0.8, omega_max=1.5)
        p1 = CBFMPPIParams(**common, **cbf_kw)
        c1 = CBFMPPIController(m1, p1)
        configs.append(ControllerConfig("CBF-MPPI", c1, m1, "#1f77b4"))

        # Shield-MPPI
        m2 = DifferentialDriveKinematic(v_max=0.8, omega_max=1.5)
        p2 = ShieldMPPIParams(**common, **cbf_kw, shield_enabled=True)
        c2 = ShieldMPPIController(m2, p2)
        configs.append(ControllerConfig("Shield-MPPI", c2, m2, "#ff7f0e"))

        # CBF-MPPI with higher alpha
        m3 = DifferentialDriveKinematic(v_max=0.8, omega_max=1.5)
        cbf_kw3 = dict(cbf_kw)
        cbf_kw3["cbf_alpha"] = 0.5
        cbf_kw3["cbf_weight"] = 2000.0
        p3 = CBFMPPIParams(**common, **cbf_kw3)
        c3 = CBFMPPIController(m3, p3)
        configs.append(ControllerConfig("CBF-Aggressive", c3, m3, "#2ca02c"))

        return configs

    def draw_environment(self, ax, t=0.0):
        super().draw_environment(ax, t)
        # 경로 중심선 표시
        path = np.array(self._path_points)
        ax.plot(path[:, 0], path[:, 1], "g--", linewidth=1.5, alpha=0.5, label="Path")


def run_scenario(live=False, no_plot=False, seed=42):
    env = NarrowCorridorEnv(seed=seed)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    initial_state = env.get_initial_state()
    duration = env.config.duration

    if live:
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
                history, obstacles_fn=obstacles_fn,
                goal=np.array([12.0, 0.0]),
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
                save_path="plots/s9_narrow_corridor.png",
            )
            plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S9: Narrow Corridor")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("S9: Narrow Corridor + 90-deg Turns".center(70))
    print("CBF-MPPI vs Shield-MPPI vs CBF-Aggressive".center(70))
    print("=" * 70 + "\n")

    run_scenario(live=args.live, no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
