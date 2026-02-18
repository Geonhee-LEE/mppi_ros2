#!/usr/bin/env python3
"""
S6: Drifting Disturbance

프로세스 노이즈(바람/미끄러짐)가 있는 환경에서의 경로 추종.
Vanilla MPPI vs Tube-MPPI vs Risk-Aware MPPI 3-way 비교.

Usage:
    python drifting_disturbance.py
    python drifting_disturbance.py --live
    python drifting_disturbance.py --noise 0.5
    python drifting_disturbance.py --no-plot
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
    TubeMPPIParams,
    RiskAwareMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


class DriftingDisturbanceEnv(SimulationEnvironment):
    """프로세스 노이즈 환경 시나리오"""

    def __init__(self, noise_level: float = 0.3, seed: int = 42):
        config = EnvironmentConfig(
            name=f"S6: Drifting Disturbance (noise={noise_level})",
            duration=20.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            Qf=np.array([20.0, 20.0, 2.0]),
            process_noise_std=np.array([noise_level, noise_level, noise_level * 0.5]),
            seed=seed,
        )
        super().__init__(config)
        self.noise_level = noise_level
        self._trajectory_fn = create_trajectory_function("figure8", scale=4.0, period=20.0)

    def get_initial_state(self):
        return self._trajectory_fn(0.0)

    def get_obstacles(self, t=0.0):
        return []  # 장애물 없음 — 순수 추적 성능 비교

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

        configs = []

        # Vanilla MPPI
        m1 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p1 = MPPIParams(**common)
        c1 = MPPIController(m1, p1)
        configs.append(ControllerConfig("Vanilla", c1, m1, "#1f77b4"))

        # Tube-MPPI
        m2 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p2 = TubeMPPIParams(**common, tube_enabled=True, tube_margin=0.15)
        c2 = TubeMPPIController(m2, p2)
        configs.append(ControllerConfig("Tube-MPPI", c2, m2, "#ff7f0e"))

        # Risk-Aware MPPI (CVaR)
        m3 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p3 = RiskAwareMPPIParams(**common, cvar_alpha=0.3)
        c3 = RiskAwareMPPIController(m3, p3)
        configs.append(ControllerConfig("Risk-Aware", c3, m3, "#2ca02c"))

        return configs


def run_scenario(noise_level=0.3, live=False, no_plot=False, seed=42):
    env = DriftingDisturbanceEnv(noise_level=noise_level, seed=seed)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    initial_state = env.get_initial_state()
    duration = env.config.duration
    noise_std = env.config.process_noise_std

    if live:
        simulators = {}
        ref_fns = {}
        colors = {}
        for cc in configs:
            sim = Simulator(cc.model, cc.controller, env.config.dt,
                            process_noise_std=noise_std)
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
            np.random.seed(seed)  # 동일 노이즈 시퀀스
            sim = Simulator(cc.model, cc.controller, env.config.dt,
                            process_noise_std=noise_std)
            sim.reset(initial_state)
            history = sim.run(ref_fn, duration)
            histories[cc.name] = history
            all_metrics[cc.name] = compute_metrics(history)
            all_env_metrics[cc.name] = compute_env_metrics(history)
            colors[cc.name] = cc.color

        for name in histories:
            print_metrics(all_metrics[name], title=name)
        print_env_comparison(all_env_metrics, title=env.name)

        if not no_plot:
            viz = EnvVisualizer(env)
            fig = viz.run_and_plot(
                histories, all_metrics, all_env_metrics,
                controller_colors=colors,
                save_path="plots/s6_drifting_disturbance.png",
            )
            plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S6: Drifting Disturbance")
    parser.add_argument("--noise", type=float, default=0.3,
                        help="Process noise level")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"S6: Drifting Disturbance (noise={args.noise})".center(70))
    print("Vanilla vs Tube-MPPI vs Risk-Aware MPPI".center(70))
    print("=" * 70 + "\n")

    run_scenario(noise_level=args.noise, live=args.live,
                 no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
