#!/usr/bin/env python3
"""
S5: Waypoint Navigation

6-8개 웨이포인트를 순차 네비게이션 (대기 시간 포함) + 정적 장애물.
Vanilla MPPI vs CBF-MPPI 비교.

Usage:
    python waypoint_navigation.py
    python waypoint_navigation.py --live
    python waypoint_navigation.py --no-plot
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
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.obstacle_field import generate_random_field
from common.waypoint_manager import WaypointStateMachine
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


class WaypointNavigationEnv(SimulationEnvironment):
    """순차 웨이포인트 네비게이션 시나리오"""

    def __init__(self, seed: int = 42):
        config = EnvironmentConfig(
            name="S5: Waypoint Navigation",
            duration=30.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([12.0, 12.0, 1.5]),
            R=np.array([0.1, 0.1]),
            Qf=np.array([25.0, 25.0, 3.0]),
            seed=seed,
        )
        super().__init__(config)

        # 웨이포인트 정의 (x, y, theta)
        self._waypoint_list = [
            (3.0, 0.0, 0.0),
            (5.0, 3.0, np.pi / 4),
            (3.0, 6.0, np.pi / 2),
            (0.0, 6.0, np.pi),
            (-3.0, 4.0, -np.pi * 3 / 4),
            (-3.0, 0.0, -np.pi / 2),
            (0.0, -2.0, 0.0),
            (3.0, -2.0, 0.0),
        ]

        # 상태 머신 (각 컨트롤러별)
        self._state_machines = {}

        # 장애물: 웨이포인트 사이에 배치
        self._obstacles = generate_random_field(
            n=8,
            x_range=(-4.0, 6.0),
            y_range=(-3.0, 7.0),
            radius_range=(0.3, 0.5),
            exclusion_zones=[
                (wp[0], wp[1], 1.0) for wp in self._waypoint_list
            ] + [(0.0, 0.0, 1.0)],
            seed=seed,
        )

    def get_initial_state(self):
        return np.array([0.0, 0.0, 0.0])

    def get_obstacles(self, t=0.0):
        return self._obstacles

    def get_reference_fn(self):
        # 개별 컨트롤러가 각자의 state machine 사용
        # 여기서는 기본 ref_fn 반환 (on_step에서 개별 처리)
        N = self.config.N
        dt = self.config.dt

        def ref_fn(t):
            # 기본: 첫 웨이포인트로 이동
            ref = np.zeros((N + 1, 3))
            wp = self._waypoint_list[0]
            for i in range(N + 1):
                ref[i] = np.array(wp)
            return ref

        return ref_fn

    def create_state_machine(self, name: str) -> WaypointStateMachine:
        sm = WaypointStateMachine(
            waypoints=self._waypoint_list,
            arrival_threshold=0.4,
            dwell_time=0.5,
        )
        self._state_machines[name] = sm
        return sm

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

        # Vanilla MPPI
        m1 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p1 = MPPIParams(**common)
        c1 = MPPIController(m1, p1)
        configs.append(ControllerConfig("Vanilla", c1, m1, "#1f77b4"))

        # CBF-MPPI
        m2 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p2 = CBFMPPIParams(**common, **cbf_kw)
        c2 = CBFMPPIController(m2, p2)
        configs.append(ControllerConfig("CBF-MPPI", c2, m2, "#ff7f0e"))

        return configs

    def draw_environment(self, ax, t=0.0):
        super().draw_environment(ax, t)
        # 웨이포인트 표시
        for i, wp in enumerate(self._waypoint_list):
            ax.plot(wp[0], wp[1], "g^", markersize=12, alpha=0.7)
            ax.annotate(f"WP{i + 1}", (wp[0] + 0.2, wp[1] + 0.2),
                        fontsize=8, color="green")
        # 웨이포인트 연결선
        wps = np.array(self._waypoint_list)
        ax.plot(wps[:, 0], wps[:, 1], "g--", alpha=0.3, linewidth=1)


def run_scenario(live=False, no_plot=False, seed=42):
    env = WaypointNavigationEnv(seed=seed)
    configs = env.get_controller_configs()
    initial_state = env.get_initial_state()
    duration = env.config.duration
    dt = env.config.dt
    N = env.config.N

    if no_plot and not live:
        matplotlib.use("Agg")

    histories = {}
    all_metrics = {}
    all_env_metrics = {}
    colors = {}

    obstacles = env.get_obstacles()
    obstacles_fn = lambda t: obstacles

    for cc in configs:
        print(f"  Running {cc.name}...")
        sm = env.create_state_machine(cc.name)
        sim = Simulator(cc.model, cc.controller, dt)
        sim.reset(initial_state)

        ref_fn = sm.get_reference_fn(N, dt)
        num_steps = int(duration / dt)

        for step in range(num_steps):
            t = sim.t
            sm.update(sim.state, t)
            ref_traj = ref_fn(t)
            sim.step(ref_traj)

            if sm.is_completed:
                break

        history = sim.get_history()
        histories[cc.name] = history
        all_metrics[cc.name] = compute_metrics(history)

        final_wp = env._waypoint_list[-1]
        all_env_metrics[cc.name] = compute_env_metrics(
            history,
            obstacles_fn=obstacles_fn,
            goal=np.array(final_wp),
            goal_threshold=0.5,
        )
        colors[cc.name] = cc.color

        info = sm.get_info()
        print(f"    Progress: {info['progress']:.0%}, State: {info['state']}")

    for name in histories:
        print_metrics(all_metrics[name], title=name)
    print_env_comparison(all_env_metrics, title=env.name)

    if not no_plot:
        viz = EnvVisualizer(env)
        fig = viz.run_and_plot(
            histories, all_metrics, all_env_metrics,
            controller_colors=colors,
            save_path="plots/s5_waypoint_navigation.png",
        )
        plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S5: Waypoint Navigation")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("S5: Waypoint Navigation".center(70))
    print("Vanilla MPPI vs CBF-MPPI".center(70))
    print("=" * 70 + "\n")

    run_scenario(live=args.live, no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
