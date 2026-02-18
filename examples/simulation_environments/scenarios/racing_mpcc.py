#!/usr/bin/env python3
"""
S8: Racing MPCC

레이스 트랙에서 MPCC vs StateTracking 비교.
MPCCCost + PathParameterization으로 contouring/lag 분리.

Usage:
    python racing_mpcc.py
    python racing_mpcc.py --no-plot
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mpcc_cost import MPCCCost, PathParameterization
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import generate_reference_trajectory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.obstacle_field import generate_corridor
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


def generate_racetrack_waypoints(
    n_points: int = 60,
    a: float = 6.0,
    b: float = 3.0,
) -> np.ndarray:
    """타원형 레이스 트랙 웨이포인트 생성"""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    # 닫힌 경로
    waypoints = np.column_stack([x, y])
    waypoints = np.vstack([waypoints, waypoints[0]])
    return waypoints


class RacingMPCCEnv(SimulationEnvironment):
    """MPCC 레이싱 시나리오"""

    def __init__(self, seed: int = 42):
        config = EnvironmentConfig(
            name="S8: Racing MPCC",
            duration=25.0,
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

        # 레이스 트랙 생성
        self._track_waypoints = generate_racetrack_waypoints(n_points=60, a=6.0, b=3.0)

        # 트랙 벽 장애물
        track_points = [(wp[0], wp[1]) for wp in self._track_waypoints]
        self._track_walls = generate_corridor(
            track_points, width=1.5, thickness=0.12, spacing=0.25,
        )

        self._path_param = PathParameterization(self._track_waypoints)

    def get_initial_state(self):
        # 트랙 시작점
        wp = self._track_waypoints[0]
        # 진행 방향
        next_wp = self._track_waypoints[1]
        heading = np.arctan2(next_wp[1] - wp[1], next_wp[0] - wp[0])
        return np.array([wp[0], wp[1], heading])

    def get_obstacles(self, t=0.0):
        return self._track_walls

    def get_reference_fn(self):
        """StateTracking용 레퍼런스 생성"""
        N = self.config.N
        dt = self.config.dt
        waypoints = self._track_waypoints
        total_points = len(waypoints)

        speed = 0.8
        # 호장 기반
        diffs = np.diff(waypoints[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cum_lengths[-1]

        def point_at_dist(s):
            s = s % total_length  # 순환
            idx = np.searchsorted(cum_lengths[1:], s, side="right")
            idx = min(idx, len(seg_lengths) - 1)
            local = s - cum_lengths[idx]
            t_ratio = local / max(seg_lengths[idx], 1e-6)
            p = waypoints[idx, :2] + t_ratio * diffs[idx]
            heading = np.arctan2(diffs[idx][1], diffs[idx][0])
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

        configs = []

        # MPCC
        m1 = DifferentialDriveKinematic(v_max=1.2, omega_max=1.5)
        mpcc_cost = MPCCCost(
            reference_path=self._track_waypoints,
            Q_c=50.0,
            Q_l=10.0,
            Q_theta=5.0,
            Q_heading=1.0,
        )
        obs_cost1 = ObstacleCost(obstacles=self._track_walls, cost_weight=500.0)
        ctrl_cost1 = ControlEffortCost(c.R)
        cost1 = CompositeMPPICost([mpcc_cost, obs_cost1, ctrl_cost1])
        p1 = MPPIParams(**common)
        c1 = MPPIController(m1, p1, cost_function=cost1)
        configs.append(ControllerConfig("MPCC", c1, m1, "#1f77b4"))

        # StateTracking (baseline)
        m2 = DifferentialDriveKinematic(v_max=1.2, omega_max=1.5)
        track_cost = StateTrackingCost(c.Q)
        term_cost = TerminalCost(c.Qf)
        obs_cost2 = ObstacleCost(obstacles=self._track_walls, cost_weight=500.0)
        ctrl_cost2 = ControlEffortCost(c.R)
        cost2 = CompositeMPPICost([track_cost, term_cost, obs_cost2, ctrl_cost2])
        p2 = MPPIParams(**common)
        c2 = MPPIController(m2, p2, cost_function=cost2)
        configs.append(ControllerConfig("Tracking", c2, m2, "#ff7f0e"))

        return configs

    def draw_environment(self, ax, t=0.0):
        super().draw_environment(ax, t)
        # 트랙 중심선
        ax.plot(self._track_waypoints[:, 0], self._track_waypoints[:, 1],
                "g-", linewidth=1.5, alpha=0.4, label="Centerline")

    def get_extra_metrics(self, histories):
        """경로 오차 메트릭 (contouring/lag)"""
        extras = {}
        for name, h in histories.items():
            states = h["state"]
            x, y = states[:, 0], states[:, 1]
            theta_star, psi_path, closest = self._path_param.project(x, y)

            dx = x - closest[:, 0]
            dy = y - closest[:, 1]

            # contouring error (수직)
            e_c = -dx * np.sin(psi_path) + dy * np.cos(psi_path)
            # lag error (수평)
            e_l = dx * np.cos(psi_path) + dy * np.sin(psi_path)

            # progress (총 이동 호장)
            progress = theta_star[-1] - theta_star[0]

            extras[name] = {
                "mean_contouring_error": float(np.mean(np.abs(e_c))),
                "mean_lag_error": float(np.mean(np.abs(e_l))),
                "max_contouring_error": float(np.max(np.abs(e_c))),
                "progress": float(progress),
            }
        return extras


def run_scenario(no_plot=False, seed=42):
    env = RacingMPCCEnv(seed=seed)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    initial_state = env.get_initial_state()
    duration = env.config.duration
    dt = env.config.dt

    if no_plot:
        matplotlib.use("Agg")

    histories = {}
    all_metrics = {}
    all_env_metrics = {}
    colors = {}

    obstacles_fn = lambda t: env.get_obstacles(t)

    for cc in configs:
        print(f"  Running {cc.name}...")
        sim = Simulator(cc.model, cc.controller, dt)
        sim.reset(initial_state)
        history = sim.run(ref_fn, duration)
        histories[cc.name] = history
        all_metrics[cc.name] = compute_metrics(history)
        all_env_metrics[cc.name] = compute_env_metrics(
            history, obstacles_fn=obstacles_fn,
        )
        colors[cc.name] = cc.color

    for name in histories:
        print_metrics(all_metrics[name], title=name)

    # MPCC 전용 메트릭
    extras = env.get_extra_metrics(histories)
    print("\n" + "=" * 60)
    print("MPCC Racing Metrics".center(60))
    print("=" * 60)
    for name, e in extras.items():
        print(f"  {name:>10s}: contouring={e['mean_contouring_error']:.4f}m, "
              f"lag={e['mean_lag_error']:.4f}m, progress={e['progress']:.1f}m")
    print("=" * 60)

    print_env_comparison(all_env_metrics, title=env.name)

    if not no_plot:
        viz = EnvVisualizer(env)
        fig = viz.run_and_plot(
            histories, all_metrics, all_env_metrics,
            controller_colors=colors,
            save_path="plots/s8_racing_mpcc.png",
        )
        plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S8: Racing MPCC")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("S8: Racing MPCC".center(70))
    print("MPCC vs StateTracking".center(70))
    print("=" * 70 + "\n")

    run_scenario(no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
