#!/usr/bin/env python3
"""
S10: Mixed Challenge

복합 시나리오: 정적 장애물 → 동적 장애물 → 좁은 통로 → 목표 도달.
Shield-MPPI 단일 컨트롤러, 완주 및 안전성 측정.

Usage:
    python mixed_challenge.py
    python mixed_challenge.py --no-plot
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import ShieldMPPIParams
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.obstacle_field import generate_random_field, generate_corridor, generate_funnel
from common.dynamic_obstacle import DynamicObstacle, BouncingMotion, CrossingMotion
from common.waypoint_manager import WaypointStateMachine
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


class MixedChallengeEnv(SimulationEnvironment):
    """복합 챌린지 시나리오"""

    def __init__(self, seed: int = 42):
        config = EnvironmentConfig(
            name="S10: Mixed Challenge",
            duration=40.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([12.0, 12.0, 1.5]),
            R=np.array([0.08, 0.08]),
            Qf=np.array([25.0, 25.0, 3.0]),
            seed=seed,
        )
        super().__init__(config)

        # ── Zone 1: 정적 장애물 필드 (0 ≤ x ≤ 6) ──
        self._static_obstacles = generate_random_field(
            n=8,
            x_range=(1.0, 5.0),
            y_range=(-2.0, 2.0),
            radius_range=(0.25, 0.45),
            exclusion_zones=[(0.0, 0.0, 0.8), (6.0, 0.0, 0.8)],
            seed=seed,
        )

        # ── Zone 2: 동적 장애물 (6 ≤ x ≤ 12) ──
        rng = np.random.RandomState(seed + 1)
        self._dynamic_obstacles = [
            DynamicObstacle(
                CrossingMotion(center=(8.0, 0.0), amplitude=2.0, period=4.0,
                               direction=np.pi / 2),
                radius=0.4,
            ),
            DynamicObstacle(
                CrossingMotion(center=(10.0, 0.0), amplitude=1.5, period=5.0,
                               direction=np.pi / 2),
                radius=0.35,
            ),
            DynamicObstacle(
                BouncingMotion(start=(7.0, 1.5), velocity=(0.3, -0.4),
                               bounds=(6.0, -3.0, 12.0, 3.0)),
                radius=0.3,
            ),
        ]

        # ── Zone 3: 좁은 통로 (12 ≤ x ≤ 16) ──
        corridor_path = [(12.0, 0.0), (14.0, 0.0), (14.0, 3.0), (16.0, 3.0)]
        self._corridor_obstacles = generate_corridor(
            corridor_path, width=1.0, thickness=0.12, spacing=0.2,
        )

        # 웨이포인트
        self._waypoint_list = [
            (3.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
            (9.0, 0.0, 0.0),
            (12.0, 0.0, 0.0),
            (14.0, 0.0, np.pi / 2),
            (14.0, 3.0, 0.0),
            (16.0, 3.0, 0.0),
        ]

        self._state_machines = {}
        self._controllers = {}

    def get_initial_state(self):
        return np.array([0.0, 0.0, 0.0])

    def get_obstacles(self, t=0.0):
        """시간에 따른 장애물 (정적 + 동적 + 복도)"""
        obs = list(self._static_obstacles)
        obs.extend([d.as_3tuple(t) for d in self._dynamic_obstacles])
        obs.extend(self._corridor_obstacles)
        return obs

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        waypoints = self._waypoint_list

        # 기본 레퍼런스 (웨이포인트 기반)
        def ref_fn(t):
            ref = np.zeros((N + 1, 3))
            wp = waypoints[0]
            for i in range(N + 1):
                ref[i] = np.array(wp)
            return ref

        return ref_fn

    def create_state_machine(self, name):
        sm = WaypointStateMachine(
            waypoints=self._waypoint_list,
            arrival_threshold=0.5,
            dwell_time=0.3,
        )
        self._state_machines[name] = sm
        return sm

    def get_controller_configs(self):
        c = self.config
        common = dict(
            N=c.N, dt=c.dt, K=c.K, lambda_=c.lambda_,
            sigma=c.sigma, Q=c.Q, R=c.R, Qf=c.Qf,
        )
        initial_obs = self.get_obstacles(0.0)
        cbf_kw = dict(
            cbf_obstacles=initial_obs,
            cbf_weight=1200.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        )

        configs = []

        # Shield-MPPI
        m1 = DifferentialDriveKinematic(v_max=0.8, omega_max=1.2)
        p1 = ShieldMPPIParams(**common, **cbf_kw, shield_enabled=True)
        c1 = ShieldMPPIController(m1, p1)
        self._controllers["Shield-MPPI"] = c1
        configs.append(ControllerConfig("Shield-MPPI", c1, m1, "#1f77b4"))

        return configs

    def on_step(self, t, states, controls, infos):
        """장애물 업데이트"""
        current_obs = self.get_obstacles(t)
        for name, ctrl in self._controllers.items():
            if hasattr(ctrl, "update_obstacles"):
                ctrl.update_obstacles(current_obs)

    def draw_environment(self, ax, t=0.0):
        import matplotlib.pyplot as plt

        # Zone 구분선
        ax.axvline(x=6.0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=12.0, color="gray", linestyle=":", alpha=0.5)
        ax.text(3.0, -3.0, "Zone 1\nStatic", ha="center", fontsize=8, color="gray")
        ax.text(9.0, -3.0, "Zone 2\nDynamic", ha="center", fontsize=8, color="gray")
        ax.text(14.0, -1.5, "Zone 3\nCorridor", ha="center", fontsize=8, color="gray")

        # 정적 장애물
        for ox, oy, r in self._static_obstacles:
            circle = plt.Circle((ox, oy), r, color="red", alpha=0.25)
            ax.add_patch(circle)

        # 동적 장애물
        for obs in self._dynamic_obstacles:
            px, py = obs.get_position(t)
            circle = plt.Circle((px, py), obs.radius, color="orange", alpha=0.35)
            ax.add_patch(circle)

        # 복도 장애물
        for ox, oy, r in self._corridor_obstacles:
            circle = plt.Circle((ox, oy), r, color="brown", alpha=0.2)
            ax.add_patch(circle)

        # 웨이포인트
        for i, wp in enumerate(self._waypoint_list):
            ax.plot(wp[0], wp[1], "g^", markersize=10, alpha=0.7)
            ax.annotate(f"WP{i + 1}", (wp[0], wp[1] + 0.3), fontsize=7,
                        ha="center", color="green")


def run_scenario(no_plot=False, seed=42):
    env = MixedChallengeEnv(seed=seed)
    configs = env.get_controller_configs()
    initial_state = env.get_initial_state()
    duration = env.config.duration
    dt = env.config.dt
    N = env.config.N

    if no_plot:
        matplotlib.use("Agg")

    histories = {}
    all_metrics = {}
    all_env_metrics = {}
    colors = {}

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
            env.on_step(t, {cc.name: sim.state}, {}, {})
            ref_traj = ref_fn(t)
            sim.step(ref_traj)

            if sm.is_completed:
                print(f"    Completed at t={t:.1f}s!")
                break

        history = sim.get_history()
        histories[cc.name] = history
        all_metrics[cc.name] = compute_metrics(history)

        final_wp = env._waypoint_list[-1]
        all_env_metrics[cc.name] = compute_env_metrics(
            history,
            obstacles_fn=lambda t: env.get_obstacles(t),
            goal=np.array(final_wp),
            goal_threshold=0.6,
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
            save_path="plots/s10_mixed_challenge.png",
        )
        plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S10: Mixed Challenge")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("S10: Mixed Challenge".center(70))
    print("Static + Dynamic + Corridor (Shield-MPPI)".center(70))
    print("=" * 70 + "\n")

    run_scenario(no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
