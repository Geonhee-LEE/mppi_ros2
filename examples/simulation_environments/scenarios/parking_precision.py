#!/usr/bin/env python3
"""
S7: Parking Precision

Ackermann 모델로 주차 슬롯에 정밀 주차.
Vanilla vs SVG-MPPI vs Smooth MPPI 3-way 비교.
SuperellipsoidCost로 직사각형 주차 경계 모델링.

Usage:
    python parking_precision.py
    python parking_precision.py --no-plot
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.superellipsoid_cost import (
    SuperellipsoidObstacle,
    SuperellipsoidCost,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


class ParkingPrecisionEnv(SimulationEnvironment):
    """Ackermann 주차 시나리오"""

    def __init__(self, seed: int = 42):
        config = EnvironmentConfig(
            name="S7: Parking Precision (Ackermann)",
            duration=15.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.3, 0.3]),  # [v, phi]
            Q=np.array([15.0, 15.0, 3.0, 0.5]),  # [x, y, theta, delta]
            R=np.array([0.1, 0.2]),
            robot_model_type="ackermann",
            seed=seed,
        )
        super().__init__(config)

        # 주차 슬롯 정의
        self._parking_target = np.array([6.0, 0.0, 0.0, 0.0])  # [x, y, theta, delta]

        # 주차 경계 (superellipsoid 장애물)
        self._parking_walls = [
            # 주차장 위쪽 벽
            SuperellipsoidObstacle(cx=6.0, cy=1.5, a=2.5, b=0.2, n=4.0, theta=0.0),
            # 주차장 아래쪽 벽
            SuperellipsoidObstacle(cx=6.0, cy=-1.5, a=2.5, b=0.2, n=4.0, theta=0.0),
            # 왼쪽 차량
            SuperellipsoidObstacle(cx=3.0, cy=0.0, a=0.8, b=0.5, n=4.0, theta=0.0),
            # 오른쪽 차량
            SuperellipsoidObstacle(cx=9.0, cy=0.0, a=0.8, b=0.5, n=4.0, theta=0.0),
        ]

        # 원형 장애물 근사 (CBF 호환)
        self._circle_obstacles = [
            (6.0, 1.5, 0.3), (6.0, -1.5, 0.3),
            (3.0, 0.8, 0.5), (3.0, -0.8, 0.5),
            (9.0, 0.8, 0.5), (9.0, -0.8, 0.5),
        ]

    def get_initial_state(self):
        return np.array([0.0, 2.0, 0.0, 0.0])  # [x, y, theta, delta]

    def get_obstacles(self, t=0.0):
        return self._circle_obstacles

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        target = self._parking_target

        def ref_fn(t):
            ref = np.zeros((N + 1, 4))
            current_target = target.copy()
            for i in range(N + 1):
                ref[i] = current_target
            return ref

        return ref_fn

    def get_controller_configs(self):
        c = self.config
        common = dict(
            N=c.N, dt=c.dt, K=c.K, lambda_=c.lambda_,
            sigma=c.sigma, Q=c.Q, R=c.R,
        )

        # Superellipsoid cost
        se_cost = SuperellipsoidCost(
            obstacles=self._parking_walls,
            cbf_weight=800.0,
        )

        configs = []

        # Vanilla MPPI
        m1 = AckermannKinematic(wheelbase=0.5, v_max=0.8, max_steer=0.5)
        base_costs1 = [
            StateTrackingCost(c.Q),
            TerminalCost(c.Q * 2),
            ControlEffortCost(c.R),
            se_cost,
        ]
        p1 = MPPIParams(**common)
        c1 = MPPIController(m1, p1, cost_function=CompositeMPPICost(base_costs1))
        configs.append(ControllerConfig("Vanilla", c1, m1, "#1f77b4"))

        # High-K MPPI (more samples for better exploration)
        m2 = AckermannKinematic(wheelbase=0.5, v_max=0.8, max_steer=0.5)
        base_costs2 = [
            StateTrackingCost(c.Q),
            TerminalCost(c.Q * 3),
            ControlEffortCost(c.R),
            SuperellipsoidCost(obstacles=self._parking_walls, cbf_weight=800.0),
        ]
        common2 = dict(common)
        common2["K"] = 2048
        common2["lambda_"] = 0.5
        p2 = MPPIParams(**common2)
        c2 = MPPIController(m2, p2, cost_function=CompositeMPPICost(base_costs2))
        configs.append(ControllerConfig("MPPI-2K", c2, m2, "#ff7f0e"))

        # Fine-sigma MPPI (narrow noise for precision)
        m3 = AckermannKinematic(wheelbase=0.5, v_max=0.8, max_steer=0.5)
        base_costs3 = [
            StateTrackingCost(c.Q * 2),
            TerminalCost(c.Q * 4),
            ControlEffortCost(c.R * 0.5),
            SuperellipsoidCost(obstacles=self._parking_walls, cbf_weight=800.0),
        ]
        common3 = dict(common)
        common3["sigma"] = np.array([0.15, 0.15])
        p3 = MPPIParams(**common3)
        c3 = MPPIController(m3, p3, cost_function=CompositeMPPICost(base_costs3))
        configs.append(ControllerConfig("MPPI-Fine", c3, m3, "#2ca02c"))

        return configs

    def draw_environment(self, ax, t=0.0):
        # 주차 슬롯 직사각형
        for wall in self._parking_walls:
            cx, cy, a, b = wall.cx, wall.cy, wall.a, wall.b
            rect = FancyBboxPatch(
                (cx - a, cy - b), 2 * a, 2 * b,
                boxstyle="round,pad=0.02",
                facecolor="gray", edgecolor="black", alpha=0.4,
            )
            ax.add_patch(rect)

        # 목표 표시
        tx, ty = self._parking_target[0], self._parking_target[1]
        ax.plot(tx, ty, "g*", markersize=20, alpha=0.7, label="Parking Target")
        target_rect = plt.Rectangle(
            (tx - 1.2, ty - 0.8), 2.4, 1.6,
            fill=False, edgecolor="green", linestyle="--", linewidth=2,
        )
        ax.add_patch(target_rect)

    def get_extra_metrics(self, histories):
        """주차 정밀도 메트릭"""
        target = self._parking_target
        extras = {}
        for name, h in histories.items():
            final = h["state"][-1]
            pos_err = np.linalg.norm(final[:2] - target[:2])
            heading_err = abs(np.arctan2(np.sin(final[2] - target[2]),
                                         np.cos(final[2] - target[2])))
            extras[name] = {
                "final_position_error": pos_err,
                "final_heading_error": np.degrees(heading_err),
            }
        return extras


def run_scenario(no_plot=False, seed=42):
    env = ParkingPrecisionEnv(seed=seed)
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
            history,
            obstacles_fn=obstacles_fn,
            goal=env._parking_target[:2],
            goal_threshold=0.1,
        )
        colors[cc.name] = cc.color

    for name in histories:
        print_metrics(all_metrics[name], title=name)

    # 주차 정밀도 출력
    extras = env.get_extra_metrics(histories)
    print("\n" + "=" * 50)
    print("Parking Precision".center(50))
    print("=" * 50)
    for name, e in extras.items():
        print(f"  {name:>15s}: pos_err={e['final_position_error']:.4f}m, "
              f"heading_err={e['final_heading_error']:.1f}deg")
    print("=" * 50)

    print_env_comparison(all_env_metrics, title=env.name)

    if not no_plot:
        viz = EnvVisualizer(env)
        fig = viz.run_and_plot(
            histories, all_metrics, all_env_metrics,
            controller_colors=colors,
            save_path="plots/s7_parking_precision.png",
        )
        plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S7: Parking Precision")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("S7: Parking Precision (Ackermann)".center(70))
    print("Vanilla vs MPPI-2K vs MPPI-Fine".center(70))
    print("=" * 70 + "\n")

    run_scenario(no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
