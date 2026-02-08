#!/usr/bin/env python3
"""
Safety Phase S3 Comparison Demo

3 시나리오:
  1. backup_cbf — Standard CBF vs Backup CBF vs Gatekeeper
  2. multi_robot — 3 로봇 교차 경로 (분산 CBF)
  3. mpcc — MPCC vs StateTrackingCost 곡선 경로 비교

Usage:
    python safety_s3_comparison_demo.py                            # all scenarios (batch)
    python safety_s3_comparison_demo.py --scenario backup_cbf      # single scenario
    python safety_s3_comparison_demo.py --scenario multi_robot
    python safety_s3_comparison_demo.py --scenario mpcc
    python safety_s3_comparison_demo.py --no-plot                  # no plot (CI)
    python safety_s3_comparison_demo.py --live                     # live animation
"""

import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    StateTrackingCost,
    TerminalCost,
    CompositeMPPICost,
)
from mppi_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter
from mppi_controller.controllers.mppi.backup_cbf_filter import BackupCBFSafetyFilter
from mppi_controller.controllers.mppi.backup_controller import BrakeBackupController
from mppi_controller.controllers.mppi.gatekeeper import Gatekeeper
from mppi_controller.controllers.mppi.multi_robot_cbf import (
    RobotAgent,
    MultiRobotCBFFilter,
    MultiRobotCoordinator,
)
from mppi_controller.controllers.mppi.mpcc_cost import MPCCCost
from mppi_controller.simulation.simulator import Simulator


def generate_reference(start, goal, N, dt):
    """직선 레퍼런스 궤적 생성"""
    t_total = N * dt
    ref = np.zeros((N + 1, 3))
    for i in range(N + 1):
        alpha = i / N
        ref[i, 0] = start[0] + alpha * (goal[0] - start[0])
        ref[i, 1] = start[1] + alpha * (goal[1] - start[1])
        ref[i, 2] = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    return ref


def run_backup_cbf_scenario(show_plot=True, live=False):
    """시나리오 1: Standard CBF vs Backup CBF vs Gatekeeper"""
    print("\n" + "=" * 60)
    print("Scenario: Backup CBF Comparison".center(60))
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(3.0, 0.3, 0.4), (5.0, -0.2, 0.3)]
    dt = 0.05
    N = 20
    duration = 12.0
    n_steps = int(duration / dt)
    initial_state = np.array([0.0, 0.0, 0.0])
    goal = np.array([8.0, 0.0, 0.0])

    params = MPPIParams(K=256, N=N, dt=dt, lambda_=1.0, sigma=np.array([0.3, 0.3]))
    cost = CompositeMPPICost([
        StateTrackingCost(np.array([10.0, 10.0, 1.0])),
        TerminalCost(np.array([20.0, 20.0, 2.0])),
    ])

    # 3 controllers + filters
    methods = {
        "Standard CBF": CBFSafetyFilter(
            obstacles=obstacles, cbf_alpha=0.3, safety_margin=0.1,
        ),
        "Backup CBF": BackupCBFSafetyFilter(
            backup_controller=BrakeBackupController(),
            model=model, obstacles=obstacles,
            dt=dt, backup_horizon=15, cbf_alpha=0.3, safety_margin=0.1,
        ),
        "Gatekeeper": Gatekeeper(
            backup_controller=BrakeBackupController(),
            model=model, obstacles=obstacles,
            safety_margin=0.1, backup_horizon=20, dt=dt,
        ),
    }

    results = {}
    for name, safety_filter in methods.items():
        print(f"\n  Running {name}...")
        controller = MPPIController(model, params, cost)
        state = initial_state.copy()
        states = [state.copy()]

        for step in range(n_steps):
            ref = generate_reference(state, goal, N, dt)
            u_mppi, _ = controller.compute_control(state, ref)

            if isinstance(safety_filter, Gatekeeper):
                u_safe, info = safety_filter.filter(state, u_mppi)
            else:
                u_safe, info = safety_filter.filter_control(
                    state, u_mppi,
                    u_min=np.array([-1.0, -1.0]),
                    u_max=np.array([1.0, 1.0]),
                )

            state = model.step(state, u_safe, dt)
            states.append(state.copy())

        states = np.array(states)

        # 최소 장애물 거리 계산
        min_obs_dist = float("inf")
        for obs in obstacles:
            dists = np.sqrt((states[:, 0] - obs[0])**2 + (states[:, 1] - obs[1])**2) - obs[2]
            min_obs_dist = min(min_obs_dist, float(np.min(dists)))

        # 목표 거리
        final_dist = np.sqrt((states[-1, 0] - goal[0])**2 + (states[-1, 1] - goal[1])**2)

        results[name] = {
            "states": states,
            "final_dist": final_dist,
            "min_obs_dist": min_obs_dist,
        }
        print(f"    Final dist to goal: {final_dist:.4f}m")
        print(f"    Min obstacle dist: {min_obs_dist:.4f}m")

    # Plot
    if show_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            colors = {"Standard CBF": "blue", "Backup CBF": "red", "Gatekeeper": "green"}

            for name, data in results.items():
                ax.plot(data["states"][:, 0], data["states"][:, 1],
                        label=name, color=colors[name], linewidth=2)

            for obs in obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color="gray", alpha=0.5)
                ax.add_patch(circle)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Backup CBF Comparison")
            ax.legend()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("safety_s3_backup_cbf.png", dpi=150)
            print(f"\n  Saved: safety_s3_backup_cbf.png")
            plt.close()
        except ImportError:
            print("  matplotlib not available, skipping plot")

    return results


def run_multi_robot_scenario(show_plot=True, live=False):
    """시나리오 2: 3 로봇 교차 경로"""
    print("\n" + "=" * 60)
    print("Scenario: Multi-Robot CBF (3 Agents)".center(60))
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=0.8, omega_max=1.0)
    dt = 0.05
    N = 15
    n_steps = 150

    params = MPPIParams(K=128, N=N, dt=dt, lambda_=1.0, sigma=np.array([0.3, 0.3]))

    # 3 로봇: 삼각형 꼭짓점에서 대각선 반대로
    robot_configs = [
        (np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])),       # →
        (np.array([4.0, 0.0, np.pi]), np.array([0.0, 0.0, np.pi])),    # ←
        (np.array([2.0, -2.0, np.pi/2]), np.array([2.0, 2.0, np.pi/2])),  # ↑
    ]

    agents = []
    for i, (start, goal) in enumerate(robot_configs):
        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([20.0, 20.0, 2.0])),
        ])
        ctrl = MPPIController(model, params, cost)
        agents.append(
            RobotAgent(id=i, state=start.copy(), radius=0.2, model=model, controller=ctrl)
        )

    coordinator = MultiRobotCoordinator(
        agents, dt=dt, cbf_alpha=0.3, safety_margin=0.15,
        use_cost=False, use_filter=True,
    )

    # 시뮬레이션
    all_states = {i: [agents[i].state.copy()] for i in range(3)}
    filter_counts = {i: 0 for i in range(3)}

    for step in range(n_steps):
        refs = {}
        for i, (start, goal) in enumerate(robot_configs):
            refs[i] = generate_reference(coordinator.agents[i].state, goal, N, dt)

        results = coordinator.step(refs)

        for i in range(3):
            all_states[i].append(coordinator.agents[i].state.copy())
            if results[i][1].get("cbf_filtered", False):
                filter_counts[i] += 1

    # 최소 로봇 간 거리 계산
    min_inter_dist = float("inf")
    for step in range(n_steps + 1):
        for i in range(3):
            for j in range(i + 1, 3):
                s_i = all_states[i][step]
                s_j = all_states[j][step]
                dist = np.sqrt((s_i[0] - s_j[0])**2 + (s_i[1] - s_j[1])**2)
                min_inter_dist = min(min_inter_dist, dist)

    print(f"\n  Min inter-robot distance: {min_inter_dist:.4f}m")
    print(f"  Robot radii: 0.2m + margin 0.15m = 0.55m required")
    for i in range(3):
        final = all_states[i][-1]
        goal = robot_configs[i][1]
        dist_to_goal = np.sqrt((final[0] - goal[0])**2 + (final[1] - goal[1])**2)
        print(f"  Robot {i}: final pos=({final[0]:.2f}, {final[1]:.2f}), "
              f"dist_to_goal={dist_to_goal:.2f}m, filtered={filter_counts[i]}")

    # Plot
    if show_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            colors = ["blue", "red", "green"]
            labels = ["Robot 0 (→)", "Robot 1 (←)", "Robot 2 (↑)"]

            for i in range(3):
                traj = np.array(all_states[i])
                ax.plot(traj[:, 0], traj[:, 1], color=colors[i], label=labels[i], linewidth=2)
                ax.plot(traj[0, 0], traj[0, 1], "o", color=colors[i], markersize=10)
                ax.plot(traj[-1, 0], traj[-1, 1], "s", color=colors[i], markersize=10)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Multi-Robot CBF Coordination (3 Agents)")
            ax.legend()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("safety_s3_multi_robot.png", dpi=150)
            print(f"\n  Saved: safety_s3_multi_robot.png")
            plt.close()
        except ImportError:
            print("  matplotlib not available, skipping plot")


def run_mpcc_scenario(show_plot=True, live=False):
    """시나리오 3: MPCC vs Tracking Cost 곡선 경로"""
    print("\n" + "=" * 60)
    print("Scenario: MPCC vs Tracking Cost".center(60))
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.5)
    dt = 0.05
    N = 20
    duration = 15.0
    n_steps = int(duration / dt)

    # S-curve 경로
    t_path = np.linspace(0, 10, 100)
    waypoints = np.stack([t_path, 2.0 * np.sin(t_path * 0.5)], axis=-1)

    # 두 비용 함수
    mpcc_cost = MPCCCost(
        reference_path=waypoints,
        Q_c=80.0, Q_l=15.0, Q_theta=3.0, Q_heading=2.0,
    )
    tracking_cost = CompositeMPPICost([
        StateTrackingCost(np.array([10.0, 10.0, 1.0])),
        TerminalCost(np.array([20.0, 20.0, 2.0])),
    ])

    params = MPPIParams(K=256, N=N, dt=dt, lambda_=1.0, sigma=np.array([0.3, 0.3]))

    initial_state = np.array([0.0, 0.0, 0.0])

    results = {}
    for name, cost_fn in [("MPCC", mpcc_cost), ("Tracking", tracking_cost)]:
        print(f"\n  Running {name}...")
        controller = MPPIController(model, params, cost_fn)
        state = initial_state.copy()
        states = [state.copy()]

        for step in range(n_steps):
            # 레퍼런스 생성 (S-curve 상의 가장 가까운 점부터)
            idx = min(step, len(waypoints) - N - 2)
            ref = np.zeros((N + 1, 3))
            for i in range(N + 1):
                wp_idx = min(idx + i, len(waypoints) - 1)
                ref[i, 0] = waypoints[wp_idx, 0]
                ref[i, 1] = waypoints[wp_idx, 1]
                if wp_idx < len(waypoints) - 1:
                    ref[i, 2] = np.arctan2(
                        waypoints[wp_idx + 1, 1] - waypoints[wp_idx, 1],
                        waypoints[wp_idx + 1, 0] - waypoints[wp_idx, 0],
                    )

            u, _ = controller.compute_control(state, ref)
            state = model.step(state, u, dt)
            states.append(state.copy())

        states = np.array(states)

        # 경로 추종 오차 계산 (경로에 대한 수직 거리)
        from mppi_controller.controllers.mppi.mpcc_cost import PathParameterization
        path_param = PathParameterization(waypoints)
        _, _, closest = path_param.project(states[:, 0], states[:, 1])
        path_error = np.sqrt(
            (states[:, 0] - closest[:, 0])**2 +
            (states[:, 1] - closest[:, 1])**2
        )
        mean_error = float(np.mean(path_error))
        max_error = float(np.max(path_error))

        results[name] = {
            "states": states,
            "mean_path_error": mean_error,
            "max_path_error": max_error,
        }
        print(f"    Mean path error: {mean_error:.4f}m")
        print(f"    Max path error: {max_error:.4f}m")

    # Plot
    if show_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            # 경로
            ax.plot(waypoints[:, 0], waypoints[:, 1], "k--", linewidth=1, alpha=0.5, label="Path")

            colors = {"MPCC": "red", "Tracking": "blue"}
            for name, data in results.items():
                ax.plot(data["states"][:, 0], data["states"][:, 1],
                        label=f"{name} (err={data['mean_path_error']:.3f}m)",
                        color=colors[name], linewidth=2)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("MPCC vs Tracking Cost (S-Curve)")
            ax.legend()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("safety_s3_mpcc.png", dpi=150)
            print(f"\n  Saved: safety_s3_mpcc.png")
            plt.close()
        except ImportError:
            print("  matplotlib not available, skipping plot")

    return results


def main():
    parser = argparse.ArgumentParser(description="Safety S3 Comparison Demo")
    parser.add_argument(
        "--scenario",
        choices=["backup_cbf", "multi_robot", "mpcc", "all"],
        default="all",
        help="Which scenario to run",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--live", action="store_true", help="Live animation")
    args = parser.parse_args()

    show_plot = not args.no_plot

    print("\n" + "=" * 60)
    print("Safety Phase S3 Comparison Demo".center(60))
    print("=" * 60)

    scenarios = {
        "backup_cbf": run_backup_cbf_scenario,
        "multi_robot": run_multi_robot_scenario,
        "mpcc": run_mpcc_scenario,
    }

    if args.scenario == "all":
        for name, fn in scenarios.items():
            fn(show_plot=show_plot, live=args.live)
    else:
        scenarios[args.scenario](show_plot=show_plot, live=args.live)

    print("\n" + "=" * 60)
    print("Done!".center(60))
    print("=" * 60)


if __name__ == "__main__":
    main()
