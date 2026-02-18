#!/usr/bin/env python3
"""
S4: Multi-Robot Coordination

3-4 로봇이 교차/교환 위치 — 상호 충돌 회피.
No CBF vs Cost-only vs Filter-only vs Cost+Filter 4-way 비교.

Usage:
    python multi_robot_coordination.py
    python multi_robot_coordination.py --no-plot
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
from mppi_controller.controllers.mppi.multi_robot_cbf import (
    MultiRobotCBFCost,
    MultiRobotCBFFilter,
    MultiRobotCoordinator,
    RobotAgent,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig
from common.env_metrics import compute_env_metrics, print_env_comparison

import matplotlib
import matplotlib.pyplot as plt


ROBOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def create_agents(n_robots: int, start_positions, goal_positions, common_params):
    """에이전트 생성"""
    agents = []
    for i in range(n_robots):
        model = DifferentialDriveKinematic(v_max=0.8, omega_max=1.0)
        params = MPPIParams(**common_params)
        controller = MPPIController(model, params)
        agent = RobotAgent(
            id=i,
            state=np.array(start_positions[i]),
            radius=0.25,
            model=model,
            controller=controller,
        )
        agents.append(agent)
    return agents


def run_multi_robot_sim(
    agents, goal_positions, coordinator, duration, dt, N,
    label="",
):
    """다중 로봇 시뮬레이션 실행"""
    num_steps = int(duration / dt)
    n_robots = len(agents)

    # 히스토리
    histories = {i: {"time": [], "state": [], "control": []} for i in range(n_robots)}

    # 초기 상태
    for agent in agents:
        agent.state = agent.state.copy()

    for step in range(num_steps):
        t = step * dt

        # 각 로봇의 레퍼런스 궤적
        ref_trajs = {}
        for i, agent in enumerate(agents):
            goal = goal_positions[i]
            # 간단한 직선 레퍼런스
            ref = np.zeros((N + 1, 3))
            current = agent.state
            direction = goal[:2] - current[:2]
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                heading = np.arctan2(direction[1], direction[0])
                speed = min(0.5, dist / (N * dt))
                for j in range(N + 1):
                    progress = min(1.0, speed * j * dt / max(dist, 1e-6))
                    ref[j, :2] = current[:2] + progress * direction
                    ref[j, 2] = heading
            else:
                ref[:, :2] = goal[:2]
                ref[:, 2] = goal[2] if len(goal) > 2 else 0.0
            ref_trajs[agent.id] = ref

        # 조율 스텝
        results = coordinator.step(ref_trajs)

        # 상태 업데이트
        for agent in agents:
            control, info = results.get(agent.id, (np.zeros(2), {}))
            new_state = agent.model.step(agent.state, control, dt)
            agent.state = new_state
            agent.velocity = np.array([
                control[0] * np.cos(new_state[2]),
                control[0] * np.sin(new_state[2]),
            ])

            histories[agent.id]["time"].append(t)
            histories[agent.id]["state"].append(new_state.copy())
            histories[agent.id]["control"].append(control.copy())

    # numpy 변환
    for i in range(n_robots):
        for key in ["time", "state", "control"]:
            histories[i][key] = np.array(histories[i][key])

    return histories


def run_scenario(no_plot=False, seed=42):
    np.random.seed(seed)

    N = 30
    dt = 0.05
    duration = 12.0
    n_robots = 4

    common_params = dict(
        N=N, dt=dt, K=512, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )

    # 교차 배치: 대각선 교환
    start_positions = [
        [3.0, 3.0, -np.pi * 3 / 4],
        [-3.0, 3.0, -np.pi / 4],
        [-3.0, -3.0, np.pi / 4],
        [3.0, -3.0, np.pi * 3 / 4],
    ]
    goal_positions = [
        np.array([-3.0, -3.0, np.pi / 4]),
        np.array([3.0, -3.0, np.pi * 3 / 4]),
        np.array([3.0, 3.0, -np.pi * 3 / 4]),
        np.array([-3.0, 3.0, -np.pi / 4]),
    ]

    if no_plot:
        matplotlib.use("Agg")

    # 4가지 설정 비교
    configs = [
        ("No CBF", False, False),
        ("Cost Only", True, False),
        ("Filter Only", False, True),
        ("Cost+Filter", True, True),
    ]

    all_results = {}

    for label, use_cost, use_filter in configs:
        print(f"  Running {label}...")
        agents = create_agents(n_robots, start_positions, goal_positions, common_params)
        coordinator = MultiRobotCoordinator(
            agents=agents,
            dt=dt,
            cbf_alpha=0.3,
            safety_margin=0.25,
            use_cost=use_cost,
            use_filter=use_filter,
        )
        histories = run_multi_robot_sim(
            agents, goal_positions, coordinator,
            duration, dt, N, label=label,
        )
        all_results[label] = histories

    # 메트릭 계산
    comparison_metrics = {}
    for label, histories in all_results.items():
        # 로봇 간 최소 거리 계산
        min_inter_dist = float("inf")
        collision_count = 0
        T = len(histories[0]["time"])

        for t_idx in range(T):
            for i in range(n_robots):
                for j in range(i + 1, n_robots):
                    si = histories[i]["state"][t_idx]
                    sj = histories[j]["state"][t_idx]
                    d = np.linalg.norm(si[:2] - sj[:2]) - 0.5  # 2 * radius
                    min_inter_dist = min(min_inter_dist, d)
                    if d < 0:
                        collision_count += 1

        # 목표 도달 거리
        goal_errors = []
        for i in range(n_robots):
            final_state = histories[i]["state"][-1]
            goal_err = np.linalg.norm(final_state[:2] - goal_positions[i][:2])
            goal_errors.append(goal_err)

        comparison_metrics[label] = {
            "collision_count": collision_count,
            "min_clearance": min_inter_dist,
            "safety_rate": 1.0 - collision_count / max(T * n_robots * (n_robots - 1) // 2, 1),
            "path_length": sum(
                float(np.sum(np.sqrt(np.sum(np.diff(histories[i]["state"][:, :2], axis=0) ** 2, axis=1))))
                for i in range(n_robots)
            ) / n_robots,
            "path_efficiency": 1.0,
            "completion_time": None,
            "mean_tracking_error": float(np.mean(goal_errors)),
            "clearances": np.array([min_inter_dist]),
        }

    print_env_comparison(comparison_metrics, title="S4: Multi-Robot Coordination")

    if not no_plot:
        # 2x2 플롯 (각 설정의 XY 궤적)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("S4: Multi-Robot Coordination", fontsize=16, fontweight="bold")

        for idx, (label, histories) in enumerate(all_results.items()):
            ax = axes[idx // 2, idx % 2]
            ax.set_title(f"{label}", fontsize=12)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

            for i in range(n_robots):
                states = histories[i]["state"]
                ax.plot(states[:, 0], states[:, 1],
                        color=ROBOT_COLORS[i], linewidth=2, label=f"Robot {i}")
                ax.plot(states[0, 0], states[0, 1], "o",
                        color=ROBOT_COLORS[i], markersize=8)
                ax.plot(states[-1, 0], states[-1, 1], "s",
                        color=ROBOT_COLORS[i], markersize=8)

                # 목표 표시
                gp = goal_positions[i]
                ax.plot(gp[0], gp[1], "*",
                        color=ROBOT_COLORS[i], markersize=15, alpha=0.5)

            m = comparison_metrics[label]
            ax.text(0.02, 0.98,
                    f"Collisions: {m['collision_count']}\n"
                    f"MinDist: {m['min_clearance']:.3f}m",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.legend(fontsize=7, loc="lower right")

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/s4_multi_robot_coordination.png", dpi=150, bbox_inches="tight")
        print("Plot saved to: plots/s4_multi_robot_coordination.png")
        plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S4: Multi-Robot Coordination")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("S4: Multi-Robot Coordination".center(70))
    print("No CBF vs Cost Only vs Filter Only vs Cost+Filter".center(70))
    print("=" * 70 + "\n")

    run_scenario(no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
