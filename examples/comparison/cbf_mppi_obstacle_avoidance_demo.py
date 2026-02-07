#!/usr/bin/env python3
"""
CBF-MPPI 장애물 회피 비교 데모

3-way 비교:
1. Vanilla MPPI (장애물 인식 없음)
2. MPPI + ObstacleCost (기존 지수 페널티)
3. MPPI + CBF (Control Barrier Function 비용)

Usage:
    python cbf_mppi_obstacle_avoidance_demo.py
    python cbf_mppi_obstacle_avoidance_demo.py --duration 15 --seed 42
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, CBFMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_simulation(model, controller, params, trajectory_fn, duration, label):
    """시뮬레이션 실행 및 결과 반환"""
    sim = Simulator(model, controller, params.dt)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    initial_state = trajectory_fn(0.0)
    sim.reset(initial_state)
    print(f"  Running {label}...")

    try:
        history = sim.run(reference_fn, duration, realtime=False)
        metrics = compute_metrics(history)
        print(f"  ✓ {label} completed (RMSE: {metrics['position_rmse']:.4f}m)")
        return history, metrics, True
    except Exception as e:
        print(f"  ✗ {label} FAILED: {e}")
        return None, None, False


def compute_min_distances(states, obstacles):
    """각 시간스텝에서 가장 가까운 장애물까지 거리 계산"""
    min_distances = np.full(len(states), np.inf)
    for obs_x, obs_y, obs_r in obstacles:
        distances = np.sqrt(
            (states[:, 0] - obs_x) ** 2 + (states[:, 1] - obs_y) ** 2
        ) - obs_r
        min_distances = np.minimum(min_distances, distances)
    return min_distances


def main():
    parser = argparse.ArgumentParser(
        description="CBF-MPPI Obstacle Avoidance Comparison"
    )
    parser.add_argument(
        "--duration", type=float, default=12.0, help="Simulation duration (s)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("CBF-MPPI Obstacle Avoidance Comparison".center(80))
    print("=" * 80)

    # 장애물 정의
    obstacles = [
        (5.0, 0.0, 0.8),
        (0.0, 5.0, 0.6),
        (-3.5, 3.5, 0.7),
    ]

    print(f"Duration: {args.duration}s")
    print(f"Obstacles: {len(obstacles)}")
    for i, (x, y, r) in enumerate(obstacles):
        print(f"  [{i}] pos=({x:.1f}, {y:.1f}), r={r:.1f}")
    print("=" * 80 + "\n")

    # 공통 파라미터
    common_kwargs = dict(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )

    trajectory_fn = create_trajectory_function("circle", radius=5.0)

    # ==================== 1. Vanilla MPPI ====================
    print("[1/3] Vanilla MPPI (no obstacle awareness)")
    vanilla_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    vanilla_params = MPPIParams(**common_kwargs)
    vanilla_controller = MPPIController(vanilla_model, vanilla_params)
    vanilla_history, vanilla_metrics, vanilla_ok = run_simulation(
        vanilla_model, vanilla_controller, vanilla_params,
        trajectory_fn, args.duration, "Vanilla MPPI"
    )

    # ==================== 2. MPPI + ObstacleCost ====================
    print("\n[2/3] MPPI + ObstacleCost (exponential penalty)")
    obstacle_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacle_params = MPPIParams(**common_kwargs)
    obstacle_cost = CompositeMPPICost([
        StateTrackingCost(obstacle_params.Q),
        TerminalCost(obstacle_params.Qf),
        ControlEffortCost(obstacle_params.R),
        ObstacleCost(obstacles, safety_margin=0.2, cost_weight=100.0),
    ])
    obstacle_controller = MPPIController(
        obstacle_model, obstacle_params, cost_function=obstacle_cost
    )
    obstacle_history, obstacle_metrics, obstacle_ok = run_simulation(
        obstacle_model, obstacle_controller, obstacle_params,
        trajectory_fn, args.duration, "MPPI + ObstacleCost"
    )

    # ==================== 3. MPPI + CBF ====================
    print("\n[3/3] MPPI + CBF (Control Barrier Function)")
    cbf_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    cbf_params = CBFMPPIParams(
        **common_kwargs,
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.1,
        cbf_safety_margin=0.1,
        cbf_use_safety_filter=False,
    )
    cbf_controller = CBFMPPIController(cbf_model, cbf_params)
    cbf_history, cbf_metrics, cbf_ok = run_simulation(
        cbf_model, cbf_controller, cbf_params,
        trajectory_fn, args.duration, "MPPI + CBF"
    )

    # ==================== 메트릭 출력 ====================
    print("\n")
    if vanilla_ok:
        print_metrics(vanilla_metrics, title="Vanilla MPPI")
    if obstacle_ok:
        print_metrics(obstacle_metrics, title="MPPI + ObstacleCost")
    if cbf_ok:
        print_metrics(cbf_metrics, title="MPPI + CBF")

    if cbf_ok:
        cbf_stats = cbf_controller.get_cbf_statistics()
        print("=" * 60)
        print("CBF Statistics".center(60))
        print("=" * 60)
        print(f"  Mean min barrier: {cbf_stats['mean_min_barrier']:.4f}")
        print(f"  Min min barrier: {cbf_stats['min_min_barrier']:.4f}")
        print(f"  Safety rate: {cbf_stats['safety_rate']:.2%}")
        print("=" * 60 + "\n")

    # ==================== 시각화 ====================
    if not all([vanilla_ok, obstacle_ok, cbf_ok]):
        print("Some simulations failed. Skipping plots.")
        return

    print("Generating comparison plots...\n")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        "CBF-MPPI Obstacle Avoidance Comparison",
        fontsize=16, fontweight="bold",
    )

    vanilla_states = vanilla_history["state"]
    obstacle_states = obstacle_history["state"]
    cbf_states = cbf_history["state"]
    vanilla_refs = vanilla_history["reference"]
    times = vanilla_history["time"]

    colors = {"vanilla": "#1f77b4", "obstacle": "#ff7f0e", "cbf": "#2ca02c"}

    # ==================== [0,0] XY 궤적 + 장애물 ====================
    ax = axes[0, 0]
    # 레퍼런스
    ax.plot(
        vanilla_refs[:, 0], vanilla_refs[:, 1],
        "k--", label="Reference", linewidth=1.5, alpha=0.5,
    )
    # 궤적
    ax.plot(
        vanilla_states[:, 0], vanilla_states[:, 1],
        color=colors["vanilla"], label="Vanilla", linewidth=2, alpha=0.8,
    )
    ax.plot(
        obstacle_states[:, 0], obstacle_states[:, 1],
        color=colors["obstacle"], label="ObstacleCost", linewidth=2, alpha=0.8,
    )
    ax.plot(
        cbf_states[:, 0], cbf_states[:, 1],
        color=colors["cbf"], label="CBF", linewidth=2, alpha=0.8,
    )
    # 장애물
    for obs_x, obs_y, obs_r in obstacles:
        circle = plt.Circle(
            (obs_x, obs_y), obs_r, color="red", alpha=0.3, label="_nolegend_"
        )
        ax.add_patch(circle)
        margin_circle = plt.Circle(
            (obs_x, obs_y), obs_r + 0.1, color="red",
            alpha=0.1, linestyle="--", fill=False, label="_nolegend_"
        )
        ax.add_patch(margin_circle)
        ax.plot(obs_x, obs_y, "rx", markersize=8)
    # 시작점
    ax.plot(vanilla_states[0, 0], vanilla_states[0, 1], "ko", markersize=8, label="Start")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories + Obstacles")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # ==================== [0,1] 위치 오차 ====================
    ax = axes[0, 1]
    vanilla_errors = np.linalg.norm(
        vanilla_states[:, :2] - vanilla_refs[:, :2], axis=1
    )
    obstacle_errors = np.linalg.norm(
        obstacle_states[:, :2] - vanilla_refs[:, :2], axis=1
    )
    cbf_errors = np.linalg.norm(
        cbf_states[:, :2] - vanilla_refs[:, :2], axis=1
    )

    ax.plot(times, vanilla_errors, color=colors["vanilla"], label="Vanilla", linewidth=2)
    ax.plot(times, obstacle_errors, color=colors["obstacle"], label="ObstacleCost", linewidth=2)
    ax.plot(times, cbf_errors, color=colors["cbf"], label="CBF", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ==================== [1,0] Minimum Barrier Value ====================
    ax = axes[1, 0]
    if cbf_ok:
        cbf_infos = cbf_history["info"]
        min_barriers = [info.get("min_barrier", 0.0) for info in cbf_infos]
        ax.plot(times, min_barriers, color=colors["cbf"], linewidth=2, label="CBF min barrier")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="h=0 (boundary)")
        ax.fill_between(
            times, min_barriers, 0,
            where=[b < 0 for b in min_barriers],
            color="red", alpha=0.2, label="Unsafe region"
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Barrier Value h(x)")
    ax.set_title("Minimum Barrier Value (CBF)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ==================== [1,1] 제어 입력 ====================
    ax = axes[1, 1]
    vanilla_controls = vanilla_history["control"]
    obstacle_controls = obstacle_history["control"]
    cbf_controls = cbf_history["control"]

    ax.plot(times, vanilla_controls[:, 0], color=colors["vanilla"],
            linewidth=1.5, alpha=0.7, label="Vanilla v")
    ax.plot(times, obstacle_controls[:, 0], color=colors["obstacle"],
            linewidth=1.5, alpha=0.7, label="ObstacleCost v")
    ax.plot(times, cbf_controls[:, 0], color=colors["cbf"],
            linewidth=1.5, alpha=0.7, label="CBF v")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("Control Inputs (Linear Velocity)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ==================== [2,0] 최소 장애물 거리 ====================
    ax = axes[2, 0]
    vanilla_min_dist = compute_min_distances(vanilla_states, obstacles)
    obstacle_min_dist = compute_min_distances(obstacle_states, obstacles)
    cbf_min_dist = compute_min_distances(cbf_states, obstacles)

    ax.plot(times, vanilla_min_dist, color=colors["vanilla"], label="Vanilla", linewidth=2)
    ax.plot(times, obstacle_min_dist, color=colors["obstacle"], label="ObstacleCost", linewidth=2)
    ax.plot(times, cbf_min_dist, color=colors["cbf"], label="CBF", linewidth=2)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Collision")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min Distance to Obstacle (m)")
    ax.set_title("Minimum Distance to Nearest Obstacle")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ==================== [2,1] 요약 메트릭 ====================
    ax = axes[2, 1]
    ax.axis("off")

    # 최소 거리 계산
    v_min_d = np.min(vanilla_min_dist)
    o_min_d = np.min(obstacle_min_dist)
    c_min_d = np.min(cbf_min_dist)

    # 충돌 여부
    v_collision = "YES" if v_min_d < 0 else "NO"
    o_collision = "YES" if o_min_d < 0 else "NO"
    c_collision = "YES" if c_min_d < 0 else "NO"

    summary_text = f"""
    ┌──────────────────────────────────────────────┐
    │           Comparison Summary                 │
    ├──────────────────────────────────────────────┤
    │                                              │
    │  Vanilla MPPI:                               │
    │    RMSE:      {vanilla_metrics['position_rmse']:8.4f} m                 │
    │    Time:      {vanilla_metrics['mean_solve_time']:8.2f} ms                │
    │    Min dist:  {v_min_d:8.4f} m                 │
    │    Collision: {v_collision:>3s}                           │
    │                                              │
    │  MPPI + ObstacleCost:                        │
    │    RMSE:      {obstacle_metrics['position_rmse']:8.4f} m                 │
    │    Time:      {obstacle_metrics['mean_solve_time']:8.2f} ms                │
    │    Min dist:  {o_min_d:8.4f} m                 │
    │    Collision: {o_collision:>3s}                           │
    │                                              │
    │  MPPI + CBF:                                 │
    │    RMSE:      {cbf_metrics['position_rmse']:8.4f} m                 │
    │    Time:      {cbf_metrics['mean_solve_time']:8.2f} ms                │
    │    Min dist:  {c_min_d:8.4f} m                 │
    │    Collision: {c_collision:>3s}                           │
    │    Safety:    {cbf_stats['safety_rate']:7.1%}                   │
    │                                              │
    │  Obstacles: {len(obstacles)} (circle trajectory)          │
    │  Duration:  {args.duration:.1f}s, Seed: {args.seed}                │
    └──────────────────────────────────────────────┘
    """

    ax.text(
        0.05, 0.5, summary_text,
        fontsize=9, verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    # 저장
    os.makedirs("plots", exist_ok=True)
    output_path = "plots/cbf_mppi_obstacle_avoidance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
