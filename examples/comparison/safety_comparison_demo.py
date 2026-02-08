#!/usr/bin/env python3
"""
Safety-Critical Control Comparison Demo

Compares 5 safety control methods:
  1. Standard CBF (distance-based)
  2. C3BF (Collision Cone CBF — velocity-aware)
  3. DPCBF (Dynamic Parabolic CBF — LoS adaptive boundary)
  4. Optimal-Decay CBF (relaxable safety filter)
  5. Gatekeeper (backup trajectory-based infinite-time safety)

Usage:
    python safety_comparison_demo.py                          # batch mode (save PNG)
    python safety_comparison_demo.py --scenario crossing      # crossing scenario
    python safety_comparison_demo.py --live                   # live animation
    python safety_comparison_demo.py --live --scenario narrow  # live + narrow
    python safety_comparison_demo.py --no-plot
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
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, CBFMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.controllers.mppi.c3bf_cost import CollisionConeCBFCost
from mppi_controller.controllers.mppi.dpcbf_cost import DynamicParabolicCBFCost
from mppi_controller.controllers.mppi.optimal_decay_cbf_filter import (
    OptimalDecayCBFSafetyFilter,
)
from mppi_controller.controllers.mppi.gatekeeper import Gatekeeper
from mppi_controller.controllers.mppi.backup_controller import BrakeBackupController
from mppi_controller.controllers.mppi.superellipsoid_cost import (
    SuperellipsoidObstacle,
    SuperellipsoidCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics


def create_scenario(scenario_type):
    """Create scenario with obstacles, initial state, goal, and duration."""
    if scenario_type == "static":
        # Static obstacle avoidance
        obstacles = [
            (3.0, 0.5, 0.4),
            (5.0, -0.3, 0.3),
            (7.0, 0.2, 0.5),
        ]
        dynamic_obstacles = [(o[0], o[1], o[2], 0.0, 0.0) for o in obstacles]
        initial_state = np.array([0.0, 0.0, 0.0])
        goal = np.array([10.0, 0.0, 0.0])
        duration = 15.0
        return obstacles, dynamic_obstacles, initial_state, goal, duration

    elif scenario_type == "crossing":
        # Crossing dynamic obstacles
        obstacles = [
            (4.0, 2.0, 0.3),   # top to bottom
            (6.0, -2.0, 0.3),  # bottom to top
        ]
        dynamic_obstacles = [
            (4.0, 2.0, 0.3, 0.0, -0.3),
            (6.0, -2.0, 0.3, 0.0, 0.3),
        ]
        initial_state = np.array([0.0, 0.0, 0.0])
        goal = np.array([10.0, 0.0, 0.0])
        duration = 15.0
        return obstacles, dynamic_obstacles, initial_state, goal, duration

    else:  # narrow
        # Narrow passage
        obstacles = [
            (3.0, 1.0, 0.4),
            (3.0, -1.0, 0.4),
            (5.0, 0.8, 0.3),
            (5.0, -0.8, 0.3),
        ]
        dynamic_obstacles = [(o[0], o[1], o[2], 0.0, 0.0) for o in obstacles]
        initial_state = np.array([0.0, 0.0, 0.0])
        goal = np.array([8.0, 0.0, 0.0])
        duration = 15.0
        return obstacles, dynamic_obstacles, initial_state, goal, duration


def make_reference(goal, N, dt, speed=0.5):
    """Generate straight-line reference trajectory toward goal."""
    def ref_fn(t):
        ref = np.zeros((N + 1, 3))
        for i in range(N + 1):
            ti = t + i * dt
            progress = min(ti * speed / max(np.linalg.norm(goal[:2]), 1e-6), 1.0)
            ref[i, :2] = progress * goal[:2]
            ref[i, 2] = np.arctan2(goal[1], goal[0])
        return ref
    return ref_fn


def create_all_controllers(scenario_type):
    """Create all 5 safety controllers (shared by run_comparison / run_live)."""
    from mppi_controller.controllers.mppi.cost_functions import (
        CompositeMPPICost, StateTrackingCost, TerminalCost, ControlEffortCost,
    )

    obstacles, dyn_obstacles, initial_state, goal, duration = create_scenario(scenario_type)
    dt = 0.05
    N = 20
    K = 512

    base_Q = np.array([10.0, 10.0, 1.0])
    base_R = np.array([0.1, 0.1])
    ref_fn = make_reference(goal, N, dt, speed=0.5)

    controllers = []
    colors = ["steelblue", "coral", "seagreen", "orchid", "goldenrod"]
    names = ["Standard CBF", "C3BF (Cone)", "DPCBF (Parabolic)",
             "Optimal-Decay", "Gatekeeper"]

    # Independent model per controller (state isolation)
    def make_model():
        return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # 1. Standard CBF
    params_cbf = CBFMPPIParams(
        N=N, dt=dt, K=K, lambda_=1.0,
        sigma=np.array([0.5, 0.5]), Q=base_Q, R=base_R,
        cbf_obstacles=obstacles, cbf_weight=1000.0, cbf_alpha=0.1,
        cbf_safety_margin=0.15, cbf_use_safety_filter=False,
    )
    controllers.append({
        "name": names[0], "color": colors[0],
        "controller": CBFMPPIController(make_model(), params_cbf),
        "model": make_model(), "gatekeeper": None,
    })

    # 2. C3BF
    c3bf_cost = CollisionConeCBFCost(
        obstacles=dyn_obstacles, cbf_weight=1000.0, safety_margin=0.15, dt=dt,
    )
    params_c3bf = MPPIParams(
        N=N, dt=dt, K=K, lambda_=1.0,
        sigma=np.array([0.5, 0.5]), Q=base_Q, R=base_R,
    )
    c3bf_composite = CompositeMPPICost([
        StateTrackingCost(base_Q), TerminalCost(base_Q), ControlEffortCost(base_R),
        c3bf_cost,
    ])
    controllers.append({
        "name": names[1], "color": colors[1],
        "controller": MPPIController(make_model(), params_c3bf, cost_function=c3bf_composite),
        "model": make_model(), "gatekeeper": None,
    })

    # 3. DPCBF
    dpcbf_cost = DynamicParabolicCBFCost(
        obstacles=dyn_obstacles, cbf_weight=1000.0,
        safety_margin=0.15, a_base=0.3, a_vel=0.5, dt=dt,
    )
    dpcbf_composite = CompositeMPPICost([
        StateTrackingCost(base_Q), TerminalCost(base_Q), ControlEffortCost(base_R),
        dpcbf_cost,
    ])
    params_dpcbf = MPPIParams(
        N=N, dt=dt, K=K, lambda_=1.0,
        sigma=np.array([0.5, 0.5]), Q=base_Q, R=base_R,
    )
    controllers.append({
        "name": names[2], "color": colors[2],
        "controller": MPPIController(make_model(), params_dpcbf, cost_function=dpcbf_composite),
        "model": make_model(), "gatekeeper": None,
    })

    # 4. Optimal-Decay CBF
    params_od = CBFMPPIParams(
        N=N, dt=dt, K=K, lambda_=1.0,
        sigma=np.array([0.5, 0.5]), Q=base_Q, R=base_R,
        cbf_obstacles=obstacles, cbf_weight=1000.0, cbf_alpha=0.1,
        cbf_safety_margin=0.15, cbf_use_safety_filter=True,
    )
    ctrl_od = CBFMPPIController(make_model(), params_od)
    ctrl_od.safety_filter = OptimalDecayCBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.1, safety_margin=0.15,
        penalty_weight=1e4,
    )
    controllers.append({
        "name": names[3], "color": colors[3],
        "controller": ctrl_od,
        "model": make_model(), "gatekeeper": None,
    })

    # 5. Gatekeeper
    params_gk = CBFMPPIParams(
        N=N, dt=dt, K=K, lambda_=1.0,
        sigma=np.array([0.5, 0.5]), Q=base_Q, R=base_R,
        cbf_obstacles=obstacles, cbf_weight=1000.0, cbf_alpha=0.1,
        cbf_safety_margin=0.15, cbf_use_safety_filter=False,
    )
    gk_model = make_model()
    gatekeeper = Gatekeeper(
        backup_controller=BrakeBackupController(),
        model=gk_model, obstacles=obstacles,
        safety_margin=0.2, backup_horizon=30, dt=dt,
    )
    controllers.append({
        "name": names[4], "color": colors[4],
        "controller": CBFMPPIController(make_model(), params_gk),
        "model": make_model(), "gatekeeper": gatekeeper,
    })

    return controllers, obstacles, dyn_obstacles, initial_state, goal, duration, ref_fn, dt


def run_method(name, controller, model, ref_fn, initial_state, duration, dt,
               obstacles=None, gatekeeper=None):
    """Run a single method and collect results."""
    sim = Simulator(model, controller, dt=dt)
    sim.reset(initial_state)

    state = initial_state.copy()
    states = [state.copy()]
    controls_log = []
    solve_times = []
    min_distances = []

    n_steps = int(duration / dt)
    for step in range(n_steps):
        t = step * dt
        ref = ref_fn(t)

        t0 = time.time()
        control, info = controller.compute_control(state, ref)

        # Apply Gatekeeper
        if gatekeeper is not None:
            control, gk_info = gatekeeper.filter(state, control)

        solve_times.append(time.time() - t0)

        state = model.step(state, control, dt)
        states.append(state.copy())
        controls_log.append(control.copy())

        # Min obstacle distance
        if obstacles:
            dists = [np.sqrt((state[0]-o[0])**2 + (state[1]-o[1])**2) - o[2]
                     for o in obstacles]
            min_distances.append(min(dists))

    states = np.array(states)
    mean_solve = np.mean(solve_times) * 1000

    # Distance to goal
    final_dist = np.linalg.norm(states[-1, :2] - np.array([10.0, 0.0]))

    # Min obstacle clearance
    min_clearance = min(min_distances) if min_distances else float('inf')
    collision = min_clearance < 0

    return {
        "name": name,
        "states": states,
        "mean_solve_ms": mean_solve,
        "final_dist": final_dist,
        "min_clearance": min_clearance,
        "collision": collision,
    }


def run_comparison(scenario_type, plot=True):
    """Compare 5 safety-critical control methods (batch mode)."""

    print("\n" + "=" * 70)
    print("Safety-Critical Control Comparison".center(70))
    print("=" * 70)
    print(f"  Scenario: {scenario_type}")
    print("=" * 70 + "\n")

    ctrls, obstacles, dyn_obstacles, initial_state, goal, duration, ref_fn, dt = \
        create_all_controllers(scenario_type)

    results = []
    for i, c in enumerate(ctrls):
        print(f"  [{i+1}/5] {c['name']}...")
        results.append(run_method(
            c["name"], c["controller"], c["model"], ref_fn,
            initial_state, duration, dt, obstacles, gatekeeper=c["gatekeeper"],
        ))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Results".center(70))
    print("=" * 70)
    print(f"{'Method':>20s} | {'Solve(ms)':>10s} | {'Goal Dist':>10s} | "
          f"{'Min Clear':>10s} | {'Collision':>10s}")
    print("─" * 70)

    for r in results:
        coll_str = "YES!" if r["collision"] else "No"
        print(f"{r['name']:>20s} | {r['mean_solve_ms']:>10.2f} | "
              f"{r['final_dist']:>10.3f} | {r['min_clearance']:>10.3f} | "
              f"{coll_str:>10s}")

    print("=" * 70)

    # ── Plot ──
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(
                f"Safety-Critical Control Comparison ({scenario_type})",
                fontsize=14,
            )

            colors = ["steelblue", "coral", "seagreen", "orchid", "goldenrod"]

            for idx, r in enumerate(results):
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]

                # Draw obstacles
                for obs in obstacles:
                    circle = plt.Circle(
                        (obs[0], obs[1]), obs[2], color="red", alpha=0.3
                    )
                    ax.add_patch(circle)
                    # Safety margin
                    safe_circle = plt.Circle(
                        (obs[0], obs[1]), obs[2] + 0.15, color="red",
                        alpha=0.1, linestyle="--", fill=False,
                    )
                    ax.add_patch(safe_circle)

                # Trajectory
                ax.plot(
                    r["states"][:, 0], r["states"][:, 1],
                    color=colors[idx], linewidth=2,
                )
                ax.plot(
                    r["states"][0, 0], r["states"][0, 1],
                    "o", color=colors[idx], markersize=8,
                )
                ax.plot(
                    r["states"][-1, 0], r["states"][-1, 1],
                    "s", color=colors[idx], markersize=8,
                )

                # Goal
                ax.plot(goal[0], goal[1], "r*", markersize=15)

                ax.set_title(
                    f"{r['name']}\n"
                    f"Solve: {r['mean_solve_ms']:.1f}ms, "
                    f"Clear: {r['min_clearance']:.3f}m",
                    fontsize=10,
                )
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-1, 12)
                ax.set_ylim(-3, 3)

            # 6th subplot: comparison bar chart
            ax = axes[1, 2]
            names = [r["name"] for r in results]
            clearances = [r["min_clearance"] for r in results]
            bar_colors = ["red" if r["collision"] else c
                          for r, c in zip(results, colors)]
            bars = ax.barh(names, clearances, color=bar_colors, alpha=0.7)
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("Min Clearance (m)")
            ax.set_title("Safety Comparison")
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()

            out_dir = os.path.join(os.path.dirname(__file__), "../../results")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"safety_comparison_{scenario_type}.png")
            plt.savefig(out_path, dpi=150)
            print(f"\nPlot saved: {out_path}")
            plt.close()

        except ImportError:
            print("\nmatplotlib not available, skipping plot.")

    return results


def run_live(scenario_type):
    """Live 2x3 animation (5 methods + comparison chart)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    ctrls, obstacles, dyn_obstacles, initial_state, goal, duration, ref_fn, dt = \
        create_all_controllers(scenario_type)

    n_steps = int(duration / dt)
    is_dynamic = (scenario_type == "crossing")

    # Dynamic obstacle velocities (crossing scenario)
    obs_velocities = [(o[3], o[4]) for o in dyn_obstacles] if is_dynamic else []

    # Per-controller simulation state
    sim_data = []
    for c in ctrls:
        sim_data.append({
            "state": initial_state.copy(),
            "xy_history": [initial_state[:2].copy()],
            "min_clearances": [],
        })

    # Current obstacle positions (for dynamic scenario)
    current_obstacles = [(o[0], o[1], o[2]) for o in obstacles]

    # ===== Figure setup (2x3) =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Safety-Critical Control Comparison — {scenario_type} (Live)",
        fontsize=14, fontweight="bold",
    )

    # Initialize 5 subplots
    traj_lines = []
    pos_markers = []
    obs_patches_per_ax = []  # obstacle patches per axis
    title_texts = []

    for idx, c in enumerate(ctrls):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 12)
        ax.set_ylim(-3, 3)

        # Goal marker
        ax.plot(goal[0], goal[1], "r*", markersize=15, zorder=5)

        # Obstacle patches
        ax_patches = []
        for obs in current_obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.3)
            margin = plt.Circle(
                (obs[0], obs[1]), obs[2] + 0.15,
                color="red", alpha=0.1, linestyle="--", fill=False,
            )
            ax.add_patch(circle)
            ax.add_patch(margin)
            ax_patches.append((circle, margin))
        obs_patches_per_ax.append(ax_patches)

        # Trajectory line + current position marker
        line, = ax.plot([], [], color=c["color"], linewidth=2)
        dot, = ax.plot([], [], "o", color=c["color"], markersize=8, zorder=4)
        # Start marker
        ax.plot(initial_state[0], initial_state[1], "ko", markersize=6, zorder=3)
        traj_lines.append(line)
        pos_markers.append(dot)

        title_text = ax.set_title(f"{c['name']}\nClearance: ---, Solve: ---")
        title_texts.append(title_text)

    # 6th panel: real-time min clearance comparison graph
    ax_cmp = axes[1, 2]
    ax_cmp.set_xlabel("Time (s)")
    ax_cmp.set_ylabel("Min Clearance (m)")
    ax_cmp.set_title("Min Clearance Comparison")
    ax_cmp.grid(True, alpha=0.3)
    ax_cmp.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    clearance_lines = []
    for c in ctrls:
        cline, = ax_cmp.plot([], [], color=c["color"], linewidth=2, label=c["name"])
        clearance_lines.append(cline)
    ax_cmp.legend(fontsize=7, loc="upper right")

    time_text = fig.text(
        0.5, 0.01, "", ha="center", fontsize=10, family="monospace",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    times_log = []

    def init():
        for line in traj_lines:
            line.set_data([], [])
        for dot in pos_markers:
            dot.set_data([], [])
        for cline in clearance_lines:
            cline.set_data([], [])
        time_text.set_text("")
        return []

    def update(frame):
        nonlocal current_obstacles
        if frame >= n_steps:
            return []

        t_current = frame * dt
        times_log.append(t_current)

        # Update dynamic obstacle positions (crossing)
        if is_dynamic:
            current_obstacles = []
            current_dyn = []
            for i, obs_base in enumerate(obstacles):
                vx, vy = obs_velocities[i]
                nx = obs_base[0] + vx * t_current
                ny = obs_base[1] + vy * t_current
                r = obs_base[2]
                current_obstacles.append((nx, ny, r))
                current_dyn.append((nx, ny, r, vx, vy))

            # Update obstacle patches (all 5 axes)
            for ax_patches in obs_patches_per_ax:
                for j, (circle, margin) in enumerate(ax_patches):
                    ox, oy, r = current_obstacles[j]
                    circle.center = (ox, oy)
                    margin.center = (ox, oy)

        # Step all 5 controllers
        for idx, c in enumerate(ctrls):
            state = sim_data[idx]["state"]
            ctrl = c["controller"]
            model = c["model"]
            gk = c["gatekeeper"]

            # Update dynamic obstacles
            if is_dynamic:
                if hasattr(ctrl, "update_obstacles"):
                    ctrl.update_obstacles(current_obstacles)
                # Update obstacles inside C3BF/DPCBF cost functions
                if hasattr(ctrl, "cost_function") and hasattr(ctrl.cost_function, "costs"):
                    for cost_fn in ctrl.cost_function.costs:
                        if hasattr(cost_fn, "update_obstacles"):
                            cost_fn.update_obstacles(current_dyn)
                if gk is not None:
                    gk.update_obstacles(current_obstacles)

            ref = ref_fn(t_current)
            t0 = time.time()
            control, info = ctrl.compute_control(state, ref)
            if gk is not None:
                control, _ = gk.filter(state, control)
            solve_ms = (time.time() - t0) * 1000

            state = model.step(state, control, dt)
            sim_data[idx]["state"] = state

            sim_data[idx]["xy_history"].append(state[:2].copy())

            # Min obstacle distance
            dists = [np.sqrt((state[0]-o[0])**2 + (state[1]-o[1])**2) - o[2]
                     for o in current_obstacles]
            min_clear = min(dists) if dists else float('inf')
            sim_data[idx]["min_clearances"].append(min_clear)

            # Update trajectory line
            xy = np.array(sim_data[idx]["xy_history"])
            traj_lines[idx].set_data(xy[:, 0], xy[:, 1])
            pos_markers[idx].set_data([state[0]], [state[1]])

            # Update title
            row, col = idx // 3, idx % 3
            axes[row, col].set_title(
                f"{c['name']}\n"
                f"Clearance: {min_clear:.3f}m, Solve: {solve_ms:.1f}ms",
                fontsize=10,
            )

        # 6th panel: clearance graph update
        times_arr = np.array(times_log)
        for idx in range(len(ctrls)):
            clearance_lines[idx].set_data(
                times_arr, sim_data[idx]["min_clearances"]
            )
        ax_cmp.relim()
        ax_cmp.autoscale_view()

        # Time text
        time_text.set_text(
            f"t = {t_current:.1f}s / {duration:.0f}s  |  "
            + "  ".join(
                f"{ctrls[i]['name']}: {sim_data[i]['min_clearances'][-1]:.3f}m"
                for i in range(len(ctrls))
            )
        )

        return []

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=n_steps, interval=1, blit=False, repeat=False,
    )
    plt.show()

    # Post-simulation summary
    print("\n" + "=" * 70)
    print("Live Simulation Results".center(70))
    print("=" * 70)
    print(f"{'Method':>20s} | {'Min Clearance':>14s} | {'Collision':>10s}")
    print("─" * 50)
    for idx, c in enumerate(ctrls):
        clears = sim_data[idx]["min_clearances"]
        min_c = min(clears) if clears else float('inf')
        coll = "YES!" if min_c < 0 else "No"
        print(f"{c['name']:>20s} | {min_c:>14.3f} | {coll:>10s}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Safety-Critical Control Comparison")
    parser.add_argument(
        "--scenario", type=str, default="static",
        choices=["static", "crossing", "narrow"],
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    args = parser.parse_args()

    if args.live:
        run_live(args.scenario)
    else:
        run_comparison(args.scenario, plot=not args.no_plot)


if __name__ == "__main__":
    main()
