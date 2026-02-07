#!/usr/bin/env python3
"""
동적 장애물 회피 데모

3-way 비교 (이동 장애물):
1. Vanilla MPPI (장애물 인식 없음)
2. CBF-MPPI (Approach A: CBF 비용 페널티 + 실시간 업데이트)
3. Shield-MPPI (Approach C: Shielded Rollout + 실시간 업데이트)

매 timestep마다 controller.update_obstacles()를 호출하여
이동하는 장애물의 현재 위치를 반영.

Usage:
    python dynamic_obstacle_avoidance_demo.py
    python dynamic_obstacle_avoidance_demo.py --live
    python dynamic_obstacle_avoidance_demo.py --duration 15 --seed 42
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    CBFMPPIParams,
    ShieldMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ===== 이동 장애물 =====

class MovingObstacle:
    """이동 장애물 시뮬레이션"""

    def __init__(self, start, velocity, radius, motion_type="linear"):
        """
        Args:
            start: (x, y) 초기 위치
            velocity: (vx, vy) 선형 속도 또는 (cx, cy, r, omega) 원형 운동
            radius: 장애물 반경
            motion_type: 'linear' 또는 'circular'
        """
        self.start = np.array(start, dtype=float)
        self.velocity = velocity
        self.radius = radius
        self.motion_type = motion_type

    def get_position(self, t):
        """시간 t에서의 위치"""
        if self.motion_type == "linear":
            vx, vy = self.velocity
            return (
                self.start[0] + vx * t,
                self.start[1] + vy * t,
            )
        elif self.motion_type == "circular":
            cx, cy, r, omega = self.velocity
            angle = omega * t
            return (
                cx + r * np.cos(angle),
                cy + r * np.sin(angle),
            )
        return tuple(self.start)

    def as_tuple(self, t):
        """(x, y, radius) 튜플 반환"""
        px, py = self.get_position(t)
        return (px, py, self.radius)


# 이동 장애물 정의 (전역)
MOVING_OBSTACLES = [
    MovingObstacle(
        start=(4.0, -2.0), velocity=(0.0, 0.6),
        radius=0.6, motion_type="linear",
    ),
    MovingObstacle(
        start=(-2.0, 4.0), velocity=(0.5, -0.3),
        radius=0.5, motion_type="linear",
    ),
    MovingObstacle(
        start=(0.0, 0.0),
        velocity=(0.0, 0.0, 3.5, 0.5),  # (cx, cy, r, omega)
        radius=0.7, motion_type="circular",
    ),
]

COLORS = {
    "vanilla": "#1f77b4",
    "cbf": "#ff7f0e",
    "shield": "#2ca02c",
}


def get_obstacles_at(t):
    """시간 t에서 모든 장애물의 (x, y, r) 리스트"""
    return [obs.as_tuple(t) for obs in MOVING_OBSTACLES]


def compute_min_distance_single(state, obstacles):
    """단일 상태에서 최소 장애물 거리"""
    min_d = np.inf
    for ox, oy, r in obstacles:
        d = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2) - r
        min_d = min(min_d, d)
    return min_d


def create_controllers(common_kwargs, cbf_kwargs):
    """3개 컨트롤러 + 시뮬레이터 생성"""
    vanilla_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    cbf_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    shield_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    vanilla_params = MPPIParams(**common_kwargs)
    cbf_params = CBFMPPIParams(**common_kwargs, **cbf_kwargs)
    shield_params = ShieldMPPIParams(**common_kwargs, **cbf_kwargs, shield_enabled=True)

    vanilla_ctrl = MPPIController(vanilla_model, vanilla_params)
    cbf_ctrl = CBFMPPIController(cbf_model, cbf_params)
    shield_ctrl = ShieldMPPIController(shield_model, shield_params)

    dt = common_kwargs["dt"]

    return {
        "vanilla": {
            "model": vanilla_model, "controller": vanilla_ctrl,
            "sim": Simulator(vanilla_model, vanilla_ctrl, dt),
        },
        "cbf": {
            "model": cbf_model, "controller": cbf_ctrl,
            "sim": Simulator(cbf_model, cbf_ctrl, dt),
        },
        "shield": {
            "model": shield_model, "controller": shield_ctrl,
            "sim": Simulator(shield_model, shield_ctrl, dt),
        },
    }


def run_live(common_kwargs, cbf_kwargs, trajectory_fn, duration):
    """실시간 4패널 애니메이션"""
    ctrls = create_controllers(common_kwargs, cbf_kwargs)
    dt = common_kwargs["dt"]
    N = common_kwargs["N"]
    num_steps = int(duration / dt)

    initial_state = trajectory_fn(0.0)
    for key in ctrls:
        ctrls[key]["sim"].reset(initial_state)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, N, dt)

    # 데이터 저장
    data = {
        k: {"xy": [], "times": [], "errors": [],
             "min_dist": [], "barriers": [], "shield_rates": []}
        for k in ctrls
    }

    # ===== Figure 설정 (2x2 4패널) =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Dynamic Obstacle Avoidance: Vanilla vs CBF vs Shield",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY 궤적 + 이동 장애물
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories + Moving Obstacles")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    # 레퍼런스 궤적
    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    # 장애물 패치 (애니메이션 업데이트)
    obs_patches = []
    obs_margins = []
    obs_markers = []
    for _ in MOVING_OBSTACLES:
        patch = plt.Circle((0, 0), 0.5, color="red", alpha=0.3)
        margin = plt.Circle((0, 0), 0.6, color="red", alpha=0.1, linestyle="--", fill=False)
        ax_xy.add_patch(patch)
        ax_xy.add_patch(margin)
        marker, = ax_xy.plot([], [], "rx", markersize=8)
        obs_patches.append(patch)
        obs_margins.append(margin)
        obs_markers.append(marker)

    # 궤적 라인
    line_v_xy, = ax_xy.plot([], [], color=COLORS["vanilla"], linewidth=2, label="Vanilla")
    line_c_xy, = ax_xy.plot([], [], color=COLORS["cbf"], linewidth=2, label="CBF")
    line_s_xy, = ax_xy.plot([], [], color=COLORS["shield"], linewidth=2, label="Shield")
    dot_v, = ax_xy.plot([], [], "o", color=COLORS["vanilla"], markersize=8)
    dot_c, = ax_xy.plot([], [], "o", color=COLORS["cbf"], markersize=8)
    dot_s, = ax_xy.plot([], [], "o", color=COLORS["shield"], markersize=8)
    ax_xy.legend(loc="upper left", fontsize=7)

    # [0,1] 최소 장애물 거리
    ax_dist = axes[0, 1]
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Min Distance (m)")
    ax_dist.set_title("Min Distance to Obstacle")
    ax_dist.grid(True, alpha=0.3)
    ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    line_v_dist, = ax_dist.plot([], [], color=COLORS["vanilla"], linewidth=2, label="Vanilla")
    line_c_dist, = ax_dist.plot([], [], color=COLORS["cbf"], linewidth=2, label="CBF")
    line_s_dist, = ax_dist.plot([], [], color=COLORS["shield"], linewidth=2, label="Shield")
    ax_dist.legend(fontsize=7)

    # [1,0] 위치 추적 오차
    ax_err = axes[1, 0]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Position Tracking Error")
    ax_err.grid(True, alpha=0.3)
    line_v_err, = ax_err.plot([], [], color=COLORS["vanilla"], linewidth=2, label="Vanilla")
    line_c_err, = ax_err.plot([], [], color=COLORS["cbf"], linewidth=2, label="CBF")
    line_s_err, = ax_err.plot([], [], color=COLORS["shield"], linewidth=2, label="Shield")
    ax_err.legend(fontsize=7)

    # [1,1] Shield 개입률 + Barrier
    ax_shield = axes[1, 1]
    ax_shield.set_xlabel("Time (s)")
    ax_shield.set_ylabel("Intervention Rate / Barrier")
    ax_shield.set_title("Shield Intervention & Barrier Value")
    ax_shield.grid(True, alpha=0.3)
    ax_shield.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    line_rate, = ax_shield.plot([], [], color=COLORS["shield"], linewidth=2, label="Intervention Rate")
    line_c_bar, = ax_shield.plot([], [], color=COLORS["cbf"], linewidth=1.5, alpha=0.7, linestyle="--", label="CBF barrier")
    line_s_bar, = ax_shield.plot([], [], color=COLORS["shield"], linewidth=1.5, alpha=0.7, linestyle=":", label="Shield barrier")
    ax_shield.legend(fontsize=7)

    time_text = ax_xy.text(
        0.5, -0.12, "", transform=ax_xy.transAxes,
        ha="center", fontsize=9, family="monospace",
    )

    plt.tight_layout()

    all_artists = [
        line_v_xy, line_c_xy, line_s_xy, dot_v, dot_c, dot_s,
        line_v_dist, line_c_dist, line_s_dist,
        line_v_err, line_c_err, line_s_err,
        line_rate, line_c_bar, line_s_bar,
        time_text,
    ] + obs_markers

    def init():
        for a in all_artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
        time_text.set_text("")
        return all_artists

    def update(frame):
        if frame >= num_steps:
            return all_artists

        # 현재 시간의 장애물 위치
        t_current = frame * dt
        current_obstacles = get_obstacles_at(t_current)

        # 장애물 패치 업데이트
        for i, obs in enumerate(MOVING_OBSTACLES):
            px, py = obs.get_position(t_current)
            obs_patches[i].center = (px, py)
            obs_patches[i].radius = obs.radius
            obs_margins[i].center = (px, py)
            obs_margins[i].radius = obs.radius + 0.1
            obs_markers[i].set_data([px], [py])

        # 각 시뮬레이터 1스텝 실행 (장애물 업데이트 포함)
        for key in ctrls:
            ctrl = ctrls[key]["controller"]
            sim = ctrls[key]["sim"]

            # 동적 장애물 업데이트 (CBF/Shield만)
            if hasattr(ctrl, "update_obstacles"):
                ctrl.update_obstacles(current_obstacles)

            ref_traj = reference_fn(sim.t)
            step_info = sim.step(ref_traj)

            state = sim.state
            ref_pt = ref_traj[0, :2]
            t = sim.t

            data[key]["xy"].append(state[:2].copy())
            data[key]["times"].append(t)
            data[key]["errors"].append(np.linalg.norm(state[:2] - ref_pt))
            data[key]["min_dist"].append(
                compute_min_distance_single(state, current_obstacles)
            )

            info = step_info["info"]
            if key in ("cbf", "shield"):
                data[key]["barriers"].append(info.get("min_barrier", 0.0))
            if key == "shield":
                data[key]["shield_rates"].append(
                    info.get("shield_intervention_rate", 0.0)
                )

        # numpy 변환
        times = np.array(data["vanilla"]["times"])
        v_xy = np.array(data["vanilla"]["xy"])
        c_xy = np.array(data["cbf"]["xy"])
        s_xy = np.array(data["shield"]["xy"])

        # [0,0] XY 궤적
        line_v_xy.set_data(v_xy[:, 0], v_xy[:, 1])
        line_c_xy.set_data(c_xy[:, 0], c_xy[:, 1])
        line_s_xy.set_data(s_xy[:, 0], s_xy[:, 1])
        dot_v.set_data([v_xy[-1, 0]], [v_xy[-1, 1]])
        dot_c.set_data([c_xy[-1, 0]], [c_xy[-1, 1]])
        dot_s.set_data([s_xy[-1, 0]], [s_xy[-1, 1]])

        # Auto-scale XY
        all_x = np.concatenate([v_xy[:, 0], c_xy[:, 0], s_xy[:, 0]])
        all_y = np.concatenate([v_xy[:, 1], c_xy[:, 1], s_xy[:, 1]])
        for obs in MOVING_OBSTACLES:
            px, py = obs.get_position(t_current)
            all_x = np.append(all_x, px)
            all_y = np.append(all_y, py)
        margin = 2.0
        ax_xy.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        ax_xy.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)

        # [0,1] 최소 거리
        line_v_dist.set_data(times, data["vanilla"]["min_dist"])
        line_c_dist.set_data(times, data["cbf"]["min_dist"])
        line_s_dist.set_data(times, data["shield"]["min_dist"])
        ax_dist.relim(); ax_dist.autoscale_view()

        # [1,0] 추적 오차
        line_v_err.set_data(times, data["vanilla"]["errors"])
        line_c_err.set_data(times, data["cbf"]["errors"])
        line_s_err.set_data(times, data["shield"]["errors"])
        ax_err.relim(); ax_err.autoscale_view()

        # [1,1] Shield 개입률 + barrier
        if data["shield"]["shield_rates"]:
            line_rate.set_data(times, data["shield"]["shield_rates"])
        if data["cbf"]["barriers"]:
            line_c_bar.set_data(times, data["cbf"]["barriers"])
        if data["shield"]["barriers"]:
            line_s_bar.set_data(times, data["shield"]["barriers"])
        ax_shield.relim(); ax_shield.autoscale_view()

        # 시간 텍스트
        t_now = times[-1]
        v_rmse = np.sqrt(np.mean(np.array(data["vanilla"]["errors"]) ** 2))
        c_rmse = np.sqrt(np.mean(np.array(data["cbf"]["errors"]) ** 2))
        s_rmse = np.sqrt(np.mean(np.array(data["shield"]["errors"]) ** 2))
        time_text.set_text(
            f"t={t_now:.1f}s | "
            f"RMSE  V:{v_rmse:.3f}m  CBF:{c_rmse:.3f}m  Shield:{s_rmse:.3f}m"
        )

        return all_artists

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=num_steps, interval=1, blit=False, repeat=False,
    )
    plt.show()

    # 종료 후 통계
    print("\n" + "=" * 60)
    print("Final Statistics".center(60))
    print("=" * 60)
    for key in ctrls:
        errs = data[key]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        min_d = np.min(data[key]["min_dist"]) if data[key]["min_dist"] else 0
        collisions = sum(1 for d in data[key]["min_dist"] if d < 0)
        print(f"  {key:>8s}: RMSE={rmse:.4f}m, MinDist={min_d:.4f}m, Collisions={collisions}")

    shield_ctrl = ctrls["shield"]["controller"]
    ss = shield_ctrl.get_shield_statistics()
    cs = shield_ctrl.get_cbf_statistics()
    print(f"\n  Shield safety rate:       {cs['safety_rate']:.1%}")
    print(f"  Shield intervention rate: {ss['mean_intervention_rate']:.1%}")
    print("=" * 60 + "\n")


def run_batch(common_kwargs, cbf_kwargs, trajectory_fn, duration, seed):
    """배치 모드: 순차 실행 후 정적 6패널 플롯 저장"""
    matplotlib.use("Agg")

    ctrls = create_controllers(common_kwargs, cbf_kwargs)
    dt = common_kwargs["dt"]
    N = common_kwargs["N"]

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, N, dt)

    initial_state = trajectory_fn(0.0)
    num_steps = int(duration / dt)

    # 각 컨트롤러에 대해 수동 시뮬레이션 (동적 장애물 업데이트 필요)
    results = {}
    for i, (key, label) in enumerate([
        ("vanilla", "Vanilla MPPI"),
        ("cbf", "CBF-MPPI"),
        ("shield", "Shield-MPPI"),
    ]):
        print(f"\n[{i + 1}/3] {label}")
        sim = ctrls[key]["sim"]
        ctrl = ctrls[key]["controller"]
        sim.reset(initial_state)

        print(f"  Running {label} with dynamic obstacles...")
        for step in range(num_steps):
            t = sim.t
            current_obstacles = get_obstacles_at(t)

            # 동적 장애물 업데이트
            if hasattr(ctrl, "update_obstacles"):
                ctrl.update_obstacles(current_obstacles)

            ref_traj = reference_fn(t)
            sim.step(ref_traj)

        history = sim.get_history()
        metrics = compute_metrics(history)
        print(f"  {label} completed (RMSE: {metrics['position_rmse']:.4f}m)")

        # 장애물 거리 계산 (시간별 이동 장애물 기반)
        states = history["state"]
        times = history["time"]
        min_distances = []
        for j, t_val in enumerate(times):
            obs_at_t = get_obstacles_at(t_val)
            min_distances.append(compute_min_distance_single(states[j], obs_at_t))

        results[key] = {
            "history": history,
            "metrics": metrics,
            "min_distances": np.array(min_distances),
        }

    # 메트릭 출력
    print("\n")
    for key, label in [("vanilla", "Vanilla"), ("cbf", "CBF-MPPI"), ("shield", "Shield-MPPI")]:
        print_metrics(results[key]["metrics"], title=label)

    cbf_stats = ctrls["cbf"]["controller"].get_cbf_statistics()
    shield_cbf_stats = ctrls["shield"]["controller"].get_cbf_statistics()
    shield_stats = ctrls["shield"]["controller"].get_shield_statistics()

    print("=" * 60)
    print("Safety Statistics".center(60))
    print("=" * 60)
    for key, label in [("vanilla", "Vanilla"), ("cbf", "CBF-MPPI"), ("shield", "Shield-MPPI")]:
        md = results[key]["min_distances"]
        collisions = np.sum(md < 0)
        print(f"  {label:>12s}: MinDist={np.min(md):.4f}m, Collisions={collisions}")
    print(f"\n  CBF-MPPI safety rate:         {cbf_stats['safety_rate']:.2%}")
    print(f"  Shield-MPPI safety rate:      {shield_cbf_stats['safety_rate']:.2%}")
    print(f"  Shield intervention rate:     {shield_stats['mean_intervention_rate']:.2%}")
    print("=" * 60 + "\n")

    # ===== 6패널 정적 플롯 =====
    print("Generating comparison plots...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        "Dynamic Obstacle Avoidance: Vanilla vs CBF vs Shield",
        fontsize=16, fontweight="bold",
    )

    vh = results["vanilla"]["history"]
    ch = results["cbf"]["history"]
    sh = results["shield"]["history"]
    times = vh["time"]
    v_states, c_states, s_states = vh["state"], ch["state"], sh["state"]
    refs = vh["reference"]

    # [0,0] XY 궤적 + 장애물 궤적
    ax = axes[0, 0]
    ax.plot(refs[:, 0], refs[:, 1], "k--", label="Reference", linewidth=1.5, alpha=0.5)
    ax.plot(v_states[:, 0], v_states[:, 1], color=COLORS["vanilla"], label="Vanilla", linewidth=2)
    ax.plot(c_states[:, 0], c_states[:, 1], color=COLORS["cbf"], label="CBF-MPPI", linewidth=2)
    ax.plot(s_states[:, 0], s_states[:, 1], color=COLORS["shield"], label="Shield-MPPI", linewidth=2)

    # 장애물 궤적 (시간에 따른 위치)
    for obs in MOVING_OBSTACLES:
        obs_positions = np.array([obs.get_position(t) for t in times])
        ax.plot(obs_positions[:, 0], obs_positions[:, 1],
                "r-", alpha=0.3, linewidth=1)
        # 시작/끝 위치
        px0, py0 = obs.get_position(times[0])
        pxf, pyf = obs.get_position(times[-1])
        ax.add_patch(plt.Circle((px0, py0), obs.radius, color="red", alpha=0.15))
        ax.add_patch(plt.Circle((pxf, pyf), obs.radius, color="red", alpha=0.3))
        ax.plot(px0, py0, "r^", markersize=6)
        ax.plot(pxf, pyf, "rs", markersize=6)

    ax.plot(v_states[0, 0], v_states[0, 1], "ko", markersize=8, label="Start")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories + Moving Obstacles")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3); ax.axis("equal")

    # [0,1] 위치 추적 오차
    ax = axes[0, 1]
    v_err = np.linalg.norm(v_states[:, :2] - refs[:, :2], axis=1)
    c_err = np.linalg.norm(c_states[:, :2] - refs[:, :2], axis=1)
    s_err = np.linalg.norm(s_states[:, :2] - refs[:, :2], axis=1)
    ax.plot(times, v_err, color=COLORS["vanilla"], label="Vanilla", linewidth=2)
    ax.plot(times, c_err, color=COLORS["cbf"], label="CBF-MPPI", linewidth=2)
    ax.plot(times, s_err, color=COLORS["shield"], label="Shield-MPPI", linewidth=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [1,0] Minimum Barrier Value
    ax = axes[1, 0]
    c_bar = [info.get("min_barrier", 0.0) for info in ch["info"]]
    s_bar = [info.get("min_barrier", 0.0) for info in sh["info"]]
    ax.plot(times, c_bar, color=COLORS["cbf"], linewidth=2, label="CBF-MPPI")
    ax.plot(times, s_bar, color=COLORS["shield"], linewidth=2, label="Shield-MPPI")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="h=0")
    ax.fill_between(times, c_bar, 0, where=[b < 0 for b in c_bar], color=COLORS["cbf"], alpha=0.15)
    ax.fill_between(times, s_bar, 0, where=[b < 0 for b in s_bar], color=COLORS["shield"], alpha=0.15)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Barrier h(x)")
    ax.set_title("Minimum Barrier Value"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [1,1] Shield 개입률
    ax = axes[1, 1]
    s_rates = [info.get("shield_intervention_rate", 0.0) for info in sh["info"]]
    ax.plot(times, s_rates, color=COLORS["shield"], linewidth=2, label="Shield rate")
    ax.fill_between(times, s_rates, alpha=0.2, color=COLORS["shield"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Intervention Rate")
    ax.set_title("Shield Intervention Rate"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [2,0] 최소 장애물 거리 (동적)
    ax = axes[2, 0]
    ax.plot(times, results["vanilla"]["min_distances"], color=COLORS["vanilla"], label="Vanilla", linewidth=2)
    ax.plot(times, results["cbf"]["min_distances"], color=COLORS["cbf"], label="CBF-MPPI", linewidth=2)
    ax.plot(times, results["shield"]["min_distances"], color=COLORS["shield"], label="Shield-MPPI", linewidth=2)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Collision")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Min Distance (m)")
    ax.set_title("Min Distance to Dynamic Obstacle"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [2,1] 요약
    ax = axes[2, 1]; ax.axis("off")
    vm = results["vanilla"]["metrics"]
    cm = results["cbf"]["metrics"]
    sm = results["shield"]["metrics"]
    v_md = results["vanilla"]["min_distances"]
    c_md = results["cbf"]["min_distances"]
    s_md = results["shield"]["min_distances"]
    summary = f"""
    +-----------------------------------------------------+
    |        Dynamic Obstacle Avoidance Summary            |
    +-----------------------------------------------------+
    |  Vanilla:  RMSE={vm['position_rmse']:.4f}m  Time={vm['mean_solve_time']:.1f}ms |
    |            MinDist={np.min(v_md):.4f}m  Collisions={np.sum(v_md < 0)}          |
    |  CBF:      RMSE={cm['position_rmse']:.4f}m  Time={cm['mean_solve_time']:.1f}ms |
    |            MinDist={np.min(c_md):.4f}m  Safety={cbf_stats['safety_rate']:.0%}   |
    |  Shield:   RMSE={sm['position_rmse']:.4f}m  Time={sm['mean_solve_time']:.1f}ms |
    |            MinDist={np.min(s_md):.4f}m  Safety={shield_cbf_stats['safety_rate']:.0%}   |
    |            Intervention={shield_stats['mean_intervention_rate']:.0%}             |
    +-----------------------------------------------------+
    |  Moving Obstacles: {len(MOVING_OBSTACLES)}, Duration: {duration:.0f}s, Seed: {seed}     |
    +-----------------------------------------------------+
    """
    ax.text(0.05, 0.5, summary, fontsize=9, va="center", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    output_path = "plots/dynamic_obstacle_avoidance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    print("\nDynamic obstacle avoidance comparison complete!")


def main():
    parser = argparse.ArgumentParser(description="Dynamic Obstacle Avoidance Demo")
    parser.add_argument("--duration", type=float, default=12.0, help="Simulation duration (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("Dynamic Obstacle Avoidance Demo".center(80))
    print("Vanilla MPPI vs CBF-MPPI vs Shield-MPPI".center(80))
    print("=" * 80)
    print(f"Duration: {args.duration}s  |  Live: {args.live}  |  Seed: {args.seed}")
    print(f"Moving Obstacles: {len(MOVING_OBSTACLES)}")
    for i, obs in enumerate(MOVING_OBSTACLES):
        px, py = obs.get_position(0)
        print(f"  [{i}] start=({px:.1f}, {py:.1f}), r={obs.radius:.1f}, type={obs.motion_type}")
    print("=" * 80 + "\n")

    common_kwargs = dict(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )

    # 초기 장애물 위치로 CBF 파라미터 설정
    initial_obstacles = get_obstacles_at(0.0)
    cbf_kwargs = dict(
        cbf_obstacles=initial_obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        cbf_use_safety_filter=False,
    )
    trajectory_fn = create_trajectory_function("circle", radius=5.0)

    if args.live:
        run_live(common_kwargs, cbf_kwargs, trajectory_fn, args.duration)
    else:
        run_batch(common_kwargs, cbf_kwargs, trajectory_fn, args.duration, args.seed)


if __name__ == "__main__":
    main()
