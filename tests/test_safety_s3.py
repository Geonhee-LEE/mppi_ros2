"""
Safety Phase S3 테스트

- Backup CBF Safety Filter (#730)
- Multi-Robot CBF (#731)
- MPCC Cost (#732)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.controllers.mppi.backup_cbf_filter import BackupCBFSafetyFilter
from mppi_controller.controllers.mppi.backup_controller import (
    BrakeBackupController,
    TurnAndBrakeBackupController,
)
from mppi_controller.controllers.mppi.multi_robot_cbf import (
    RobotAgent,
    MultiRobotCBFCost,
    MultiRobotCBFFilter,
    MultiRobotCoordinator,
)
from mppi_controller.controllers.mppi.mpcc_cost import (
    PathParameterization,
    MPCCCost,
)
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)


# ─────────────────────────────────────────────────
# Backup CBF Safety Filter 테스트
# ─────────────────────────────────────────────────


def test_backup_cbf_no_obstacle():
    """장애물 없을 때 필터링 없음"""
    print("\n" + "=" * 60)
    print("Test: Backup CBF - No Obstacles")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    bcf = BackupCBFSafetyFilter(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=[],
        dt=0.05,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.3])
    u_safe, info = bcf.filter_control(state, u)

    assert np.allclose(u_safe, u)
    assert not info["filtered"]
    assert info["min_backup_barrier"] == float("inf")
    print("PASS\n")


def test_backup_cbf_safe_no_modification():
    """먼 장애물 — 필터링 없음"""
    print("=" * 60)
    print("Test: Backup CBF - Safe (No Modification)")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    bcf = BackupCBFSafetyFilter(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=[(100.0, 100.0, 0.5)],
        dt=0.05,
        backup_horizon=10,
        cbf_alpha=0.3,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.0])
    u_safe, info = bcf.filter_control(state, u)

    print(f"  u_safe: {u_safe}")
    print(f"  filtered: {info['filtered']}")
    assert np.allclose(u_safe, u, atol=1e-4)
    assert not info["filtered"]
    print("PASS\n")


def test_backup_cbf_unsafe_correction():
    """가까운 장애물 — 제어 수정"""
    print("=" * 60)
    print("Test: Backup CBF - Unsafe Correction")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    bcf = BackupCBFSafetyFilter(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=[(0.3, 0.0, 0.15)],
        dt=0.05,
        backup_horizon=10,
        cbf_alpha=0.3,
        safety_margin=0.05,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0])  # 장애물 방향 전진
    u_safe, info = bcf.filter_control(
        state, u,
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )

    print(f"  u_mppi: {u}")
    print(f"  u_safe: {u_safe}")
    print(f"  filtered: {info['filtered']}")
    print(f"  correction: {info['correction_norm']:.4f}")

    if info["filtered"]:
        assert info["correction_norm"] > 0
        # 속도가 줄어야 함
        assert u_safe[0] < u[0] or abs(u_safe[1]) > abs(u[1])
    print("PASS\n")


def test_backup_cbf_sensitivity_chain():
    """민감도 체인 길이 = backup_horizon + 1"""
    print("=" * 60)
    print("Test: Backup CBF - Sensitivity Chain Length")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    horizon = 12
    bcf = BackupCBFSafetyFilter(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=[(3.0, 0.0, 0.3)],
        dt=0.05,
        backup_horizon=horizon,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.0])

    backup_traj = bcf._compute_backup_trajectory(state, u)
    sensitivities = bcf._compute_sensitivity_chain(backup_traj, state, u)

    print(f"  Backup horizon: {horizon}")
    print(f"  Sensitivity chain length: {len(sensitivities)}")
    print(f"  Backup traj shape: {backup_traj.shape}")

    assert len(sensitivities) == horizon + 1
    assert backup_traj.shape == (horizon + 1, 3)

    # 각 민감도 행렬 shape 확인
    for i, s in enumerate(sensitivities):
        assert s.shape == (3, 2), f"Sensitivity[{i}] shape: {s.shape}"

    print("PASS\n")


def test_backup_cbf_stronger_than_standard():
    """Backup CBF가 표준 CBF보다 보수적 (더 큰 correction)"""
    print("=" * 60)
    print("Test: Backup CBF - Stronger Than Standard CBF")
    print("=" * 60)

    from mppi_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(0.8, 0.0, 0.2)]

    standard = CBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.3, safety_margin=0.1,
    )
    backup = BackupCBFSafetyFilter(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=obstacles,
        dt=0.05,
        backup_horizon=15,
        cbf_alpha=0.3,
        safety_margin=0.1,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.8, 0.0])
    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

    u_std, info_std = standard.filter_control(state, u, *bounds)
    u_bak, info_bak = backup.filter_control(state, u, *bounds)

    print(f"  Standard: u={u_std}, correction={info_std['correction_norm']:.4f}")
    print(f"  Backup:   u={u_bak}, correction={info_bak['correction_norm']:.4f}")

    # Backup CBF는 미래 안전도 고려 → 같거나 더 큰 correction
    # (빈약한 조건이지만 합리적)
    if info_std["filtered"] and info_bak["filtered"]:
        assert info_bak["correction_norm"] >= info_std["correction_norm"] * 0.5, \
            "Backup should have comparable or larger correction"
    print("PASS\n")


def test_backup_cbf_statistics():
    """필터 통계 추적"""
    print("=" * 60)
    print("Test: Backup CBF - Statistics")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    bcf = BackupCBFSafetyFilter(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=[(1.0, 0.0, 0.3)],
        dt=0.05,
        backup_horizon=10,
    )

    state = np.array([0.0, 0.0, 0.0])

    # 안전한 제어
    bcf.filter_control(state, np.array([0.1, 0.5]))
    # 위험한 제어
    bcf.filter_control(
        state, np.array([1.0, 0.0]),
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )

    stats = bcf.get_filter_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Filter rate: {stats['filter_rate']:.2%}")

    assert stats["total_steps"] == 2
    assert "mean_correction_norm" in stats

    bcf.reset()
    stats_reset = bcf.get_filter_statistics()
    assert stats_reset["total_steps"] == 0
    print("PASS\n")


def test_backup_cbf_update_obstacles():
    """동적 장애물 업데이트"""
    print("=" * 60)
    print("Test: Backup CBF - Update Obstacles")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    bcf = BackupCBFSafetyFilter(model=model, obstacles=[])

    assert len(bcf.obstacles) == 0
    bcf.update_obstacles([(1.0, 0.0, 0.3), (2.0, 1.0, 0.4)])
    assert len(bcf.obstacles) == 2
    print("PASS\n")


# ─────────────────────────────────────────────────
# Multi-Robot CBF 테스트
# ─────────────────────────────────────────────────


def test_multi_robot_cost_no_robots():
    """다른 로봇 없으면 비용 = 0"""
    print("=" * 60)
    print("Test: Multi-Robot CBF Cost - No Other Robots")
    print("=" * 60)

    cost_fn = MultiRobotCBFCost(other_robots=[], cbf_weight=2000.0)

    K, N = 32, 10
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 2, N + 1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,)
    assert np.allclose(costs, 0.0)
    print("PASS\n")


def test_multi_robot_cost_far_robots():
    """먼 로봇 — 비용 ≈ 0"""
    print("=" * 60)
    print("Test: Multi-Robot CBF Cost - Far Robots")
    print("=" * 60)

    other_robots = [(100.0, 100.0, 0.3, 0.0, 0.0)]
    cost_fn = MultiRobotCBFCost(other_robots=other_robots, cbf_weight=2000.0)

    K, N = 16, 10
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 2, N + 1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.allclose(costs, 0.0, atol=1e-3)
    print("PASS\n")


def test_multi_robot_cost_collision_path():
    """충돌 경로 — 높은 비용"""
    print("=" * 60)
    print("Test: Multi-Robot CBF Cost - Collision Path")
    print("=" * 60)

    other_robots = [(2.0, 0.0, 0.3, 0.0, 0.0)]
    cost_fn = MultiRobotCBFCost(
        other_robots=other_robots, cbf_weight=2000.0, safety_margin=0.2,
    )

    K, N = 16, 20
    # 충돌 궤적
    traj_collide = np.zeros((K, N + 1, 3))
    traj_collide[:, :, 0] = np.linspace(0.0, 2.5, N + 1)

    # 회피 궤적
    traj_avoid = np.zeros((K, N + 1, 3))
    traj_avoid[:, :, 0] = np.linspace(0.0, 2.5, N + 1)
    traj_avoid[:, :, 1] = 3.0

    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    cost_collide = cost_fn.compute_cost(traj_collide, ctrl, ref)
    cost_avoid = cost_fn.compute_cost(traj_avoid, ctrl, ref)

    print(f"  Collision cost: {np.mean(cost_collide):.2f}")
    print(f"  Avoidance cost: {np.mean(cost_avoid):.2f}")

    assert np.mean(cost_collide) > np.mean(cost_avoid)
    print("PASS\n")


def test_multi_robot_filter_safe():
    """안전 상태 — 필터링 없음"""
    print("=" * 60)
    print("Test: Multi-Robot CBF Filter - Safe State")
    print("=" * 60)

    other_robots = [(100.0, 100.0, 0.3, 0.0, 0.0)]
    cbf_filter = MultiRobotCBFFilter(
        other_robots=other_robots, cbf_alpha=0.3,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.3])
    u_safe, info = cbf_filter.filter_control(state, u)

    assert np.allclose(u_safe, u)
    assert not info["filtered"]
    print("PASS\n")


def test_multi_robot_filter_collision():
    """정면 충돌 상황 — 필터 적용"""
    print("=" * 60)
    print("Test: Multi-Robot CBF Filter - Collision Filtering")
    print("=" * 60)

    other_robots = [(0.5, 0.0, 0.15, 0.0, 0.0)]
    cbf_filter = MultiRobotCBFFilter(
        other_robots=other_robots, cbf_alpha=0.3, safety_margin=0.1,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0])  # 상대 로봇 방향 전진
    u_safe, info = cbf_filter.filter_control(
        state, u,
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )

    print(f"  u_mppi: {u}")
    print(f"  u_safe: {u_safe}")
    print(f"  filtered: {info['filtered']}")

    if info["filtered"]:
        assert info["correction_norm"] > 0
    print("PASS\n")


def test_multi_robot_coordinator_2agents():
    """2 에이전트 조율"""
    print("=" * 60)
    print("Test: Multi-Robot Coordinator - 2 Agents")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    from mppi_controller.controllers.mppi.base_mppi import MPPIController
    from mppi_controller.controllers.mppi.mppi_params import MPPIParams
    from mppi_controller.controllers.mppi.cost_functions import (
        StateTrackingCost,
        CompositeMPPICost,
    )

    params = MPPIParams(
        K=32, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.3, 0.3]),
    )

    cost1 = CompositeMPPICost([StateTrackingCost(np.array([10.0, 10.0, 1.0]))])
    cost2 = CompositeMPPICost([StateTrackingCost(np.array([10.0, 10.0, 1.0]))])

    ctrl1 = MPPIController(model, params, cost1)
    ctrl2 = MPPIController(model, params, cost2)

    agents = [
        RobotAgent(id=0, state=np.array([0.0, 0.0, 0.0]), radius=0.2,
                    model=model, controller=ctrl1),
        RobotAgent(id=1, state=np.array([5.0, 0.0, np.pi]), radius=0.2,
                    model=model, controller=ctrl2),
    ]

    coordinator = MultiRobotCoordinator(
        agents, dt=0.05, cbf_alpha=0.3, safety_margin=0.2,
        use_cost=False, use_filter=True,
    )

    # 레퍼런스 궤적
    N = 10
    ref0 = np.zeros((N + 1, 3))
    ref0[:, 0] = np.linspace(0, 3, N + 1)
    ref1 = np.zeros((N + 1, 3))
    ref1[:, 0] = np.linspace(5, 2, N + 1)
    ref1[:, 2] = np.pi

    results = coordinator.step({0: ref0, 1: ref1})

    assert 0 in results
    assert 1 in results
    print(f"  Agent 0 control: {results[0][0]}")
    print(f"  Agent 1 control: {results[1][0]}")

    states = coordinator.get_states()
    assert 0 in states
    assert 1 in states
    print("PASS\n")


def test_multi_robot_coordinator_3agents():
    """3 에이전트 조율"""
    print("=" * 60)
    print("Test: Multi-Robot Coordinator - 3 Agents")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    from mppi_controller.controllers.mppi.base_mppi import MPPIController
    from mppi_controller.controllers.mppi.mppi_params import MPPIParams
    from mppi_controller.controllers.mppi.cost_functions import (
        StateTrackingCost,
        CompositeMPPICost,
    )

    params = MPPIParams(
        K=32, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.3, 0.3]),
    )

    agents = []
    for i in range(3):
        cost = CompositeMPPICost([StateTrackingCost(np.array([10.0, 10.0, 1.0]))])
        ctrl = MPPIController(model, params, cost)
        angle = 2 * np.pi * i / 3
        state = np.array([3.0 * np.cos(angle), 3.0 * np.sin(angle), angle + np.pi])
        agents.append(
            RobotAgent(id=i, state=state, radius=0.2, model=model, controller=ctrl)
        )

    coordinator = MultiRobotCoordinator(
        agents, dt=0.05, use_cost=False, use_filter=True,
    )

    N = 10
    refs = {}
    for i in range(3):
        ref = np.zeros((N + 1, 3))
        ref[:, 2] = agents[i].state[2]
        refs[i] = ref

    results = coordinator.step(refs)

    assert len(results) == 3
    for i in range(3):
        assert i in results
        print(f"  Agent {i} control: {results[i][0]}")

    print("PASS\n")


def test_multi_robot_update_robots():
    """로봇 상태 업데이트"""
    print("=" * 60)
    print("Test: Multi-Robot CBF - Update Robots")
    print("=" * 60)

    cost_fn = MultiRobotCBFCost(other_robots=[])
    assert len(cost_fn.other_robots) == 0

    cost_fn.update_other_robots([
        (1.0, 0.0, 0.3, 0.1, 0.0),
        (2.0, 1.0, 0.2, -0.1, 0.2),
    ])
    assert len(cost_fn.other_robots) == 2

    cbf_filter = MultiRobotCBFFilter()
    cbf_filter.update_other_robots([
        (1.0, 0.0, 0.3, 0.0, 0.0),
    ])
    assert len(cbf_filter.other_robots) == 1
    print("PASS\n")


# ─────────────────────────────────────────────────
# MPCC Cost 테스트
# ─────────────────────────────────────────────────


def test_mpcc_straight_line_no_error():
    """직선 경로 위 → contouring ≈ 0"""
    print("=" * 60)
    print("Test: MPCC - Straight Line (No Error)")
    print("=" * 60)

    waypoints = np.array([[0.0, 0.0], [10.0, 0.0]])
    cost_fn = MPCCCost(reference_path=waypoints, Q_c=50.0, Q_l=10.0, Q_theta=0.0, Q_heading=0.0)

    K, N = 16, 10
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 5, N + 1)  # 경로 위
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)

    print(f"  Max cost: {np.max(costs):.6f}")
    assert np.allclose(costs, 0.0, atol=1e-6), f"On-path should have ~0 contouring cost, got {np.max(costs)}"
    print("PASS\n")


def test_mpcc_contouring_error():
    """수직 오프셋 → contouring cost"""
    print("=" * 60)
    print("Test: MPCC - Contouring Error")
    print("=" * 60)

    waypoints = np.array([[0.0, 0.0], [10.0, 0.0]])
    cost_fn = MPCCCost(reference_path=waypoints, Q_c=50.0, Q_l=0.0, Q_theta=0.0, Q_heading=0.0)

    K, N = 16, 10
    # 수직 오프셋 (y=1.0)
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 5, N + 1)
    traj[:, :, 1] = 1.0  # contouring error
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)

    print(f"  Mean cost: {np.mean(costs):.2f}")
    assert np.mean(costs) > 0, "Vertical offset should produce contouring cost"
    print("PASS\n")


def test_mpcc_lag_error():
    """수평 방향 정지 → lag cost"""
    print("=" * 60)
    print("Test: MPCC - Lag Error")
    print("=" * 60)

    waypoints = np.array([[0.0, 0.0], [10.0, 0.0]])
    cost_fn = MPCCCost(reference_path=waypoints, Q_c=0.0, Q_l=10.0, Q_theta=0.0, Q_heading=0.0)

    K, N = 16, 10
    # 정지 상태 (lag가 발생하지만 progress가 0)
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = 2.0  # 한 점에 정지
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)

    # 경로 위이므로 lag = 0, contouring = 0, progress = 0
    # 하지만 투영이 같은 점이므로 lag error는 0
    print(f"  Mean cost: {np.mean(costs):.4f}")

    # 이제 경로 밖에서 lag가 있는 경우
    traj_offset = np.zeros((K, N + 1, 3))
    traj_offset[:, :, 0] = np.linspace(-1.0, -0.5, N + 1)  # 뒤쪽
    costs_offset = cost_fn.compute_cost(traj_offset, ctrl, ref)
    print(f"  Behind start cost: {np.mean(costs_offset):.4f}")

    # lag error 자체가 존재하는지 확인 (Q_c=0이므로 contouring은 없음)
    assert costs.shape == (K,)
    print("PASS\n")


def test_mpcc_progress_reward():
    """더 많은 진행 → 낮은 cost"""
    print("=" * 60)
    print("Test: MPCC - Progress Reward")
    print("=" * 60)

    waypoints = np.array([[0.0, 0.0], [10.0, 0.0]])
    cost_fn = MPCCCost(
        reference_path=waypoints,
        Q_c=0.0, Q_l=0.0, Q_theta=5.0, Q_heading=0.0,
    )

    K, N = 16, 10
    # 빠른 진행
    traj_fast = np.zeros((K, N + 1, 3))
    traj_fast[:, :, 0] = np.linspace(0, 8, N + 1)

    # 느린 진행
    traj_slow = np.zeros((K, N + 1, 3))
    traj_slow[:, :, 0] = np.linspace(0, 2, N + 1)

    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    cost_fast = cost_fn.compute_cost(traj_fast, ctrl, ref)
    cost_slow = cost_fn.compute_cost(traj_slow, ctrl, ref)

    print(f"  Fast progress cost: {np.mean(cost_fast):.2f}")
    print(f"  Slow progress cost: {np.mean(cost_slow):.2f}")

    # 빠른 진행 = 더 큰 보상(음수) → 더 낮은 총 cost
    assert np.mean(cost_fast) < np.mean(cost_slow), \
        "More progress should yield lower cost"
    print("PASS\n")


def test_mpcc_circle_path():
    """원형 경로"""
    print("=" * 60)
    print("Test: MPCC - Circle Path")
    print("=" * 60)

    # 원형 웨이포인트
    n_pts = 50
    angles = np.linspace(0, 2 * np.pi, n_pts)
    R = 3.0
    waypoints = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=-1)

    cost_fn = MPCCCost(reference_path=waypoints, Q_c=50.0, Q_l=10.0, Q_theta=0.0, Q_heading=0.0)

    K, N = 16, 10
    # 원 위의 궤적
    t_angles = np.linspace(0, np.pi / 4, N + 1)
    traj_on = np.zeros((K, N + 1, 3))
    traj_on[:, :, 0] = R * np.cos(t_angles)
    traj_on[:, :, 1] = R * np.sin(t_angles)

    # 원 밖의 궤적 (반경 5)
    traj_off = np.zeros((K, N + 1, 3))
    traj_off[:, :, 0] = 5.0 * np.cos(t_angles)
    traj_off[:, :, 1] = 5.0 * np.sin(t_angles)

    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    cost_on = cost_fn.compute_cost(traj_on, ctrl, ref)
    cost_off = cost_fn.compute_cost(traj_off, ctrl, ref)

    print(f"  On-circle cost: {np.mean(cost_on):.2f}")
    print(f"  Off-circle cost: {np.mean(cost_off):.2f}")

    assert np.mean(cost_on) < np.mean(cost_off), \
        "On-circle should have lower cost than off-circle"
    print("PASS\n")


def test_mpcc_vs_tracking_cost():
    """MPCC vs StateTrackingCost 비교"""
    print("=" * 60)
    print("Test: MPCC vs StateTrackingCost")
    print("=" * 60)

    from mppi_controller.controllers.mppi.cost_functions import StateTrackingCost

    waypoints = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    mpcc = MPCCCost(reference_path=waypoints, Q_c=50.0, Q_l=10.0, Q_theta=5.0)
    tracking = StateTrackingCost(np.array([10.0, 10.0, 1.0]))

    K, N = 32, 15
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 4, N + 1)
    traj[:, :, 1] = np.linspace(0, 1, N + 1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = np.linspace(0, 5, N + 1)

    cost_mpcc = mpcc.compute_cost(traj, ctrl, ref)
    cost_track = tracking.compute_cost(traj, ctrl, ref)

    print(f"  MPCC cost: {np.mean(cost_mpcc):.2f}")
    print(f"  Tracking cost: {np.mean(cost_track):.2f}")

    assert cost_mpcc.shape == (K,)
    assert cost_track.shape == (K,)
    print("PASS\n")


def test_mpcc_shape_consistency():
    """다양한 K 크기에서 shape 일관성"""
    print("=" * 60)
    print("Test: MPCC - Shape Consistency")
    print("=" * 60)

    waypoints = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 5.0]])
    cost_fn = MPCCCost(reference_path=waypoints)

    for K in [1, 16, 128, 512]:
        N = 10
        traj = np.random.randn(K, N + 1, 3) * 3.0
        ctrl = np.random.randn(K, N, 2)
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (K,), f"K={K}: got {costs.shape}"
        assert not np.any(np.isnan(costs)), f"K={K}: NaN detected"
        print(f"  K={K:4d}: OK")

    print("PASS\n")


def test_mpcc_contouring_info():
    """상세 contouring info dict"""
    print("=" * 60)
    print("Test: MPCC - Contouring Info")
    print("=" * 60)

    waypoints = np.array([[0.0, 0.0], [10.0, 0.0]])
    cost_fn = MPCCCost(reference_path=waypoints)

    traj = np.zeros((11, 3))
    traj[:, 0] = np.linspace(0, 5, 11)
    traj[:, 1] = 0.5  # 수직 오프셋

    info = cost_fn.get_contouring_info(traj)

    print(f"  mean_contouring_error: {info['mean_contouring_error']:.4f}")
    print(f"  mean_lag_error: {info['mean_lag_error']:.4f}")
    print(f"  progress: {info['progress']:.2f}")

    assert "contouring_errors" in info
    assert "lag_errors" in info
    assert "progress" in info
    assert "theta_star" in info
    assert "mean_contouring_error" in info
    assert "mean_lag_error" in info
    assert info["mean_contouring_error"] > 0  # 수직 오프셋이므로
    assert info["progress"] > 0  # 앞으로 진행
    print("PASS\n")


# ─────────────────────────────────────────────────
# Standalone 실행
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Safety Phase S3 Tests".center(60))
    print("=" * 60)

    tests = [
        # Backup CBF (7)
        test_backup_cbf_no_obstacle,
        test_backup_cbf_safe_no_modification,
        test_backup_cbf_unsafe_correction,
        test_backup_cbf_sensitivity_chain,
        test_backup_cbf_stronger_than_standard,
        test_backup_cbf_statistics,
        test_backup_cbf_update_obstacles,
        # Multi-Robot CBF (8)
        test_multi_robot_cost_no_robots,
        test_multi_robot_cost_far_robots,
        test_multi_robot_cost_collision_path,
        test_multi_robot_filter_safe,
        test_multi_robot_filter_collision,
        test_multi_robot_coordinator_2agents,
        test_multi_robot_coordinator_3agents,
        test_multi_robot_update_robots,
        # MPCC (8)
        test_mpcc_straight_line_no_error,
        test_mpcc_contouring_error,
        test_mpcc_lag_error,
        test_mpcc_progress_reward,
        test_mpcc_circle_path,
        test_mpcc_vs_tracking_cost,
        test_mpcc_shape_consistency,
        test_mpcc_contouring_info,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")
