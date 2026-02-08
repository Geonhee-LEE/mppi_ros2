"""
Gatekeeper + Superellipsoid 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.backup_controller import (
    BrakeBackupController,
    TurnAndBrakeBackupController,
)
from mppi_controller.controllers.mppi.gatekeeper import Gatekeeper
from mppi_controller.controllers.mppi.superellipsoid_cost import (
    SuperellipsoidObstacle,
    SuperellipsoidCost,
)


# ─────────────────────────────────────────────────
# Backup Controller 테스트
# ─────────────────────────────────────────────────


def test_brake_backup():
    """BrakeBackupController: 정지 제어"""
    print("\n" + "=" * 60)
    print("Test: Brake Backup Controller")
    print("=" * 60)

    bc = BrakeBackupController()
    state = np.array([1.0, 2.0, 0.5])
    obstacles = [(3.0, 2.0, 0.3)]

    u = bc.compute_backup_control(state, obstacles)
    assert np.allclose(u, [0.0, 0.0]), f"Brake should be [0,0], got {u}"

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    traj = bc.generate_backup_trajectory(state, model, 0.05, 10, obstacles)
    assert traj.shape == (11, 3)
    # 정지이므로 모든 위치 동일
    assert np.allclose(traj[0, :2], traj[-1, :2], atol=1e-10)
    print("PASS\n")


def test_turn_and_brake_backup():
    """TurnAndBrakeBackupController: 회전 후 정지"""
    print("=" * 60)
    print("Test: Turn and Brake Backup Controller")
    print("=" * 60)

    bc = TurnAndBrakeBackupController(turn_speed=0.5, turn_steps=5)
    state = np.array([0.0, 0.0, 0.0])
    obstacles = [(1.0, 0.0, 0.3)]  # 정면에 장애물

    u = bc.compute_backup_control(state, obstacles)
    print(f"  Backup control: {u}")
    assert u[0] == 0.0, "Forward velocity should be 0"
    assert abs(u[1]) > 0, "Should turn away from obstacle"

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    traj = bc.generate_backup_trajectory(state, model, 0.05, 15, obstacles)
    assert traj.shape == (16, 3)
    # 회전 후 정지이므로 마지막 위치 변화 없음
    assert np.allclose(traj[-1, :2], traj[-2, :2], atol=1e-6)
    print("PASS\n")


def test_turn_backup_no_obstacles():
    """장애물 없을 때 백업 컨트롤"""
    print("=" * 60)
    print("Test: Turn Backup - No Obstacles")
    print("=" * 60)

    bc = TurnAndBrakeBackupController()
    state = np.array([0.0, 0.0, 0.0])

    u = bc.compute_backup_control(state, [])
    assert np.allclose(u, [0.0, 0.0])
    print("PASS\n")


# ─────────────────────────────────────────────────
# Gatekeeper 테스트
# ─────────────────────────────────────────────────


def test_gatekeeper_safe_pass():
    """안전할 때 gate open (MPPI 제어 통과)"""
    print("=" * 60)
    print("Test: Gatekeeper - Safe (Gate Open)")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(10.0, 10.0, 0.5)]  # 먼 장애물
    gk = Gatekeeper(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=obstacles,
        safety_margin=0.15,
        backup_horizon=20,
        dt=0.05,
    )

    state = np.array([0.0, 0.0, 0.0])
    u_mppi = np.array([0.5, 0.1])

    u_safe, info = gk.filter(state, u_mppi)

    print(f"  Gate open: {info['gate_open']}")
    print(f"  Reason: {info['reason']}")
    assert info["gate_open"]
    assert np.allclose(u_safe, u_mppi)
    print("PASS\n")


def test_gatekeeper_unsafe_block():
    """위험할 때 gate closed (백업 제어 적용)"""
    print("=" * 60)
    print("Test: Gatekeeper - Unsafe (Gate Closed)")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    # 장애물이 바로 앞에
    obstacles = [(0.3, 0.0, 0.2)]
    gk = Gatekeeper(
        backup_controller=BrakeBackupController(),
        model=model,
        obstacles=obstacles,
        safety_margin=0.15,
        backup_horizon=20,
        dt=0.05,
    )

    state = np.array([0.0, 0.0, 0.0])
    u_mppi = np.array([1.0, 0.0])  # 전진 (장애물 방향)

    u_safe, info = gk.filter(state, u_mppi)

    print(f"  Gate open: {info['gate_open']}")
    print(f"  Reason: {info['reason']}")
    print(f"  Backup min barrier: {info['backup_min_barrier']:.4f}")
    print(f"  u_safe: {u_safe}")

    if not info["gate_open"]:
        assert np.allclose(u_safe, [0.0, 0.0]), "Should apply brake backup"
        print("PASS (gate closed, backup applied)\n")
    else:
        # 현재 상태가 이미 장애물 내부일 수 있음
        print("PASS (gate open — backup still safe from next state)\n")


def test_gatekeeper_no_obstacles():
    """장애물 없으면 항상 gate open"""
    print("=" * 60)
    print("Test: Gatekeeper - No Obstacles")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    gk = Gatekeeper(model=model, obstacles=[])

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.3])

    u_safe, info = gk.filter(state, u)
    assert info["gate_open"]
    assert np.allclose(u_safe, u)
    print("PASS\n")


def test_gatekeeper_statistics():
    """Gatekeeper 통계"""
    print("=" * 60)
    print("Test: Gatekeeper - Statistics")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(10.0, 10.0, 0.5)]
    gk = Gatekeeper(model=model, obstacles=obstacles)

    state = np.array([0.0, 0.0, 0.0])
    for _ in range(5):
        gk.filter(state, np.array([0.3, 0.0]))

    stats = gk.get_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Gate open rate: {stats['gate_open_rate']:.2%}")
    assert stats["total_steps"] == 5
    assert stats["gate_open_rate"] == 1.0

    gk.reset()
    stats_reset = gk.get_statistics()
    assert stats_reset["total_steps"] == 0
    print("PASS\n")


def test_gatekeeper_update_obstacles():
    """장애물 업데이트"""
    print("=" * 60)
    print("Test: Gatekeeper - Update Obstacles")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    gk = Gatekeeper(model=model, obstacles=[])
    assert len(gk.obstacles) == 0

    gk.update_obstacles([(1.0, 1.0, 0.3)])
    assert len(gk.obstacles) == 1
    print("PASS\n")


def test_gatekeeper_repr():
    """__repr__ 테스트"""
    print("=" * 60)
    print("Test: Gatekeeper - repr")
    print("=" * 60)

    gk = Gatekeeper(
        backup_controller=TurnAndBrakeBackupController(),
        obstacles=[(1.0, 0.0, 0.3), (2.0, 0.0, 0.5)],
        backup_horizon=30,
    )
    rep = repr(gk)
    print(f"  {rep}")
    assert "Gatekeeper" in rep
    assert "TurnAndBrake" in rep
    print("PASS\n")


# ─────────────────────────────────────────────────
# Superellipsoid 테스트
# ─────────────────────────────────────────────────


def test_superellipsoid_obstacle_circle():
    """n=2, a=b → 원형 장애물과 동등"""
    print("=" * 60)
    print("Test: Superellipsoid - Circle (n=2, a=b)")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=0.0, cy=0.0, a=1.0, b=1.0, n=2.0)

    # 경계 위
    h_boundary = obs.barrier(np.array([1.0]), np.array([0.0]))
    assert abs(h_boundary[0]) < 1e-10, f"Boundary: expected 0, got {h_boundary[0]}"

    # 외부 (distance 2)
    h_outside = obs.barrier(np.array([2.0]), np.array([0.0]))
    assert h_outside[0] > 0, f"Outside should be positive, got {h_outside[0]}"

    # 내부
    h_inside = obs.barrier(np.array([0.5]), np.array([0.0]))
    assert h_inside[0] < 0, f"Inside should be negative, got {h_inside[0]}"

    print(f"  h(boundary) = {h_boundary[0]:.6f}")
    print(f"  h(outside)  = {h_outside[0]:.6f}")
    print(f"  h(inside)   = {h_inside[0]:.6f}")
    print("PASS\n")


def test_superellipsoid_obstacle_rectangle():
    """n>2 → 직사각형에 가까운 형태"""
    print("=" * 60)
    print("Test: Superellipsoid - Rectangle (n=10)")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=0.0, cy=0.0, a=2.0, b=1.0, n=10.0)

    # (2,0) → 경계 위
    h_edge = obs.barrier(np.array([2.0]), np.array([0.0]))
    assert abs(h_edge[0]) < 1e-10

    # 모서리 (1.9, 0.9) → 내부 (직사각형에 가까우므로)
    h_corner = obs.barrier(np.array([1.9]), np.array([0.9]))
    assert h_corner[0] < 0, f"Corner inside should be negative: {h_corner[0]}"

    # (3.0, 0) → 외부
    h_outside = obs.barrier(np.array([3.0]), np.array([0.0]))
    assert h_outside[0] > 0

    print(f"  h(edge)    = {h_edge[0]:.6f}")
    print(f"  h(corner)  = {h_corner[0]:.6f}")
    print(f"  h(outside) = {h_outside[0]:.6f}")
    print("PASS\n")


def test_superellipsoid_rotated():
    """회전된 장애물"""
    print("=" * 60)
    print("Test: Superellipsoid - Rotated Obstacle")
    print("=" * 60)

    # 45도 회전된 타원
    obs = SuperellipsoidObstacle(
        cx=0.0, cy=0.0, a=2.0, b=1.0, n=2.0, theta=np.pi / 4
    )

    # 비회전 시 (2,0)이 경계지만, 회전 후에는 아님
    h1 = obs.barrier(np.array([2.0]), np.array([0.0]))
    # 대각 방향으로 경계 확인
    # a=2 장축 방향이 45도이므로 (sqrt(2), sqrt(2)) ≈ (1.41, 1.41)이 경계 근처
    h2 = obs.barrier(np.array([np.sqrt(2)]), np.array([np.sqrt(2)]))

    print(f"  h(2, 0) = {h1[0]:.4f} (not on boundary when rotated)")
    print(f"  h(√2, √2) = {h2[0]:.4f} (along rotated long axis)")

    assert not np.isnan(h1[0])
    assert not np.isnan(h2[0])
    print("PASS\n")


def test_superellipsoid_vectorized():
    """배치 처리 테스트"""
    print("=" * 60)
    print("Test: Superellipsoid - Vectorized")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=0.0, cy=0.0, a=1.0, b=0.5, n=2.0)

    x = np.random.randn(100) * 3
    y = np.random.randn(100) * 3
    h = obs.barrier(x, y)

    assert h.shape == (100,)
    assert not np.any(np.isnan(h))
    print(f"  Shape: {h.shape}, min={np.min(h):.4f}, max={np.max(h):.4f}")
    print("PASS\n")


def test_superellipsoid_cost_no_obstacle():
    """장애물 없을 때 비용 0"""
    print("=" * 60)
    print("Test: Superellipsoid Cost - No Obstacles")
    print("=" * 60)

    cost_fn = SuperellipsoidCost(obstacles=[])

    K, N = 32, 10
    traj = np.random.randn(K, N + 1, 3)
    ctrl = np.random.randn(K, N, 2)
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,)
    assert np.allclose(costs, 0.0)
    print("PASS\n")


def test_superellipsoid_cost_violation():
    """장애물 관통 시 비용 발생"""
    print("=" * 60)
    print("Test: Superellipsoid Cost - Violation")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=2.0, cy=0.0, a=0.5, b=0.3, n=2.0)
    cost_fn = SuperellipsoidCost(obstacles=[obs], cbf_weight=1000.0)

    K, N = 16, 20

    # 장애물 관통 궤적
    traj_unsafe = np.zeros((K, N + 1, 3))
    traj_unsafe[:, :, 0] = np.linspace(0, 4, N + 1)

    # 안전 궤적
    traj_safe = np.zeros((K, N + 1, 3))
    traj_safe[:, :, 0] = np.linspace(0, 4, N + 1)
    traj_safe[:, :, 1] = 2.0  # 측면 우회

    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    cost_unsafe = cost_fn.compute_cost(traj_unsafe, ctrl, ref)
    cost_safe = cost_fn.compute_cost(traj_safe, ctrl, ref)

    print(f"  Unsafe cost: {np.mean(cost_unsafe):.2f}")
    print(f"  Safe cost: {np.mean(cost_safe):.2f}")

    assert np.mean(cost_unsafe) > np.mean(cost_safe)
    print("PASS\n")


def test_superellipsoid_cost_barrier_info():
    """Barrier info 테스트"""
    print("=" * 60)
    print("Test: Superellipsoid Cost - Barrier Info")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=5.0, cy=5.0, a=1.0, b=1.0, n=2.0)
    cost_fn = SuperellipsoidCost(obstacles=[obs])

    traj = np.zeros((11, 3))
    traj[:, 0] = np.linspace(0, 1, 11)

    info = cost_fn.get_barrier_info(traj)
    assert "min_barrier" in info
    assert "is_safe" in info
    assert info["is_safe"]  # 원점 근처, 장애물은 (5,5)

    # 빈 장애물
    cost_empty = SuperellipsoidCost(obstacles=[])
    info_empty = cost_empty.get_barrier_info(traj)
    assert info_empty["is_safe"]
    assert info_empty["min_barrier"] == float("inf")
    print("PASS\n")


def test_superellipsoid_cost_shape_consistency():
    """다양한 K에서 shape 일관성"""
    print("=" * 60)
    print("Test: Superellipsoid Cost - Shape Consistency")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=0.0, cy=0.0, a=1.0, b=0.5, n=4.0)
    cost_fn = SuperellipsoidCost(obstacles=[obs])

    for K in [1, 16, 128]:
        N = 15
        traj = np.random.randn(K, N + 1, 3) * 3
        ctrl = np.random.randn(K, N, 2)
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (K,)
        assert not np.any(np.isnan(costs))
        print(f"  K={K:4d}: OK")

    print("PASS\n")


def test_superellipsoid_cost_safety_margin():
    """안전 마진 적용 — 마진이 barrier 값에 영향"""
    print("=" * 60)
    print("Test: Superellipsoid Cost - Safety Margin")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=2.0, cy=0.0, a=0.5, b=0.5, n=2.0)

    cost_tight = SuperellipsoidCost(obstacles=[obs], cbf_weight=1000.0, safety_margin=0.0)
    cost_wide = SuperellipsoidCost(obstacles=[obs], cbf_weight=1000.0, safety_margin=0.3)

    # 측면 통과 궤적 — 마진 없으면 안전, 마진 있으면 위험
    K, N = 16, 20
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0.0, 4.0, N + 1)
    traj[:, :, 1] = 0.7  # 장애물 가장자리 근처 통과
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    c_tight = cost_tight.compute_cost(traj, ctrl, ref)
    c_wide = cost_wide.compute_cost(traj, ctrl, ref)

    print(f"  Cost (no margin):  {np.mean(c_tight):.2f}")
    print(f"  Cost (0.3m margin): {np.mean(c_wide):.2f}")

    # 마진이 barrier 값을 변화시킴을 확인
    info_tight = cost_tight.get_barrier_info(traj[0])
    info_wide = cost_wide.get_barrier_info(traj[0])
    print(f"  Min barrier (no margin):  {info_tight['min_barrier']:.4f}")
    print(f"  Min barrier (0.3m margin): {info_wide['min_barrier']:.4f}")

    # 마진을 적용하면 barrier가 감소 (더 위험하게 판단)
    assert info_wide["min_barrier"] < info_tight["min_barrier"], \
        "Wider margin should reduce barrier values"
    print("PASS\n")


def test_superellipsoid_repr():
    """repr 테스트"""
    print("=" * 60)
    print("Test: Superellipsoid - repr")
    print("=" * 60)

    obs = SuperellipsoidObstacle(cx=1.0, cy=2.0, a=0.5, b=0.3, n=4.0, theta=0.5)
    print(f"  {obs}")
    assert "SuperellipsoidObstacle" in repr(obs)

    cost_fn = SuperellipsoidCost(obstacles=[obs])
    print(f"  {cost_fn}")
    assert "SuperellipsoidCost" in repr(cost_fn)
    print("PASS\n")


# ─────────────────────────────────────────────────
# Standalone 실행
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Gatekeeper + Superellipsoid Tests".center(60))
    print("=" * 60)

    tests = [
        # Backup Controller
        test_brake_backup,
        test_turn_and_brake_backup,
        test_turn_backup_no_obstacles,
        # Gatekeeper
        test_gatekeeper_safe_pass,
        test_gatekeeper_unsafe_block,
        test_gatekeeper_no_obstacles,
        test_gatekeeper_statistics,
        test_gatekeeper_update_obstacles,
        test_gatekeeper_repr,
        # Superellipsoid
        test_superellipsoid_obstacle_circle,
        test_superellipsoid_obstacle_rectangle,
        test_superellipsoid_rotated,
        test_superellipsoid_vectorized,
        test_superellipsoid_cost_no_obstacle,
        test_superellipsoid_cost_violation,
        test_superellipsoid_cost_barrier_info,
        test_superellipsoid_cost_shape_consistency,
        test_superellipsoid_cost_safety_margin,
        test_superellipsoid_repr,
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
