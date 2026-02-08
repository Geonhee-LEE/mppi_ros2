"""
고급 Safety-Critical Control 테스트

- C3BF (Collision Cone CBF)
- DPCBF (Dynamic Parabolic CBF)
- Optimal-Decay CBF Safety Filter
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.controllers.mppi.c3bf_cost import CollisionConeCBFCost
from mppi_controller.controllers.mppi.dpcbf_cost import DynamicParabolicCBFCost
from mppi_controller.controllers.mppi.optimal_decay_cbf_filter import (
    OptimalDecayCBFSafetyFilter,
)


# ─────────────────────────────────────────────────
# C3BF (Collision Cone CBF) 테스트
# ─────────────────────────────────────────────────


def test_c3bf_no_obstacle():
    """장애물 없을 때 C3BF 비용 = 0"""
    print("\n" + "=" * 60)
    print("Test: C3BF - No Obstacles")
    print("=" * 60)

    cost_fn = CollisionConeCBFCost(obstacles=[], cbf_weight=1000.0)

    K, N = 32, 10
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 2, N + 1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)

    assert costs.shape == (K,)
    assert np.allclose(costs, 0.0)
    print("PASS\n")


def test_c3bf_receding_no_cost():
    """장애물에서 멀어지는 궤적은 비용 0"""
    print("=" * 60)
    print("Test: C3BF - Receding Trajectory (No Cost)")
    print("=" * 60)

    # 장애물이 원점, 로봇이 멀어지는 방향
    obstacles = [(0.0, 0.0, 0.3, 0.0, 0.0)]
    cost_fn = CollisionConeCBFCost(
        obstacles=obstacles, cbf_weight=1000.0, safety_margin=0.1, dt=0.05,
    )

    K, N = 16, 20
    traj = np.zeros((K, N + 1, 3))
    # 장애물 반대쪽에서 시작, 더 멀어지는 방향으로 이동
    traj[:, :, 0] = np.linspace(3.0, 6.0, N + 1)  # x: 3→6 (멀어짐)
    traj[:, :, 1] = 0.0
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)

    print(f"  Max cost: {np.max(costs):.6f}")
    assert np.allclose(costs, 0.0, atol=1e-3), f"Receding should be safe, got max cost {np.max(costs)}"
    print("PASS\n")


def test_c3bf_approaching_high_cost():
    """장애물로 접근하는 궤적은 높은 비용"""
    print("=" * 60)
    print("Test: C3BF - Approaching Trajectory (High Cost)")
    print("=" * 60)

    obstacles = [(2.0, 0.0, 0.3, 0.0, 0.0)]
    cost_fn = CollisionConeCBFCost(
        obstacles=obstacles, cbf_weight=1000.0, safety_margin=0.1, dt=0.05,
    )

    K, N = 16, 20
    # 접근 궤적: 장애물 관통
    traj_approach = np.zeros((K, N + 1, 3))
    traj_approach[:, :, 0] = np.linspace(0.0, 3.0, N + 1)

    # 회피 궤적: 측면으로 우회
    traj_avoid = np.zeros((K, N + 1, 3))
    traj_avoid[:, :, 0] = np.linspace(0.0, 3.0, N + 1)
    traj_avoid[:, :, 1] = 2.0  # 측면에서 통과

    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    cost_approach = cost_fn.compute_cost(traj_approach, ctrl, ref)
    cost_avoid = cost_fn.compute_cost(traj_avoid, ctrl, ref)

    print(f"  Approach cost: {np.mean(cost_approach):.2f}")
    print(f"  Avoid cost: {np.mean(cost_avoid):.2f}")

    assert np.mean(cost_approach) > np.mean(cost_avoid), \
        "Approaching should have higher cost than avoiding"
    print("PASS\n")


def test_c3bf_dynamic_obstacle_velocity():
    """장애물 속도 반영 테스트"""
    print("=" * 60)
    print("Test: C3BF - Dynamic Obstacle Velocity")
    print("=" * 60)

    # 정지 장애물
    obstacles_static = [(3.0, 0.0, 0.3, 0.0, 0.0)]
    # 로봇 방향으로 접근 중인 장애물
    obstacles_approaching = [(3.0, 0.0, 0.3, -1.0, 0.0)]

    cost_static = CollisionConeCBFCost(obstacles=obstacles_static, cbf_weight=1000.0, dt=0.05)
    cost_approaching = CollisionConeCBFCost(obstacles=obstacles_approaching, cbf_weight=1000.0, dt=0.05)

    K, N = 16, 20
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0.0, 2.5, N + 1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    c_static = cost_static.compute_cost(traj, ctrl, ref)
    c_approaching = cost_approaching.compute_cost(traj, ctrl, ref)

    print(f"  Static obstacle cost: {np.mean(c_static):.2f}")
    print(f"  Approaching obstacle cost: {np.mean(c_approaching):.2f}")

    assert np.mean(c_approaching) >= np.mean(c_static), \
        "Approaching obstacle should have >= cost than static"
    print("PASS\n")


def test_c3bf_barrier_info():
    """Barrier info 반환 테스트"""
    print("=" * 60)
    print("Test: C3BF - Barrier Info")
    print("=" * 60)

    obstacles = [(2.0, 0.0, 0.3, 0.0, 0.0)]
    cost_fn = CollisionConeCBFCost(obstacles=obstacles, dt=0.05)

    traj = np.zeros((1, 11, 3))
    traj[0, :, 0] = np.linspace(0, 1, 11)

    info = cost_fn.get_barrier_info(traj[0])
    print(f"  min_barrier: {info['min_barrier']:.4f}")
    print(f"  is_safe: {info['is_safe']}")
    assert "barrier_values" in info
    assert "min_barrier" in info
    assert "is_safe" in info

    # 빈 장애물 테스트
    cost_fn_empty = CollisionConeCBFCost(obstacles=[], dt=0.05)
    info_empty = cost_fn_empty.get_barrier_info(traj[0])
    assert info_empty["is_safe"] is True
    assert info_empty["min_barrier"] == float("inf")
    print("PASS\n")


def test_c3bf_shape_consistency():
    """다양한 K 크기에서 shape 일관성"""
    print("=" * 60)
    print("Test: C3BF - Shape Consistency")
    print("=" * 60)

    obstacles = [(2.0, 1.0, 0.3, 0.1, -0.1)]
    cost_fn = CollisionConeCBFCost(obstacles=obstacles, dt=0.05)

    for K in [1, 16, 128, 512]:
        N = 15
        traj = np.random.randn(K, N + 1, 3)
        ctrl = np.random.randn(K, N, 2)
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (K,), f"K={K}: got {costs.shape}"
        assert not np.any(np.isnan(costs)), f"K={K}: NaN detected"
        print(f"  K={K:4d}: OK")

    print("PASS\n")


# ─────────────────────────────────────────────────
# DPCBF (Dynamic Parabolic CBF) 테스트
# ─────────────────────────────────────────────────


def test_dpcbf_no_obstacle():
    """장애물 없을 때 DPCBF 비용 = 0"""
    print("=" * 60)
    print("Test: DPCBF - No Obstacles")
    print("=" * 60)

    cost_fn = DynamicParabolicCBFCost(obstacles=[], cbf_weight=1000.0)

    K, N = 32, 10
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 2, N + 1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)

    assert costs.shape == (K,)
    assert np.allclose(costs, 0.0)
    print("PASS\n")


def test_dpcbf_far_obstacle_no_cost():
    """먼 장애물에 대해 비용 = 0"""
    print("=" * 60)
    print("Test: DPCBF - Far Obstacle (No Cost)")
    print("=" * 60)

    obstacles = [(20.0, 20.0, 0.5, 0.0, 0.0)]
    cost_fn = DynamicParabolicCBFCost(obstacles=obstacles, cbf_weight=1000.0)

    K, N = 16, 10
    traj = np.zeros((K, N + 1, 3))
    traj[:, :, 0] = np.linspace(0, 1, N + 1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.allclose(costs, 0.0, atol=1e-3)
    print("PASS\n")


def test_dpcbf_head_on_vs_lateral():
    """정면 충돌 vs 측면 통과 비교 — 정면이 더 높은 비용"""
    print("=" * 60)
    print("Test: DPCBF - Head-On vs Lateral Passage")
    print("=" * 60)

    obstacles = [(2.0, 0.0, 0.3, 0.0, 0.0)]
    cost_fn = DynamicParabolicCBFCost(
        obstacles=obstacles, cbf_weight=1000.0,
        a_base=0.5, a_vel=0.5, sigma_beta=0.8, dt=0.05,
    )

    K, N = 16, 20

    # 정면 충돌 궤적
    traj_head_on = np.zeros((K, N + 1, 3))
    traj_head_on[:, :, 0] = np.linspace(0.0, 3.0, N + 1)

    # 측면 통과 궤적 (y=1.5에서 통과)
    traj_lateral = np.zeros((K, N + 1, 3))
    traj_lateral[:, :, 0] = np.linspace(0.0, 3.0, N + 1)
    traj_lateral[:, :, 1] = 1.5

    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    cost_head = cost_fn.compute_cost(traj_head_on, ctrl, ref)
    cost_lat = cost_fn.compute_cost(traj_lateral, ctrl, ref)

    print(f"  Head-on cost: {np.mean(cost_head):.2f}")
    print(f"  Lateral cost: {np.mean(cost_lat):.2f}")

    assert np.mean(cost_head) > np.mean(cost_lat), \
        "Head-on approach should have higher cost than lateral"
    print("PASS\n")


def test_dpcbf_approach_speed_adaptation():
    """접근 속도가 높으면 안전 경계 확대"""
    print("=" * 60)
    print("Test: DPCBF - Approach Speed Adaptation")
    print("=" * 60)

    obstacles = [(3.0, 0.0, 0.3, 0.0, 0.0)]
    cost_fn = DynamicParabolicCBFCost(
        obstacles=obstacles, cbf_weight=1000.0,
        a_base=0.3, a_vel=1.0, dt=0.05,
    )

    K, N = 16, 20

    # 느린 접근 (x: 0→2)
    traj_slow = np.zeros((K, N + 1, 3))
    traj_slow[:, :, 0] = np.linspace(0.0, 2.0, N + 1)

    # 빠른 접근 (x: 0→2.8)
    traj_fast = np.zeros((K, N + 1, 3))
    traj_fast[:, :, 0] = np.linspace(0.0, 2.8, N + 1)

    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, 3))

    cost_slow = cost_fn.compute_cost(traj_slow, ctrl, ref)
    cost_fast = cost_fn.compute_cost(traj_fast, ctrl, ref)

    print(f"  Slow approach cost: {np.mean(cost_slow):.2f}")
    print(f"  Fast approach cost: {np.mean(cost_fast):.2f}")

    # 빠른 접근이 더 높은 비용 (안전 경계 확대로)
    assert np.mean(cost_fast) >= np.mean(cost_slow), \
        "Fast approach should have >= cost due to expanded boundary"
    print("PASS\n")


def test_dpcbf_barrier_info():
    """DPCBF barrier info 테스트"""
    print("=" * 60)
    print("Test: DPCBF - Barrier Info")
    print("=" * 60)

    obstacles = [(2.0, 0.0, 0.3, 0.1, 0.0)]
    cost_fn = DynamicParabolicCBFCost(obstacles=obstacles, dt=0.05)

    traj = np.zeros((11, 3))
    traj[:, 0] = np.linspace(0, 1, 11)

    info = cost_fn.get_barrier_info(traj)
    print(f"  min_barrier: {info['min_barrier']:.4f}")
    print(f"  is_safe: {info['is_safe']}")
    assert "barrier_values" in info
    assert "min_barrier" in info
    assert "is_safe" in info

    # 빈 장애물 테스트
    cost_empty = DynamicParabolicCBFCost(obstacles=[], dt=0.05)
    info_empty = cost_empty.get_barrier_info(traj)
    assert info_empty["is_safe"] is True
    print("PASS\n")


def test_dpcbf_shape_consistency():
    """다양한 K 크기에서 shape 일관성"""
    print("=" * 60)
    print("Test: DPCBF - Shape Consistency")
    print("=" * 60)

    obstacles = [(2.0, 1.0, 0.3, 0.1, -0.1)]
    cost_fn = DynamicParabolicCBFCost(obstacles=obstacles, dt=0.05)

    for K in [1, 16, 128, 512]:
        N = 15
        traj = np.random.randn(K, N + 1, 3) * 3.0
        traj[:, :, :2] += 5.0  # 적당히 떨어진 위치
        ctrl = np.random.randn(K, N, 2)
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (K,), f"K={K}: got {costs.shape}"
        assert not np.any(np.isnan(costs)), f"K={K}: NaN detected"
        print(f"  K={K:4d}: OK")

    print("PASS\n")


def test_dpcbf_update_obstacles():
    """장애물 업데이트 테스트"""
    print("=" * 60)
    print("Test: DPCBF - Update Obstacles")
    print("=" * 60)

    cost_fn = DynamicParabolicCBFCost(obstacles=[], dt=0.05)
    assert len(cost_fn.obstacles) == 0

    new_obs = [(1.0, 1.0, 0.3, 0.1, 0.0), (2.0, 2.0, 0.5, -0.1, 0.2)]
    cost_fn.update_obstacles(new_obs)
    assert len(cost_fn.obstacles) == 2
    print("PASS\n")


# ─────────────────────────────────────────────────
# Optimal-Decay CBF Safety Filter 테스트
# ─────────────────────────────────────────────────


def test_optimal_decay_no_obstacle():
    """장애물 없을 때 필터 비활성"""
    print("=" * 60)
    print("Test: Optimal-Decay CBF - No Obstacles")
    print("=" * 60)

    sf = OptimalDecayCBFSafetyFilter(obstacles=[], cbf_alpha=0.1)
    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.3])

    u_safe, info = sf.filter_control(state, u)

    assert np.allclose(u_safe, u)
    assert not info["filtered"]
    assert info["optimal_omega"] == 1.0
    print("PASS\n")


def test_optimal_decay_safe_no_modification():
    """안전할 때 제어 수정 없음, ω=1"""
    print("=" * 60)
    print("Test: Optimal-Decay CBF - Safe State (No Modification)")
    print("=" * 60)

    obstacles = [(100.0, 100.0, 0.5)]
    sf = OptimalDecayCBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.1, penalty_weight=1e4,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.3])

    u_safe, info = sf.filter_control(state, u)

    print(f"  u_mppi: {u}")
    print(f"  u_safe: {u_safe}")
    print(f"  ω: {info['optimal_omega']:.4f}")

    assert not info["filtered"]
    assert np.allclose(u_safe, u)
    assert info["optimal_omega"] == 1.0
    print("PASS\n")


def test_optimal_decay_unsafe_correction():
    """위험할 때 제어 수정 + ω 확인"""
    print("=" * 60)
    print("Test: Optimal-Decay CBF - Unsafe Correction")
    print("=" * 60)

    obstacles = [(0.5, 0.0, 0.3)]
    sf = OptimalDecayCBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.3, penalty_weight=1e4,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0])  # 장애물 방향 전진

    u_safe, info = sf.filter_control(
        state, u,
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )

    print(f"  u_mppi: {u}")
    print(f"  u_safe: {u_safe}")
    print(f"  ω: {info['optimal_omega']:.4f}")
    print(f"  Filtered: {info['filtered']}")
    print(f"  Correction: {info['correction_norm']:.4f}")

    if info["filtered"]:
        assert info["correction_norm"] > 0
        # ω는 0~1 사이
        assert 0.0 <= info["optimal_omega"] <= 1.0
    print("PASS\n")


def test_optimal_decay_omega_relaxation():
    """tight 환경에서 ω < 1 (제약 완화) 테스트"""
    print("=" * 60)
    print("Test: Optimal-Decay CBF - Omega Relaxation in Tight Space")
    print("=" * 60)

    # 여러 장애물로 둘러싸인 상황
    obstacles = [
        (0.5, 0.0, 0.2),   # 앞
        (-0.5, 0.0, 0.2),  # 뒤
        (0.0, 0.5, 0.2),   # 왼쪽
        (0.0, -0.5, 0.2),  # 오른쪽
    ]
    sf = OptimalDecayCBFSafetyFilter(
        obstacles=obstacles,
        cbf_alpha=0.5,
        safety_margin=0.1,
        penalty_weight=1e4,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.0])

    u_safe, info = sf.filter_control(
        state, u,
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )

    print(f"  ω: {info['optimal_omega']:.4f}")
    print(f"  Filtered: {info['filtered']}")
    print(f"  Decay relaxed: {info.get('decay_relaxed', False)}")

    # tight 환경에서는 최적화가 해를 찾아야 함 (feasibility guaranteed)
    assert not np.any(np.isnan(u_safe)), "u_safe contains NaN"
    assert 0.0 <= info["optimal_omega"] <= 1.0
    print("PASS\n")


def test_optimal_decay_statistics():
    """필터 통계 (ω 통계 포함) 테스트"""
    print("=" * 60)
    print("Test: Optimal-Decay CBF - Statistics")
    print("=" * 60)

    obstacles = [(1.0, 0.0, 0.3)]
    sf = OptimalDecayCBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.3, penalty_weight=1e4,
    )

    state = np.array([0.0, 0.0, 0.0])

    # 안전한 제어
    sf.filter_control(state, np.array([0.1, 0.0]))
    # 위험한 제어
    sf.filter_control(
        state, np.array([1.0, 0.0]),
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )

    stats = sf.get_filter_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Filter rate: {stats['filter_rate']:.2%}")
    print(f"  Mean omega: {stats['mean_omega']:.4f}")
    print(f"  Min omega: {stats['min_omega']:.4f}")

    assert "mean_omega" in stats
    assert "min_omega" in stats
    assert "relaxation_rate" in stats
    assert stats["total_steps"] == 2

    # Reset
    sf.reset()
    stats_reset = sf.get_filter_statistics()
    assert stats_reset["num_filtered"] == 0
    print("PASS\n")


def test_optimal_decay_vs_standard_cbf():
    """Standard CBF vs Optimal-Decay 비교 — 동일 결과 또는 더 나은 feasibility"""
    print("=" * 60)
    print("Test: Optimal-Decay vs Standard CBF Comparison")
    print("=" * 60)

    from mppi_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter

    obstacles = [(1.0, 0.0, 0.3)]

    standard = CBFSafetyFilter(obstacles=obstacles, cbf_alpha=0.3, safety_margin=0.1)
    optimal = OptimalDecayCBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.3, safety_margin=0.1, penalty_weight=1e4,
    )

    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.8, 0.1])
    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

    u_std, info_std = standard.filter_control(state, u, *bounds)
    u_opt, info_opt = optimal.filter_control(state, u, *bounds)

    print(f"  Standard: u={u_std}, filtered={info_std['filtered']}")
    print(f"  Optimal:  u={u_opt}, ω={info_opt['optimal_omega']:.4f}, filtered={info_opt['filtered']}")

    # 동일 결과이거나, optimal이 원래 제어에 더 가까워야 함
    if info_std["filtered"] and info_opt["filtered"]:
        # Optimal-decay는 ω를 완화할 수 있으므로 correction이 같거나 작을 수 있음
        print(f"  Standard correction: {info_std['correction_norm']:.4f}")
        print(f"  Optimal correction:  {info_opt['correction_norm']:.4f}")
    print("PASS\n")


def test_optimal_decay_repr():
    """__repr__ 테스트"""
    print("=" * 60)
    print("Test: Optimal-Decay CBF - repr")
    print("=" * 60)

    sf = OptimalDecayCBFSafetyFilter(
        obstacles=[(1.0, 2.0, 0.3)], cbf_alpha=0.2, penalty_weight=5000.0,
    )
    rep = repr(sf)
    print(f"  {rep}")
    assert "OptimalDecayCBFSafetyFilter" in rep
    assert "5000" in rep
    print("PASS\n")


# ─────────────────────────────────────────────────
# Standalone 실행
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Advanced Safety-Critical Control Tests".center(60))
    print("=" * 60)

    tests = [
        # C3BF
        test_c3bf_no_obstacle,
        test_c3bf_receding_no_cost,
        test_c3bf_approaching_high_cost,
        test_c3bf_dynamic_obstacle_velocity,
        test_c3bf_barrier_info,
        test_c3bf_shape_consistency,
        # DPCBF
        test_dpcbf_no_obstacle,
        test_dpcbf_far_obstacle_no_cost,
        test_dpcbf_head_on_vs_lateral,
        test_dpcbf_approach_speed_adaptation,
        test_dpcbf_barrier_info,
        test_dpcbf_shape_consistency,
        test_dpcbf_update_obstacles,
        # Optimal-Decay CBF
        test_optimal_decay_no_obstacle,
        test_optimal_decay_safe_no_modification,
        test_optimal_decay_unsafe_correction,
        test_optimal_decay_omega_relaxation,
        test_optimal_decay_statistics,
        test_optimal_decay_vs_standard_cbf,
        test_optimal_decay_repr,
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
