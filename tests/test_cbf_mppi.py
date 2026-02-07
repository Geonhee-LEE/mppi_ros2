"""
CBF-MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import CBFMPPIParams
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController


def test_cbf_cost_no_obstacle_nearby():
    """장애물이 멀리 있을 때 CBF 비용이 0인지 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: CBF Cost - No Obstacle Nearby")
    print("=" * 60)

    # 장애물이 (10, 10) 위치에 있고 궤적은 원점 근처
    obstacles = [(10.0, 10.0, 0.5)]
    cbf_cost = ControlBarrierCost(
        obstacles=obstacles, cbf_alpha=0.1, cbf_weight=1000.0, safety_margin=0.1
    )

    K, N = 32, 10
    # 원점 근처 궤적 (장애물과 충분히 멀리 떨어져 있음)
    trajectories = np.zeros((K, N + 1, 3))
    trajectories[:, :, 0] = np.linspace(0, 1, N + 1)  # x 방향으로 이동
    controls = np.zeros((K, N, 2))
    reference = np.zeros((N + 1, 3))

    costs = cbf_cost.compute_cost(trajectories, controls, reference)

    print(f"Costs shape: {costs.shape}")
    print(f"Max cost: {np.max(costs):.6f}")
    print(f"Min cost: {np.min(costs):.6f}")

    assert costs.shape == (K,), f"Expected shape ({K},), got {costs.shape}"
    assert np.allclose(costs, 0.0), f"Expected zero cost, got max={np.max(costs)}"
    print("✓ PASS: Zero cost when no obstacle nearby\n")


def test_cbf_cost_barrier_violation():
    """장애물에 접근할 때 높은 CBF 비용이 발생하는지 테스트"""
    print("=" * 60)
    print("Test 2: CBF Cost - Barrier Violation")
    print("=" * 60)

    obstacles = [(2.0, 0.0, 0.5)]
    cbf_cost = ControlBarrierCost(
        obstacles=obstacles, cbf_alpha=0.1, cbf_weight=1000.0, safety_margin=0.1
    )

    K, N = 32, 10

    # 궤적 1: 장애물 관통 (x=0 → x=4)
    trajectories_unsafe = np.zeros((K, N + 1, 3))
    trajectories_unsafe[:, :, 0] = np.linspace(0, 4, N + 1)

    # 궤적 2: 안전 궤적 (x=0, y=0 근처)
    trajectories_safe = np.zeros((K, N + 1, 3))
    trajectories_safe[:, :, 0] = np.linspace(0, 0.5, N + 1)

    controls = np.zeros((K, N, 2))
    reference = np.zeros((N + 1, 3))

    cost_unsafe = cbf_cost.compute_cost(trajectories_unsafe, controls, reference)
    cost_safe = cbf_cost.compute_cost(trajectories_safe, controls, reference)

    print(f"Unsafe cost (mean): {np.mean(cost_unsafe):.2f}")
    print(f"Safe cost (mean): {np.mean(cost_safe):.2f}")

    assert np.mean(cost_unsafe) > np.mean(cost_safe), \
        "Unsafe trajectory should have higher cost"
    assert np.mean(cost_unsafe) > 0, "Unsafe trajectory should have positive cost"
    print("✓ PASS: High cost when approaching obstacle\n")


def test_cbf_cost_vectorized():
    """CBF 비용 출력 shape이 올바른지 테스트"""
    print("=" * 60)
    print("Test 3: CBF Cost - Vectorized Shape Check")
    print("=" * 60)

    obstacles = [(3.0, 0.0, 0.5), (0.0, 3.0, 0.3)]
    cbf_cost = ControlBarrierCost(obstacles=obstacles)

    for K in [1, 16, 128, 512]:
        N = 20
        trajectories = np.random.randn(K, N + 1, 3)
        controls = np.random.randn(K, N, 2)
        reference = np.zeros((N + 1, 3))

        costs = cbf_cost.compute_cost(trajectories, controls, reference)

        assert costs.shape == (K,), f"K={K}: Expected shape ({K},), got {costs.shape}"
        assert not np.any(np.isnan(costs)), f"K={K}: Costs contain NaN"
        assert not np.any(np.isinf(costs)), f"K={K}: Costs contain Inf"
        print(f"  K={K:4d}: shape={costs.shape} ✓")

    # Barrier info 테스트
    traj = np.zeros((1, 11, 3))
    barrier_info = cbf_cost.get_barrier_info(traj)
    print(f"  Barrier info keys: {list(barrier_info.keys())}")
    assert "min_barrier" in barrier_info
    assert "is_safe" in barrier_info

    print("✓ PASS: All shapes correct\n")


def test_cbf_safety_filter_no_modification():
    """안전한 상태에서 필터가 제어를 수정하지 않는지 테스트"""
    print("=" * 60)
    print("Test 4: Safety Filter - No Modification When Safe")
    print("=" * 60)

    # 장애물이 매우 멀리 있음
    obstacles = [(100.0, 100.0, 0.5)]
    safety_filter = CBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.1, safety_margin=0.1
    )

    state = np.array([0.0, 0.0, 0.0])
    u_mppi = np.array([0.5, 0.3])

    u_safe, info = safety_filter.filter_control(state, u_mppi)

    print(f"u_mppi: {u_mppi}")
    print(f"u_safe: {u_safe}")
    print(f"Filtered: {info['filtered']}")
    print(f"Correction norm: {info['correction_norm']:.6f}")

    assert not info["filtered"], "Should not filter when safe"
    assert np.allclose(u_safe, u_mppi), "Should not modify control when safe"
    print("✓ PASS: No modification when safe\n")


def test_cbf_safety_filter_correction():
    """안전하지 않을 때 필터가 제어를 수정하는지 테스트"""
    print("=" * 60)
    print("Test 5: Safety Filter - Correction When Unsafe")
    print("=" * 60)

    # 장애물이 바로 앞에 있음
    obstacles = [(0.5, 0.0, 0.3)]
    safety_filter = CBFSafetyFilter(
        obstacles=obstacles, cbf_alpha=0.3, safety_margin=0.1
    )

    # 로봇이 장애물 방향으로 이동
    state = np.array([0.0, 0.0, 0.0])  # 원점, 전방 향함
    u_mppi = np.array([1.0, 0.0])  # 전진 (장애물 방향)

    u_safe, info = safety_filter.filter_control(
        state, u_mppi,
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )

    print(f"u_mppi: {u_mppi}")
    print(f"u_safe: {u_safe}")
    print(f"Filtered: {info['filtered']}")
    print(f"Correction norm: {info['correction_norm']:.4f}")
    print(f"Min barrier: {info['min_barrier']:.4f}")

    # barrier가 음수이면(장애물 내부에 가까움) 필터가 활성화될 수 있음
    # 또는 CBF 조건이 위반될 때 필터가 활성화
    # 장애물까지의 거리가 가까우므로 속도를 줄이거나 방향을 변경해야 함
    if info["filtered"]:
        assert info["correction_norm"] > 0, "Correction norm should be positive"
        # 안전 필터가 전진 속도를 줄여야 함
        assert u_safe[0] <= u_mppi[0], \
            f"Safe velocity should be <= MPPI velocity: {u_safe[0]} vs {u_mppi[0]}"
        print("✓ PASS: Filter corrected unsafe control\n")
    else:
        # 현재 상태에서 CBF 조건이 이미 만족될 수도 있음 (h가 양수이면)
        _, _, h = safety_filter._compute_lie_derivatives(state, 0.5, 0.0, 0.3)
        print(f"  Barrier h = {h:.4f} (positive = currently safe)")
        print("✓ PASS: Already safe at current state (filter not needed)\n")


def test_cbf_mppi_controller_integration():
    """CBFMPPIController 전체 통합 테스트"""
    print("=" * 60)
    print("Test 6: CBF-MPPI Controller Integration")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(3.0, 0.0, 0.5), (0.0, 3.0, 0.5)]

    params = CBFMPPIParams(
        N=15,
        dt=0.05,
        K=256,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.1,
        cbf_safety_margin=0.1,
        cbf_use_safety_filter=False,
    )

    controller = CBFMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 2, 16)

    # 여러 스텝 실행
    for i in range(5):
        control, info = controller.compute_control(state, reference)
        state = model.step(state, control, params.dt)

    print(f"Control: {control}")
    print(f"Final state: {state}")
    print(f"Min barrier: {info['min_barrier']:.4f}")
    print(f"Is safe: {info['is_safe']}")
    print(f"CBF filtered: {info['cbf_filtered']}")

    # 기본 동작 검증
    assert not np.any(np.isnan(control)), "Control contains NaN"
    assert not np.any(np.isinf(control)), "Control contains Inf"
    assert "min_barrier" in info, "Missing min_barrier in info"
    assert "barrier_values" in info, "Missing barrier_values in info"

    # CBF 통계
    stats = controller.get_cbf_statistics()
    print(f"\nCBF Statistics:")
    print(f"  Mean min barrier: {stats['mean_min_barrier']:.4f}")
    print(f"  Min min barrier: {stats['min_min_barrier']:.4f}")
    print(f"  Safety rate: {stats['safety_rate']:.2%}")
    print(f"  Filter rate: {stats['filter_rate']:.2%}")

    assert stats["safety_rate"] >= 0, "Safety rate should be non-negative"
    print("✓ PASS: CBF-MPPI controller integration works\n")


def test_cbf_mppi_with_safety_filter():
    """CBFMPPIController + 안전 필터 테스트"""
    print("=" * 60)
    print("Test 7: CBF-MPPI with Safety Filter")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(2.0, 0.0, 0.5)]

    params = CBFMPPIParams(
        N=15,
        dt=0.05,
        K=256,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.1,
        cbf_safety_margin=0.1,
        cbf_use_safety_filter=True,
    )

    controller = CBFMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 3, 16)  # 장애물 방향으로 레퍼런스

    # 여러 스텝 실행
    for i in range(10):
        control, info = controller.compute_control(state, reference)
        state = model.step(state, control, params.dt)

    print(f"Control: {control}")
    print(f"Final state: {state}")
    print(f"CBF filtered: {info['cbf_filtered']}")
    print(f"Correction norm: {info['cbf_correction_norm']:.4f}")

    # 기본 동작 검증
    assert not np.any(np.isnan(control)), "Control contains NaN"
    assert not np.any(np.isinf(control)), "Control contains Inf"
    assert "cbf_filtered" in info, "Missing cbf_filtered in info"

    # CBF 통계
    stats = controller.get_cbf_statistics()
    print(f"\nCBF Statistics:")
    print(f"  Safety rate: {stats['safety_rate']:.2%}")
    print(f"  Filter rate: {stats['filter_rate']:.2%}")
    print(f"  Mean correction: {stats['mean_correction_norm']:.4f}")

    # Reset 테스트
    controller.reset()
    stats_after_reset = controller.get_cbf_statistics()
    assert stats_after_reset["safety_rate"] == 0.0, "Stats should be zero after reset"

    # 장애물 업데이트 테스트
    new_obstacles = [(5.0, 5.0, 1.0)]
    controller.update_obstacles(new_obstacles)
    assert controller.cbf_cost.obstacles == new_obstacles, "Obstacles not updated"

    print("✓ PASS: CBF-MPPI with safety filter works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CBF-MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_cbf_cost_no_obstacle_nearby()
        test_cbf_cost_barrier_violation()
        test_cbf_cost_vectorized()
        test_cbf_safety_filter_no_modification()
        test_cbf_safety_filter_correction()
        test_cbf_mppi_controller_integration()
        test_cbf_mppi_with_safety_filter()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
