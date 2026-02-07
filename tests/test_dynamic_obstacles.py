"""
동적 장애물 통합 테스트

Shield-MPPI/CBF-MPPI의 동적 장애물 업데이트, 파이프라인 통합 검증.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    CBFMPPIParams,
    ShieldMPPIParams,
)
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.perception.obstacle_detector import ObstacleDetector, DetectedObstacle
from mppi_controller.perception.obstacle_tracker import ObstacleTracker


def test_shield_update_obstacles():
    """Shield-MPPI 동적 장애물 업데이트 반영"""
    print("\n" + "=" * 60)
    print("Test 1: Shield update_obstacles()")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles_initial = [(10.0, 10.0, 0.5)]

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles_initial,
        cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        shield_enabled=True,
    )
    controller = ShieldMPPIController(model, params)

    # 초기: 장애물이 먼 곳
    assert controller.cbf_params.cbf_obstacles == obstacles_initial
    assert controller.cbf_cost.obstacles == obstacles_initial

    # 업데이트: 장애물이 가까이
    new_obstacles = [(0.8, 0.0, 0.3)]
    controller.update_obstacles(new_obstacles)

    # 모든 내부 참조가 갱신되었는지 확인
    assert controller.cbf_cost.obstacles == new_obstacles, \
        "cbf_cost.obstacles not updated"
    assert controller.cbf_params.cbf_obstacles == new_obstacles, \
        "cbf_params.cbf_obstacles not updated"

    print("  cbf_cost.obstacles updated: OK")
    print("  cbf_params.cbf_obstacles updated: OK")
    print("PASS: Shield update_obstacles works\n")


def test_cbf_update_obstacles():
    """CBF-MPPI 동적 장애물 업데이트 반영"""
    print("=" * 60)
    print("Test 2: CBF update_obstacles()")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles_initial = [(10.0, 10.0, 0.5)]

    params = CBFMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles_initial,
        cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
    )
    controller = CBFMPPIController(model, params)

    new_obstacles = [(1.0, 0.0, 0.5), (0.0, 1.0, 0.3)]
    controller.update_obstacles(new_obstacles)

    assert controller.cbf_cost.obstacles == new_obstacles
    print("  cbf_cost.obstacles updated: OK")
    print("PASS: CBF update_obstacles works\n")


def test_shield_cbf_shield_batch_uses_updated_obstacles():
    """Shield의 _cbf_shield_batch가 업데이트된 장애물을 사용하는지 확인"""
    print("=" * 60)
    print("Test 3: _cbf_shield_batch uses updated obstacles")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=32, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=[(100.0, 100.0, 0.5)],  # 먼 장애물
        cbf_alpha=0.3, cbf_safety_margin=0.1,
        shield_enabled=True,
    )
    controller = ShieldMPPIController(model, params)

    K = 32
    states = np.zeros((K, 3))
    controls = np.column_stack([np.full(K, 1.0), np.zeros(K)])

    # 먼 장애물 → 개입 없음
    safe1, intervened1, _ = controller._cbf_shield_batch(states, controls)
    assert np.sum(intervened1) == 0, "Should not intervene with far obstacle"

    # 장애물을 가까이 이동
    controller.update_obstacles([(0.8, 0.0, 0.3)])

    # 가까운 장애물 → 개입 발생
    safe2, intervened2, _ = controller._cbf_shield_batch(states, controls)
    assert np.sum(intervened2) > 0, "Should intervene with close obstacle"
    assert np.all(safe2[:, 0] < controls[:, 0]), "Velocity should be reduced"

    print(f"  Far obstacle: {np.sum(intervened1)} interventions")
    print(f"  Close obstacle: {np.sum(intervened2)} interventions")
    print("PASS: _cbf_shield_batch uses updated obstacles\n")


def test_dynamic_obstacle_safety():
    """이동 장애물 환경에서 Shield-MPPI 안전성"""
    print("=" * 60)
    print("Test 4: Dynamic Obstacle Safety")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=256, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=[(3.0, 0.0, 0.5)],
        cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        shield_enabled=True,
    )
    controller = ShieldMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 3, 16)

    np.random.seed(42)
    min_dist_global = np.inf

    for step in range(30):
        # 장애물이 서서히 접근
        t = step * 0.05
        obs_x = 3.0 - t * 0.5  # 점점 가까워짐
        obstacles = [(obs_x, 0.0, 0.5)]
        controller.update_obstacles(obstacles)

        control, info = controller.compute_control(state, reference)
        state = model.step(state, control, 0.05)

        # 안전 거리 확인
        dist = np.sqrt((state[0] - obs_x) ** 2 + (state[1] - 0.0) ** 2) - 0.5
        min_dist_global = min(min_dist_global, dist)

    print(f"  Min distance to moving obstacle: {min_dist_global:.4f}m")
    # Shield는 CBF로 속도를 제한하므로 충돌을 방지해야 함
    # 약간의 수치 오차는 허용
    assert min_dist_global > -0.1, \
        f"Shield should prevent collision, but min dist = {min_dist_global:.4f}"
    print("PASS: Dynamic obstacle safety maintained\n")


def test_detector_tracker_pipeline():
    """ObstacleDetector → ObstacleTracker → controller 전체 파이프라인"""
    print("=" * 60)
    print("Test 5: Detector → Tracker → Controller Pipeline")
    print("=" * 60)

    # 1. 가상 LaserScan 생성 (전방 2m에 장애물)
    detector = ObstacleDetector(cluster_threshold=0.3, min_cluster_size=3)

    num_rays = 360
    angle_min = -np.pi
    angle_increment = 2 * np.pi / num_rays
    ranges = np.full(num_rays, 10.0)

    # 전방에 장애물
    center_idx = num_rays // 2
    for i in range(-4, 5):
        ranges[(center_idx + i) % num_rays] = 2.0

    # 2. 감지
    detections = detector.detect(ranges, angle_min, angle_increment)
    assert len(detections) >= 1, "Should detect at least one obstacle"
    print(f"  Detections: {len(detections)}")

    # 3. 추적
    tracker = ObstacleTracker()
    tracker.update(detections, dt=0.1)
    obstacles = tracker.get_obstacles_as_tuples()
    assert len(obstacles) >= 1, "Should have at least one tracked obstacle"
    print(f"  Tracked obstacles: {obstacles}")

    # 4. 컨트롤러 업데이트
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = ShieldMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=[],
        cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        shield_enabled=True,
    )
    controller = ShieldMPPIController(model, params)
    controller.update_obstacles(obstacles)

    # 5. 제어 계산
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 2, 16)

    control, info = controller.compute_control(state, reference)
    assert not np.any(np.isnan(control)), "Control contains NaN"

    print(f"  Control: v={control[0]:.3f}, w={control[1]:.3f}")
    print(f"  Shield intervention rate: {info['shield_intervention_rate']:.2%}")
    print("PASS: Full pipeline works\n")


def test_update_with_empty_obstacles():
    """장애물이 없을 때 업데이트"""
    print("=" * 60)
    print("Test 6: Update with Empty Obstacles")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = ShieldMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=[(3.0, 0.0, 0.5)],
        cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        shield_enabled=True,
    )
    controller = ShieldMPPIController(model, params)

    # 장애물 제거
    controller.update_obstacles([])
    assert controller.cbf_cost.obstacles == []
    assert controller.cbf_params.cbf_obstacles == []

    # 제어 계산이 에러 없이 동작
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 2, 16)

    control, info = controller.compute_control(state, reference)
    assert not np.any(np.isnan(control))
    print(f"  Control with no obstacles: v={control[0]:.3f}, w={control[1]:.3f}")
    print("PASS: Empty obstacles handled\n")


def test_frequent_obstacle_updates():
    """빈번한 장애물 업데이트 안정성"""
    print("=" * 60)
    print("Test 7: Frequent Obstacle Updates Stability")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = ShieldMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=[(5.0, 0.0, 0.5)],
        cbf_weight=1000.0, cbf_alpha=0.3, cbf_safety_margin=0.1,
        shield_enabled=True,
    )
    controller = ShieldMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 2, 16)

    np.random.seed(42)

    for i in range(20):
        # 매 스텝마다 장애물 위치 변경
        obs_x = 3.0 + np.sin(i * 0.5)
        obs_y = np.cos(i * 0.3)
        controller.update_obstacles([(obs_x, obs_y, 0.5)])

        control, info = controller.compute_control(state, reference)
        state = model.step(state, control, 0.05)

        assert not np.any(np.isnan(control)), f"NaN at step {i}"
        assert not np.any(np.isinf(control)), f"Inf at step {i}"

    print(f"  20 steps with frequent updates completed")
    print(f"  Final state: {state}")
    print("PASS: Frequent updates stable\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Dynamic Obstacles Integration Tests".center(60))
    print("=" * 60)

    try:
        test_shield_update_obstacles()
        test_cbf_update_obstacles()
        test_shield_cbf_shield_batch_uses_updated_obstacles()
        test_dynamic_obstacle_safety()
        test_detector_tracker_pipeline()
        test_update_with_empty_obstacles()
        test_frequent_obstacle_updates()

        print("=" * 60)
        print("All 7 Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
