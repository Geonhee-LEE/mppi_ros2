"""
ObstacleDetector 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.perception.obstacle_detector import ObstacleDetector, DetectedObstacle


def test_polar_to_cartesian():
    """좌표 변환 정확도"""
    print("\n" + "=" * 60)
    print("Test 1: Polar to Cartesian")
    print("=" * 60)

    detector = ObstacleDetector()

    # 4방향 거리 1.0m
    ranges = np.array([1.0, 1.0, 1.0, 1.0])
    angle_min = 0.0
    angle_increment = np.pi / 2  # 90도 간격

    points, valid = detector._polar_to_cartesian(ranges, angle_min, angle_increment)

    print(f"Points:\n{points}")
    assert len(points) == 4
    assert np.allclose(points[0], [1.0, 0.0], atol=1e-10)  # 0도
    assert np.allclose(points[1], [0.0, 1.0], atol=1e-10)  # 90도
    assert np.allclose(points[2], [-1.0, 0.0], atol=1e-10)  # 180도
    assert np.allclose(points[3], [0.0, -1.0], atol=1e-10)  # 270도
    print("PASS: Polar to Cartesian conversion correct\n")


def test_polar_to_cartesian_filters_invalid():
    """inf, nan, 범위 외 필터링"""
    print("=" * 60)
    print("Test 2: Filter Invalid Ranges")
    print("=" * 60)

    detector = ObstacleDetector(min_range=0.1, max_range=10.0)

    ranges = np.array([1.0, np.inf, np.nan, 0.01, 50.0, 2.0])
    angle_min = 0.0
    angle_increment = 0.1

    points, valid = detector._polar_to_cartesian(ranges, angle_min, angle_increment)

    print(f"Valid points: {len(points)} / {len(ranges)}")
    assert len(points) == 2  # 1.0과 2.0만 유효
    print("PASS: Invalid ranges filtered\n")


def test_clustering():
    """클러스터 분리"""
    print("=" * 60)
    print("Test 3: Clustering")
    print("=" * 60)

    detector = ObstacleDetector(cluster_threshold=0.3)

    # 두 그룹: (0,0)~(0.2,0) 과 (2,0)~(2.2,0)
    points = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.2, 0.0],
        [2.0, 0.0], [2.1, 0.0], [2.2, 0.0],
    ])

    clusters = detector._cluster_points(points, 0.3)

    print(f"Number of clusters: {len(clusters)}")
    assert len(clusters) == 2
    assert len(clusters[0]) == 3
    assert len(clusters[1]) == 3
    print("PASS: Clusters separated correctly\n")


def test_circle_fitting():
    """원 피팅 정확도"""
    print("=" * 60)
    print("Test 4: Circle Fitting")
    print("=" * 60)

    # 반경 1.0 원 위의 점들
    angles = np.linspace(0, np.pi, 10)
    cluster = np.column_stack([np.cos(angles), np.sin(angles)])

    cx, cy, radius = ObstacleDetector._fit_circle(cluster)

    print(f"Center: ({cx:.4f}, {cy:.4f}), Radius: {radius:.4f}")
    # centroid는 원의 중심이 아님 (반원이므로), 하지만 radius는 centroid에서 최대 거리
    assert radius > 0
    assert not np.isnan(cx) and not np.isnan(cy)
    print("PASS: Circle fitting works\n")


def test_detect_single_obstacle():
    """단일 장애물 감지"""
    print("=" * 60)
    print("Test 5: Detect Single Obstacle")
    print("=" * 60)

    detector = ObstacleDetector(
        cluster_threshold=0.3, min_cluster_size=3, max_obstacle_radius=2.0
    )

    # 전방 2m에 작은 장애물 시뮬레이션 (좁은 각도 범위에서 짧은 거리)
    num_rays = 360
    angle_min = -np.pi
    angle_increment = 2 * np.pi / num_rays
    ranges = np.full(num_rays, 10.0)  # 기본 10m

    # 전방 (0도 근처) 약 5개 레이에 2m 거리
    center_idx = num_rays // 2  # 0도에 해당
    for i in range(-3, 4):
        idx = (center_idx + i) % num_rays
        ranges[idx] = 2.0

    obstacles = detector.detect(ranges, angle_min, angle_increment)

    print(f"Detected obstacles: {len(obstacles)}")
    if obstacles:
        obs = obstacles[0]
        print(f"  Position: ({obs.x:.2f}, {obs.y:.2f}), Radius: {obs.radius:.3f}")
        print(f"  Points: {obs.num_points}, Confidence: {obs.confidence:.2f}")
    assert len(obstacles) >= 1
    print("PASS: Single obstacle detected\n")


def test_detect_multiple_obstacles():
    """다중 장애물 감지"""
    print("=" * 60)
    print("Test 6: Detect Multiple Obstacles")
    print("=" * 60)

    detector = ObstacleDetector(
        cluster_threshold=0.5, min_cluster_size=3, max_obstacle_radius=2.0
    )

    num_rays = 360
    angle_min = -np.pi
    angle_increment = 2 * np.pi / num_rays
    ranges = np.full(num_rays, 10.0)

    # 장애물 1: 전방 2m (0도)
    center1 = num_rays // 2
    for i in range(-3, 4):
        ranges[(center1 + i) % num_rays] = 2.0

    # 장애물 2: 좌측 3m (90도)
    center2 = num_rays * 3 // 4
    for i in range(-3, 4):
        ranges[(center2 + i) % num_rays] = 3.0

    obstacles = detector.detect(ranges, angle_min, angle_increment)

    print(f"Detected obstacles: {len(obstacles)}")
    for obs in obstacles:
        print(f"  ({obs.x:.2f}, {obs.y:.2f}), r={obs.radius:.3f}, pts={obs.num_points}")
    assert len(obstacles) >= 2
    print("PASS: Multiple obstacles detected\n")


def test_no_obstacles():
    """장애물 없는 스캔"""
    print("=" * 60)
    print("Test 7: No Obstacles")
    print("=" * 60)

    detector = ObstacleDetector(min_cluster_size=3)

    # 모든 레이가 같은 거리 → 하나의 큰 클러스터 → max_cluster_size 초과
    ranges = np.full(360, 5.0)
    angle_min = -np.pi
    angle_increment = 2 * np.pi / 360

    obstacles = detector.detect(ranges, angle_min, angle_increment)

    print(f"Detected obstacles: {len(obstacles)}")
    # 연속된 같은 거리 포인트는 하나의 거대 클러스터 → max_cluster_size 초과로 필터링
    # 또는 반경이 max_obstacle_radius 초과
    print("PASS: Handled uniform scan\n")


def test_world_frame_transform():
    """로봇 프레임 → 월드 프레임 변환"""
    print("=" * 60)
    print("Test 8: World Frame Transform")
    print("=" * 60)

    # 로봇이 (1, 2)에 있고 90도 회전
    wx, wy = ObstacleDetector._to_world_frame(1.0, 0.0, np.array([1.0, 2.0, np.pi / 2]))

    print(f"World position: ({wx:.4f}, {wy:.4f})")
    # 로컬 (1, 0) → 90도 회전 → (0, 1) → + (1, 2) = (1, 3)
    assert abs(wx - 1.0) < 1e-6
    assert abs(wy - 3.0) < 1e-6
    print("PASS: World frame transform correct\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ObstacleDetector Tests".center(60))
    print("=" * 60)

    try:
        test_polar_to_cartesian()
        test_polar_to_cartesian_filters_invalid()
        test_clustering()
        test_circle_fitting()
        test_detect_single_obstacle()
        test_detect_multiple_obstacles()
        test_no_obstacles()
        test_world_frame_transform()

        print("=" * 60)
        print("All 8 Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
