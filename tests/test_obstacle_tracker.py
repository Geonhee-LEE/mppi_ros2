"""
ObstacleTracker 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.perception.obstacle_detector import DetectedObstacle
from mppi_controller.perception.obstacle_tracker import ObstacleTracker, TrackedObstacle


def test_single_track_creation():
    """트랙 생성"""
    print("\n" + "=" * 60)
    print("Test 1: Single Track Creation")
    print("=" * 60)

    tracker = ObstacleTracker()

    detections = [DetectedObstacle(x=1.0, y=2.0, radius=0.5, num_points=10, confidence=0.5)]
    tracks = tracker.update(detections, dt=0.1)

    print(f"Tracks: {len(tracks)}")
    assert len(tracks) == 1
    assert abs(tracks[0].x - 1.0) < 1e-6
    assert abs(tracks[0].y - 2.0) < 1e-6
    assert abs(tracks[0].radius - 0.5) < 1e-6
    assert tracks[0].id == 0
    assert tracks[0].vx == 0.0
    assert tracks[0].vy == 0.0
    print("PASS: Single track created\n")


def test_track_association():
    """매칭 정확도"""
    print("=" * 60)
    print("Test 2: Track Association")
    print("=" * 60)

    tracker = ObstacleTracker(max_association_dist=1.0)

    # 프레임 1: 장애물 생성
    dets1 = [DetectedObstacle(x=1.0, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets1, dt=0.1)

    # 프레임 2: 장애물이 약간 이동
    dets2 = [DetectedObstacle(x=1.2, y=0.1, radius=0.5, num_points=10, confidence=0.5)]
    tracks = tracker.update(dets2, dt=0.1)

    print(f"Tracks: {len(tracks)}")
    assert len(tracks) == 1  # 같은 트랙으로 매칭
    assert abs(tracks[0].x - 1.2) < 1e-6
    assert abs(tracks[0].y - 0.1) < 1e-6
    assert tracks[0].age == 1
    print("PASS: Track associated correctly\n")


def test_velocity_estimation():
    """속도 추정"""
    print("=" * 60)
    print("Test 3: Velocity Estimation")
    print("=" * 60)

    tracker = ObstacleTracker(velocity_smoothing=1.0)  # No smoothing

    dt = 0.1

    # 프레임 1
    dets1 = [DetectedObstacle(x=0.0, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets1, dt=dt)

    # 프레임 2: 이동 (vx=1.0, vy=0.5)
    dets2 = [DetectedObstacle(x=0.1, y=0.05, radius=0.5, num_points=10, confidence=0.5)]
    tracks = tracker.update(dets2, dt=dt)

    expected_vx = 0.1 / dt  # 1.0
    expected_vy = 0.05 / dt  # 0.5

    print(f"Estimated velocity: ({tracks[0].vx:.2f}, {tracks[0].vy:.2f})")
    print(f"Expected velocity: ({expected_vx:.2f}, {expected_vy:.2f})")
    assert abs(tracks[0].vx - expected_vx) < 1e-6
    assert abs(tracks[0].vy - expected_vy) < 1e-6
    print("PASS: Velocity estimated correctly\n")


def test_velocity_ema_smoothing():
    """속도 EMA 스무딩"""
    print("=" * 60)
    print("Test 4: Velocity EMA Smoothing")
    print("=" * 60)

    tracker = ObstacleTracker(velocity_smoothing=0.5)

    dt = 0.1

    # 프레임 1
    dets1 = [DetectedObstacle(x=0.0, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets1, dt=dt)

    # 프레임 2: vx=1.0
    dets2 = [DetectedObstacle(x=0.1, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets2, dt=dt)

    # EMA: alpha=0.5, vx = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    assert abs(tracker.tracks[0].vx - 0.5) < 1e-6

    # 프레임 3: vx=1.0 다시
    dets3 = [DetectedObstacle(x=0.2, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets3, dt=dt)

    # EMA: vx = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    print(f"Smoothed velocity: {tracker.tracks[0].vx:.4f}")
    assert abs(tracker.tracks[0].vx - 0.75) < 1e-6
    print("PASS: EMA smoothing correct\n")


def test_track_deletion():
    """미매칭 트랙 삭제"""
    print("=" * 60)
    print("Test 5: Track Deletion")
    print("=" * 60)

    tracker = ObstacleTracker(max_lost_frames=2)

    # 프레임 1: 장애물 생성
    dets1 = [DetectedObstacle(x=1.0, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets1, dt=0.1)
    assert len(tracker.tracks) == 1

    # 프레임 2: 장애물 사라짐 (빈 검출)
    tracker.update([], dt=0.1)
    assert len(tracker.tracks) == 1  # lost_count=1, 아직 삭제 안됨

    # 프레임 3: 여전히 없음
    tracker.update([], dt=0.1)
    assert len(tracker.tracks) == 1  # lost_count=2, 아직 허용 범위

    # 프레임 4: 초과
    tracker.update([], dt=0.1)
    assert len(tracker.tracks) == 0  # lost_count=3 > max_lost_frames=2

    print("PASS: Lost tracks deleted correctly\n")


def test_multiple_tracks():
    """다중 트랙 관리"""
    print("=" * 60)
    print("Test 6: Multiple Tracks")
    print("=" * 60)

    tracker = ObstacleTracker(max_association_dist=1.0)

    # 프레임 1: 2개 장애물
    dets1 = [
        DetectedObstacle(x=1.0, y=0.0, radius=0.5, num_points=10, confidence=0.5),
        DetectedObstacle(x=5.0, y=5.0, radius=0.3, num_points=8, confidence=0.4),
    ]
    tracker.update(dets1, dt=0.1)
    assert len(tracker.tracks) == 2

    # 프레임 2: 둘 다 약간 이동
    dets2 = [
        DetectedObstacle(x=1.1, y=0.1, radius=0.5, num_points=10, confidence=0.5),
        DetectedObstacle(x=5.1, y=5.1, radius=0.3, num_points=8, confidence=0.4),
    ]
    tracks = tracker.update(dets2, dt=0.1)
    assert len(tracks) == 2

    # 각 트랙이 올바르게 업데이트되었는지 (id로 확인)
    ids = {t.id for t in tracks}
    assert len(ids) == 2  # 고유 ID
    print(f"Track IDs: {ids}")
    print("PASS: Multiple tracks managed\n")


def test_get_obstacles_as_tuples():
    """controller 호환 튜플 반환"""
    print("=" * 60)
    print("Test 7: Get Obstacles as Tuples")
    print("=" * 60)

    tracker = ObstacleTracker()

    dets = [
        DetectedObstacle(x=1.0, y=2.0, radius=0.5, num_points=10, confidence=0.5),
        DetectedObstacle(x=3.0, y=4.0, radius=0.3, num_points=8, confidence=0.4),
    ]
    tracker.update(dets, dt=0.1)

    tuples = tracker.get_obstacles_as_tuples()
    print(f"Obstacle tuples: {tuples}")

    assert len(tuples) == 2
    assert tuples[0] == (1.0, 2.0, 0.5)
    assert tuples[1] == (3.0, 4.0, 0.3)
    print("PASS: Tuples returned correctly\n")


def test_predicted_obstacles():
    """속도 기반 미래 위치 예측"""
    print("=" * 60)
    print("Test 8: Predicted Obstacles")
    print("=" * 60)

    tracker = ObstacleTracker(velocity_smoothing=1.0)

    # 프레임 1
    dets1 = [DetectedObstacle(x=0.0, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets1, dt=0.1)

    # 프레임 2: vx=1.0m/s
    dets2 = [DetectedObstacle(x=0.1, y=0.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets2, dt=0.1)

    # 0.5초 후 예측
    predicted = tracker.get_predicted_obstacles(dt_ahead=0.5)
    print(f"Predicted: {predicted}")

    # x = 0.1 + 1.0 * 0.5 = 0.6
    assert abs(predicted[0][0] - 0.6) < 1e-6
    assert abs(predicted[0][1] - 0.0) < 1e-6
    print("PASS: Future position predicted\n")


def test_reset():
    """트래커 초기화"""
    print("=" * 60)
    print("Test 9: Reset")
    print("=" * 60)

    tracker = ObstacleTracker()

    dets = [DetectedObstacle(x=1.0, y=2.0, radius=0.5, num_points=10, confidence=0.5)]
    tracker.update(dets, dt=0.1)
    assert len(tracker.tracks) == 1

    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker._next_id == 0
    print("PASS: Tracker reset\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ObstacleTracker Tests".center(60))
    print("=" * 60)

    try:
        test_single_track_creation()
        test_track_association()
        test_velocity_estimation()
        test_velocity_ema_smoothing()
        test_track_deletion()
        test_multiple_tracks()
        test_get_obstacles_as_tuples()
        test_predicted_obstacles()
        test_reset()

        print("=" * 60)
        print("All 9 Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
