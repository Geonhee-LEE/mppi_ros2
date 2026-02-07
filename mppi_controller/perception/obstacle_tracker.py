"""
다중 객체 추적 (Multi-Object Tracker)

프레임 간 장애물 매칭 + 속도 추정.
Nearest-neighbor association 기반.

Pipeline:
    1. 기존 트랙과 새 검출 간 거리 행렬 계산
    2. Greedy nearest-neighbor 매칭
    3. 매칭된 트랙 업데이트 (위치, 속도 EMA)
    4. 미매칭 검출 → 새 트랙 생성
    5. 미매칭 트랙 → lost count 증가 → 삭제
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from mppi_controller.perception.obstacle_detector import DetectedObstacle


@dataclass
class TrackedObstacle:
    """추적 중인 장애물"""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.5
    age: int = 0          # 추적 프레임 수
    lost_count: int = 0   # 연속 미매칭 수


class ObstacleTracker:
    """
    다중 객체 추적기 (nearest-neighbor association)

    Args:
        max_association_dist: 매칭 최대 거리 (m)
        max_lost_frames: 미매칭 허용 프레임 수 (초과 시 삭제)
        velocity_smoothing: 속도 EMA 계수 (0~1, 낮을수록 smooth)
    """

    def __init__(
        self,
        max_association_dist: float = 1.0,
        max_lost_frames: int = 5,
        velocity_smoothing: float = 0.3,
    ):
        self.max_association_dist = max_association_dist
        self.max_lost_frames = max_lost_frames
        self.velocity_smoothing = velocity_smoothing

        self.tracks: List[TrackedObstacle] = []
        self._next_id = 0

    def update(
        self, detections: List[DetectedObstacle], dt: float
    ) -> List[TrackedObstacle]:
        """
        트래커 업데이트

        Args:
            detections: 현재 프레임 감지 결과
            dt: 프레임 간 시간 간격 (s)

        Returns:
            현재 활성 트랙 리스트
        """
        if dt <= 0:
            return self.tracks

        num_tracks = len(self.tracks)
        num_dets = len(detections)

        # 매칭할 것이 없는 경우
        if num_tracks == 0 and num_dets == 0:
            return self.tracks

        # 트랙만 있고 검출이 없는 경우
        if num_dets == 0:
            self._increment_lost_counts()
            self._remove_lost_tracks()
            return self.tracks

        # 검출만 있고 트랙이 없는 경우
        if num_tracks == 0:
            for det in detections:
                self._create_track(det)
            return self.tracks

        # 1. 거리 행렬 계산
        dist_matrix = self._compute_distance_matrix(detections)

        # 2. Greedy nearest-neighbor 매칭
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = (
            self._greedy_matching(dist_matrix, num_tracks, num_dets)
        )

        # 3. 매칭된 트랙 업데이트
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            self._update_track(self.tracks[track_idx], detections[det_idx], dt)

        # 4. 미매칭 검출 → 새 트랙
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # 5. 미매칭 트랙 → lost count 증가
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].lost_count += 1

        # 6. lost 트랙 삭제
        self._remove_lost_tracks()

        return self.tracks

    def get_obstacles_as_tuples(self) -> List[Tuple[float, float, float]]:
        """controller.update_obstacles()에 전달할 (x, y, r) 리스트"""
        return [(t.x, t.y, t.radius) for t in self.tracks]

    def get_predicted_obstacles(
        self, dt_ahead: float
    ) -> List[Tuple[float, float, float]]:
        """속도 기반 미래 위치 예측 (MPPI horizon 고려)"""
        predicted = []
        for t in self.tracks:
            px = t.x + t.vx * dt_ahead
            py = t.y + t.vy * dt_ahead
            predicted.append((px, py, t.radius))
        return predicted

    def reset(self):
        """트래커 초기화"""
        self.tracks = []
        self._next_id = 0

    def _compute_distance_matrix(
        self, detections: List[DetectedObstacle]
    ) -> np.ndarray:
        """트랙-검출 간 유클리드 거리 행렬 (num_tracks x num_dets)"""
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        dist_matrix = np.zeros((num_tracks, num_dets))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                dist_matrix[i, j] = np.sqrt(
                    (track.x - det.x) ** 2 + (track.y - det.y) ** 2
                )

        return dist_matrix

    def _greedy_matching(
        self,
        dist_matrix: np.ndarray,
        num_tracks: int,
        num_dets: int,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Greedy nearest-neighbor 매칭

        Returns:
            (matched_track_indices, matched_det_indices,
             unmatched_track_indices, unmatched_det_indices)
        """
        matched_tracks = []
        matched_dets = []
        used_tracks = set()
        used_dets = set()

        # 거리 기반 정렬 후 greedy 매칭
        flat_indices = np.argsort(dist_matrix, axis=None)

        for flat_idx in flat_indices:
            track_idx = flat_idx // num_dets
            det_idx = flat_idx % num_dets

            if track_idx in used_tracks or det_idx in used_dets:
                continue

            if dist_matrix[track_idx, det_idx] > self.max_association_dist:
                break

            matched_tracks.append(track_idx)
            matched_dets.append(det_idx)
            used_tracks.add(track_idx)
            used_dets.add(det_idx)

        unmatched_tracks = [i for i in range(num_tracks) if i not in used_tracks]
        unmatched_dets = [j for j in range(num_dets) if j not in used_dets]

        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets

    def _update_track(
        self, track: TrackedObstacle, det: DetectedObstacle, dt: float
    ):
        """매칭된 트랙 업데이트 (위치 + 속도 EMA)"""
        # 속도 계산
        new_vx = (det.x - track.x) / dt
        new_vy = (det.y - track.y) / dt

        # EMA smoothing
        alpha = self.velocity_smoothing
        track.vx = alpha * new_vx + (1 - alpha) * track.vx
        track.vy = alpha * new_vy + (1 - alpha) * track.vy

        # 위치 및 반경 업데이트
        track.x = det.x
        track.y = det.y
        track.radius = det.radius

        track.age += 1
        track.lost_count = 0

    def _create_track(self, det: DetectedObstacle):
        """새 트랙 생성"""
        track = TrackedObstacle(
            id=self._next_id,
            x=det.x,
            y=det.y,
            radius=det.radius,
        )
        self.tracks.append(track)
        self._next_id += 1

    def _increment_lost_counts(self):
        """모든 트랙의 lost_count 증가"""
        for track in self.tracks:
            track.lost_count += 1

    def _remove_lost_tracks(self):
        """lost_count 초과 트랙 삭제"""
        self.tracks = [
            t for t in self.tracks if t.lost_count <= self.max_lost_frames
        ]
