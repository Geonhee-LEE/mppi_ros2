"""
LaserScan 기반 장애물 감지

LaserScan 데이터(polar coordinates)에서 원형 장애물을 추출.
순수 numpy 구현으로 ROS2 의존성 없음.

Pipeline:
    1. Polar → Cartesian 변환
    2. 유효 포인트 필터링 (inf, nan 제거)
    3. Sequential distance-based 클러스터링 (1D scan ordering)
    4. 클러스터별 최소 외접원 피팅
    5. (선택) robot_pose로 world frame 변환
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DetectedObstacle:
    """감지된 장애물"""
    x: float
    y: float
    radius: float
    num_points: int
    confidence: float  # num_points 기반 신뢰도 [0, 1]


class ObstacleDetector:
    """
    LaserScan → 원형 장애물 추출

    Args:
        cluster_threshold: 인접 포인트 간 최대 거리 (m) - 클러스터 분리 기준
        min_cluster_size: 최소 클러스터 포인트 수
        max_cluster_size: 최대 클러스터 포인트 수 (벽 등 제외)
        max_obstacle_radius: 최대 장애물 반경 (m)
        min_range: 최소 유효 거리 (m)
        max_range: 최대 유효 거리 (m)
    """

    def __init__(
        self,
        cluster_threshold: float = 0.3,
        min_cluster_size: int = 3,
        max_cluster_size: int = 50,
        max_obstacle_radius: float = 2.0,
        min_range: float = 0.05,
        max_range: float = 30.0,
    ):
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.max_obstacle_radius = max_obstacle_radius
        self.min_range = min_range
        self.max_range = max_range

    def detect(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        robot_pose: Optional[np.ndarray] = None,
    ) -> List[DetectedObstacle]:
        """
        LaserScan에서 장애물 감지

        Args:
            ranges: (M,) 거리 배열
            angle_min: 시작 각도 (rad)
            angle_increment: 각도 증분 (rad)
            robot_pose: (3,) [x, y, theta] - world frame 변환용 (None이면 로봇 프레임)

        Returns:
            감지된 장애물 리스트
        """
        # 1. Polar → Cartesian
        points, valid_mask = self._polar_to_cartesian(
            ranges, angle_min, angle_increment
        )

        if len(points) < self.min_cluster_size:
            return []

        # 2. 클러스터링
        clusters = self._cluster_points(points, self.cluster_threshold)

        # 3. 클러스터별 원 피팅
        obstacles = []
        for cluster in clusters:
            n = len(cluster)
            if n < self.min_cluster_size or n > self.max_cluster_size:
                continue

            cx, cy, radius = self._fit_circle(cluster)

            if radius > self.max_obstacle_radius:
                continue

            # 신뢰도: 포인트 수 기반 (많을수록 높음, 최대 1.0)
            confidence = min(1.0, n / 20.0)

            # World frame 변환
            if robot_pose is not None:
                wx, wy = self._to_world_frame(cx, cy, robot_pose)
            else:
                wx, wy = cx, cy

            obstacles.append(DetectedObstacle(
                x=wx, y=wy, radius=radius,
                num_points=n, confidence=confidence,
            ))

        return obstacles

    def _polar_to_cartesian(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Polar → Cartesian 변환 + 유효 포인트 필터링

        Returns:
            points: (N, 2) 유효 Cartesian 좌표
            valid_mask: (M,) 유효 인덱스 마스크
        """
        M = len(ranges)
        angles = angle_min + np.arange(M) * angle_increment

        # 유효 거리 필터링
        valid = (
            np.isfinite(ranges)
            & (ranges >= self.min_range)
            & (ranges <= self.max_range)
        )

        valid_ranges = ranges[valid]
        valid_angles = angles[valid]

        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)

        points = np.column_stack([x, y])
        return points, valid

    def _cluster_points(
        self, points: np.ndarray, threshold: float
    ) -> List[np.ndarray]:
        """
        Sequential distance-based 클러스터링 (1D scan ordering)

        인접 포인트 간 거리가 threshold 이하이면 같은 클러스터.
        """
        if len(points) == 0:
            return []

        clusters = []
        current_cluster = [points[0]]

        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i - 1])
            if dist <= threshold:
                current_cluster.append(points[i])
            else:
                clusters.append(np.array(current_cluster))
                current_cluster = [points[i]]

        # 마지막 클러스터
        clusters.append(np.array(current_cluster))

        return clusters

    @staticmethod
    def _fit_circle(cluster: np.ndarray) -> Tuple[float, float, float]:
        """
        최소 외접원 피팅: center = centroid, radius = max distance from center

        Args:
            cluster: (N, 2) 클러스터 포인트

        Returns:
            (cx, cy, radius)
        """
        centroid = np.mean(cluster, axis=0)
        distances = np.linalg.norm(cluster - centroid, axis=1)
        radius = np.max(distances)

        # 최소 반경 보장 (포인트가 거의 같은 위치)
        radius = max(radius, 0.05)

        return float(centroid[0]), float(centroid[1]), float(radius)

    @staticmethod
    def _to_world_frame(
        local_x: float, local_y: float, robot_pose: np.ndarray
    ) -> Tuple[float, float]:
        """로봇 프레임 → 월드 프레임 변환"""
        rx, ry, rtheta = robot_pose[0], robot_pose[1], robot_pose[2]
        cos_t = np.cos(rtheta)
        sin_t = np.sin(rtheta)
        wx = rx + cos_t * local_x - sin_t * local_y
        wy = ry + sin_t * local_x + cos_t * local_y
        return float(wx), float(wy)
