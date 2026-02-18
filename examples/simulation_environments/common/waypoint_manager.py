"""
WaypointStateMachine — 순차적 웨이포인트 네비게이션.

상태 전이: NAVIGATING → DWELLING → NAVIGATING → ... → COMPLETED

각 웨이포인트에 도착하면 dwell_time만큼 대기 후 다음 웨이포인트로 전환.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from enum import Enum


class WaypointState(Enum):
    NAVIGATING = "navigating"
    DWELLING = "dwelling"
    COMPLETED = "completed"


class WaypointStateMachine:
    """
    순차적 웨이포인트 네비게이션 상태 머신.

    Args:
        waypoints: [(x, y, theta), ...] 웨이포인트 목록
        arrival_threshold: 도착 판정 거리 (m)
        dwell_time: 각 웨이포인트에서의 대기 시간 (s)
        heading_weight: 레퍼런스 heading 보간 가중치
    """

    def __init__(
        self,
        waypoints: List[Tuple[float, float, float]],
        arrival_threshold: float = 0.3,
        dwell_time: float = 1.0,
        heading_weight: float = 1.0,
    ):
        self._waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self._threshold = arrival_threshold
        self._dwell_time = dwell_time
        self._heading_weight = heading_weight

        self._current_idx = 0
        self._state = WaypointState.NAVIGATING
        self._dwell_start = 0.0

    @property
    def state(self) -> WaypointState:
        return self._state

    @property
    def current_waypoint_idx(self) -> int:
        return self._current_idx

    @property
    def current_waypoint(self) -> Optional[np.ndarray]:
        if self._current_idx < len(self._waypoints):
            return self._waypoints[self._current_idx]
        return None

    @property
    def progress(self) -> float:
        """완료 비율 (0~1)"""
        total = len(self._waypoints)
        if total == 0:
            return 1.0
        return min(1.0, self._current_idx / total)

    @property
    def is_completed(self) -> bool:
        return self._state == WaypointState.COMPLETED

    def reset(self):
        self._current_idx = 0
        self._state = WaypointState.NAVIGATING
        self._dwell_start = 0.0

    def update(self, state: np.ndarray, t: float) -> str:
        """
        상태 업데이트.

        Args:
            state: (nx,) 현재 로봇 상태 [x, y, theta, ...]
            t: 현재 시간

        Returns:
            상태 문자열: 'navigating', 'dwelling', 'completed'
        """
        if self._state == WaypointState.COMPLETED:
            return "completed"

        wp = self._waypoints[self._current_idx]
        dist = np.sqrt((state[0] - wp[0]) ** 2 + (state[1] - wp[1]) ** 2)

        if self._state == WaypointState.NAVIGATING:
            if dist < self._threshold:
                self._state = WaypointState.DWELLING
                self._dwell_start = t

        elif self._state == WaypointState.DWELLING:
            if t - self._dwell_start >= self._dwell_time:
                self._current_idx += 1
                if self._current_idx >= len(self._waypoints):
                    self._state = WaypointState.COMPLETED
                else:
                    self._state = WaypointState.NAVIGATING

        return self._state.value

    def get_reference_fn(
        self,
        N: int,
        dt: float,
        nx: int = 3,
    ) -> Callable[[float], np.ndarray]:
        """
        현재 웨이포인트 기반 레퍼런스 궤적 함수 생성.

        반환하는 함수: t -> (N+1, nx)

        Args:
            N: 예측 구간 길이
            dt: 시간 간격
            nx: 상태 차원
        """
        def reference_fn(t: float) -> np.ndarray:
            ref = np.zeros((N + 1, nx))

            if self._state == WaypointState.COMPLETED:
                # 마지막 웨이포인트에 정지
                last_wp = self._waypoints[-1]
                for i in range(N + 1):
                    ref[i, :min(nx, len(last_wp))] = last_wp[:min(nx, len(last_wp))]
                return ref

            # 현재 + 남은 웨이포인트
            remaining = self._waypoints[self._current_idx:]

            # 각 스텝에 대해 가장 가까운 미래 웨이포인트 할당
            for i in range(N + 1):
                # 간단한 직선 보간: 현재 웨이포인트를 유지
                wp_idx = min(i // max(1, N // max(1, len(remaining))),
                             len(remaining) - 1)
                wp = remaining[wp_idx]
                ref[i, :min(nx, len(wp))] = wp[:min(nx, len(wp))]

            return ref

        return reference_fn

    def get_info(self) -> dict:
        """현재 상태 정보"""
        return {
            "state": self._state.value,
            "current_idx": self._current_idx,
            "total_waypoints": len(self._waypoints),
            "progress": self.progress,
            "current_waypoint": (
                self._waypoints[self._current_idx].tolist()
                if self._current_idx < len(self._waypoints)
                else None
            ),
        }
