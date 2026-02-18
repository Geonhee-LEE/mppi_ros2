"""
Dynamic obstacle with various motion strategies.

동적 장애물과 다양한 운동 전략:
- BouncingMotion: 벽에 반사
- ChasingMotion: 로봇 추적
- EvadingMotion: 경로를 막으며 회피
- CrossingMotion: 주기적으로 경로 횡단
- CircularMotion: 원형 궤도
"""

import numpy as np
from typing import Tuple, Optional


class DynamicObstacle:
    """
    동적 장애물.

    motion_strategy에 따라 시간에 따른 위치/속도를 계산.
    """

    def __init__(self, motion_strategy, radius: float = 0.4):
        self.motion = motion_strategy
        self.radius = radius

    def get_position(self, t: float) -> Tuple[float, float]:
        return self.motion.get_position(t)

    def get_velocity(self, t: float) -> Tuple[float, float]:
        return self.motion.get_velocity(t)

    def as_3tuple(self, t: float) -> Tuple[float, float, float]:
        """(x, y, radius) — static cost function용"""
        px, py = self.get_position(t)
        return (px, py, self.radius)

    def as_5tuple(self, t: float) -> Tuple[float, float, float, float, float]:
        """(x, y, radius, vx, vy) — C3BF/DPCBF용"""
        px, py = self.get_position(t)
        vx, vy = self.get_velocity(t)
        return (px, py, self.radius, vx, vy)


# ── Motion Strategies ───────────────────────────────────────────────────────

class BouncingMotion:
    """벽에 반사하며 이동"""

    def __init__(
        self,
        start: Tuple[float, float],
        velocity: Tuple[float, float],
        bounds: Tuple[float, float, float, float] = (-6.0, -6.0, 6.0, 6.0),
    ):
        self._start = np.array(start, dtype=float)
        self._vel = np.array(velocity, dtype=float)
        self._bounds = bounds  # (x_min, y_min, x_max, y_max)
        # precompute trajectory cache
        self._cache_dt = 0.01
        self._cache = None
        self._cache_duration = 0.0

    def _ensure_cache(self, t: float):
        if self._cache is not None and t <= self._cache_duration:
            return
        duration = max(t + 1.0, 30.0)
        n = int(duration / self._cache_dt) + 1
        positions = np.zeros((n, 2))
        velocities = np.zeros((n, 2))
        pos = self._start.copy()
        vel = self._vel.copy()
        x_min, y_min, x_max, y_max = self._bounds

        for i in range(n):
            positions[i] = pos
            velocities[i] = vel
            pos = pos + vel * self._cache_dt
            if pos[0] <= x_min or pos[0] >= x_max:
                vel[0] = -vel[0]
                pos[0] = np.clip(pos[0], x_min, x_max)
            if pos[1] <= y_min or pos[1] >= y_max:
                vel[1] = -vel[1]
                pos[1] = np.clip(pos[1], y_min, y_max)

        self._cache = positions
        self._vel_cache = velocities
        self._cache_duration = duration

    def get_position(self, t: float) -> Tuple[float, float]:
        self._ensure_cache(t)
        idx = min(int(t / self._cache_dt), len(self._cache) - 1)
        p = self._cache[idx]
        return (float(p[0]), float(p[1]))

    def get_velocity(self, t: float) -> Tuple[float, float]:
        self._ensure_cache(t)
        idx = min(int(t / self._cache_dt), len(self._vel_cache) - 1)
        v = self._vel_cache[idx]
        return (float(v[0]), float(v[1]))


class ChasingMotion:
    """로봇을 추적하는 운동 (비례 추적)"""

    def __init__(
        self,
        start: Tuple[float, float],
        speed: float = 0.4,
        dt: float = 0.01,
    ):
        self._pos = np.array(start, dtype=float)
        self._speed = speed
        self._dt = dt
        self._target = np.array(start, dtype=float)
        self._vel = np.zeros(2)
        self._positions = [self._pos.copy()]
        self._velocities = [self._vel.copy()]
        self._times = [0.0]

    def set_target(self, target_x: float, target_y: float):
        """로봇 위치로 추적 목표 업데이트"""
        self._target = np.array([target_x, target_y])

    def _advance_to(self, t: float):
        while self._times[-1] < t - 1e-9:
            direction = self._target - self._pos
            dist = np.linalg.norm(direction)
            if dist > 0.05:
                self._vel = direction / dist * self._speed
            else:
                self._vel = np.zeros(2)
            self._pos = self._pos + self._vel * self._dt
            self._positions.append(self._pos.copy())
            self._velocities.append(self._vel.copy())
            self._times.append(self._times[-1] + self._dt)

    def get_position(self, t: float) -> Tuple[float, float]:
        self._advance_to(t)
        p = self._positions[-1]
        return (float(p[0]), float(p[1]))

    def get_velocity(self, t: float) -> Tuple[float, float]:
        self._advance_to(t)
        v = self._velocities[-1]
        return (float(v[0]), float(v[1]))

    def reset(self, start: Tuple[float, float]):
        self._pos = np.array(start, dtype=float)
        self._vel = np.zeros(2)
        self._positions = [self._pos.copy()]
        self._velocities = [self._vel.copy()]
        self._times = [0.0]


class EvadingMotion:
    """로봇 경로를 막으며 회피하는 운동"""

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        speed: float = 0.3,
        dt: float = 0.01,
        evade_distance: float = 1.5,
    ):
        self._pos = np.array(start, dtype=float)
        self._goal = np.array(goal, dtype=float)
        self._speed = speed
        self._dt = dt
        self._evade_dist = evade_distance
        self._robot_pos = np.array(start, dtype=float)
        self._vel = np.zeros(2)
        self._positions = [self._pos.copy()]
        self._velocities = [self._vel.copy()]
        self._times = [0.0]

    def set_robot_position(self, rx: float, ry: float):
        self._robot_pos = np.array([rx, ry])

    def _advance_to(self, t: float):
        while self._times[-1] < t - 1e-9:
            to_robot = self._robot_pos - self._pos
            dist_to_robot = np.linalg.norm(to_robot)

            if dist_to_robot < self._evade_dist:
                # 로봇이 가까우면 로봇-목표 사이를 유지하며 회피
                to_goal = self._goal - self._robot_pos
                if np.linalg.norm(to_goal) > 0.1:
                    block_pos = self._robot_pos + to_goal / np.linalg.norm(to_goal) * 1.5
                    direction = block_pos - self._pos
                else:
                    direction = -to_robot
            else:
                # 로봇이 멀면 목표 경로의 중간점으로 이동
                midpoint = (self._robot_pos + self._goal) / 2
                direction = midpoint - self._pos

            dist = np.linalg.norm(direction)
            if dist > 0.05:
                self._vel = direction / dist * self._speed
            else:
                self._vel = np.zeros(2)

            self._pos = self._pos + self._vel * self._dt
            self._positions.append(self._pos.copy())
            self._velocities.append(self._vel.copy())
            self._times.append(self._times[-1] + self._dt)

    def get_position(self, t: float) -> Tuple[float, float]:
        self._advance_to(t)
        p = self._positions[-1]
        return (float(p[0]), float(p[1]))

    def get_velocity(self, t: float) -> Tuple[float, float]:
        self._advance_to(t)
        v = self._velocities[-1]
        return (float(v[0]), float(v[1]))


class CrossingMotion:
    """주기적으로 경로를 횡단하는 운동"""

    def __init__(
        self,
        center: Tuple[float, float],
        amplitude: float = 2.0,
        period: float = 5.0,
        direction: float = 0.0,
    ):
        self._center = np.array(center, dtype=float)
        self._amplitude = amplitude
        self._period = period
        self._cos_d = np.cos(direction)
        self._sin_d = np.sin(direction)

    def get_position(self, t: float) -> Tuple[float, float]:
        offset = self._amplitude * np.sin(2.0 * np.pi * t / self._period)
        x = self._center[0] + offset * self._cos_d
        y = self._center[1] + offset * self._sin_d
        return (float(x), float(y))

    def get_velocity(self, t: float) -> Tuple[float, float]:
        omega = 2.0 * np.pi / self._period
        v_mag = self._amplitude * omega * np.cos(omega * t)
        vx = v_mag * self._cos_d
        vy = v_mag * self._sin_d
        return (float(vx), float(vy))


class CircularMotion:
    """원형 궤도 운동"""

    def __init__(
        self,
        center: Tuple[float, float],
        orbit_radius: float = 2.0,
        angular_velocity: float = 0.5,
        phase: float = 0.0,
    ):
        self._center = np.array(center, dtype=float)
        self._orbit_r = orbit_radius
        self._omega = angular_velocity
        self._phase = phase

    def get_position(self, t: float) -> Tuple[float, float]:
        angle = self._omega * t + self._phase
        x = self._center[0] + self._orbit_r * np.cos(angle)
        y = self._center[1] + self._orbit_r * np.sin(angle)
        return (float(x), float(y))

    def get_velocity(self, t: float) -> Tuple[float, float]:
        angle = self._omega * t + self._phase
        vx = -self._orbit_r * self._omega * np.sin(angle)
        vy = self._orbit_r * self._omega * np.cos(angle)
        return (float(vx), float(vy))
