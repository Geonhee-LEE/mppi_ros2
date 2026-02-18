"""
Extended environment metrics.

시뮬레이션 환경 전용 확장 메트릭:
- 충돌 횟수, 최소 간격, 안전율
- 경로 길이, 효율, 완주 시간
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_env_metrics(
    history: dict,
    obstacles_fn=None,
    robot_radius: float = 0.15,
    goal: Optional[np.ndarray] = None,
    goal_threshold: float = 0.3,
) -> dict:
    """
    환경 확장 메트릭 계산.

    Args:
        history: Simulator.get_history() 반환값
        obstacles_fn: t -> [(x,y,r), ...] 시간별 장애물 함수 (None이면 장애물 무시)
        robot_radius: 로봇 반경
        goal: 최종 목표점 (None이면 완주 판정 안함)
        goal_threshold: 목표 도달 판정 거리

    Returns:
        dict with keys:
        - collision_count: 충돌 발생 스텝 수
        - min_clearance: 최소 간격 (m)
        - safety_rate: 안전 비율 (0~1)
        - path_length: 실제 경로 길이 (m)
        - path_efficiency: 경로 효율 (직선 거리 / 실제 거리)
        - completion_time: 목표 도달 시간 (s, None이면 미달)
        - mean_tracking_error: 평균 추적 오차 (m)
    """
    states = history["state"]  # (T, nx)
    times = history["time"]  # (T,)
    refs = history["reference"]  # (T, nx)
    T = len(times)

    # 경로 길이
    diffs = np.diff(states[:, :2], axis=0)
    path_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

    # 직선 거리 (시작→끝)
    straight_dist = float(np.linalg.norm(states[-1, :2] - states[0, :2]))
    path_efficiency = straight_dist / max(path_length, 1e-6)

    # 추적 오차
    tracking_errors = np.linalg.norm(states[:, :2] - refs[:, :2], axis=1)
    mean_tracking_error = float(np.mean(tracking_errors))

    # 장애물 관련 메트릭
    collision_count = 0
    min_clearance = float("inf")
    clearances = []

    if obstacles_fn is not None:
        for i in range(T):
            t = times[i]
            obs = obstacles_fn(t)
            if not obs:
                clearances.append(float("inf"))
                continue
            px, py = states[i, 0], states[i, 1]
            min_d = float("inf")
            for ox, oy, r in obs:
                d = np.sqrt((px - ox) ** 2 + (py - oy) ** 2) - r - robot_radius
                min_d = min(min_d, d)
            clearances.append(min_d)
            if min_d < 0:
                collision_count += 1
            min_clearance = min(min_clearance, min_d)

    if not clearances:
        clearances = [float("inf")]
        min_clearance = float("inf")

    safety_rate = 1.0 - collision_count / max(T, 1)

    # 완주 시간
    completion_time = None
    if goal is not None:
        for i in range(T):
            dist_to_goal = np.linalg.norm(states[i, :2] - goal[:2])
            if dist_to_goal < goal_threshold:
                completion_time = float(times[i])
                break

    return {
        "collision_count": collision_count,
        "min_clearance": float(min_clearance),
        "safety_rate": safety_rate,
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "completion_time": completion_time,
        "mean_tracking_error": mean_tracking_error,
        "clearances": np.array(clearances),
    }


def print_env_comparison(
    results: Dict[str, dict],
    title: str = "Environment Comparison",
):
    """
    여러 컨트롤러의 환경 메트릭을 비교 출력.

    Args:
        results: {controller_name: env_metrics_dict}
        title: 제목
    """
    names = list(results.keys())
    if not names:
        return

    print()
    print("=" * 78)
    print(f"  {title}".center(78))
    print("=" * 78)

    # 헤더
    header = f"{'Controller':>18s} | {'Collisions':>10s} | {'MinClear':>8s} | {'Safety':>7s} | {'PathLen':>7s} | {'TrkErr':>7s} | {'Complete':>8s}"
    print(header)
    print("-" * 78)

    for name in names:
        m = results[name]
        collision_str = f"{m['collision_count']:>10d}"
        clear_str = f"{m['min_clearance']:>8.3f}" if m['min_clearance'] < 1e6 else f"{'N/A':>8s}"
        safety_str = f"{m['safety_rate']:>6.1%}"
        path_str = f"{m['path_length']:>7.2f}"
        trk_str = f"{m['mean_tracking_error']:>7.3f}"
        comp_str = (
            f"{m['completion_time']:>7.1f}s"
            if m['completion_time'] is not None
            else f"{'---':>8s}"
        )
        print(f"{name:>18s} | {collision_str} | {clear_str} | {safety_str} | {path_str} | {trk_str} | {comp_str}")

    print("=" * 78)
    print()
