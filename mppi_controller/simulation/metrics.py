"""
시뮬레이션 메트릭 계산

MPPI 성능 평가를 위한 메트릭 계산 함수.
"""

import numpy as np
from typing import Dict


def compute_metrics(history: Dict) -> Dict:
    """
    시뮬레이션 메트릭 계산

    Args:
        history: dict
            - state: (T, nx)
            - control: (T, nu)
            - reference: (T, nx)
            - solve_time: (T,)

    Returns:
        metrics: dict
            - position_rmse: float (m)
            - max_position_error: float (m)
            - heading_rmse: float (rad)
            - max_heading_error: float (rad)
            - control_rate: float (제어 변화율 평균)
            - max_control_rate: float (제어 변화율 최대)
            - mean_solve_time: float (ms)
            - max_solve_time: float (ms)
            - std_solve_time: float (ms)
    """
    states = history["state"]  # (T, nx)
    controls = history["control"]  # (T, nu)
    references = history["reference"]  # (T, nx)
    solve_times = history["solve_time"]  # (T,)

    T, nx = states.shape

    # 1. 위치 오차 (x, y)
    position_errors = np.linalg.norm(states[:, :2] - references[:, :2], axis=1)
    position_rmse = np.sqrt(np.mean(position_errors**2))
    max_position_error = np.max(position_errors)

    # 2. 각도 오차 (θ) - 각도 차이는 [-π, π]로 정규화
    if nx >= 3:
        heading_errors = angle_difference(states[:, 2], references[:, 2])
        heading_rmse = np.sqrt(np.mean(heading_errors**2))
        max_heading_error = np.max(np.abs(heading_errors))
    else:
        heading_rmse = 0.0
        max_heading_error = 0.0

    # 3. 제어 변화율 (부드러움)
    if T > 1:
        control_diffs = np.diff(controls, axis=0)  # (T-1, nu)
        control_rates = np.linalg.norm(control_diffs, axis=1)  # (T-1,)
        control_rate = np.mean(control_rates)
        max_control_rate = np.max(control_rates)
    else:
        control_rate = 0.0
        max_control_rate = 0.0

    # 4. 계산 시간 (ms)
    mean_solve_time = np.mean(solve_times) * 1000  # ms
    max_solve_time = np.max(solve_times) * 1000  # ms
    std_solve_time = np.std(solve_times) * 1000  # ms

    metrics = {
        "position_rmse": position_rmse,
        "max_position_error": max_position_error,
        "heading_rmse": heading_rmse,
        "max_heading_error": max_heading_error,
        "control_rate": control_rate,
        "max_control_rate": max_control_rate,
        "mean_solve_time": mean_solve_time,
        "max_solve_time": max_solve_time,
        "std_solve_time": std_solve_time,
    }

    return metrics


def angle_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    각도 차이 계산 ([-π, π] 범위로 정규화)

    Args:
        angle1: (T,) 각도 1 (rad)
        angle2: (T,) 각도 2 (rad)

    Returns:
        diff: (T,) 각도 차이 (rad)
    """
    diff = angle1 - angle2
    # [-π, π] 범위로 정규화
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    메트릭을 ASCII 테이블로 출력

    Args:
        metrics: 메트릭 딕셔너리
        title: 테이블 제목
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    print(f"{'Metric':<30} {'Value':>20}")
    print(f"{'-' * 60}")

    # 위치 메트릭
    print(f"{'Position RMSE (m)':<30} {metrics['position_rmse']:>20.4f}")
    print(f"{'Max Position Error (m)':<30} {metrics['max_position_error']:>20.4f}")

    # 각도 메트릭
    print(f"{'Heading RMSE (rad)':<30} {metrics['heading_rmse']:>20.4f}")
    print(f"{'Max Heading Error (rad)':<30} {metrics['max_heading_error']:>20.4f}")

    # 제어 메트릭
    print(f"{'Control Rate (avg)':<30} {metrics['control_rate']:>20.4f}")
    print(f"{'Control Rate (max)':<30} {metrics['max_control_rate']:>20.4f}")

    # 계산 시간 메트릭
    print(f"{'Mean Solve Time (ms)':<30} {metrics['mean_solve_time']:>20.2f}")
    print(f"{'Max Solve Time (ms)':<30} {metrics['max_solve_time']:>20.2f}")
    print(f"{'Std Solve Time (ms)':<30} {metrics['std_solve_time']:>20.2f}")

    print(f"{'=' * 60}\n")


def compare_metrics(metrics_list: list, labels: list, title: str = "Comparison"):
    """
    여러 메트릭 비교 테이블 출력

    Args:
        metrics_list: List[dict] 메트릭 딕셔너리 리스트
        labels: List[str] 각 메트릭 라벨
        title: 테이블 제목
    """
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")

    # 헤더
    header = f"{'Metric':<30}"
    for label in labels:
        header += f"{label:>15}"
    print(header)
    print(f"{'-' * 80}")

    # 메트릭 키
    metric_keys = [
        ("Position RMSE (m)", "position_rmse"),
        ("Max Position Error (m)", "max_position_error"),
        ("Heading RMSE (rad)", "heading_rmse"),
        ("Max Heading Error (rad)", "max_heading_error"),
        ("Control Rate (avg)", "control_rate"),
        ("Control Rate (max)", "max_control_rate"),
        ("Mean Solve Time (ms)", "mean_solve_time"),
        ("Max Solve Time (ms)", "max_solve_time"),
    ]

    for display_name, key in metric_keys:
        row = f"{display_name:<30}"
        for metrics in metrics_list:
            value = metrics[key]
            row += f"{value:>15.4f}"
        print(row)

    print(f"{'=' * 80}\n")
