"""
Static obstacle field generators.

다양한 형태의 정적 장애물 배치를 생성:
- generate_random_field: 무작위 분포 (Poisson-disk 유사)
- generate_corridor: 복도형 벽
- generate_slalom: 슬라럼 게이트
- generate_funnel: 좁아지는 깔때기
"""

import numpy as np
from typing import List, Tuple, Optional


def generate_random_field(
    n: int,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    radius_range: Tuple[float, float] = (0.2, 0.5),
    exclusion_zones: Optional[List[Tuple[float, float, float]]] = None,
    min_spacing: float = 0.3,
    seed: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """
    무작위 장애물 필드 생성 (최소 간격 보장).

    Args:
        n: 장애물 수
        x_range: x 범위
        y_range: y 범위
        radius_range: 반경 범위
        exclusion_zones: 장애물을 배치하지 않을 영역 [(x, y, radius), ...]
        min_spacing: 장애물 간 최소 간격
        seed: 랜덤 시드

    Returns:
        [(x, y, radius), ...]
    """
    rng = np.random.RandomState(seed)
    exclusion_zones = exclusion_zones or []
    obstacles = []
    max_attempts = n * 50

    for _ in range(max_attempts):
        if len(obstacles) >= n:
            break

        x = rng.uniform(x_range[0], x_range[1])
        y = rng.uniform(y_range[0], y_range[1])
        r = rng.uniform(radius_range[0], radius_range[1])

        # 제외 영역 확인
        in_exclusion = False
        for ex, ey, er in exclusion_zones:
            if (x - ex) ** 2 + (y - ey) ** 2 < (er + r + min_spacing) ** 2:
                in_exclusion = True
                break
        if in_exclusion:
            continue

        # 기존 장애물과의 간격 확인
        too_close = False
        for ox, oy, or_ in obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 < (r + or_ + min_spacing) ** 2:
                too_close = True
                break
        if too_close:
            continue

        obstacles.append((x, y, r))

    return obstacles


def generate_corridor(
    path_points: List[Tuple[float, float]],
    width: float = 1.2,
    thickness: float = 0.15,
    spacing: float = 0.2,
) -> List[Tuple[float, float, float]]:
    """
    경로를 따라 복도 벽 생성.

    Args:
        path_points: 복도 중심선 웨이포인트 [(x, y), ...]
        width: 복도 폭
        thickness: 벽 두께 (원형 장애물 반경)
        spacing: 벽 장애물 간격

    Returns:
        [(x, y, radius), ...] 벽을 구성하는 원형 장애물들
    """
    obstacles = []
    half_w = width / 2.0

    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-6:
            continue

        # 법선 벡터
        nx_ = -dy / length
        ny_ = dx / length

        n_circles = max(2, int(length / spacing) + 1)
        for j in range(n_circles):
            t = j / (n_circles - 1)
            cx = x1 + t * dx
            cy = y1 + t * dy

            # 좌측 벽
            obstacles.append((cx + half_w * nx_, cy + half_w * ny_, thickness))
            # 우측 벽
            obstacles.append((cx - half_w * nx_, cy - half_w * ny_, thickness))

    return obstacles


def generate_slalom(
    n_gates: int = 5,
    spacing: float = 2.0,
    gate_width: float = 1.5,
    pillar_radius: float = 0.25,
    start_x: float = 0.0,
    start_y: float = 0.0,
    direction: float = 0.0,
    alternating_offset: float = 1.0,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    """
    슬라럼 게이트 생성.

    Args:
        n_gates: 게이트 수
        spacing: 게이트 간 간격
        gate_width: 게이트 폭
        pillar_radius: 기둥 반경
        start_x, start_y: 시작점
        direction: 진행 방향 (rad)
        alternating_offset: 좌우 교대 오프셋

    Returns:
        (obstacles, gate_centers): 장애물 목록과 게이트 중심점들
    """
    cos_d = np.cos(direction)
    sin_d = np.sin(direction)
    # 법선 (진행 방향의 왼쪽)
    nx_ = -sin_d
    ny_ = cos_d

    obstacles = []
    gate_centers = []

    for i in range(n_gates):
        # 게이트 중심
        offset = alternating_offset * ((-1) ** i)
        cx = start_x + (i + 1) * spacing * cos_d + offset * nx_
        cy = start_y + (i + 1) * spacing * sin_d + offset * ny_

        gate_centers.append((cx, cy))

        # 좌측/우측 기둥
        lx = cx + (gate_width / 2) * nx_
        ly = cy + (gate_width / 2) * ny_
        rx = cx - (gate_width / 2) * nx_
        ry = cy - (gate_width / 2) * ny_

        obstacles.append((lx, ly, pillar_radius))
        obstacles.append((rx, ry, pillar_radius))

    return obstacles, gate_centers


def generate_funnel(
    start_width: float = 3.0,
    end_width: float = 0.8,
    length: float = 4.0,
    start_x: float = 0.0,
    start_y: float = 0.0,
    direction: float = 0.0,
    thickness: float = 0.15,
    spacing: float = 0.2,
) -> List[Tuple[float, float, float]]:
    """
    좁아지는 깔때기 장애물 생성.

    Args:
        start_width: 입구 폭
        end_width: 출구 폭
        length: 길이
        start_x, start_y: 시작점
        direction: 진행 방향 (rad)
        thickness: 벽 두께
        spacing: 벽 장애물 간격

    Returns:
        [(x, y, radius), ...]
    """
    cos_d = np.cos(direction)
    sin_d = np.sin(direction)
    nx_ = -sin_d
    ny_ = cos_d

    n_circles = max(3, int(length / spacing) + 1)
    obstacles = []

    for i in range(n_circles):
        t = i / (n_circles - 1)
        w = start_width + t * (end_width - start_width)
        half_w = w / 2.0

        cx = start_x + t * length * cos_d
        cy = start_y + t * length * sin_d

        # 좌측 벽
        obstacles.append((cx + half_w * nx_, cy + half_w * ny_, thickness))
        # 우측 벽
        obstacles.append((cx - half_w * nx_, cy - half_w * ny_, thickness))

    return obstacles
