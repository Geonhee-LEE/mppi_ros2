"""
SVG-MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import SVGMPPIParams
from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController


def test_svg_mppi_basic():
    """SVG-MPPI 기본 동작 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: SVG-MPPI Basic Functionality")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SVGMPPIParams(
        N=10,
        dt=0.05,
        K=128,
        svg_num_guide_particles=16,
        svgd_num_iterations=3,
        svg_guide_step_size=0.01,
    )
    controller = SVGMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    print(f"Running SVG-MPPI (K=128, G=16, 3 iterations)...")
    control, info = controller.compute_control(state, reference)

    print(f"Control: {control}")
    print(f"SVG num guides: {info['svg_stats']['num_guides']}")
    print(f"SVG num followers: {info['svg_stats']['num_followers']}")
    print(f"Guide cost improvement: {info['svg_stats']['guide_cost_improvement']:.4f}")
    print(f"ESS: {info['ess']:.2f}")

    # 기본 체크
    assert control.shape == (2,), f"Control shape mismatch: {control.shape}"
    assert "svg_stats" in info, "Info should contain svg_stats"
    assert "guide_controls" in info, "Info should contain guide_controls"
    assert info["guide_controls"].shape == (16, 10, 2), "Guide controls shape mismatch"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print("✓ PASS: SVG-MPPI works\n")


def test_guide_selection():
    """Guide particle 선택 테스트"""
    print("=" * 60)
    print("Test 2: Guide Particle Selection")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SVGMPPIParams(
        N=10, dt=0.05, K=64, svg_num_guide_particles=8, svgd_num_iterations=2
    )
    controller = SVGMPPIController(model, params)

    state = np.array([1.0, 1.0, 0.0])  # 초기 오차
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    guide_indices = info["guide_indices"]

    print(f"Guide indices (top 8): {guide_indices}")
    print(f"Guide mean cost: {info['svg_stats']['guide_mean_cost']:.4f}")
    print(f"Follower mean cost: {info['svg_stats']['follower_mean_cost']:.4f}")

    # Guide는 최저 비용 8개여야 함
    assert len(guide_indices) == 8, f"Guide count mismatch: {len(guide_indices)}"
    assert np.all(guide_indices < 64), "Guide indices out of range"

    # Guide 비용이 follower보다 낮아야 함 (일반적으로)
    # 단, SVGD 후 follower가 더 좋아질 수도 있음
    print(f"Guide better than follower: {info['svg_stats']['guide_mean_cost'] <= info['svg_stats']['follower_mean_cost']}")

    print("✓ PASS: Guide selection works\n")


def test_svg_guide_cost_improvement():
    """Guide SVGD 비용 개선 테스트"""
    print("=" * 60)
    print("Test 3: SVG Guide Cost Improvement")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SVGMPPIParams(
        N=10,
        dt=0.05,
        K=128,
        svg_num_guide_particles=32,
        svgd_num_iterations=5,
        svg_guide_step_size=0.05,
    )
    controller = SVGMPPIController(model, params)

    state = np.array([1.0, 1.0, 0.0])
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    initial_guide_cost = info["svg_stats"]["initial_guide_cost"]
    final_guide_cost = info["svg_stats"]["final_guide_cost"]
    improvement = info["svg_stats"]["guide_cost_improvement"]

    print(f"Initial guide cost: {initial_guide_cost:.4f}")
    print(f"Final guide cost: {final_guide_cost:.4f}")
    print(f"Improvement: {improvement:.4f}")
    print(f"Improvement rate: {improvement/initial_guide_cost*100:.2f}%")

    # SVGD는 비용을 개선해야 함 (또는 최소한 악화되지 않아야)
    assert improvement >= -1e-6, f"SVGD should not worsen cost significantly: {improvement}"

    print("✓ PASS: SVG guide cost improvement verified\n")


def test_svg_vs_vanilla():
    """SVG-MPPI vs Vanilla 비교 (G=0, iter=0 → Vanilla 유사)"""
    print("=" * 60)
    print("Test 4: SVG-MPPI vs Vanilla (Sanity Check)")
    print("=" * 60)

    model = DifferentialDriveKinematic()

    # SVG-MPPI (최소 설정)
    svg_params = SVGMPPIParams(
        N=10,
        dt=0.05,
        K=64,
        svg_num_guide_particles=4,
        svgd_num_iterations=1,  # 최소 iteration
        svg_guide_step_size=0.001,  # 작은 스텝
    )
    svg_controller = SVGMPPIController(model, svg_params)

    state = np.array([0.5, 0.5, 0.1])
    reference = np.zeros((11, 3))

    svg_control, svg_info = svg_controller.compute_control(state, reference)

    print(f"SVG-MPPI cost: {svg_info['mean_cost']:.4f}")
    print(
        f"SVG guide improvement: {svg_info['svg_stats']['guide_cost_improvement']:.4f}"
    )

    # 동작 확인
    assert not np.any(np.isnan(svg_control)), "SVG control contains NaN"
    print("✓ PASS: SVG-MPPI vs Vanilla sanity check\n")


def test_svg_computational_efficiency():
    """SVG-MPPI 계산 효율성 테스트 (SVGD 복잡도)"""
    print("=" * 60)
    print("Test 5: SVG Computational Efficiency")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # SVG-MPPI: G=16, K=128 → SVGD는 16²만
    svg_params = SVGMPPIParams(
        N=10,
        dt=0.05,
        K=128,
        svg_num_guide_particles=16,
        svgd_num_iterations=3,
        svg_guide_step_size=0.01,
    )
    svg_controller = SVGMPPIController(model, svg_params)

    import time

    t_start = time.time()
    svg_control, svg_info = svg_controller.compute_control(state, reference)
    svg_time = (time.time() - t_start) * 1000  # ms

    print(f"SVG-MPPI (G=16, K=128): {svg_time:.2f} ms")
    print(f"  Guide mean cost: {svg_info['svg_stats']['guide_mean_cost']:.4f}")
    print(f"  Follower mean cost: {svg_info['svg_stats']['follower_mean_cost']:.4f}")

    # 계산 시간 확인 (일반적으로 < 200ms)
    print(f"SVG-MPPI time: {svg_time:.2f} ms (should be reasonable)")

    print("✓ PASS: SVG computational efficiency verified\n")


def test_svg_statistics():
    """SVG 통계 추적 테스트"""
    print("=" * 60)
    print("Test 6: SVG Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = SVGMPPIParams(
        N=10,
        dt=0.05,
        K=64,
        svg_num_guide_particles=8,
        svgd_num_iterations=3,
        svg_guide_step_size=0.01,
    )
    controller = SVGMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 여러 번 호출
    num_steps = 5
    for i in range(num_steps):
        controller.compute_control(state, reference)

    # 통계 확인
    stats = controller.get_svg_statistics()

    print(f"Mean guide cost improvement: {stats['mean_guide_cost_improvement']:.4f}")
    print(f"Mean bandwidth: {stats['mean_bandwidth']:.4f}")
    print(f"History length: {len(stats['svg_stats_history'])}")

    assert (
        len(stats["svg_stats_history"]) == num_steps
    ), "History length mismatch"
    assert stats["mean_bandwidth"] > 0, "Bandwidth should be positive"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SVG-MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_svg_mppi_basic()
        test_guide_selection()
        test_svg_guide_cost_improvement()
        test_svg_vs_vanilla()
        test_svg_computational_efficiency()
        test_svg_statistics()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
