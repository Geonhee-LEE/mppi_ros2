"""
Log-MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, LogMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.log_mppi import LogMPPIController


def test_log_mppi_numerical_stability():
    """수치 안정성 테스트 (극단적 비용 차이)"""
    print("\n" + "=" * 60)
    print("Test 1: Numerical Stability (Extreme Cost Differences)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = LogMPPIParams(N=10, dt=0.05, K=128, lambda_=0.01)  # 작은 λ로 극단 상황
    controller = LogMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.tile(state, (11, 1))

    # 제어 계산 (극단적 비용 차이 발생 가능)
    control, info = controller.compute_control(state, reference)

    print(f"Lambda: {params.lambda_}")
    print(f"Control: {control}")
    print(f"Weights sum: {np.sum(info['sample_weights']):.10f}")
    print(f"Min weight: {np.min(info['sample_weights']):.2e}")
    print(f"Max weight: {np.max(info['sample_weights']):.2e}")
    print(f"ESS: {info['ess']:.2f} / {params.K}")

    # 가중치 합이 1인지 확인 (수치 안정성)
    assert np.isclose(
        np.sum(info["sample_weights"]), 1.0, atol=1e-10
    ), f"Weights sum: {np.sum(info['sample_weights'])}"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    assert not np.any(np.isinf(control)), "Control contains Inf"
    print("✓ PASS: Numerically stable\n")


def test_log_mppi_vs_vanilla():
    """Log-MPPI vs Vanilla MPPI 가중치 동등성 테스트"""
    print("=" * 60)
    print("Test 2: Log-MPPI vs Vanilla MPPI (Weight Equivalence)")
    print("=" * 60)

    model_vanilla = DifferentialDriveKinematic()
    model_log = DifferentialDriveKinematic()

    params_vanilla = MPPIParams(N=10, dt=0.05, K=128, lambda_=1.0)
    params_log = LogMPPIParams(N=10, dt=0.05, K=128, lambda_=1.0, use_baseline=False)

    controller_vanilla = MPPIController(model_vanilla, params_vanilla)
    controller_log = LogMPPIController(model_log, params_log)

    # 동일한 비용 배열 생성
    np.random.seed(42)
    costs = np.random.uniform(0, 100, 128)

    # 가중치 계산
    weights_vanilla = controller_vanilla._compute_weights(costs, params_vanilla.lambda_)
    weights_log = controller_log._compute_weights(costs, params_log.lambda_)

    print(f"Vanilla weights sum: {np.sum(weights_vanilla):.10f}")
    print(f"Log-MPPI weights sum: {np.sum(weights_log):.10f}")
    print(f"Weights difference: {np.linalg.norm(weights_vanilla - weights_log):.2e}")
    print(f"Max weight diff: {np.max(np.abs(weights_vanilla - weights_log)):.2e}")

    # 가중치가 동일해야 함 (수치 오차 범위)
    assert np.allclose(
        weights_vanilla, weights_log, atol=1e-10
    ), f"Weights differ: {np.linalg.norm(weights_vanilla - weights_log)}"
    print("✓ PASS: Log-MPPI and Vanilla MPPI produce identical weights\n")


def test_baseline_effect():
    """Baseline 적용 효과 테스트"""
    print("=" * 60)
    print("Test 3: Baseline Effect")
    print("=" * 60)

    model = DifferentialDriveKinematic()

    # Baseline 없음
    params_no_baseline = LogMPPIParams(N=10, dt=0.05, K=128, use_baseline=False)
    controller_no_baseline = LogMPPIController(model, params_no_baseline)

    # Baseline 있음
    params_with_baseline = LogMPPIParams(N=10, dt=0.05, K=128, use_baseline=True)
    controller_with_baseline = LogMPPIController(model, params_with_baseline)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 동일한 랜덤 시드로 비교
    np.random.seed(42)
    control_no_baseline, info_no_baseline = controller_no_baseline.compute_control(
        state, reference
    )

    np.random.seed(42)
    control_with_baseline, info_with_baseline = controller_with_baseline.compute_control(
        state, reference
    )

    # Log 통계 확인
    stats_no_baseline = controller_no_baseline.get_log_weight_statistics()
    stats_with_baseline = controller_with_baseline.get_log_weight_statistics()

    print(f"No baseline - Mean log Z: {stats_no_baseline['mean_log_Z']:.4f}")
    print(f"With baseline - Mean log Z: {stats_with_baseline['mean_log_Z']:.4f}")
    print(f"Control difference: {np.linalg.norm(control_no_baseline - control_with_baseline):.2e}")

    # Baseline은 가중치 분포를 안정화시키지만 최종 제어는 거의 동일해야 함
    assert not np.any(np.isnan(control_no_baseline)), "Control contains NaN"
    assert not np.any(np.isnan(control_with_baseline)), "Control contains NaN"
    print("✓ PASS: Baseline works\n")


def test_log_weight_statistics():
    """Log 가중치 통계 추적 테스트"""
    print("=" * 60)
    print("Test 4: Log Weight Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = LogMPPIParams(N=10, dt=0.05, K=128)
    controller = LogMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 여러 번 호출
    num_steps = 10
    for i in range(num_steps):
        controller.compute_control(state, reference)

    # 통계 확인
    stats = controller.get_log_weight_statistics()

    print(f"Mean log Z: {stats['mean_log_Z']:.4f}")
    print(f"Mean ESS ratio: {stats['mean_ess_ratio']:.4f}")
    print(f"History length: {len(stats['log_weight_stats_history'])}")

    assert len(stats["log_weight_stats_history"]) == num_steps, "History length mismatch"
    assert stats["mean_ess_ratio"] > 0, "ESS ratio should be positive"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Log-MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_log_mppi_numerical_stability()
        test_log_mppi_vs_vanilla()
        test_baseline_effect()
        test_log_weight_statistics()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
