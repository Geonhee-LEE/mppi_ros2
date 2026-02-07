"""
Risk-Aware MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    RiskAwareMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController


def test_cvar_alpha_1():
    """α=1.0일 때 Vanilla MPPI와 동등한지 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: CVaR α=1.0 = Vanilla MPPI")
    print("=" * 60)

    model_vanilla = DifferentialDriveKinematic()
    model_risk = DifferentialDriveKinematic()

    params_vanilla = MPPIParams(N=10, dt=0.05, K=128, lambda_=1.0)
    params_risk = RiskAwareMPPIParams(
        N=10, dt=0.05, K=128, lambda_=1.0, cvar_alpha=1.0
    )

    controller_vanilla = MPPIController(model_vanilla, params_vanilla)
    controller_risk = RiskAwareMPPIController(model_risk, params_risk)

    # 동일한 비용으로 가중치 비교
    np.random.seed(42)
    costs = np.random.uniform(0, 100, 128)

    weights_vanilla = controller_vanilla._compute_weights(costs, params_vanilla.lambda_)
    weights_risk = controller_risk._compute_weights(costs, params_risk.lambda_)

    print(f"Vanilla weights sum: {np.sum(weights_vanilla):.10f}")
    print(f"Risk-Aware (α=1.0) weights sum: {np.sum(weights_risk):.10f}")
    print(f"Weights difference: {np.linalg.norm(weights_vanilla - weights_risk):.2e}")

    # Risk 통계
    stats = controller_risk.risk_stats_history[-1]
    print(f"CVaR α: {stats['cvar_alpha']}")
    print(f"CVaR samples: {stats['num_cvar_samples']} / {len(costs)}")
    print(f"Zero weights: {stats['num_zero_weights']}")

    # α=1.0일 때 Vanilla와 동일해야 함
    assert np.allclose(
        weights_vanilla, weights_risk, atol=1e-6
    ), f"Weights differ: {np.linalg.norm(weights_vanilla - weights_risk)}"
    assert stats["num_zero_weights"] == 0, "Should have no zero weights at α=1.0"
    print("✓ PASS: CVaR α=1.0 equivalent to Vanilla\n")


def test_risk_averse_behavior(alpha=0.5):
    """α<1일 때 위험 회피 행동 테스트"""
    print("=" * 60)
    print(f"Test 2: CVaR α={alpha} (Risk-Averse)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = RiskAwareMPPIParams(N=10, dt=0.05, K=128, lambda_=1.0, cvar_alpha=alpha)
    controller = RiskAwareMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    # Risk 통계
    stats = controller.get_risk_statistics()

    print(f"CVaR α: {alpha}")
    print(f"Control: {control}")
    print(f"Mean VaR cost: {stats['mean_var_cost']:.4f}")
    print(f"Mean CVaR cost: {stats['mean_cvar_cost']:.4f}")
    print(f"Mean ESS ratio: {stats['mean_ess_ratio']:.4f}")
    print(f"Mean zero weights: {stats['mean_zero_weights']:.2f}")

    # α<1이면 일부 샘플이 가중치 0
    expected_zero_weights = int(128 * (1 - alpha))
    actual_zero_weights = int(stats["mean_zero_weights"])

    print(
        f"Expected ~{expected_zero_weights} zero weights, got {actual_zero_weights}"
    )

    assert stats["mean_zero_weights"] > 0, "Should have some zero weights at α<1"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print(f"✓ PASS: CVaR α={alpha} works (risk-averse)\n")


def test_extreme_risk_aversion(alpha=0.1):
    """극단적 위험 회피 (α=0.1) 테스트"""
    print("=" * 60)
    print(f"Test 3: CVaR α={alpha} (Extreme Risk-Averse)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = RiskAwareMPPIParams(N=10, dt=0.05, K=128, lambda_=1.0, cvar_alpha=alpha)
    controller = RiskAwareMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    # Risk 통계
    stats = controller.get_risk_statistics()

    print(f"CVaR α: {alpha}")
    print(f"Control: {control}")
    print(f"Mean zero weights: {stats['mean_zero_weights']:.2f}")

    # α=0.1이면 90% 샘플이 가중치 0
    expected_zero_weights = int(128 * (1 - alpha))
    actual_zero_weights = int(stats["mean_zero_weights"])

    print(
        f"Expected ~{expected_zero_weights} zero weights, got {actual_zero_weights}"
    )

    assert stats["mean_zero_weights"] > 100, "Should have many zero weights at α=0.1"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print(f"✓ PASS: CVaR α={alpha} works (extreme risk-averse)\n")


def test_var_cvar_relationship():
    """VaR와 CVaR 관계 테스트"""
    print("=" * 60)
    print("Test 4: VaR and CVaR Relationship")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = RiskAwareMPPIParams(N=10, dt=0.05, K=128, cvar_alpha=0.9)
    controller = RiskAwareMPPIController(model, params)

    # 단순 비용 배열
    costs = np.arange(128, dtype=float)  # 0, 1, 2, ..., 127

    weights = controller._compute_weights(costs, lambda_=1.0)
    stats = controller.risk_stats_history[-1]

    var_cost = stats["var_cost"]
    cvar_cost = stats["cvar_cost"]

    print(f"VaR (α=0.9): {var_cost:.2f}")
    print(f"CVaR (α=0.9): {cvar_cost:.2f}")

    # CVaR는 항상 VaR 이하 (CVaR set의 평균이므로)
    assert cvar_cost <= var_cost, "CVaR should be <= VaR"

    # α=0.9이면 상위 90% (0~114)의 평균
    expected_cvar = np.mean(costs[: int(128 * 0.9)])
    print(f"Expected CVaR: {expected_cvar:.2f}")
    print(f"Actual CVaR: {cvar_cost:.2f}")

    # 거의 같아야 함
    assert np.isclose(
        cvar_cost, expected_cvar, atol=1.0
    ), f"CVaR mismatch: {cvar_cost} vs {expected_cvar}"
    print("✓ PASS: VaR and CVaR relationship correct\n")


def test_numerical_stability():
    """다양한 α 값에서 수치 안정성 테스트"""
    print("=" * 60)
    print("Test 5: Numerical Stability Across α Values")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    for alpha in alpha_values:
        params = RiskAwareMPPIParams(N=10, dt=0.05, K=128, lambda_=1.0, cvar_alpha=alpha)
        controller = RiskAwareMPPIController(model, params)

        try:
            control, info = controller.compute_control(state, reference)
            stats = controller.get_risk_statistics()

            print(
                f"  α={alpha:.1f}: ESS ratio={stats['mean_ess_ratio']:.4f}, "
                f"Zero weights={stats['mean_zero_weights']:.0f} → ✓"
            )

            assert not np.any(np.isnan(control)), f"Control contains NaN at α={alpha}"
            assert not np.any(np.isinf(control)), f"Control contains Inf at α={alpha}"

        except Exception as e:
            print(f"  α={alpha:.1f}: FAILED - {e}")
            raise

    print("✓ PASS: All α values numerically stable\n")


def test_risk_statistics():
    """Risk 통계 추적 테스트"""
    print("=" * 60)
    print("Test 6: Risk Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = RiskAwareMPPIParams(N=10, dt=0.05, K=128, cvar_alpha=0.8)
    controller = RiskAwareMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 여러 번 호출
    num_steps = 10
    for i in range(num_steps):
        controller.compute_control(state, reference)

    # 통계 확인
    stats = controller.get_risk_statistics()

    print(f"Mean VaR cost: {stats['mean_var_cost']:.4f}")
    print(f"Mean CVaR cost: {stats['mean_cvar_cost']:.4f}")
    print(f"Mean ESS ratio: {stats['mean_ess_ratio']:.4f}")
    print(f"History length: {len(stats['risk_stats_history'])}")

    assert len(stats["risk_stats_history"]) == num_steps, "History length mismatch"
    assert stats["mean_var_cost"] >= 0, "VaR cost should be non-negative"
    assert stats["mean_cvar_cost"] >= 0, "CVaR cost should be non-negative"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Risk-Aware MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_cvar_alpha_1()
        test_risk_averse_behavior(alpha=0.5)
        test_extreme_risk_aversion(alpha=0.1)
        test_var_cvar_relationship()
        test_numerical_stability()
        test_risk_statistics()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
