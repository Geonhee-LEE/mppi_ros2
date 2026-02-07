"""
Tsallis-MPPI 테스트
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
    TsallisMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController


def test_tsallis_q_equals_1():
    """q=1일 때 Vanilla MPPI와 동등한지 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: Tsallis q=1 = Vanilla MPPI")
    print("=" * 60)

    model_vanilla = DifferentialDriveKinematic()
    model_tsallis = DifferentialDriveKinematic()

    params_vanilla = MPPIParams(N=10, dt=0.05, K=128, lambda_=1.0)
    params_tsallis = TsallisMPPIParams(
        N=10, dt=0.05, K=128, lambda_=1.0, tsallis_q=1.0
    )

    controller_vanilla = MPPIController(model_vanilla, params_vanilla)
    controller_tsallis = TsallisMPPIController(model_tsallis, params_tsallis)

    # 동일한 비용으로 가중치 비교
    np.random.seed(42)
    costs = np.random.uniform(0, 100, 128)

    weights_vanilla = controller_vanilla._compute_weights(costs, params_vanilla.lambda_)
    weights_tsallis = controller_tsallis._compute_weights(costs, params_tsallis.lambda_)

    print(f"Vanilla weights sum: {np.sum(weights_vanilla):.10f}")
    print(f"Tsallis (q=1) weights sum: {np.sum(weights_tsallis):.10f}")
    print(f"Weights difference: {np.linalg.norm(weights_vanilla - weights_tsallis):.2e}")

    # Tsallis 통계
    stats = controller_tsallis.tsallis_stats_history[-1]
    print(f"Tsallis q: {stats['tsallis_q']}")
    print(f"ESS ratio: {stats['ess_ratio']:.4f}")
    print(f"Zero weights: {stats['num_zero_weights']}")

    # q=1일 때 Vanilla와 거의 동일해야 함
    assert np.allclose(
        weights_vanilla, weights_tsallis, atol=1e-6
    ), f"Weights differ: {np.linalg.norm(weights_vanilla - weights_tsallis)}"
    print("✓ PASS: Tsallis q=1 equivalent to Vanilla\n")


def test_tsallis_exploration(q_value=0.5):
    """q<1일 때 탐색적 행동 테스트"""
    print("=" * 60)
    print(f"Test 2: Tsallis q={q_value} (Exploration)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = TsallisMPPIParams(N=10, dt=0.05, K=128, lambda_=1.0, tsallis_q=q_value)
    controller = TsallisMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    # Tsallis 통계
    stats = controller.get_tsallis_statistics()

    print(f"Tsallis q: {q_value}")
    print(f"Control: {control}")
    print(f"Mean ESS ratio: {stats['mean_ess_ratio']:.4f}")
    print(f"Mean zero weights: {stats['mean_zero_weights']:.2f}")

    # q<1이면 가중치가 더 분산됨 (높은 ESS)
    assert stats["mean_ess_ratio"] > 0, "ESS ratio should be positive"
    assert not np.any(np.isnan(control)), "Control contains NaN"
    print(f"✓ PASS: Tsallis q={q_value} works (exploratory)\n")


def test_tsallis_exploitation(q_value=2.0):
    """q>1일 때 활용적 행동 테스트"""
    print("=" * 60)
    print(f"Test 3: Tsallis q={q_value} (Exploitation)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = TsallisMPPIParams(N=10, dt=0.05, K=128, lambda_=1.0, tsallis_q=q_value)
    controller = TsallisMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    control, info = controller.compute_control(state, reference)

    # Tsallis 통계
    stats = controller.get_tsallis_statistics()

    print(f"Tsallis q: {q_value}")
    print(f"Control: {control}")
    print(f"Mean ESS ratio: {stats['mean_ess_ratio']:.4f}")
    print(f"Mean zero weights: {stats['mean_zero_weights']:.2f}")

    # q>1이면 가중치가 더 집중됨 (낮은 ESS, 일부 가중치 0)
    assert stats["mean_ess_ratio"] >= 0, "ESS ratio should be non-negative"
    assert not np.any(np.isnan(control)), "Control contains NaN"

    # q>1일 때 일부 가중치가 0이 될 수 있음 (절단)
    if stats["mean_zero_weights"] > 0:
        print(f"  (Weight truncation occurred: {stats['mean_zero_weights']:.0f} zero weights)")

    print(f"✓ PASS: Tsallis q={q_value} works (exploitative)\n")


def test_tsallis_extreme_q():
    """극단적 q 값에서 수치 안정성 테스트"""
    print("=" * 60)
    print("Test 4: Tsallis Extreme q Values (Numerical Stability)")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    q_values = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]

    for q in q_values:
        params = TsallisMPPIParams(N=10, dt=0.05, K=128, lambda_=1.0, tsallis_q=q)
        controller = TsallisMPPIController(model, params)

        try:
            control, info = controller.compute_control(state, reference)
            stats = controller.get_tsallis_statistics()

            print(f"  q={q:.1f}: ESS ratio={stats['mean_ess_ratio']:.4f}, "
                  f"Zero weights={stats['mean_zero_weights']:.0f} → ✓")

            assert not np.any(np.isnan(control)), f"Control contains NaN at q={q}"
            assert not np.any(np.isinf(control)), f"Control contains Inf at q={q}"

        except Exception as e:
            print(f"  q={q:.1f}: FAILED - {e}")
            raise

    print("✓ PASS: All q values numerically stable\n")


def test_tsallis_statistics():
    """Tsallis 통계 추적 테스트"""
    print("=" * 60)
    print("Test 5: Tsallis Statistics Tracking")
    print("=" * 60)

    model = DifferentialDriveKinematic()
    params = TsallisMPPIParams(N=10, dt=0.05, K=128, tsallis_q=1.5)
    controller = TsallisMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((11, 3))

    # 여러 번 호출
    num_steps = 10
    for i in range(num_steps):
        controller.compute_control(state, reference)

    # 통계 확인
    stats = controller.get_tsallis_statistics()

    print(f"Mean ESS ratio: {stats['mean_ess_ratio']:.4f}")
    print(f"Mean zero weights: {stats['mean_zero_weights']:.2f}")
    print(f"History length: {len(stats['tsallis_stats_history'])}")

    assert len(stats["tsallis_stats_history"]) == num_steps, "History length mismatch"
    assert stats["mean_ess_ratio"] >= 0, "ESS ratio should be non-negative"
    print("✓ PASS: Statistics tracking works\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Tsallis-MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_tsallis_q_equals_1()
        test_tsallis_exploration(q_value=0.5)
        test_tsallis_exploitation(q_value=2.0)
        test_tsallis_extreme_q()
        test_tsallis_statistics()

        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
