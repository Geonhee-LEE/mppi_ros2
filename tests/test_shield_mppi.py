"""
Shield-MPPI 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import ShieldMPPIParams
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController


def test_shield_batch_no_constraint():
    """장애물이 멀리 있을 때 CBF shield가 제어를 수정하지 않는지 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: Shield Batch - No Constraint When Far")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(100.0, 100.0, 0.5)]  # 매우 먼 장애물

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=64, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    controller = ShieldMPPIController(model, params)

    # 원점 근처 상태, 전진 제어
    K = 64
    states = np.zeros((K, 3))
    controls = np.column_stack([
        np.full(K, 0.5),  # v = 0.5
        np.full(K, 0.3),  # omega = 0.3
    ])

    safe_controls, intervened, vel_reduction = controller._cbf_shield_batch(
        states, controls
    )

    print(f"Interventions: {np.sum(intervened)} / {K}")
    print(f"Max vel reduction: {np.max(vel_reduction):.6f}")

    assert np.sum(intervened) == 0, "Should not intervene when obstacle is far"
    assert np.allclose(safe_controls, controls), "Controls should be unchanged"
    print("PASS: No constraint when far from obstacles\n")


def test_shield_batch_clips_velocity():
    """장애물에 접근할 때 속도를 제한하는지 테스트"""
    print("=" * 60)
    print("Test 2: Shield Batch - Clips Velocity When Approaching")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    # 장애물이 전방 0.8m에 있음 (r=0.3, margin=0.1 → effective_r=0.4)
    obstacles = [(0.8, 0.0, 0.3)]

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=32, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    controller = ShieldMPPIController(model, params)

    K = 32
    states = np.zeros((K, 3))  # 원점, theta=0 (전방)
    controls = np.column_stack([
        np.full(K, 1.0),  # v = 1.0 (최대 속도로 전진)
        np.zeros(K),       # omega = 0
    ])

    safe_controls, intervened, vel_reduction = controller._cbf_shield_batch(
        states, controls
    )

    print(f"Original v: {controls[0, 0]:.3f}")
    print(f"Safe v: {safe_controls[0, 0]:.3f}")
    print(f"Interventions: {np.sum(intervened)} / {K}")
    print(f"Mean vel reduction: {np.mean(vel_reduction):.4f}")

    # h = 0.8^2 - 0.4^2 = 0.64 - 0.16 = 0.48
    # Lg_h_v = 2*0.8*1 + 2*0*0 = 1.6 > 0 → 이탈 방향
    # 사실 원점에서 전방 장애물로 접근 시 Lg_h_v > 0 (위치 벡터와 속도 방향이 같음)
    # → 이 경우 접근이 아니라 "이미 떨어지고 있는" 상태
    # Lg_h_v = 2*(x-xo)*cos(θ) = 2*(-0.8)*1 = -1.6 < 0 → 접근!
    # 수정: dx = 0 - 0.8 = -0.8
    # Lg_h_v = 2*(-0.8)*cos(0) + 2*(0)*sin(0) = -1.6 → 접근
    # v_ceiling = alpha * h / |Lg_h_v| = 0.3 * 0.48 / 1.6 = 0.09

    assert np.all(intervened), "Should intervene for all samples"
    assert np.all(safe_controls[:, 0] < controls[:, 0]), \
        "Safe velocity should be less than original"
    assert np.all(safe_controls[:, 0] >= 0), "Safe velocity should be non-negative"
    print("PASS: Velocity clipped when approaching obstacle\n")


def test_shield_batch_omega_unconstrained():
    """ω가 항상 자유(미수정)인지 테스트"""
    print("=" * 60)
    print("Test 3: Shield Batch - Omega Unconstrained")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(0.8, 0.0, 0.3)]

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=32, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    controller = ShieldMPPIController(model, params)

    K = 32
    states = np.zeros((K, 3))
    omega_values = np.random.uniform(-1.0, 1.0, K)
    controls = np.column_stack([
        np.full(K, 1.0),
        omega_values,
    ])

    safe_controls, _, _ = controller._cbf_shield_batch(states, controls)

    print(f"Original omega: {omega_values[:5]}")
    print(f"Safe omega: {safe_controls[:5, 1]}")

    assert np.allclose(safe_controls[:, 1], omega_values), \
        "Omega should not be modified by CBF shield"
    print("PASS: Omega unconstrained\n")


def test_shielded_rollout_all_safe():
    """핵심 테스트: K=512 전체 궤적이 h>0 (안전)인지 검증"""
    print("=" * 60)
    print("Test 4: Shielded Rollout - All Trajectories Safe")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [
        (2.0, 0.0, 0.5),
        (0.0, 2.0, 0.5),
        (-1.5, 1.5, 0.4),
    ]

    params = ShieldMPPIParams(
        N=20, dt=0.05, K=512, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    controller = ShieldMPPIController(model, params)

    # 랜덤 제어 생성
    np.random.seed(42)
    K, N = 512, 20
    controls = np.random.uniform(
        [-1.0, -1.0], [1.0, 1.0], size=(K, N, 2)
    )

    initial_state = np.array([0.0, 0.0, 0.0])
    trajectories, shielded_controls, info = controller._shielded_rollout(
        initial_state, controls
    )

    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Intervention rate: {info['intervention_rate']:.2%}")
    print(f"Total interventions: {info['total_interventions']}")

    # 모든 궤적의 모든 시간스텝에서 장애물과의 거리 확인
    positions = trajectories[:, :, :2]  # (K, N+1, 2)
    all_safe = True
    min_h_global = np.inf

    for obs_x, obs_y, obs_r in obstacles:
        effective_r = obs_r + params.cbf_safety_margin
        dx = positions[:, :, 0] - obs_x
        dy = positions[:, :, 1] - obs_y
        h = dx**2 + dy**2 - effective_r**2  # (K, N+1)
        min_h = np.min(h)
        min_h_global = min(min_h_global, min_h)
        if min_h < -1e-6:  # 수치 오차 허용
            all_safe = False

    print(f"Min barrier value across all trajectories: {min_h_global:.6f}")
    assert all_safe, \
        f"All trajectories should be safe, but min h = {min_h_global:.6f}"
    print("PASS: All K=512 trajectories are safe!\n")


def test_shield_vs_cbf_safety():
    """Shield-MPPI가 CBF-MPPI보다 높은 안전율을 보이는지 테스트"""
    print("=" * 60)
    print("Test 5: Shield vs CBF Safety Rate")
    print("=" * 60)

    from mppi_controller.controllers.mppi.mppi_params import CBFMPPIParams
    from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(2.0, 0.0, 0.5)]

    common_kwargs = dict(
        N=15, dt=0.05, K=256, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        cbf_use_safety_filter=False,
    )

    cbf_params = CBFMPPIParams(**common_kwargs)
    shield_params = ShieldMPPIParams(**common_kwargs, shield_enabled=True)

    cbf_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    shield_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    cbf_controller = CBFMPPIController(cbf_model, cbf_params)
    shield_controller = ShieldMPPIController(shield_model, shield_params)

    np.random.seed(42)
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 3, 16)

    cbf_state = state.copy()
    shield_state = state.copy()

    for i in range(15):
        control_cbf, _ = cbf_controller.compute_control(cbf_state, reference)
        cbf_state = cbf_model.step(cbf_state, control_cbf, 0.05)

        np.random.seed(42 + i)
        control_shield, _ = shield_controller.compute_control(shield_state, reference)
        shield_state = shield_model.step(shield_state, control_shield, 0.05)

    cbf_stats = cbf_controller.get_cbf_statistics()
    shield_stats = shield_controller.get_cbf_statistics()

    print(f"CBF-MPPI safety rate:   {cbf_stats['safety_rate']:.2%}")
    print(f"Shield-MPPI safety rate: {shield_stats['safety_rate']:.2%}")

    # Shield-MPPI는 shielded rollout 덕분에 일반적으로 더 높은 안전율
    assert shield_stats["safety_rate"] >= cbf_stats["safety_rate"] - 0.1, \
        "Shield-MPPI should have comparable or better safety rate"
    print("PASS: Shield-MPPI has good safety rate\n")


def test_shield_mppi_integration():
    """ShieldMPPIController 전체 통합 테스트"""
    print("=" * 60)
    print("Test 6: Shield-MPPI Integration")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(3.0, 0.0, 0.5), (0.0, 3.0, 0.5)]

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=256, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        shield_enabled=True,
    )

    controller = ShieldMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 2, 16)

    for i in range(5):
        control, info = controller.compute_control(state, reference)
        state = model.step(state, control, params.dt)

    print(f"Control: {control}")
    print(f"Final state: {state}")
    print(f"Min barrier: {info['min_barrier']:.4f}")
    print(f"Shield intervention rate: {info['shield_intervention_rate']:.2%}")

    assert not np.any(np.isnan(control)), "Control contains NaN"
    assert not np.any(np.isinf(control)), "Control contains Inf"
    assert "shield_intervention_rate" in info
    assert "shield_mean_vel_reduction" in info
    assert "shielded_controls" in info
    print("PASS: Shield-MPPI integration works\n")


def test_shield_disabled_fallback():
    """shield_enabled=False일 때 CBF-MPPI로 폴백하는지 테스트"""
    print("=" * 60)
    print("Test 7: Shield Disabled Fallback")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(3.0, 0.0, 0.5)]

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        shield_enabled=False,  # Shield 비활성화
    )

    controller = ShieldMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 2, 16)

    control, info = controller.compute_control(state, reference)

    print(f"Control: {control}")
    print(f"Has shield info: {'shield_intervention_rate' in info}")

    # shield 비활성화 시 CBF-MPPI 방식 (shield 관련 info 없음)
    assert not np.any(np.isnan(control)), "Control contains NaN"
    assert "min_barrier" in info, "Should still have CBF info"

    # set_shield_enabled으로 런타임 전환 테스트
    controller.set_shield_enabled(True)
    control2, info2 = controller.compute_control(state, reference)
    assert "shield_intervention_rate" in info2, \
        "Should have shield info after enabling"

    print("PASS: Fallback to CBF-MPPI when shield disabled\n")


def test_shield_multiple_obstacles():
    """여러 장애물 중 가장 보수적 제약이 적용되는지 테스트"""
    print("=" * 60)
    print("Test 8: Shield Multiple Obstacles - Most Conservative")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # 3개 장애물: 거리 순 (가까운 것 → 먼 것)
    obstacles_near = [(0.8, 0.0, 0.2)]   # 가장 가까운 장애물 하나
    obstacles_all = [
        (0.8, 0.0, 0.2),   # 가장 가까운
        (1.5, 0.0, 0.3),   # 중간
        (3.0, 0.0, 0.5),   # 먼
    ]

    params_near = ShieldMPPIParams(
        N=15, dt=0.05, K=32, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles_near,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    params_all = ShieldMPPIParams(
        N=15, dt=0.05, K=32, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles_all,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )

    controller_near = ShieldMPPIController(model, params_near)
    controller_all = ShieldMPPIController(model, params_all)

    K = 32
    states = np.zeros((K, 3))
    controls = np.column_stack([
        np.full(K, 1.0),
        np.zeros(K),
    ])

    safe_near, _, _ = controller_near._cbf_shield_batch(states, controls)
    safe_all, _, _ = controller_all._cbf_shield_batch(states, controls)

    print(f"v with nearest only: {safe_near[0, 0]:.4f}")
    print(f"v with all obstacles: {safe_all[0, 0]:.4f}")

    # 여러 장애물이 있으면 더 보수적이거나 동일해야 함
    assert np.all(safe_all[:, 0] <= safe_near[:, 0] + 1e-10), \
        "Multiple obstacles should be more conservative"
    print("PASS: Most conservative constraint applied\n")


def test_shield_statistics():
    """Shield 통계 수집 및 반환 테스트"""
    print("=" * 60)
    print("Test 9: Shield Statistics")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(2.0, 0.0, 0.5)]

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=256, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        shield_enabled=True,
    )

    controller = ShieldMPPIController(model, params)

    # 초기 통계 (비어있음)
    stats = controller.get_shield_statistics()
    assert stats["num_steps"] == 0, "Initial stats should be empty"
    assert stats["mean_intervention_rate"] == 0.0

    # 여러 스텝 실행
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 3, 16)

    for i in range(5):
        control, info = controller.compute_control(state, reference)
        state = model.step(state, control, params.dt)

    stats = controller.get_shield_statistics()
    print(f"Shield Statistics:")
    print(f"  Mean intervention rate: {stats['mean_intervention_rate']:.2%}")
    print(f"  Max intervention rate: {stats['max_intervention_rate']:.2%}")
    print(f"  Mean vel reduction: {stats['mean_vel_reduction']:.4f}")
    print(f"  Total interventions: {stats['total_interventions']}")
    print(f"  Num steps: {stats['num_steps']}")

    assert stats["num_steps"] == 5, f"Expected 5 steps, got {stats['num_steps']}"
    assert "mean_intervention_rate" in stats
    assert "max_intervention_rate" in stats
    assert "mean_vel_reduction" in stats
    assert "total_interventions" in stats

    # Reset 테스트
    controller.reset()
    stats_after = controller.get_shield_statistics()
    assert stats_after["num_steps"] == 0, "Stats should be empty after reset"
    print("PASS: Shield statistics work correctly\n")


def test_shielded_noise_weight_update():
    """shielded noise로 가중 업데이트가 올바르게 수행되는지 테스트"""
    print("=" * 60)
    print("Test 10: Shielded Noise Weight Update")
    print("=" * 60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    obstacles = [(2.0, 0.0, 0.5)]

    params = ShieldMPPIParams(
        N=15, dt=0.05, K=256, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        shield_enabled=True,
    )

    controller = ShieldMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((16, 3))
    reference[:, 0] = np.linspace(0, 3, 16)

    control, info = controller.compute_control(state, reference)

    # shielded_controls가 info에 있는지 확인
    assert "shielded_controls" in info, "Missing shielded_controls in info"
    shielded = info["shielded_controls"]
    assert shielded.shape == (256, 15, 2), \
        f"Expected shape (256, 15, 2), got {shielded.shape}"

    # 제어가 유효한 범위 내인지
    assert not np.any(np.isnan(control)), "Control contains NaN"
    assert not np.any(np.isinf(control)), "Control contains Inf"

    # 가중치가 올바르게 정규화되었는지
    weights = info["sample_weights"]
    assert abs(np.sum(weights) - 1.0) < 1e-6, \
        f"Weights should sum to 1, got {np.sum(weights)}"

    print(f"Shielded controls shape: {shielded.shape}")
    print(f"Weights sum: {np.sum(weights):.6f}")
    print(f"ESS: {info['ess']:.1f}")
    print("PASS: Shielded noise weight update correct\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Shield-MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_shield_batch_no_constraint()
        test_shield_batch_clips_velocity()
        test_shield_batch_omega_unconstrained()
        test_shielded_rollout_all_safe()
        test_shield_vs_cbf_safety()
        test_shield_mppi_integration()
        test_shield_disabled_fallback()
        test_shield_multiple_obstacles()
        test_shield_statistics()
        test_shielded_noise_weight_update()

        print("=" * 60)
        print("All 10 Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
