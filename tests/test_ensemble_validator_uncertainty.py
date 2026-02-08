"""
P2 테스트: EnsembleNeuralDynamics, ModelValidator, UncertaintyAwareCost

Tests:
    - EnsembleTrainer: 학습, 예측, save/load, 불확실성
    - EnsembleNeuralDynamics: forward_dynamics, predict_with_uncertainty
    - ModelValidator: evaluate, compare, per-dim
    - UncertaintyAwareCost: compute_cost
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from mppi_controller.learning.ensemble_trainer import EnsembleTrainer
from mppi_controller.models.learned.ensemble_dynamics import EnsembleNeuralDynamics
from mppi_controller.learning.model_validator import ModelValidator
from mppi_controller.controllers.mppi.uncertainty_cost import UncertaintyAwareCost


def _make_data(state_dim=3, control_dim=2, N=200):
    """합성 선형 데이터"""
    np.random.seed(42)
    input_dim = state_dim + control_dim
    inputs = np.random.randn(N, input_dim).astype(np.float32)
    A = np.random.randn(state_dim, input_dim).astype(np.float32) * 0.1
    targets = inputs @ A.T + np.random.randn(N, state_dim).astype(np.float32) * 0.01

    num_train = int(N * 0.8)
    norm_stats = {
        "state_mean": np.zeros(state_dim),
        "state_std": np.ones(state_dim),
        "control_mean": np.zeros(control_dim),
        "control_std": np.ones(control_dim),
        "state_dot_mean": np.zeros(state_dim),
        "state_dot_std": np.ones(state_dim),
    }
    return inputs, targets, num_train, norm_stats


# ========== EnsembleTrainer Tests ==========


def test_ensemble_trainer_train():
    """앙상블 학습 기본"""
    print("\n" + "=" * 60)
    print("Test: EnsembleTrainer train")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = EnsembleTrainer(
        state_dim=3, control_dim=2,
        num_models=3, hidden_dims=[16, 16],
        save_dir=save_dir,
    )
    inputs, targets, num_train, norm_stats = _make_data()

    history = trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=20, verbose=False,
    )

    assert len(trainer.models) == 3
    assert len(history["train_loss"]) > 0
    assert len(history["val_loss"]) > 0
    assert history["train_loss"][-1] < history["train_loss"][0]
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print("PASS")


def test_ensemble_trainer_predict():
    """앙상블 예측 형상 + 불확실성"""
    print("\n" + "=" * 60)
    print("Test: EnsembleTrainer predict with uncertainty")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = EnsembleTrainer(
        state_dim=3, control_dim=2,
        num_models=3, hidden_dims=[16],
        save_dir=save_dir,
    )
    inputs, targets, num_train, norm_stats = _make_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=10, verbose=False,
    )

    # Single
    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    mean, std = trainer.predict(state, control, return_uncertainty=True)
    assert mean.shape == (3,), f"mean: {mean.shape}"
    assert std.shape == (3,), f"std: {std.shape}"
    assert np.all(std >= 0)

    # Batch
    states = np.random.randn(8, 3).astype(np.float32)
    controls = np.random.randn(8, 2).astype(np.float32)
    mean, std = trainer.predict(states, controls, return_uncertainty=True)
    assert mean.shape == (8, 3)
    assert std.shape == (8, 3)
    print("PASS")


def test_ensemble_trainer_save_load():
    """앙상블 save/load"""
    print("\n" + "=" * 60)
    print("Test: EnsembleTrainer save/load")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = EnsembleTrainer(
        state_dim=3, control_dim=2,
        num_models=3, hidden_dims=[16],
        save_dir=save_dir,
    )
    inputs, targets, num_train, norm_stats = _make_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=5, verbose=False,
    )

    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    pred_before, _ = trainer.predict(state, control)

    trainer.save_model("ensemble.pth")

    trainer2 = EnsembleTrainer(
        state_dim=3, control_dim=2,
        num_models=3, hidden_dims=[16],
        save_dir=save_dir,
    )
    trainer2.load_model("ensemble.pth")

    pred_after, _ = trainer2.predict(state, control)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)
    print("PASS")


# ========== EnsembleNeuralDynamics Tests ==========


def test_ensemble_dynamics_no_model():
    """모델 미로드"""
    print("\n" + "=" * 60)
    print("Test: EnsembleNeuralDynamics no model")
    print("=" * 60)

    model = EnsembleNeuralDynamics(state_dim=3, control_dim=2)
    assert model.models is None
    assert model.model_type == "learned"

    info = model.get_model_info()
    assert info["loaded"] is False

    try:
        model.forward_dynamics(np.zeros(3), np.zeros(2))
        assert False, "Should have raised"
    except RuntimeError:
        pass
    print("PASS")


def test_ensemble_dynamics_loaded():
    """학습된 앙상블 모델 로드 + 추론"""
    print("\n" + "=" * 60)
    print("Test: EnsembleNeuralDynamics loaded")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = EnsembleTrainer(
        state_dim=3, control_dim=2,
        num_models=3, hidden_dims=[16],
        save_dir=save_dir,
    )
    inputs, targets, num_train, norm_stats = _make_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=5, verbose=False,
    )
    trainer.save_model("ens.pth")

    model = EnsembleNeuralDynamics(
        state_dim=3, control_dim=2,
        model_path=os.path.join(save_dir, "ens.pth"),
    )

    assert model.num_models == 3
    info = model.get_model_info()
    assert info["loaded"]
    assert info["num_models"] == 3

    # forward
    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    result = model.forward_dynamics(state, control)
    assert result.shape == (3,)

    # uncertainty
    mean, std = model.predict_with_uncertainty(state, control)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert np.all(std >= 0)

    # repr
    r = repr(model)
    assert "M=3" in r
    print(f"repr: {r}")
    print("PASS")


# ========== ModelValidator Tests ==========


def test_validator_evaluate():
    """ModelValidator evaluate"""
    print("\n" + "=" * 60)
    print("Test: ModelValidator evaluate")
    print("=" * 60)

    np.random.seed(42)
    N, nx, nu = 100, 3, 2
    test_states = np.random.randn(N, nx)
    test_controls = np.random.randn(N, nu)
    # "Perfect" model
    test_targets = test_states * 0.1

    def perfect_fn(states, controls):
        return states * 0.1

    validator = ModelValidator()
    metrics = validator.evaluate(perfect_fn, test_states, test_controls, test_targets)

    assert metrics["rmse"] < 1e-10
    assert metrics["mae"] < 1e-10
    assert metrics["r2"] > 0.999
    assert metrics["max_error"] < 1e-10
    assert metrics["per_dim_rmse"].shape == (nx,)
    print(f"RMSE: {metrics['rmse']:.2e}, R²: {metrics['r2']:.4f}")
    print("PASS")


def test_validator_evaluate_noisy():
    """ModelValidator evaluate with noisy prediction"""
    print("\n" + "=" * 60)
    print("Test: ModelValidator evaluate noisy")
    print("=" * 60)

    np.random.seed(42)
    N, nx, nu = 200, 3, 2
    test_states = np.random.randn(N, nx)
    test_controls = np.random.randn(N, nu)
    test_targets = test_states * 0.1

    def noisy_fn(states, controls):
        return states * 0.1 + np.random.randn(*states.shape) * 0.01

    validator = ModelValidator()
    metrics = validator.evaluate(noisy_fn, test_states, test_controls, test_targets)

    assert metrics["rmse"] > 0
    assert metrics["rmse"] < 0.05  # small noise
    assert 0 < metrics["r2"] < 1.0
    print(f"RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.4f}")
    print("PASS")


def test_validator_compare():
    """ModelValidator compare"""
    print("\n" + "=" * 60)
    print("Test: ModelValidator compare")
    print("=" * 60)

    np.random.seed(42)
    N, nx, nu = 100, 3, 2
    test_states = np.random.randn(N, nx)
    test_controls = np.random.randn(N, nu)
    test_targets = test_states * 0.1

    def good_fn(s, c):
        return s * 0.1 + np.random.randn(*s.shape) * 0.001

    def bad_fn(s, c):
        return np.zeros_like(s)

    validator = ModelValidator()
    results = validator.compare(
        {"Good": good_fn, "Bad": bad_fn},
        test_states, test_controls, test_targets,
    )

    assert "Good" in results
    assert "Bad" in results
    assert results["Good"]["rmse"] < results["Bad"]["rmse"]

    # Print
    ModelValidator.print_comparison(results)
    print("PASS")


def test_validator_tuple_prediction():
    """predict_fn이 (mean, std) 튜플 반환 시 처리"""
    print("\n" + "=" * 60)
    print("Test: ModelValidator tuple prediction")
    print("=" * 60)

    N, nx, nu = 50, 3, 2
    test_states = np.random.randn(N, nx)
    test_controls = np.random.randn(N, nu)
    test_targets = test_states * 0.1

    def tuple_fn(s, c):
        return (s * 0.1, np.ones_like(s) * 0.01)

    validator = ModelValidator()
    metrics = validator.evaluate(tuple_fn, test_states, test_controls, test_targets)
    assert metrics["rmse"] < 1e-10
    print("PASS")


def test_validator_rollout():
    """ModelValidator rollout evaluation"""
    print("\n" + "=" * 60)
    print("Test: ModelValidator evaluate_rollout")
    print("=" * 60)

    from mppi_controller.models.kinematic.differential_drive_kinematic import (
        DifferentialDriveKinematic,
    )

    model = DifferentialDriveKinematic()
    dt = 0.05

    # Generate ground truth trajectories
    M, T = 5, 20
    nx, nu = 3, 2
    initial_states = np.random.randn(M, nx) * 0.1
    control_sequences = np.random.randn(M, T, nu) * 0.3

    true_trajs = np.zeros((M, T + 1, nx))
    for m in range(M):
        state = initial_states[m].copy()
        true_trajs[m, 0] = state
        for t in range(T):
            state = model.step(state, control_sequences[m, t], dt)
            state = model.normalize_state(state)
            true_trajs[m, t + 1] = state

    validator = ModelValidator()
    metrics = validator.evaluate_rollout(
        model, initial_states, control_sequences, true_trajs, dt
    )

    # Same model → zero error
    assert metrics["mean_rollout_rmse"] < 1e-8
    assert metrics["worst_case_rmse"] < 1e-8
    assert metrics["per_step_rmse"].shape == (T + 1,)
    print(f"Mean rollout RMSE: {metrics['mean_rollout_rmse']:.2e}")
    print("PASS")


# ========== UncertaintyAwareCost Tests ==========


def test_uncertainty_cost_basic():
    """UncertaintyAwareCost 기본"""
    print("\n" + "=" * 60)
    print("Test: UncertaintyAwareCost basic")
    print("=" * 60)

    # 상수 불확실성
    def const_uncertainty(states, controls):
        return np.ones((states.shape[0], 3)) * 0.5

    cost_fn = UncertaintyAwareCost(
        uncertainty_fn=const_uncertainty, beta=1.0, reduce="sum"
    )

    K, N, nx, nu = 10, 5, 3, 2
    trajectories = np.random.randn(K, N + 1, nx)
    controls = np.random.randn(K, N, nu)
    ref = np.zeros((N + 1, nx))

    costs = cost_fn.compute_cost(trajectories, controls, ref)
    assert costs.shape == (K,), f"Expected (K,), got {costs.shape}"
    assert np.all(costs > 0)

    # 상수 불확실성 → 모든 K 동일
    expected = 1.0 * N * 3 * 0.25  # beta * N * nx * 0.5²
    np.testing.assert_allclose(costs, expected, atol=1e-10)
    print(f"Costs: {costs[:3]}")
    print("PASS")


def test_uncertainty_cost_varies():
    """불확실성 크기에 따른 비용 변화"""
    print("\n" + "=" * 60)
    print("Test: UncertaintyAwareCost varies with uncertainty")
    print("=" * 60)

    def varying_uncertainty(states, controls):
        # 원점에서 멀수록 불확실성 증가
        return np.abs(states)

    cost_fn = UncertaintyAwareCost(
        uncertainty_fn=varying_uncertainty, beta=1.0
    )

    K, N, nx, nu = 2, 5, 3, 2
    # Trajectory 1: 원점 근처
    traj_near = np.ones((1, N + 1, nx)) * 0.1
    # Trajectory 2: 원점에서 멀리
    traj_far = np.ones((1, N + 1, nx)) * 2.0

    trajectories = np.concatenate([traj_near, traj_far], axis=0)
    controls = np.zeros((K, N, nu))
    ref = np.zeros((N + 1, nx))

    costs = cost_fn.compute_cost(trajectories, controls, ref)
    assert costs[1] > costs[0], "Far trajectory should have higher uncertainty cost"
    print(f"Near cost: {costs[0]:.4f}, Far cost: {costs[1]:.4f}")
    print("PASS")


def test_uncertainty_cost_reduce_modes():
    """reduce 모드 테스트"""
    print("\n" + "=" * 60)
    print("Test: UncertaintyAwareCost reduce modes")
    print("=" * 60)

    def const_unc(states, controls):
        return np.ones((states.shape[0], 3)) * 0.5

    K, N, nx, nu = 4, 3, 3, 2
    trajectories = np.random.randn(K, N + 1, nx)
    controls = np.random.randn(K, N, nu)
    ref = np.zeros((N + 1, nx))

    for reduce in ["sum", "max", "mean"]:
        cost_fn = UncertaintyAwareCost(
            uncertainty_fn=const_unc, beta=1.0, reduce=reduce
        )
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (K,)
        assert np.all(costs > 0)

    print("PASS")


def test_uncertainty_cost_zero_beta():
    """beta=0 → 비용 0"""
    print("\n" + "=" * 60)
    print("Test: UncertaintyAwareCost zero beta")
    print("=" * 60)

    def some_unc(states, controls):
        return np.ones((states.shape[0], 3))

    cost_fn = UncertaintyAwareCost(uncertainty_fn=some_unc, beta=0.0)

    K, N, nx, nu = 4, 3, 3, 2
    costs = cost_fn.compute_cost(
        np.random.randn(K, N + 1, nx),
        np.random.randn(K, N, nu),
        np.zeros((N + 1, nx)),
    )
    np.testing.assert_allclose(costs, 0.0, atol=1e-15)
    print("PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("P2 Tests: Ensemble, Validator, UncertaintyCost".center(60))
    print("=" * 60)

    try:
        # Ensemble
        test_ensemble_trainer_train()
        test_ensemble_trainer_predict()
        test_ensemble_trainer_save_load()
        test_ensemble_dynamics_no_model()
        test_ensemble_dynamics_loaded()

        # Validator
        test_validator_evaluate()
        test_validator_evaluate_noisy()
        test_validator_compare()
        test_validator_tuple_prediction()
        test_validator_rollout()

        # UncertaintyCost
        test_uncertainty_cost_basic()
        test_uncertainty_cost_varies()
        test_uncertainty_cost_reduce_modes()
        test_uncertainty_cost_zero_beta()

        print("\n" + "=" * 60)
        print("All P2 Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
