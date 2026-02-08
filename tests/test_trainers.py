"""
NeuralNetworkTrainer + GaussianProcessTrainer 테스트 (P0-8)

Tests:
    - NN Trainer: 합성 데이터 학습, predict 형상, save/load, early stopping, history
    - GP Trainer: 합성 데이터 학습, predict 형상, save/load, 모델 누적 수정 확인
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from mppi_controller.learning.neural_network_trainer import (
    DynamicsMLPModel,
    NeuralNetworkTrainer,
)

gpytorch = None
try:
    import gpytorch as _gpytorch
    gpytorch = _gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False

if HAS_GPYTORCH:
    from mppi_controller.learning.gaussian_process_trainer import (
        GaussianProcessTrainer,
    )


def _make_nn_data(state_dim=3, control_dim=2, N=200):
    """합성 데이터 생성 (NN용)"""
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


# ========== DynamicsMLPModel Tests ==========


def test_mlp_model_forward():
    """DynamicsMLPModel forward pass"""
    print("\n" + "=" * 60)
    print("Test: DynamicsMLPModel forward pass")
    print("=" * 60)

    model = DynamicsMLPModel(input_dim=5, output_dim=3, hidden_dims=[32, 32])
    x = torch.randn(10, 5)
    y = model(x)

    assert y.shape == (10, 3), f"Expected (10, 3), got {y.shape}"
    assert model.input_dim == 5
    assert model.output_dim == 3
    assert model.hidden_dims == [32, 32]
    print("PASS")


def test_mlp_model_activations():
    """다양한 활성화 함수 테스트"""
    print("\n" + "=" * 60)
    print("Test: DynamicsMLPModel activations")
    print("=" * 60)

    for act in ["relu", "tanh", "elu"]:
        model = DynamicsMLPModel(input_dim=5, output_dim=3, hidden_dims=[16], activation=act)
        y = model(torch.randn(4, 5))
        assert y.shape == (4, 3), f"{act}: shape {y.shape}"

    # Unknown activation
    try:
        DynamicsMLPModel(input_dim=5, output_dim=3, activation="unknown")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS")


# ========== NeuralNetworkTrainer Tests ==========


def test_nn_trainer_train():
    """NN Trainer 학습 테스트"""
    print("\n" + "=" * 60)
    print("Test: NeuralNetworkTrainer train")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        learning_rate=1e-3,
        save_dir=save_dir,
    )

    inputs, targets, num_train, norm_stats = _make_nn_data()

    history = trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats,
        epochs=30,
        batch_size=32,
        early_stopping_patience=50,
        verbose=False,
    )

    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) > 0
    assert len(history["val_loss"]) > 0
    # Loss should decrease
    assert history["train_loss"][-1] < history["train_loss"][0], "Training loss should decrease"
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print("PASS")


def test_nn_trainer_predict_single():
    """NN Trainer predict (단일)"""
    print("\n" + "=" * 60)
    print("Test: NeuralNetworkTrainer predict single")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2, hidden_dims=[32], save_dir=save_dir
    )
    inputs, targets, num_train, norm_stats = _make_nn_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=5, verbose=False,
    )

    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    pred = trainer.predict(state, control)

    assert pred.shape == (3,), f"Expected (3,), got {pred.shape}"
    assert np.all(np.isfinite(pred))
    print(f"Prediction: {pred}")
    print("PASS")


def test_nn_trainer_predict_batch():
    """NN Trainer predict (배치)"""
    print("\n" + "=" * 60)
    print("Test: NeuralNetworkTrainer predict batch")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2, hidden_dims=[32], save_dir=save_dir
    )
    inputs, targets, num_train, norm_stats = _make_nn_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=5, verbose=False,
    )

    states = np.random.randn(16, 3)
    controls = np.random.randn(16, 2)
    pred = trainer.predict(states, controls)

    assert pred.shape == (16, 3), f"Expected (16, 3), got {pred.shape}"
    print("PASS")


def test_nn_trainer_save_load():
    """NN Trainer save/load 왕복"""
    print("\n" + "=" * 60)
    print("Test: NeuralNetworkTrainer save/load roundtrip")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2, hidden_dims=[32], save_dir=save_dir
    )
    inputs, targets, num_train, norm_stats = _make_nn_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=5, verbose=False,
    )

    # Predict before save
    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    pred_before = trainer.predict(state, control)

    # Save
    trainer.save_model("test_nn.pth")

    # Load into new trainer
    trainer2 = NeuralNetworkTrainer(
        state_dim=3, control_dim=2, hidden_dims=[32], save_dir=save_dir
    )
    trainer2.load_model("test_nn.pth")

    pred_after = trainer2.predict(state, control)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
    print("PASS")


def test_nn_trainer_early_stopping():
    """Early stopping 동작"""
    print("\n" + "=" * 60)
    print("Test: NeuralNetworkTrainer early stopping")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2, hidden_dims=[32], save_dir=save_dir
    )
    inputs, targets, num_train, norm_stats = _make_nn_data()

    history = trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=1000, early_stopping_patience=5, verbose=False,
    )

    # Early stopping should have terminated before 1000 epochs
    num_epochs = len(history["train_loss"])
    assert num_epochs < 1000, f"Expected early stopping, ran {num_epochs} epochs"
    print(f"Stopped at epoch {num_epochs}")
    print("PASS")


def test_nn_trainer_model_summary():
    """모델 요약"""
    print("\n" + "=" * 60)
    print("Test: NeuralNetworkTrainer get_model_summary")
    print("=" * 60)

    trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2, hidden_dims=[32, 32],
        save_dir=tempfile.mkdtemp(),
    )
    summary = trainer.get_model_summary()
    assert "Input dim: 5" in summary
    assert "Output dim: 3" in summary
    print(summary)
    print("PASS")


# ========== GaussianProcessTrainer Tests ==========


def _skip_if_no_gpytorch():
    if not HAS_GPYTORCH:
        print("  SKIPPED (gpytorch not installed)")
        return True
    return False


def _make_gp_data(state_dim=3, control_dim=2, N=50):
    """합성 데이터 (GP용, 적은 데이터)"""
    np.random.seed(42)
    input_dim = state_dim + control_dim
    inputs = np.random.randn(N, input_dim).astype(np.float32)
    A = np.random.randn(state_dim, input_dim).astype(np.float32) * 0.1
    targets = inputs @ A.T

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


def test_gp_trainer_train():
    """GP Trainer 학습"""
    print("\n" + "=" * 60)
    print("Test: GaussianProcessTrainer train")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    trainer = GaussianProcessTrainer(
        state_dim=3, control_dim=2,
        kernel_type="rbf",
        save_dir=tempfile.mkdtemp(),
    )
    inputs, targets, num_train, norm_stats = _make_gp_data()

    history = trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats,
        num_iterations=10,
        verbose=False,
    )

    assert len(trainer.gp_models) == 3, f"Expected 3 models, got {len(trainer.gp_models)}"
    assert len(trainer.likelihoods) == 3
    assert "val_loss" in history
    print("PASS")


def test_gp_trainer_no_accumulation():
    """P0-3: train() 반복 호출 시 모델 누적 안됨"""
    print("\n" + "=" * 60)
    print("Test: GP Trainer no model accumulation on repeated train()")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    trainer = GaussianProcessTrainer(
        state_dim=3, control_dim=2,
        save_dir=tempfile.mkdtemp(),
    )
    inputs, targets, num_train, norm_stats = _make_gp_data()

    # Train twice
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, num_iterations=5, verbose=False,
    )
    assert len(trainer.gp_models) == 3, f"After 1st train: {len(trainer.gp_models)}"

    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, num_iterations=5, verbose=False,
    )
    assert len(trainer.gp_models) == 3, (
        f"After 2nd train: {len(trainer.gp_models)} (should be 3, not 6)"
    )
    print("PASS")


def test_gp_trainer_predict():
    """GP Trainer predict 형상"""
    print("\n" + "=" * 60)
    print("Test: GaussianProcessTrainer predict")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    trainer = GaussianProcessTrainer(
        state_dim=3, control_dim=2,
        save_dir=tempfile.mkdtemp(),
    )
    inputs, targets, num_train, norm_stats = _make_gp_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, num_iterations=10, verbose=False,
    )

    # Single
    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    mean, std = trainer.predict(state, control, return_uncertainty=True)
    assert mean.shape == (3,), f"mean: {mean.shape}"
    assert std.shape == (3,), f"std: {std.shape}"

    # Batch
    states = np.random.randn(8, 3).astype(np.float32)
    controls = np.random.randn(8, 2).astype(np.float32)
    mean, std = trainer.predict(states, controls, return_uncertainty=True)
    assert mean.shape == (8, 3), f"mean: {mean.shape}"
    assert std.shape == (8, 3), f"std: {std.shape}"
    print("PASS")


def test_gp_trainer_save_load():
    """P0-2: GP Trainer save/load 왕복 (모델 재구성 확인)"""
    print("\n" + "=" * 60)
    print("Test: GaussianProcessTrainer save/load roundtrip")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    save_dir = tempfile.mkdtemp()
    trainer = GaussianProcessTrainer(
        state_dim=3, control_dim=2,
        save_dir=save_dir,
    )
    inputs, targets, num_train, norm_stats = _make_gp_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, num_iterations=10, verbose=False,
    )

    # Predict before save
    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    pred_before, _ = trainer.predict(state, control)

    trainer.save_model("test_gp.pth")

    # Load into new trainer
    trainer2 = GaussianProcessTrainer(
        state_dim=3, control_dim=2,
        save_dir=save_dir,
    )
    trainer2.load_model("test_gp.pth")

    assert len(trainer2.gp_models) == 3, (
        f"Loaded {len(trainer2.gp_models)} models (expected 3)"
    )

    pred_after, _ = trainer2.predict(state, control)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-4)
    print("PASS")


def test_gp_trainer_model_summary():
    """모델 요약"""
    print("\n" + "=" * 60)
    print("Test: GaussianProcessTrainer get_model_summary")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    trainer = GaussianProcessTrainer(
        state_dim=3, control_dim=2,
        save_dir=tempfile.mkdtemp(),
    )

    # Before training
    summary = trainer.get_model_summary()
    assert "No models trained" in summary

    # After training
    inputs, targets, num_train, norm_stats = _make_gp_data()
    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, num_iterations=5, verbose=False,
    )
    summary = trainer.get_model_summary()
    assert "Num models: 3" in summary
    print(summary)
    print("PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuralNetworkTrainer + GaussianProcessTrainer Tests".center(60))
    print("=" * 60)

    try:
        # MLP Model
        test_mlp_model_forward()
        test_mlp_model_activations()

        # NN Trainer
        test_nn_trainer_train()
        test_nn_trainer_predict_single()
        test_nn_trainer_predict_batch()
        test_nn_trainer_save_load()
        test_nn_trainer_early_stopping()
        test_nn_trainer_model_summary()

        # GP Trainer
        test_gp_trainer_train()
        test_gp_trainer_no_accumulation()
        test_gp_trainer_predict()
        test_gp_trainer_save_load()
        test_gp_trainer_model_summary()

        print("\n" + "=" * 60)
        print("All Trainer Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
