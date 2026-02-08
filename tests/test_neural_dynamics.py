"""
NeuralDynamics 모델 테스트 (P0-5)

Tests:
    - 생성자 (모델 미로드)
    - forward_dynamics (모델 없을 때 에러)
    - forward_dynamics (단일/배치)
    - normalization
    - model_info
    - repr
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
from mppi_controller.learning.neural_network_trainer import (
    DynamicsMLPModel,
    NeuralNetworkTrainer,
)


def _create_trained_model(state_dim=3, control_dim=2, hidden_dims=None):
    """Helper: 합성 데이터로 학습된 모델 생성"""
    if hidden_dims is None:
        hidden_dims = [32, 32]

    trainer = NeuralNetworkTrainer(
        state_dim=state_dim,
        control_dim=control_dim,
        hidden_dims=hidden_dims,
        learning_rate=1e-3,
        save_dir=tempfile.mkdtemp(),
    )

    # Synthetic data: state_dot = A @ [state, control]
    np.random.seed(42)
    N = 200
    inputs = np.random.randn(N, state_dim + control_dim).astype(np.float32)
    A = np.random.randn(state_dim, state_dim + control_dim).astype(np.float32) * 0.1
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

    trainer.train(
        inputs[:num_train],
        targets[:num_train],
        inputs[num_train:],
        targets[num_train:],
        norm_stats,
        epochs=20,
        batch_size=32,
        early_stopping_patience=50,
        verbose=False,
    )

    return trainer


def test_constructor_no_model():
    """모델 미로드 상태에서 생성자 테스트"""
    print("\n" + "=" * 60)
    print("Test: Constructor without model")
    print("=" * 60)

    model = NeuralDynamics(state_dim=3, control_dim=2)
    assert model.state_dim == 3
    assert model.control_dim == 2
    assert model.model is None
    assert model.model_type == "learned"
    print("PASS")


def test_forward_dynamics_no_model_raises():
    """모델 없을 때 forward_dynamics 호출 시 RuntimeError"""
    print("\n" + "=" * 60)
    print("Test: forward_dynamics without model raises RuntimeError")
    print("=" * 60)

    model = NeuralDynamics(state_dim=3, control_dim=2)
    state = np.array([1.0, 2.0, 0.5])
    control = np.array([0.5, 0.1])

    try:
        model.forward_dynamics(state, control)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("PASS")


def test_forward_dynamics_single():
    """단일 입력 forward_dynamics"""
    print("\n" + "=" * 60)
    print("Test: forward_dynamics single input")
    print("=" * 60)

    trainer = _create_trained_model()
    # Save and load
    model_path = os.path.join(trainer.save_dir, "best_model.pth")
    neural = NeuralDynamics(state_dim=3, control_dim=2, model_path=model_path)

    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    result = neural.forward_dynamics(state, control)

    assert result.shape == (3,), f"Expected (3,), got {result.shape}"
    assert np.all(np.isfinite(result)), "Output contains NaN/Inf"
    print(f"Output: {result}")
    print("PASS")


def test_forward_dynamics_batch():
    """배치 입력 forward_dynamics"""
    print("\n" + "=" * 60)
    print("Test: forward_dynamics batch input")
    print("=" * 60)

    trainer = _create_trained_model()
    model_path = os.path.join(trainer.save_dir, "best_model.pth")
    neural = NeuralDynamics(state_dim=3, control_dim=2, model_path=model_path)

    batch_size = 16
    states = np.random.randn(batch_size, 3)
    controls = np.random.randn(batch_size, 2)
    result = neural.forward_dynamics(states, controls)

    assert result.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {result.shape}"
    assert np.all(np.isfinite(result)), "Output contains NaN/Inf"
    print(f"Output shape: {result.shape}")
    print("PASS")


def test_normalization_effect():
    """정규화 통계 적용 확인"""
    print("\n" + "=" * 60)
    print("Test: Normalization effect")
    print("=" * 60)

    trainer = _create_trained_model()
    model_path = os.path.join(trainer.save_dir, "best_model.pth")
    neural = NeuralDynamics(state_dim=3, control_dim=2, model_path=model_path)

    assert neural.norm_stats is not None, "norm_stats should be loaded"
    assert "state_mean" in neural.norm_stats
    assert "state_std" in neural.norm_stats
    assert "state_dot_mean" in neural.norm_stats
    print("PASS")


def test_model_info():
    """모델 정보 확인"""
    print("\n" + "=" * 60)
    print("Test: get_model_info")
    print("=" * 60)

    # Unloaded model
    neural_unloaded = NeuralDynamics(state_dim=3, control_dim=2)
    info = neural_unloaded.get_model_info()
    assert info["loaded"] is False

    # Loaded model
    trainer = _create_trained_model()
    model_path = os.path.join(trainer.save_dir, "best_model.pth")
    neural_loaded = NeuralDynamics(state_dim=3, control_dim=2, model_path=model_path)
    info = neural_loaded.get_model_info()
    assert info["loaded"] is True
    assert info["input_dim"] == 5  # 3 + 2
    assert info["output_dim"] == 3
    assert info["num_parameters"] > 0
    print(f"Model info: {info}")
    print("PASS")


def test_repr():
    """__repr__ 출력 확인"""
    print("\n" + "=" * 60)
    print("Test: __repr__")
    print("=" * 60)

    neural_unloaded = NeuralDynamics(state_dim=3, control_dim=2)
    r = repr(neural_unloaded)
    assert "loaded=False" in r

    trainer = _create_trained_model()
    model_path = os.path.join(trainer.save_dir, "best_model.pth")
    neural_loaded = NeuralDynamics(state_dim=3, control_dim=2, model_path=model_path)
    r = repr(neural_loaded)
    assert "loaded=True" in r
    print(f"repr: {r}")
    print("PASS")


def test_load_nonexistent_file():
    """존재하지 않는 파일 로드 시 FileNotFoundError"""
    print("\n" + "=" * 60)
    print("Test: Load non-existent file raises FileNotFoundError")
    print("=" * 60)

    try:
        NeuralDynamics(state_dim=3, control_dim=2, model_path="/tmp/nonexistent_model.pth")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    print("PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuralDynamics Tests".center(60))
    print("=" * 60)

    try:
        test_constructor_no_model()
        test_forward_dynamics_no_model_raises()
        test_forward_dynamics_single()
        test_forward_dynamics_batch()
        test_normalization_effect()
        test_model_info()
        test_repr()
        test_load_nonexistent_file()

        print("\n" + "=" * 60)
        print("All NeuralDynamics Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
