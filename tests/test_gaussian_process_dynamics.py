"""
GaussianProcessDynamics 모델 테스트 (P0-6)

Tests:
    - 생성자 (모델 미로드)
    - forward_dynamics (모델 없을 때 에러)
    - forward_dynamics (단일/배치)
    - predict_with_uncertainty 형상
    - lengthscales, noise_variance
    - model_info
    - repr
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

gpytorch = None
torch = None
try:
    import torch as _torch
    import gpytorch as _gpytorch
    torch = _torch
    gpytorch = _gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False

if HAS_GPYTORCH:
    from mppi_controller.models.learned.gaussian_process_dynamics import (
        GaussianProcessDynamics,
    )
    from mppi_controller.learning.gaussian_process_trainer import (
        GaussianProcessTrainer,
    )


def _skip_if_no_gpytorch():
    if not HAS_GPYTORCH:
        print("  SKIPPED (gpytorch not installed)")
        return True
    return False


def _create_trained_gp(state_dim=3, control_dim=2):
    """Helper: 합성 데이터로 학습된 GP 모델 생성 + 저장"""
    save_dir = tempfile.mkdtemp()
    trainer = GaussianProcessTrainer(
        state_dim=state_dim,
        control_dim=control_dim,
        kernel_type="rbf",
        use_sparse=False,
        use_ard=True,
        save_dir=save_dir,
    )

    np.random.seed(42)
    N = 50  # GP는 데이터 적어도 됨
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

    trainer.train(
        inputs[:num_train],
        targets[:num_train],
        inputs[num_train:],
        targets[num_train:],
        norm_stats,
        num_iterations=10,
        verbose=False,
    )

    filename = "gp_model.pth"
    trainer.save_model(filename)
    return trainer, os.path.join(save_dir, filename)


def test_constructor_no_model():
    """모델 미로드 상태 생성자"""
    print("\n" + "=" * 60)
    print("Test: Constructor without model")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    model = GaussianProcessDynamics(state_dim=3, control_dim=2)
    assert model.state_dim == 3
    assert model.control_dim == 2
    assert model.gp_models is None
    assert model.model_type == "learned"
    print("PASS")


def test_forward_dynamics_no_model_raises():
    """모델 없을 때 RuntimeError"""
    print("\n" + "=" * 60)
    print("Test: forward_dynamics without model raises RuntimeError")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    model = GaussianProcessDynamics(state_dim=3, control_dim=2)
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
    if _skip_if_no_gpytorch():
        return

    _, model_path = _create_trained_gp()
    gp = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path=model_path)

    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    result = gp.forward_dynamics(state, control)

    assert result.shape == (3,), f"Expected (3,), got {result.shape}"
    assert np.all(np.isfinite(result)), "Output contains NaN/Inf"
    print(f"Output: {result}")
    print("PASS")


def test_forward_dynamics_batch():
    """배치 입력 forward_dynamics"""
    print("\n" + "=" * 60)
    print("Test: forward_dynamics batch input")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    _, model_path = _create_trained_gp()
    gp = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path=model_path)

    batch_size = 8
    states = np.random.randn(batch_size, 3).astype(np.float32)
    controls = np.random.randn(batch_size, 2).astype(np.float32)
    result = gp.forward_dynamics(states, controls)

    assert result.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {result.shape}"
    assert np.all(np.isfinite(result))
    print(f"Output shape: {result.shape}")
    print("PASS")


def test_predict_with_uncertainty_shape():
    """predict_with_uncertainty 출력 형상 (단일/배치)"""
    print("\n" + "=" * 60)
    print("Test: predict_with_uncertainty shape")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    _, model_path = _create_trained_gp()
    gp = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path=model_path)

    # Single
    state = np.array([0.5, -0.3, 0.1])
    control = np.array([0.2, -0.1])
    mean, std = gp.predict_with_uncertainty(state, control)
    assert mean.shape == (3,), f"mean shape: {mean.shape}"
    assert std.shape == (3,), f"std shape: {std.shape}"
    assert np.all(std >= 0), "std should be non-negative"

    # Batch
    batch_size = 8
    states = np.random.randn(batch_size, 3).astype(np.float32)
    controls = np.random.randn(batch_size, 2).astype(np.float32)
    mean, std = gp.predict_with_uncertainty(states, controls)
    assert mean.shape == (batch_size, 3), f"mean shape: {mean.shape}"
    assert std.shape == (batch_size, 3), f"std shape: {std.shape}"
    assert np.all(std >= 0), "std should be non-negative"

    print("PASS")


def test_lengthscales():
    """학습된 lengthscales 확인"""
    print("\n" + "=" * 60)
    print("Test: get_lengthscales")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    _, model_path = _create_trained_gp()
    gp = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path=model_path)

    ls = gp.get_lengthscales()
    assert len(ls) == 3, f"Expected 3 output dims, got {len(ls)}"
    for dim, values in ls.items():
        assert values.shape == (5,), f"dim {dim}: expected (5,), got {values.shape}"
        assert np.all(values > 0), f"dim {dim}: lengthscales should be positive"
    print("PASS")


def test_noise_variance():
    """학습된 noise variance 확인"""
    print("\n" + "=" * 60)
    print("Test: get_noise_variance")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    _, model_path = _create_trained_gp()
    gp = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path=model_path)

    nv = gp.get_noise_variance()
    assert len(nv) == 3, f"Expected 3, got {len(nv)}"
    for dim, val in nv.items():
        assert val > 0, f"dim {dim}: noise variance should be positive"
    print("PASS")


def test_model_info():
    """get_model_info"""
    print("\n" + "=" * 60)
    print("Test: get_model_info")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    # Unloaded
    gp_unloaded = GaussianProcessDynamics(state_dim=3, control_dim=2)
    info = gp_unloaded.get_model_info()
    assert info["loaded"] is False

    # Loaded
    _, model_path = _create_trained_gp()
    gp_loaded = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path=model_path)
    info = gp_loaded.get_model_info()
    assert info["loaded"] is True
    assert info["num_models"] == 3
    assert info["num_parameters"] > 0
    print(f"Model info: {info}")
    print("PASS")


def test_repr():
    """__repr__"""
    print("\n" + "=" * 60)
    print("Test: __repr__")
    print("=" * 60)
    if _skip_if_no_gpytorch():
        return

    gp = GaussianProcessDynamics(state_dim=3, control_dim=2)
    r = repr(gp)
    assert "loaded=False" in r

    _, model_path = _create_trained_gp()
    gp_loaded = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path=model_path)
    r = repr(gp_loaded)
    assert "loaded=True" in r
    print(f"repr: {r}")
    print("PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GaussianProcessDynamics Tests".center(60))
    print("=" * 60)

    try:
        test_constructor_no_model()
        test_forward_dynamics_no_model_raises()
        test_forward_dynamics_single()
        test_forward_dynamics_batch()
        test_predict_with_uncertainty_shape()
        test_lengthscales()
        test_noise_variance()
        test_model_info()
        test_repr()

        print("\n" + "=" * 60)
        print("All GaussianProcessDynamics Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
