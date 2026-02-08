#!/usr/bin/env python3
"""
P3 테스트: MC-Dropout Bayesian NN + 체크포인트 버전 관리

MC-Dropout 불확실성 추정 및 OnlineLearner 체크포인트 롤백 검증.
"""

import sys
import os
import numpy as np
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ──────────────────────────────────────────────
# MC-Dropout Tests
# ──────────────────────────────────────────────


def test_mc_dropout_constructor():
    """MCDropoutDynamics 생성자 테스트"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    model = MCDropoutDynamics(state_dim=3, control_dim=2, num_samples=10)
    assert model.state_dim == 3
    assert model.control_dim == 2
    assert model.num_samples == 10
    assert model.model is None
    assert model.model_type == "learned"
    print("  [PASS] test_mc_dropout_constructor")


def test_mc_dropout_repr_unloaded():
    """MCDropoutDynamics repr (미로드 상태)"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    model = MCDropoutDynamics(state_dim=3, control_dim=2)
    r = repr(model)
    assert "loaded=False" in r
    assert "MCDropoutDynamics" in r
    print("  [PASS] test_mc_dropout_repr_unloaded")


def test_mc_dropout_model_info_unloaded():
    """MCDropoutDynamics model_info (미로드)"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    model = MCDropoutDynamics(state_dim=3, control_dim=2)
    info = model.get_model_info()
    assert info["loaded"] is False
    print("  [PASS] test_mc_dropout_model_info_unloaded")


def test_mc_dropout_forward_raises_without_model():
    """모델 미로드 시 RuntimeError"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    model = MCDropoutDynamics(state_dim=3, control_dim=2)
    try:
        model.forward_dynamics(np.zeros(3), np.zeros(2))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("  [PASS] test_mc_dropout_forward_raises_without_model")


def _create_mc_dropout_model_checkpoint(tmpdir, dropout_rate=0.3):
    """MC-Dropout용 모델 체크포인트 생성 헬퍼"""
    import torch
    from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel

    state_dim, control_dim = 3, 2
    model = DynamicsMLPModel(
        input_dim=state_dim + control_dim,
        output_dim=state_dim,
        hidden_dims=[32, 32],
        dropout_rate=dropout_rate,
    )

    norm_stats = {
        "state_mean": np.zeros(state_dim),
        "state_std": np.ones(state_dim),
        "control_mean": np.zeros(control_dim),
        "control_std": np.ones(control_dim),
        "state_dot_mean": np.zeros(state_dim),
        "state_dot_std": np.ones(state_dim),
    }

    filepath = os.path.join(tmpdir, "mc_dropout_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "state_dim": state_dim,
            "control_dim": control_dim,
            "hidden_dims": [32, 32],
            "activation": "relu",
            "dropout_rate": dropout_rate,
        },
        "norm_stats": norm_stats,
    }, filepath)

    return filepath


def test_mc_dropout_load_and_predict():
    """MC-Dropout 모델 로드 및 예측"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _create_mc_dropout_model_checkpoint(tmpdir)

        model = MCDropoutDynamics(
            state_dim=3, control_dim=2,
            model_path=path,
            num_samples=10,
        )

        state = np.array([1.0, 2.0, 0.5])
        control = np.array([0.5, 0.3])

        result = model.forward_dynamics(state, control)
        assert result.shape == (3,), f"Expected (3,), got {result.shape}"
        assert np.all(np.isfinite(result))
    print("  [PASS] test_mc_dropout_load_and_predict")


def test_mc_dropout_uncertainty_shape():
    """MC-Dropout 불확실성 형상 검증"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _create_mc_dropout_model_checkpoint(tmpdir)

        model = MCDropoutDynamics(
            state_dim=3, control_dim=2,
            model_path=path,
            num_samples=30,
        )

        state = np.array([1.0, 2.0, 0.5])
        control = np.array([0.5, 0.3])

        mean, std = model.predict_with_uncertainty(state, control)
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert np.all(std >= 0), "std should be non-negative"
    print("  [PASS] test_mc_dropout_uncertainty_shape")


def test_mc_dropout_batch_prediction():
    """MC-Dropout 배치 예측"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _create_mc_dropout_model_checkpoint(tmpdir)

        model = MCDropoutDynamics(
            state_dim=3, control_dim=2,
            model_path=path,
            num_samples=10,
        )

        states = np.random.randn(5, 3)
        controls = np.random.randn(5, 2)

        mean, std = model.predict_with_uncertainty(states, controls)
        assert mean.shape == (5, 3)
        assert std.shape == (5, 3)
    print("  [PASS] test_mc_dropout_batch_prediction")


def test_mc_dropout_uncertainty_nonzero_with_dropout():
    """dropout > 0 일 때 불확실성이 0이 아닌지 확인"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _create_mc_dropout_model_checkpoint(tmpdir, dropout_rate=0.5)

        model = MCDropoutDynamics(
            state_dim=3, control_dim=2,
            model_path=path,
            num_samples=50,  # 충분한 샘플
        )

        state = np.array([1.0, 2.0, 0.5])
        control = np.array([0.5, 0.3])

        _, std = model.predict_with_uncertainty(state, control)
        # dropout=0.5이면 일반적으로 std > 0
        assert np.sum(std) > 0, "With dropout=0.5, std should generally be > 0"
    print("  [PASS] test_mc_dropout_uncertainty_nonzero_with_dropout")


def test_mc_dropout_repr_loaded():
    """MC-Dropout repr (로드 상태)"""
    from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _create_mc_dropout_model_checkpoint(tmpdir)

        model = MCDropoutDynamics(
            state_dim=3, control_dim=2,
            model_path=path,
            num_samples=20,
        )

        r = repr(model)
        assert "loaded=True" in r
        assert "M=20" in r
        assert "dropout=0.3" in r

        info = model.get_model_info()
        assert info["loaded"] is True
        assert info["num_samples"] == 20
        assert info["dropout_rate"] == 0.3
    print("  [PASS] test_mc_dropout_repr_loaded")


# ──────────────────────────────────────────────
# Checkpoint Versioning Tests
# ──────────────────────────────────────────────


def _create_simple_learner(tmpdir, checkpoint_dir=None):
    """테스트용 OnlineLearner + NeuralNetworkTrainer 생성 헬퍼"""
    import torch
    from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
    from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
    from mppi_controller.learning.online_learner import OnlineLearner

    state_dim, control_dim = 3, 2

    model = NeuralDynamics(state_dim=state_dim, control_dim=control_dim)

    save_dir = os.path.join(tmpdir, "trainer_models")
    trainer = NeuralNetworkTrainer(
        state_dim=state_dim,
        control_dim=control_dim,
        hidden_dims=[16, 16],
        save_dir=save_dir,
    )

    learner = OnlineLearner(
        model=model,
        trainer=trainer,
        buffer_size=200,
        min_samples_for_update=20,
        update_interval=50,
        verbose=False,
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=5,
    )

    return learner


def test_checkpoint_disabled_by_default():
    """체크포인트 기본 비활성화"""
    from mppi_controller.learning.online_learner import OnlineLearner

    with tempfile.TemporaryDirectory() as tmpdir:
        learner = _create_simple_learner(tmpdir)
        assert learner.checkpoint_dir is None
        assert len(learner.checkpoint_versions) == 0
    print("  [PASS] test_checkpoint_disabled_by_default")


def test_checkpoint_dir_created():
    """체크포인트 디렉토리 자동 생성"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = os.path.join(tmpdir, "checkpoints")
        learner = _create_simple_learner(tmpdir, checkpoint_dir=cp_dir)
        assert os.path.isdir(cp_dir)
    print("  [PASS] test_checkpoint_dir_created")


def test_checkpoint_saved_on_update():
    """update_model() 시 체크포인트 저장"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = os.path.join(tmpdir, "checkpoints")
        learner = _create_simple_learner(tmpdir, checkpoint_dir=cp_dir)

        # 데이터 추가
        for _ in range(30):
            state = np.random.randn(3)
            control = np.random.randn(2)
            next_state = state + np.random.randn(3) * 0.01
            learner.buffer.add(state, control, next_state, 0.05)

        learner.update_model(num_epochs=2)

        assert len(learner.checkpoint_versions) == 1
        assert learner.checkpoint_versions[0]["version"] == 1
        assert learner.checkpoint_versions[0]["val_loss"] > 0
    print("  [PASS] test_checkpoint_saved_on_update")


def test_checkpoint_history():
    """체크포인트 히스토리 조회"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = os.path.join(tmpdir, "checkpoints")
        learner = _create_simple_learner(tmpdir, checkpoint_dir=cp_dir)

        # 2회 업데이트
        for _ in range(2):
            for _ in range(30):
                state = np.random.randn(3)
                control = np.random.randn(2)
                next_state = state + np.random.randn(3) * 0.01
                learner.buffer.add(state, control, next_state, 0.05)
            learner.update_model(num_epochs=2)

        history = learner.get_checkpoint_history()
        assert len(history) >= 1
        # At least one should be best
        bests = [h for h in history if h["is_best"]]
        assert len(bests) == 1
    print("  [PASS] test_checkpoint_history")


def test_get_best_checkpoint():
    """최적 체크포인트 조회"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = os.path.join(tmpdir, "checkpoints")
        learner = _create_simple_learner(tmpdir, checkpoint_dir=cp_dir)

        # 빈 상태
        assert learner._get_best_checkpoint() is None

        # 업데이트
        for _ in range(30):
            state = np.random.randn(3)
            control = np.random.randn(2)
            next_state = state + np.random.randn(3) * 0.01
            learner.buffer.add(state, control, next_state, 0.05)
        learner.update_model(num_epochs=2)

        best = learner._get_best_checkpoint()
        assert best is not None
        assert "val_loss" in best
    print("  [PASS] test_get_best_checkpoint")


def test_rollback_no_checkpoints():
    """체크포인트 없이 롤백 시 False 반환"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = os.path.join(tmpdir, "checkpoints")
        learner = _create_simple_learner(tmpdir, checkpoint_dir=cp_dir)

        result = learner.rollback()
        assert result is False
    print("  [PASS] test_rollback_no_checkpoints")


def test_rollback_invalid_version():
    """존재하지 않는 버전 롤백 시 False"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = os.path.join(tmpdir, "checkpoints")
        learner = _create_simple_learner(tmpdir, checkpoint_dir=cp_dir)

        # 1회 업데이트
        for _ in range(30):
            state = np.random.randn(3)
            control = np.random.randn(2)
            next_state = state + np.random.randn(3) * 0.01
            learner.buffer.add(state, control, next_state, 0.05)
        learner.update_model(num_epochs=2)

        result = learner.rollback(version=999)
        assert result is False
    print("  [PASS] test_rollback_invalid_version")


def test_reset_clears_checkpoints():
    """reset() 시 체크포인트 목록 초기화"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = os.path.join(tmpdir, "checkpoints")
        learner = _create_simple_learner(tmpdir, checkpoint_dir=cp_dir)

        # 업데이트
        for _ in range(30):
            state = np.random.randn(3)
            control = np.random.randn(2)
            next_state = state + np.random.randn(3) * 0.01
            learner.buffer.add(state, control, next_state, 0.05)
        learner.update_model(num_epochs=2)

        assert len(learner.checkpoint_versions) > 0
        learner.reset()
        assert len(learner.checkpoint_versions) == 0
    print("  [PASS] test_reset_clears_checkpoints")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # MC-Dropout
        test_mc_dropout_constructor,
        test_mc_dropout_repr_unloaded,
        test_mc_dropout_model_info_unloaded,
        test_mc_dropout_forward_raises_without_model,
        test_mc_dropout_load_and_predict,
        test_mc_dropout_uncertainty_shape,
        test_mc_dropout_batch_prediction,
        test_mc_dropout_uncertainty_nonzero_with_dropout,
        test_mc_dropout_repr_loaded,
        # Checkpoint versioning
        test_checkpoint_disabled_by_default,
        test_checkpoint_dir_created,
        test_checkpoint_saved_on_update,
        test_checkpoint_history,
        test_get_best_checkpoint,
        test_rollback_no_checkpoints,
        test_rollback_invalid_version,
        test_reset_clears_checkpoints,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed}/{passed + failed} passed")
    if failed > 0:
        sys.exit(1)
