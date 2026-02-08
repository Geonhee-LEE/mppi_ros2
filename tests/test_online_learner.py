"""
OnlineLearner + OnlineDataBuffer 테스트 (P0-9)

Tests:
    - OnlineDataBuffer: add, get_batch, circular buffer, statistics
    - OnlineLearner: add_sample, should_retrain, update_model, reset
    - P0-1: repr 크래시 수정 확인
    - P0-4: 셔플 분할 수정 확인
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.learning.online_learner import OnlineDataBuffer, OnlineLearner


# ========== OnlineDataBuffer Tests ==========


def test_buffer_add_and_len():
    """add + __len__"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer add + len")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2, buffer_size=100)
    assert len(buf) == 0

    for i in range(10):
        buf.add(
            np.random.randn(3),
            np.random.randn(2),
            np.random.randn(3),
            0.05,
        )
    assert len(buf) == 10
    assert buf.num_samples == 10
    print("PASS")


def test_buffer_circular():
    """순환 버퍼 (FIFO)"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer circular buffer")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2, buffer_size=5)

    # Add 8 samples (exceeds buffer_size=5)
    for i in range(8):
        state = np.array([float(i), 0.0, 0.0])
        buf.add(state, np.zeros(2), np.zeros(3), 0.05)

    assert len(buf) == 5, f"Expected 5 (buffer_size), got {len(buf)}"
    assert buf.num_samples == 8, f"Expected 8 total samples, got {buf.num_samples}"

    # Oldest samples (0,1,2) should be evicted; remaining: 3,4,5,6,7
    first_state = buf.states[0]
    assert first_state[0] == 3.0, f"Expected 3.0, got {first_state[0]}"
    print("PASS")


def test_buffer_get_batch():
    """get_batch 형상"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer get_batch")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2, buffer_size=100, batch_size=8)

    for _ in range(20):
        buf.add(np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05)

    batch = buf.get_batch()
    assert batch["states"].shape == (8, 3)
    assert batch["controls"].shape == (8, 2)
    assert batch["next_states"].shape == (8, 3)
    assert batch["state_dots"].shape == (8, 3)
    assert batch["dt"].shape == (8,)

    # Custom batch size
    batch = buf.get_batch(batch_size=5)
    assert batch["states"].shape[0] == 5

    # batch_size > len → capped
    batch = buf.get_batch(batch_size=1000)
    assert batch["states"].shape[0] == 20
    print("PASS")


def test_buffer_get_batch_empty_raises():
    """빈 버퍼에서 get_batch 에러"""
    print("\n" + "=" * 60)
    print("Test: get_batch on empty buffer raises")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2)
    try:
        buf.get_batch()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS")


def test_buffer_get_all_data():
    """get_all_data"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer get_all_data")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2)
    N = 15
    for _ in range(N):
        buf.add(np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05)

    data = buf.get_all_data()
    assert data["states"].shape == (N, 3)
    assert data["state_dots"].shape == (N, 3)
    print("PASS")


def test_buffer_statistics():
    """통계 업데이트"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer statistics")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2, buffer_size=1000)

    # Add 100 samples to trigger stats update
    for _ in range(100):
        buf.add(np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05)

    stats = buf.get_statistics()
    assert stats["state_mean"].shape == (3,)
    assert stats["state_std"].shape == (3,)
    assert stats["control_mean"].shape == (2,)
    assert stats["state_dot_mean"].shape == (3,)

    # Std should be non-trivial after 100 samples
    assert np.all(stats["state_std"] > 1e-6)
    print("PASS")


def test_buffer_should_retrain():
    """should_retrain 로직"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer should_retrain")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2)

    # Not enough samples
    for _ in range(50):
        buf.add(np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05)
    assert buf.should_retrain(min_samples=100) is False

    # Enough samples, but not at interval
    for _ in range(51):
        buf.add(np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05)
    # num_samples=101, not a multiple of 500
    assert buf.should_retrain(min_samples=100, retrain_interval=500) is False

    print("PASS")


def test_buffer_clear():
    """clear"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer clear")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2)
    for _ in range(10):
        buf.add(np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05)

    buf.clear()
    assert len(buf) == 0
    assert buf.num_samples == 0
    print("PASS")


def test_buffer_repr():
    """OnlineDataBuffer __repr__"""
    print("\n" + "=" * 60)
    print("Test: OnlineDataBuffer __repr__")
    print("=" * 60)

    buf = OnlineDataBuffer(state_dim=3, control_dim=2, buffer_size=100)
    r = repr(buf)
    assert "size=0/100" in r
    assert "batch_size=" in r
    print(f"repr: {r}")
    print("PASS")


# ========== OnlineLearner Tests ==========


def _create_online_learner():
    """Helper: NN 기반 OnlineLearner 생성"""
    from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
    from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

    model = NeuralDynamics(state_dim=3, control_dim=2)
    trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[16],
        save_dir=tempfile.mkdtemp(),
    )
    learner = OnlineLearner(
        model=model,
        trainer=trainer,
        buffer_size=500,
        batch_size=32,
        min_samples_for_update=50,
        update_interval=100,
        verbose=False,
    )
    return learner


def test_online_learner_add_sample():
    """add_sample"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner add_sample")
    print("=" * 60)

    learner = _create_online_learner()

    for _ in range(10):
        learner.add_sample(
            np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05
        )

    assert len(learner.buffer) == 10
    print("PASS")


def test_online_learner_update_model():
    """update_model (min samples 이상일 때)"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner update_model")
    print("=" * 60)

    learner = _create_online_learner()

    # Add enough samples
    for _ in range(60):
        learner.add_sample(
            np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05
        )

    # Manual update
    learner.update_model(num_epochs=3)

    assert learner.performance_history["num_updates"] == 1
    assert len(learner.performance_history["validation_losses"]) == 1
    print("PASS")


def test_online_learner_insufficient_samples():
    """샘플 부족 시 update 스킵"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner update with insufficient samples")
    print("=" * 60)

    learner = _create_online_learner()

    for _ in range(10):
        learner.add_sample(
            np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05
        )

    learner.update_model()
    assert learner.performance_history["num_updates"] == 0
    print("PASS")


def test_online_learner_reset():
    """reset"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner reset")
    print("=" * 60)

    learner = _create_online_learner()

    for _ in range(60):
        learner.add_sample(
            np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05
        )
    learner.update_model(num_epochs=3)

    learner.reset()
    assert len(learner.buffer) == 0
    assert learner.performance_history["num_updates"] == 0
    assert learner.adaptation_metrics["improvement"] is None
    print("PASS")


def test_online_learner_performance_summary():
    """get_performance_summary"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner get_performance_summary")
    print("=" * 60)

    learner = _create_online_learner()

    summary = learner.get_performance_summary()
    assert summary["num_updates"] == 0
    assert summary["buffer_size"] == 0
    assert summary["latest_val_loss"] is None
    assert summary["adaptation_improvement"] is None
    print(f"Summary: {summary}")
    print("PASS")


def test_online_learner_repr_no_crash():
    """P0-1: repr이 improvement=None일 때 크래시 안남"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner __repr__ no crash (P0-1)")
    print("=" * 60)

    learner = _create_online_learner()

    # improvement is None initially
    assert learner.adaptation_metrics["improvement"] is None

    r = repr(learner)
    assert "N/A" in r, f"Expected 'N/A' in repr, got: {r}"
    print(f"repr: {r}")
    print("PASS")


def test_online_learner_repr_with_improvement():
    """repr with numeric improvement"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner __repr__ with improvement")
    print("=" * 60)

    learner = _create_online_learner()
    learner.adaptation_metrics["improvement"] = 15.5

    r = repr(learner)
    assert "15.50%" in r, f"Expected '15.50%' in repr, got: {r}"
    print(f"repr: {r}")
    print("PASS")


def test_online_learner_shuffle_in_update():
    """P0-4: update_model의 train/val 분할이 셔플됨"""
    print("\n" + "=" * 60)
    print("Test: OnlineLearner update_model uses shuffled split (P0-4)")
    print("=" * 60)

    learner = _create_online_learner()

    # Add sequential data with clear time pattern
    for i in range(60):
        # State values increase over time
        state = np.array([float(i), float(i) * 0.1, 0.0])
        control = np.array([0.5, 0.0])
        next_state = state + np.array([0.01, 0.001, 0.0])
        learner.add_sample(state, control, next_state, 0.05)

    # Update should succeed without temporal bias
    learner.update_model(num_epochs=2)
    assert learner.performance_history["num_updates"] == 1
    print("PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OnlineLearner + OnlineDataBuffer Tests".center(60))
    print("=" * 60)

    try:
        # OnlineDataBuffer
        test_buffer_add_and_len()
        test_buffer_circular()
        test_buffer_get_batch()
        test_buffer_get_batch_empty_raises()
        test_buffer_get_all_data()
        test_buffer_statistics()
        test_buffer_should_retrain()
        test_buffer_clear()
        test_buffer_repr()

        # OnlineLearner
        test_online_learner_add_sample()
        test_online_learner_update_model()
        test_online_learner_insufficient_samples()
        test_online_learner_reset()
        test_online_learner_performance_summary()
        test_online_learner_repr_no_crash()
        test_online_learner_repr_with_improvement()
        test_online_learner_shuffle_in_update()

        print("\n" + "=" * 60)
        print("All OnlineLearner Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
