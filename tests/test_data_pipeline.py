"""
DataCollector + DynamicsDataset 테스트 (P0-7)

Tests:
    - DataCollector: add_sample, end_episode, get_data 형상
    - DataCollector: save/load 왕복
    - DataCollector: truncation
    - DynamicsDataset: 정규화, train/val 분할
    - DynamicsDataset: get_normalization_stats
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset


def test_add_sample_and_get_data():
    """add_sample + end_episode + get_data 형상"""
    print("\n" + "=" * 60)
    print("Test: add_sample + end_episode + get_data")
    print("=" * 60)

    state_dim, control_dim = 3, 2
    collector = DataCollector(state_dim=state_dim, control_dim=control_dim)

    N = 50
    dt = 0.05
    for i in range(N):
        state = np.random.randn(state_dim)
        control = np.random.randn(control_dim)
        next_state = state + np.random.randn(state_dim) * 0.01
        collector.add_sample(state, control, next_state, dt)
    collector.end_episode()

    assert len(collector) == N, f"Expected {N}, got {len(collector)}"

    data = collector.get_data()
    assert data["states"].shape == (N, state_dim)
    assert data["controls"].shape == (N, control_dim)
    assert data["next_states"].shape == (N, state_dim)
    assert data["state_dots"].shape == (N, state_dim)
    assert data["dt"].shape == (N,)

    # state_dot = (next_state - state) / dt
    expected_dots = (data["next_states"] - data["states"]) / data["dt"][:, np.newaxis]
    np.testing.assert_allclose(data["state_dots"], expected_dots, atol=1e-10)

    print(f"Data shapes: states={data['states'].shape}, controls={data['controls'].shape}")
    print("PASS")


def test_multiple_episodes():
    """여러 에피소드 추가"""
    print("\n" + "=" * 60)
    print("Test: Multiple episodes")
    print("=" * 60)

    collector = DataCollector(state_dim=3, control_dim=2)

    ep_lengths = [10, 20, 30]
    for ep_len in ep_lengths:
        for _ in range(ep_len):
            collector.add_sample(
                np.random.randn(3),
                np.random.randn(2),
                np.random.randn(3),
                0.05,
            )
        collector.end_episode()

    assert len(collector) == sum(ep_lengths)
    assert len(collector.metadata["episodes"]) == 3
    print(f"Total samples: {len(collector)}, Episodes: {len(collector.metadata['episodes'])}")
    print("PASS")


def test_empty_episode_ignored():
    """빈 에피소드는 무시"""
    print("\n" + "=" * 60)
    print("Test: Empty episode ignored")
    print("=" * 60)

    collector = DataCollector(state_dim=3, control_dim=2)
    collector.end_episode()  # 빈 에피소드
    assert len(collector) == 0
    assert len(collector.metadata["episodes"]) == 0
    print("PASS")


def test_save_load_roundtrip():
    """save/load 왕복 테스트"""
    print("\n" + "=" * 60)
    print("Test: Save/Load roundtrip")
    print("=" * 60)

    save_dir = tempfile.mkdtemp()
    collector = DataCollector(state_dim=3, control_dim=2, save_dir=save_dir)

    np.random.seed(42)
    N = 30
    for _ in range(N):
        collector.add_sample(
            np.random.randn(3),
            np.random.randn(2),
            np.random.randn(3),
            0.05,
        )
    collector.end_episode()

    original_data = collector.get_data()
    collector.save("test_data.pkl")

    # Load into new collector
    collector2 = DataCollector(state_dim=3, control_dim=2, save_dir=save_dir)
    collector2.load("test_data.pkl")

    loaded_data = collector2.get_data()

    np.testing.assert_allclose(original_data["states"], loaded_data["states"], atol=1e-10)
    np.testing.assert_allclose(original_data["controls"], loaded_data["controls"], atol=1e-10)
    np.testing.assert_allclose(original_data["next_states"], loaded_data["next_states"], atol=1e-10)
    np.testing.assert_allclose(original_data["dt"], loaded_data["dt"], atol=1e-10)

    print("PASS")


def test_truncation():
    """max_samples 초과 시 truncation"""
    print("\n" + "=" * 60)
    print("Test: Truncation on max_samples exceeded")
    print("=" * 60)

    collector = DataCollector(state_dim=3, control_dim=2, max_samples=20)

    for _ in range(30):
        collector.add_sample(
            np.random.randn(3),
            np.random.randn(2),
            np.random.randn(3),
            0.05,
        )
    collector.end_episode()

    assert len(collector) == 20, f"Expected 20, got {len(collector)}"
    print("PASS")


def test_get_data_empty_raises():
    """데이터 없을 때 get_data 에러"""
    print("\n" + "=" * 60)
    print("Test: get_data on empty collector raises ValueError")
    print("=" * 60)

    collector = DataCollector(state_dim=3, control_dim=2)
    try:
        collector.get_data()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS")


def test_clear():
    """clear 테스트"""
    print("\n" + "=" * 60)
    print("Test: clear")
    print("=" * 60)

    collector = DataCollector(state_dim=3, control_dim=2)
    for _ in range(10):
        collector.add_sample(
            np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05
        )
    collector.end_episode()
    assert len(collector) == 10

    collector.clear()
    assert len(collector) == 0
    assert collector.metadata["num_samples"] == 0
    assert len(collector.metadata["episodes"]) == 0
    print("PASS")


def test_get_statistics():
    """통계 계산 테스트"""
    print("\n" + "=" * 60)
    print("Test: get_statistics")
    print("=" * 60)

    collector = DataCollector(state_dim=3, control_dim=2)
    for _ in range(20):
        collector.add_sample(
            np.random.randn(3), np.random.randn(2), np.random.randn(3), 0.05
        )
    collector.end_episode()

    stats = collector.get_statistics()
    assert stats["num_samples"] == 20
    assert stats["num_episodes"] == 1
    assert stats["state_mean"].shape == (3,)
    assert stats["state_std"].shape == (3,)
    assert stats["control_mean"].shape == (2,)
    assert stats["state_dot_mean"].shape == (3,)
    print("PASS")


def test_repr():
    """__repr__"""
    print("\n" + "=" * 60)
    print("Test: DataCollector __repr__")
    print("=" * 60)

    collector = DataCollector(state_dim=3, control_dim=2)
    r = repr(collector)
    assert "samples=0" in r
    assert "state_dim=3" in r
    print(f"repr: {r}")
    print("PASS")


# ========== DynamicsDataset Tests ==========


def _make_synthetic_data(N=100, state_dim=3, control_dim=2):
    """합성 데이터 생성"""
    np.random.seed(42)
    return {
        "states": np.random.randn(N, state_dim),
        "controls": np.random.randn(N, control_dim),
        "state_dots": np.random.randn(N, state_dim),
        "next_states": np.random.randn(N, state_dim),
        "dt": np.full(N, 0.05),
    }


def test_dataset_train_val_split():
    """학습/검증 분할 비율"""
    print("\n" + "=" * 60)
    print("Test: DynamicsDataset train/val split")
    print("=" * 60)

    data = _make_synthetic_data(N=100)
    dataset = DynamicsDataset(data, train_ratio=0.8, normalize=False, shuffle=False)

    assert len(dataset.train_states) == 80
    assert len(dataset.val_states) == 20

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()

    assert train_inputs.shape == (80, 5)  # 3 + 2
    assert train_targets.shape == (80, 3)
    assert val_inputs.shape == (20, 5)
    assert val_targets.shape == (20, 3)
    print("PASS")


def test_dataset_normalization():
    """정규화 적용 확인"""
    print("\n" + "=" * 60)
    print("Test: DynamicsDataset normalization")
    print("=" * 60)

    data = _make_synthetic_data(N=200)
    dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True, shuffle=False)

    # 정규화된 학습 데이터의 평균 ≈ 0, 표준편차 ≈ 1
    train_states_mean = np.mean(dataset.train_states, axis=0)
    train_states_std = np.std(dataset.train_states, axis=0)

    np.testing.assert_allclose(train_states_mean, 0.0, atol=0.1)
    np.testing.assert_allclose(train_states_std, 1.0, atol=0.2)

    print(f"Train states mean: {train_states_mean}")
    print(f"Train states std: {train_states_std}")
    print("PASS")


def test_dataset_normalization_stats():
    """정규화 통계 반환"""
    print("\n" + "=" * 60)
    print("Test: DynamicsDataset get_normalization_stats")
    print("=" * 60)

    data = _make_synthetic_data()
    dataset = DynamicsDataset(data, normalize=True)

    norm_stats = dataset.get_normalization_stats()
    assert "state_mean" in norm_stats
    assert "state_std" in norm_stats
    assert "control_mean" in norm_stats
    assert "control_std" in norm_stats
    assert "state_dot_mean" in norm_stats
    assert "state_dot_std" in norm_stats

    assert norm_stats["state_mean"].shape == (3,)
    assert norm_stats["state_std"].shape == (3,)
    assert norm_stats["control_mean"].shape == (2,)
    print("PASS")


def test_dataset_no_normalization():
    """정규화 비활성 시 원본 데이터 유지"""
    print("\n" + "=" * 60)
    print("Test: DynamicsDataset without normalization")
    print("=" * 60)

    data = _make_synthetic_data(N=100)
    dataset = DynamicsDataset(data, normalize=False, shuffle=False)

    # 원본 데이터와 동일
    np.testing.assert_allclose(
        dataset.train_states, data["states"][:80], atol=1e-10
    )
    print("PASS")


def test_dataset_shuffle():
    """셔플 동작 확인"""
    print("\n" + "=" * 60)
    print("Test: DynamicsDataset shuffle")
    print("=" * 60)

    data = _make_synthetic_data(N=100)

    dataset_no_shuffle = DynamicsDataset(data, normalize=False, shuffle=False)
    np.random.seed(123)
    dataset_shuffle = DynamicsDataset(data, normalize=False, shuffle=True)

    # 셔플 시 순서가 달라야 함
    is_different = not np.allclose(
        dataset_no_shuffle.train_states, dataset_shuffle.train_states
    )
    assert is_different, "Shuffled data should differ from unshuffled"
    print("PASS")


def test_dataset_repr():
    """DynamicsDataset __repr__"""
    print("\n" + "=" * 60)
    print("Test: DynamicsDataset __repr__")
    print("=" * 60)

    data = _make_synthetic_data()
    dataset = DynamicsDataset(data, train_ratio=0.8)
    r = repr(dataset)
    assert "train=80" in r
    assert "val=20" in r
    print(f"repr: {r}")
    print("PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DataCollector + DynamicsDataset Tests".center(60))
    print("=" * 60)

    try:
        # DataCollector
        test_add_sample_and_get_data()
        test_multiple_episodes()
        test_empty_episode_ignored()
        test_save_load_roundtrip()
        test_truncation()
        test_get_data_empty_raises()
        test_clear()
        test_get_statistics()
        test_repr()

        # DynamicsDataset
        test_dataset_train_val_split()
        test_dataset_normalization()
        test_dataset_normalization_stats()
        test_dataset_no_normalization()
        test_dataset_shuffle()
        test_dataset_repr()

        print("\n" + "=" * 60)
        print("All DataCollector + DynamicsDataset Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
