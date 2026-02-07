#!/usr/bin/env python3
"""
데이터 수집 파이프라인

시뮬레이션 또는 실제 로봇에서 동역학 학습 데이터 수집.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import pickle
import os
from pathlib import Path


class DataCollector:
    """
    동역학 학습 데이터 수집기

    데이터 형식:
        - states: (N, nx) - 현재 상태
        - controls: (N, nu) - 제어 입력
        - next_states: (N, nx) - 다음 상태
        - dt: float - 시간 간격
        - state_dots: (N, nx) - 상태 미분 (자동 계산)

    사용 예시:
        collector = DataCollector(state_dim=3, control_dim=2)

        for episode in range(100):
            state = env.reset()
            for t in range(100):
                control = controller.compute_control(state)
                next_state = env.step(control)
                collector.add_sample(state, control, next_state, dt=0.05)
                state = next_state

        collector.save("dynamics_data.pkl")
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        max_samples: int = 100000,
        save_dir: str = "data/learned_models",
    ):
        """
        Args:
            state_dim: 상태 벡터 차원
            control_dim: 제어 벡터 차원
            max_samples: 최대 샘플 수 (메모리 제한)
            save_dir: 데이터 저장 디렉토리
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.max_samples = max_samples
        self.save_dir = save_dir

        # 데이터 버퍼
        self.states: List[np.ndarray] = []
        self.controls: List[np.ndarray] = []
        self.next_states: List[np.ndarray] = []
        self.dt_values: List[float] = []

        # 메타데이터
        self.metadata = {
            "state_dim": state_dim,
            "control_dim": control_dim,
            "num_samples": 0,
            "episodes": [],
        }

        # 현재 에피소드
        self.current_episode = {
            "states": [],
            "controls": [],
            "next_states": [],
            "dt_values": [],
        }

        # 저장 디렉토리 생성
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def add_sample(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
        dt: float,
    ):
        """
        단일 샘플 추가

        Args:
            state: (nx,) 현재 상태
            control: (nu,) 제어 입력
            next_state: (nx,) 다음 상태
            dt: 시간 간격
        """
        assert state.shape == (self.state_dim,), f"State shape mismatch: {state.shape} vs {(self.state_dim,)}"
        assert control.shape == (self.control_dim,), f"Control shape mismatch: {control.shape} vs {(self.control_dim,)}"
        assert next_state.shape == (self.state_dim,), f"Next state shape mismatch: {next_state.shape} vs {(self.state_dim,)}"

        # 현재 에피소드에 추가
        self.current_episode["states"].append(state.copy())
        self.current_episode["controls"].append(control.copy())
        self.current_episode["next_states"].append(next_state.copy())
        self.current_episode["dt_values"].append(dt)

    def end_episode(self):
        """현재 에피소드 종료 및 저장"""
        if len(self.current_episode["states"]) == 0:
            return

        # 에피소드 데이터 추가
        self.states.extend(self.current_episode["states"])
        self.controls.extend(self.current_episode["controls"])
        self.next_states.extend(self.current_episode["next_states"])
        self.dt_values.extend(self.current_episode["dt_values"])

        # 메타데이터 업데이트
        self.metadata["episodes"].append({
            "length": len(self.current_episode["states"]),
            "total_samples": len(self.states),
        })
        self.metadata["num_samples"] = len(self.states)

        # 현재 에피소드 초기화
        self.current_episode = {
            "states": [],
            "controls": [],
            "next_states": [],
            "dt_values": [],
        }

        # 최대 샘플 수 체크
        if len(self.states) > self.max_samples:
            print(f"[DataCollector] Warning: Exceeded max_samples ({self.max_samples}). Truncating oldest data.")
            self._truncate_to_max()

    def _truncate_to_max(self):
        """최대 샘플 수로 자르기 (FIFO)"""
        excess = len(self.states) - self.max_samples
        if excess > 0:
            self.states = self.states[excess:]
            self.controls = self.controls[excess:]
            self.next_states = self.next_states[excess:]
            self.dt_values = self.dt_values[excess:]
            self.metadata["num_samples"] = len(self.states)

    def get_data(self) -> Dict[str, np.ndarray]:
        """
        수집된 데이터 반환

        Returns:
            dict:
                - states: (N, nx)
                - controls: (N, nu)
                - next_states: (N, nx)
                - state_dots: (N, nx) - 유한차분으로 계산
                - dt: (N,)
        """
        if len(self.states) == 0:
            raise ValueError("No data collected yet!")

        states = np.array(self.states)
        controls = np.array(self.controls)
        next_states = np.array(self.next_states)
        dt = np.array(self.dt_values)

        # state_dot 계산 (유한차분)
        state_dots = (next_states - states) / dt[:, np.newaxis]

        return {
            "states": states,
            "controls": controls,
            "next_states": next_states,
            "state_dots": state_dots,
            "dt": dt,
        }

    def get_statistics(self) -> Dict:
        """데이터 통계"""
        if len(self.states) == 0:
            return {"num_samples": 0}

        data = self.get_data()

        return {
            "num_samples": len(self.states),
            "num_episodes": len(self.metadata["episodes"]),
            "state_mean": np.mean(data["states"], axis=0),
            "state_std": np.std(data["states"], axis=0),
            "control_mean": np.mean(data["controls"], axis=0),
            "control_std": np.std(data["controls"], axis=0),
            "state_dot_mean": np.mean(data["state_dots"], axis=0),
            "state_dot_std": np.std(data["state_dots"], axis=0),
        }

    def save(self, filename: str):
        """
        데이터 저장

        Args:
            filename: 저장 파일명 (.pkl)
        """
        if len(self.states) == 0:
            print("[DataCollector] Warning: No data to save!")
            return

        filepath = os.path.join(self.save_dir, filename)

        # 데이터 준비
        data = self.get_data()
        data["metadata"] = self.metadata

        # 저장
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"[DataCollector] Saved {len(self.states)} samples to {filepath}")
        print(f"  Episodes: {len(self.metadata['episodes'])}")
        print(f"  State shape: {data['states'].shape}")
        print(f"  Control shape: {data['controls'].shape}")

    def load(self, filename: str):
        """
        데이터 로드

        Args:
            filename: 로드 파일명 (.pkl)
        """
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # 데이터 복원
        self.states = data["states"].tolist()
        self.controls = data["controls"].tolist()
        self.next_states = data["next_states"].tolist()
        self.dt_values = data["dt"].tolist()
        self.metadata = data.get("metadata", {"num_samples": len(self.states)})

        print(f"[DataCollector] Loaded {len(self.states)} samples from {filepath}")

    def clear(self):
        """모든 데이터 초기화"""
        self.states = []
        self.controls = []
        self.next_states = []
        self.dt_values = []
        self.metadata["num_samples"] = 0
        self.metadata["episodes"] = []
        self.current_episode = {
            "states": [],
            "controls": [],
            "next_states": [],
            "dt_values": [],
        }

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return (
            f"DataCollector("
            f"samples={len(self.states)}, "
            f"episodes={len(self.metadata['episodes'])}, "
            f"state_dim={self.state_dim}, "
            f"control_dim={self.control_dim})"
        )


class DynamicsDataset:
    """
    동역학 학습용 데이터셋

    학습/검증 분할, 정규화, 배치 샘플링 지원.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        train_ratio: float = 0.8,
        normalize: bool = True,
        shuffle: bool = True,
    ):
        """
        Args:
            data: DataCollector.get_data() 결과
            train_ratio: 학습 데이터 비율
            normalize: 데이터 정규화 여부
            shuffle: 데이터 셔플 여부
        """
        self.states = data["states"]
        self.controls = data["controls"]
        self.state_dots = data["state_dots"]
        self.normalize = normalize

        # 셔플
        if shuffle:
            indices = np.random.permutation(len(self.states))
            self.states = self.states[indices]
            self.controls = self.controls[indices]
            self.state_dots = self.state_dots[indices]

        # Train/Val 분할
        num_train = int(len(self.states) * train_ratio)
        self.train_states = self.states[:num_train]
        self.train_controls = self.controls[:num_train]
        self.train_state_dots = self.state_dots[:num_train]

        self.val_states = self.states[num_train:]
        self.val_controls = self.controls[num_train:]
        self.val_state_dots = self.state_dots[num_train:]

        # 정규화 통계 계산 (학습 데이터 기준)
        if normalize:
            self.state_mean = np.mean(self.train_states, axis=0)
            self.state_std = np.std(self.train_states, axis=0) + 1e-6
            self.control_mean = np.mean(self.train_controls, axis=0)
            self.control_std = np.std(self.train_controls, axis=0) + 1e-6
            self.state_dot_mean = np.mean(self.train_state_dots, axis=0)
            self.state_dot_std = np.std(self.train_state_dots, axis=0) + 1e-6

            # 정규화 적용
            self._normalize_data()
        else:
            self.state_mean = np.zeros(self.states.shape[1])
            self.state_std = np.ones(self.states.shape[1])
            self.control_mean = np.zeros(self.controls.shape[1])
            self.control_std = np.ones(self.controls.shape[1])
            self.state_dot_mean = np.zeros(self.state_dots.shape[1])
            self.state_dot_std = np.ones(self.state_dots.shape[1])

    def _normalize_data(self):
        """데이터 정규화"""
        self.train_states = (self.train_states - self.state_mean) / self.state_std
        self.train_controls = (self.train_controls - self.control_mean) / self.control_std
        self.train_state_dots = (self.train_state_dots - self.state_dot_mean) / self.state_dot_std

        self.val_states = (self.val_states - self.state_mean) / self.state_std
        self.val_controls = (self.val_controls - self.control_mean) / self.control_std
        self.val_state_dots = (self.val_state_dots - self.state_dot_mean) / self.state_dot_std

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터 반환

        Returns:
            inputs: (N_train, nx + nu) - [state, control] concatenated
            targets: (N_train, nx) - state_dot
        """
        inputs = np.concatenate([self.train_states, self.train_controls], axis=1)
        targets = self.train_state_dots
        return inputs, targets

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """검증 데이터 반환"""
        inputs = np.concatenate([self.val_states, self.val_controls], axis=1)
        targets = self.val_state_dots
        return inputs, targets

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """정규화 통계 반환 (추론 시 필요)"""
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "control_mean": self.control_mean,
            "control_std": self.control_std,
            "state_dot_mean": self.state_dot_mean,
            "state_dot_std": self.state_dot_std,
        }

    def __repr__(self) -> str:
        return (
            f"DynamicsDataset("
            f"train={len(self.train_states)}, "
            f"val={len(self.val_states)}, "
            f"normalized={self.normalize})"
        )
