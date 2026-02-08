#!/usr/bin/env python3
"""
온라인 학습 파이프라인

실시간 데이터 수집 및 모델 업데이트.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from collections import deque
import time
import os
from pathlib import Path


class OnlineDataBuffer:
    """
    온라인 데이터 순환 버퍼

    실시간 데이터 스트림을 관리하고, 학습을 위한 배치 샘플링 제공.

    Features:
        - 순환 버퍼 (FIFO)
        - 자동 통계 업데이트
        - 배치 샘플링
        - 트리거 기반 학습
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        buffer_size: int = 1000,
        batch_size: int = 64,
    ):
        """
        Args:
            state_dim: 상태 벡터 차원
            control_dim: 제어 벡터 차원
            buffer_size: 최대 버퍼 크기
            batch_size: 학습 배치 크기
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Circular buffers
        self.states = deque(maxlen=buffer_size)
        self.controls = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dt_values = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)

        # Running statistics
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.control_mean = np.zeros(control_dim)
        self.control_std = np.ones(control_dim)
        self.state_dot_mean = np.zeros(state_dim)
        self.state_dot_std = np.ones(state_dim)

        self.num_samples = 0

    def add(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
        dt: float,
        timestamp: Optional[float] = None,
    ):
        """
        새 샘플 추가

        Args:
            state: (nx,) 현재 상태
            control: (nu,) 제어 입력
            next_state: (nx,) 다음 상태
            dt: 시간 간격
            timestamp: 시간 스탬프 (None이면 현재 시간)
        """
        self.states.append(state.copy())
        self.controls.append(control.copy())
        self.next_states.append(next_state.copy())
        self.dt_values.append(dt)
        self.timestamps.append(timestamp or time.time())

        self.num_samples += 1

        # Update statistics periodically
        if self.num_samples % 100 == 0:
            self._update_statistics()

    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        랜덤 배치 샘플링

        Args:
            batch_size: 배치 크기 (None이면 self.batch_size)

        Returns:
            batch: {"states", "controls", "next_states", "state_dots", "dt"}
        """
        if len(self.states) == 0:
            raise ValueError("Buffer is empty!")

        batch_size = batch_size or self.batch_size
        batch_size = min(batch_size, len(self.states))

        # Random sampling
        indices = np.random.choice(len(self.states), batch_size, replace=False)

        states = np.array([self.states[i] for i in indices])
        controls = np.array([self.controls[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dt = np.array([self.dt_values[i] for i in indices])

        # Compute state_dot
        state_dots = (next_states - states) / dt[:, np.newaxis]

        return {
            "states": states,
            "controls": controls,
            "next_states": next_states,
            "state_dots": state_dots,
            "dt": dt,
        }

    def get_all_data(self) -> Dict[str, np.ndarray]:
        """모든 데이터 반환"""
        if len(self.states) == 0:
            raise ValueError("Buffer is empty!")

        states = np.array(list(self.states))
        controls = np.array(list(self.controls))
        next_states = np.array(list(self.next_states))
        dt = np.array(list(self.dt_values))

        state_dots = (next_states - states) / dt[:, np.newaxis]

        return {
            "states": states,
            "controls": controls,
            "next_states": next_states,
            "state_dots": state_dots,
            "dt": dt,
        }

    def _update_statistics(self):
        """통계 업데이트"""
        if len(self.states) < 10:
            return

        data = self.get_all_data()

        self.state_mean = np.mean(data["states"], axis=0)
        self.state_std = np.std(data["states"], axis=0) + 1e-6
        self.control_mean = np.mean(data["controls"], axis=0)
        self.control_std = np.std(data["controls"], axis=0) + 1e-6
        self.state_dot_mean = np.mean(data["state_dots"], axis=0)
        self.state_dot_std = np.std(data["state_dots"], axis=0) + 1e-6

    def get_statistics(self) -> Dict[str, np.ndarray]:
        """통계 반환"""
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "control_mean": self.control_mean,
            "control_std": self.control_std,
            "state_dot_mean": self.state_dot_mean,
            "state_dot_std": self.state_dot_std,
        }

    def should_retrain(
        self,
        min_samples: int = 100,
        retrain_interval: int = 500,
    ) -> bool:
        """
        재학습 필요 여부 판단

        Args:
            min_samples: 최소 샘플 수
            retrain_interval: 재학습 주기

        Returns:
            should_retrain: 재학습 필요 여부
        """
        if len(self.states) < min_samples:
            return False

        # 주기적 재학습
        if self.num_samples % retrain_interval == 0:
            return True

        return False

    def clear(self):
        """버퍼 초기화"""
        self.states.clear()
        self.controls.clear()
        self.next_states.clear()
        self.dt_values.clear()
        self.timestamps.clear()
        self.num_samples = 0

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return (
            f"OnlineDataBuffer("
            f"size={len(self.states)}/{self.buffer_size}, "
            f"batch_size={self.batch_size}, "
            f"total_samples={self.num_samples})"
        )


class OnlineLearner:
    """
    온라인 학습 관리자

    실시간 데이터 수집 및 모델 업데이트 관리.

    Features:
        - 온라인 데이터 버퍼 관리
        - Neural Network Fine-tuning
        - GP Online Update
        - 학습 스케줄링
        - 성능 모니터링
    """

    def __init__(
        self,
        model,  # NeuralDynamics or GaussianProcessDynamics
        trainer,  # NeuralNetworkTrainer or GaussianProcessTrainer
        buffer_size: int = 1000,
        batch_size: int = 64,
        min_samples_for_update: int = 100,
        update_interval: int = 500,
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        max_checkpoints: int = 10,
    ):
        """
        Args:
            model: 학습 모델 (Neural or GP)
            trainer: 학습 파이프라인
            buffer_size: 데이터 버퍼 크기
            batch_size: 학습 배치 크기
            min_samples_for_update: 업데이트 최소 샘플 수
            update_interval: 업데이트 주기 (샘플 수)
            verbose: 로그 출력
            checkpoint_dir: 체크포인트 저장 디렉토리 (None이면 비활성화)
            max_checkpoints: 최대 체크포인트 보관 수
        """
        self.model = model
        self.trainer = trainer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_samples_for_update = min_samples_for_update
        self.update_interval = update_interval
        self.verbose = verbose

        # Checkpoint versioning
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_versions: List[Dict] = []  # [{version, path, val_loss, timestamp}]
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Online data buffer
        self.buffer = OnlineDataBuffer(
            state_dim=model.state_dim,
            control_dim=model.control_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # Performance monitoring
        self.performance_history = {
            "num_updates": 0,
            "update_timestamps": [],
            "validation_losses": [],
            "buffer_sizes": [],
        }

        # Adaptation metrics
        self.adaptation_metrics = {
            "initial_error": None,
            "current_error": None,
            "improvement": None,
        }

    def add_sample(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
        dt: float,
    ):
        """
        새 샘플 추가

        Args:
            state: (nx,) 현재 상태
            control: (nu,) 제어 입력
            next_state: (nx,) 다음 상태
            dt: 시간 간격
        """
        self.buffer.add(state, control, next_state, dt)

        # Check if update is needed
        if self.buffer.should_retrain(
            min_samples=self.min_samples_for_update,
            retrain_interval=self.update_interval,
        ):
            self.update_model()

    def update_model(self, num_epochs: int = 10):
        """
        모델 업데이트 (fine-tuning)

        Args:
            num_epochs: Fine-tuning epochs
        """
        if len(self.buffer) < self.min_samples_for_update:
            if self.verbose:
                print(f"[OnlineLearner] Insufficient samples: {len(self.buffer)}")
            return

        if self.verbose:
            print(f"\n[OnlineLearner] Updating model...")
            print(f"  Buffer size: {len(self.buffer)}")
            print(f"  Total samples: {self.buffer.num_samples}")

        # Get data
        data = self.buffer.get_all_data()

        # Split train/val (80/20) with shuffle to avoid temporal bias
        n_samples = len(data["states"])
        indices = np.random.permutation(n_samples)
        num_train = int(n_samples * 0.8)

        train_idx = indices[:num_train]
        val_idx = indices[num_train:]

        train_states = data["states"][train_idx]
        train_controls = data["controls"][train_idx]
        train_state_dots = data["state_dots"][train_idx]

        val_states = data["states"][val_idx]
        val_controls = data["controls"][val_idx]
        val_state_dots = data["state_dots"][val_idx]

        # Prepare inputs/targets
        train_inputs = np.concatenate([train_states, train_controls], axis=1)
        train_targets = train_state_dots
        val_inputs = np.concatenate([val_states, val_controls], axis=1)
        val_targets = val_state_dots

        # Get normalization stats
        norm_stats = self.buffer.get_statistics()

        # Fine-tuning
        start_time = time.time()

        if hasattr(self.trainer, "train"):  # Neural or GP trainer
            # Neural Network or GP
            try:
                history = self.trainer.train(
                    train_inputs,
                    train_targets,
                    val_inputs,
                    val_targets,
                    norm_stats,
                    epochs=num_epochs,  # Short fine-tuning
                    batch_size=self.batch_size,
                    early_stopping_patience=5,
                    verbose=False,
                )

                val_loss = history["val_loss"][-1] if len(history["val_loss"]) > 0 else 0.0

            except Exception as e:
                if self.verbose:
                    print(f"[OnlineLearner] Update failed: {e}")
                return
        else:
            if self.verbose:
                print("[OnlineLearner] Unsupported trainer type")
            return

        update_time = time.time() - start_time

        # Check for performance degradation and rollback if needed
        prev_val_loss = (
            self.performance_history["validation_losses"][-1]
            if len(self.performance_history["validation_losses"]) > 0
            else None
        )

        # Update performance history
        self.performance_history["num_updates"] += 1
        self.performance_history["update_timestamps"].append(time.time())
        self.performance_history["validation_losses"].append(val_loss)
        self.performance_history["buffer_sizes"].append(len(self.buffer))

        # Checkpoint versioning
        if self.checkpoint_dir is not None:
            version = self.performance_history["num_updates"]

            # Check for significant degradation (>50% worse than best)
            best_checkpoint = self._get_best_checkpoint()
            if (
                best_checkpoint is not None
                and val_loss > best_checkpoint["val_loss"] * 1.5
            ):
                if self.verbose:
                    print(
                        f"  Performance degradation detected! "
                        f"(val_loss={val_loss:.6f} vs best={best_checkpoint['val_loss']:.6f})"
                    )
                    print(f"  Rolling back to v{best_checkpoint['version']}...")
                self.rollback(best_checkpoint["version"])
                return

            # Save checkpoint
            self._save_checkpoint(version, val_loss)

        if self.verbose:
            print(f"  Update completed in {update_time:.2f}s")
            print(f"  Val loss: {val_loss:.6f}")
            print(f"  Total updates: {self.performance_history['num_updates']}")

    def compute_adaptation_error(
        self,
        test_states: np.ndarray,
        test_controls: np.ndarray,
        test_targets: np.ndarray,
    ) -> float:
        """
        적응 오차 계산

        Args:
            test_states: (N, nx) 테스트 상태
            test_controls: (N, nu) 테스트 제어
            test_targets: (N, nx) 실제 state_dot

        Returns:
            rmse: RMSE
        """
        # Predict
        predictions = self.trainer.predict(
            test_states,
            test_controls,
            denormalize=True,
        )

        if isinstance(predictions, tuple):  # GP returns (mean, std)
            predictions = predictions[0]

        # RMSE
        errors = predictions - test_targets
        rmse = np.sqrt(np.mean(errors ** 2))

        return rmse

    def monitor_adaptation(
        self,
        test_data: Dict[str, np.ndarray],
    ):
        """
        적응 성능 모니터링

        Args:
            test_data: {"states", "controls", "state_dots"}
        """
        rmse = self.compute_adaptation_error(
            test_data["states"],
            test_data["controls"],
            test_data["state_dots"],
        )

        # Update metrics
        if self.adaptation_metrics["initial_error"] is None:
            self.adaptation_metrics["initial_error"] = rmse

        self.adaptation_metrics["current_error"] = rmse
        self.adaptation_metrics["improvement"] = (
            self.adaptation_metrics["initial_error"] - rmse
        ) / self.adaptation_metrics["initial_error"] * 100

        if self.verbose:
            print(f"\n[Adaptation Monitoring]")
            print(f"  Initial error: {self.adaptation_metrics['initial_error']:.6f}")
            print(f"  Current error: {rmse:.6f}")
            print(f"  Improvement: {self.adaptation_metrics['improvement']:.2f}%")

    def get_performance_summary(self) -> Dict:
        """성능 요약"""
        return {
            "num_updates": self.performance_history["num_updates"],
            "buffer_size": len(self.buffer),
            "total_samples": self.buffer.num_samples,
            "latest_val_loss": self.performance_history["validation_losses"][-1]
            if len(self.performance_history["validation_losses"]) > 0
            else None,
            "adaptation_improvement": self.adaptation_metrics["improvement"],
        }

    def _save_checkpoint(self, version: int, val_loss: float):
        """체크포인트 저장"""
        if self.checkpoint_dir is None:
            return

        filename = f"model_v{version}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        if hasattr(self.trainer, "save_model"):
            self.trainer.save_model(filename)

        self.checkpoint_versions.append({
            "version": version,
            "path": filepath,
            "val_loss": val_loss,
            "timestamp": time.time(),
        })

        # Prune old checkpoints (keep best + recent max_checkpoints)
        if len(self.checkpoint_versions) > self.max_checkpoints:
            best = self._get_best_checkpoint()
            # Sort by version (oldest first), remove oldest non-best
            to_remove = []
            for cp in self.checkpoint_versions[:-self.max_checkpoints]:
                if cp["version"] != best["version"]:
                    to_remove.append(cp)

            for cp in to_remove:
                if os.path.exists(cp["path"]):
                    os.remove(cp["path"])
                self.checkpoint_versions.remove(cp)

        if self.verbose:
            print(f"  Checkpoint saved: {filename} (val_loss={val_loss:.6f})")

    def _get_best_checkpoint(self) -> Optional[Dict]:
        """최적 체크포인트 반환 (최저 val_loss)"""
        if not self.checkpoint_versions:
            return None
        return min(self.checkpoint_versions, key=lambda c: c["val_loss"])

    def rollback(self, version: Optional[int] = None):
        """
        이전 체크포인트로 롤백

        Args:
            version: 롤백할 버전 (None이면 최적 체크포인트)
        """
        if not self.checkpoint_versions:
            if self.verbose:
                print("[OnlineLearner] No checkpoints available for rollback")
            return False

        if version is None:
            target = self._get_best_checkpoint()
        else:
            targets = [c for c in self.checkpoint_versions if c["version"] == version]
            if not targets:
                if self.verbose:
                    print(f"[OnlineLearner] Checkpoint v{version} not found")
                return False
            target = targets[0]

        if not os.path.exists(target["path"]):
            if self.verbose:
                print(f"[OnlineLearner] Checkpoint file not found: {target['path']}")
            return False

        # Load checkpoint
        filename = os.path.basename(target["path"])
        if hasattr(self.trainer, "load_model"):
            self.trainer.load_model(filename)

        if self.verbose:
            print(
                f"[OnlineLearner] Rolled back to v{target['version']} "
                f"(val_loss={target['val_loss']:.6f})"
            )
        return True

    def get_checkpoint_history(self) -> List[Dict]:
        """체크포인트 히스토리 반환"""
        return [
            {
                "version": cp["version"],
                "val_loss": cp["val_loss"],
                "timestamp": cp["timestamp"],
                "is_best": cp == self._get_best_checkpoint(),
            }
            for cp in self.checkpoint_versions
        ]

    def reset(self):
        """리셋"""
        self.buffer.clear()
        self.performance_history = {
            "num_updates": 0,
            "update_timestamps": [],
            "validation_losses": [],
            "buffer_sizes": [],
        }
        self.adaptation_metrics = {
            "initial_error": None,
            "current_error": None,
            "improvement": None,
        }
        self.checkpoint_versions = []

    def __repr__(self) -> str:
        improvement = self.adaptation_metrics['improvement']
        improvement_str = f"{improvement:.2f}%" if improvement is not None else "N/A"
        return (
            f"OnlineLearner("
            f"buffer={len(self.buffer)}/{self.buffer_size}, "
            f"updates={self.performance_history['num_updates']}, "
            f"improvement={improvement_str})"
        )
