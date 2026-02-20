"""
Reptile 메타 학습 파이프라인

FOMAML의 대안으로 더 간단하고 안정적인 메타 학습 알고리즘.
태스크별 K-step SGD 후 메타 파라미터를 적응된 파라미터 방향으로 이동.

알고리즘:
    for each iteration:
        θ' = θ  (copy)
        for each task:
            θ_adapted = SGD(θ', task_data, K steps)
            θ' += ε * (θ_adapted - θ')  (interpolate)
        θ = θ'

Usage:
    trainer = ReptileTrainer(state_dim=5, control_dim=2)
    trainer.meta_train(n_iterations=500)
    trainer.save_meta_model("reptile_meta_model.pth")
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
from typing import Dict, List, Tuple
from pathlib import Path

from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel
from mppi_controller.learning.maml_trainer import MAMLTrainer


class ReptileTrainer(MAMLTrainer):
    """
    Reptile 메타 학습 파이프라인.

    MAMLTrainer를 상속하여 태스크 생성, 데이터 생성, norm_stats 등을 재사용.
    meta_train()만 Reptile 알고리즘으로 오버라이드.

    Reptile: meta_params += ε * (adapted_params - meta_params)
    FOMAML보다 구현이 간단하고 안정적.

    Args:
        epsilon: outer-loop interpolation rate (default: 0.1)
        기타 인자: MAMLTrainer와 동일
    """

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 2,
        hidden_dims: List[int] = None,
        inner_lr: float = 0.01,
        inner_steps: int = 10,
        meta_lr: float = 1e-3,
        epsilon: float = 0.1,
        task_batch_size: int = 4,
        support_size: int = 100,
        query_size: int = 0,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    ):
        super().__init__(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dims=hidden_dims,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            meta_lr=meta_lr,
            task_batch_size=task_batch_size,
            support_size=support_size,
            query_size=query_size,
            device=device,
            save_dir=save_dir,
        )
        self.epsilon = epsilon

    def meta_train(self, n_iterations: int = 1000, verbose: bool = True):
        """
        Reptile 메타 학습 루프.

        1. norm_stats 계산
        2. 매 iteration:
           a. 현재 메타 파라미터 저장
           b. task_batch_size 개의 태스크별로 K-step SGD
           c. 적응된 파라미터 평균과 메타 파라미터 interpolation
        """
        if verbose:
            print(f"\n  Reptile Meta-Training")
            print(f"    Iterations: {n_iterations}")
            print(f"    Task batch size: {self.task_batch_size}")
            print(f"    Support size: {self.support_size}")
            print(f"    Inner LR: {self.inner_lr}, Steps: {self.inner_steps}")
            print(f"    Epsilon: {self.epsilon}")

        # Step 1: Pre-collect data for norm_stats
        if verbose:
            print("    Computing normalization stats...")

        n_total = self.support_size + max(self.query_size, 1)
        gen_fn = self._generate_task_data_5d if self.state_dim == 5 else self._generate_task_data

        pre_data = []
        for _ in range(self.task_batch_size * 2):
            task = self._sample_task()
            data = gen_fn(task, n_total)
            pre_data.append(data)

        self.norm_stats = self._compute_norm_stats(pre_data)

        # Step 2: Reptile meta-training loop
        if verbose:
            print("    Starting meta-training...")

        for iteration in range(n_iterations):
            # 메타 파라미터 저장
            meta_state = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }

            # 태스크별 적응 → 적응된 파라미터 수집
            adapted_states = []
            total_loss = 0.0

            for _ in range(self.task_batch_size):
                task = self._sample_task()
                states, controls, next_states = gen_fn(task, n_total)

                support_s = states[:self.support_size]
                support_c = controls[:self.support_size]
                support_ns = next_states[:self.support_size]

                inputs, targets = self._prepare_batch(
                    support_s, support_c, support_ns
                )

                # 메타 파라미터에서 시작하여 K-step SGD
                self.model.load_state_dict(meta_state)
                self.model.train()
                optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.inner_lr
                )

                for _ in range(self.inner_steps):
                    optimizer.zero_grad()
                    pred = self.model(inputs)
                    loss = F.mse_loss(pred, targets)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                adapted_states.append({
                    k: v.clone() for k, v in self.model.state_dict().items()
                })

            # Reptile 업데이트: θ += ε * mean(θ_adapted - θ)
            with torch.no_grad():
                for key in meta_state:
                    # 평균 적응된 파라미터
                    adapted_avg = torch.stack([
                        s[key] for s in adapted_states
                    ]).mean(dim=0)
                    # Interpolation
                    meta_state[key] += self.epsilon * (adapted_avg - meta_state[key])

            self.model.load_state_dict(meta_state)

            avg_loss = total_loss / self.task_batch_size
            self.history["meta_loss"].append(avg_loss)

            if verbose and (iteration + 1) % 50 == 0:
                print(
                    f"    Iter {iteration + 1}/{n_iterations} | "
                    f"Meta Loss: {avg_loss:.6f}"
                )

        self.model.eval()
        if verbose:
            final_loss = self.history["meta_loss"][-1] if self.history["meta_loss"] else float("nan")
            print(f"\n    Reptile meta-training complete. Final loss: {final_loss:.6f}")
