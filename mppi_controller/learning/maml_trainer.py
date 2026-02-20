"""
FOMAML (First-Order MAML) 메타 학습 파이프라인

다양한 DynamicWorld 설정을 "태스크"로 샘플링하여
few-shot 적응에 최적화된 메타 파라미터를 학습.

Usage:
    trainer = MAMLTrainer(state_dim=3, control_dim=2)
    trainer.meta_train(n_iterations=500)
    trainer.save_meta_model("maml_meta_model.pth")
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel


class MAMLTrainer:
    """
    FOMAML 메타 학습 파이프라인.

    다양한 DynamicWorld 파라미터(c_v, c_omega)를 태스크로 샘플링하여
    inner-loop 적응 후 query loss가 최소화되도록 메타 파라미터를 업데이트.

    FOMAML: create_graph=False → 1차 근사, 2차 미분 불필요로 효율적.

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        hidden_dims: MLP hidden layer 차원
        inner_lr: inner-loop 학습률
        inner_steps: inner-loop gradient step 수
        meta_lr: outer-loop (메타) 학습률
        task_batch_size: 메타 배치당 태스크 수
        support_size: support set 크기
        query_size: query set 크기
        device: 'cpu' or 'cuda'
        save_dir: 모델 저장 경로
    """

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 2,
        hidden_dims: List[int] = None,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_lr: float = 1e-3,
        task_batch_size: int = 4,
        support_size: int = 50,
        query_size: int = 50,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    ):
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        self.task_batch_size = task_batch_size
        self.support_size = support_size
        self.query_size = query_size
        self.device = torch.device(device)
        self.save_dir = save_dir

        input_dim = state_dim + control_dim
        output_dim = state_dim

        self.model = DynamicsMLPModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            dropout_rate=0.0,
        ).to(self.device)

        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=meta_lr
        )

        self.norm_stats = None
        self.history = {"meta_loss": []}

        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def _sample_task(self) -> Dict:
        """랜덤 DynamicWorld 설정 생성 (c_v, c_omega 범위 내 균일 분포)."""
        c_v = np.random.uniform(0.1, 0.8)
        c_omega = np.random.uniform(0.1, 0.5)
        return {
            "c_v": c_v,
            "c_omega": c_omega,
            "k_v": 5.0,
            "k_omega": 5.0,
        }

    def _generate_task_data(
        self, task_params: Dict, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        DynamicWorld에서 데이터 생성 — 궤적 추종 패턴 + 랜덤 혼합.

        실제 MPPI 제어 상황을 모사하기 위해:
        - 50% forward driving (v > 0, 다양한 omega)
        - 30% curved driving (v > 0, large omega)
        - 20% pure random

        Returns:
            states: (n_samples, state_dim)
            controls: (n_samples, control_dim)
            next_states: (n_samples, state_dim)
        """
        # Lazy import to avoid circular dependency
        from examples.comparison.model_mismatch_comparison_demo import DynamicWorld
        from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

        world = DynamicWorld(
            c_v=task_params["c_v"],
            c_omega=task_params["c_omega"],
            k_v=task_params["k_v"],
            k_omega=task_params["k_omega"],
            process_noise_std=np.zeros(5),
        )

        states = []
        controls = []
        next_states = []

        dt = 0.05
        state_3d = np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-3, 3),
            np.random.uniform(-np.pi, np.pi),
        ])
        world.reset(state_3d)
        obs = state_3d.copy()

        for _ in range(n_samples):
            r = np.random.random()
            if r < 0.5:
                # Forward driving: v ∈ [0.2, 1.0], small omega
                control = np.array([
                    np.random.uniform(0.2, 1.0),
                    np.random.uniform(-0.3, 0.3),
                ])
            elif r < 0.8:
                # Curved driving: v ∈ [0.1, 0.8], larger omega
                control = np.array([
                    np.random.uniform(0.1, 0.8),
                    np.random.uniform(-1.0, 1.0),
                ])
            else:
                # Pure random
                control = np.array([
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0),
                ])

            states.append(obs.copy())
            controls.append(control.copy())

            obs = world.step(control, dt, add_noise=False)
            next_states.append(obs.copy())

            # 주기적 리셋 (다양한 상태 커버)
            if np.random.random() < 0.05:
                state_3d = np.array([
                    np.random.uniform(-3, 3),
                    np.random.uniform(-3, 3),
                    np.random.uniform(-np.pi, np.pi),
                ])
                world.reset(state_3d)
                obs = state_3d.copy()

        return np.array(states), np.array(controls), np.array(next_states)

    def _generate_task_data_5d(
        self, task_params: Dict, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        DynamicWorld에서 5D 데이터 생성 — world.get_full_state() 사용.

        5D state = [x, y, θ, v, ω]로 MAML이 속도/각속도를 관측하여
        관성/마찰 보정을 학습할 수 있도록 한다.

        Returns:
            states: (n_samples, 5)
            controls: (n_samples, 2)
            next_states: (n_samples, 5)
        """
        from examples.comparison.model_mismatch_comparison_demo import DynamicWorld

        world = DynamicWorld(
            c_v=task_params["c_v"],
            c_omega=task_params["c_omega"],
            k_v=task_params["k_v"],
            k_omega=task_params["k_omega"],
            process_noise_std=np.zeros(5),
        )

        states = []
        controls = []
        next_states = []

        dt = 0.05
        state_3d = np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-3, 3),
            np.random.uniform(-np.pi, np.pi),
        ])
        world.reset(state_3d)

        for _ in range(n_samples):
            r = np.random.random()
            if r < 0.5:
                control = np.array([
                    np.random.uniform(0.2, 1.0),
                    np.random.uniform(-0.3, 0.3),
                ])
            elif r < 0.8:
                control = np.array([
                    np.random.uniform(0.1, 0.8),
                    np.random.uniform(-1.0, 1.0),
                ])
            else:
                control = np.array([
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0),
                ])

            state_5d = world.get_full_state()
            states.append(state_5d.copy())
            controls.append(control.copy())

            world.step(control, dt, add_noise=False)
            next_state_5d = world.get_full_state()
            next_states.append(next_state_5d.copy())

            # 주기적 리셋
            if np.random.random() < 0.05:
                state_3d = np.array([
                    np.random.uniform(-3, 3),
                    np.random.uniform(-3, 3),
                    np.random.uniform(-np.pi, np.pi),
                ])
                world.reset(state_3d)

        return np.array(states), np.array(controls), np.array(next_states)

    def _compute_norm_stats(self, all_data: List[Tuple]) -> Dict[str, np.ndarray]:
        """전체 태스크 데이터에서 normalization 통계 계산."""
        all_states = []
        all_controls = []
        all_dots = []

        dt = 0.05
        for states, controls, next_states in all_data:
            dots = (next_states - states) / dt
            # Angle wrapping
            theta_diff = next_states[:, 2] - states[:, 2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            dots[:, 2] = theta_diff / dt

            all_states.append(states)
            all_controls.append(controls)
            all_dots.append(dots)

        all_states = np.concatenate(all_states, axis=0)
        all_controls = np.concatenate(all_controls, axis=0)
        all_dots = np.concatenate(all_dots, axis=0)

        return {
            "state_mean": np.mean(all_states, axis=0),
            "state_std": np.std(all_states, axis=0) + 1e-8,
            "control_mean": np.mean(all_controls, axis=0),
            "control_std": np.std(all_controls, axis=0) + 1e-8,
            "state_dot_mean": np.mean(all_dots, axis=0),
            "state_dot_std": np.std(all_dots, axis=0) + 1e-8,
        }

    def _prepare_batch(
        self, states, controls, next_states, dt=0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """데이터를 normalized tensor로 변환."""
        targets = (next_states - states) / dt
        # Angle wrapping
        theta_diff = next_states[:, 2] - states[:, 2]
        theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
        targets[:, 2] = theta_diff / dt

        if self.norm_stats is not None:
            state_norm = (states - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_norm = (controls - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
            targets_norm = (targets - self.norm_stats["state_dot_mean"]) / self.norm_stats["state_dot_std"]
        else:
            state_norm = states
            control_norm = controls
            targets_norm = targets

        inputs = np.concatenate([state_norm, control_norm], axis=1)
        inputs_t = torch.FloatTensor(inputs).to(self.device)
        targets_t = torch.FloatTensor(targets_norm).to(self.device)

        return inputs_t, targets_t

    def _inner_loop(
        self, support_inputs: torch.Tensor, support_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Inner loop: support set으로 K-step SGD.

        FOMAML: create_graph=False → 1차 근사.
        모델 파라미터를 복사하여 적응 후 반환.
        """
        # Copy model parameters
        adapted_params = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in self.model.state_dict().items()
        }

        for _ in range(self.inner_steps):
            # Forward with adapted params
            pred = self._forward_with_params(support_inputs, adapted_params)
            loss = F.mse_loss(pred, support_targets)

            # Compute gradients
            grads = torch.autograd.grad(
                loss, list(adapted_params.values()),
                create_graph=False,  # FOMAML: no second-order gradients
            )

            # SGD update
            adapted_params = {
                k: v - self.inner_lr * g
                for (k, v), g in zip(adapted_params.items(), grads)
            }

        return adapted_params

    def _forward_with_params(
        self, inputs: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """주어진 파라미터로 forward pass (functional forward)."""
        x = inputs
        layer_idx = 0
        param_keys = list(params.keys())

        # DynamicsMLPModel의 network는 Sequential(Linear, ReLU, ..., Linear)
        # 파라미터 이름: network.0.weight, network.0.bias, network.2.weight, ...
        for key in param_keys:
            if "weight" in key:
                bias_key = key.replace("weight", "bias")
                bias = params.get(bias_key)
                x = F.linear(x, params[key], bias)
            elif "bias" in key:
                continue  # bias는 weight와 함께 처리됨
            else:
                continue

            # ReLU after each hidden layer (not after last layer)
            # Check if next param is another weight (meaning there's an activation between)
            weight_keys = [k for k in param_keys if "weight" in k]
            current_weight_idx = weight_keys.index(key)
            if current_weight_idx < len(weight_keys) - 1:
                x = F.relu(x)

        return x

    def _compute_query_loss(
        self, query_inputs: torch.Tensor, query_targets: torch.Tensor,
        adapted_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """적응된 파라미터로 query set loss 계산."""
        pred = self._forward_with_params(query_inputs, adapted_params)
        return F.mse_loss(pred, query_targets)

    def meta_train(self, n_iterations: int = 1000, verbose: bool = True):
        """
        FOMAML 메타 학습 루프.

        1. norm_stats 계산을 위해 초기 데이터 수집
        2. 매 iteration:
           a. task_batch_size 개의 태스크 샘플
           b. 각 태스크에서 support/query 분할
           c. Inner loop (support) → 적응된 파라미터
           d. Query loss 계산
        3. Outer loop: 평균 query loss로 메타 파라미터 업데이트

        Args:
            n_iterations: 메타 학습 반복 횟수
            verbose: 진행 상황 출력
        """
        if verbose:
            print(f"\n  FOMAML Meta-Training")
            print(f"    Iterations: {n_iterations}")
            print(f"    Task batch size: {self.task_batch_size}")
            print(f"    Support/Query: {self.support_size}/{self.query_size}")
            print(f"    Inner LR: {self.inner_lr}, Steps: {self.inner_steps}")
            print(f"    Meta LR: {self.meta_lr}")

        # Step 1: Pre-collect data for norm_stats
        if verbose:
            print("    Computing normalization stats...")

        n_total = self.support_size + self.query_size
        gen_fn = self._generate_task_data_5d if self.state_dim == 5 else self._generate_task_data

        pre_data = []
        for _ in range(self.task_batch_size * 2):
            task = self._sample_task()
            data = gen_fn(task, n_total)
            pre_data.append(data)

        self.norm_stats = self._compute_norm_stats(pre_data)

        # Step 2: Meta-training loop
        if verbose:
            print("    Starting meta-training...")

        for iteration in range(n_iterations):
            total_query_loss = 0.0

            self.meta_optimizer.zero_grad()

            for _ in range(self.task_batch_size):
                task = self._sample_task()
                states, controls, next_states = gen_fn(
                    task, n_total
                )

                # Support/Query split
                support_s = states[:self.support_size]
                support_c = controls[:self.support_size]
                support_ns = next_states[:self.support_size]
                query_s = states[self.support_size:]
                query_c = controls[self.support_size:]
                query_ns = next_states[self.support_size:]

                support_inputs, support_targets = self._prepare_batch(
                    support_s, support_c, support_ns
                )
                query_inputs, query_targets = self._prepare_batch(
                    query_s, query_c, query_ns
                )

                # Inner loop
                adapted_params = self._inner_loop(support_inputs, support_targets)

                # Query loss — FOMAML: 적응된 파라미터로 직접 forward
                # 그래디언트는 meta model 파라미터로 흐름
                # FOMAML 근사: adapted_params에서의 loss gradient를
                # 원래 파라미터에 대한 gradient로 근사
                self.model.train()
                # Load adapted params temporarily
                original_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

                # Use adapted params for query evaluation
                adapted_state = {k: v.detach() for k, v in adapted_params.items()}
                self.model.load_state_dict(adapted_state)

                # Compute query loss with model's parameters (for meta gradient)
                pred = self.model(query_inputs)
                query_loss = F.mse_loss(pred, query_targets)
                query_loss.backward()

                # Restore original parameters (meta params)
                self.model.load_state_dict(original_state)

                total_query_loss += query_loss.item()

            # Meta update: average gradients across tasks
            # Scale gradients by 1/task_batch_size
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= self.task_batch_size

            self.meta_optimizer.step()

            avg_loss = total_query_loss / self.task_batch_size
            self.history["meta_loss"].append(avg_loss)

            if verbose and (iteration + 1) % 50 == 0:
                print(
                    f"    Iter {iteration + 1}/{n_iterations} | "
                    f"Meta Loss: {avg_loss:.6f}"
                )

        if verbose:
            final_loss = self.history["meta_loss"][-1] if self.history["meta_loss"] else float("nan")
            print(f"\n    Meta-training complete. Final loss: {final_loss:.6f}")

    def save_meta_model(self, filename: str = "maml_meta_model.pth"):
        """메타 모델 + norm_stats 저장."""
        filepath = os.path.join(self.save_dir, filename)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "state_dim": self.state_dim,
                "control_dim": self.control_dim,
                "hidden_dims": self.hidden_dims,
                "activation": "relu",
                "dropout_rate": 0.0,
            },
        }, filepath)

        print(f"  [MAMLTrainer] Meta model saved to {filepath}")

    def load_meta_model(self, filename: str = "maml_meta_model.pth"):
        """메타 모델 로드."""
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Meta model not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.norm_stats = checkpoint.get("norm_stats")
        self.history = checkpoint.get("history", {"meta_loss": []})

        self.model.eval()
        print(f"  [MAMLTrainer] Meta model loaded from {filepath}")
