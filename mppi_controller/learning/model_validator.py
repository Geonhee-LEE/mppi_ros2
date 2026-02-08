"""
모델 검증 프레임워크

학습된 동역학 모델의 성능을 평가하고 비교하는 도구.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple


class ModelValidator:
    """
    학습 모델 검증 프레임워크

    Metrics:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - R²: Coefficient of Determination
        - Per-dimension error: 차원별 오차 분석
        - Rollout error: 다단계 예측 누적 오차

    사용 예시:
        validator = ModelValidator()

        # 단일 모델 평가
        metrics = validator.evaluate(model, test_states, test_controls, test_targets)
        print(f"RMSE: {metrics['rmse']:.6f}")

        # 여러 모델 비교
        results = validator.compare(
            {"Neural": nn_model, "GP": gp_model, "Physics": physics_model},
            test_states, test_controls, test_targets
        )
        validator.print_comparison(results)
    """

    def evaluate(
        self,
        predict_fn: Callable,
        test_states: np.ndarray,
        test_controls: np.ndarray,
        test_targets: np.ndarray,
    ) -> Dict:
        """
        모델 평가

        Args:
            predict_fn: (states, controls) → predictions
                        RobotModel.forward_dynamics 또는 Trainer.predict
            test_states: (N, nx) 테스트 상태
            test_controls: (N, nu) 테스트 제어
            test_targets: (N, nx) 실제 state_dot

        Returns:
            metrics: {rmse, mae, r2, per_dim_rmse, per_dim_mae, per_dim_r2,
                      max_error, predictions}
        """
        predictions = predict_fn(test_states, test_controls)

        # 튜플 반환 시 (mean, std) → mean만 사용
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        errors = predictions - test_targets  # (N, nx)

        # 전체 메트릭
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        # R² (결정 계수)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((test_targets - np.mean(test_targets, axis=0)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

        # 차원별 메트릭
        nx = test_targets.shape[1]
        per_dim_rmse = np.zeros(nx)
        per_dim_mae = np.zeros(nx)
        per_dim_r2 = np.zeros(nx)

        for d in range(nx):
            per_dim_rmse[d] = np.sqrt(np.mean(errors[:, d] ** 2))
            per_dim_mae[d] = np.mean(np.abs(errors[:, d]))
            ss_res_d = np.sum(errors[:, d] ** 2)
            ss_tot_d = np.sum((test_targets[:, d] - np.mean(test_targets[:, d])) ** 2)
            per_dim_r2[d] = 1.0 - ss_res_d / (ss_tot_d + 1e-12)

        # 최대 오차
        max_error = float(np.max(np.abs(errors)))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "per_dim_rmse": per_dim_rmse,
            "per_dim_mae": per_dim_mae,
            "per_dim_r2": per_dim_r2,
            "max_error": max_error,
            "predictions": predictions,
        }

    def evaluate_rollout(
        self,
        model,
        initial_states: np.ndarray,
        control_sequences: np.ndarray,
        true_trajectories: np.ndarray,
        dt: float,
    ) -> Dict:
        """
        롤아웃 오차 평가

        초기 상태에서 다단계 예측을 수행하고 실제 궤적과 비교.

        Args:
            model: RobotModel (step() 메서드 필요)
            initial_states: (M, nx) M개의 초기 상태
            control_sequences: (M, T, nu) M개의 제어 시퀀스
            true_trajectories: (M, T+1, nx) M개의 실제 궤적
            dt: 시간 간격

        Returns:
            metrics: {mean_rollout_rmse, per_step_rmse, worst_case_rmse}
        """
        M, T, _ = control_sequences.shape
        nx = initial_states.shape[1]

        per_step_errors = np.zeros(T + 1)
        all_pred_trajs = np.zeros_like(true_trajectories)

        for m in range(M):
            state = initial_states[m].copy()
            all_pred_trajs[m, 0] = state

            for t in range(T):
                state = model.step(state, control_sequences[m, t], dt)
                if hasattr(model, 'normalize_state'):
                    state = model.normalize_state(state)
                all_pred_trajs[m, t + 1] = state

        # Per-step RMSE
        for t in range(T + 1):
            errors = all_pred_trajs[:, t] - true_trajectories[:, t]
            per_step_errors[t] = np.sqrt(np.mean(errors ** 2))

        mean_rollout_rmse = float(np.mean(per_step_errors))
        worst_case_rmse = float(np.max(per_step_errors))

        return {
            "mean_rollout_rmse": mean_rollout_rmse,
            "per_step_rmse": per_step_errors,
            "worst_case_rmse": worst_case_rmse,
            "predicted_trajectories": all_pred_trajs,
        }

    def compare(
        self,
        models: Dict[str, Callable],
        test_states: np.ndarray,
        test_controls: np.ndarray,
        test_targets: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        여러 모델 비교 평가

        Args:
            models: {name: predict_fn} 딕셔너리
            test_states: (N, nx)
            test_controls: (N, nu)
            test_targets: (N, nx)

        Returns:
            results: {name: metrics} 딕셔너리
        """
        results = {}
        for name, predict_fn in models.items():
            results[name] = self.evaluate(
                predict_fn, test_states, test_controls, test_targets
            )
        return results

    @staticmethod
    def print_comparison(results: Dict[str, Dict]):
        """비교 결과 테이블 출력"""
        header = f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MaxErr':>10}"
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for name, metrics in results.items():
            print(
                f"{name:<20} "
                f"{metrics['rmse']:>10.6f} "
                f"{metrics['mae']:>10.6f} "
                f"{metrics['r2']:>10.4f} "
                f"{metrics['max_error']:>10.6f}"
            )
        print("=" * len(header))

    @staticmethod
    def print_per_dim(metrics: Dict, dim_names: Optional[List[str]] = None):
        """차원별 결과 출력"""
        nx = len(metrics["per_dim_rmse"])
        if dim_names is None:
            dim_names = [f"dim_{i}" for i in range(nx)]

        header = f"{'Dim':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10}"
        print(header)
        print("-" * len(header))

        for d in range(nx):
            print(
                f"{dim_names[d]:<12} "
                f"{metrics['per_dim_rmse'][d]:>10.6f} "
                f"{metrics['per_dim_mae'][d]:>10.6f} "
                f"{metrics['per_dim_r2'][d]:>10.4f}"
            )
