"""
PyTorch GPU 노이즈 샘플러

GPU에서 직접 가우시안 노이즈를 생성하여 CPU↔GPU 전송 제거.
"""

import torch
import numpy as np


class TorchGaussianSampler:
    """
    GPU 가우시안 노이즈 샘플러

    Args:
        sigma: (nu,) 표준편차 (numpy array 또는 리스트)
        device: torch device
    """

    def __init__(self, sigma, device="cuda"):
        self.device = torch.device(device)
        self.sigma = torch.tensor(
            np.asarray(sigma), device=self.device, dtype=torch.float32
        )

    def sample(self, U_shape, K):
        """
        GPU에서 가우시안 노이즈 생성

        Args:
            U_shape: (N, nu) 형태의 튜플 또는 numpy 배열의 shape
            K: 샘플 개수

        Returns:
            noise: (K, N, nu) torch tensor (GPU에 유지)
        """
        if hasattr(U_shape, "shape"):
            N, nu = U_shape.shape
        else:
            N, nu = U_shape

        noise = torch.randn(K, N, nu, device=self.device, dtype=torch.float32)
        noise = noise * self.sigma  # 브로드캐스트: (K, N, nu) * (nu,)
        return noise
