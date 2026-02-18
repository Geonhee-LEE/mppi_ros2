"""
노이즈 샘플러 (sampling.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.controllers.mppi.sampling import (
    NoiseSampler,
    GaussianSampler,
    ColoredNoiseSampler,
    RectifiedGaussianSampler,
)

K, N, nu = 1024, 20, 2
sigma = np.array([0.5, 0.3])


# ── Tests ──────────────────────────────────────────────────


def test_gaussian_shape():
    print("\n" + "=" * 60)
    print("Test: GaussianSampler output shape (K, N, nu)")
    print("=" * 60)

    sampler = GaussianSampler(sigma, seed=42)
    U = np.zeros((N, nu))
    noise = sampler.sample(U, K)

    assert noise.shape == (K, N, nu), f"shape: {noise.shape}"
    print(f"  shape={noise.shape}")
    print("PASS")


def test_gaussian_mean_zero():
    print("\n" + "=" * 60)
    print("Test: GaussianSampler mean ~ 0 (law of large numbers)")
    print("=" * 60)

    sampler = GaussianSampler(sigma, seed=42)
    U = np.zeros((N, nu))
    noise = sampler.sample(U, 10000)

    mean = np.mean(noise, axis=0)  # (N, nu)
    assert np.allclose(mean, 0.0, atol=0.05), f"mean too far: max={np.max(np.abs(mean))}"
    print(f"  max_abs_mean={np.max(np.abs(mean)):.4f}")
    print("PASS")


def test_gaussian_seed_reproducibility():
    print("\n" + "=" * 60)
    print("Test: same seed -> same noise")
    print("=" * 60)

    U = np.zeros((N, nu))

    s1 = GaussianSampler(sigma, seed=123)
    n1 = s1.sample(U, K)

    s2 = GaussianSampler(sigma, seed=123)
    n2 = s2.sample(U, K)

    assert np.allclose(n1, n2), "seed reproducibility failed"
    print("PASS")


def test_gaussian_constraint_clipping():
    print("\n" + "=" * 60)
    print("Test: GaussianSampler respects control bounds")
    print("=" * 60)

    sampler = GaussianSampler(np.array([2.0, 2.0]), seed=42)
    U = np.zeros((N, nu))
    u_min = np.array([-0.5, -0.3])
    u_max = np.array([0.5, 0.3])

    noise = sampler.sample(U, K, u_min, u_max)
    sampled = U + noise

    assert np.all(sampled >= u_min - 1e-10), f"below min: {sampled.min(axis=(0,1))}"
    assert np.all(sampled <= u_max + 1e-10), f"above max: {sampled.max(axis=(0,1))}"
    print("PASS")


def test_colored_noise_shape():
    print("\n" + "=" * 60)
    print("Test: ColoredNoiseSampler output shape")
    print("=" * 60)

    sampler = ColoredNoiseSampler(sigma, theta=np.array([1.0, 1.0]), dt=0.05, seed=42)
    U = np.zeros((N, nu))
    noise = sampler.sample(U, K)

    assert noise.shape == (K, N, nu), f"shape: {noise.shape}"
    print("PASS")


def test_colored_noise_correlation():
    print("\n" + "=" * 60)
    print("Test: ColoredNoise has temporal correlation > 0")
    print("=" * 60)

    sampler = ColoredNoiseSampler(sigma, theta=np.array([0.5, 0.5]), dt=0.05, seed=42)
    U = np.zeros((N, nu))
    noise = sampler.sample(U, 256)

    # Average auto-correlation between consecutive timesteps
    autocorr = np.mean(noise[:, :-1, :] * noise[:, 1:, :])
    print(f"  autocorrelation={autocorr:.4f}")
    assert autocorr > 0, f"no temporal correlation: {autocorr}"
    print("PASS")


def test_colored_noise_theta_zero():
    print("\n" + "=" * 60)
    print("Test: ColoredNoise theta=0 -> brownian motion (variance grows)")
    print("=" * 60)

    sampler = ColoredNoiseSampler(
        sigma=np.array([1.0, 1.0]),
        theta=np.array([0.0, 0.0]),
        dt=0.05,
        seed=42,
    )
    U = np.zeros((N, nu))
    noise = sampler.sample(U, 2000)

    # Variance should grow with time (no mean-reversion)
    var_early = np.var(noise[:, 2, :])
    var_late = np.var(noise[:, -1, :])
    print(f"  var_early(t=2)={var_early:.4f}, var_late(t={N-1})={var_late:.4f}")
    assert var_late > var_early, f"variance not growing: {var_late} <= {var_early}"
    print("PASS")


def test_rectified_shape():
    print("\n" + "=" * 60)
    print("Test: RectifiedGaussianSampler output shape")
    print("=" * 60)

    sampler = RectifiedGaussianSampler(sigma, seed=42)
    U = np.zeros((N, nu))
    noise = sampler.sample(U, K)

    assert noise.shape == (K, N, nu), f"shape: {noise.shape}"
    print("PASS")


def test_rectified_constraint_satisfaction():
    print("\n" + "=" * 60)
    print("Test: RectifiedGaussianSampler satisfies bounds")
    print("=" * 60)

    sampler = RectifiedGaussianSampler(np.array([2.0, 2.0]), max_retries=10, seed=42)
    U = np.zeros((N, nu))
    u_min = np.array([-0.3, -0.3])
    u_max = np.array([0.3, 0.3])

    noise = sampler.sample(U, K, u_min, u_max)
    sampled = U + noise

    assert np.all(sampled >= u_min - 1e-10), f"below min"
    assert np.all(sampled <= u_max + 1e-10), f"above max"
    print("PASS")


def test_sampler_does_not_modify_U():
    print("\n" + "=" * 60)
    print("Test: sample() does not modify original U")
    print("=" * 60)

    sampler = GaussianSampler(sigma, seed=42)
    U = np.ones((N, nu)) * 0.5
    U_copy = U.copy()

    sampler.sample(U, K)
    assert np.allclose(U, U_copy), "U was modified by sample()"
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Sampling Unit Tests")
    print("=" * 60)

    tests = [
        test_gaussian_shape,
        test_gaussian_mean_zero,
        test_gaussian_seed_reproducibility,
        test_gaussian_constraint_clipping,
        test_colored_noise_shape,
        test_colored_noise_correlation,
        test_colored_noise_theta_zero,
        test_rectified_shape,
        test_rectified_constraint_satisfaction,
        test_sampler_does_not_modify_U,
    ]

    try:
        for t in tests:
            t()
        print(f"\n{'=' * 60}")
        print(f"  All {len(tests)} Tests Passed!")
        print(f"{'=' * 60}")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
