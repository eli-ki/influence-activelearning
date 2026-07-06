"""Noisy pool utilities for Phase 2 ADS-style experiments."""

from __future__ import annotations

import numpy as np


def inject_label_noise(
    y: np.ndarray,
    noise_fraction: float,
    n_classes: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Flip labels on a random fraction of samples (pool corruption)."""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n = len(y)
    n_flip = int(n * noise_fraction)
    if n_flip == 0:
        return y_noisy
    idx = rng.choice(n, n_flip, replace=False)
    if n_classes is None:
        n_classes = int(np.max(y) + 1)
    for i in idx:
        choices = [c for c in range(n_classes) if c != y[i]]
        if choices:
            y_noisy[i] = rng.choice(choices)
    return y_noisy


def inject_ood_rows(
    X: np.ndarray,
    y: np.ndarray,
    ood_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace a fraction of rows with OOD Gaussian noise and random labels."""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_ood = int(n * ood_fraction)
    X_out = X.copy()
    y_out = y.copy()
    if n_ood == 0:
        return X_out, y_out
    idx = rng.choice(n, n_ood, replace=False)
    X_out[idx] = rng.normal(0, 3, size=(n_ood, X.shape[1]))
    n_classes = int(np.max(y) + 1)
    y_out[idx] = rng.integers(0, n_classes, size=n_ood)
    return X_out, y_out
