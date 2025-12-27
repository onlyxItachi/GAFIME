"""Permutation test utilities."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def permutation_test(
    feature: np.ndarray,
    target: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 100,
    seed: int | None = None,
) -> Dict[str, float]:
    """Compute a simple permutation test p-value for a metric."""
    rng = np.random.default_rng(seed)
    observed = metric_fn(feature, target)
    if n_permutations <= 0:
        return {"observed": float(observed), "p_value": 1.0}

    exceed = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(target)
        score = metric_fn(feature, permuted)
        if abs(score) >= abs(observed):
            exceed += 1
    p_value = (exceed + 1) / (n_permutations + 1)
    return {"observed": float(observed), "p_value": float(p_value)}
