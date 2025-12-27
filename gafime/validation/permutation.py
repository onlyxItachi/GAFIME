"""Permutation test utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PermutationResult:
    observed_score: float
    p_value: float


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    permutations: int = 20,
    seed: int = 0,
) -> PermutationResult:
    """Run a simple permutation test based on a score function."""
    rng = np.random.default_rng(seed)
    observed = score_fn(x, y)
    if permutations <= 0:
        return PermutationResult(observed_score=observed, p_value=1.0)

    count = 0
    for _ in range(permutations):
        y_perm = rng.permutation(y)
        score = score_fn(x, y_perm)
        if score >= observed:
            count += 1
    p_value = (count + 1) / (permutations + 1)
    return PermutationResult(observed_score=observed, p_value=p_value)
