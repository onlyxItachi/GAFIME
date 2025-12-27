"""Permutation-based validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PermutationResult:
    observed_score: float
    pvalue: float


def permutation_test(
    score_fn: Callable[[np.ndarray], float],
    target: np.ndarray,
    *,
    n_permutations: int = 25,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """Compute permutation p-value for a scalar score function."""
    rng = rng or np.random.default_rng()
    target = np.asarray(target)
    observed = float(score_fn(target))
    if n_permutations <= 0:
        return PermutationResult(observed_score=observed, pvalue=1.0)

    hits = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(target)
        if float(score_fn(shuffled)) >= observed:
            hits += 1
    pvalue = (hits + 1) / (n_permutations + 1)
    return PermutationResult(observed_score=observed, pvalue=pvalue)
