"""Permutation testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gafime.backend.metrics import pearson_corr


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    n_permutations: int


def permutation_test_max_pearson(
    features: np.ndarray,
    target: np.ndarray,
    n_permutations: int = 50,
    random_state: int | None = None,
) -> PermutationResult:
    rng = np.random.default_rng(random_state)
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("features must be 2D and target must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and target must share the same number of rows")

    observed = max(abs(pearson_corr(x[:, i], y)) for i in range(x.shape[1]))
    exceed = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_score = max(abs(pearson_corr(x[:, i], y_perm)) for i in range(x.shape[1]))
        if perm_score >= observed:
            exceed += 1
    p_value = (exceed + 1) / (n_permutations + 1)
    return PermutationResult(observed=observed, p_value=p_value, n_permutations=n_permutations)
