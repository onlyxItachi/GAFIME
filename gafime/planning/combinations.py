from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import ComputeBudget


def plan_unary(
    n_features: int,
    max_count: int,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int]], List[str]]:
    indices = list(range(n_features))
    warnings: List[str] = []
    if n_features <= max_count:
        return [(idx,) for idx in indices], warnings

    rng.shuffle(indices)
    selected = indices[:max_count]
    warnings.append("Unary combinations capped by max_combinations_per_k.")
    return [(idx,) for idx in selected], warnings


def select_top_features(
    feature_scores: Dict[int, float],
    top_n: int,
) -> List[int]:
    if top_n <= 0:
        return []
    ordered = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    return [idx for idx, _score in ordered[:top_n]]


def plan_higher_order(
    feature_indices: Sequence[int],
    max_comb_size: int,
    max_combinations_per_k: int,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    warnings: List[str] = []
    if max_comb_size < 2:
        return [], warnings

    indices = list(feature_indices)
    if not indices:
        return [], warnings

    rng.shuffle(indices)
    combos: List[Tuple[int, ...]] = []
    for k in range(2, max_comb_size + 1):
        combos_k: List[Tuple[int, ...]] = []
        for combo in itertools.combinations(indices, k):
            combos_k.append(combo)
            if len(combos_k) >= max_combinations_per_k:
                warnings.append(f"k={k} combinations capped by max_combinations_per_k.")
                break
        combos.extend(combos_k)
        if len(indices) < k + 1:
            break
    return combos, warnings


def plan_combinations(
    n_features: int,
    budget: ComputeBudget,
    feature_scores: Dict[int, float] | None,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    warnings: List[str] = []
    unary, unary_warnings = plan_unary(n_features, budget.max_combinations_per_k, rng)
    warnings.extend(unary_warnings)

    if budget.max_comb_size < 2:
        return unary, warnings

    if feature_scores is None:
        top_indices = [idx for (idx,) in unary]
    else:
        top_indices = select_top_features(feature_scores, budget.top_features_for_higher_k)

    higher, higher_warnings = plan_higher_order(
        top_indices,
        budget.max_comb_size,
        budget.max_combinations_per_k,
        rng,
    )
    warnings.extend(higher_warnings)
    return unary + higher, warnings
