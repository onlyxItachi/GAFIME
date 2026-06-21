from __future__ import annotations

import math
from typing import List

from ..config import ComputeBudget


def estimate_combinations(n_features: int, k: int) -> int:
    if k < 1 or k > n_features:
        return 0
    return math.comb(n_features, k)


def validate_budget(n_features: int, budget: ComputeBudget) -> List[str]:
    warnings: List[str] = []

    if budget.max_comb_size < 1:
        raise ValueError("max_comb_size must be >= 1.")
    if budget.max_combinations_per_k < 1:
        raise ValueError("max_combinations_per_k must be >= 1.")
    if budget.max_discrete_candidates < 0:
        raise ValueError("max_discrete_candidates must be >= 0.")
    if budget.max_thresholds_per_feature < 0:
        raise ValueError("max_thresholds_per_feature must be >= 0.")
    if budget.max_intervals_per_feature < 0:
        raise ValueError("max_intervals_per_feature must be >= 0.")
    if budget.max_feature_pairs_for_rectangles < 0:
        raise ValueError("max_feature_pairs_for_rectangles must be >= 0.")
    if budget.top_k_features_for_discrete < 0:
        raise ValueError("top_k_features_for_discrete must be >= 0.")
    if budget.max_time_series_candidates < 0:
        raise ValueError("max_time_series_candidates must be >= 0.")
    if budget.top_k_features_for_time_series < 0:
        raise ValueError("top_k_features_for_time_series must be >= 0.")
    if budget.max_feature_candidate is not None and budget.max_feature_candidate < -1:
        raise ValueError("max_feature_candidate must be >= 0 or -1 for power-user mode.")
    if budget.top_features_for_higher_k < 1 and budget.max_comb_size > 1:
        warnings.append("top_features_for_higher_k < 1; higher-order combos will be empty.")
    if budget.max_comb_size > n_features:
        warnings.append("max_comb_size exceeds feature count; will cap to n_features.")

    return warnings


def cap_combinations(count: int, limit: int) -> int:
    return min(count, limit)
