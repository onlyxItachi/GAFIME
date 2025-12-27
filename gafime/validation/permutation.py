from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..backends.base import Backend
from ..metrics import MetricSuite
from ..reporting import PermutationResult


class PermutationTester:
    def __init__(self, metric_suite: MetricSuite, backend: Backend) -> None:
        self.metric_suite = metric_suite
        self.backend = backend

    def test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combos: Iterable[Tuple[int, ...]],
        num_permutations: int,
        rng: np.random.Generator,
        actual_scores: Dict[Tuple[int, ...], Dict[str, float]] | None = None,
    ) -> List[PermutationResult]:
        if num_permutations <= 0:
            return []

        combos_list = list(combos)
        if actual_scores is None:
            actual_scores = {
                combo: self.metric_suite.score(self.backend.build_interaction_vector(X, combo), y)
                for combo in combos_list
            }

        exceed_counts = {
            combo: {name: 0 for name in self.metric_suite.metric_names}
            for combo in combos_list
        }

        for _ in range(num_permutations):
            y_perm = self.backend.permute(y, rng)
            for combo in combos_list:
                vector = self.backend.build_interaction_vector(X, combo)
                metrics = self.metric_suite.score(vector, y_perm)
                for name, value in metrics.items():
                    if _exceeds_null(value, actual_scores[combo][name], name):
                        exceed_counts[combo][name] += 1

        results: List[PermutationResult] = []
        for combo, counts in exceed_counts.items():
            p_values = {
                name: float((count + 1) / (num_permutations + 1))
                for name, count in counts.items()
            }
            results.append(PermutationResult(combo=combo, p_values=p_values))

        return results


def _exceeds_null(null_value: float, actual_value: float, metric_name: str) -> bool:
    if metric_name in ("pearson", "spearman"):
        return abs(null_value) >= abs(actual_value)
    return null_value >= actual_value
