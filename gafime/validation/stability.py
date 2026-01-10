from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from ..backends.base import Backend
from ..metrics import MetricSuite
from ..reporting import StabilityResult


class StabilityAnalyzer:
    def __init__(self, metric_suite: MetricSuite, backend: Backend) -> None:
        self.metric_suite = metric_suite
        self.backend = backend

    def assess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        combos: Iterable[Tuple[int, ...]],
        repeats: int,
        rng: np.random.Generator,
    ) -> List[StabilityResult]:
        if repeats <= 1:
            return []

        combos_list = list(combos)
        scores_by_combo = {
            combo: {name: [] for name in self.metric_suite.metric_names}
            for combo in combos_list
        }

        n_samples = X.shape[0]
        for _ in range(repeats):
            indices = self.backend.sample_indices(n_samples, rng)
            X_sample = X[indices]
            y_sample = y[indices]
            metrics_by_combo = self.backend.score_combos(X_sample, y_sample, combos_list, self.metric_suite)
            for combo, metrics in metrics_by_combo.items():
                for name, value in metrics.items():
                    scores_by_combo[combo][name].append(value)

        results: List[StabilityResult] = []
        for combo, metric_lists in scores_by_combo.items():
            metrics_mean = {name: float(np.mean(values)) for name, values in metric_lists.items()}
            metrics_std = {name: float(np.std(values)) for name, values in metric_lists.items()}
            results.append(StabilityResult(combo=combo, metrics_mean=metrics_mean, metrics_std=metrics_std))

        return results
