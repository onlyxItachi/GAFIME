from __future__ import annotations

from typing import Any


class BackendSession:
    """Compiled backend session shell.

    This delegates to the selected backend for the API skeleton checkpoint and
    gives later checkpoints a stable object for resident native handles.
    """

    def __init__(
        self,
        backend: Any,
        X: Any,
        y: Any,
        scenario_plan: Any,
        metric_suite: Any,
        flags: Any,
    ) -> None:
        self.backend = backend
        self.X = X
        self.y = y
        self.scenario_plan = scenario_plan
        self.compiled_metric_suite = metric_suite
        self.flags = flags
        self.warnings: list[str] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def info(self):
        return self.backend.info()

    def metric_suite(self, config):
        return self.backend.metric_suite(config)

    def check_budget(self, X, y, budget):
        return self.backend.check_budget(X, y, budget)

    def to_device(self, data):
        return self.backend.to_device(data)

    def to_host(self, data):
        return self.backend.to_host(data)

    def build_interaction_vector(self, X, combo):
        return self.backend.build_interaction_vector(X, combo)

    def score_combos(self, X, y, combos, metric_suite):
        return self.backend.score_combos(X, y, combos, metric_suite)

    def score_discrete_candidates(self, X, y, candidates, metric_suite):
        return self.backend.score_discrete_candidates(X, y, candidates, metric_suite)

    def score_discrete_selection_candidates(self, X, y, candidates, *, baseline_pred=None, mi_bins=96):
        return self.backend.score_discrete_selection_candidates(
            X,
            y,
            candidates,
            baseline_pred=baseline_pred,
            mi_bins=mi_bins,
        )

    def score_time_series_candidates(self, X, y, candidates, metric_suite):
        return self.backend.score_time_series_candidates(X, y, candidates, metric_suite)

    def sample_indices(self, n_samples, rng):
        return self.backend.sample_indices(n_samples, rng)

    def permute(self, y, rng):
        return self.backend.permute(y, rng)
