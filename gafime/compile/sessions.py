from __future__ import annotations

import ctypes
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


class ResidentContinuousMatrixSession(BackendSession):
    """Session that keeps a native matrix handle alive across scoring phases."""

    def __init__(
        self,
        backend: Any,
        X: Any,
        y: Any,
        scenario_plan: Any,
        metric_suite: Any,
        flags: Any,
        *,
        allocate_matrix,
        free_matrix,
        launch_global_batch,
        scheduler_batches,
        stats_metric_names,
        stats_to_metrics,
        complete_report_metrics,
        max_arity: int,
    ) -> None:
        super().__init__(backend, X, y, scenario_plan, metric_suite, flags)
        self._free_matrix = free_matrix
        self._launch_global_batch = launch_global_batch
        self._scheduler_batches = scheduler_batches
        self._stats_metric_names = stats_metric_names
        self._stats_to_metrics = stats_to_metrics
        self._complete_report_metrics = complete_report_metrics
        self._max_arity = int(max_arity)
        self._matrix, self._retained_buffers = allocate_matrix(X, y)
        self.feature_matrix_handle = _native_handle_value(self._matrix)
        if getattr(flags, "graph", False):
            self.warnings.append(
                f"{backend.info().name} graph capture requested; using resident normal launches."
            )

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self._matrix is not None:
                self._free_matrix(self._matrix)
        finally:
            self._matrix = None
            self._retained_buffers = []
            super().close()

    def score_combos(self, X, y, combos, metric_suite):
        if self.closed:
            raise RuntimeError("Compiled backend session is closed.")
        combos_list = [tuple(int(idx) for idx in combo) for combo in combos]
        if not combos_list:
            return {}
        invalid = [
            combo for combo in combos_list
            if len(combo) < 1 or len(combo) > self._max_arity
        ]
        if invalid:
            raise ValueError(
                f"{self.backend.info().name} compiled continuous session supports combo arity 1 through "
                f"{self._max_arity}."
            )

        stats_metric_names = self._stats_metric_names(metric_suite.metric_names)
        scores = {combo: {} for combo in combos_list}
        if stats_metric_names:
            for batch in self._scheduler_batches(combos_list):
                _kinds, indices, _ops, _interact, _ts_params, arity, batch_size = batch
                if batch_size <= 0:
                    continue
                stats = self._launch_global_batch(
                    self._matrix,
                    indices,
                    int(arity),
                    int(batch_size),
                )
                for row_idx, row in enumerate(stats):
                    combo = tuple(
                        int(indices[row_idx * int(arity) + col])
                        for col in range(int(arity))
                    )
                    scores[combo] = self._stats_to_metrics(row, stats_metric_names)
        self._complete_report_metrics(X, y, combos_list, metric_suite, scores)
        return scores


def _native_handle_value(handle: Any) -> int | None:
    if handle is None:
        return None
    if isinstance(handle, ctypes.c_void_p):
        return int(handle.value or 0)
    value = getattr(handle, "value", None)
    if value is None:
        return None
    return int(value)
