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
        self.feature_matrix_handle = getattr(X, "buffer", X)
        self.result_table_handle = None
        self.graph_requested = bool(getattr(flags, "graph", False))
        self.graph_backend = backend.info().name
        self.graph_status = "disabled"
        self.graph_captured_shapes: set[tuple[int, int]] = set()
        if self.graph_requested and type(self) is BackendSession:
            self.graph_status = "fallback"
            self.warnings.append(_graph_fallback_warning(self.graph_backend))
        self._candidate_tables: dict[tuple[str, tuple[Any, ...]], CandidateDescriptorTable] = {}
        self.candidate_table_handle: CandidateDescriptorTable | None = None
        self.discrete_candidate_table_handle: CandidateDescriptorTable | None = None
        self.time_series_candidate_table_handle: CandidateDescriptorTable | None = None
        self._continuous_combo_cache: dict[
            tuple[tuple[int, ...], int, int],
            tuple[tuple[tuple[int, ...], ...], tuple[str, ...]],
        ] = {}
        self.continuous_combo_cache_hits = 0
        self._discrete_plan_cache: dict[
            tuple[Any, ...],
            tuple[tuple[Any, ...], tuple[str, ...]],
        ] = {}
        self._time_series_plan_cache: dict[
            tuple[Any, ...],
            tuple[tuple[Any, ...], tuple[str, ...]],
        ] = {}
        self.discrete_plan_cache_hits = 0
        self.time_series_plan_cache_hits = 0
        self.candidate_table_cache_hits = 0

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

    def plan_unary(self, n_features, max_count, rng):
        from ..planning.combinations import plan_unary

        return plan_unary(n_features, max_count, rng)

    def plan_higher_order(self, feature_indices, max_comb_size, max_combinations_per_k, rng):
        if max_comb_size < 2:
            return [], []
        indices = [int(idx) for idx in feature_indices]
        if not indices:
            return [], []

        rng.shuffle(indices)
        key = (tuple(indices), int(max_comb_size), int(max_combinations_per_k))
        cached = self._continuous_combo_cache.get(key)
        if cached is not None:
            self.continuous_combo_cache_hits += 1
            combos, warnings = cached
            return list(combos), list(warnings)

        from ..planning.combinations import _python_higher_order_combos, _rust_continuous_combos

        rust_result = _rust_continuous_combos(
            indices,
            min_arity=2,
            max_arity=max_comb_size,
            max_combinations_per_k=max_combinations_per_k,
        )
        if rust_result is None:
            combos, warnings = _python_higher_order_combos(
                indices,
                max_comb_size=max_comb_size,
                max_combinations_per_k=max_combinations_per_k,
            )
        else:
            combos, warnings = rust_result
        frozen = (tuple(combos), tuple(warnings))
        self._continuous_combo_cache[key] = frozen
        return list(frozen[0]), list(frozen[1])

    def plan_discrete_candidates(self, X, feature_scores, config):
        key = (
            _matrix_key(X),
            _feature_score_key(feature_scores),
            _discrete_config_key(config),
        )
        cached = self._discrete_plan_cache.get(key)
        if cached is not None:
            self.discrete_plan_cache_hits += 1
            candidates, warnings = cached
            return list(candidates), list(warnings)

        from ..planning.discrete import plan_discrete_candidates

        candidates, warnings = plan_discrete_candidates(X, feature_scores, config)
        frozen = (tuple(candidates), tuple(warnings))
        self._discrete_plan_cache[key] = frozen
        return list(frozen[0]), list(frozen[1])

    def plan_time_series_candidates(self, n_features, feature_scores, config):
        key = (
            int(n_features),
            _feature_score_key(feature_scores),
            _time_series_config_key(config),
        )
        cached = self._time_series_plan_cache.get(key)
        if cached is not None:
            self.time_series_plan_cache_hits += 1
            candidates, warnings = cached
            return list(candidates), list(warnings)

        from ..planning.time_series import plan_time_series_candidates

        candidates, warnings = plan_time_series_candidates(n_features, feature_scores, config)
        frozen = (tuple(candidates), tuple(warnings))
        self._time_series_plan_cache[key] = frozen
        return list(frozen[0]), list(frozen[1])

    def score_combos(self, X, y, combos, metric_suite):
        return self.backend.score_combos(X, y, combos, metric_suite)

    def score_discrete_candidates(self, X, y, candidates, metric_suite):
        candidates_list = list(candidates)
        self._candidate_table("discrete", candidates_list)
        return self.backend.score_discrete_candidates(X, y, candidates_list, metric_suite)

    def score_discrete_selection_candidates(self, X, y, candidates, *, baseline_pred=None, mi_bins=96):
        candidates_list = list(candidates)
        self._candidate_table("discrete", candidates_list)
        return self.backend.score_discrete_selection_candidates(
            X,
            y,
            candidates_list,
            baseline_pred=baseline_pred,
            mi_bins=mi_bins,
        )

    def score_time_series_candidates(self, X, y, candidates, metric_suite):
        candidates_list = list(candidates)
        self._candidate_table("time_series", candidates_list)
        return self.backend.score_time_series_candidates(X, y, candidates_list, metric_suite)

    def sample_indices(self, n_samples, rng):
        return self.backend.sample_indices(n_samples, rng)

    def permute(self, y, rng):
        return self.backend.permute(y, rng)

    def _candidate_table(self, family: str, candidates: list[Any]) -> "CandidateDescriptorTable":
        key = (family, tuple(candidates))
        table = self._candidate_tables.get(key)
        if table is None:
            table = CandidateDescriptorTable(family=family, candidates=tuple(candidates))
            self._candidate_tables[key] = table
        else:
            self.candidate_table_cache_hits += 1
        self.candidate_table_handle = table
        if family == "discrete":
            self.discrete_candidate_table_handle = table
        elif family == "time_series":
            self.time_series_candidate_table_handle = table
        return table


class CandidateDescriptorTable:
    def __init__(self, *, family: str, candidates: tuple[Any, ...]) -> None:
        self.family = family
        self.candidates = candidates
        self.handle = id(self)

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)


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
        graph_backend: str | None = None,
        graph_capture_supported: bool = False,
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
        self._resident_X_buffer = getattr(X, "buffer", X)
        self._resident_y_buffer = getattr(y, "buffer", y)
        self.feature_matrix_handle = _native_handle_value(self._matrix)
        self.graph_backend = graph_backend or backend.info().name
        if self.graph_requested:
            self.graph_status = "captured" if graph_capture_supported else "fallback"
            if not graph_capture_supported:
                self.warnings.append(_graph_fallback_warning(self.graph_backend))

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
        if not self._prepare_resident_inputs(X, y):
            return self.backend.score_combos(X, y, combos_list, metric_suite)
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
                if self.graph_requested:
                    self.graph_captured_shapes.add((int(arity), int(batch_size)))
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

    def score_time_series_candidates(self, X, y, candidates, metric_suite):
        if self.closed:
            raise RuntimeError("Compiled backend session is closed.")
        candidates_list = list(candidates)
        self._candidate_table("time_series", candidates_list)
        if not candidates_list:
            return {}

        resident_score = getattr(self.backend, "score_time_series_candidates_resident", None)
        if not callable(resident_score):
            return self.backend.score_time_series_candidates(X, y, candidates_list, metric_suite)
        if not self._prepare_resident_inputs(X, y):
            return self.backend.score_time_series_candidates(X, y, candidates_list, metric_suite)
        return resident_score(self._matrix, X, y, candidates_list, metric_suite)

    def _prepare_resident_inputs(self, X, y) -> bool:
        if getattr(X, "buffer", X) is not self._resident_X_buffer:
            return False
        y_buffer = getattr(y, "buffer", y)
        if y_buffer is self._resident_y_buffer:
            return True

        pred = getattr(self.backend, "supports_resident_target_update", None)
        updater = getattr(self.backend, "update_resident_target", None)
        if not callable(pred) or not pred() or not callable(updater):
            return False
        updater(self._matrix, y)
        self._resident_y_buffer = y_buffer
        return True


def _native_handle_value(handle: Any) -> int | None:
    if handle is None:
        return None
    if isinstance(handle, ctypes.c_void_p):
        return int(handle.value or 0)
    value = getattr(handle, "value", None)
    if value is None:
        return None
    return int(value)


def _matrix_key(X: Any) -> tuple[int, tuple[int, ...]]:
    shape = tuple(int(value) for value in getattr(X, "shape", (0, 0)))
    return id(getattr(X, "buffer", X)), shape


def _feature_score_key(feature_scores: Any) -> tuple[tuple[int, float], ...]:
    return tuple(
        sorted(
            (int(feature), float(score))
            for feature, score in dict(feature_scores or {}).items()
        )
    )


def _discrete_config_key(config: Any) -> tuple[Any, ...]:
    budget = config.budget
    return (
        bool(config.enable_discrete_functions),
        str(config.discrete_mode),
        str(config.discrete_threshold_source),
        float(config.discrete_gate_sharpness),
        tuple(float(value) for value in config.discrete_quantiles),
        int(budget.max_discrete_candidates),
        int(budget.max_thresholds_per_feature),
        int(budget.max_intervals_per_feature),
        int(budget.max_feature_pairs_for_rectangles),
        int(budget.top_k_features_for_discrete),
    )


def _time_series_config_key(config: Any) -> tuple[Any, ...]:
    budget = config.budget
    return (
        bool(config.enable_time_series_functions),
        tuple(int(value) for value in config.time_series_lags),
        tuple(int(value) for value in config.time_series_windows),
        int(budget.max_time_series_candidates),
        int(budget.top_k_features_for_time_series),
    )


def _graph_fallback_warning(graph_backend: str) -> str:
    label = str(graph_backend).lower()
    if label == "cuda":
        return "CUDA graph capture requested but native graph APIs are unavailable; using normal launches."
    if label in {"hip", "rocm"}:
        return "HIP graph capture requested but native graph APIs are unavailable; using normal launches."
    if label == "metal":
        return "Metal graph=True is unsupported in v0.5; using normal command-buffer launches."
    return f"{graph_backend} graph capture requested but unsupported; using normal launches."
