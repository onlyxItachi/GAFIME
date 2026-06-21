from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Tuple

from .backends import resolve_backend
from .config import EngineConfig
from .discrete import (
    DiscreteFunctionCandidate,
    GPU_HARD_MODE_ERROR,
    describe_discrete_candidate,
    discrete_feature_names,
    rank_discrete_selection_scores,
)
from .native_data import NativeMatrix, NativeVector, build_interaction_vector, mean, std
from .planning.combinations import plan_higher_order, plan_unary, select_top_features
from .planning.discrete import plan_discrete_candidates
from .planning.time_series import plan_time_series_candidates
from .reporting import (
    Decision,
    DiagnosticReport,
    InteractionResult,
    NativeReportBuilder,
    PermutationResult,
    StabilityResult,
)
from .time_series import (
    TimeSeriesCandidate,
    describe_time_series_candidate,
)
from .utils.arrays import coerce_inputs
from .utils.safety import validate_budget
from .validation import PermutationTester, StabilityAnalyzer


class GafimeEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.metric_suite = None
        self.backend = None

    def analyze(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
    ) -> DiagnosticReport:
        if os.environ.get("GAFIME_USE_LEGACY_ENGINE") == "1":
            return self._analyze_legacy(X, y, feature_names)
        artifact = self.compile(X, y, feature_names=feature_names)
        try:
            return artifact.analyze()
        finally:
            artifact.close()

    def compile(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
        *,
        flags=None,
    ):
        from .compile import CompileFlags, CompiledGafime

        return CompiledGafime.from_engine(
            self,
            X,
            y,
            feature_names=feature_names,
            flags=flags or CompileFlags(),
        )

    def _analyze_legacy(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
    ) -> DiagnosticReport:
        X_array, y_array, names = coerce_inputs(X, y, feature_names)
        return self._analyze_native(X_array, y_array, names)

    def _analyze_native(
        self,
        X_array,
        y_array,
        names: List[str],
        *,
        initial_warnings: Iterable[str] | None = None,
        backend=None,
        executor=None,
        prevalidated: bool = False,
    ) -> DiagnosticReport:
        warnings = list(initial_warnings or [])
        if prevalidated and backend is None:
            raise RuntimeError("prevalidated native analysis requires a backend.")
        if not prevalidated:
            warnings.extend(validate_budget(X_array.shape[1], self.config.budget))
            backend, backend_warnings = resolve_backend(self.config, X_array, y_array)
            warnings.extend(backend_warnings)
        elif backend is None:
            backend, backend_warnings = resolve_backend(self.config, X_array, y_array)
            warnings.extend(backend_warnings)

        active_backend = executor or backend
        self.backend = active_backend
        self.metric_suite = active_backend.metric_suite(self.config)
        backend_info = active_backend.info()
        if (
            self.config.enable_discrete_functions
            and self.config.discrete_mode == "hard"
            and backend_info.is_gpu
        ):
            raise ValueError(GPU_HARD_MODE_ERROR)
        if self.config.enable_discrete_functions:
            _validate_discrete_ranking(self.config.discrete_ranking)

        rng = random.Random(self.config.random_seed)
        X_data = active_backend.to_device(X_array)
        y_data = active_backend.to_device(y_array)
        unary_combos, unary_warnings = _plan_unary(
            self.backend,
            X_array.shape[1],
            self.config.budget.max_combinations_per_k,
            rng,
        )
        warnings.extend(unary_warnings)

        interaction_builder = NativeReportBuilder("interaction")
        _unary_results, unary_scores = self._score_combos(
            X_data,
            y_data,
            unary_combos,
            names,
            result_builder=interaction_builder,
        )

        feature_scores = {combo[0]: _metric_strength(metrics) for combo, metrics in unary_scores.items()}
        top_features = select_top_features(feature_scores, self.config.budget.top_features_for_higher_k)

        higher_combos, higher_warnings = _plan_higher_order(
            self.backend,
            top_features,
            self.config.budget.max_comb_size,
            self.config.budget.max_combinations_per_k,
            rng,
        )
        warnings.extend(higher_warnings)

        _higher_results, higher_scores = self._score_combos(
            X_data,
            y_data,
            higher_combos,
            names,
            result_builder=interaction_builder,
        )
        discrete_candidates, discrete_warnings = _plan_discrete_candidates(
            self.backend,
            X_array,
            feature_scores,
            self.config,
        )
        warnings.extend(discrete_warnings)

        baseline_pred = None
        if discrete_candidates and self.config.discrete_ranking == "split_aware":
            baseline_pred = _continuous_baseline_prediction(
                X_array,
                y_array,
                {**unary_scores, **higher_scores},
            )

        _discrete_results, discrete_scores = self._score_discrete_candidates(
            X_data,
            y_data,
            discrete_candidates,
            names,
            baseline_pred=baseline_pred,
            result_builder=interaction_builder,
        )
        time_series_candidates, time_series_warnings = _plan_time_series_candidates(
            self.backend,
            X_array.shape[1],
            feature_scores,
            self.config,
        )
        warnings.extend(time_series_warnings)
        _time_series_results, time_series_scores = self._score_time_series_candidates(
            X_data,
            y_data,
            time_series_candidates,
            names,
            result_builder=interaction_builder,
        )

        interactions = interaction_builder.sequence()
        interaction_scores = {**unary_scores, **higher_scores, **discrete_scores, **time_series_scores}
        all_combos = unary_combos + higher_combos

        stability = StabilityAnalyzer(self.metric_suite, active_backend).assess(
            X_data,
            y_data,
            all_combos,
            self.config.num_repeats,
            rng,
        )
        permutations = PermutationTester(self.metric_suite, active_backend).test(
            X_data,
            y_data,
            all_combos,
            self.config.permutation_tests,
            rng,
            actual_scores=interaction_scores,
        )
        if discrete_candidates:
            stability.extend(
                self._assess_discrete_candidates(
                    X_data,
                    y_data,
                    discrete_candidates,
                    rng,
                    names,
                )
            )
            permutations.extend(
                self._test_discrete_candidates(
                    X_data,
                    y_data,
                    discrete_candidates,
                    rng,
                    actual_scores=discrete_scores,
                    feature_names=names,
                )
            )
        if time_series_candidates:
            stability.extend(
                self._assess_time_series_candidates(
                    X_data,
                    y_data,
                    time_series_candidates,
                    rng,
                    names,
                )
            )
            permutations.extend(
                self._test_time_series_candidates(
                    X_data,
                    y_data,
                    time_series_candidates,
                    rng,
                    actual_scores=time_series_scores,
                    feature_names=names,
                )
            )

        decision = _make_decision(
            interaction_scores,
            permutations,
            stability,
            self.config,
        )

        return DiagnosticReport(
            config=self.config,
            feature_names=names,
            interactions=interactions,
            stability=stability,
            permutations=permutations,
            warnings=warnings,
            decision=decision,
            backend=backend_info,
        )

    def _score_combos(
        self,
        X,
        y,
        combos: Iterable[Tuple[int, ...]],
        feature_names: List[str],
        result_builder: NativeReportBuilder | None = None,
    ) -> Tuple[List[InteractionResult], Dict[Tuple[int, ...], Dict[str, float]]]:
        combos_list = list(combos)
        scores = self.backend.score_combos(X, y, combos_list, self.metric_suite)
        results: List[InteractionResult] = []
        for combo in combos_list:
            metrics = scores[combo]
            if result_builder is None:
                results.append(
                    InteractionResult(
                        combo=combo,
                        feature_names=tuple(feature_names[idx] for idx in combo),
                        metrics=metrics,
                    )
                )
            else:
                result_builder.append_interaction(
                    combo=combo,
                    feature_names=tuple(feature_names[idx] for idx in combo),
                    metrics=metrics,
                )
        return results, scores

    def _score_time_series_candidates(
        self,
        X,
        y,
        candidates: Iterable[TimeSeriesCandidate],
        feature_names: List[str],
        result_builder: NativeReportBuilder | None = None,
    ) -> Tuple[List[InteractionResult], Dict[TimeSeriesCandidate, Dict[str, float]]]:
        candidates_list = list(candidates)
        if not candidates_list:
            return [], {}
        scores = self.backend.score_time_series_candidates(X, y, candidates_list, self.metric_suite)
        candidates_list.sort(
            key=lambda candidate: (
                -_metric_strength(scores[candidate]),
                candidate.candidate_id,
            )
        )
        results: List[InteractionResult] = []
        for candidate in candidates_list:
            if result_builder is None:
                results.append(
                    InteractionResult(
                        combo=candidate.combo,
                        feature_names=(feature_names[candidate.feature_index],),
                        metrics=scores[candidate],
                        family="time_series_function",
                        expression=describe_time_series_candidate(candidate, feature_names),
                        params=candidate.params(),
                        candidate_id=candidate.candidate_id,
                    )
                )
            else:
                result_builder.append_interaction(
                    combo=candidate.combo,
                    feature_names=(feature_names[candidate.feature_index],),
                    metrics=scores[candidate],
                    family="time_series_function",
                    expression=describe_time_series_candidate(candidate, feature_names),
                    params=candidate.params(),
                    candidate_id=candidate.candidate_id,
                )
        return results, scores

    def _score_discrete_candidates(
        self,
        X,
        y,
        candidates: Iterable[DiscreteFunctionCandidate],
        feature_names: List[str],
        baseline_pred=None,
        result_builder: NativeReportBuilder | None = None,
    ) -> Tuple[
        List[InteractionResult],
        Dict[DiscreteFunctionCandidate, Dict[str, float]],
    ]:
        candidates_list = list(candidates)
        if not candidates_list:
            return [], {}

        scores = self.backend.score_discrete_candidates(X, y, candidates_list, self.metric_suite)
        ranking = self.config.discrete_ranking
        selector_scores: Dict[DiscreteFunctionCandidate, Dict[str, float]] = {}
        if ranking == "split_aware":
            selector_scores = self.backend.score_discrete_selection_candidates(
                X,
                y,
                candidates_list,
                baseline_pred=baseline_pred,
                mi_bins=self.config.mi_bins,
            )
            rank_scores = rank_discrete_selection_scores(selector_scores)
        elif ranking == "metric":
            rank_scores = {
                candidate: _metric_strength(scores[candidate])
                for candidate in candidates_list
            }
        else:
            rank_scores = {candidate: 0.0 for candidate in candidates_list}

        if ranking != "none":
            candidates_list.sort(
                key=lambda candidate: (
                    -rank_scores.get(candidate, 0.0),
                    candidate.candidate_id,
                )
            )
        results: List[InteractionResult] = []
        for candidate in candidates_list:
            params = candidate.params()
            params["discrete_ranking"] = ranking
            params["selection_score"] = float(rank_scores.get(candidate, 0.0))
            if selector_scores:
                params["selection_scores"] = {
                    name: float(value)
                    for name, value in selector_scores.get(candidate, {}).items()
                }
            if result_builder is None:
                results.append(
                    InteractionResult(
                        combo=candidate.combo,
                        feature_names=discrete_feature_names(candidate, feature_names),
                        metrics=scores[candidate],
                        family="discrete_function",
                        expression=describe_discrete_candidate(candidate, feature_names),
                        params=params,
                        candidate_id=candidate.candidate_id,
                    )
                )
            else:
                result_builder.append_interaction(
                    combo=candidate.combo,
                    feature_names=discrete_feature_names(candidate, feature_names),
                    metrics=scores[candidate],
                    family="discrete_function",
                    expression=describe_discrete_candidate(candidate, feature_names),
                    params=params,
                    candidate_id=candidate.candidate_id,
                )
        return results, scores

    def _assess_discrete_candidates(
        self,
        X,
        y,
        candidates: Iterable[DiscreteFunctionCandidate],
        rng: random.Random,
        feature_names: List[str],
    ) -> List[StabilityResult]:
        if self.config.num_repeats <= 1:
            return []

        candidates_list = list(candidates)
        scores_by_candidate = {
            candidate: {name: [] for name in self.metric_suite.metric_names}
            for candidate in candidates_list
        }

        n_samples = X.shape[0]
        for _ in range(self.config.num_repeats):
            indices = self.backend.sample_indices(n_samples, rng)
            X_sample = X.select_rows(indices)
            y_sample = y.select(indices)
            metrics_by_candidate = self.backend.score_discrete_candidates(
                X_sample,
                y_sample,
                candidates_list,
                self.metric_suite,
            )
            for candidate, metrics in metrics_by_candidate.items():
                for name, value in metrics.items():
                    scores_by_candidate[candidate][name].append(value)

        results: List[StabilityResult] = []
        for candidate, metric_lists in scores_by_candidate.items():
            metrics_mean = {name: float(mean(values)) for name, values in metric_lists.items()}
            metrics_std = {name: float(std(values)) for name, values in metric_lists.items()}
            results.append(
                StabilityResult(
                    combo=candidate.combo,
                    metrics_mean=metrics_mean,
                    metrics_std=metrics_std,
                    family="discrete_function",
                    expression=describe_discrete_candidate(candidate, feature_names),
                    params=candidate.params(),
                    candidate_id=candidate.candidate_id,
                )
            )
        return results

    def _test_discrete_candidates(
        self,
        X,
        y,
        candidates: Iterable[DiscreteFunctionCandidate],
        rng: random.Random,
        actual_scores: Dict[DiscreteFunctionCandidate, Dict[str, float]],
        feature_names: List[str],
    ) -> List[PermutationResult]:
        if self.config.permutation_tests <= 0:
            return []

        candidates_list = list(candidates)
        exceed_counts = {
            candidate: {name: 0 for name in self.metric_suite.metric_names}
            for candidate in candidates_list
        }

        for _ in range(self.config.permutation_tests):
            y_perm = self.backend.permute(y, rng)
            metrics_by_candidate = self.backend.score_discrete_candidates(
                X,
                y_perm,
                candidates_list,
                self.metric_suite,
            )
            for candidate, metrics in metrics_by_candidate.items():
                for name, value in metrics.items():
                    if _exceeds_null(value, actual_scores[candidate][name], name):
                        exceed_counts[candidate][name] += 1

        results: List[PermutationResult] = []
        for candidate, counts in exceed_counts.items():
            p_values = {
                name: float((count + 1) / (self.config.permutation_tests + 1))
                for name, count in counts.items()
            }
            results.append(
                PermutationResult(
                    combo=candidate.combo,
                    p_values=p_values,
                    family="discrete_function",
                    expression=describe_discrete_candidate(candidate, feature_names),
                    params=candidate.params(),
                    candidate_id=candidate.candidate_id,
                )
            )
        return results

    def _assess_time_series_candidates(
        self,
        X,
        y,
        candidates: Iterable[TimeSeriesCandidate],
        rng: random.Random,
        feature_names: List[str],
    ) -> List[StabilityResult]:
        if self.config.num_repeats <= 1:
            return []
        candidates_list = list(candidates)
        scores_by_candidate = {
            candidate: {name: [] for name in self.metric_suite.metric_names}
            for candidate in candidates_list
        }
        n_samples = X.shape[0]
        for _ in range(self.config.num_repeats):
            indices = self.backend.sample_indices(n_samples, rng)
            metrics_by_candidate = self.backend.score_time_series_candidates(
                X.select_rows(indices),
                y.select(indices),
                candidates_list,
                self.metric_suite,
            )
            for candidate, metrics in metrics_by_candidate.items():
                for name, value in metrics.items():
                    scores_by_candidate[candidate][name].append(value)
        results: List[StabilityResult] = []
        for candidate, metric_lists in scores_by_candidate.items():
            results.append(
                StabilityResult(
                    combo=candidate.combo,
                    metrics_mean={name: float(mean(values)) for name, values in metric_lists.items()},
                    metrics_std={name: float(std(values)) for name, values in metric_lists.items()},
                    family="time_series_function",
                    expression=describe_time_series_candidate(candidate, feature_names),
                    params=candidate.params(),
                    candidate_id=candidate.candidate_id,
                )
            )
        return results

    def _test_time_series_candidates(
        self,
        X,
        y,
        candidates: Iterable[TimeSeriesCandidate],
        rng: random.Random,
        actual_scores: Dict[TimeSeriesCandidate, Dict[str, float]],
        feature_names: List[str],
    ) -> List[PermutationResult]:
        if self.config.permutation_tests <= 0:
            return []
        candidates_list = list(candidates)
        exceed_counts = {
            candidate: {name: 0 for name in self.metric_suite.metric_names}
            for candidate in candidates_list
        }
        for _ in range(self.config.permutation_tests):
            y_perm = self.backend.permute(y, rng)
            metrics_by_candidate = self.backend.score_time_series_candidates(
                X,
                y_perm,
                candidates_list,
                self.metric_suite,
            )
            for candidate, metrics in metrics_by_candidate.items():
                for name, value in metrics.items():
                    if _exceeds_null(value, actual_scores[candidate][name], name):
                        exceed_counts[candidate][name] += 1
        return [
            PermutationResult(
                combo=candidate.combo,
                p_values={
                    name: float((count + 1) / (self.config.permutation_tests + 1))
                    for name, count in counts.items()
                },
                family="time_series_function",
                expression=describe_time_series_candidate(candidate, feature_names),
                params=candidate.params(),
                candidate_id=candidate.candidate_id,
            )
            for candidate, counts in exceed_counts.items()
        ]


def _metric_strength(metrics: Dict[str, float]) -> float:
    strengths: List[float] = []
    for name, value in metrics.items():
        if name in ("pearson", "spearman"):
            strengths.append(abs(value))
        else:
            strengths.append(value)
    return max(strengths) if strengths else 0.0


def _plan_unary(planner, n_features: int, max_count: int, rng: random.Random):
    plan_fn = getattr(planner, "plan_unary", None)
    if callable(plan_fn):
        return plan_fn(n_features, max_count, rng)
    return plan_unary(n_features, max_count, rng)


def _plan_higher_order(
    planner,
    feature_indices,
    max_comb_size: int,
    max_combinations_per_k: int,
    rng: random.Random,
):
    plan_fn = getattr(planner, "plan_higher_order", None)
    if callable(plan_fn):
        return plan_fn(feature_indices, max_comb_size, max_combinations_per_k, rng)
    return plan_higher_order(feature_indices, max_comb_size, max_combinations_per_k, rng)


def _plan_discrete_candidates(
    planner,
    X: NativeMatrix,
    feature_scores: Dict[int, float],
    config: EngineConfig,
):
    plan_fn = getattr(planner, "plan_discrete_candidates", None)
    if callable(plan_fn):
        return plan_fn(X, feature_scores, config)
    return plan_discrete_candidates(X, feature_scores, config)


def _plan_time_series_candidates(
    planner,
    n_features: int,
    feature_scores: Dict[int, float],
    config: EngineConfig,
):
    plan_fn = getattr(planner, "plan_time_series_candidates", None)
    if callable(plan_fn):
        return plan_fn(n_features, feature_scores, config)
    return plan_time_series_candidates(n_features, feature_scores, config)


def _validate_discrete_ranking(value: str) -> None:
    allowed = {"split_aware", "metric", "none"}
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"discrete_ranking must be one of: {allowed_text}.")


def _continuous_baseline_prediction(
    X: NativeMatrix,
    y: NativeVector,
    scores: Dict[Tuple[int, ...], Dict[str, float]],
    *,
    max_terms: int = 64,
    alpha: float = 1.0,
) -> List[float]:
    """Fit a small ridge baseline for residual-aware discrete ranking."""
    y_values = y.to_list()
    if X.n_samples != len(y_values) or not scores:
        return [mean(y_values)] * len(y_values)

    ordered_combos = sorted(
        scores,
        key=lambda combo: _metric_strength(scores[combo]),
        reverse=True,
    )[:max_terms]
    columns: List[List[float]] = []
    for combo in ordered_combos:
        try:
            vector = build_interaction_vector(X, combo)
        except Exception:
            continue
        if len(vector) == len(y_values):
            columns.append([float(value) for value in vector])

    if not columns:
        return [mean(y_values)] * len(y_values)

    col_means = [mean(column) for column in columns]
    col_stds = [std(column) for column in columns]
    keep = [idx for idx, value in enumerate(col_stds) if value > 1e-12]
    if not keep:
        return [mean(y_values)] * len(y_values)

    y_mean = mean(y_values)
    n_terms = len(keep)
    xtx = [[0.0 for _ in range(n_terms)] for _ in range(n_terms)]
    xty = [0.0 for _ in range(n_terms)]
    design_rows: List[List[float]] = []
    for row in range(len(y_values)):
        row_values = [
            (columns[col][row] - col_means[col]) / col_stds[col]
            for col in keep
        ]
        design_rows.append(row_values)
        y_centered = y_values[row] - y_mean
        for i, xi in enumerate(row_values):
            xty[i] += xi * y_centered
            for j, xj in enumerate(row_values):
                xtx[i][j] += xi * xj
    for i in range(n_terms):
        xtx[i][i] += float(alpha)
    coef = _solve_linear_system(xtx, xty)
    if coef is None:
        return [y_mean] * len(y_values)
    return [
        y_mean + sum(value * coef[idx] for idx, value in enumerate(row_values))
        for row_values in design_rows
    ]


def _solve_linear_system(matrix: List[List[float]], rhs: List[float]) -> List[float] | None:
    n = len(rhs)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) <= 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[row][n] for row in range(n)]


def _make_decision(
    scores: Dict[object, Dict[str, float]],
    permutations: List,
    stability: List,
    config: EngineConfig,
) -> Decision:
    perm_map = {_result_key(result): result.p_values for result in permutations}
    stab_map = {_result_key(result): result.metrics_std for result in stability}

    signal_detected = False
    for candidate, metrics in scores.items():
        key = _candidate_key(candidate)
        p_values = perm_map.get(key, {})
        stds = stab_map.get(key, {})
        for name, value in metrics.items():
            p_ok = True
            if p_values:
                p_ok = p_values.get(name, 1.0) <= config.permutation_p_threshold
            s_ok = True
            if stds:
                s_ok = stds.get(name, 1.0) <= config.stability_std_threshold
            if p_ok and s_ok and _metric_strength({name: value}) > 0:
                signal_detected = True
                break
        if signal_detected:
            break

    message = (
        "Learnable feature-based signal detected."
        if signal_detected
        else "No learnable feature-based signal detected for this target."
    )
    return Decision(signal_detected=signal_detected, message=message)


def _candidate_key(candidate: object) -> str:
    if isinstance(candidate, DiscreteFunctionCandidate):
        return candidate.candidate_id
    candidate_id = getattr(candidate, "candidate_id", "")
    if candidate_id:
        return str(candidate_id)
    return _combo_key(candidate)


def _result_key(result) -> str:
    candidate_id = getattr(result, "candidate_id", "")
    if candidate_id:
        return candidate_id
    return _combo_key(result.combo)


def _combo_key(combo: object) -> str:
    if isinstance(combo, tuple):
        return "interaction:" + ",".join(str(idx) for idx in combo)
    return str(combo)


def _exceeds_null(null_value: float, actual_value: float, metric_name: str) -> bool:
    if metric_name in ("pearson", "spearman"):
        return abs(null_value) >= abs(actual_value)
    return null_value >= actual_value
