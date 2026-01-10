from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .backends import resolve_backend
from .config import EngineConfig
from .planning.combinations import plan_higher_order, plan_unary, select_top_features
from .reporting import Decision, DiagnosticReport, InteractionResult
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
        X_array, y_array, names = coerce_inputs(X, y, feature_names)
        warnings = validate_budget(X_array.shape[1], self.config.budget)
        backend, backend_warnings = resolve_backend(self.config, X_array, y_array)
        warnings.extend(backend_warnings)
        self.backend = backend
        self.metric_suite = backend.metric_suite(self.config)
        backend_info = backend.info()

        rng = np.random.default_rng(self.config.random_seed)
        X_data = backend.to_device(X_array)
        y_data = backend.to_device(y_array)
        unary_combos, unary_warnings = plan_unary(
            X_array.shape[1],
            self.config.budget.max_combinations_per_k,
            rng,
        )
        warnings.extend(unary_warnings)

        unary_results, unary_scores = self._score_combos(
            X_data,
            y_data,
            unary_combos,
            names,
        )

        feature_scores = {combo[0]: _metric_strength(metrics) for combo, metrics in unary_scores.items()}
        top_features = select_top_features(feature_scores, self.config.budget.top_features_for_higher_k)

        higher_combos, higher_warnings = plan_higher_order(
            top_features,
            self.config.budget.max_comb_size,
            self.config.budget.max_combinations_per_k,
            rng,
        )
        warnings.extend(higher_warnings)

        higher_results, higher_scores = self._score_combos(
            X_data,
            y_data,
            higher_combos,
            names,
        )
        interactions = unary_results + higher_results
        interaction_scores = {**unary_scores, **higher_scores}
        all_combos = unary_combos + higher_combos

        stability = StabilityAnalyzer(self.metric_suite, backend).assess(
            X_data,
            y_data,
            all_combos,
            self.config.num_repeats,
            rng,
        )
        permutations = PermutationTester(self.metric_suite, backend).test(
            X_data,
            y_data,
            all_combos,
            self.config.permutation_tests,
            rng,
            actual_scores=interaction_scores,
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
        chunk_size: int = 1024,
    ) -> Tuple[List[InteractionResult], Dict[Tuple[int, ...], Dict[str, float]]]:
        combos_list = list(combos)
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}

        for i in range(0, len(combos_list), chunk_size):
            chunk = combos_list[i : i + chunk_size]
            chunk_scores = self.backend.score_combos(X, y, chunk, self.metric_suite)
            scores.update(chunk_scores)

        results: List[InteractionResult] = []
        for combo in combos_list:
            metrics = scores[combo]
            results.append(
                InteractionResult(
                    combo=combo,
                    feature_names=tuple(feature_names[idx] for idx in combo),
                    metrics=metrics,
                )
            )
        return results, scores


def _metric_strength(metrics: Dict[str, float]) -> float:
    strengths: List[float] = []
    for name, value in metrics.items():
        if name in ("pearson", "spearman"):
            strengths.append(abs(value))
        else:
            strengths.append(value)
    return max(strengths) if strengths else 0.0


def _make_decision(
    scores: Dict[Tuple[int, ...], Dict[str, float]],
    permutations: List,
    stability: List,
    config: EngineConfig,
) -> Decision:
    perm_map = {result.combo: result.p_values for result in permutations}
    stab_map = {result.combo: result.metrics_std for result in stability}

    signal_detected = False
    for combo, metrics in scores.items():
        p_values = perm_map.get(combo, {})
        stds = stab_map.get(combo, {})
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
