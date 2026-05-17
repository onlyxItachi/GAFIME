"""
Focused high-budget California Housing benchmark for GAFIME v0.4.0.

This is an application benchmark, not a pytest test. It intentionally uses the
local checkout and native backend, fits feature generation on train only, then
applies the fixed generated transforms to the held-out test split.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gafime import ComputeBudget, EngineConfig, GafimeEngine
from gafime.discrete import (
    DiscreteFunctionCandidate,
    discrete_candidate_from_result,
    evaluate_discrete_candidate,
    rank_discrete_selection_scores,
)


@dataclass(frozen=True)
class GeneratedFeature:
    name: str
    family: str
    strength: float
    selection_scores: dict[str, float]
    candidate: DiscreteFunctionCandidate | None
    transform: Callable[[np.ndarray], np.ndarray]


def main() -> None:
    data = fetch_california_housing(as_frame=True)
    X = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64)
    names = list(data.data.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    engine = GafimeEngine(_heavy_config())
    started = time.perf_counter()
    report = engine.analyze(X_train, y_train, feature_names=names)
    engine_sec = time.perf_counter() - started

    continuous = _continuous_features(report, X_train)
    discrete = _discrete_features(report)
    baseline_pred_train = _fit_predict_ridge(_augment(X_train, continuous), y_train)
    selection_pool = _selection_pool(discrete, per_group=96)
    selector_started = time.perf_counter()
    selection_scores = engine.backend.score_discrete_selection_candidates(
        X_train,
        y_train,
        [feature.candidate for feature in selection_pool if feature.candidate is not None],
        baseline_pred=baseline_pred_train,
    )
    selector_sec = time.perf_counter() - selector_started
    balanced_rank = rank_discrete_selection_scores(selection_scores)
    residual_rank = rank_discrete_selection_scores(
        selection_scores,
        weights={
            "mutual_info": 0.15,
            "variance_reduction": 0.20,
            "residual_r2_gain": 0.65,
        },
    )
    discrete_ranked = _with_selection_scores(selection_pool, selection_scores, balanced_rank)
    residual_ranked = _with_selection_scores(selection_pool, selection_scores, residual_rank)
    discrete_top = _select_top(discrete_ranked, 2000)
    discrete_diverse = _select_diverse(discrete_ranked, 2000, per_group=16)
    residual_top = _select_top(residual_ranked, 2000)
    residual_diverse = _select_diverse(residual_ranked, 2000, per_group=16)

    results = {
        "dataset": "california_housing",
        "samples": int(X.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "backend": report.backend.name,
        "engine_metric": "pearson",
        "engine_time_sec": round(engine_sec, 4),
        "n_interactions": len(report.interactions),
        "n_continuous_features": len(continuous),
        "n_discrete_features": len(discrete),
        "n_selection_pool": len(selection_pool),
        "selection_backend": engine.backend.name,
        "selection_time_sec": round(selector_sec, 4),
        "scores": {},
        "cv_scores": {},
        "top_discrete": [item.name for item in discrete_top[:15]],
        "top_diverse_discrete": [item.name for item in discrete_diverse[:15]],
        "top_residual_discrete": [item.name for item in residual_top[:15]],
        "top_residual_diverse_discrete": [item.name for item in residual_diverse[:15]],
        "top_discrete_selection_scores": [item.selection_scores for item in discrete_top[:5]],
        "top_residual_selection_scores": [item.selection_scores for item in residual_top[:5]],
    }

    results["scores"]["raw_ridgecv"] = _score_ridge(X_train, y_train, X_test, y_test)
    results["scores"]["continuous_all_ridgecv"] = _score_ridge(
        _augment(X_train, continuous),
        y_train,
        _augment(X_test, continuous),
        y_test,
    )

    for k in (25, 50, 100, 200, 500, 1000, 2000):
        top_k = discrete_top[:k]
        diverse_k = discrete_diverse[:k]
        results["scores"][f"continuous_plus_top{k}_ridgecv"] = _score_ridge(
            _augment(X_train, [*continuous, *top_k]),
            y_train,
            _augment(X_test, [*continuous, *top_k]),
            y_test,
        )
        results["scores"][f"continuous_plus_diverse{k}_ridgecv"] = _score_ridge(
            _augment(X_train, [*continuous, *diverse_k]),
            y_train,
            _augment(X_test, [*continuous, *diverse_k]),
            y_test,
        )
        if k in (500, 1000, 2000):
            residual_top_k = residual_top[:k]
            residual_diverse_k = residual_diverse[:k]
            results["scores"][f"continuous_plus_residual_top{k}_ridgecv"] = _score_ridge(
                _augment(X_train, [*continuous, *residual_top_k]),
                y_train,
                _augment(X_test, [*continuous, *residual_top_k]),
                y_test,
            )
            results["scores"][f"continuous_plus_residual_diverse{k}_ridgecv"] = _score_ridge(
                _augment(X_train, [*continuous, *residual_diverse_k]),
                y_train,
                _augment(X_test, [*continuous, *residual_diverse_k]),
                y_test,
            )

    results["scores"]["hist_gradient_boosting_raw"] = _score_hgb(X_train, y_train, X_test, y_test)
    results["cv_scores"]["raw_ridgecv"] = _score_cv_ridge(X_train, y_train)
    results["cv_scores"]["continuous_all_ridgecv"] = _score_cv_ridge(
        _augment(X_train, continuous),
        y_train,
    )
    results["cv_scores"]["continuous_plus_top500_ridgecv"] = _score_cv_ridge(
        _augment(X_train, [*continuous, *discrete_top[:500]]),
        y_train,
    )
    results["cv_scores"]["continuous_plus_diverse500_ridgecv"] = _score_cv_ridge(
        _augment(X_train, [*continuous, *discrete_diverse[:500]]),
        y_train,
    )
    results["cv_scores"]["continuous_plus_residual_diverse500_ridgecv"] = _score_cv_ridge(
        _augment(X_train, [*continuous, *residual_diverse[:500]]),
        y_train,
    )

    print(json.dumps(results, indent=2, sort_keys=True))


def _heavy_config() -> EngineConfig:
    dense_quantiles = (
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.98,
    )
    return EngineConfig(
        backend="auto",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=42,
        enable_discrete_functions=True,
        discrete_mode="soft",
        discrete_gate_sharpness=12.0,
        discrete_quantiles=dense_quantiles,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=100_000,
            top_features_for_higher_k=8,
            keep_in_vram=True,
            vram_budget_mb=6144,
            max_discrete_candidates=400_000,
            max_thresholds_per_feature=len(dense_quantiles),
            max_intervals_per_feature=60,
            max_feature_pairs_for_rectangles=28,
            top_k_features_for_discrete=8,
        ),
    )


def _continuous_features(report, X_train: np.ndarray) -> List[GeneratedFeature]:
    means = X_train.mean(axis=0)
    features: List[GeneratedFeature] = []
    for item in report.interactions:
        if item.family != "interaction" or len(item.combo) != 2:
            continue
        i, j = item.combo
        features.append(
            GeneratedFeature(
                name=" x ".join(item.feature_names),
                family="interaction",
                strength=_strength(item.metrics),
                selection_scores={},
                candidate=None,
                transform=lambda X, i=i, j=j: (X[:, i] - means[i]) * (X[:, j] - means[j]),
            )
        )
    features.sort(key=lambda item: item.strength, reverse=True)
    return features


def _discrete_features(report) -> List[GeneratedFeature]:
    features: List[GeneratedFeature] = []
    for item in report.interactions:
        if item.family != "discrete_function":
            continue
        candidate = discrete_candidate_from_result(item)
        features.append(
            GeneratedFeature(
                name=item.expression,
                family="discrete_function",
                strength=_strength(item.metrics),
                selection_scores={},
                candidate=candidate,
                transform=lambda X, candidate=candidate: evaluate_discrete_candidate(X, candidate),
            )
        )
    return features


def _select_top(features: List[GeneratedFeature], k: int) -> List[GeneratedFeature]:
    return sorted(features, key=lambda item: item.strength, reverse=True)[:k]


def _selection_pool(
    features: List[GeneratedFeature],
    *,
    per_group: int,
) -> List[GeneratedFeature]:
    groups: dict[tuple[object, ...], List[GeneratedFeature]] = {}
    for feature in features:
        candidate = feature.candidate
        if candidate is None:
            continue
        key = (
            candidate.kind,
            candidate.feature_indices,
            candidate.value_feature,
        )
        groups.setdefault(key, []).append(feature)

    pool: List[GeneratedFeature] = []
    for group_features in groups.values():
        if len(group_features) <= per_group:
            pool.extend(group_features)
            continue
        indices = np.linspace(0, len(group_features) - 1, num=per_group, dtype=np.int64)
        pool.extend(group_features[int(idx)] for idx in indices)
    return pool


def _with_selection_scores(
    features: List[GeneratedFeature],
    scores: dict[DiscreteFunctionCandidate, dict[str, float]],
    ranks: dict[DiscreteFunctionCandidate, float],
) -> List[GeneratedFeature]:
    ranked: List[GeneratedFeature] = []
    for feature in features:
        candidate = feature.candidate
        if candidate is None:
            continue
        selection_scores = dict(scores.get(candidate, {}))
        selection_score = float(ranks.get(candidate, 0.0))
        selection_scores["selection_score"] = selection_score
        ranked.append(
            GeneratedFeature(
                name=feature.name,
                family=feature.family,
                strength=selection_score,
                selection_scores=selection_scores,
                candidate=candidate,
                transform=feature.transform,
            )
        )
    ranked.sort(key=lambda item: item.strength, reverse=True)
    return ranked


def _select_diverse(
    features: List[GeneratedFeature],
    k: int,
    *,
    per_group: int,
) -> List[GeneratedFeature]:
    selected: List[GeneratedFeature] = []
    counts: dict[str, int] = {}
    for feature in features:
        group = _feature_group(feature.name)
        count = counts.get(group, 0)
        if count >= per_group:
            continue
        selected.append(feature)
        counts[group] = count + 1
        if len(selected) >= k:
            break
    return selected


def _feature_group(name: str) -> str:
    if "rect(" in name:
        return name.split("rect(", 1)[1].split(")", 1)[0]
    if "mask(" in name:
        return name.split("mask(", 1)[1].split(" ", 1)[0]
    if "(" in name:
        return name.split("(", 1)[1].split(" ", 1)[0]
    return name


def _augment(X: np.ndarray, features: Iterable[GeneratedFeature]) -> np.ndarray:
    feature_list = list(features)
    if not feature_list:
        return X
    generated = [feature.transform(X) for feature in feature_list]
    return np.column_stack([X, *generated])


def _score_ridge(X_train, y_train, X_test, y_test) -> float:
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 5, 24)),
    )
    model.fit(X_train, y_train)
    return round(float(r2_score(y_test, model.predict(X_test))), 6)


def _fit_predict_ridge(X_train, y_train) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 5, 24)),
    )
    model.fit(X_train, y_train)
    return model.predict(X_train)


def _score_cv_ridge(X_train, y_train) -> float:
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 5, 24)),
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    return round(float(np.mean(scores)), 6)


def _score_hgb(X_train, y_train, X_test, y_test) -> float:
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return round(float(r2_score(y_test, model.predict(X_test))), 6)


def _strength(metrics: dict[str, float]) -> float:
    values = []
    for name, value in metrics.items():
        values.append(abs(value) if name in ("pearson", "spearman") else value)
    return max(values) if values else 0.0


if __name__ == "__main__":
    main()
