"""
GAFIME v0.4.0 discrete-function benchmark runner.

This script is intentionally not a pytest test. It is an application benchmark
that compares:

1. Raw linear model.
2. Continuous-only GAFIME pair features.
3. GAFIME pair + discrete function features.
4. Optional tree baselines when xgboost/lightgbm/catboost are installed.

Feature discovery is fit on the training split only. The same fixed means,
thresholds, intervals, and scales are then applied to the test split.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_friedman1,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gafime import ComputeBudget, EngineConfig, GafimeEngine
from gafime.discrete import (
    discrete_candidate_from_result,
    evaluate_discrete_candidate,
    rank_discrete_selection_scores,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    loader: Callable[[int, int], Tuple[np.ndarray, np.ndarray, List[str]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "cuda", "gpu", "metal", "cpu", "numpy", "core", "cpp"),
        help="Engine backend request. Use auto to exercise the local native build when available.",
    )
    parser.add_argument(
        "--engine-metrics",
        default="pearson,spearman,mutual_info,r2",
        help="Comma-separated GAFIME feature-scoring metrics to benchmark.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON lines")
    args = parser.parse_args()

    specs = [
        DatasetSpec("california_housing", "regression", load_california_housing),
        DatasetSpec("diabetes", "regression", load_diabetes_regression),
        DatasetSpec("friedman1", "regression", load_friedman1),
        DatasetSpec("breast_cancer", "classification", load_breast_cancer_binary),
        DatasetSpec("wine_class_0_vs_rest", "classification", load_wine_class_0_vs_rest),
        DatasetSpec("synthetic_threshold_regression", "regression", load_threshold_regression),
        DatasetSpec("synthetic_threshold_classification", "classification", load_threshold_classification),
        DatasetSpec("synthetic_rectangle_region", "classification", load_rectangle_region),
    ]

    engine_metrics = _parse_engine_metrics(args.engine_metrics)
    for spec in specs:
        for engine_metric in engine_metrics:
            try:
                result = run_one(
                    spec,
                    n_samples=args.samples,
                    seed=args.seed,
                    top_k=args.top_k,
                    backend=args.backend,
                    engine_metric=engine_metric,
                )
            except Exception as exc:
                result = {
                    "dataset": spec.name,
                    "task": spec.task,
                    "backend_request": args.backend,
                    "engine_metric": engine_metric,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            if args.json:
                print(json.dumps(result, sort_keys=True))
            else:
                print_result(result)


def run_one(
    spec: DatasetSpec,
    n_samples: int,
    seed: int,
    top_k: int,
    backend: str,
    engine_metric: str,
) -> Dict[str, object]:
    X, y, feature_names = spec.loader(n_samples, seed)
    stratify = y if spec.task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )

    started = time.perf_counter()
    continuous_features = fit_continuous_gafime_features(
        X_train,
        y_train,
        feature_names,
        task=spec.task,
        top_k=top_k,
        seed=seed,
        backend=backend,
        engine_metric=engine_metric,
    )
    baseline_pred_train = fit_predict_linear_train(
        spec.task,
        augment(X_train, continuous_features),
        y_train,
    )
    discrete_features = fit_discrete_gafime_features(
        X_train,
        y_train,
        feature_names,
        task=spec.task,
        top_k=top_k,
        seed=seed,
        backend=backend,
        engine_metric=engine_metric,
        baseline_pred=baseline_pred_train,
    )
    feature_time = time.perf_counter() - started

    raw_score = fit_eval_linear(spec.task, X_train, y_train, X_test, y_test)
    continuous_score = fit_eval_linear(
        spec.task,
        augment(X_train, continuous_features),
        y_train,
        augment(X_test, continuous_features),
        y_test,
    )
    discrete_score = fit_eval_linear(
        spec.task,
        augment(X_train, continuous_features + discrete_features),
        y_train,
        augment(X_test, continuous_features + discrete_features),
        y_test,
    )

    tree_scores = fit_eval_tree_baselines(spec.task, X_train, y_train, X_test, y_test, seed)

    return {
        "dataset": spec.name,
        "task": spec.task,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "backend_request": backend,
        "engine_metric": engine_metric,
        "continuous_backend": continuous_features[0]["backend"] if continuous_features else None,
        "discrete_backend": discrete_features[0]["backend"] if discrete_features else None,
        "eval_metric": "roc_auc" if spec.task == "classification" else "r2",
        "feature_time_sec": round(feature_time, 4),
        "raw_linear": round(raw_score, 6),
        "gafime_continuous": round(continuous_score, 6),
        "gafime_discrete": round(discrete_score, 6),
        "continuous_features": [item["name"] for item in continuous_features],
        "discrete_features": [item["name"] for item in discrete_features],
        "tree_baselines": tree_scores,
    }


def fit_continuous_gafime_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    *,
    task: str,
    top_k: int,
    seed: int,
    backend: str,
    engine_metric: str,
) -> List[Dict[str, object]]:
    report = GafimeEngine(
        _config(
            enable_discrete=False,
            task=task,
            seed=seed,
            backend=backend,
            engine_metric=engine_metric,
        )
    ).analyze(
        X_train,
        y_train,
        feature_names=feature_names,
    )
    means = X_train.mean(axis=0)
    pairs = [
        item
        for item in report.interactions
        if item.family == "interaction" and len(item.combo) == 2
    ]
    pairs.sort(key=lambda item: _strength(item.metrics), reverse=True)

    features = []
    for item in pairs[:top_k]:
        i, j = item.combo
        features.append(
            {
                "name": " x ".join(item.feature_names),
                "backend": report.backend.name,
                "transform": lambda X, i=i, j=j: (X[:, i] - means[i]) * (X[:, j] - means[j]),
            }
        )
    return features


def fit_discrete_gafime_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    *,
    task: str,
    top_k: int,
    seed: int,
    backend: str,
    engine_metric: str,
    baseline_pred: np.ndarray,
) -> List[Dict[str, object]]:
    engine = GafimeEngine(
        _config(
            enable_discrete=True,
            task=task,
            seed=seed,
            backend=backend,
            engine_metric=engine_metric,
        )
    )
    report = engine.analyze(
        X_train,
        y_train,
        feature_names=feature_names,
    )
    discrete = [item for item in report.interactions if item.family == "discrete_function"]
    candidate_pairs = [(item, discrete_candidate_from_result(item)) for item in discrete]
    selection_scores = engine.backend.score_discrete_selection_candidates(
        X_train,
        y_train,
        [candidate for _, candidate in candidate_pairs],
        baseline_pred=baseline_pred,
    )
    selection_rank = rank_discrete_selection_scores(selection_scores)
    candidate_pairs.sort(
        key=lambda pair: selection_rank.get(pair[1], 0.0),
        reverse=True,
    )

    features = []
    for item, candidate in candidate_pairs[:top_k]:
        features.append(
            {
                "name": item.expression,
                "backend": report.backend.name,
                "selection_score": selection_rank.get(candidate, 0.0),
                "selection_scores": selection_scores.get(candidate, {}),
                "transform": lambda X, candidate=candidate: evaluate_discrete_candidate(X, candidate),
            }
        )
    return features


def augment(X: np.ndarray, features: Iterable[Dict[str, object]]) -> np.ndarray:
    feature_list = list(features)
    if not feature_list:
        return X
    generated = [item["transform"](X) for item in feature_list]
    return np.column_stack([X, *generated])


def fit_eval_linear(task: str, X_train, y_train, X_test, y_test) -> float:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if task == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        return float(roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    return float(r2_score(y_test, model.predict(X_test_scaled)))


def fit_predict_linear_train(task: str, X_train, y_train) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if task == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        return model.predict_proba(X_train_scaled)[:, 1]

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_train_scaled)


def fit_eval_tree_baselines(task: str, X_train, y_train, X_test, y_test, seed: int) -> Dict[str, object]:
    baselines: Dict[str, object] = {}
    for name, factory in _tree_factories(task, seed).items():
        try:
            model = factory()
            model.fit(X_train, y_train)
            if task == "classification":
                score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                score = r2_score(y_test, model.predict(X_test))
            baselines[name] = round(float(score), 6)
        except Exception as exc:
            baselines[name] = f"skipped: {type(exc).__name__}: {exc}"
    return baselines


def _tree_factories(task: str, seed: int):
    factories = {}
    try:
        import xgboost as xgb

        if task == "classification":
            factories["xgboost"] = lambda: xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=2,
            )
        else:
            factories["xgboost"] = lambda: xgb.XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                random_state=seed,
                n_jobs=2,
            )
    except Exception:
        pass

    try:
        import lightgbm as lgb

        if task == "classification":
            factories["lightgbm"] = lambda: lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                random_state=seed,
                verbose=-1,
            )
        else:
            factories["lightgbm"] = lambda: lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.05,
                random_state=seed,
                verbose=-1,
            )
    except Exception:
        pass

    try:
        import catboost as cb

        if task == "classification":
            factories["catboost"] = lambda: cb.CatBoostClassifier(
                iterations=120,
                depth=4,
                learning_rate=0.05,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
        else:
            factories["catboost"] = lambda: cb.CatBoostRegressor(
                iterations=120,
                depth=4,
                learning_rate=0.05,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
    except Exception:
        pass
    return factories


def _config(
    *,
    enable_discrete: bool,
    task: str,
    seed: int,
    backend: str,
    engine_metric: str,
) -> EngineConfig:
    return EngineConfig(
        backend=backend,
        metric_names=(engine_metric,),
        permutation_tests=0,
        num_repeats=1,
        random_seed=seed,
        enable_discrete_functions=enable_discrete,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=120,
            top_features_for_higher_k=24,
            max_discrete_candidates=20_000,
            max_thresholds_per_feature=9,
            max_intervals_per_feature=10,
            max_feature_pairs_for_rectangles=120,
            top_k_features_for_discrete=24,
        ),
    )


def load_california_housing(n_samples: int, seed: int):
    data = fetch_california_housing(as_frame=True)
    X = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64)
    names = list(data.data.columns)
    return _subsample(X, y, names, n_samples, seed)


def load_diabetes_regression(n_samples: int, seed: int):
    data = load_diabetes(as_frame=True)
    X = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.float64)
    names = list(data.data.columns)
    return _subsample(X, y, names, n_samples, seed)


def load_friedman1(n_samples: int, seed: int):
    X, y = make_friedman1(n_samples=n_samples, n_features=10, noise=1.0, random_state=seed)
    return X.astype(np.float64), y.astype(np.float64), [f"f{i}" for i in range(X.shape[1])]


def load_breast_cancer_binary(n_samples: int, seed: int):
    data = load_breast_cancer(as_frame=True)
    X = data.data.to_numpy(dtype=np.float64)
    y = data.target.to_numpy(dtype=np.int64)
    names = list(data.data.columns)
    return _subsample(X, y, names, n_samples, seed)


def load_wine_class_0_vs_rest(n_samples: int, seed: int):
    data = load_wine(as_frame=True)
    X = data.data.to_numpy(dtype=np.float64)
    y = (data.target.to_numpy(dtype=np.int64) == 0).astype(np.int64)
    names = list(data.data.columns)
    return _subsample(X, y, names, n_samples, seed)


def load_threshold_regression(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 12))
    y = (X[:, 0] > 0.35).astype(float) + 0.5 * (X[:, 1] <= -0.75).astype(float)
    y += 0.08 * rng.normal(size=n_samples)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


def load_threshold_classification(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 12))
    logits = 2.0 * (X[:, 0] > 0.2).astype(float) + 1.5 * (X[:, 2] <= -0.5).astype(float)
    logits += 0.15 * rng.normal(size=n_samples)
    y = (logits > np.median(logits)).astype(int)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


def load_rectangle_region(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 12))
    y = (
        (X[:, 0] >= -0.6)
        & (X[:, 0] <= 0.7)
        & (X[:, 1] >= 0.15)
        & (X[:, 1] <= 1.25)
    ).astype(int)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


def _subsample(X, y, names, n_samples: int, seed: int):
    if X.shape[0] <= n_samples:
        return X, y, names
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    return X[idx], y[idx], names


def _strength(metrics: Dict[str, float]) -> float:
    values = []
    for name, value in metrics.items():
        values.append(abs(value) if name in ("pearson", "spearman") else value)
    return max(values) if values else 0.0


def _parse_engine_metrics(value: str) -> List[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    supported = {"pearson", "spearman", "mutual_info", "r2"}
    unsupported = [item for item in metrics if item not in supported]
    if unsupported:
        raise ValueError(f"Unsupported engine metric(s): {unsupported}")
    return metrics or ["pearson"]


def print_result(result: Dict[str, object]) -> None:
    print("\n" + "=" * 80)
    print(
        f"{result.get('dataset')} ({result.get('task')}) "
        f"| engine_metric={result.get('engine_metric')}"
    )
    print("=" * 80)
    if "error" in result:
        print(result["error"])
        return
    print(
        f"eval_metric: {result['eval_metric']} | backend_request: {result['backend_request']} "
        f"| continuous_backend: {result['continuous_backend']} | discrete_backend: {result['discrete_backend']}"
    )
    print(f"samples: {result['n_samples']} | features: {result['n_features']}")
    print(f"feature discovery time: {result['feature_time_sec']}s")
    print(f"raw linear       : {result['raw_linear']}")
    print(f"GAFIME continuous: {result['gafime_continuous']}")
    print(f"GAFIME discrete  : {result['gafime_discrete']}")
    print("tree baselines   :", result["tree_baselines"])
    print("top discrete features:")
    for name in result["discrete_features"][:5]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
