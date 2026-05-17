import numpy as np

from gafime import ComputeBudget, EngineConfig, GafimeEngine


def _benchmark_budget():
    return ComputeBudget(
        max_comb_size=2,
        max_combinations_per_k=20,
        top_features_for_higher_k=6,
        max_discrete_candidates=2000,
        max_thresholds_per_feature=9,
        max_intervals_per_feature=10,
        max_feature_pairs_for_rectangles=8,
        top_k_features_for_discrete=6,
    )


def _run_engine(X, y, *, enable_discrete_functions):
    config = EngineConfig(
        backend="numpy",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=123,
        enable_discrete_functions=enable_discrete_functions,
        budget=_benchmark_budget(),
    )
    return GafimeEngine(config).analyze(X, y)


def _top_abs_pearson(report, *, family=None):
    values = []
    for item in report.interactions:
        if family is not None and item.family != family:
            continue
        values.append(abs(item.metrics.get("pearson", 0.0)))
    return max(values) if values else 0.0


def test_discrete_threshold_improves_step_signal_representation():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(1200, 6))
    y = (X[:, 0] > 0.35).astype(float)
    y += 0.03 * rng.normal(size=X.shape[0])

    continuous = _run_engine(X, y, enable_discrete_functions=False)
    discrete = _run_engine(X, y, enable_discrete_functions=True)

    continuous_score = _top_abs_pearson(continuous)
    discrete_score = _top_abs_pearson(discrete, family="discrete_function")

    assert discrete_score > continuous_score + 0.05


def test_discrete_rectangle_improves_region_signal_representation():
    rng = np.random.default_rng(7)
    X = rng.uniform(-2.0, 2.0, size=(1800, 6))
    in_rectangle = (
        (X[:, 0] >= -0.6)
        & (X[:, 0] <= 0.7)
        & (X[:, 1] >= 0.15)
        & (X[:, 1] <= 1.25)
    )
    y = in_rectangle.astype(float) + 0.03 * rng.normal(size=X.shape[0])

    continuous = _run_engine(X, y, enable_discrete_functions=False)
    discrete = _run_engine(X, y, enable_discrete_functions=True)

    continuous_score = _top_abs_pearson(continuous)
    discrete_rectangle_score = max(
        abs(item.metrics.get("pearson", 0.0))
        for item in discrete.interactions
        if item.candidate_id.startswith("discrete_function_soft_rectangle")
    )

    assert discrete_rectangle_score > continuous_score + 0.10

