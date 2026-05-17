import numpy as np
import pytest

import gafime.engine as engine_module
from gafime import ComputeBudget, EngineConfig, GafimeEngine
from gafime.backends.base import Backend, BackendInfo
from gafime.config import DEFAULT_DISCRETE_QUANTILES
from gafime.discrete import (
    GPU_HARD_MODE_ERROR,
    DiscreteFunctionCandidate,
    discrete_candidate_from_result,
    evaluate_discrete_candidate,
)


def _threshold_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 5)).astype(np.float64)
    y = (X[:, 0] > 0.25).astype(np.float64)
    y += 0.05 * rng.normal(size=X.shape[0])
    return X, y


def _small_discrete_budget():
    return ComputeBudget(
        max_comb_size=2,
        max_combinations_per_k=10,
        top_features_for_higher_k=4,
        max_discrete_candidates=120,
        max_thresholds_per_feature=5,
        max_intervals_per_feature=4,
        max_feature_pairs_for_rectangles=2,
        top_k_features_for_discrete=3,
    )


def test_default_discrete_quantiles_include_requested_variants():
    assert 0.40 in DEFAULT_DISCRETE_QUANTILES
    assert 0.65 in DEFAULT_DISCRETE_QUANTILES
    assert len(DEFAULT_DISCRETE_QUANTILES) == 9


def test_engine_adds_discrete_candidates_and_keeps_user_metric():
    X, y = _threshold_data()
    config = EngineConfig(
        backend="numpy",
        metric_names=("mutual_info",),
        permutation_tests=0,
        num_repeats=1,
        enable_discrete_functions=True,
        budget=_small_discrete_budget(),
    )

    report = GafimeEngine(config).analyze(X, y)
    discrete = [item for item in report.interactions if item.family == "discrete_function"]

    assert discrete
    assert all(tuple(item.metrics) == ("mutual_info",) for item in discrete)
    assert any(item.candidate_id.startswith("discrete_function_soft_threshold") for item in discrete)
    assert any(item.candidate_id.startswith("discrete_function_soft_interval") for item in discrete)
    assert any(item.candidate_id.startswith("discrete_function_value_gated_threshold") for item in discrete)
    assert any(item.candidate_id.startswith("discrete_function_soft_rectangle") for item in discrete)
    assert any(item.candidate_id.startswith("discrete_function_value_in_soft_rectangle") for item in discrete)

    reconstructed = discrete_candidate_from_result(discrete[0])
    assert reconstructed.candidate_id == discrete[0].candidate_id


def test_engine_ranks_discrete_candidates_with_split_aware_default():
    X, y = _threshold_data(seed=12)
    config = EngineConfig(
        backend="numpy",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        enable_discrete_functions=True,
        budget=_small_discrete_budget(),
    )

    report = GafimeEngine(config).analyze(X, y)
    discrete = [item for item in report.interactions if item.family == "discrete_function"]
    scores = [item.params["selection_score"] for item in discrete]

    assert discrete
    assert discrete[0].params["discrete_ranking"] == "split_aware"
    assert "selection_scores" in discrete[0].params
    assert scores == sorted(scores, reverse=True)
    assert any(item.params["selection_scores"]["mutual_info"] > 0.0 for item in discrete)


def test_engine_can_rank_discrete_candidates_by_report_metric():
    X, y = _threshold_data(seed=13)
    config = EngineConfig(
        backend="numpy",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        enable_discrete_functions=True,
        discrete_ranking="metric",
        budget=_small_discrete_budget(),
    )

    report = GafimeEngine(config).analyze(X, y)
    discrete = [item for item in report.interactions if item.family == "discrete_function"]
    strengths = [abs(item.metrics["pearson"]) for item in discrete]

    assert discrete
    assert discrete[0].params["discrete_ranking"] == "metric"
    assert "selection_scores" not in discrete[0].params
    assert strengths == sorted(strengths, reverse=True)


def test_numpy_hard_mode_discrete_threshold_outputs_binary_mask():
    X, _y = _threshold_data()
    candidate = DiscreteFunctionCandidate(
        kind="discrete_function_soft_threshold",
        feature_indices=(0,),
        thresholds=(0.0,),
        direction="ge",
        mode="hard",
        candidate_id="hard-threshold-test",
    )

    mask = evaluate_discrete_candidate(X, candidate)

    assert set(np.unique(mask)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(mask, (X[:, 0] >= 0.0).astype(float))


def test_gpu_hard_mode_raises(monkeypatch):
    class FakeGpuBackend(Backend):
        def info(self):
            return BackendInfo(
                name="fake-gpu",
                device="cuda:test",
                is_gpu=True,
                memory_total_mb=1024,
                memory_free_mb=512,
            )

    def fake_resolve_backend(_config, _X, _y):
        return FakeGpuBackend(), []

    monkeypatch.setattr(engine_module, "resolve_backend", fake_resolve_backend)

    X, y = _threshold_data()
    config = EngineConfig(
        backend="cuda",
        metric_names=("pearson",),
        enable_discrete_functions=True,
        discrete_mode="hard",
        budget=_small_discrete_budget(),
    )

    with pytest.raises(ValueError, match=GPU_HARD_MODE_ERROR):
        GafimeEngine(config).analyze(X, y)
