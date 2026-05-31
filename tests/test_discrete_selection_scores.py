import numpy as np
import pytest

from gafime.backends.base import Backend
from gafime.discrete import (
    DiscreteFunctionCandidate,
    batch_discrete_selection_candidates_cache_aware,
    evaluate_discrete_candidate,
    evaluate_discrete_mask,
    mi_bin_template_from_discrete_selection_template,
    rank_discrete_selection_scores,
    score_discrete_candidates,
    score_discrete_selection_candidate,
    score_discrete_selection_candidates,
)
from gafime.metrics import MetricSuite
from gafime.metrics.cpu_metrics import (
    adaptive_min_effective_support,
    dense_mutual_info,
    mi_bin_template_capacity,
    select_adaptive_mi_bins,
    soft_binary_mutual_info,
)


def test_adaptive_mi_bins_scale_with_sample_count_and_cap():
    assert select_adaptive_mi_bins(50, max_bins=96, samples_per_bin=8, dimensions=2) == 2
    assert select_adaptive_mi_bins(1_000, max_bins=96, samples_per_bin=8, dimensions=2) == 8
    assert select_adaptive_mi_bins(100_000, max_bins=96, samples_per_bin=25, dimensions=1) == 96
    assert select_adaptive_mi_bins(100_000, max_bins=16, samples_per_bin=25, dimensions=1) == 16


def test_mi_bin_template_capacity_rounds_to_static_kernel_shape():
    assert mi_bin_template_capacity(1) == 2
    assert mi_bin_template_capacity(3) == 4
    assert mi_bin_template_capacity(17) == 32
    assert mi_bin_template_capacity(96) == 96


def test_effective_support_floor_adapts_for_small_datasets():
    assert adaptive_min_effective_support(96) == pytest.approx(3.0)
    assert adaptive_min_effective_support(250) == pytest.approx(5.0)
    assert adaptive_min_effective_support(5000) == pytest.approx(8.0)


def test_discrete_selection_batches_are_homogeneous_by_mi_template():
    candidates = [
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(0,),
            thresholds=(0.0,),
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_rectangle",
            feature_indices=(0, 1),
            intervals=((-1.0, 1.0), (-1.0, 1.0)),
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(2,),
            thresholds=(0.0,),
        ),
    ]

    batches = batch_discrete_selection_candidates_cache_aware(
        candidates,
        mi_bin_template=32,
        max_blocks=1,
    )
    flattened = [candidate for _, batch in batches for candidate in batch]

    assert sorted(flattened, key=id) == sorted(candidates, key=id)
    assert all(
        mi_bin_template_from_discrete_selection_template(template_id) == 32
        for template_id, _ in batches
    )


def test_dense_mi_independent_features_are_bias_corrected():
    rng = np.random.default_rng(7)
    values = []
    for _ in range(16):
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        values.append(dense_mutual_info(x, y, max_bins=96))

    assert float(np.mean(values)) < 0.03


def test_soft_binary_mi_handles_small_and_large_sample_sizes():
    rng = np.random.default_rng(8)
    for n in (80, 20_000):
        x = rng.normal(size=n)
        mask = 1.0 / (1.0 + np.exp(-12.0 * (x - 0.15)))
        y_signal = (x > 0.15).astype(float) + 0.05 * rng.normal(size=n)
        y_noise = rng.normal(size=n)

        assert soft_binary_mutual_info(mask, y_signal, max_bins=96) > 0.1
        assert soft_binary_mutual_info(mask, y_noise, max_bins=96) < 0.05


def test_soft_binary_mi_matches_manual_corrected_binary_contingency():
    rng = np.random.default_rng(19)
    mask = (rng.random(1000) > 0.45).astype(float)
    y = (mask + 0.35 * rng.normal(size=mask.shape[0]) > 0.5).astype(float)

    actual = soft_binary_mutual_info(mask, y, max_bins=96)
    expected = _manual_corrected_mi(
        np.array(
            [
                [
                    float(np.sum((mask == 1.0) & (y == 0.0))),
                    float(np.sum((mask == 1.0) & (y == 1.0))),
                ],
                [
                    float(np.sum((mask == 0.0) & (y == 0.0))),
                    float(np.sum((mask == 0.0) & (y == 1.0))),
                ],
            ]
        )
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_discrete_selector_ranking_matches_sklearn_mi_on_planted_regions():
    feature_selection = pytest.importorskip("sklearn.feature_selection")

    for seed in (101, 202, 303):
        X, y, candidates, true_ids = _planted_region_case(seed=seed, n_samples=5000)
        scores = score_discrete_selection_candidates(X, y, candidates, mi_bins=96)
        gafime_mi = np.array([scores[candidate]["mutual_info"] for candidate in candidates])
        gafime_composite = rank_discrete_selection_scores(scores)
        masks = np.column_stack([evaluate_discrete_mask(X, candidate) for candidate in candidates])
        sklearn_mi = feature_selection.mutual_info_regression(
            masks,
            y,
            random_state=seed,
            n_neighbors=5,
        )

        sklearn_top = _top_ids(candidates, sklearn_mi, k=len(true_ids))
        gafime_mi_top = _top_ids(candidates, gafime_mi, k=len(true_ids))
        gafime_composite_top = [
            candidate.candidate_id
            for candidate, _ in sorted(
                gafime_composite.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: len(true_ids)]
        ]

        assert set(sklearn_top) == true_ids
        assert set(gafime_mi_top) == true_ids
        assert set(gafime_composite_top) == true_ids

        noise_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.candidate_id not in true_ids
        ]
        signal_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.candidate_id in true_ids
        ]
        assert float(np.min(gafime_mi[signal_indices])) > float(np.max(gafime_mi[noise_indices])) + 0.05
        assert float(np.min(sklearn_mi[signal_indices])) > float(np.max(sklearn_mi[noise_indices])) + 0.05


def test_small_sample_rectangle_is_not_pruned_by_support_guard():
    X, y, candidates, true_ids = _planted_region_case(seed=11, n_samples=96)
    scores = score_discrete_selection_candidates(X, y, candidates, mi_bins=96)
    ranked = rank_discrete_selection_scores(scores)
    top = [
        candidate.candidate_id
        for candidate, _ in sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:2]
    ]

    assert set(top) == true_ids
    assert scores[candidates[1]]["mutual_info"] > 0.01
    assert scores[candidates[1]]["variance_reduction"] > 0.01


def test_discrete_selector_noise_floor_stays_low_against_sklearn_mi():
    feature_selection = pytest.importorskip("sklearn.feature_selection")

    for seed in (404, 505, 606):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(4000, 12)).astype(np.float32)
        y = rng.normal(size=X.shape[0])
        candidates = _noise_candidates(n_features=X.shape[1])

        scores = score_discrete_selection_candidates(X, y, candidates, mi_bins=96)
        gafime_mi = np.array([scores[candidate]["mutual_info"] for candidate in candidates])
        masks = np.column_stack([evaluate_discrete_mask(X, candidate) for candidate in candidates])
        sklearn_mi = feature_selection.mutual_info_regression(
            masks,
            y,
            random_state=seed,
            n_neighbors=5,
        )

        assert float(np.max(gafime_mi)) < 0.025
        assert float(np.mean(gafime_mi)) < 0.010
        assert float(np.max(sklearn_mi)) < 0.050


def test_discrete_selector_mi_increases_with_signal_strength_like_sklearn():
    feature_selection = pytest.importorskip("sklearn.feature_selection")

    rng = np.random.default_rng(707)
    X = rng.normal(size=(5000, 5)).astype(np.float32)
    candidate = DiscreteFunctionCandidate(
        kind="discrete_function_soft_threshold",
        feature_indices=(0,),
        thresholds=(0.15,),
        direction="ge",
        scales=(1.0,),
        sharpness=22.0,
        candidate_id="threshold",
    )
    mask = evaluate_discrete_mask(X, candidate)
    noise = rng.normal(size=X.shape[0])
    gafime_values = []
    sklearn_values = []

    for strength in (0.25, 0.75, 1.50):
        y = strength * mask + 0.4 * noise
        scores = score_discrete_selection_candidate(X, y, candidate, mi_bins=96)
        gafime_values.append(scores["mutual_info"])
        sklearn_values.append(
            float(
                feature_selection.mutual_info_regression(
                    mask.reshape(-1, 1),
                    y,
                    random_state=int(strength * 100),
                    n_neighbors=5,
                )[0]
            )
        )

    assert gafime_values[0] < gafime_values[1] < gafime_values[2]
    assert sklearn_values[0] < sklearn_values[1] < sklearn_values[2]


def test_core_backend_mutual_info_matches_python_when_available():
    try:
        from gafime.backends.core_backend import CoreBackend
    except Exception as exc:
        pytest.skip(f"Core backend unavailable: {exc}")

    rng = np.random.default_rng(9)
    X = rng.normal(size=(600, 5))
    y = np.sin(X[:, 0]) + (X[:, 1] > 0.25).astype(float)
    combos = [(0,), (1,), (0, 1), (2, 3)]
    suite = MetricSuite(("mutual_info",), mi_bins=96)

    try:
        core = CoreBackend()
    except Exception as exc:
        pytest.skip(f"Core backend unavailable: {exc}")

    expected = Backend().score_combos(X, y, combos, suite)
    actual = core.score_combos(X, y, combos, suite)

    for combo in combos:
        assert actual[combo]["mutual_info"] == pytest.approx(
            expected[combo]["mutual_info"], abs=1e-10
        )


def test_discrete_selection_scores_prefer_rectangle_region():
    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(1200, 4))
    y = (
        (X[:, 0] >= -0.5)
        & (X[:, 0] <= 0.6)
        & (X[:, 1] >= 0.1)
        & (X[:, 1] <= 1.1)
    ).astype(float)

    good = DiscreteFunctionCandidate(
        kind="discrete_function_soft_rectangle",
        feature_indices=(0, 1),
        intervals=((-0.5, 0.6), (0.1, 1.1)),
        scales=(1.0, 1.0),
        sharpness=20.0,
    )
    bad = DiscreteFunctionCandidate(
        kind="discrete_function_soft_rectangle",
        feature_indices=(2, 3),
        intervals=((-0.5, 0.6), (0.1, 1.1)),
        scales=(1.0, 1.0),
        sharpness=20.0,
    )

    scores = score_discrete_selection_candidates(X, y, [good, bad])
    ranked = rank_discrete_selection_scores(scores)

    assert scores[good]["mutual_info"] > scores[bad]["mutual_info"]
    assert scores[good]["variance_reduction"] > scores[bad]["variance_reduction"] + 0.10
    assert ranked[good] > ranked[bad]


def test_residual_gain_scores_signal_after_baseline():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(1500, 3))
    baseline = 2.0 * X[:, 0]
    y = baseline + 1.5 * (X[:, 1] > 0.25).astype(float)
    y += 0.05 * rng.normal(size=X.shape[0])

    candidate = DiscreteFunctionCandidate(
        kind="discrete_function_soft_threshold",
        feature_indices=(1,),
        thresholds=(0.25,),
        direction="ge",
        scales=(1.0,),
        sharpness=20.0,
    )

    scores = score_discrete_selection_candidate(
        X,
        y,
        candidate,
        baseline_pred=baseline,
    )

    assert scores["residual_abs_corr"] > 0.90
    assert scores["residual_r2_gain"] > 0.80
    assert scores["variance_reduction"] > 0.05


def test_variance_reduction_matches_manual_sse_and_sklearn_tree_stump():
    tree_mod = pytest.importorskip("sklearn.tree")

    rng = np.random.default_rng(808)
    X = rng.normal(size=(2000, 4))
    y = 1.7 * (X[:, 0] > 0.15).astype(float) + 0.2 * rng.normal(size=X.shape[0])
    candidate = DiscreteFunctionCandidate(
        kind="discrete_function_soft_threshold",
        feature_indices=(0,),
        thresholds=(0.15,),
        direction="ge",
        scales=(1.0,),
        mode="hard",
        candidate_id="hard_threshold",
    )
    mask = evaluate_discrete_mask(X, candidate)
    scores = score_discrete_selection_candidate(X, y, candidate)

    manual_gain = _manual_variance_reduction(y, mask)
    stump = tree_mod.DecisionTreeRegressor(max_depth=1, random_state=0)
    stump.fit(mask.reshape(-1, 1), y)
    sklearn_gain = float(stump.score(mask.reshape(-1, 1), y))

    assert scores["variance_reduction"] == pytest.approx(manual_gain, abs=1e-12)
    assert scores["variance_reduction"] == pytest.approx(sklearn_gain, abs=1e-12)


def test_residual_r2_gain_matches_sklearn_linear_regression():
    linear_model = pytest.importorskip("sklearn.linear_model")

    rng = np.random.default_rng(909)
    X = rng.normal(size=(2500, 5))
    baseline = 2.0 * X[:, 0] - 0.5 * X[:, 2]
    y = baseline + 1.4 * (X[:, 1] > -0.2).astype(float) + 0.08 * rng.normal(size=X.shape[0])
    candidate = DiscreteFunctionCandidate(
        kind="discrete_function_soft_threshold",
        feature_indices=(1,),
        thresholds=(-0.2,),
        direction="ge",
        scales=(1.0,),
        sharpness=24.0,
        candidate_id="residual_threshold",
    )

    scores = score_discrete_selection_candidate(
        X,
        y,
        candidate,
        baseline_pred=baseline,
    )
    feature = evaluate_discrete_candidate(X, candidate).reshape(-1, 1)
    residual = y - baseline
    model = linear_model.LinearRegression().fit(feature, residual)

    assert scores["residual_r2_gain"] == pytest.approx(
        float(model.score(feature, residual)),
        abs=1e-12,
    )


def test_public_r2_metric_matches_sklearn_univariate_linear_regression():
    linear_model = pytest.importorskip("sklearn.linear_model")

    rng = np.random.default_rng(1001)
    X = rng.normal(size=(1800, 4))
    y = 1.3 * np.tanh(X[:, 0]) + 0.2 * rng.normal(size=X.shape[0])
    candidate = DiscreteFunctionCandidate(
        kind="discrete_function_value_gated_threshold",
        feature_indices=(1,),
        thresholds=(-0.1,),
        direction="ge",
        value_feature=0,
        scales=(1.0,),
        sharpness=18.0,
        candidate_id="value_gate",
    )
    suite = MetricSuite(("r2",))
    scores = score_discrete_candidates(X, y, [candidate], suite)
    feature = evaluate_discrete_candidate(X, candidate).reshape(-1, 1)
    model = linear_model.LinearRegression().fit(feature, y)

    assert scores[candidate]["r2"] == pytest.approx(
        float(model.score(feature, y)),
        abs=1e-12,
    )


def test_selector_ranking_matches_sklearn_cv_ridge_gain_on_powered_candidates():
    linear_model = pytest.importorskip("sklearn.linear_model")
    model_selection = pytest.importorskip("sklearn.model_selection")

    X, y, candidates, true_ids = _planted_region_case(seed=1111, n_samples=3000)
    scores = score_discrete_selection_candidates(X, y, candidates, mi_bins=96)
    gafime_top = _top_ranked_ids(candidates, rank_discrete_selection_scores(scores), k=2)

    cv = model_selection.KFold(n_splits=4, shuffle=True, random_state=1111)
    ridge_scores = []
    for candidate in candidates:
        feature = evaluate_discrete_candidate(X, candidate).reshape(-1, 1)
        ridge_scores.append(
            float(
                np.mean(
                    model_selection.cross_val_score(
                        linear_model.Ridge(alpha=1.0),
                        feature,
                        y,
                        cv=cv,
                        scoring="r2",
                    )
                )
            )
        )

    assert set(gafime_top) == true_ids
    assert set(_top_ids(candidates, ridge_scores, k=2)) == true_ids


def test_selector_ranking_matches_sklearn_cv_gbm_gain_on_powered_candidates():
    ensemble = pytest.importorskip("sklearn.ensemble")
    model_selection = pytest.importorskip("sklearn.model_selection")

    X, y, candidates, true_ids = _planted_region_case(seed=1212, n_samples=2500)
    scores = score_discrete_selection_candidates(X, y, candidates, mi_bins=96)
    gafime_top = _top_ranked_ids(candidates, rank_discrete_selection_scores(scores), k=2)

    cv = model_selection.KFold(n_splits=3, shuffle=True, random_state=1212)
    gbm_scores = []
    for candidate in candidates:
        feature = evaluate_discrete_candidate(X, candidate).reshape(-1, 1)
        estimator = ensemble.HistGradientBoostingRegressor(
            max_iter=40,
            max_leaf_nodes=8,
            learning_rate=0.08,
            random_state=1212,
        )
        gbm_scores.append(
            float(
                np.mean(
                    model_selection.cross_val_score(
                        estimator,
                        feature,
                        y,
                        cv=cv,
                        scoring="r2",
                    )
                )
            )
        )

    assert set(gafime_top) == true_ids
    assert set(_top_ids(candidates, gbm_scores, k=2)) == true_ids


def _manual_corrected_mi(joint: np.ndarray) -> float:
    total = float(np.sum(joint))
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)
    pxy = joint / total
    expected = (px / total)[:, None] * (py / total)[None, :]
    valid = (pxy > 0.0) & (expected > 0.0)
    raw = float(np.sum(pxy[valid] * np.log(pxy[valid] / expected[valid])))
    nonzero_rows = int(np.count_nonzero(px > 0.0))
    nonzero_cols = int(np.count_nonzero(py > 0.0))
    bias = ((nonzero_rows - 1) * (nonzero_cols - 1)) / (2.0 * total)
    return max(raw - bias, 0.0)


def _manual_variance_reduction(y: np.ndarray, mask: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    total_sse = float(np.sum((y - np.mean(y)) ** 2))
    if total_sse <= 0.0:
        return 0.0
    left = mask > 0.5
    right = ~left
    split_sse = float(np.sum((y[left] - np.mean(y[left])) ** 2))
    split_sse += float(np.sum((y[right] - np.mean(y[right])) ** 2))
    return max((total_sse - split_sse) / total_sse, 0.0)


def _planted_region_case(seed: int, n_samples: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 12)).astype(np.float32)
    threshold = X[:, 0] > 0.35
    rectangle = (
        (X[:, 1] > -0.8)
        & (X[:, 1] < 0.4)
        & (X[:, 2] > 0.2)
        & (X[:, 2] < 1.2)
    )
    y = (
        1.5 * threshold.astype(float)
        + 1.8 * rectangle.astype(float)
        + 0.12 * rng.normal(size=n_samples)
    )
    candidates = [
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(0,),
            thresholds=(0.35,),
            direction="ge",
            scales=(1.0,),
            sharpness=24.0,
            candidate_id="true_threshold",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_rectangle",
            feature_indices=(1, 2),
            intervals=((-0.8, 0.4), (0.2, 1.2)),
            scales=(1.0, 1.0),
            sharpness=24.0,
            candidate_id="true_rectangle",
        ),
    ]
    candidates.extend(_noise_candidates(n_features=X.shape[1]))
    return X, y, candidates, {"true_threshold", "true_rectangle"}


def _noise_candidates(n_features: int):
    candidates = []
    for feature in range(3, n_features):
        candidates.append(
            DiscreteFunctionCandidate(
                kind="discrete_function_soft_threshold",
                feature_indices=(feature,),
                thresholds=(0.0,),
                direction="ge",
                scales=(1.0,),
                sharpness=24.0,
                candidate_id=f"noise_threshold_{feature}",
            )
        )
    for feature_a, feature_b in ((3, 4), (5, 6), (7, 8), (9, 10)):
        if feature_b < n_features:
            candidates.append(
                DiscreteFunctionCandidate(
                    kind="discrete_function_soft_rectangle",
                    feature_indices=(feature_a, feature_b),
                    intervals=((-0.75, 0.25), (-0.5, 0.6)),
                    scales=(1.0, 1.0),
                    sharpness=24.0,
                    candidate_id=f"noise_rectangle_{feature_a}_{feature_b}",
                )
            )
    return candidates


def _top_ids(candidates, values, k: int):
    return [
        candidates[index].candidate_id
        for index in np.argsort(np.asarray(values, dtype=float))[::-1][:k]
    ]


def _top_ranked_ids(candidates, ranked, k: int):
    candidates_set = set(candidates)
    return [
        candidate.candidate_id
        for candidate, _ in sorted(
            ((candidate, value) for candidate, value in ranked.items() if candidate in candidates_set),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
    ]
