import numpy as np

from gafime.discrete import (
    DiscreteFunctionCandidate,
    rank_discrete_selection_scores,
    score_discrete_selection_candidate,
    score_discrete_selection_candidates,
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

