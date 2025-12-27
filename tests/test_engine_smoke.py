import numpy as np

from gafime.engine import DEFAULT_DIAGNOSIS, GAFIMEEngine


def test_engine_smoke_detects_signal():
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    target = 2.0 * x + rng.normal(scale=0.1, size=200)
    features = np.column_stack([x, y])

    engine = GAFIMEEngine(max_comb_size=2, permutation_rounds=10)
    report = engine.analyze(features, target, feature_names=["x", "y"])

    assert "summary" in report
    assert "feature_scores" in report
    assert "interaction_scores" in report
    assert report["summary"]["diagnosis"] != DEFAULT_DIAGNOSIS
    assert report["summary"]["max_abs_pearson"] > 0.5
    assert len(report["interaction_scores"]) >= 1
