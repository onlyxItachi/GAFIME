"""Smoke tests for GAFIME engine."""

import numpy as np

from gafime.engine import GAFIMEEngine


def test_engine_runs_and_reports_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] * 0.8 + rng.normal(scale=0.1, size=200)

    engine = GAFIMEEngine()
    report = engine.analyze(X, y)

    assert "summary" in report
    assert "unary_scores" in report
    assert report["unary_scores"]
    assert report["summary"].startswith("Learnable")
