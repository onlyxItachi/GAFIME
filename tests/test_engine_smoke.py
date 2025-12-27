"""Basic smoke tests for engine."""
import numpy as np

from gafime.engine import EngineConfig, GAFIMEEngine


def test_engine_runs_unary_and_pairwise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))
    y = x[:, 0] * 0.5 + rng.normal(size=50) * 0.1

    config = EngineConfig(max_comb_size=2, permutation_runs=5, use_gpu=False)
    engine = GAFIMEEngine(config)
    report = engine.run(x, y)

    assert report["status"] == "ok"
    assert report["unary_metrics"]
    assert "interaction_metrics" in report
    assert "validation" in report
