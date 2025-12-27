import numpy as np

from gafime.engine import GAFIMEEngine


def test_engine_smoke_cpu():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 4))
    y = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=128)

    engine = GAFIMEEngine(max_comb_size=2, use_gpu=False, permutation_tests=5)
    report = engine.analyze(x, y)

    assert "summary" in report
    assert report["summary"]["num_features"] == 4
    assert len(report["unary_results"]) == 4
    assert report["pairwise_results"]
    assert "diagnostics" in report
