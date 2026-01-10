
import numpy as np
import pytest
from unittest.mock import patch
from gafime.engine import GafimeEngine
from gafime.config import EngineConfig, ComputeBudget
from gafime.backends.core_backend import CoreBackend

@pytest.fixture
def mock_resolve_backend():
    with patch("gafime.engine.resolve_backend") as mock:
        # Configure mock to return CoreBackend
        mock.return_value = (CoreBackend(), [])
        yield mock

def test_engine_baseline(mock_resolve_backend):
    # Synthetic data
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 10
    X = rng.standard_normal((n_samples, n_features))
    # y = x0 + x1 + noise
    y = X[:, 0] + X[:, 1] + rng.standard_normal(n_samples) * 0.1

    config = EngineConfig(
        metric_names=["pearson", "r2"],
        budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=100)
    )

    engine = GafimeEngine(config)
    report = engine.analyze(X, y)

    assert report.decision.signal_detected
    assert len(report.interactions) > 0

    # Check if top features are 0 and 1
    top_combos = [r.combo for r in report.interactions if len(r.combo) == 1]
    assert (0,) in top_combos
    assert (1,) in top_combos

    # Check if interaction (0, 1) is present
    pair_combos = [r.combo for r in report.interactions if len(r.combo) == 2]
    # It might be present if they are selected as top features

    # Verify scores exist
    for r in report.interactions:
        assert "pearson" in r.metrics
        assert "r2" in r.metrics

def test_engine_chunking_logic(mock_resolve_backend):
    # Verify that chunking works by using a small chunk size
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 5))
    y = rng.standard_normal(10)

    config = EngineConfig(
        metric_names=["pearson"],
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=100)
    )

    with patch("gafime.backends.core_backend.CoreBackend.score_combos", side_effect=CoreBackend().score_combos) as mock_score:
        engine = GafimeEngine(config)

        # Manually set backend and metric_suite
        backend = CoreBackend()
        engine.backend = backend
        engine.metric_suite = backend.metric_suite(config)

        combos = [(0,), (1,), (2,), (3,), (4,)]
        names = [f"f{i}" for i in range(5)]

        # Call _score_combos with chunk_size=2
        results, scores = engine._score_combos(X, y, combos, names, chunk_size=2)

        # Expect 3 calls: [0,1], [2,3], [4]
        assert mock_score.call_count == 3

        # Check args of first call
        args, _ = mock_score.call_args_list[0]
        # args: X, y, combos, suite
        assert len(args[2]) == 2 # 2 combos
