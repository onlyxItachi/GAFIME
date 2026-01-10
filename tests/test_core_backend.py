
import numpy as np
import pytest
from gafime.backends.core_backend import CoreBackend
from gafime.config import EngineConfig
from gafime.metrics import MetricSuite

def test_core_backend_pearson():
    backend = CoreBackend()
    config = EngineConfig(metric_names=["pearson"])
    suite = backend.metric_suite(config)

    X = np.array([
        [1.0, 3.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 1.0, 3.0]
    ])

    y = np.array([1.0, 2.0, 3.0])
    combos = [(0,), (1,), (2,)]

    # Debugging
    combos_list = list(combos)
    indices, offsets = backend.core.pack_combos(combos_list)
    print(f"Indices: {indices}")
    print(f"Offsets: {offsets}")

    scores = backend.score_combos(X, y, combos, suite)

    assert len(scores) == 3
    assert scores[(0,)]["pearson"] == pytest.approx(1.0)
    assert scores[(1,)]["pearson"] == pytest.approx(-1.0)
    assert scores[(2,)]["pearson"] == pytest.approx(1.0)

def test_core_backend_interaction():
    backend = CoreBackend()
    config = EngineConfig(metric_names=["pearson"])
    suite = backend.metric_suite(config)

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ])
    y = np.array([1.0, 1.0, 2.0, 0.0])

    combos = [(0, 1)]

    scores = backend.score_combos(X, y, combos, suite)
    assert (0, 1) in scores
    assert "pearson" in scores[(0, 1)]
