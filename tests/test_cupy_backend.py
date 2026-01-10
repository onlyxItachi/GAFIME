
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_cupy_module():
    # Mock cupy
    mock_cupy = MagicMock()
    mock_cupy.asnumpy = np.array
    mock_cupy.asarray = np.array
    mock_cupy.cuda.Device = MagicMock()
    mock_cupy.matmul = np.matmul
    mock_cupy.dot = np.dot
    mock_cupy.mean = np.mean
    mock_cupy.sqrt = np.sqrt
    mock_cupy.sum = np.sum
    mock_cupy.float64 = np.float64
    mock_cupy.empty = np.empty
    mock_cupy.nan_to_num = np.nan_to_num

    with patch.dict(sys.modules, {"cupy": mock_cupy}):
        yield mock_cupy

@pytest.fixture
def patch_cuda_paths():
    with patch("gafime.backends.cupy_backend._augment_cuda_paths"):
        yield

def test_cupy_backend_vectorized_pearson(mock_cupy_module, patch_cuda_paths):
    # Import inside test to ensure mock is active
    from gafime.backends.cupy_backend import CupyBackend
    from gafime.config import EngineConfig
    from gafime.metrics import MetricSuite

    backend = CupyBackend()
    config = EngineConfig(metric_names=["pearson", "r2"])
    suite = backend.metric_suite(config)

    X = np.array([
        [1.0, 3.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 1.0, 3.0]
    ])

    y = np.array([1.0, 2.0, 3.0])

    combos = [(0,), (1,), (2,)]

    scores = backend.score_combos(X, y, combos, suite)

    assert scores[(0,)]["pearson"] == pytest.approx(1.0)
    assert scores[(1,)]["pearson"] == pytest.approx(-1.0)
    assert scores[(2,)]["pearson"] == pytest.approx(1.0)

    assert scores[(0,)]["r2"] == pytest.approx(1.0)

def test_cupy_backend_interaction(mock_cupy_module, patch_cuda_paths):
    from gafime.backends.cupy_backend import CupyBackend
    from gafime.config import EngineConfig
    from gafime.metrics import MetricSuite

    backend = CupyBackend()
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
