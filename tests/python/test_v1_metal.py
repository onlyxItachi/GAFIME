"""Metal backend is wired into the v1 Python boundary.

On non-Apple CI/dev boxes this should fail at the explicit payload environment
variable, not at backend parsing or a Python fallback.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import math

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if not os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") and str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402


@pytest.mark.parametrize(
    "metric_names",
    [
        ("pearson", "r2"),
        ("mutual_info",),
        ("spearman",),
        ("pearson", "mutual_info", "spearman"),
    ],
)
def test_metal_backend_is_wired_and_reaches_library_load(metric_names):
    cfg = EngineConfig(
        backend="metal",
        metric_names=metric_names,
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
        mi_bins=16,
    )
    if os.environ.get("GAFIME_METAL_V1_LIB"):
        report = GafimeEngine(cfg).analyze(
            [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]],
            [1.0, 2.0, 3.0, 4.0],
            ["a", "b"],
        )
        assert report.backend is not None
        assert report.backend.name == "v1-metal-cabi"
        assert report.backend.is_gpu
        assert report.interactions
        for item in report.interactions:
            assert set(item.metrics) == set(metric_names)
            assert all(math.isfinite(value) for value in item.metrics.values())
        return

    with pytest.raises(Exception) as excinfo:
        GafimeEngine(cfg).analyze([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], ["a"])
    message = str(excinfo.value)
    # MI/Spearman are now part of the Metal metric surface, so every metric set
    # must reach the payload/library-load boundary rather than being rejected as
    # an unsupported metric earlier in the Rust layer.
    assert "GAFIME_METAL_V1_LIB" in message
    assert "not wired" not in message.lower()
    assert "unsupported" not in message.lower()


def test_auto_selects_metal_when_it_is_the_configured_gpu_payload():
    if not os.environ.get("GAFIME_METAL_V1_LIB"):
        pytest.skip("Metal payload not configured")
    if os.environ.get("GAFIME_CUDA_V1_LIB") or os.environ.get("GAFIME_ROCM_V1_LIB"):
        pytest.skip("auto ranking should be tested with only the Metal payload configured")

    report = GafimeEngine(
        EngineConfig(
            backend="auto",
            metric_names=("pearson", "r2"),
            permutation_tests=0,
            num_repeats=1,
            budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
        )
    ).analyze(
        [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]],
        [1.0, 2.0, 3.0, 4.0],
        ["a", "b"],
    )
    assert report.backend is not None
    assert report.backend.name == "v1-metal-cabi"
