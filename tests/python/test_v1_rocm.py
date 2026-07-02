"""P-E: ROCm backend is wired into the v1 Python boundary.

`backend="rocm"` must reach the native ROCm payload loader (not be rejected as an
unsupported backend). When the payload is present it executes on the AMD GPU
(covered by the hardware e2e / gpu-sys parity test); when absent it must fail on
the MISSING LIBRARY, which is the CI-observable proof that ROCm is first-class.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402


def test_rocm_backend_is_wired_and_reaches_library_load():
    if os.environ.get("GAFIME_ROCM_V1_LIB"):
        pytest.skip("ROCm payload present; execution covered by the hardware e2e")
    cfg = EngineConfig(
        backend="rocm",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )
    with pytest.raises(Exception) as excinfo:
        GafimeEngine(cfg).analyze([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], ["a"])
    message = str(excinfo.value)
    # Reached the payload loader -> ROCm is wired (not rejected as unsupported).
    assert "GAFIME_ROCM_V1_LIB" in message
    assert "not wired" not in message.lower()


def test_hip_alias_maps_to_rocm():
    if os.environ.get("GAFIME_ROCM_V1_LIB"):
        pytest.skip("ROCm payload present; execution covered by the hardware e2e")
    cfg = EngineConfig(
        backend="hip",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )
    with pytest.raises(Exception) as excinfo:
        GafimeEngine(cfg).analyze([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], ["a"])
    assert "GAFIME_ROCM_V1_LIB" in str(excinfo.value)
