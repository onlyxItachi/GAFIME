"""Metal backend is wired into the v1 Python boundary.

On non-Apple CI/dev boxes this should fail at the explicit payload environment
variable, not at backend parsing or a Python fallback.
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


def test_metal_backend_is_wired_and_reaches_library_load():
    if os.environ.get("GAFIME_METAL_V1_LIB"):
        pytest.skip("Metal payload present; execution requires Apple Metal hardware")
    cfg = EngineConfig(
        backend="metal",
        metric_names=("pearson", "r2"),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )
    with pytest.raises(Exception) as excinfo:
        GafimeEngine(cfg).analyze([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], ["a"])
    message = str(excinfo.value)
    assert "GAFIME_METAL_V1_LIB" in message
    assert "not wired" not in message.lower()
