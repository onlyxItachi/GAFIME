"""P-D: resident-session reuse — compile once, swap the target, re-analyze.

The compiled artifact keeps the feature matrix resident (uploaded on GPU / held
on CPU); update_target replaces only y, so a re-analyze reuses the resident
matrix and must match a fresh compile+analyze with the same y.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig  # noqa: E402

# Import the compile *function* from the subpackage: `gafime.compile` is both a
# function and a submodule, and once the submodule is imported elsewhere it
# shadows the function on the top-level package.
from gafime.compile import compile as gafime_compile  # noqa: E402


def _metrics_by_combo(report):
    return {tuple(it.combo): it.metrics["pearson"] for it in report.interactions}


def test_update_target_reuses_resident_matrix_and_matches_fresh():
    X = [[float(i), float((i * i) % 5)] for i in range(30)]
    y1 = [float(i) for i in range(30)]           # tracks feature 0
    y2 = [float((i * i) % 5) for i in range(30)]  # tracks feature 1
    cfg = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )

    artifact = gafime_compile(X, y1, ["a", "b"], config=cfg)
    m1 = _metrics_by_combo(artifact.analyze())

    artifact.update_target(y2)
    m2 = _metrics_by_combo(artifact.analyze())

    fresh = _metrics_by_combo(gafime_compile(X, y2, ["a", "b"], config=cfg).analyze())

    # Reused-with-new-target must equal a fresh compile+analyze with that target.
    assert set(m2) == set(fresh)
    for combo in m2:
        assert m2[combo] == pytest.approx(fresh[combo], abs=1e-6)
    # And the swap actually changed the result vs the original target.
    assert any(abs(m1[c] - m2[c]) > 1e-6 for c in m1)


def test_update_target_validates_length_and_closed():
    X = [[float(i)] for i in range(8)]
    y = [float(i) for i in range(8)]
    cfg = EngineConfig(metric_names=("pearson",), budget=ComputeBudget(max_comb_size=1))
    artifact = gafime_compile(X, y, ["a"], config=cfg)
    with pytest.raises(Exception):
        artifact.update_target([1.0, 2.0])  # wrong length
    artifact.close()
    with pytest.raises(Exception):
        artifact.update_target(y)  # closed
