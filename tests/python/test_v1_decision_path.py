"""P-C: native decision_path family, end to end (requires the built wheel).

Enables `enable_decision_path_functions` and checks that the engine discovers
depth-k GBDT conjunction paths, appends their membership columns, and mines them
through the continuous path — recovering an AND-structured signal that a single
raw feature cannot express.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402


def _and_dataset(per_quadrant=20):
    """y is high only where f0 AND f1 are both high — a pure interaction that no
    single raw feature separates."""
    X, y = [], []
    for q0 in (0, 1):
        for q1 in (0, 1):
            for k in range(per_quadrant):
                f0 = (0.2 if q0 == 0 else 0.8) + 0.001 * k
                f1 = (0.2 if q1 == 0 else 0.8) + 0.001 * k
                X.append([f0, f1])
                y.append(5.0 if (q0 == 1 and q1 == 1) else 0.0)
    return X, y


def _config(**overrides) -> EngineConfig:
    base = EngineConfig(
        enable_decision_path_functions=True,
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=64),
        decision_path_max_depth=2,
        decision_path_rounds=1,
        decision_path_max_paths=8,
        decision_path_min_leaf=5,
        decision_path_learning_rate=1.0,
    )
    return replace(base, **overrides) if overrides else base


def test_decision_path_discovers_and_conjunction_feature():
    X, y = _and_dataset()
    report = GafimeEngine(_config()).analyze(X, y, feature_names=["f0", "f1"])

    # Path membership columns are appended after the two base features.
    assert len(report.feature_names) > 2
    path_names = [n for n in report.feature_names if n.startswith("path[")]
    assert path_names, f"expected a path[...] feature, got {report.feature_names}"
    assert report.warnings and "decision_path" in report.warnings[0]

    # The AND region's membership indicator matches y exactly -> near-perfect
    # pearson, which no single raw feature (f0 or f1 alone) can reach.
    best_pearson = max(abs(item.metrics["pearson"]) for item in report.interactions)
    assert best_pearson >= 0.9, f"path feature should strongly separate y, got {best_pearson}"


def test_decision_path_carries_significance_when_requested():
    X, y = _and_dataset()
    report = GafimeEngine(_config(permutation_tests=50, num_repeats=5)).analyze(
        X, y, feature_names=["f0", "f1"]
    )
    assert len(report.permutations) > 0
    assert len(report.stability) > 0
    # A perfect AND-membership feature is a real, stable signal.
    assert report.decision.signal_detected is True
