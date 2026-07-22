"""P-D: sklearn-style GafimeSelector transformer (requires the built wheel).

The transformer fits GAFIME to discover the top feature interactions and appends
their operator-combined columns to X — the legacy fit/transform surface, now on
the v1 native engine.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime.sklearn import GafimeSelector  # noqa: E402


def _interaction_dataset(n=120, seed=0):
    """y = x0 * x1 — a pairwise interaction no single feature explains; x2 is noise."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x0 = rng.uniform(-1.0, 1.0)
        x1 = rng.uniform(-1.0, 1.0)
        x2 = rng.uniform(-1.0, 1.0)
        X.append([x0, x1, x2])
        y.append(x0 * x1)
    return X, y


def test_selector_discovers_top_pair_and_appends_interaction_column():
    X, y = _interaction_dataset()
    selector = GafimeSelector(k=1, metric="pearson", operator="multiply", n_jobs=1)
    out = selector.fit_transform(X, y)

    assert selector.n_features_in_ == 3
    assert len(selector.top_interactions_) == 1
    # The discovered pair is (x0, x1) regardless of internal ordering.
    assert set(selector.top_interactions_[0]) == {0, 1}

    # Each row gains exactly one appended interaction column (x0 * x1).
    assert len(out) == len(X)
    assert all(len(row) == 4 for row in out)
    for original, transformed in zip(X, out):
        assert transformed[3] == pytest.approx(original[0] * original[1], abs=1e-6)


def test_transform_requires_fit_and_matches_feature_count():
    X, y = _interaction_dataset(n=40)
    selector = GafimeSelector(k=1, n_jobs=1)
    with pytest.raises(RuntimeError):
        selector.transform(X)

    selector.fit(X, y)
    with pytest.raises(ValueError):
        selector.transform([[1.0, 2.0]])  # wrong feature count


def test_operator_add_appends_sum():
    X, y = _interaction_dataset(n=60)
    selector = GafimeSelector(k=1, operator="add", n_jobs=1)
    out = selector.fit_transform(X, y)
    i, j = selector.top_interactions_[0]
    for original, transformed in zip(X, out):
        assert transformed[-1] == pytest.approx(original[i] + original[j], abs=1e-6)


def test_selector_supports_sklearn_clone_and_pipeline_fit():
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    X, y = _interaction_dataset(n=80)
    selector = GafimeSelector(
        k=1,
        backend="core",
        metric="pearson",
        operator="multiply",
        n_jobs=1,
    )
    cloned = clone(selector)

    assert cloned.get_params() == selector.get_params()
    pipeline = Pipeline([("gafime", cloned), ("model", Ridge())])
    pipeline.fit(X, y)
    assert len(pipeline.named_steps["gafime"].top_interactions_) == 1
