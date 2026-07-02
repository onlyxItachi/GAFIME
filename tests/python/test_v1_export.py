"""P-D: zero-copy Arrow result export via CompileFlags(export=True).

The compiled artifact surfaces the compact result table over the Arrow C Data
Interface (the same capsule path Polars/pyarrow/torch consume), gated on the
export flag. Requires the built wheel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, compile as gafime_compile  # noqa: E402
from gafime.compile import CompileFlags  # noqa: E402
from gafime.errors import V1UnsupportedError  # noqa: E402


def _dataset(n=24):
    X = [[float(i), float((i * i) % 5)] for i in range(n)]
    y = [float(i) for i in range(n)]
    return X, y


def _config():
    return EngineConfig(
        metric_names=("pearson", "r2"),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=16),
    )


def test_export_flag_yields_arrow_capsule_pair():
    X, y = _dataset()
    artifact = gafime_compile(X, y, ["a", "b"], config=_config(), flags=CompileFlags(export=True))
    artifact.analyze()
    schema_capsule, array_capsule = artifact.export_arrow()
    assert type(schema_capsule).__name__ == "PyCapsule"
    assert type(array_capsule).__name__ == "PyCapsule"
    # __arrow_c_array__ is the dunder consumers call directly.
    again = artifact.__arrow_c_array__()
    assert len(again) == 2


def test_export_without_flag_raises():
    X, y = _dataset()
    artifact = gafime_compile(X, y, ["a", "b"], config=_config())  # no export flag
    with pytest.raises(V1UnsupportedError):
        artifact.export_arrow()


def test_exported_table_roundtrips_through_pyarrow_when_available():
    pa = pytest.importorskip("pyarrow")
    X, y = _dataset()
    artifact = gafime_compile(X, y, ["a", "b"], config=_config(), flags=CompileFlags(export=True))
    report = artifact.analyze()
    table = pa.array(artifact)  # consumes __arrow_c_array__ zero-copy
    # One row per scored candidate; struct columns include combo + metrics.
    assert len(table) == len(report.interactions)
    field_names = {f.name for f in table.type}
    assert {"candidate_id", "rank", "combo", "metrics"}.issubset(field_names)
