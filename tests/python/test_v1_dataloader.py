"""Pure-Python tests for the Polars dataloader's column resolution.

These run without Polars or the native wheel; the full read+analyze path is
exercised in the wheel/integration phase.
"""
import sys
from pathlib import Path

import pytest

# Resolve the v1 wheel source (python/), not the legacy ./gafime/ at the repo root.
_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime.dataloader import _resolve_feature_columns  # noqa: E402


def test_defaults_to_every_column_except_target():
    cols = ["a", "b", "c", "y"]
    assert _resolve_feature_columns(cols, "y", None) == ["a", "b", "c"]


def test_explicit_features_are_used_in_order():
    cols = ["a", "b", "c", "y"]
    assert _resolve_feature_columns(cols, "y", ["c", "a"]) == ["c", "a"]


def test_missing_target_raises():
    with pytest.raises(ValueError, match="target column 'z' not found"):
        _resolve_feature_columns(["a", "b"], "z", None)


def test_missing_feature_raises():
    with pytest.raises(ValueError, match="feature columns not found"):
        _resolve_feature_columns(["a", "b", "y"], "y", ["a", "nope"])


def test_target_as_feature_raises():
    with pytest.raises(ValueError, match="cannot also be a feature"):
        _resolve_feature_columns(["a", "y"], "y", ["a", "y"])


def test_no_features_resolved_raises():
    with pytest.raises(ValueError, match="no feature columns resolved"):
        _resolve_feature_columns(["y"], "y", None)


def test_target_must_name_exactly_one_column():
    with pytest.raises(ValueError, match="exactly one column"):
        _resolve_feature_columns(["a", "y"], ["y"], None)


def test_target_must_appear_exactly_once_in_schema():
    with pytest.raises(ValueError, match="exactly once"):
        _resolve_feature_columns(["a", "y", "y"], "y", None)


def test_dataload_is_exported_at_top_level():
    import gafime

    assert hasattr(gafime, "dataload")
    assert "dataload" in gafime.__all__
