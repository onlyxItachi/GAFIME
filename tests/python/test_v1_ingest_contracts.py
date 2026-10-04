"""Focused source-dtype, seed and candidate-cap contracts for dataload.

Set GAFIME_TEST_INSTALLED_PACKAGE=1 outside the checkout import path to require
the installed native boundary; missing native dependencies then fail, not skip.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import sys

import pytest

_INSTALLED = os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") == "1"
_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if not _INSTALLED and str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime  # noqa: E402
from gafime import v1_adapter  # noqa: E402

_F32_MAX = float.fromhex("0x1.fffffep+127")


@pytest.fixture
def polars():
    if _INSTALLED:
        import polars as pl
        return pl
    return pytest.importorskip("polars")


@pytest.fixture
def native(polars):
    if _INSTALLED:
        import gafime.gafime_py  # noqa: F401
    else:
        pytest.importorskip("gafime.gafime_py")
    return polars


def _config(precision="mixed", *, shortcut=True, seed=7, **budget):
    return gafime.EngineConfig(
        backend="cpu",
        precision=precision,
        metric_names=("pearson",),
        num_repeats=1,
        permutation_tests=0,
        random_seed=seed,
        budget=gafime.ComputeBudget(
            max_comb_size=budget.pop("max_comb_size", 1),
            max_combinations_per_k=budget.pop("max_combinations_per_k", 4),
            keep_in_vram=shortcut,
            **budget,
        ),
    )


def _direct(frame, config):
    names = [name for name in frame.columns if name != "target"]
    return gafime.GafimeEngine(config).analyze(
        frame.select(names).rows(), frame["target"].to_list(), names
    )


def _records(report):
    return [
        (row.candidate_id, row.combo, row.metrics)
        for row in report.interactions
    ]


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("file_kind", ["parquet", "csv", "ipc"])
def test_loader_preserves_wide_source_before_ingest(
    polars, monkeypatch, tmp_path, precision, file_kind
):
    frame = polars.DataFrame({"x": [1e39, 1e100], "target": [1e100, 1e39]})
    path = tmp_path / f"wide.{file_kind}"
    getattr(frame, f"write_{file_kind}")(path)
    sentinel = object()

    def capture(config, features, target, names):
        assert config.precision == precision
        assert names == ["x"]
        assert features.dtypes == target.dtypes == [polars.Float64]
        assert features["x"].to_list() == [1e39, 1e100]
        assert target["target"].to_list() == [1e100, 1e39]
        return sentinel

    monkeypatch.setattr(v1_adapter, "analyze_arrow_with_v1_boundary", capture)
    assert gafime.dataload(path, "target", config=_config(precision)) is sentinel


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize("seed", [7, 123, (1 << 129) + 123, None])
def test_dataload_seed_matches_direct_with_binding_caps(
    native, monkeypatch, tmp_path, precision, shortcut, seed
):
    # There are 16 unary, 120 pairwise and 560 triple candidates before caps;
    # sampling four per arity therefore exercises the planning seed.
    data = {
        f"x{col}": [
            math.sin((row + 1) * (col + 2) * 0.173)
            + math.cos((row + 3) * (col + 1) * 0.071)
            for row in range(32)
        ]
        for col in range(16)
    }
    data["target"] = [math.sin(row * 0.321) for row in range(32)]
    dtype = native.Float64 if precision == "fp64" else native.Float32
    frame = native.DataFrame(data).cast(dtype)
    path = tmp_path / "seeded.parquet"
    frame.write_parquet(path)
    config = _config(precision, shortcut=shortcut, seed=seed, max_comb_size=3)
    assert v1_adapter._raw_arrow_config_supported(config) is shortcut
    assert v1_adapter._raw_arrow_dtypes_supported(
        precision, frame.select(frame.columns[:-1]), frame.select("target")
    )
    if seed is None:
        # Control entropy, not planner internals, to compare a None request with
        # the exact arbitrary-size integer which the ordinary adapter resolves.
        resolved_seed = (1 << 200) + 987
        entropy_calls = []

        def fresh_seed():
            entropy_calls.append(True)
            return resolved_seed

        monkeypatch.setattr(v1_adapter, "_fresh_random_seed", fresh_seed)
        direct = _direct(frame, replace(config, random_seed=resolved_seed))
    else:
        direct = _direct(frame, config)
    loaded = gafime.dataload(path, "target", config=config)

    assert _records(loaded) == _records(direct)
    assert loaded.feature_names == direct.feature_names
    assert len(loaded.interactions) > 4
    assert all(
        sum(len(row.combo) == arity for row in loaded.interactions) <= 4
        for arity in (1, 2, 3)
    )
    if seed is None:
        assert len(entropy_calls) == 1


def test_dataload_preserves_ineligible_feature_caps(native, tmp_path):
    frame = native.DataFrame(
        {f"x{col}": [float((row * (col + 1)) % 17) for row in range(32)]
         for col in range(12)}
        | {"target": [float(row % 11) for row in range(32)]}
    ).cast(native.Float32)
    path = tmp_path / "feature_caps.parquet"
    frame.write_parquet(path)
    config = _config(
        max_comb_size=3, max_feature_candidate=6, top_features_for_higher_k=3,
        seed=(1 << 129) + 123,
    )

    loaded = gafime.dataload(path, "target", config=config)

    assert _records(loaded) == _records(_direct(frame, config))
    assert all(index < 6 for row in loaded.interactions for index in row.combo)


@pytest.mark.parametrize("precision", ["fp32", "mixed"])
@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize("column", ["x", "target"])
@pytest.mark.parametrize("value", [1e39, 1e100, -1e39, -1e100])
def test_dataload_rejects_finite_f32_overflow_before_narrowing(
    native, tmp_path, precision, shortcut, column, value
):
    data = {"x": [1.0, 2.0, 3.0, 4.0], "target": [4.0, 1.0, 3.0, 2.0]}
    data[column][1] = value
    frame = native.DataFrame(data)
    path = tmp_path / "overflow.parquet"
    frame.write_parquet(path)
    config = _config(precision, shortcut=shortcut)

    with pytest.raises(ValueError, match="outside fp32 range"):
        gafime.dataload(path, "target", config=config)
    with pytest.raises(ValueError, match="outside fp32 range"):
        _direct(frame, config)


@pytest.mark.parametrize("precision", ["fp32", "mixed"])
@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize("column", ["x", "target"])
@pytest.mark.parametrize("value", [_F32_MAX, -_F32_MAX, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("source_dtype", ["Float32", "Float64"])
def test_dataload_preserves_f32_limits_and_source_nonfinite_semantics(
    native, tmp_path, precision, shortcut, column, value, source_dtype
):
    data = {"x": [1.0, 2.0, 3.0, 4.0], "target": [4.0, 1.0, 3.0, 2.0]}
    data[column][1] = value
    frame = native.DataFrame(data).cast(getattr(native, source_dtype))
    path = tmp_path / "limits.parquet"
    frame.write_parquet(path)
    config = _config(precision, shortcut=shortcut)

    loaded = gafime.dataload(path, "target", config=config)
    direct = _direct(frame, config)

    assert len(loaded.interactions) == len(direct.interactions)
    assert loaded.backend.effective_precision == precision
    assert loaded.backend.storage_dtype == "float32"


@pytest.mark.parametrize(
    "precision,source_dtype,expected_dtype",
    [("fp32", "Float64", "Float32"), ("mixed", "Float64", "Float32"),
     ("fp64", "Float32", "Float64")],
)
def test_raw_arrow_still_rejects_mismatched_dtype(
    native, precision, source_dtype, expected_dtype
):
    from gafime import gafime_py as boundary

    features = native.DataFrame({"x": [1.0, 2.0]}).cast(getattr(native, source_dtype))
    target = native.DataFrame({"target": [2.0, 1.0]}).cast(getattr(native, source_dtype))

    with pytest.raises(ValueError, match=expected_dtype):
        boundary.analyze_continuous_arrow(features, target, precision=precision)


@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize("column", ["x", "target"])
@pytest.mark.parametrize("value", [1e39, 1e100, _F32_MAX])
def test_dataload_fp64_keeps_wide_finite_values(native, tmp_path, shortcut, column, value):
    data = {"x": [1.0, 2.0, 3.0, 4.0], "target": [4.0, 1.0, 3.0, 2.0]}
    data[column][1] = value
    frame = native.DataFrame(data)
    path = tmp_path / "fp64.parquet"
    frame.write_parquet(path)
    config = _config("fp64", shortcut=shortcut)

    loaded = gafime.dataload(path, "target", config=config)

    assert _records(loaded) == _records(_direct(frame, config))
    assert loaded.backend.storage_dtype == "float64"


@pytest.mark.parametrize("shortcut", [True, False])
def test_dataload_fp64_has_no_f32_intermediate(native, tmp_path, shortcut):
    values = [1.0 + row * 2.0**-30 for row in range(16)]
    frame = native.DataFrame({"x": values, "target": values})
    path = tmp_path / "adjacent.parquet"
    frame.write_parquet(path)

    report = gafime.dataload(path, "target", config=_config("fp64", shortcut=shortcut))

    assert len(report.interactions) == 1
    assert report.interactions[0].metrics["pearson"] == pytest.approx(1.0, abs=1e-12)
