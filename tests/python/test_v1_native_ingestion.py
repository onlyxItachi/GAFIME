"""Installed public-file acquisition contracts, independent of timing claims."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import struct
import sys
import weakref

import pytest

_INSTALLED = os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") == "1"
if not _INSTALLED:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import gafime  # noqa: E402
from gafime import v1_adapter  # noqa: E402


@pytest.fixture
def native():
    if _INSTALLED:
        import polars as pl
        from gafime import gafime_py as boundary

        assert callable(getattr(boundary, "_acquire_arrow_input", None))
    else:
        pl = pytest.importorskip("polars")
        boundary = pytest.importorskip("gafime.gafime_py")
        if not callable(getattr(boundary, "_acquire_arrow_input", None)):
            pytest.skip("requires RC3 native acquisition")
    v1_adapter._clear_analyze_cache_for_tests()
    yield pl, boundary
    v1_adapter._clear_analyze_cache_for_tests()


def _frame(pl, rows=48, cols=6):
    return pl.DataFrame(
        {
            **{
                f"x{col}": [
                    math.sin((row + 1) * (col + 2) * 0.17)
                    + math.cos((row + 3) * (col + 1) * 0.071)
                    for row in range(rows)
                ]
                for col in range(cols)
            },
            "target": [math.sin(row * 0.321) for row in range(rows)],
        }
    )


def _stable(value):
    if isinstance(value, float):
        return struct.pack("<d", value)
    if isinstance(value, dict):
        return tuple((key, _stable(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_stable(item) for item in value)
    return value


def _snapshot(report):
    return (
        tuple(report.feature_names),
        tuple(_stable(asdict(item)) for item in report.interactions),
        tuple(_stable(asdict(item)) for item in report.stability),
        tuple(_stable(asdict(item)) for item in report.permutations),
        tuple(report.warnings),
        report.decision.signal_detected,
        report.backend.selected_backend,
        report.backend.effective_precision,
    )


def _direct(frame, config):
    names = frame.columns[:-1]
    # Deliberately independent adapter acquisition: NumPy contiguous inputs,
    # not the Arrow or generic row shim under test.
    return gafime.GafimeEngine(config).analyze(
        frame.select(names).to_numpy(), frame["target"].to_numpy(), names
    )


def _forbid_materialization(monkeypatch, pl):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "numeric dataload reached Python materialization/transport"
        )

    for name in ("iter_rows", "rows", "to_dicts", "rechunk"):
        monkeypatch.setattr(pl.DataFrame, name, forbidden)
    monkeypatch.setattr(pl.Series, "to_list", forbidden)
    for name in (
        "_coerce_row_major_f32",
        "_coerce_row_major_f32_for_cache",
        "_numeric_storage_to_le_bytes",
        "_sequence",
    ):
        monkeypatch.setattr(v1_adapter, name, forbidden)


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("file_kind", ["parquet", "csv", "ipc"])
@pytest.mark.parametrize(
    "workflow", ["default", "configured", "time_series", "decision_path"]
)
def test_public_files_use_native_acquisition_with_full_config(
    native, monkeypatch, tmp_path, precision, file_kind, workflow
):
    pl, _ = native
    for variable in ("GAFIME_CUDA_V1_LIB", "GAFIME_ROCM_V1_LIB", "GAFIME_METAL_V1_LIB"):
        monkeypatch.setenv(variable, str(tmp_path / "unavailable-payload"))
    frame = _frame(pl)
    path = tmp_path / f"full-config.{file_kind}"
    getattr(frame, f"write_{file_kind}")(path)
    # CSV parity is against its parsed values, not presumed writer roundtrip.
    parsed = getattr(pl, f"read_{file_kind}")(path)
    config = gafime.EngineConfig(precision=precision)
    if workflow != "default":
        config = replace(
            config,
            backend="core",
            permutation_tests=7,
            num_repeats=3,
            significance_top_n=4,
            random_seed=(1 << 129) + 7,
            budget=gafime.ComputeBudget(
                max_comb_size=2,
                max_combinations_per_k=4,
                top_features_for_higher_k=3,
                max_time_series_candidates=6,
                top_k_features_for_time_series=2,
            ),
            enable_time_series_functions=workflow == "time_series",
            enable_decision_path_functions=workflow == "decision_path",
            time_series_lags=(1, 2),
            time_series_windows=(3,),
            decision_path_max_paths=4,
            decision_path_top_k_features=2,
        )
    direct = _direct(parsed, config)
    v1_adapter._clear_analyze_cache_for_tests()
    _forbid_materialization(monkeypatch, pl)
    loaded = gafime.dataload(path, "target", config=config)
    assert _snapshot(loaded) == _snapshot(direct)


@pytest.mark.parametrize("scalar_fallback", [False, True])
@pytest.mark.parametrize(
    "workflow", ["resident", "uncached", "time_series", "decision_path"]
)
def test_loader_releases_foreign_frames_before_native_resident_allocation(
    native, monkeypatch, tmp_path, workflow, scalar_fallback
):
    import gafime.dataloader as loader

    pl, boundary = native
    source = _frame(pl)
    if scalar_fallback:
        source = source.cast(pl.String)
    path = tmp_path / "ownership.parquet"
    source.write_parquet(path)
    config = gafime.EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=3,
        num_repeats=2,
        significance_top_n=4,
        enable_time_series_functions=workflow == "time_series",
        enable_decision_path_functions=workflow == "decision_path",
        time_series_lags=(1,),
        time_series_windows=(3,),
        decision_path_max_paths=4,
        decision_path_top_k_features=2,
        budget=gafime.ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=4,
            keep_in_vram=workflow != "uncached",
            top_k_features_for_time_series=2,
            max_time_series_candidates=6,
        ),
    )
    direct = _direct(source, config)
    v1_adapter._clear_analyze_cache_for_tests()
    owners = []
    read_frame = loader._read_frame
    select = pl.DataFrame.select

    def observed_read(*args, **kwargs):
        frame = read_frame(*args, **kwargs)
        owners.append(weakref.ref(frame))
        return frame

    def observed_select(frame, *args, **kwargs):
        selected = select(frame, *args, **kwargs)
        if any(owner() is frame for owner in owners):
            owners.append(weakref.ref(selected))
        return selected

    compile_calls = []
    entrypoint = (
        "_compile_acquired_time_series"
        if workflow == "time_series"
        else "_compile_acquired_decision_path"
        if workflow == "decision_path"
        else "_compile_acquired_continuous"
    )
    compile_input = getattr(boundary, entrypoint)

    def observed_compile(*args, **kwargs):
        # Every loader-owned frame/view is dead before resident allocation.
        # The returned input cannot retain a Python iterator or foreign frame.
        assert len(owners) == 3
        assert all(owner() is None for owner in owners)
        compile_calls.append(True)
        return compile_input(*args, **kwargs)

    execute = v1_adapter._analyze_prepared_frame_acquisition
    executions = []

    def observed_execution(*args, **kwargs):
        assert len(owners) % 3 == 0
        assert all(owner() is None for owner in owners)
        executions.append(True)
        return execute(*args, **kwargs)

    monkeypatch.setattr(loader, "_read_frame", observed_read)
    monkeypatch.setattr(pl.DataFrame, "select", observed_select)
    monkeypatch.setattr(boundary, entrypoint, observed_compile)
    monkeypatch.setattr(
        v1_adapter, "_analyze_prepared_frame_acquisition", observed_execution
    )
    loaded = gafime.dataload(path, "target", config=config)
    assert compile_calls == [True]
    assert _snapshot(loaded) == _snapshot(direct)
    if workflow == "resident":
        hit = gafime.dataload(path, "target", config=config)
        assert _snapshot(hit) == _snapshot(direct)
        changed = source.with_columns(pl.col("target").reverse())
        changed.write_parquet(path)
        updated = gafime.dataload(path, "target", config=config)
        assert _snapshot(updated) == _snapshot(_direct(changed, config))
        assert compile_calls == [True]
        assert len(executions) == 3
    else:
        assert executions == [True]


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_acquired_fingerprints_bind_the_selected_owned_values(native, precision):
    pl, boundary = native
    frame = _frame(pl)
    config = gafime.EngineConfig(backend="core", precision=precision)
    acquired = boundary._acquire_arrow_input(
        v1_adapter._config_payload(config),
        frame.select(frame.columns[:-1]),
        frame.select("target"),
    )
    expected = v1_adapter._coerce_row_major_f32_for_cache(
        frame.select(frame.columns[:-1]).to_numpy(),
        frame["target"].to_numpy(),
        frame.columns[:-1],
        precision=precision,
        include_digests=True,
    )
    assert acquired.feature_digest == expected.feature_digest
    assert acquired.target_digest == expected.target_digest
    handle = boundary._compile_acquired_continuous(
        v1_adapter._config_payload(config), acquired
    )
    try:
        with pytest.raises(ValueError, match="consum"):
            boundary._compile_acquired_continuous(
                v1_adapter._config_payload(config), acquired
            )
        assert len(handle.analyze()) > 0
    finally:
        handle.close()


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_cache_reuses_feature_owner_and_moves_only_changed_target(
    native, monkeypatch, tmp_path, precision
):
    pl, boundary = native
    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    frame = _frame(pl)
    config = gafime.EngineConfig(
        backend="core", precision=precision, permutation_tests=3
    )
    path = tmp_path / "cached.parquet"
    frame.write_parquet(path)
    original_compile = boundary._compile_acquired_continuous
    compiled = []

    def record_compile(*args, **kwargs):
        compiled.append(True)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(boundary, "_compile_acquired_continuous", record_compile)
    expected = _direct(frame, config)
    v1_adapter._clear_analyze_cache_for_tests()
    assert _snapshot(gafime.dataload(path, "target", config=config)) == _snapshot(
        expected
    )
    first_entry = next(iter(v1_adapter._current_analyze_cache().values()))
    assert _snapshot(gafime.dataload(path, "target", config=config)) == _snapshot(
        expected
    )
    updated = frame.with_columns(pl.col("target") * -1)
    updated.write_parquet(path)
    # Direct analysis shares this cache by exact content/config identity too.
    expected_update = _direct(
        updated, replace(config, budget=replace(config.budget, keep_in_vram=False))
    )
    assert _snapshot(gafime.dataload(path, "target", config=config)) == _snapshot(
        expected_update
    )
    assert next(iter(v1_adapter._current_analyze_cache().values())) is first_entry
    assert len(compiled) == 1
    changed = updated.with_columns(pl.col("x0") + 0.5)
    changed.write_parquet(path)
    gafime.dataload(path, "target", config=config)
    assert len(compiled) == 2


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("expected_rows", [None, 9])
def test_native_multibatch_input_has_independent_feature_target_chunking(
    native, precision, expected_rows
):
    pl, boundary = native
    frame = _frame(pl, rows=9, cols=3)
    features = pl.concat([frame[:2], frame[2:5], frame[5:]], rechunk=False).select(
        frame.columns[:-1]
    )
    target = pl.concat([frame[:4], frame[4:]], rechunk=False).select("target")
    config = gafime.EngineConfig(
        backend="core", precision=precision, permutation_tests=0, num_repeats=1
    )
    acquired = boundary._acquire_arrow_input(
        v1_adapter._config_payload(config),
        features,
        target,
        expected_rows=expected_rows,
    )
    expected = v1_adapter._coerce_row_major_f32_for_cache(
        frame.select(frame.columns[:-1]).to_numpy(),
        frame["target"].to_numpy(),
        frame.columns[:-1],
        precision=precision,
        include_digests=True,
    )
    assert (acquired.rows, acquired.cols) == (9, 3)
    assert acquired.feature_digest == expected.feature_digest
    assert acquired.target_digest == expected.target_digest


def test_numeric_string_compatibility_uses_bounded_native_row_acquisition(
    native, monkeypatch, tmp_path
):
    pl, boundary = native
    frame = pl.DataFrame(
        {"x": ["1.5", " 2 ", "3", "4"], "target": [4.0, 1.0, 3.0, 2.0]}
    )
    path = tmp_path / "strings.parquet"
    frame.write_parquet(path)
    config = gafime.EngineConfig(
        backend="core",
        permutation_tests=0,
        num_repeats=1,
        budget=gafime.ComputeBudget(keep_in_vram=False),
    )
    expected = _direct(
        frame.with_columns(pl.col("x").str.strip_chars().cast(pl.Float64)), config
    )
    calls = []
    original = boundary._acquire_rows_input

    def record_rows(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary, "_acquire_rows_input", record_rows)

    def forbidden(*args, **kwargs):
        raise AssertionError("compatibility route collected an entire row iterator")

    monkeypatch.setattr(v1_adapter, "_sequence", forbidden)
    assert _snapshot(gafime.dataload(path, "target", config=config)) == _snapshot(
        expected
    )
    assert calls == [True]


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("dtype", ["Int8", "UInt32", "Int64", "UInt64", "Boolean"])
def test_primitive_arrow_columns_preserve_scalar_dataload_semantics(
    native, monkeypatch, tmp_path, precision, dtype
):
    pl, _ = native
    values = [True, False, True, False] if dtype == "Boolean" else [1, 2, 3, 4]
    if dtype in {"Int64", "UInt64"}:
        # Old dataload converts scalar ints through Python float. At this
        # midpoint int->f32 and int->f64->f32 differ, so use the established
        # scalar reference rather than silently changing that boundary policy.
        values = [1 << 60, (1 << 60) + (1 << 36) + 1, (1 << 60) + (1 << 38), 0]
    frame = pl.DataFrame(
        {
            "x": pl.Series(values, dtype=getattr(pl, dtype)),
            "target": [4.0, 1.0, 3.0, 2.0],
        }
    )
    config = gafime.EngineConfig(
        backend="core", precision=precision, permutation_tests=0, num_repeats=1
    )
    reference = gafime.GafimeEngine(config).analyze(
        ([value] for value in values), [4.0, 1.0, 3.0, 2.0], ["x"]
    )
    path = tmp_path / "primitives.parquet"
    frame.write_parquet(path)
    _forbid_materialization(monkeypatch, pl)
    assert _snapshot(gafime.dataload(path, "target", config=config)) == _snapshot(
        reference
    )


@pytest.mark.parametrize("precision", ["fp32", "mixed"])
@pytest.mark.parametrize("column", ["x", "target"])
def test_arrow_finite_value_just_above_f32_max_is_rejected_before_rounding(
    native, tmp_path, precision, column
):
    pl, _ = native
    maximum = float.fromhex("0x1.fffffep+127")
    value = math.nextafter(maximum, math.inf)
    data = {"x": [1.0, 2.0, 3.0, 4.0], "target": [4.0, 1.0, 3.0, 2.0]}
    data[column][1] = value
    frame = pl.DataFrame(data)
    path = tmp_path / "pre-cast-trap.parquet"
    frame.write_parquet(path)
    config = gafime.EngineConfig(backend="core", precision=precision)
    for run in (
        lambda: _direct(frame, config),
        lambda: gafime.dataload(path, "target", config=config),
    ):
        with pytest.raises(ValueError, match="outside fp32 range"):
            run()


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_arrow_special_values_preserve_profile_typed_content(native, precision):
    pl, boundary = native
    values = [
        -0.0,
        math.inf,
        -math.inf,
        math.nan,
        1e-310,
        1e-40,
        1e-46,
        float.fromhex("0x1.fffffep+127"),
    ]
    frame = pl.DataFrame({"x": values, "target": list(reversed(values))})
    config = gafime.EngineConfig(backend="core", precision=precision)
    acquired = boundary._acquire_arrow_input(
        v1_adapter._config_payload(config), frame.select("x"), frame.select("target")
    )
    expected = v1_adapter._coerce_row_major_f32_for_cache(
        frame.select("x").to_numpy(),
        frame["target"].to_numpy(),
        ["x"],
        precision=precision,
        include_digests=True,
    )
    assert acquired.feature_digest == expected.feature_digest
    assert acquired.target_digest == expected.target_digest


@pytest.mark.parametrize("column", ["x", "target"])
def test_native_numeric_nulls_fail_closed(native, tmp_path, column):
    pl, _ = native
    data = {"x": [1.0, 2.0, 3.0, 4.0], "target": [4.0, 1.0, 3.0, 2.0]}
    data[column][1] = None
    path = tmp_path / "null.parquet"
    pl.DataFrame(data).write_parquet(path)
    with pytest.raises(ValueError, match="null"):
        gafime.dataload(path, "target")


def test_native_target_alignment_and_empty_input_fail_closed(native):
    pl, boundary = native
    config = v1_adapter._config_payload(gafime.EngineConfig(backend="core"))
    with pytest.raises(ValueError, match="length"):
        boundary._acquire_arrow_input(
            config, pl.DataFrame({"x": [1.0, 2.0]}), pl.DataFrame({"target": [1.0]})
        )
    with pytest.raises(ValueError):
        boundary._acquire_arrow_input(
            config,
            pl.DataFrame({"x": []}, schema={"x": pl.Float64}),
            pl.DataFrame({"target": []}, schema={"target": pl.Float64}),
        )


@pytest.mark.parametrize(
    "feature_rows,target_rows,expected_rows,label",
    [(3, 3, 2, "X"), (3, 3, 4, "X"), (3, 2, 3, "y"), (3, 4, 3, "y")],
)
def test_native_expected_rows_is_checked_not_trusted(
    native, feature_rows, target_rows, expected_rows, label
):
    pl, boundary = native
    config = v1_adapter._config_payload(gafime.EngineConfig(backend="core"))
    with pytest.raises(ValueError, match=f"Arrow {label} row count.*expected_rows"):
        boundary._acquire_arrow_input(
            config,
            pl.DataFrame({"x": [1.0] * feature_rows}),
            pl.DataFrame({"target": [1.0] * target_rows}),
            expected_rows=expected_rows,
        )


def test_native_expected_rows_size_overflow_fails_closed(native):
    pl, boundary = native
    config = v1_adapter._config_payload(gafime.EngineConfig(backend="core"))
    with pytest.raises(ValueError, match="rows\\*cols"):
        boundary._acquire_arrow_input(
            config,
            pl.DataFrame({"x": [1.0], "z": [2.0]}),
            pl.DataFrame({"target": [1.0]}),
            expected_rows=(1 << 64) - 1,
        )


def test_public_dataload_forwards_known_frame_height(native, monkeypatch, tmp_path):
    pl, boundary = native
    frame = _frame(pl, rows=9, cols=3)
    path = tmp_path / "known-rows.parquet"
    frame.write_parquet(path)
    original = boundary._acquire_arrow_input
    calls = []

    def record_hint(*args, **kwargs):
        calls.append(kwargs.get("expected_rows"))
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary, "_acquire_arrow_input", record_hint)
    gafime.dataload(path, "target", config=gafime.EngineConfig(backend="core"))
    assert calls == [9]


def test_arrow_export_receives_explicit_no_schema_request(native, monkeypatch):
    pl, boundary = native
    original = pl.DataFrame.__arrow_c_stream__
    calls = []

    # The contracted Polars 1.3 floor requires this argument, whereas recent
    # releases default it. Explicit None works with both protocol shapes.
    def required_schema_request(self, requested_schema):
        assert requested_schema is None
        calls.append(True)
        return original(self, requested_schema)

    monkeypatch.setattr(pl.DataFrame, "__arrow_c_stream__", required_schema_request)
    config = v1_adapter._config_payload(gafime.EngineConfig(backend="core"))
    acquired = boundary._acquire_arrow_input(
        config, pl.DataFrame({"x": [1.0, 2.0]}), pl.DataFrame({"target": [2.0, 1.0]})
    )
    assert (acquired.rows, acquired.cols) == (2, 1)
    assert calls == [True, True]
