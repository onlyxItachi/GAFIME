"""Installed-package, hardware-conditional semantic accelerator coverage.

These are public lifecycle tests, not a payload probe or a CPU substitute.  A
Core-only host skips only when the requested payload environment is absent after
installed-package discovery.  A configured payload that is stale, incomplete,
or cannot execute is a test failure: accepting it as a skip would hide an
explicit-backend negotiation regression.
"""

from __future__ import annotations

import os
import sys
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))


pytest.importorskip("gafime.gafime_py")

from gafime import EngineConfig, semantic

_PAYLOAD_ENV = {
    "cuda": "GAFIME_CUDA_V1_LIB",
    "rocm": "GAFIME_ROCM_V1_LIB",
}
_NAMES = ("left", "right", "anchor")
_KEYS = tuple(range(10_001, 10_033))
_ROWS = tuple(
    (
        float(index),
        float((index * 7 + 3) % 19),
        float((index * index + 5 * index + 11) % 23),
    )
    for index in range(len(_KEYS))
)
_PAIRED_ROWS = tuple(
    (
        row[0] * 1.5 + 0.25,
        row[1] * -0.75 + 2.0,
        row[2] * 0.5 - 1.0,
    )
    for row in _ROWS
)
_INFERENCE_ROWS = ((41.0, 6.0, 9.0),)
_LABEL_ROWS = tuple(_KEYS[index] for index in range(0, len(_KEYS), 2))
_LABEL_VALUES = tuple(float((index * 5 + 1) % 13) for index in range(len(_LABEL_ROWS)))
_GRAPH_LEFT = _KEYS[:-1]
_GRAPH_RIGHT = _KEYS[1:]
_GRAPH_WEIGHTS = tuple(1.0 + (index % 3) * 0.25 for index in range(len(_GRAPH_LEFT)))


def _matrix(
    rows: Iterable[Iterable[float]], *, typecode: str
) -> tuple[array, memoryview]:
    materialized = tuple(tuple(row) for row in rows)
    if not materialized or not materialized[0]:
        raise AssertionError("semantic accelerator fixture must be nonempty")
    width = len(materialized[0])
    if any(len(row) != width for row in materialized):
        raise AssertionError("semantic accelerator fixture must be rectangular")
    storage = array(typecode, (value for row in materialized for value in row))
    view = (
        memoryview(storage).cast("B").cast(typecode, shape=(len(materialized), width))
    )
    return storage, view


def _session(
    backend: str,
    precision: str,
    rows: Iterable[Iterable[float]] = _ROWS,
    *,
    row_keys: Iterable[int] = _KEYS,
    feature_names: Iterable[str] = _NAMES,
    row_domain: str = "native-semantic-parity",
    role: str = "discovery",
    provenance: str | None = None,
) -> tuple[semantic.TabularSession, array]:
    typecode = "d" if precision == "fp64" else "f"
    storage, matrix = _matrix(rows, typecode=typecode)
    session = semantic.TabularSession(
        matrix,
        config=EngineConfig(backend=backend, precision=precision),
        feature_names=list(feature_names),
        row_keys=list(row_keys),
        row_domain=row_domain,
        provenance=provenance or f"native-semantic-{backend}-{precision}",
        role=role,
    )
    return session, storage


def _require_accelerator_session(
    backend: str, precision: str, **session_kwargs: Any
) -> tuple[semantic.TabularSession, array]:
    """Open an explicit accelerator session or report only real absence as skip."""

    try:
        return _session(backend, precision, **session_kwargs)
    except (NotImplementedError, RuntimeError, ValueError) as error:
        payload_env = _PAYLOAD_ENV[backend]
        if not os.environ.get(payload_env):
            pytest.skip(
                f"{backend} semantic accelerator skipped: {payload_env} is not configured "
                f"after installed-package discovery ({type(error).__name__}: {error})"
            )
        raise


def _assert_gpu_capabilities(
    session: semantic.TabularSession, backend: str, precision: str
) -> None:
    capabilities = session.capabilities
    assert capabilities["configured_backend"] == backend
    assert capabilities["selected_backend"] == backend
    assert capabilities["configured_device_id"] == 0
    assert capabilities["selected_device_id"] == 0
    assert capabilities["precision"] == precision
    assert capabilities["programs"] == [
        "source",
        "absolute_difference",
        "softsign",
        "centered_product",
    ]
    assert capabilities["statistics"] == ["pearson", "graph_energy"]
    assert capabilities["contexts"] == ["reference", "paired_view", "labels", "graph"]
    assert capabilities["source"] == "runtime"
    assert isinstance(capabilities["payload"], str) and capabilities["payload"]
    assert isinstance(capabilities["primitive_abi_version"], int)
    assert "no Core substitution" in capabilities["selection_reason"]


def _record_values(
    report: semantic.EvidenceReport,
    candidates: Iterable[semantic.Candidate],
    channels: Iterable[semantic.Evidence],
) -> tuple[dict[str, Any], ...]:
    records = []
    for candidate in candidates:
        for channel in channels:
            records.append(dict(report.value(candidate, channel)))
    return tuple(records)


def _feature_columns(
    table: semantic.FeatureTable, pa: Any
) -> dict[str, tuple[Any, ...]]:
    exported = pa.array(table)
    assert exported.type.names == ["__gafime_row_key__", *table.feature_names]
    return {
        name: tuple(exported.field(name).to_pylist()) for name in exported.type.names
    }


def _assert_same_records(
    core: tuple[dict[str, Any], ...],
    accelerator: tuple[dict[str, Any], ...],
    precision: str,
) -> None:
    assert len(accelerator) == len(core)
    absolute, relative = (2e-4, 2e-5) if precision == "fp32" else (2e-10, 2e-10)
    for expected, actual in zip(core, accelerator, strict=True):
        assert actual["state"] == expected["state"]
        assert actual["support"] == expected["support"]
        assert actual["reason"] == expected["reason"]
        if expected["state"] == "measured":
            assert actual["value"] == pytest.approx(
                expected["value"], abs=absolute, rel=relative
            )
        else:
            assert actual["value"] is None


def _assert_same_feature_columns(
    core: dict[str, tuple[Any, ...]],
    accelerator: dict[str, tuple[Any, ...]],
    precision: str,
) -> None:
    assert accelerator.keys() == core.keys()
    absolute, relative = (2e-5, 2e-5) if precision == "fp32" else (2e-11, 2e-11)
    for name, expected in core.items():
        actual = accelerator[name]
        assert len(actual) == len(expected)
        if name == "__gafime_row_key__":
            assert actual == expected
        else:
            assert actual == pytest.approx(expected, abs=absolute, rel=relative)


def _run_lifecycle(
    session: semantic.TabularSession,
    *,
    backend: str,
    precision: str,
    pa: Any,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, tuple[Any, ...]],
    dict[str, tuple[Any, ...]],
]:
    """Exercise one public discovery/reuse/inference lifecycle on one backend."""

    try:
        if backend != "core":
            _assert_gpu_capabilities(session, backend, precision)
        else:
            assert session.selected_backend == "core"
            assert session.capabilities["statistics"] == [
                "pearson",
                "spearman",
                "fixed_nmi",
                "graph_energy",
            ]

        assert session.begin_round() == 1
        left = session.source("left")
        right = session.source("right")
        anchor = session.source("anchor")
        difference = session.absolute_difference(left, right)
        softened = session.softsign(difference)
        product = session.centered_product([left, anchor], [15.5, 11.0])
        candidates = (left, difference, softened, product)

        paired_storage, paired_matrix = _matrix(
            _PAIRED_ROWS, typecode="d" if precision == "fp64" else "f"
        )
        paired = session.snapshot(
            paired_matrix,
            feature_names=list(_NAMES),
            row_keys=list(_KEYS),
            row_domain="native-semantic-parity",
            provenance=f"paired-{backend}-{precision}",
            role="discovery",
        )
        labels = session.frame.labels(
            row_keys=list(_LABEL_ROWS),
            values=list(_LABEL_VALUES),
            provenance=f"partial-labels-{backend}-{precision}",
        )
        graph = session.frame.graph(
            left_keys=list(_GRAPH_LEFT),
            right_keys=list(_GRAPH_RIGHT),
            weights=list(_GRAPH_WEIGHTS),
            provenance=f"graph-{backend}-{precision}",
        )
        channels = (
            semantic.Evidence.reference("reference", right),
            semantic.Evidence.paired("paired", paired),
            semantic.Evidence.labels("partial-labels", labels),
            semantic.Evidence.graph("graph", graph),
        )
        report = session.evaluate(candidates, channels)
        assert report.backend == backend
        assert report.precision == precision
        records = _record_values(report, candidates, channels)
        assert all(record["state"] == "measured" for record in records)
        assert {record["support"] for record in records[2 :: len(channels)]} == {
            len(_LABEL_ROWS)
        }
        assert {record["support"] for record in records[3 :: len(channels)]} == {
            len(_GRAPH_LEFT)
        }

        if backend != "core":
            for statistic, bins in (("spearman", None), ("fixed_nmi", 2)):
                unsupported = semantic.Evidence.reference(
                    f"unsupported-{statistic}", right, statistic=statistic, bins=bins
                )
                with pytest.raises(
                    NotImplementedError,
                    match="Pearson only; no Core substitution occurs",
                ):
                    session.evaluate([left], [unsupported])

        accepted = session.select(
            report,
            semantic.SelectionPolicy(channels[2], direction="maximize", limit=2),
        )
        assert len(accepted) == 2
        assert session.begin_round(accepted) == 2
        reused = session.softsign(accepted[0])
        reuse_report = session.evaluate([reused], [channels[2]])
        second_accepted = session.select(
            reuse_report,
            semantic.SelectionPolicy(channels[2], direction="maximize", limit=1),
        )
        assert len(second_accepted) == 1
        empty = session.select(
            reuse_report,
            semantic.SelectionPolicy(channels[2], direction="maximize", limit=0),
        )
        assert len(empty) == 0
        assert session.begin_round([accepted, second_accepted]) == 3
        composite = session.absolute_difference(accepted[0], second_accepted[0])
        composite_report = session.evaluate([composite], [channels[2]])
        composite_accepted = session.select(
            composite_report,
            semantic.SelectionPolicy(channels[2], direction="maximize", limit=1),
        )
        assert len(composite_accepted) == 1

        inference_storage, inference_matrix = _matrix(
            _INFERENCE_ROWS, typecode="d" if precision == "fp64" else "f"
        )
        inference = session.snapshot(
            inference_matrix,
            feature_names=list(_NAMES),
            row_keys=[90_001],
            row_domain="native-semantic-inference",
            provenance=f"inference-{backend}-{precision}",
        )
        features = session.transform(accepted, inference)
        composite_features = session.transform(composite_accepted, inference)
        empty_features = session.transform(empty, inference)
        assert features.feature_names == ["feature_0", "feature_1"]
        assert features.row_keys == [90_001]
        assert features.rows == 1
        assert features.precision == precision
        assert composite_features.feature_names == ["feature_0"]
        assert composite_features.row_keys == [90_001]
        assert composite_features.rows == 1
        assert composite_features.precision == precision
        assert empty_features.feature_names == []
        assert empty_features.row_keys == [90_001]
        assert empty_features.rows == 1
        assert _feature_columns(empty_features, pa) == {"__gafime_row_key__": (90_001,)}
        columns = _feature_columns(features, pa)
        composite_columns = _feature_columns(composite_features, pa)

        diagnostics = session.diagnostics
        if backend != "core":
            assert set(diagnostics) == {
                "backend",
                "native_work_counters_available",
                "retained_bytes",
            }
            assert diagnostics["backend"] == backend
            assert diagnostics["native_work_counters_available"] is False
            assert diagnostics["retained_bytes"] >= 0

        session.close()
        # Arrow-owned outputs remain usable after the backend/session lifetime.
        assert _feature_columns(features, pa) == columns
        assert _feature_columns(composite_features, pa) == composite_columns
        assert _feature_columns(empty_features, pa) == {"__gafime_row_key__": (90_001,)}
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.capabilities
        _ = paired_storage
        _ = inference_storage
        return records, columns, composite_columns
    finally:
        session.close()


@pytest.mark.parametrize("backend", ("cuda", "rocm"))
@pytest.mark.parametrize("precision", ("fp32", "mixed", "fp64"))
def test_explicit_accelerator_semantic_lifecycle_matches_core(
    backend: str, precision: str
) -> None:
    """A real CUDA/ROCm route matches the Core lifecycle without fallback."""

    pa = pytest.importorskip("pyarrow")
    core, core_storage = _session("core", precision)
    accelerator, accelerator_storage = _require_accelerator_session(backend, precision)
    try:
        core_records, core_features, core_composite_features = _run_lifecycle(
            core, backend="core", precision=precision, pa=pa
        )
        (
            accelerator_records,
            accelerator_features,
            accelerator_composite_features,
        ) = _run_lifecycle(accelerator, backend=backend, precision=precision, pa=pa)
        _assert_same_records(core_records, accelerator_records, precision)
        _assert_same_feature_columns(core_features, accelerator_features, precision)
        _assert_same_feature_columns(
            core_composite_features, accelerator_composite_features, precision
        )
    finally:
        core.close()
        accelerator.close()
        _ = core_storage
        _ = accelerator_storage


_DEFINEDNESS_NAMES = (
    "underflow",
    "underflow_reference",
    "constant",
    "zero_left",
    "zero_reference",
)
_DEFINEDNESS_KEYS = (20_001, 20_002, 20_003, 20_004)


def _definedness_rows(scale: float) -> tuple[tuple[float, ...], ...]:
    """Finite nonconstant axes whose variance product, not either variance, underflows."""

    underflow = (0.0, scale, 2.0 * scale, 3.0 * scale)
    underflow_reference = (0.0, 2.0 * scale, scale, 3.0 * scale)
    constant = (7.0, 7.0, 7.0, 7.0)
    zero_left = (1.0, -1.0, 1.0, -1.0)
    zero_reference = (1.0, 1.0, -1.0, -1.0)
    return tuple(
        zip(
            underflow,
            underflow_reference,
            constant,
            zero_left,
            zero_reference,
            strict=True,
        )
    )


def _definedness_records(session: semantic.TabularSession) -> dict[str, dict[str, Any]]:
    """Return exact public state records for definedness, not approximate scores."""

    assert session.begin_round() == 1
    underflow = session.source("underflow")
    underflow_reference = session.source("underflow_reference")
    constant = session.source("constant")
    zero_left = session.source("zero_left")
    zero_reference = session.source("zero_reference")

    underflow_channel = semantic.Evidence.reference("underflow", underflow_reference)
    constant_channel = semantic.Evidence.reference("constant", zero_reference)
    zero_channel = semantic.Evidence.reference("measured-zero", zero_reference)
    missing_labels_channel = semantic.Evidence.labels("missing-labels")

    underflow_report = session.evaluate([underflow], [underflow_channel])
    constant_report = session.evaluate([constant], [constant_channel])
    zero_report = session.evaluate([zero_left], [zero_channel])
    missing_labels_report = session.evaluate([underflow], [missing_labels_channel])
    return {
        "underflow": dict(underflow_report.value(underflow, underflow_channel)),
        "constant": dict(constant_report.value(constant, constant_channel)),
        "measured_zero": dict(zero_report.value(zero_left, zero_channel)),
        "missing_labels": dict(
            missing_labels_report.value(underflow, missing_labels_channel)
        ),
    }


@pytest.mark.parametrize("backend", ("cuda", "rocm"))
@pytest.mark.parametrize(
    ("precision", "scale"), (("fp32", 1.0e-15), ("fp64", 1.0e-100))
)
def test_explicit_accelerator_matches_core_pearson_definedness_states(
    backend: str, precision: str, scale: float
) -> None:
    """Underflowed normalization is degenerate, distinct from zero/constant/missing."""

    rows = _definedness_rows(scale)
    kwargs: dict[str, Any] = {
        "rows": rows,
        "row_keys": _DEFINEDNESS_KEYS,
        "feature_names": _DEFINEDNESS_NAMES,
        "row_domain": "native-semantic-definedness",
        "provenance": f"definedness-{precision}",
    }
    accelerator, accelerator_storage = _require_accelerator_session(
        backend, precision, **kwargs
    )
    core, core_storage = _session("core", precision, **kwargs)
    try:
        expected = _definedness_records(core)
        assert expected == {
            "underflow": {
                "state": "unavailable",
                "value": None,
                "support": 4,
                "reason": "degenerate_reduction",
            },
            "constant": {
                "state": "unavailable",
                "value": None,
                "support": 4,
                "reason": "constant_operand",
            },
            "measured_zero": {
                "state": "measured",
                "value": 0.0,
                "support": 4,
                "reason": None,
            },
            "missing_labels": {
                "state": "unavailable",
                "value": None,
                "support": 0,
                "reason": "missing_labels",
            },
        }
        assert _definedness_records(accelerator) == expected
    finally:
        core.close()
        accelerator.close()
        _ = core_storage
        _ = accelerator_storage


_OVERFLOW_NAMES = (
    "positive",
    "negative",
    "product_left",
    "product_right",
    "reference",
)
_OVERFLOW_KEYS = (30_001, 30_002, 30_003, 30_004)


def _overflow_rows(precision: str) -> tuple[tuple[float, ...], ...]:
    difference_scale = 2.0e38 if precision != "fp64" else 1.0e308
    product_scale = 2.0e20 if precision != "fp64" else 1.0e200
    return (
        (
            difference_scale,
            -difference_scale,
            product_scale,
            product_scale,
            1.0,
        ),
        (
            0.99 * difference_scale,
            -0.99 * difference_scale,
            0.9 * product_scale,
            1.1 * product_scale,
            2.0,
        ),
        (
            0.98 * difference_scale,
            -0.98 * difference_scale,
            1.1 * product_scale,
            0.9 * product_scale,
            3.0,
        ),
        (
            0.97 * difference_scale,
            -0.97 * difference_scale,
            1.2 * product_scale,
            1.2 * product_scale,
            4.0,
        ),
    )


def _assert_finite_program_overflow(session: semantic.TabularSession) -> None:
    """A finite pointwise overflow fails materialization before it emits evidence."""

    assert session.begin_round() == 1
    positive = session.source("positive")
    negative = session.source("negative")
    product_left = session.source("product_left")
    product_right = session.source("product_right")
    reference = session.source("reference")
    channel = semantic.Evidence.reference("reference", reference)

    difference = session.absolute_difference(positive, negative)
    with pytest.raises(ValueError):
        session.evaluate([difference], [channel])

    product = session.centered_product([product_left, product_right], [0.0, 0.0])
    with pytest.raises(ValueError):
        session.evaluate([product], [channel])


@pytest.mark.parametrize("backend", ("cuda", "rocm"))
@pytest.mark.parametrize("precision", ("fp32", "mixed", "fp64"))
def test_explicit_accelerator_matches_core_finite_program_overflow_failures(
    backend: str, precision: str
) -> None:
    """Finite absolute-difference/product overflow is a ValueError, never evidence."""

    kwargs: dict[str, Any] = {
        "rows": _overflow_rows(precision),
        "row_keys": _OVERFLOW_KEYS,
        "feature_names": _OVERFLOW_NAMES,
        "row_domain": "native-semantic-overflow",
        "provenance": f"overflow-{precision}",
    }
    accelerator, accelerator_storage = _require_accelerator_session(
        backend, precision, **kwargs
    )
    core, core_storage = _session("core", precision, **kwargs)
    try:
        _assert_finite_program_overflow(core)
        _assert_finite_program_overflow(accelerator)
    finally:
        core.close()
        accelerator.close()
        _ = core_storage
        _ = accelerator_storage


_ROW_BOUNDARY_COUNTS = (2, 33, 65, 257, 1_025, 8_192)


def _row_boundary_fixture(
    row_count: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[int, ...]]:
    """Use non-contiguous row keys across native lane and block boundaries."""

    keys = tuple(40_003 + index * 17 for index in range(row_count))
    rows = tuple(
        (
            float(index * 3 + 1),
            float((index * 17 + 5) % 31 - 15),
            float((index * index * 3 + index * 7 + 11) % 37 - 18),
        )
        for index in range(row_count)
    )
    return rows, keys


def _row_boundary_records(
    session: semantic.TabularSession, *, row_keys: tuple[int, ...]
) -> tuple[dict[str, Any], ...]:
    """Evaluate one source and one native-derived feature over Pearson and graph."""

    assert session.begin_round() == 1
    left = session.source("left")
    right = session.source("right")
    difference = session.absolute_difference(left, right)
    graph = session.frame.graph(
        left_keys=list(row_keys[:-1]),
        right_keys=list(row_keys[1:]),
        weights=[1.0 + (index % 5) * 0.125 for index in range(len(row_keys) - 1)],
        provenance="native-semantic-row-boundary-graph",
    )
    channels = (
        semantic.Evidence.reference("row-boundary-reference", right),
        semantic.Evidence.graph("row-boundary-graph", graph),
    )
    candidates = (left, difference)
    report = session.evaluate(candidates, channels)
    records = _record_values(report, candidates, channels)
    assert all(record["state"] == "measured" for record in records)
    assert {record["support"] for record in records[0 :: len(channels)]} == {
        len(row_keys)
    }
    assert {record["support"] for record in records[1 :: len(channels)]} == {
        len(row_keys) - 1
    }
    return records


@pytest.mark.parametrize("backend", ("cuda", "rocm"))
@pytest.mark.parametrize("precision", ("fp32", "mixed", "fp64"))
@pytest.mark.parametrize("row_count", _ROW_BOUNDARY_COUNTS)
def test_explicit_accelerator_matches_core_across_row_dispatch_boundaries(
    backend: str, precision: str, row_count: int
) -> None:
    """Physical payload parity spans short lanes and native block-boundary row counts."""

    rows, row_keys = _row_boundary_fixture(row_count)
    kwargs: dict[str, Any] = {
        "rows": rows,
        "row_keys": row_keys,
        "row_domain": "native-semantic-row-boundary",
        "provenance": f"row-boundary-{precision}-{row_count}",
    }
    accelerator, accelerator_storage = _require_accelerator_session(
        backend, precision, **kwargs
    )
    core, core_storage = _session("core", precision, **kwargs)
    try:
        _assert_gpu_capabilities(accelerator, backend, precision)
        assert core.selected_backend == "core"
        core_records = _row_boundary_records(core, row_keys=row_keys)
        accelerator_records = _row_boundary_records(accelerator, row_keys=row_keys)
        _assert_same_records(core_records, accelerator_records, precision)
    finally:
        core.close()
        accelerator.close()
        _ = core_storage
        _ = accelerator_storage


def _batched_centered_product_case(
    session: semantic.TabularSession,
    *,
    backend: str,
    precision: str,
    pa: Any,
) -> tuple[tuple[dict[str, Any], ...], dict[str, tuple[Any, ...]]]:
    """Exercise distinct centered-product descriptors in one native batch."""

    assert session.begin_round() == 1
    left = session.source("left")
    right = session.source("right")
    anchor = session.source("anchor")
    first_product = session.centered_product([left, anchor], [15.5, 11.0])
    difference = session.absolute_difference(left, right)
    softened = session.softsign(difference)
    second_product = session.centered_product([softened, right], [0.5, 9.25])
    candidates = (first_product, difference, softened, second_product)
    assert first_product != second_product
    assert difference != softened

    channel = semantic.Evidence.reference("descriptor-reference", anchor)
    report = session.evaluate(candidates, [channel])
    assert report.backend == backend
    assert report.precision == precision
    records = _record_values(report, candidates, [channel])
    assert all(record["state"] == "measured" for record in records)
    separately_evaluated = tuple(
        dict(session.evaluate([candidate], [channel]).value(candidate, channel))
        for candidate in candidates
    )
    _assert_same_records(records, separately_evaluated, precision)

    accepted = session.select(
        report,
        semantic.SelectionPolicy(channel, direction="maximize", limit=len(candidates)),
    )
    assert len(accepted) == len(candidates)
    inference_storage, inference_matrix = _matrix(
        _INFERENCE_ROWS, typecode="d" if precision == "fp64" else "f"
    )
    inference = session.snapshot(
        inference_matrix,
        feature_names=list(_NAMES),
        row_keys=[90_101],
        row_domain="native-semantic-descriptor-inference",
        provenance=f"descriptor-inference-{backend}-{precision}",
    )
    transformed = session.transform(accepted, inference)
    assert transformed.feature_names == [
        f"feature_{index}" for index in range(len(candidates))
    ]
    assert transformed.row_keys == [90_101]
    assert transformed.rows == 1
    assert transformed.precision == precision
    columns = _feature_columns(transformed, pa)
    _ = inference_storage
    return records, columns


@pytest.mark.parametrize("backend", ("cuda", "rocm"))
@pytest.mark.parametrize("precision", ("fp32", "mixed", "fp64"))
def test_explicit_accelerator_batched_centered_product_descriptors_match_core(
    backend: str, precision: str
) -> None:
    """No derived descriptor may overwrite an earlier queued centered product."""

    pa = pytest.importorskip("pyarrow")
    accelerator, accelerator_storage = _require_accelerator_session(backend, precision)
    core, core_storage = _session("core", precision)
    try:
        _assert_gpu_capabilities(accelerator, backend, precision)
        assert core.selected_backend == "core"
        core_records, core_columns = _batched_centered_product_case(
            core, backend="core", precision=precision, pa=pa
        )
        accelerator_records, accelerator_columns = _batched_centered_product_case(
            accelerator, backend=backend, precision=precision, pa=pa
        )
        _assert_same_records(core_records, accelerator_records, precision)
        _assert_same_feature_columns(core_columns, accelerator_columns, precision)
    finally:
        core.close()
        accelerator.close()
        _ = core_storage
        _ = accelerator_storage
