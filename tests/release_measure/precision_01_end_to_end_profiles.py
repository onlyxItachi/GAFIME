#!/usr/bin/env python3
"""Physical end-to-end gate for the three public precision profiles.

Run this script from a fresh installed wheel. Backends are explicit so that a
missing requested payload or device is a failure, never an optional skip::

    python precision_01_end_to_end_profiles.py --backend core
    python precision_01_end_to_end_profiles.py \
        --backend core --backend cuda --backend rocm

The bounded workloads cover the four public arithmetic domains through all
four v1 metrics, all v1 families, eager/resident/compiled execution, graph
replay where advertised, target replacement, significance, deterministic
candidate identity, visible-score ranking, profile-keyed caching, NumPy
ingest, and the Polars/Arrow dataload path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any

import gafime
import polars as pl


ALL_METRICS = ("pearson", "spearman", "mutual_info", "r2")
BACKEND_ORDER = ("core", "cuda", "rocm", "metal")
PROFILE_MATRIX = {
    "core": ("fp32", "mixed", "fp64"),
    "cuda": ("fp32", "mixed", "fp64"),
    "rocm": ("fp32", "mixed", "fp64"),
    "metal": ("fp32",),
}
EXPECTED_DOMAINS = {
    "fp32": {
        "storage_dtype": "float32",
        "interaction_arithmetic": "float32",
        "reduction_dtype": "float32",
        "result_dtype": "float32",
    },
    "mixed": {
        "storage_dtype": "float32",
        "interaction_arithmetic": "float32",
        "reduction_dtype": "float64",
        "result_dtype": "float64",
    },
    "fp64": {
        "storage_dtype": "float64",
        "interaction_arithmetic": "float64",
        "reduction_dtype": "float64",
        "result_dtype": "float64",
    },
}


@dataclass
class GateStats:
    backend_profile_cases: int = 0
    family_cases: int = 0
    eager_analyses: int = 0
    compiled_analyses: int = 0
    graph_profile_family_cases: int = 0
    graph_replays: int = 0
    target_replacements: int = 0
    permutation_reports: int = 0
    stability_reports: int = 0
    cache_profile_entries: int = 0
    numpy_ingest_cases: int = 0
    arrow_dataload_cases: int = 0
    cross_backend_comparisons: int = 0
    mi_boundary_cases: int = 0
    filtered_spearman_cases: int = 0
    covariance_nonfinite_cases: int = 0
    overflow_ratio_cases: int = 0
    timings: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class FamilyCase:
    name: str
    features: Any
    target: Any
    replacement_target: Any
    feature_names: tuple[str, ...]
    config_fields: Mapping[str, object]
    required_family: str
    required_arities: frozenset[int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        choices=(*BACKEND_ORDER, "all"),
        dest="backends",
        help=(
            "backend to validate; repeat for a matrix. The default is core. "
            "'all' explicitly requires every distributed backend to be available."
        ),
    )
    return parser.parse_args()


def _requested_backends(arguments: argparse.Namespace) -> tuple[str, ...]:
    requested = arguments.backends or ["core"]
    expanded: set[str] = set()
    for backend in requested:
        if backend == "all":
            expanded.update(BACKEND_ORDER)
        else:
            expanded.add(backend)
    return tuple(backend for backend in BACKEND_ORDER if backend in expanded)


def _expected_profiles(backend: str) -> tuple[str, ...]:
    return PROFILE_MATRIX[backend]


def _assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _assert_finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"{label}: expected a finite value, got {number!r}")
    return number


def _tolerances(precision: str, *, cross_backend: bool) -> tuple[float, float]:
    if precision == "fp32":
        return (2.0e-4, 2.0e-5) if cross_backend else (2.0e-5, 2.0e-6)
    return (2.0e-9, 2.0e-10) if cross_backend else (2.0e-10, 2.0e-11)


def _assert_close(
    actual: object,
    expected: object,
    precision: str,
    label: str,
    *,
    cross_backend: bool = False,
) -> None:
    actual_value = _assert_finite(actual, f"{label}/actual")
    expected_value = _assert_finite(expected, f"{label}/expected")
    absolute, relative = _tolerances(precision, cross_backend=cross_backend)
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=relative,
        abs_tol=absolute,
    ):
        raise AssertionError(
            f"{label}: {actual_value!r} != {expected_value!r} "
            f"(abs_tol={absolute}, rel_tol={relative})"
        )


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _item_identity(item: object) -> tuple[object, ...]:
    return (
        str(getattr(item, "candidate_id")),
        str(getattr(item, "family")),
        tuple(int(index) for index in getattr(item, "combo")),
        str(getattr(item, "expression")),
        _freeze(getattr(item, "params")),
    )


def _semantic_identity(item: object) -> tuple[object, ...]:
    params = dict(getattr(item, "params"))
    # These two fields mirror the backend-local candidate-number assignment;
    # the path definition itself is the cross-backend semantic identity.
    params.pop("candidate_id", None)
    params.pop("native_candidate_id", None)
    return (
        str(getattr(item, "family")),
        tuple(sorted(int(index) for index in getattr(item, "combo"))),
        tuple(sorted(str(name) for name in getattr(item, "feature_names"))),
        _freeze(params),
    )


def _candidate_number(candidate_id: str) -> int:
    try:
        return int(candidate_id.rsplit(":", 1)[-1])
    except ValueError as exc:
        raise AssertionError(
            f"candidate id does not end in a numeric stable identity: {candidate_id!r}"
        ) from exc


def _validate_backend_info(report: object, backend: str, precision: str) -> None:
    info = getattr(report, "backend", None)
    if info is None:
        raise AssertionError(f"{backend}/{precision}: report has no backend identity")
    _assert_equal(info.selected_backend, backend, f"{backend}/{precision}/selected")
    _assert_equal(
        info.requested_precision, precision, f"{backend}/{precision}/requested"
    )
    _assert_equal(
        info.effective_precision, precision, f"{backend}/{precision}/effective"
    )
    for name, expected in EXPECTED_DOMAINS[precision].items():
        _assert_equal(getattr(info, name), expected, f"{backend}/{precision}/{name}")
    expected_accumulator = EXPECTED_DOMAINS[precision]["reduction_dtype"]
    for metric in ALL_METRICS:
        _assert_equal(
            info.metric_accumulators.get(metric),
            expected_accumulator,
            f"{backend}/{precision}/{metric}/accumulator",
        )


def _validate_ranking(report: object, precision: str, label: str) -> None:
    interactions = getattr(report, "interactions")
    for metric in ALL_METRICS:
        ranked = list(interactions.ranked(metric_name=metric, descending=True))
        if len(ranked) != len(interactions):
            raise AssertionError(f"{label}/{metric}: ranking lost interaction rows")
        previous_value: float | None = None
        previous_id: int | None = None
        for item in ranked:
            value = _assert_finite(item.metrics[metric], f"{label}/{metric}/rank")
            candidate_number = _candidate_number(str(item.candidate_id))
            if previous_value is not None and value > previous_value:
                raise AssertionError(
                    f"{label}/{metric}: visible result ranking is not descending"
                )
            if (
                previous_value is not None
                and value == previous_value
                and previous_id is not None
                and candidate_number < previous_id
            ):
                raise AssertionError(
                    f"{label}/{metric}: visible-score tie did not use candidate id"
                )
            previous_value = value
            previous_id = candidate_number


def _validate_public_result_width(report: object, precision: str, label: str) -> None:
    native = report.interactions.native_handle
    expected_typecode = "f" if precision == "fp32" else "d"
    expected_arrow_dtype = pl.Float32 if precision == "fp32" else pl.Float64

    metric_values = native.metric_values(0)
    _assert_equal(metric_values.typecode, expected_typecode, f"{label}/metric-typecode")
    _assert_equal(
        native.record(0).metrics.typecode,
        expected_typecode,
        f"{label}/record-metric-typecode",
    )
    result_frame = pl.DataFrame(native)
    _assert_equal(
        result_frame.schema["metrics"].inner,
        expected_arrow_dtype,
        f"{label}/arrow-result-dtype",
    )

    for getter_name in (
        "significance_pvalues",
        "significance_means",
        "significance_stds",
    ):
        rows = getattr(native, getter_name)()
        if not rows:
            raise AssertionError(f"{label}/{getter_name}: no significance rows")
        if any(row.typecode != expected_typecode for row in rows):
            raise AssertionError(
                f"{label}/{getter_name}: expected array({expected_typecode!r}) rows"
            )

    for item in report.interactions:
        if item.family != "decision_path" or len(item.combo) != 1:
            continue
        params = native.decision_path_params(item.combo[0])
        if params is None:
            continue
        expected_threshold_typecode = "d" if precision == "fp64" else "f"
        _assert_equal(
            params["thresholds"].typecode,
            expected_threshold_typecode,
            f"{label}/decision-threshold-typecode",
        )
        if not isinstance(params["support"], int):
            raise AssertionError(f"{label}: decision support is not an integer count")
        break


def _validate_report(
    report: object,
    backend: str,
    precision: str,
    case: FamilyCase,
    label: str,
) -> None:
    _validate_backend_info(report, backend, precision)
    interactions = list(report.interactions)
    if not interactions:
        raise AssertionError(f"{label}: no interaction rows")
    identities = [str(item.candidate_id) for item in interactions]
    if len(set(identities)) != len(identities):
        raise AssertionError(f"{label}: candidate ids are not unique")
    families = {str(item.family) for item in interactions}
    if case.required_family not in families:
        raise AssertionError(
            f"{label}: required family {case.required_family!r} is absent from {families}"
        )
    observed_arities = {len(item.combo) for item in interactions}
    if not case.required_arities.issubset(observed_arities):
        raise AssertionError(
            f"{label}: required arities {sorted(case.required_arities)} are not "
            f"contained in {sorted(observed_arities)}"
        )
    for item in interactions:
        _assert_equal(
            set(item.metrics), set(ALL_METRICS), f"{label}/{item.candidate_id}/metrics"
        )
        for metric, value in item.metrics.items():
            _assert_finite(value, f"{label}/{item.candidate_id}/{metric}")

    interaction_ids = set(identities)
    permutations = list(report.permutations)
    stability = list(report.stability)
    if not permutations:
        raise AssertionError(f"{label}: nonzero permutation request produced no rows")
    if not stability:
        raise AssertionError(f"{label}: bootstrap request produced no rows")
    for item in permutations:
        if item.candidate_id not in interaction_ids:
            raise AssertionError(
                f"{label}: permutation identity {item.candidate_id!r} has no result row"
            )
        _assert_equal(
            set(item.p_values),
            set(ALL_METRICS),
            f"{label}/{item.candidate_id}/p-values",
        )
        for metric, value in item.p_values.items():
            probability = _assert_finite(
                value, f"{label}/{item.candidate_id}/{metric}/p-value"
            )
            if not 0.0 < probability <= 1.0:
                raise AssertionError(
                    f"{label}/{item.candidate_id}/{metric}: invalid p={probability}"
                )
    for item in stability:
        if item.candidate_id not in interaction_ids:
            raise AssertionError(
                f"{label}: stability identity {item.candidate_id!r} has no result row"
            )
        _assert_equal(
            set(item.metrics_mean),
            set(ALL_METRICS),
            f"{label}/{item.candidate_id}/bootstrap-means",
        )
        _assert_equal(
            set(item.metrics_std),
            set(ALL_METRICS),
            f"{label}/{item.candidate_id}/bootstrap-stds",
        )
        for metric in ALL_METRICS:
            _assert_finite(
                item.metrics_mean[metric],
                f"{label}/{item.candidate_id}/{metric}/bootstrap-mean",
            )
            standard_deviation = _assert_finite(
                item.metrics_std[metric],
                f"{label}/{item.candidate_id}/{metric}/bootstrap-std",
            )
            if standard_deviation < 0.0:
                raise AssertionError(
                    f"{label}/{item.candidate_id}/{metric}: negative bootstrap std"
                )
    if case.required_family != "interaction":
        if not any(item.family == case.required_family for item in permutations):
            raise AssertionError(
                f"{label}: {case.required_family} permutation significance is absent"
            )
        if not any(item.family == case.required_family for item in stability):
            raise AssertionError(
                f"{label}: {case.required_family} stability significance is absent"
            )
    _validate_public_result_width(report, precision, label)
    _validate_ranking(report, precision, label)


def _assert_report_equivalent(
    actual: object,
    expected: object,
    precision: str,
    label: str,
    *,
    cross_backend: bool = False,
) -> None:
    _assert_equal(
        tuple(actual.feature_names), tuple(expected.feature_names), f"{label}/features"
    )
    actual_interactions = list(actual.interactions)
    expected_interactions = list(expected.interactions)
    _assert_equal(
        tuple(_item_identity(item) for item in actual_interactions),
        tuple(_item_identity(item) for item in expected_interactions),
        f"{label}/interaction-identities",
    )
    for actual_item, expected_item in zip(actual_interactions, expected_interactions):
        for metric in ALL_METRICS:
            _assert_close(
                actual_item.metrics[metric],
                expected_item.metrics[metric],
                precision,
                f"{label}/{actual_item.candidate_id}/{metric}",
                cross_backend=cross_backend,
            )

    for kind, fields in (
        ("permutations", ("p_values",)),
        ("stability", ("metrics_mean", "metrics_std")),
    ):
        actual_items = list(getattr(actual, kind))
        expected_items = list(getattr(expected, kind))
        _assert_equal(
            tuple(_item_identity(item) for item in actual_items),
            tuple(_item_identity(item) for item in expected_items),
            f"{label}/{kind}/identities",
        )
        for actual_item, expected_item in zip(actual_items, expected_items):
            for field_name in fields:
                actual_values = getattr(actual_item, field_name)
                expected_values = getattr(expected_item, field_name)
                _assert_equal(
                    set(actual_values), set(expected_values), f"{label}/{kind}/keys"
                )
                for metric in actual_values:
                    if cross_backend and field_name == "p_values":
                        # Seeded maxT is deterministic within one backend, which
                        # the eager/compiled/graph checks above enforce. Its
                        # stochastic p-values are not a cross-backend equality
                        # contract; identities and [0, 1] validity are checked.
                        continue
                    _assert_close(
                        actual_values[metric],
                        expected_values[metric],
                        precision,
                        f"{label}/{kind}/{actual_item.candidate_id}/{metric}",
                        cross_backend=cross_backend,
                    )

    for metric in ALL_METRICS:
        actual_ranked = tuple(
            item.candidate_id
            for item in actual.interactions.ranked(metric_name=metric, descending=True)
        )
        expected_ranked = tuple(
            item.candidate_id
            for item in expected.interactions.ranked(
                metric_name=metric, descending=True
            )
        )
        _assert_equal(actual_ranked, expected_ranked, f"{label}/{metric}/ranking")


def _assert_cross_backend_equivalent(
    actual: object, expected: object, precision: str, label: str
) -> None:
    """Compare deterministic arithmetic, not backend-local stochastic state."""
    _assert_equal(
        tuple(actual.feature_names), tuple(expected.feature_names), f"{label}/features"
    )
    actual_rows = {_semantic_identity(item): item for item in actual.interactions}
    expected_rows = {_semantic_identity(item): item for item in expected.interactions}
    _assert_equal(
        len(actual_rows),
        len(actual.interactions),
        f"{label}/actual-semantic-uniqueness",
    )
    _assert_equal(
        len(expected_rows),
        len(expected.interactions),
        f"{label}/expected-semantic-uniqueness",
    )
    _assert_equal(set(actual_rows), set(expected_rows), f"{label}/semantic-candidates")
    for identity, actual_item in actual_rows.items():
        expected_item = expected_rows[identity]
        for metric in ALL_METRICS:
            _assert_close(
                actual_item.metrics[metric],
                expected_item.metrics[metric],
                precision,
                f"{label}/{identity[:3]}/{metric}",
                cross_backend=True,
            )


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", float(value)))[0]


def _corrected_normalized_mi_f64(
    joint: Sequence[Sequence[int]],
) -> float:
    row_count = len(joint)
    col_count = len(joint[0]) if joint else 0
    rows = [sum(row) for row in joint]
    cols = [
        sum(joint[row][col] for row in range(row_count)) for col in range(col_count)
    ]
    total = sum(rows)
    nonzero_rows = sum(count != 0 for count in rows)
    nonzero_cols = sum(count != 0 for count in cols)
    if total == 0 or nonzero_rows < 2 or nonzero_cols < 2:
        return 0.0
    total_f = float(total)
    mutual_information = 0.0
    for row in range(row_count):
        for col in range(col_count):
            count = joint[row][col]
            if count == 0:
                continue
            pxy = count / total_f
            px = rows[row] / total_f
            py = cols[col] / total_f
            mutual_information += pxy * math.log(pxy / (px * py))
    correction = ((nonzero_rows - 1) * (nonzero_cols - 1)) / (2.0 * total_f)
    corrected = max(mutual_information - correction, 0.0)
    normalizer = math.log(min(nonzero_rows, nonzero_cols))
    return corrected / normalizer if normalizer > 0.0 else 0.0


def _run_mixed_mi_boundary_adversary(backends: Sequence[str], stats: GateStats) -> None:
    """Prove that mixed MI places stored f32 values in f32 histogram bins."""
    import numpy as np

    minimum = _f32_from_bits(0x44E83674)
    boundary = _f32_from_bits(0x45000ABE)
    maximum = _f32_from_bits(0x4553975A)
    _assert_equal(_f32_bits(minimum), 0x44E83674, "MI/minimum-bits")
    _assert_equal(_f32_bits(boundary), 0x45000ABE, "MI/boundary-bits")
    _assert_equal(_f32_bits(maximum), 0x4553975A, "MI/maximum-bits")
    inverse = _f32(_f32(8.0) / _f32(maximum - minimum))
    scaled_f32 = _f32(_f32(boundary - minimum) * inverse)
    promoted_scaled = (boundary - minimum) * (8.0 / (maximum - minimum))
    _assert_equal(_f32_bits(scaled_f32), _f32_bits(1.0), "MI/f32-boundary")
    if not promoted_scaled < 1.0:
        raise AssertionError(
            f"MI adversary lost its promoted-f64 boundary: {promoted_scaled!r}"
        )

    values = [minimum] * 128 + [boundary] * 128 + [maximum] * 256
    target = [0.0] * 128 + [1.0] * 128 + [0.0] * 256
    features = np.asarray(values, dtype=np.float32).reshape((-1, 1))
    targets = np.asarray(target, dtype=np.float32)
    joint = [[0 for _ in range(8)] for _ in range(8)]
    joint[0][0] = 128
    joint[1][7] = 128
    joint[7][0] = 256
    oracle = _corrected_normalized_mi_f64(joint)

    tested = tuple(
        backend for backend in ("core", "cuda", "rocm") if backend in backends
    )
    execution_backends = ("core", *tuple(b for b in tested if b != "core"))
    scores: dict[str, float] = {}
    for backend in execution_backends:
        config = gafime.EngineConfig(
            backend=backend,
            precision="mixed",
            metric_names=("mutual_info",),
            permutation_tests=0,
            num_repeats=1,
            random_seed=29,
            mi_bins=8,
            mi_approximate=True,
            budget=gafime.ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=1,
                keep_in_vram=False,
            ),
        )
        report = gafime.GafimeEngine(config).analyze(
            features, targets, feature_names=("boundary",)
        )
        stats.eager_analyses += 1
        _validate_backend_info(report, backend, "mixed")
        rows = list(report.interactions)
        if len(rows) != 1:
            raise AssertionError(
                f"{backend}/mixed/MI-boundary: expected one row, got {len(rows)}"
            )
        score = _assert_finite(
            rows[0].metrics["mutual_info"], f"{backend}/mixed/MI-boundary"
        )
        scores[backend] = score
        stats.mi_boundary_cases += 1
    _assert_close(scores["core"], oracle, "mixed", "Core/mixed/MI-boundary-oracle")
    for backend in execution_backends:
        if backend == "core":
            continue
        if scores[backend] != scores["core"]:
            raise AssertionError(
                f"{backend}/mixed fixed-bin MI boundary differs from Core exactly: "
                f"{scores[backend].hex()} != {scores['core'].hex()}"
            )


def _filtered_spearman_input() -> tuple[Any, Any]:
    import numpy as np

    features: list[tuple[float, float]] = [
        (_f32(1.0e20), _f32(1.0e20)),
        (_f32(-1.0e20), _f32(-1.0e20)),
    ]
    target = [_f32(16.5), _f32(48.5)]
    for value in range(1, 64):
        root = _f32(math.sqrt(value))
        features.extend(((root, root), (_f32(-root), _f32(-root))))
        target.extend((_f32(value), _f32(value)))
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(target, dtype=np.float32)
    _assert_equal(matrix.shape, (128, 2), "filtered-Spearman/input-shape")
    if not bool(np.isfinite(matrix).all() and np.isfinite(targets).all()):
        raise AssertionError("filtered-Spearman base inputs must all be finite")
    return matrix, targets


def _assert_filtered_spearman_report(
    report: object, backend: str, precision: str, label: str
) -> None:
    _validate_backend_info(report, backend, precision)
    higher = [item for item in report.interactions if len(item.combo) == 2]
    if len(higher) != 1:
        raise AssertionError(
            f"{label}: expected one binary interaction, got {len(higher)}"
        )
    candidate = higher[0]
    score = _assert_finite(candidate.metrics["spearman"], f"{label}/spearman")
    if score != 1.0:
        raise AssertionError(
            f"{label}: filtered-candidate Spearman must be exactly 1.0, got {score!r}; "
            "the matrix-wide cached target-rank result is about 0.9999711416313609"
        )
    if backend in {"core", "cuda", "rocm"}:
        _assert_equal(
            candidate.precision_diagnostics_available,
            True,
            f"{label}/diagnostics-available",
        )
        _assert_equal(candidate.source_nonfinite, False, f"{label}/finite-base-sources")
        _assert_equal(
            candidate.interaction_overflow_rows,
            2,
            f"{label}/subset-overflow-rows",
        )
        expected_ratio = _f32(2.0 / 128.0) if precision == "fp32" else 2.0 / 128.0
        _assert_equal(
            candidate.interaction_overflow_ratio,
            expected_ratio,
            f"{label}/subset-overflow-ratio-width",
        )
    top = list(report.interactions.top_k(1, metric_name="spearman"))
    _assert_equal(len(top), 1, f"{label}/top-one-size")
    _assert_equal(
        top[0].candidate_id, candidate.candidate_id, f"{label}/visible-ranking"
    )


def _run_filtered_spearman_adversary(
    backends: Sequence[str], graph_support: Mapping[str, bool], stats: GateStats
) -> None:
    features, target = _filtered_spearman_input()
    for backend in backends:
        profiles = ("fp32",) if backend == "metal" else ("fp32", "mixed")
        for precision in profiles:
            label = f"{backend}/{precision}/filtered-Spearman"
            config = gafime.EngineConfig(
                backend=backend,
                precision=precision,
                metric_names=("spearman",),
                permutation_tests=0,
                num_repeats=1,
                random_seed=29,
                budget=gafime.ComputeBudget(
                    max_comb_size=2,
                    max_combinations_per_k=2,
                    top_features_for_higher_k=2,
                    keep_in_vram=False,
                ),
            )
            engine = gafime.GafimeEngine(config)
            eager = engine.analyze(features, target, feature_names=("left", "right"))
            stats.eager_analyses += 1
            _assert_filtered_spearman_report(
                eager, backend, precision, f"{label}/eager"
            )
            artifact = engine.compile(
                features,
                target,
                feature_names=("left", "right"),
                flags=gafime.CompileFlags(
                    plan=True, graph=bool(graph_support[backend])
                ),
            )
            try:
                compiled = artifact.analyze()
                stats.compiled_analyses += 1
                _assert_filtered_spearman_report(
                    compiled, backend, precision, f"{label}/compiled"
                )
                _assert_equal(
                    tuple(_item_identity(item) for item in compiled.interactions),
                    tuple(_item_identity(item) for item in eager.interactions),
                    f"{label}/compiled-vs-eager-identities",
                )
                for compiled_item, eager_item in zip(
                    compiled.interactions, eager.interactions
                ):
                    _assert_close(
                        compiled_item.metrics["spearman"],
                        eager_item.metrics["spearman"],
                        precision,
                        f"{label}/{compiled_item.candidate_id}/compiled-vs-eager",
                    )
                if graph_support[backend]:
                    if not artifact.graph_replayed:
                        raise AssertionError(f"{label}: graph replay was not confirmed")
                    stats.graph_replays += 1
            finally:
                artifact.close()
        stats.filtered_spearman_cases += 1


def _run_overflow_ratio_width_adversary(
    backends: Sequence[str], stats: GateStats
) -> None:
    features = [[value] * 5 for value in (0.0, 0.0, 1.0e8)]
    target = [0.0, 1.0, 2.0]
    names = tuple(f"x{index}" for index in range(5))
    for backend in backends:
        profiles = ("fp32",) if backend == "metal" else ("fp32", "mixed")
        for precision in profiles:
            label = f"{backend}/{precision}/overflow-ratio-width"
            report = gafime.GafimeEngine(
                gafime.EngineConfig(
                    backend=backend,
                    precision=precision,
                    metric_names=("pearson",),
                    permutation_tests=0,
                    num_repeats=1,
                    budget=gafime.ComputeBudget(
                        max_comb_size=5,
                        max_combinations_per_k=64,
                        top_features_for_higher_k=5,
                        keep_in_vram=False,
                    ),
                )
            ).analyze(features, target, feature_names=names)
            _validate_backend_info(report, backend, precision)
            arity_five = next(
                item for item in report.interactions if len(item.combo) == 5
            )
            _assert_equal(
                arity_five.interaction_overflow_rows,
                1,
                f"{label}/overflow-rows",
            )
            expected = _f32(1.0 / 3.0) if precision == "fp32" else 1.0 / 3.0
            _assert_equal(
                arity_five.interaction_overflow_ratio,
                expected,
                label,
            )
            stats.eager_analyses += 1
            stats.overflow_ratio_cases += 1


def _run_core_covariance_nonfinite_adversary(stats: GateStats) -> None:
    import numpy as np

    target = [1.0, -1.0, -1.0, 1.0]
    for precision, magnitude in (("fp32", 1.0e20), ("fp64", 1.0e200)):
        features = np.asarray(
            [
                [target[row], magnitude if row % 2 == 0 else -magnitude]
                for row in range(4)
            ],
            dtype=np.float64,
        )
        config = gafime.EngineConfig(
            backend="core",
            precision=precision,
            metric_names=("pearson", "r2"),
            permutation_tests=3,
            num_repeats=2,
            random_seed=29,
            significance_top_n=1,
            budget=gafime.ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=2,
                keep_in_vram=False,
            ),
        )
        report = gafime.GafimeEngine(config).analyze(
            features, target, feature_names=("finite", "covariance_overflow")
        )
        stats.eager_analyses += 1
        _validate_backend_info(report, "core", precision)
        good = next(item for item in report.interactions if item.combo == (0,))
        overflow = next(item for item in report.interactions if item.combo == (1,))
        for metric in ("pearson", "r2"):
            if not math.isnan(float(overflow.metrics[metric])):
                raise AssertionError(
                    f"Core/{precision}/{metric}: covariance overflow must remain NaN, "
                    f"got {overflow.metrics[metric]!r}"
                )
        top = list(report.interactions.top_k(1, metric_name="pearson"))
        _assert_equal(
            tuple(item.candidate_id for item in top),
            (good.candidate_id,),
            f"Core/{precision}/nonfinite-ranking-exclusion",
        )
        for kind in ("permutations", "stability"):
            rows = list(getattr(report, kind))
            if not rows:
                raise AssertionError(f"Core/{precision}: {kind} shortlist is empty")
            if any(item.candidate_id == overflow.candidate_id for item in rows):
                raise AssertionError(
                    f"Core/{precision}: nonfinite candidate entered {kind} shortlist"
                )
            if any(item.candidate_id != good.candidate_id for item in rows):
                raise AssertionError(
                    f"Core/{precision}: unexpected candidate entered {kind} shortlist"
                )
        stats.permutation_reports += 1
        stats.stability_reports += 1
        stats.covariance_nonfinite_cases += 1


def _probe_backend(backend: str) -> bool:
    expected_profiles = _expected_profiles(backend)
    graph_support: bool | None = None
    for precision in expected_profiles:
        try:
            capabilities = gafime.backend_capabilities(
                backend, probe=True, precision=precision, mi_bins=8, mi_approximate=True
            )
        except Exception as exc:
            raise AssertionError(
                f"explicitly requested backend={backend!r} precision={precision!r} "
                f"could not be probed: {exc}"
            ) from exc
        if capabilities.selection_status != "available":
            raise AssertionError(
                f"explicitly requested backend={backend!r} is unavailable: "
                f"status={capabilities.selection_status!r} "
                f"detail={capabilities.selection_detail!r}"
            )
        _assert_equal(
            capabilities.selected_backend,
            backend,
            f"{backend}/{precision}/capability-selection",
        )
        contract = capabilities.precision_contract.value
        _assert_equal(
            tuple(contract["supported_profiles"]),
            expected_profiles,
            f"{backend}/{precision}/advertised-profiles",
        )
        _assert_equal(
            contract["requested"], precision, f"{backend}/{precision}/cap-requested"
        )
        _assert_equal(
            contract["effective"], precision, f"{backend}/{precision}/cap-effective"
        )
        _assert_equal(
            contract["request_supported"],
            True,
            f"{backend}/{precision}/cap-supported",
        )
        for domain, expected in EXPECTED_DOMAINS[precision].items():
            _assert_equal(
                contract[domain], expected, f"{backend}/{precision}/cap-{domain}"
            )
        expected_accumulator = EXPECTED_DOMAINS[precision]["reduction_dtype"]
        for metric in ALL_METRICS:
            _assert_equal(
                contract["accumulators"].get(metric),
                expected_accumulator,
                f"{backend}/{precision}/{metric}/cap-accumulator",
            )
        graph_value = capabilities.graph_support.value
        current_graph = (
            graph_value.get("supported") is True
            if isinstance(graph_value, Mapping)
            else graph_value is True
        )
        if graph_support is None:
            graph_support = current_graph
        elif graph_support != current_graph:
            raise AssertionError(
                f"{backend}: graph support changed across precision profiles"
            )
    assert graph_support is not None
    return graph_support


class _ExplodingInput:
    def __iter__(self):
        raise AssertionError("unsupported Metal precision iterated or coerced input")


def _validate_metal_fail_closed() -> None:
    from gafime import v1_adapter

    v1_adapter._clear_analyze_cache_for_tests()
    for precision in ("mixed", "fp64"):
        try:
            gafime.GafimeEngine(
                gafime.EngineConfig(backend="metal", precision=precision)
            ).analyze(_ExplodingInput(), _ExplodingInput())
        except (ValueError, gafime.V1UnsupportedError) as exc:
            message = str(exc).lower()
            if "metal" not in message or precision not in message:
                raise AssertionError(
                    f"Metal {precision} rejection is not actionable: {exc}"
                ) from exc
        else:
            raise AssertionError(f"Metal unexpectedly accepted precision={precision!r}")
        if v1_adapter._current_analyze_cache():
            raise AssertionError(
                f"Metal {precision} rejection mutated the resident analyze cache"
            )


def _continuous_case() -> FamilyCase:
    import numpy as np

    features = []
    target = []
    replacement = []
    for row in range(48):
        x0 = (row - 23.5) / 9.0
        x1 = math.sin(row * 0.31)
        x2 = (((row * 7) % 23) - 11) / 7.0
        x3 = math.cos(row * 0.17) + 0.025 * (row % 5)
        x4 = ((row % 9) - 4) / 4.0
        features.append((x0, x1, x2, x3, x4))
        target.append(1.3 * x0 - 0.8 * x1 + 0.5 * x2 * x3 + 0.2 * x4 * x4)
        replacement.append(-0.4 * x0 + 1.1 * x3 + 0.7 * x1 * x4 - 0.15 * x2)
    return FamilyCase(
        name="continuous",
        features=np.asarray(features, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        replacement_target=np.asarray(replacement, dtype=np.float64),
        feature_names=("trend", "wave", "modulo", "curve", "phase"),
        config_fields={},
        required_family="interaction",
        required_arities=frozenset((1, 2, 3)),
    )


def _time_series_case() -> FamilyCase:
    import numpy as np

    rows = 56
    features = []
    target = []
    replacement = []
    for row in range(rows):
        trend = row / 12.0
        features.append((trend,))
        lagged_trend = max(row - 1, 0) / 12.0
        lagged_cycle = math.sin(max(row - 2, 0) * 0.31)
        value = 0.9 * lagged_trend - 0.55 * lagged_cycle
        target.append(value)
        # The resident time-series matrix is deliberately retained across a
        # target replacement. Keep the source-screening order stable so this
        # comparison tests typed target replacement rather than asking the
        # resident API to rebuild a different feature plan.
        replacement.append(1.15 * value + 0.04 * math.cos(row * 0.13))
    return FamilyCase(
        name="time_series",
        features=np.asarray(features, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        replacement_target=np.asarray(replacement, dtype=np.float64),
        feature_names=("trend",),
        config_fields={
            "enable_time_series_functions": True,
            "time_series_lags": (1, 2),
            "time_series_windows": (3,),
        },
        required_family="time_series",
        required_arities=frozenset((1, 2, 3)),
    )


def _decision_path_case() -> FamilyCase:
    import numpy as np

    features = []
    target = []
    replacement = []
    for q0 in (0, 1):
        for q1 in (0, 1):
            for q2 in (0, 1):
                for sample in range(6):
                    f0 = (0.18 if q0 == 0 else 0.82) + 0.003 * sample
                    f1 = (0.22 if q1 == 0 else 0.78) + 0.002 * sample
                    f2 = (0.15 if q2 == 0 else 0.85) + 0.0025 * sample
                    wave = math.sin((len(features) + 1) * 0.37)
                    phase = ((len(features) * 7) % 13 - 6) / 6.0
                    features.append((f0, f1, f2, wave, phase))
                    target.append(
                        (6.0 if q0 and q1 else 0.0)
                        + (1.5 if q2 and not q1 else 0.0)
                        + 0.08 * wave
                    )
                    replacement.append(
                        (5.0 if q1 and q2 else 0.0)
                        + (1.25 if q0 and not q2 else 0.0)
                        - 0.06 * phase
                    )
    return FamilyCase(
        name="decision_path",
        features=np.asarray(features, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        replacement_target=np.asarray(replacement, dtype=np.float64),
        feature_names=("gate_a", "gate_b", "gate_c", "wave", "phase"),
        config_fields={
            "enable_decision_path_functions": True,
            "decision_path_max_depth": 3,
            "decision_path_rounds": 2,
            "decision_path_max_paths": 3,
            "decision_path_max_bins": 8,
            "decision_path_min_leaf": 3,
            "decision_path_top_k_features": 5,
        },
        required_family="decision_path",
        required_arities=frozenset((1, 2, 3, 4, 5)),
    )


def _family_config(backend: str, precision: str, case: FamilyCase) -> Any:
    max_arity = max(case.required_arities)
    maximum = 64 if case.name == "continuous" else 256
    return gafime.EngineConfig(
        backend=backend,
        precision=precision,
        metric_names=ALL_METRICS,
        permutation_tests=3,
        num_repeats=2,
        random_seed=29,
        mi_bins=8,
        mi_approximate=True,
        significance_top_n=8,
        budget=gafime.ComputeBudget(
            max_comb_size=max_arity,
            max_combinations_per_k=maximum,
            top_features_for_higher_k=64,
            keep_in_vram=False,
            max_time_series_candidates=128,
            top_k_features_for_time_series=5,
        ),
        **case.config_fields,
    )


def _run_family_case(
    backend: str,
    precision: str,
    graph_supported: bool,
    case: FamilyCase,
    stats: GateStats,
) -> tuple[object, object]:
    label = f"{backend}/{precision}/{case.name}"
    started = time.perf_counter()
    config = _family_config(backend, precision, case)
    engine = gafime.GafimeEngine(config)

    eager = engine.analyze(case.features, case.target, feature_names=case.feature_names)
    stats.eager_analyses += 1
    _validate_report(eager, backend, precision, case, f"{label}/eager")

    plain = engine.compile(
        case.features,
        case.target,
        feature_names=case.feature_names,
        flags=gafime.CompileFlags(plan=True, graph=False),
    )
    graph = None
    try:
        if plain.graph_requested:
            raise AssertionError(
                f"{label}: plain artifact unexpectedly requested graph"
            )
        plain_report = plain.analyze()
        stats.compiled_analyses += 1
        if plain.graph_replayed:
            raise AssertionError(f"{label}: plain artifact unexpectedly replayed graph")
        _validate_public_result_width(plain_report, precision, f"{label}/compiled")
        _assert_report_equivalent(
            plain_report, eager, precision, f"{label}/compiled-vs-eager"
        )

        active = plain
        active_report = plain_report
        if graph_supported:
            graph = engine.compile(
                case.features,
                case.target,
                feature_names=case.feature_names,
                flags=gafime.CompileFlags(plan=True, graph=True),
            )
            if not graph.graph_requested:
                raise AssertionError(f"{label}: graph artifact lost graph request")
            graph_report = graph.analyze()
            stats.compiled_analyses += 1
            stats.graph_replays += 1
            if not graph.graph_replayed:
                raise AssertionError(
                    f"{label}: native boundary did not confirm graph replay"
                )
            _validate_public_result_width(graph_report, precision, f"{label}/graph")
            _assert_report_equivalent(
                graph_report, plain_report, precision, f"{label}/graph-vs-plain"
            )
            active = graph
            active_report = graph_report
            stats.graph_profile_family_cases += 1

        repeated = active.analyze()
        stats.compiled_analyses += 1
        if graph_supported:
            stats.graph_replays += 1
            if not active.graph_replayed:
                raise AssertionError(
                    f"{label}: repeated graph replay was not confirmed"
                )
        _validate_public_result_width(
            repeated, precision, f"{label}/deterministic-repeat"
        )
        _assert_report_equivalent(
            repeated, active_report, precision, f"{label}/deterministic-repeat"
        )

        if active.update_target(case.replacement_target) is not active:
            raise AssertionError(
                f"{label}: update_target did not preserve artifact identity"
            )
        stats.target_replacements += 1
        updated = active.analyze()
        stats.compiled_analyses += 1
        if graph_supported:
            stats.graph_replays += 1
            if not active.graph_replayed:
                raise AssertionError(
                    f"{label}: target-update graph replay was not confirmed"
                )
        fresh = engine.analyze(
            case.features,
            case.replacement_target,
            feature_names=case.feature_names,
        )
        stats.eager_analyses += 1
        _validate_report(updated, backend, precision, case, f"{label}/updated")
        _validate_public_result_width(fresh, precision, f"{label}/target-update-fresh")
        _assert_report_equivalent(
            updated, fresh, precision, f"{label}/target-update-vs-fresh"
        )
        _assert_equal(
            tuple(active.feature_names),
            tuple(fresh.feature_names),
            f"{label}/resident-target-feature-identity",
        )
    finally:
        if graph is not None:
            graph.close()
        plain.close()

    stats.family_cases += 1
    stats.permutation_reports += 5 + int(graph_supported)
    stats.stability_reports += 5 + int(graph_supported)
    elapsed = time.perf_counter() - started
    stats.timings.append(
        {
            "backend": backend,
            "precision": precision,
            "case": case.name,
            "seconds": round(elapsed, 6),
            "rows": int(len(case.target)),
            "interaction_rows": len(eager.interactions),
            "permutation_rows": len(eager.permutations),
            "stability_rows": len(eager.stability),
            "graph": graph_supported,
        }
    )
    return eager, updated


def _adversarial_ingest_matrix() -> tuple[Any, Any]:
    import numpy as np

    delta = 2.0**-30
    values = np.asarray([1.0 + row * delta for row in range(32)], dtype=np.float64)
    return values.reshape((-1, 1)), values.copy()


def _pearson(report: object, label: str) -> float:
    rows = list(report.interactions)
    if len(rows) != 1:
        raise AssertionError(f"{label}: expected one unary result, got {len(rows)}")
    return _assert_finite(rows[0].metrics["pearson"], f"{label}/pearson")


def _assert_ingest_score(score: float, precision: str, label: str) -> None:
    expected = 1.0 if precision == "fp64" else 0.0
    tolerance = 1.0e-12 if precision == "fp64" else 0.0
    if not math.isclose(score, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"{label}: precision={precision!r} ingest score {score!r} != {expected!r}"
        )


def _run_ingest_cases(backend: str, profiles: Sequence[str], stats: GateStats) -> None:
    import polars as pl

    features, target = _adversarial_ingest_matrix()
    with tempfile.TemporaryDirectory(prefix="gafime-precision-ingest-") as directory:
        path = Path(directory) / "precision.arrow"
        pl.DataFrame({"signal": features[:, 0], "target": target}).write_ipc(path)
        for precision in profiles:
            config = gafime.EngineConfig(
                backend=backend,
                precision=precision,
                metric_names=("pearson",),
                permutation_tests=0,
                num_repeats=1,
                budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=1),
            )
            direct = gafime.GafimeEngine(config).analyze(
                features, target, feature_names=("signal",)
            )
            loaded = gafime.dataload(path, target="target", config=config)
            stats.eager_analyses += 2
            _validate_backend_info(direct, backend, precision)
            _validate_backend_info(loaded, backend, precision)
            direct_score = _pearson(direct, f"{backend}/{precision}/numpy")
            loaded_score = _pearson(loaded, f"{backend}/{precision}/arrow")
            _assert_ingest_score(direct_score, precision, f"{backend}/NumPy")
            _assert_ingest_score(loaded_score, precision, f"{backend}/Arrow")
            _assert_close(
                loaded_score,
                direct_score,
                precision,
                f"{backend}/{precision}/Arrow-vs-NumPy",
            )
            stats.numpy_ingest_cases += 1
            stats.arrow_dataload_cases += 1


def _run_cache_separation(
    backend: str, profiles: Sequence[str], stats: GateStats
) -> None:
    from gafime import v1_adapter

    features, target = _adversarial_ingest_matrix()
    previous_capacity = os.environ.get("GAFIME_V1_ANALYZE_CACHE_SIZE")
    os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = "8"
    v1_adapter._clear_analyze_cache_for_tests()
    try:
        observed: dict[str, float] = {}
        for index, precision in enumerate(profiles, start=1):
            config = gafime.EngineConfig(
                backend=backend,
                precision=precision,
                metric_names=("pearson",),
                permutation_tests=0,
                num_repeats=1,
                budget=gafime.ComputeBudget(
                    max_comb_size=1,
                    max_combinations_per_k=1,
                    keep_in_vram=True,
                ),
            )
            report = gafime.GafimeEngine(config).analyze(
                features, target, feature_names=("signal",)
            )
            _validate_backend_info(report, backend, precision)
            score = _pearson(report, f"{backend}/{precision}/cache-first")
            _assert_ingest_score(score, precision, f"{backend}/cache")
            observed[precision] = score
            _assert_equal(
                len(v1_adapter._current_analyze_cache()),
                index,
                f"{backend}/{precision}/profile-cache-identity",
            )
            repeated = gafime.GafimeEngine(config).analyze(
                features, target, feature_names=("signal",)
            )
            stats.eager_analyses += 2
            _assert_equal(
                len(v1_adapter._current_analyze_cache()),
                index,
                f"{backend}/{precision}/cache-reuse",
            )
            _assert_close(
                _pearson(repeated, f"{backend}/{precision}/cache-repeat"),
                score,
                precision,
                f"{backend}/{precision}/cache-repeat",
            )
        if "fp64" in observed and observed["fp64"] == observed.get("mixed"):
            raise AssertionError(
                f"{backend}: adversarial fp64 cache result was reused from mixed"
            )
        stats.cache_profile_entries += len(profiles)
    finally:
        v1_adapter._clear_analyze_cache_for_tests()
        if previous_capacity is None:
            os.environ.pop("GAFIME_V1_ANALYZE_CACHE_SIZE", None)
        else:
            os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = previous_capacity


def main() -> None:
    started = time.perf_counter()
    arguments = _parse_args()
    backends = _requested_backends(arguments)
    graph_support = {backend: _probe_backend(backend) for backend in backends}
    if "metal" in backends:
        _validate_metal_fail_closed()

    stats = GateStats()
    _run_mixed_mi_boundary_adversary(backends, stats)
    _run_filtered_spearman_adversary(backends, graph_support, stats)
    _run_overflow_ratio_width_adversary(backends, stats)
    _run_core_covariance_nonfinite_adversary(stats)
    parity: dict[tuple[str, str, str], tuple[str, object]] = {}
    cases = (_continuous_case(), _time_series_case(), _decision_path_case())
    for backend in backends:
        profiles = _expected_profiles(backend)
        for precision in profiles:
            stats.backend_profile_cases += 1
            for case in cases:
                initial, updated = _run_family_case(
                    backend,
                    precision,
                    graph_support[backend],
                    case,
                    stats,
                )
                for phase, report in (("initial", initial), ("updated", updated)):
                    key = (precision, case.name, phase)
                    baseline = parity.get(key)
                    if baseline is None:
                        parity[key] = (backend, report)
                    else:
                        baseline_backend, baseline_report = baseline
                        _assert_cross_backend_equivalent(
                            report,
                            baseline_report,
                            precision,
                            f"{backend}-vs-{baseline_backend}/{precision}/"
                            f"{case.name}/{phase}",
                        )
                        stats.cross_backend_comparisons += 1
        _run_ingest_cases(backend, profiles, stats)
        _run_cache_separation(backend, profiles, stats)

    elapsed = time.perf_counter() - started
    summary = {
        "status": "passed",
        "backends": list(backends),
        "profiles": {
            backend: list(_expected_profiles(backend)) for backend in backends
        },
        "counts": {
            key: value for key, value in vars(stats).items() if key != "timings"
        },
        "timings": stats.timings,
        "total_seconds": round(elapsed, 6),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
