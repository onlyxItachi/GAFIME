#!/usr/bin/env python3
"""Clean installed-wheel load, symbol, and CPU backend smoke."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.machinery
import importlib.metadata
import inspect
import math
import sys
from array import array
from pathlib import Path


REQUIRED_BOUNDARY_SYMBOLS = (
    "BOUNDARY_NAME",
    "CompiledContinuousArtifact",
    "ContinuousReport",
    "analyze_continuous",
    "analyze_continuous_cpu",
    "compile_continuous",
)
SEMANTIC_PUBLIC_EXPORTS = (
    "AcceptedSet",
    "Candidate",
    "CandidateSet",
    "Constraint",
    "Evidence",
    "EvidenceReport",
    "FeatureTable",
    "Graph",
    "Labels",
    "SelectionPolicy",
    "Snapshot",
    "TabularSession",
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _remove_checkout_paths(source_root: Path) -> None:
    clean_path = []
    for entry in sys.path:
        try:
            resolved = (Path(entry) if entry else Path.cwd()).resolve()
        except OSError:
            clean_path.append(entry)
            continue
        if not _is_within(resolved, source_root):
            clean_path.append(entry)
    sys.path[:] = clean_path


def _assert_installed(module: object, source_root: Path, label: str) -> Path:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise AssertionError(f"{label} has no import path")
    path = Path(raw_path).resolve()
    if _is_within(path, source_root):
        raise AssertionError(f"{label} imported from checkout: {path}")
    if not path.is_file():
        raise AssertionError(f"{label} import path does not exist: {path}")
    return path


def _assert_native_extension(path: Path) -> None:
    if not any(
        str(path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise AssertionError(f"native boundary is not an extension module: {path}")
    library = ctypes.CDLL(str(path))
    try:
        getattr(library, "PyInit_gafime_py")
    except AttributeError as exc:
        raise AssertionError(f"{path.name} does not export PyInit_gafime_py") from exc


def _assert_boundary_symbols(boundary: object) -> None:
    missing = [
        name for name in REQUIRED_BOUNDARY_SYMBOLS if not hasattr(boundary, name)
    ]
    if missing:
        raise AssertionError(f"native boundary is missing symbols: {missing}")
    for name in ("analyze_continuous", "analyze_continuous_cpu", "compile_continuous"):
        if not callable(getattr(boundary, name)):
            raise AssertionError(f"native boundary symbol {name} is not callable")


def _assert_direct_cpu_precision_contract(boundary: object) -> None:
    function = boundary.analyze_continuous_cpu
    signature = inspect.signature(function)
    precision_parameter = signature.parameters.get("precision")
    if (
        precision_parameter is None
        or precision_parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        or precision_parameter.default != "mixed"
    ):
        raise AssertionError(
            "analyze_continuous_cpu must expose keyword-only precision='mixed'"
        )

    delta = 2.0**-30
    features = [[1.0 + index * delta] for index in range(16)]
    target = [1.0 + index * delta for index in range(16)]
    scores = {}
    for precision in ("fp32", "mixed", "fp64"):
        report = function(features, target, 1, 1, [1], precision=precision)
        expected_storage = "float64" if precision == "fp64" else "float32"
        expected_result = "float32" if precision == "fp32" else "float64"
        actual = (report.precision, report.storage_dtype, report.result_dtype)
        expected = (precision, expected_storage, expected_result)
        if actual != expected:
            raise AssertionError(
                f"direct Core precision contract {actual!r} != {expected!r}"
            )
        scores[precision] = float(report.metric_values(0)[0])

    if scores["fp32"] != 0.0 or scores["mixed"] != 0.0:
        raise AssertionError(
            f"direct f32-storage profiles preserved sub-f32 spacing: {scores!r}"
        )
    if not math.isclose(scores["fp64"], 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(f"direct fp64 input was quantized through f32: {scores!r}")
    if function(features, target, 1, 1, [1]).precision != "mixed":
        raise AssertionError("direct Core compatibility default is not mixed")
    try:
        function(features, target, 1, 1, [1], "fp64")
    except TypeError:
        pass
    else:
        raise AssertionError("direct Core precision unexpectedly accepted positionally")


def _assert_arrow_target_contract(boundary: object) -> None:
    import polars as pl

    features = pl.DataFrame({"x": [1.0, 2.0]}).cast(pl.Float32).rechunk()
    invalid_targets = (
        (
            pl.DataFrame({"y": [1.0, 2.0], "extra": [3.0, 4.0]})
            .cast(pl.Float32)
            .rechunk(),
            "exactly one column",
        ),
        (
            pl.DataFrame({"y": [1.0, None]}).cast(pl.Float32).rechunk(),
            "null target values",
        ),
    )
    for target, expected in invalid_targets:
        try:
            boundary.analyze_continuous_arrow(
                features,
                target,
                max_arity=1,
                max_combinations_per_k=2,
                metric_ids=[1],
            )
        except ValueError as exc:
            if expected not in str(exc):
                raise AssertionError(
                    f"native Arrow rejection {exc!r} did not contain {expected!r}"
                ) from exc
        else:
            raise AssertionError(
                f"native Arrow boundary accepted invalid target: {expected}"
            )

    delta = 2.0**-30
    features_f64 = (
        pl.DataFrame({"x": [1.0 + index * delta for index in range(16)]})
        .cast(pl.Float64)
        .rechunk()
    )
    target_f64 = (
        pl.DataFrame({"y": [1.0 + index * delta for index in range(16)]})
        .cast(pl.Float64)
        .rechunk()
    )
    report = boundary.analyze_continuous_arrow(
        features_f64,
        target_f64,
        precision="fp64",
        max_arity=1,
        max_combinations_per_k=1,
        metric_ids=[1],
    )
    if report.precision != "fp64" or report.storage_dtype != "float64":
        raise AssertionError("native Arrow fp64 path did not preserve its dtype")
    if not math.isclose(
        float(report.record(0).metrics[0]),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("native Arrow fp64 path rounded through Float32")
    for precision, expected in (("mixed", "Float32"), ("fp32", "Float32")):
        try:
            boundary.analyze_continuous_arrow(
                features_f64,
                target_f64,
                precision=precision,
                max_arity=1,
                max_combinations_per_k=1,
                metric_ids=[1],
            )
        except ValueError as exc:
            if expected not in str(exc):
                raise AssertionError(
                    f"native Arrow precision={precision!r} rejection {exc!r} "
                    f"did not contain {expected!r}"
                ) from exc
        else:
            raise AssertionError(
                f"native Arrow precision={precision!r} accepted Float64 input"
            )


def _assert_significance_identity(gafime: object, precision: str) -> None:
    config = gafime.EngineConfig(
        backend="core",
        precision=precision,
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(3,),
        metric_names=("pearson",),
        permutation_tests=2,
        num_repeats=2,
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=32),
    )
    X = [[float(index), float((index * 5) % 11)] for index in range(16)]
    y = [float((index * 3 + 1) % 13) for index in range(16)]
    report = gafime.GafimeEngine(config).analyze(X, y, ["trend", "cycle"])
    interactions = {item.candidate_id: item.family for item in report.interactions}
    significance = [*report.permutations, *report.stability]
    if not significance:
        raise AssertionError("installed wheel produced no requested significance rows")
    for item in significance:
        if interactions.get(item.candidate_id) != item.family:
            raise AssertionError(
                "significance identity does not match its interaction row: "
                f"{item.candidate_id!r} family={item.family!r}"
            )
    if not any(item.family == "time_series" for item in significance):
        raise AssertionError(
            "generated time-series significance identity was not exercised"
        )


def _assert_cpu_backend(gafime: object, precision: str) -> None:
    config = gafime.EngineConfig(
        backend="core",
        precision=precision,
        metric_names=("pearson", "r2"),
        permutation_tests=0,
        num_repeats=1,
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )
    X = [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    names = ["ascending", "descending"]

    report = gafime.GafimeEngine(config).analyze(X, y, names)
    if report.backend is None or report.backend.name != "v1-rust-cpu":
        raise AssertionError(f"core resolved to unexpected backend: {report.backend!r}")
    if report.backend.is_gpu:
        raise AssertionError("core backend incorrectly reported is_gpu=True")
    expected_storage = "float64" if precision == "fp64" else "float32"
    expected_result = "float32" if precision == "fp32" else "float64"
    expected_precision_contract = (
        precision,
        precision,
        expected_storage,
        expected_storage,
        expected_result,
        expected_result,
    )
    actual_precision_contract = (
        report.backend.requested_precision,
        report.backend.effective_precision,
        report.backend.storage_dtype,
        report.backend.interaction_arithmetic,
        report.backend.reduction_dtype,
        report.backend.result_dtype,
    )
    if actual_precision_contract != expected_precision_contract:
        raise AssertionError(
            f"installed Core precision={precision!r} domains "
            f"{actual_precision_contract!r} != {expected_precision_contract!r}"
        )
    interactions = list(report.interactions)
    if not interactions:
        raise AssertionError("installed wheel CPU analysis produced no interactions")
    for item in interactions:
        if set(item.metrics) != {"pearson", "r2"}:
            raise AssertionError(
                f"installed wheel returned unexpected metrics for {item.candidate_id}: "
                f"{sorted(item.metrics)}"
            )
    expected_metrics = {
        (0,): {"pearson": 1.0, "r2": 1.0},
        (1,): {"pearson": -1.0, "r2": 1.0},
    }
    eager_by_combo = {tuple(item.combo): item for item in interactions}
    if eager_by_combo.keys() != expected_metrics.keys():
        raise AssertionError(
            f"installed wheel returned unexpected unary combos: {sorted(eager_by_combo)}"
        )
    for combo, metrics in expected_metrics.items():
        for name, expected in metrics.items():
            actual = float(eager_by_combo[combo].metrics[name])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7):
                raise AssertionError(
                    f"installed wheel {combo} {name}={actual!r}, expected {expected!r}"
                )

    compiled = gafime.compile(
        X,
        y,
        names,
        config=config,
        flags=gafime.CompileFlags(plan=True),
    )
    try:
        plan = compiled.scenario_plan
        if plan is None or int(plan.rows) != len(X) or int(plan.cols) != len(names):
            raise AssertionError(
                "installed wheel compile did not expose the expected plan"
            )
        compiled_report = compiled.analyze()
        if compiled_report.backend.name != "v1-rust-cpu":
            raise AssertionError(
                f"compiled core resolved to {compiled_report.backend.name!r}"
            )
        compiled_interactions = list(compiled_report.interactions)
        eager_by_id = {item.candidate_id: item for item in interactions}
        compiled_by_id = {item.candidate_id: item for item in compiled_interactions}
        if eager_by_id.keys() != compiled_by_id.keys():
            raise AssertionError(
                "installed wheel eager/compiled candidate identities differ"
            )
        for candidate_id, eager_item in eager_by_id.items():
            compiled_item = compiled_by_id[candidate_id]
            if (
                eager_item.family != compiled_item.family
                or tuple(eager_item.combo) != tuple(compiled_item.combo)
                or eager_item.metrics.keys() != compiled_item.metrics.keys()
            ):
                raise AssertionError(
                    f"installed wheel eager/compiled metadata differs for {candidate_id}"
                )
            for name, eager_value in eager_item.metrics.items():
                compiled_value = float(compiled_item.metrics[name])
                if not math.isclose(
                    float(eager_value), compiled_value, rel_tol=0.0, abs_tol=1.0e-7
                ):
                    raise AssertionError(
                        f"installed wheel eager/compiled {candidate_id} {name} differs: "
                        f"{eager_value!r} != {compiled_value!r}"
                    )
    finally:
        compiled.close()


def _assert_fp64_ingest_never_rounds_through_fp32(gafime: object) -> None:
    delta = 2.0**-30
    features = [[1.0 + index * delta] for index in range(16)]
    target = [1.0 + index * delta for index in range(16)]
    scores = {}
    for precision in ("fp32", "mixed", "fp64"):
        report = gafime.GafimeEngine(
            gafime.EngineConfig(
                backend="core",
                precision=precision,
                metric_names=("pearson",),
                permutation_tests=0,
                num_repeats=1,
                budget=gafime.ComputeBudget(
                    max_comb_size=1,
                    max_combinations_per_k=1,
                ),
            )
        ).analyze(features, target, ["sub-f32-spacing"])
        rows = list(report.interactions)
        if len(rows) != 1:
            raise AssertionError(
                f"precision={precision!r} adversarial ingest returned {len(rows)} rows"
            )
        scores[precision] = float(rows[0].metrics["pearson"])
    if scores["fp32"] != 0.0 or scores["mixed"] != 0.0:
        raise AssertionError(
            f"f32-storage profiles unexpectedly preserved sub-f32 spacing: {scores!r}"
        )
    if not math.isclose(scores["fp64"], 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(
            f"fp64 ingest was quantized before Core execution: {scores!r}"
        )


def _f32_matrix(rows: list[tuple[float, ...]]) -> tuple[array, memoryview]:
    """Build a 2-D stdlib buffer without introducing NumPy into wheel smoke."""

    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise AssertionError("semantic wheel-smoke fixture must be rectangular")
    storage = array("f", (value for row in rows for value in row))
    matrix = memoryview(storage).cast("B").cast("f", shape=(len(rows), len(rows[0])))
    return storage, matrix


def _assert_semantic_lifecycle(gafime: object, source_root: Path) -> None:
    semantic = importlib.import_module("gafime.semantic")
    _assert_installed(semantic, source_root, "gafime.semantic")
    if tuple(getattr(semantic, "__all__", ())) != SEMANTIC_PUBLIC_EXPORTS:
        raise AssertionError("installed semantic module changed its public inventory")
    missing = [name for name in SEMANTIC_PUBLIC_EXPORTS if not hasattr(semantic, name)]
    if missing:
        raise AssertionError(f"installed semantic module is missing exports: {missing}")

    storage, matrix = _f32_matrix([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)])
    session = semantic.TabularSession(
        matrix,
        feature_names=["left", "right"],
        row_keys=[101, 102, 103, 104],
        row_domain="installed-wheel-smoke",
        provenance="installed-wheel-smoke-input",
    )
    feature_table = None
    try:
        if session.configured_backend != "auto" or session.selected_backend != "core":
            raise AssertionError(
                "semantic session did not distinguish configured auto from selected Core"
            )
        if session.precision != "mixed":
            raise AssertionError("semantic default precision is not mixed")
        capabilities = session.capabilities
        if (
            capabilities["configured_device_id"] != 0
            or capabilities["selected_device_id"] is not None
        ):
            raise AssertionError(
                "semantic Core selection did not report requested versus selected device"
            )
        snapshot = session.frame
        if (
            snapshot.feature_names != ["left", "right"]
            or snapshot.row_keys != [101, 102, 103, 104]
            or snapshot.row_domain != "installed-wheel-smoke"
            or snapshot.role != "discovery"
        ):
            raise AssertionError("semantic snapshot lost installed input identity")
        storage[0] = 99.0
        if session.frame.feature_names != ["left", "right"]:
            raise AssertionError("semantic snapshot no longer exposes stable schema")
        if session.begin_round() != 1:
            raise AssertionError("semantic session did not begin its first round")
        left = session.source("left")
        labels = snapshot.labels(
            row_keys=[101, 102, 103, 104],
            values=[0.0, 1.0, 2.0, 3.0],
            provenance="installed-wheel-smoke-labels",
        )
        channel = semantic.Evidence.labels("outcome", labels)
        report = session.evaluate([left], [channel])
        value = report.value(left, channel)
        if (
            value["state"] != "measured"
            or value["support"] != 4
            or value["reason"] is not None
            or value["value"] is None
            or abs(value["value"] - 1.0) > 1.0e-12
        ):
            raise AssertionError(
                "semantic installed-wheel lifecycle did not preserve copied input"
            )
        accepted = session.select(
            report,
            semantic.SelectionPolicy(channel, direction="maximize", limit=1),
        )
        if len(accepted) != 1:
            raise AssertionError(
                "semantic installed-wheel selection rejected a valid row"
            )
        inference_storage, inference = _f32_matrix([(9.0, 19.0)])
        snapshot_inference = session.snapshot(
            inference,
            feature_names=["left", "right"],
            row_keys=[9001],
            row_domain="installed-wheel-inference",
            provenance="installed-wheel-inference",
        )
        feature_table = session.transform(accepted, snapshot_inference)
        if (
            feature_table.row_keys != [9001]
            or feature_table.rows != 1
            or feature_table.precision != "mixed"
            or len(feature_table.__arrow_c_array__()) != 2
        ):
            raise AssertionError(
                "semantic installed-wheel transform lost inference identity"
            )
        _ = inference_storage
    finally:
        session.close()
        session.close()

    if feature_table is None or len(feature_table.__arrow_c_array__()) != 2:
        raise AssertionError("semantic Arrow output did not outlive its closed session")

    try:
        _ = session.frame
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed semantic session still exposes a live snapshot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="checkout root that must not provide the imported gafime package",
    )
    args = parser.parse_args()
    source_root = (args.source_root or Path.cwd()).resolve()
    _remove_checkout_paths(source_root)

    gafime = importlib.import_module("gafime")
    boundary = importlib.import_module("gafime.gafime_py")
    package_path = _assert_installed(gafime, source_root, "gafime")
    boundary_path = _assert_installed(boundary, source_root, "gafime.gafime_py")
    _assert_native_extension(boundary_path)
    _assert_boundary_symbols(boundary)
    _assert_direct_cpu_precision_contract(boundary)
    _assert_arrow_target_contract(boundary)

    installed_version = importlib.metadata.version("gafime")
    if installed_version != gafime.__version__:
        raise AssertionError(
            f"distribution/package version mismatch: "
            f"{installed_version!r} != {gafime.__version__!r}"
        )
    for precision in ("fp32", "mixed", "fp64"):
        _assert_cpu_backend(gafime, precision)
        _assert_significance_identity(gafime, precision)
    _assert_fp64_ingest_never_rounds_through_fp32(gafime)
    _assert_semantic_lifecycle(gafime, source_root)
    print(
        f"INSTALLED WHEEL: PASS version={installed_version} "
        f"package={package_path} boundary={boundary_path} backend=v1-rust-cpu"
    )


if __name__ == "__main__":
    main()
