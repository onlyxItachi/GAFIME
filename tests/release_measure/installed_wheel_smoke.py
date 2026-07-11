#!/usr/bin/env python3
"""Clean installed-wheel load, symbol, and CPU backend smoke."""
from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.machinery
import importlib.metadata
import math
from pathlib import Path
import sys


REQUIRED_BOUNDARY_SYMBOLS = (
    "BOUNDARY_NAME",
    "CompiledContinuousArtifact",
    "ContinuousReport",
    "analyze_continuous",
    "analyze_continuous_cpu",
    "compile_continuous",
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
        str(path).endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
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


def _assert_significance_identity(gafime: object) -> None:
    config = gafime.EngineConfig(
        backend="core",
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
    interactions = {
        item.candidate_id: item.family for item in report.interactions
    }
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
        raise AssertionError("generated time-series significance identity was not exercised")


def _assert_cpu_backend(gafime: object) -> None:
    config = gafime.EngineConfig(
        backend="core",
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
        if (
            plan is None
            or int(plan.rows) != len(X)
            or int(plan.cols) != len(names)
        ):
            raise AssertionError("installed wheel compile did not expose the expected plan")
        compiled_report = compiled.analyze()
        if compiled_report.backend.name != "v1-rust-cpu":
            raise AssertionError(
                f"compiled core resolved to {compiled_report.backend.name!r}"
            )
        compiled_interactions = list(compiled_report.interactions)
        eager_by_id = {item.candidate_id: item for item in interactions}
        compiled_by_id = {item.candidate_id: item for item in compiled_interactions}
        if eager_by_id.keys() != compiled_by_id.keys():
            raise AssertionError("installed wheel eager/compiled candidate identities differ")
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
    _assert_arrow_target_contract(boundary)

    installed_version = importlib.metadata.version("gafime")
    if installed_version != gafime.__version__:
        raise AssertionError(
            f"distribution/package version mismatch: "
            f"{installed_version!r} != {gafime.__version__!r}"
        )
    _assert_cpu_backend(gafime)
    _assert_significance_identity(gafime)
    print(
        f"INSTALLED WHEEL: PASS version={installed_version} "
        f"package={package_path} boundary={boundary_path} backend=v1-rust-cpu"
    )


if __name__ == "__main__":
    main()
