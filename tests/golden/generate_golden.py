from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = Path(__file__).resolve().parent
SCHEMA = "gafime.golden.v1.p0"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gafime.config import ComputeBudget, EngineConfig  # noqa: E402
from gafime.engine import GafimeEngine  # noqa: E402


def _continuous_case() -> tuple[str, list[list[float]], list[float], list[str], EngineConfig]:
    rows = 48
    x = [
        [
            (i - 20) / 10.0,
            ((i * 7) % 19 - 9) / 5.0,
            ((i * 5) % 13 - 6) / 4.0,
        ]
        for i in range(rows)
    ]
    y = [row[0] * row[1] + 0.2 * row[2] for row in x]
    config = EngineConfig(
        backend="core",
        metric_names=("pearson", "r2"),
        budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=16),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
    )
    return "continuous_core", x, y, ["a", "b", "c"], config


def _time_series_case() -> tuple[str, list[list[float]], list[float], list[str], EngineConfig]:
    rows = 64
    x = [
        [
            math.sin(i / 5.0),
            math.cos(i / 7.0),
            ((i % 11) - 5) / 3.0,
        ]
        for i in range(rows)
    ]
    y = [
        0.7 * x[i - 1][0] + 0.2 * x[i][1] - 0.1 * x[i - 2][2] if i >= 2 else 0.0
        for i in range(rows)
    ]
    config = EngineConfig(
        backend="core",
        metric_names=("pearson", "r2"),
        enable_time_series_functions=True,
        time_series_lags=(1, 2),
        time_series_windows=(3,),
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=12,
            max_time_series_candidates=16,
            top_k_features_for_time_series=2,
        ),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
    )
    return "time_series_core", x, y, ["wave", "season", "saw"], config


def build_golden_cases() -> dict[str, dict[str, Any]]:
    previous = os.environ.get("GAFIME_USE_LEGACY_ENGINE")
    os.environ["GAFIME_USE_LEGACY_ENGINE"] = "1"
    try:
        return {
            name: _canonical_report(name, x, y, feature_names, config)
            for name, x, y, feature_names, config in (
                _continuous_case(),
                _time_series_case(),
            )
        }
    finally:
        if previous is None:
            os.environ.pop("GAFIME_USE_LEGACY_ENGINE", None)
        else:
            os.environ["GAFIME_USE_LEGACY_ENGINE"] = previous


def _canonical_report(
    case_name: str,
    x: list[list[float]],
    y: list[float],
    feature_names: list[str],
    config: EngineConfig,
) -> dict[str, Any]:
    report = GafimeEngine(config)._analyze_legacy(x, y, feature_names)
    return {
        "schema": SCHEMA,
        "case": case_name,
        "config": _jsonable(config),
        "data": {
            "rows": len(x),
            "features": len(feature_names),
            "feature_names": list(feature_names),
        },
        "backend": _jsonable(report.backend),
        "warnings": list(report.warnings),
        "decision": _jsonable(report.decision),
        "interactions": [_interaction_row(item) for item in report.interactions],
        "stability": [_stability_row(item) for item in report.stability],
        "permutations": [_permutation_row(item) for item in report.permutations],
    }


def _interaction_row(item: Any) -> dict[str, Any]:
    return {
        "combo": [int(value) for value in item.combo],
        "feature_names": [str(value) for value in item.feature_names],
        "family": str(item.family),
        "candidate_id": str(item.candidate_id),
        "expression": str(item.expression),
        "params": _jsonable(item.params),
        "metrics": _metric_map(item.metrics),
    }


def _stability_row(item: Any) -> dict[str, Any]:
    return {
        "combo": [int(value) for value in item.combo],
        "family": str(item.family),
        "candidate_id": str(item.candidate_id),
        "expression": str(item.expression),
        "params": _jsonable(item.params),
        "metrics_mean": _metric_map(item.metrics_mean),
        "metrics_std": _metric_map(item.metrics_std),
    }


def _permutation_row(item: Any) -> dict[str, Any]:
    return {
        "combo": [int(value) for value in item.combo],
        "family": str(item.family),
        "candidate_id": str(item.candidate_id),
        "expression": str(item.expression),
        "params": _jsonable(item.params),
        "p_values": _metric_map(item.p_values),
    }


def _metric_map(metrics: dict[str, float]) -> dict[str, float]:
    return {str(key): _float(metrics[key]) for key in sorted(metrics)}


def _float(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return _float(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def write_golden_cases(outdir: Path = OUTDIR) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, payload in build_golden_cases().items():
        path = outdir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_golden_case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_golden(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected dict, got {type(actual).__name__}")
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: key mismatch actual={sorted(actual)} expected={sorted(expected)}")
        for key in expected:
            compare_golden(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(f"{path}: expected list, got {type(actual).__name__}")
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length mismatch actual={len(actual)} expected={len(expected)}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_golden(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (float, int)):
            raise AssertionError(f"{path}: expected float, got {type(actual).__name__}")
        if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(f"{path}: actual={actual!r} expected={expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: actual={actual!r} expected={expected!r}")


def check_golden_cases(outdir: Path = OUTDIR) -> None:
    actual_cases = build_golden_cases()
    for name, actual in actual_cases.items():
        expected_path = outdir / f"{name}.json"
        if not expected_path.exists():
            raise AssertionError(f"Missing golden fixture: {expected_path}")
        compare_golden(actual, load_golden_case(expected_path), name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or check GAFIME v1 P0 golden fixtures.")
    parser.add_argument("--update", action="store_true", help="Write fixtures under tests/golden.")
    parser.add_argument("--check", action="store_true", help="Compare current legacy output to fixtures.")
    args = parser.parse_args()

    if args.update and args.check:
        parser.error("--update and --check are mutually exclusive")
    if args.update:
        write_golden_cases()
        return
    if args.check:
        check_golden_cases()
        return
    print(json.dumps(build_golden_cases(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
