"""Installed-wheel native platform validation.

This script is intentionally stdlib-only. It exercises public GAFIME APIs from
an installed wheel and writes a compact JSON result for CI artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

try:
    import gafime
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gafime

from gafime import ComputeBudget, EngineConfig, GafimeEngine, gafime_core


def run_command(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[:6000],
            "stderr": completed.stderr.strip()[:6000],
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error_type": type(exc).__name__, "error": str(exc)}


def runtime_specs(label: str) -> dict[str, Any]:
    system = platform.system().lower()
    specs: dict[str, Any] = {
        "label": label,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "gafime_version": getattr(gafime, "__version__", "unknown"),
        "precision": gafime_core.precision_name(),
        "dispatch": gafime_core.cpu_dispatch_target(),
        "available_dispatch_targets": list(gafime_core.available_cpu_dispatch_targets()),
    }
    if system == "linux":
        specs["lscpu"] = run_command(["lscpu"])
        specs["memory"] = run_command(["bash", "-lc", "grep -E 'MemTotal|MemAvailable' /proc/meminfo"])
    elif system == "darwin":
        specs["sysctl"] = run_command(
            [
                "sysctl",
                "machdep.cpu.brand_string",
                "hw.machine",
                "hw.model",
                "hw.physicalcpu",
                "hw.logicalcpu",
                "hw.memsize",
            ]
        )
        specs["sw_vers"] = run_command(["sw_vers"])
    elif system == "windows":
        specs["processor_info"] = run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Processor | "
                    "Select-Object -First 1 Name,Manufacturer,Architecture,NumberOfCores,NumberOfLogicalProcessors | "
                    "ConvertTo-Json -Compress"
                ),
            ]
        )
    return specs


def make_dataset(n_samples: int, n_features: int, seed: int) -> tuple[list[list[float]], list[float], list[str]]:
    if n_features < 12:
        raise ValueError("n_features must be at least 12")
    rng = random.Random(seed)
    rows: list[list[float]] = []
    target: list[float] = []
    for i in range(n_samples):
        row: list[float] = []
        for j in range(n_features):
            value = (
                math.sin(0.0019 * (i + 1) * (j + 1))
                + 0.33 * math.cos(0.0047 * (i + 3) * (j + 2))
                + 0.018 * (((i * (j + 11)) % 41) - 20)
                + rng.uniform(-0.003, 0.003)
            )
            row.append(float(value))
        row[-1] = 1.0
        row[-2] = -1.0 if i % 2 else 1.0
        lag = rows[max(i - 4, 0)][8] if rows else row[8]
        gate = 1.0 / (1.0 + math.exp(-12.0 * (row[2] - 0.12)))
        rectangle = (
            1.0 / (1.0 + math.exp(-10.0 * (row[6] + 0.35)))
            * 1.0 / (1.0 + math.exp(-10.0 * (0.75 - row[6])))
            * 1.0 / (1.0 + math.exp(-10.0 * (row[7] + 0.20)))
            * 1.0 / (1.0 + math.exp(-10.0 * (0.85 - row[7])))
        )
        y = (
            row[0] * row[1]
            + 0.24 * row[3] * row[4] * row[5]
            + 0.95 * gate
            + 0.70 * rectangle
            + 0.18 * lag
            + rng.uniform(-0.006, 0.006)
        )
        rows.append(row)
        target.append(float(y))
    return rows, target, [f"f{i}" for i in range(n_features)]


def validate(args: argparse.Namespace) -> dict[str, Any]:
    rows, target, names = make_dataset(args.n_samples, args.n_features, args.seed)
    budget = ComputeBudget(
        max_comb_size=args.max_comb_size,
        max_combinations_per_k=args.max_combinations_per_k,
        top_features_for_higher_k=min(args.n_features, 64),
        max_discrete_candidates=args.max_discrete_candidates,
        max_feature_pairs_for_rectangles=args.max_rectangle_pairs,
        top_k_features_for_discrete=min(args.n_features, 24),
        max_time_series_candidates=args.max_time_series_candidates,
        top_k_features_for_time_series=min(args.n_features, 16),
        vram_budget_mb=32768,
    )
    config = EngineConfig(
        backend=args.backend,
        metric_names=("pearson", "r2"),
        enable_discrete_functions=True,
        enable_time_series_functions=True,
        time_series_lags=(1, 2, 4, 8, 16),
        time_series_windows=(4, 8, 16, 32),
        permutation_tests=0,
        num_repeats=1,
        random_seed=args.seed,
        budget=budget,
    )
    started = time.perf_counter()
    report = GafimeEngine(config).analyze(rows, target, feature_names=names)
    elapsed = time.perf_counter() - started

    families: dict[str, int] = {}
    for item in report.interactions:
        families[item.family] = families.get(item.family, 0) + 1
    if not report.interactions:
        raise AssertionError("validation produced no interactions")
    if not report.decision or not report.decision.signal_detected:
        raise AssertionError("validation did not detect the planted signal")
    required = {"interaction", "discrete_function", "time_series_function"}
    if not required.issubset(families):
        raise AssertionError(f"missing expected families: {families}")

    top = sorted(
        report.interactions,
        key=lambda item: max(abs(v) if k in ("pearson", "spearman") else v for k, v in item.metrics.items()),
        reverse=True,
    )[:10]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = report.to_dict()
    deprecated_export_warned = any(item.category is DeprecationWarning for item in caught)

    return {
        "status": "ok",
        "label": args.label,
        "runtime": runtime_specs(args.label),
        "dataset": {
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "seed": args.seed,
            "max_comb_size": args.max_comb_size,
        },
        "engine": {
            "requested_backend": args.backend,
            "resolved_backend": report.backend.name if report.backend else None,
            "backend": {
                "name": report.backend.name,
                "device": report.backend.device,
                "is_gpu": report.backend.is_gpu,
                "memory_total_mb": report.backend.memory_total_mb,
                "memory_free_mb": report.backend.memory_free_mb,
            } if report.backend else None,
            "seconds": elapsed,
            "interactions": len(report.interactions),
            "families": families,
            "native_report": bool(getattr(report.interactions, "is_native_backed", False)),
            "deprecated_to_dict_warned": deprecated_export_warned,
            "warnings": report.warnings[:20],
            "decision": {
                "signal_detected": report.decision.signal_detected,
                "message": report.decision.message,
            } if report.decision else None,
            "top": [
                {
                    "combo": list(item.combo),
                    "family": item.family,
                    "features": list(item.feature_names),
                    "candidate_id": item.candidate_id,
                    "pearson": item.metrics.get("pearson"),
                    "r2": item.metrics.get("r2"),
                }
                for item in top
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--max-comb-size", type=int, default=4)
    parser.add_argument("--max-combinations-per-k", type=int, default=20000)
    parser.add_argument("--max-discrete-candidates", type=int, default=4000)
    parser.add_argument("--max-rectangle-pairs", type=int, default=400)
    parser.add_argument("--max-time-series-candidates", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    result = validate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
