"""Small distribution-wheel benchmark for GitHub-hosted runners.

This script is intentionally compact. It validates that an installed wheel can
load its native payloads, find planted interactions, tolerate small edge cases,
and report runner/backend timing data that can be copied into release notes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import gafime
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gafime

from gafime import ComputeBudget, EngineConfig, GafimeEngine, gafime_core
from gafime.backends import resolve_backend
from gafime.utils.arrays import coerce_inputs


def _run_command(args: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"{type(exc).__name__}: {exc}"
    text = (result.stdout or result.stderr).strip()
    return text[:8000]


def collect_specs(label: str, wheel_path: str | None) -> dict[str, object]:
    specs: dict[str, object] = {
        "label": label,
        "wheel_path": wheel_path,
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "gafime_version": gafime.__version__,
        "native_precision": gafime_core.precision_name(),
        "native_dispatch": gafime_core.cpu_dispatch_target(),
        "available_dispatch_targets": list(gafime_core.available_cpu_dispatch_targets()),
    }
    system = platform.system().lower()
    if system == "linux":
        specs["lscpu"] = _run_command(["lscpu"])
        specs["memory"] = _run_command(["bash", "-lc", "grep -E 'MemTotal|MemAvailable' /proc/meminfo"])
    elif system == "darwin":
        specs["sysctl"] = _run_command(
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
    elif system == "windows":
        specs["processor"] = _run_command(
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
        specs["computer"] = _run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_ComputerSystem | "
                    "Select-Object Manufacturer,Model,TotalPhysicalMemory | "
                    "ConvertTo-Json -Compress"
                ),
            ]
        )
    return specs


def make_interaction_dataset(n: int = 768, p: int = 8) -> tuple[list[list[float]], list[float], list[str]]:
    X: list[list[float]] = []
    y: list[float] = []
    for i in range(n):
        row = [
            math.sin(i * 0.013),
            math.cos(i * 0.017),
            ((i * 7) % 31 - 15) / 9.0,
            ((i * 11) % 37 - 18) / 11.0,
            math.sin(i * 0.031 + 0.4),
            math.cos(i * 0.029 - 0.2),
            ((i * i) % 43 - 21) / 13.0,
            1.0,
        ][:p]
        signal = row[0] * row[1] + 0.18 * row[2] - 0.04 * row[4]
        X.append(row)
        y.append(signal)
    return X, y, [f"f{i}" for i in range(p)]


def analyze_once(
    X: list[list[float]],
    y: list[float],
    names: list[str],
    *,
    max_comb_size: int = 2,
) -> object:
    cfg = EngineConfig(
        backend="core",
        metric_names=("pearson", "r2", "mutual_info"),
        budget=ComputeBudget(
            max_comb_size=max_comb_size,
            max_combinations_per_k=256,
        ),
        permutation_tests=0,
        num_repeats=1,
    )
    return GafimeEngine(cfg).analyze(X, y, names)


def benchmark_case(name: str, fn, repeats: int = 3) -> dict[str, object]:
    timings: list[float] = []
    payload: dict[str, object] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        payload = fn()
        timings.append(time.perf_counter() - start)
    assert payload is not None
    return {
        "name": name,
        "seconds": {
            "min": min(timings),
            "median": statistics.median(timings),
            "max": max(timings),
        },
        "payload": payload,
    }


def continuous_interaction_check() -> dict[str, object]:
    X, y, names = make_interaction_dataset()
    report = analyze_once(X, y, names, max_comb_size=2)
    ranked = sorted(report.interactions, key=lambda item: abs(item.metrics.get("pearson", 0.0)), reverse=True)
    top = ranked[:8]
    found_pair = any(set(item.combo) == {0, 1} for item in top)
    if not found_pair:
        raise AssertionError("planted f0 x f1 pair was not found in top 8 continuous interactions")
    return {
        "backend": report.backend.name,
        "dispatch": gafime_core.cpu_dispatch_target(),
        "n_interactions": len(report.interactions),
        "found_planted_pair_top8": found_pair,
        "top": [
            {
                "combo": list(item.combo),
                "family": item.family,
                "features": list(item.feature_names),
                "pearson": item.metrics.get("pearson"),
                "r2": item.metrics.get("r2"),
            }
            for item in top[:3]
        ],
    }


def edge_case_check() -> dict[str, object]:
    X, y, names = make_interaction_dataset(n=96, p=6)
    for row in X:
        row[-1] = 1.0
    report = analyze_once(X, y, names, max_comb_size=3)
    if not report.interactions:
        raise AssertionError("edge-case small dataset returned no interactions")
    X2, y2, _ = coerce_inputs(X[:16], y[:16], names)
    return {
        "small_dataset_interactions": len(report.interactions),
        "coerced_nbytes": {"X": X2.nbytes, "y": y2.nbytes},
        "top_family": report.interactions[0].family,
    }


def cuda_check() -> dict[str, object]:
    X, y, _ = coerce_inputs([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], [0.0, 1.0, 1.5, 2.0])
    try:
        backend, warnings = resolve_backend(EngineConfig(backend="cuda", metric_names=("pearson",)), X, y)
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}
    return {"status": "available", "backend": dataclasses.asdict(backend.info()), "warnings": warnings}


def run_benchmarks(label: str) -> dict[str, object]:
    available_targets = list(gafime_core.available_cpu_dispatch_targets())
    selected_targets = ["Default"]
    for target in ("SSE4.2", "AVX2", "AVX512", "NEON"):
        if target in available_targets:
            selected_targets.append(target)

    dispatch_results = []
    previous = os.environ.get("GAFIME_CPU_DISPATCH")
    try:
        for target in dict.fromkeys(selected_targets):
            if target == "Default":
                os.environ.pop("GAFIME_CPU_DISPATCH", None)
            elif target in {"SSE4.2", "AVX2", "AVX512"}:
                os.environ["GAFIME_CPU_DISPATCH"] = target
            else:
                os.environ.pop("GAFIME_CPU_DISPATCH", None)
            dispatch_results.append(benchmark_case(f"continuous_interaction_{target}", continuous_interaction_check))
    finally:
        if previous is None:
            os.environ.pop("GAFIME_CPU_DISPATCH", None)
        else:
            os.environ["GAFIME_CPU_DISPATCH"] = previous

    return {
        "label": label,
        "specs": collect_specs(label, os.environ.get("GAFIME_BENCHMARK_WHEEL")),
        "checks": {
            "edge_cases": benchmark_case("edge_cases", edge_case_check),
            "cuda": cuda_check(),
        },
        "dispatch_benchmarks": dispatch_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_benchmarks(args.label)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    specs = result["specs"]
    print(f"GAFIME wheel benchmark: {args.label}")
    print(f"  platform: {specs['platform']}")
    print(f"  machine: {specs['machine']}")
    print(f"  dispatch: {specs['native_dispatch']}")
    print(f"  available dispatch targets: {', '.join(specs['available_dispatch_targets'])}")
    for item in result["dispatch_benchmarks"]:
        seconds = item["seconds"]
        payload = item["payload"]
        print(
            f"  {item['name']}: median={seconds['median']:.6f}s "
            f"found_pair={payload['found_planted_pair_top8']} dispatch={payload['dispatch']}"
        )
    print(f"  cuda: {result['checks']['cuda']['status']}")


if __name__ == "__main__":
    main()
