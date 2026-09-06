#!/usr/bin/env python3
"""Bounded, non-release #73 native quality/CUDA reuse experiment.

Build product rlibs and the standard CUDA payload separately first. This runner
binds the standalone Rust harness to those explicit inputs, retains raw output,
and never imports a Python implementation of the candidate data plane.
Shared-desktop timing is diagnostic, not release performance qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/release_measure/issue73_quality_cuda.rs"
RUST_VERSION = "1.97.1"


def command(args, *, env=None, timeout=180):
    result = subprocess.run(
        [str(value) for value in args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {args}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def identity(path):
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def compile_harness(target, output):
    release = target / "release"
    deps = release / "deps"
    libraries = {
        name: release / f"lib{name}.rlib"
        for name in (
            "gafime_cpu",
            "gafime_gpu_sys",
            "gafime_orchestrator",
            "gafime_types",
        )
    }
    rayon = list(deps.glob("librayon-*.rlib"))
    if len(rayon) != 1:
        raise ValueError("Use an isolated target with exactly one Rayon rlib")
    libraries["rayon"] = rayon[0]
    base = [
        "rustup",
        "run",
        RUST_VERSION,
        "rustc",
        "--edition=2021",
        "-C",
        "opt-level=3",
        "-C",
        "codegen-units=1",
        "-D",
        "warnings",
        "-L",
        f"dependency={deps}",
    ]
    for name, path in libraries.items():
        base += ["--extern", f"{name}={path}"]
    build = [*base, "-C", "lto=fat", SOURCE, "-o", output / "native-probe"]
    command(build)
    test_build = [*base, "--test", SOURCE, "-o", output / "native-tests"]
    command(test_build)
    return {
        "commands": [list(map(str, build)), list(map(str, test_build))],
        "rlibs": {name: identity(path) for name, path in libraries.items()},
        "binary": identity(output / "native-probe"),
        "tests": command([output / "native-tests"]),
    }


def snapshot():
    result = {"unix_time": time.time(), "allowed_cpus": sorted(os.sched_getaffinity(0))}
    probes = {
        "gpu": [
            "nvidia-smi",
            (
                "--query-gpu=name,uuid,driver_version,utilization.gpu,"
                "memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw"
            ),
            "--format=csv",
        ],
        "cpu": ["ps", "-eo", "pid,ppid,pcpu,comm", "--sort=-pcpu"],
    }
    for name, args in probes.items():
        try:
            value = command(args, timeout=15)
            result[name] = (
                "\n".join(value.splitlines()[:25]) if name == "cpu" else value
            )
        except (OSError, subprocess.SubprocessError) as error:
            result[name] = {"unavailable": str(error)}
    ac = Path("/sys/class/power_supply/ADP0/online")
    result["ac_online"] = ac.read_text().strip() if ac.exists() else None
    return result


def source_identity():
    paths = command(
        [
            "git",
            "ls-files",
            "crates",
            "src",
            "Cargo.toml",
            "Cargo.lock",
            "tests/release_measure/issue73*",
            "tests/release_measure/run_issue73*",
        ]
    )
    return {
        "head": command(["git", "rev-parse", "HEAD"]),
        "tree": command(["git", "rev-parse", "HEAD^{tree}"]),
        "dirty": command(["git", "status", "--porcelain"]),
        "files": {path: identity(ROOT / path) for path in paths.splitlines()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-target", type=Path, required=True)
    parser.add_argument("--cuda-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timings", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = identity(args.cuda_lib)
    env = os.environ.copy()
    env["GAFIME_CUDA_V1_LIB"] = payload["path"]
    # Default production Rayon worker selection is retained; the exact observed
    # worker count is emitted by every native cell. No affinity/clock changes.
    env.pop("RAYON_NUM_THREADS", None)
    report = {
        "schema": "gafime.issue73.quality-cuda-reuse.v2",
        "source": source_identity(),
        "payload": payload,
        "preflight": snapshot(),
        "cells": [],
        "performance_claim": (
            "shared-host bounded diagnostics only; not saturation proof"
        ),
        "public_api_added": False,
    }
    report_path = output / "report.json"
    try:
        report["build"] = compile_harness(args.cargo_target.resolve(), output)
        for backend in ("core", "cuda"):
            raw = command([output / "native-probe", "quality", backend], env=env)
            (output / f"quality-{backend}.json").write_text(raw + "\n")
            report[f"quality_{backend}"] = json.loads(raw)
        if args.timings:
            cells = [
                (backend, rows, count, kind)
                for backend in ("core", "cuda")
                for rows in (512, 8192, 32768)
                for count in (15, 66, 253)
                for kind in ("absdiff", "product", "product_direct")
            ]
            random.Random(73).shuffle(cells)
            report["schedule"] = cells
            for index, (backend, rows, count, kind) in enumerate(cells):
                before = snapshot()
                raw = command(
                    [
                        output / "native-probe",
                        "bench",
                        backend,
                        rows,
                        count,
                        kind,
                        10,
                        30,
                        100,
                    ],
                    env=env,
                    timeout=180,
                )
                (output / f"cell-{index:03d}.json").write_text(raw + "\n")
                cell = json.loads(raw)
                samples = cell["resident_ns_per_call"]
                cell["median_resident_ns"] = statistics.median(samples)
                cell["candidate_rows_per_second"] = (
                    rows * count * 1e9 / statistics.median(samples)
                )
                cell["host_before"] = before
                cell["host_after"] = snapshot()
                report["cells"].append(cell)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{index + 1}/{len(cells)} {backend} {kind} {rows}x{count}",
                    flush=True,
                )
        # A successful numerical run is not evidence for a source tree or binary
        # that changed during collection. Preserve the partial report on failure.
        report["source_after"] = source_identity()
        if report["source_after"] != report["source"]:
            raise RuntimeError("source identity changed during evidence collection")
        if identity(args.cuda_lib) != payload:
            raise RuntimeError(
                "CUDA payload identity changed during evidence collection"
            )
        if identity(output / "native-probe") != report["build"]["binary"]:
            raise RuntimeError(
                "native harness identity changed during evidence collection"
            )
        report["completed"] = True
    except Exception as error:
        report["completed"] = False
        report["failure"] = str(error)
        raise
    finally:
        report["postflight"] = snapshot()
        report["source_after"] = source_identity()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(report_path, flush=True)


if __name__ == "__main__":
    main()
