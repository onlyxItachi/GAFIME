#!/usr/bin/env python3
"""Independent Polars producer controls; never acquisition or numerical execution.

Use the public collector report's exact fixtures, installed wheels and variants.
Three fresh uninstrumented processes cover each shape/format. Polars import,
public file reading and literal projection are inside the interval; authentication
matches the public worker. Frames remain alive at the frozen RSS snapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

SCHEMA = "gafime.public-ingestion-parser.v1"
READERS = {"csv": "read_csv", "parquet": "read_parquet", "ipc": "read_ipc"}


def load_public_harness():
    path = Path(__file__).with_name("perf_14_public_ingest.py")
    spec = importlib.util.spec_from_file_location("_perf14_parser_public", path)
    if spec is None or spec.loader is None:
        raise ValueError("public ingestion collector unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(spec: dict) -> dict:
    harness = load_public_harness()
    initial_policy = harness.initial_allocator_policy()
    case = spec["case"]
    if set(case) != {"rows", "cols", "format"} or case["format"] not in READERS:
        raise ValueError("unknown producer case")
    if any(type(case[name]) is not int or case[name] <= 0 for name in ("rows", "cols")):
        raise ValueError("producer dimensions must be positive integers")
    harness.check_bounds(case["rows"], case["cols"], "light")
    for name in ("GAFIME_CUDA_V1_LIB", "GAFIME_ROCM_V1_LIB", "GAFIME_METAL_V1_LIB"):
        os.environ[name] = str(Path(spec["directory"]) / "unavailable-payload")
    os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = "2"
    import importlib.metadata as metadata
    import numpy  # noqa: F401 -- match the public worker's import scope
    import gafime
    import gafime.gafime_py as native

    expected = spec["variant"]["expected_version"]
    if gafime.__version__ != expected or metadata.version("gafime") != expected:
        raise ValueError("runtime/distribution identity differs from assigned wheel")
    identity = harness.verify_wheel(
        Path(spec["variant"]["wheel"]), expected, Path(gafime.__file__).resolve().parent
    )
    harness.verify_loaded_native(Path(native.__file__), identity)
    imported = harness.memory_now()
    # Match cheap declaration setup, without invoking any product loader or
    # analyzing data. The producer itself has no precision conversion policy.
    harness.build_config(
        {
            **case,
            "precision": "mixed",
            "workflow": "light",
            "cache_state": "miss",
            "backend": "auto-core",
        }
    )
    before = harness.memory_now()
    cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
    import polars as pl

    frame = getattr(pl, READERS[case["format"]])(spec["fixture"]["file"]["path"])
    names = spec["fixture"]["feature_names"]
    features, target = frame[names], frame[["target"]]
    wall_ns, cpu_ns = (
        time.perf_counter_ns() - wall_start,
        time.process_time_ns() - cpu_start,
    )
    after = harness.memory_now()
    # Only after the snapshot: validate metadata and bind the actual consumed
    # fixture. Do not normalize product allocations into the producer floor.
    if (
        frame.height != case["rows"]
        or features.width != case["cols"]
        or target.width != 1
    ):
        raise ValueError("producer shape mismatch")
    dtype = {"float32": pl.Float32, "float64": pl.Float64}[
        spec["fixture"]["source_dtype"]
    ]
    if any(item != dtype for item in features.dtypes + target.dtypes):
        raise ValueError("producer source dtype mismatch")
    if frame.columns != [*names, "target"]:
        raise ValueError("producer source/projection names mismatch")
    for key in ("file", "features", "target"):
        actual = harness.file_identity(Path(spec["fixture"][key]["path"]))
        if actual != spec["fixture"][key]:
            raise ValueError("fixture changed after public collection")
    return {
        "schema": SCHEMA + ".worker",
        "case": case,
        "variant": spec["variant"],
        "identity": identity,
        "native": harness.file_identity(Path(native.__file__)),
        "python": {"executable": sys.executable, "version": sys.version},
        "dependencies": harness.runtime_dependencies(),
        "fixture": spec["fixture"],
        "reader": {
            "name": READERS[case["format"]],
            "kwargs": {},
            "features": names,
            "target": "target",
        },
        "instrumented": False,
        "treatment": "normal",
        "numerical_acquisition": False,
        "frames_retained_until_snapshot": True,
        "harness": harness.file_identity(Path(__file__)),
        "public_harness": harness.file_identity(Path(harness.__file__)),
        "measurement_context": harness.measurement_context(initial_policy),
        "timing": {
            "wall_nanoseconds": wall_ns,
            "process_cpu_nanoseconds": cpu_ns,
            "cpu_core_equivalents": cpu_ns / wall_ns,
        },
        "memory": {
            "after_import": imported,
            "before_call": before,
            "after_call_before_snapshot": after,
            "inherited_resource_ru_maxrss_used": False,
            "peak_scope": "fresh exec, installed imports plus Polars read/projection only",
        },
        "scope": "Independent public Polars reader; no GAFIME dataload, acquisition or executor call",
    }


def driver(args) -> int:
    harness = load_public_harness()
    source = Path(args.public_report)
    report = json.loads(source.read_text())
    if report["schema"] != harness.SCHEMA or report["status"] != "collected":
        raise ValueError("complete public report required for producer binding")
    if args.repeats < 3 or not all(
        math.isfinite(x) and x > 0 for x in (args.timeout, args.max_time_seconds)
    ):
        raise ValueError("at least three repeats and positive time bounds required")
    fixtures = {}
    for record in report["records"]:
        case = record["case"]
        if record["status"] != "passed":
            raise ValueError("failed public worker")
        key = (case["rows"], case["cols"], case["format"])
        fixture = record["result"]["fixture"]
        if key in fixtures and fixtures[key] != fixture:
            raise ValueError("mixed producer fixtures")
        fixtures[key] = fixture
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [
        (v, k, rep)
        for v in report["variants"]
        for k in sorted(fixtures)
        for rep in range(args.repeats)
    ]
    random.Random(17).shuffle(jobs)
    records, started = [], time.monotonic()
    for index, (variant, key, repeat) in enumerate(jobs):
        if time.monotonic() - started >= args.max_time_seconds:
            raise ValueError("producer campaign time budget exhausted")
        case = dict(zip(("rows", "cols", "format"), key))
        stem = output / f"parser-{index:05d}"
        spec = {
            "variant": variant,
            "case": case,
            "fixture": fixtures[key],
            "repeat": repeat,
        }
        # Use the same declared unavailable-payload directory as public workers;
        # this path is not an allocator or execution setting.
        spec["directory"] = str(Path(source).parent)
        spec_path = stem.with_suffix(".spec.json")
        spec_path.write_text(json.dumps(spec, indent=2))
        try:
            run = subprocess.run(
                [
                    variant["python"],
                    "-I",
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(spec_path),
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            stem.with_suffix(".stdout.log").write_text(run.stdout)
            stem.with_suffix(".stderr.log").write_text(run.stderr)
            record = {
                "status": "passed" if run.returncode == 0 else "failed",
                "returncode": run.returncode,
            }
            if run.returncode == 0:
                record["result"] = json.loads(run.stdout)
        except subprocess.TimeoutExpired:
            record = {"status": "failed", "reason": "timeout"}
        record.update(variant=variant["label"], case=case, repeat=repeat)
        records.append(record)
        stem.with_suffix(".result.json").write_text(json.dumps(record, indent=2))
        print(
            json.dumps({"parser_sample": index, "status": record["status"]}), flush=True
        )
    passed = len(records) == len(jobs) and all(x["status"] == "passed" for x in records)
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "collected" if passed else "failed",
        "collector_status": "passed" if passed else "failed",
        "release_ready": False,
        "requested_jobs": len(jobs),
        "records": records,
        "variants": report["variants"],
        "harness": harness.file_identity(Path(__file__)),
        "public_harness": report["harness"],
        "public_report": harness.file_identity(source),
        "scope": "Independent producer controls; no numerical execution, trimming or diagnostic treatments",
    }
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return 0 if passed else 1


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        try:
            print(
                json.dumps(
                    worker(json.loads(Path(sys.argv[2]).read_text())), allow_nan=False
                )
            )
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-time-seconds", type=float, default=240)
    args = parser.parse_args()
    try:
        return driver(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
