#!/usr/bin/env python3
"""One bounded Linux allocator investigation; never a release-gate replacement.

Runs the unchanged source collector and budget validator first. Diagnostic
workers and external /proc polling are separately labelled and cannot enter
that verdict. All subprocesses are fresh execs and run sequentially.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def run(command: list[str], stem: Path, timeout: float = 600) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = None, error.stdout or b"", error.stderr or b""
    stem.with_suffix(".stdout.log").write_bytes(stdout)
    stem.with_suffix(".stderr.log").write_bytes(stderr)
    return {"command": command, "returncode": code}


def proc_fields(text: str) -> dict:
    """Keep kernel quantities named and in KiB, not additive component costs."""
    fields = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator:
            words = value.split()
            if len(words) == 2 and words[0].isdigit() and words[1] == "kB":
                fields[name + "_kib"] = int(words[0])
            elif name == "Threads" and value.strip().isdigit():
                fields[name] = int(value)
    return fields


def sampled_run(
    command: list[str], stem: Path, timeout: float = 60, env: dict | None = None
) -> dict:
    """External parent can observe native calls even when Python holds the GIL.

    Polling is diagnostic overhead. Fast transitions may still be missed, and
    /proc/status RSS uses approximate accounting; smaps_rollup is separate.
    """
    start = time.monotonic()
    with (
        stem.with_suffix(".stdout.log").open("wb") as stdout,
        stem.with_suffix(".stderr.log").open("wb") as stderr,
        stem.with_suffix(".proc.jsonl").open("w") as samples,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        root = Path(f"/proc/{process.pid}")
        next_smaps = 0
        count = 0
        timed_out = False
        while process.poll() is None:
            now = time.monotonic_ns()
            record = {"monotonic_ns": now, "pid": process.pid}
            try:
                record["status"] = proc_fields((root / "status").read_text())
                if now >= next_smaps:
                    record["smaps_rollup"] = proc_fields(
                        (root / "smaps_rollup").read_text()
                    )
                    next_smaps = now + 5_000_000
                samples.write(json.dumps(record) + "\n")
                count += 1
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass  # Process exit races are not treated as zero-memory samples.
            if time.monotonic() - start > timeout:
                process.kill()
                timed_out = True
                break
            time.sleep(0.001)
        code = process.wait(timeout=10)
    result = {
        "command": command,
        "returncode": code,
        "timed_out": timed_out,
        "external_samples": count,
        "instrumented": True,
        "sample_interval_seconds": 0.001,
        "smaps_interval_seconds": 0.005,
        "not_a_peak_guarantee": True,
    }
    if code == 0:
        try:
            result["result"] = json.loads(stem.with_suffix(".stdout.log").read_text())
        except ValueError:
            result["invalid_json"] = True
    return result


def treatment_env(name: str, prior: dict | None = None) -> dict:
    env = (os.environ if prior is None else prior).copy()
    # These settings apply only to isolated diagnostic processes. The normal
    # control inherits the observed runner environment, not a sanitized one.
    if name in {"decay-zero", "thp-never", "thp-data", "thp-all"}:
        env.pop("POLARS_THP", None)
        env["_RJEM_MALLOC_CONF"] = {
            "decay-zero": "dirty_decay_ms:0,muzzy_decay_ms:0",
            "thp-never": "thp:never,metadata_thp:disabled",
            "thp-data": "thp:always,metadata_thp:disabled",
            "thp-all": "thp:always,metadata_thp:always",
        }[name]
    return env


def selected(case: dict, full: bool = False) -> bool:
    shape = (case["rows"], case["cols"])
    fmt, precision = case["format"], case["precision"]
    if not full:
        return fmt == "parquet" and precision in {"fp32", "fp64"}
    return (
        fmt == "parquet"
        and (shape == (100000, 20) or precision != "mixed")
        or shape == (100000, 20)
        and (
            fmt == "ipc"
            and precision != "mixed"
            or fmt == "csv"
            and precision == "fp32"
        )
    )


def validate_cost_verdict(value: dict, code: int | None) -> None:
    if (
        value.get("schema") != "gafime.public-ingestion-budget.v1.verdict"
        or value.get("status") not in {"passed", "failed"}
        or value.get("release_ready") is not False
        or "reason" in value
        or len(value.get("cells", [])) != 18
        or code != (0 if value["status"] == "passed" else 1)
    ):
        raise ValueError("budget validation failed to produce a complete valid verdict")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    for label in ("candidate", "cibw", "rc2"):
        parser.add_argument(f"--{label}-python", required=True)
        parser.add_argument(f"--{label}-wheel", type=Path, required=True)
    parser.add_argument("--rc2-sha", required=True)
    args = parser.parse_args()
    output, source = args.output.resolve(), args.source_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if (
        actual != args.source_sha
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source
        ).strip()
    ):
        parser.error("source must be the exact clean requested checkout")
    measure = source / "tests/release_measure"
    collector = measure / "perf_14_public_ingest.py"
    reference = output / "reference"
    command = [
        args.candidate_python,
        str(collector),
        "--python",
        f"candidate={args.candidate_python}",
        "--wheel",
        f"candidate={args.candidate_wheel.resolve()}",
        "--expected-version",
        "candidate=1.0.0rc2",
        "--source-sha",
        f"candidate={args.source_sha}",
        "--shape",
        "100000x20",
        "--shape",
        "100000x100",
        "--format",
        "csv,parquet,ipc",
        "--precision",
        "fp32,mixed,fp64",
        "--workflow",
        "light",
        "--cache-state",
        "miss",
        "--repeats",
        "3",
        "--max-time-seconds",
        "600",
        "--output",
        str(reference),
    ]
    collected = run(command, output / "collector", timeout=660)
    report = json.loads((reference / "report.json").read_text())
    verdict = run(
        [
            args.candidate_python,
            str(measure / "perf_14_ingest_budget.py"),
            "--report",
            str(reference / "report.json"),
            "--variant",
            "candidate",
            "--budget",
            str(measure / "ingest_cost_budget.json"),
            "--output",
            str(output / "unchanged-cost-verdict.json"),
        ],
        output / "budget",
    )
    cost_assessment = json.loads((output / "unchanged-cost-verdict.json").read_text())
    validate_cost_verdict(cost_assessment, verdict["returncode"])
    # A failed unchanged RSS verdict is data, not a reason to lose attribution.
    # Failed worker/parity instead ends this investigation fail closed.
    if collected["returncode"] != 0 or report["status"] != "collected":
        raise RuntimeError("uninstrumented candidate collection/parity failed")
    cases = sorted(reference.glob("sample-*.spec.json"))
    controls, diagnostics = [], []
    probe = Path(__file__).with_name("diag_15_ingest_memory_worker.py").resolve()
    tasks = []
    for path in cases:
        spec = json.loads(path.read_text())
        if selected(spec["case"]):
            for label in ("cibw", "rc2"):
                tasks.append((label, "normal", False, spec))
        if selected(spec["case"], full=True):
            tasks.append(("candidate", "normal", False, spec))
        case = spec["case"]
        if (
            case["format"] == "parquet"
            and case["cols"] == 20
            and case["precision"] in {"fp32", "fp64"}
        ):
            for treatment in (
                "glibc-trim",
                "decay-zero",
                "thp-never",
                "thp-data",
                "thp-all",
            ):
                tasks.append(("candidate", treatment, False, spec))
        if (
            case["route"] == "dataload"
            and case["precision"] == "fp32"
            and (case["cols"] == 20 or case["format"] == "parquet")
        ):
            tasks.append(("candidate", "normal", True, spec))
    random.Random(23).shuffle(tasks)
    for index, (label, treatment, parser_only, original) in enumerate(tasks):
        spec = json.loads(json.dumps(original))
        spec["instrument"] = False
        spec["variant"] = {
            "label": label,
            "python": getattr(args, f"{label}_python"),
            "wheel": str(getattr(args, f"{label}_wheel").resolve()),
            "expected_version": "1.0.0rc2",
            "source_sha": args.rc2_sha if label == "rc2" else args.source_sha,
        }
        stem = output / f"trial-{index:04d}"
        spec_path = stem.with_suffix(".spec.json")
        spec_path.write_text(json.dumps(spec, indent=2))
        python = spec["variant"]["python"]
        instrumented = label == "candidate"
        if instrumented:
            command = [
                python,
                "-I",
                str(probe),
                "--source-root",
                str(source),
                "--spec",
                str(spec_path),
                "--markers",
                str(stem.with_suffix(".markers.jsonl")),
                "--treatment",
                treatment,
            ]
            if parser_only:
                command.append("--parser-only")
            # Set process environment before Polars' private jemalloc starts.
            if parser_only:
                command.append("--uninstrumented-parser")
                record = run(command, stem, timeout=60)
                if record["returncode"] == 0:
                    record["result"] = json.loads(
                        stem.with_suffix(".stdout.log").read_text()
                    )
            else:
                record = sampled_run(command, stem, env=treatment_env(treatment))
            diagnostics.append(record)
        else:
            record = run(
                [python, "-I", str(collector), "--worker", str(spec_path)],
                stem,
                timeout=60,
            )
            if record["returncode"] == 0:
                record["result"] = json.loads(
                    stem.with_suffix(".stdout.log").read_text()
                )
            controls.append(record)
        record.update(
            label=label,
            treatment=treatment,
            parser_only=parser_only,
            case=spec["case"],
            repeat=spec["repeat"],
            stem=str(stem),
        )
        stem.with_suffix(".result.json").write_text(json.dumps(record, indent=2))
        print(
            json.dumps(
                {
                    "trial": index,
                    "label": label,
                    "treatment": treatment,
                    "returncode": record["returncode"],
                }
            ),
            flush=True,
        )
    comparisons = paired_results(controls + diagnostics)
    manifest = {
        "schema": "gafime.rc3-rss-diagnostic.v1",
        "release_ready": False,
        "source_sha": args.source_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector": collected,
        "unchanged_budget": verdict,
        "unchanged_budget_status": cost_assessment["status"],
        "controls": controls,
        "diagnostics": diagnostics,
        "same_variant_same_treatment_numeric_parity": comparisons,
        "scope": "same-host fresh execs; diagnostic sampling excluded from qualification; source capacities and exact private-jemalloc counters not exposed",
        "files": [
            identity(path) for path in sorted(output.rglob("*")) if path.is_file()
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return (
        0
        if all(
            item["returncode"] == 0 and not item.get("invalid_json")
            for item in controls + diagnostics
        )
        and all(item["passed"] for item in comparisons)
        else 1
    )


def paired_results(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        if record["parser_only"]:
            continue
        case = {key: value for key, value in record["case"].items() if key != "route"}
        key = json.dumps(
            [record["label"], record["treatment"], record["repeat"], case],
            sort_keys=True,
        )
        group = groups.setdefault(key, {})
        if record["case"]["route"] in group:
            raise ValueError("duplicate route in numeric parity group")
        group[record["case"]["route"]] = (
            record.get("result", {}).get("snapshot", {}).get("numeric")
        )
    return [
        {
            "cell": json.loads(key),
            "passed": set(routes) == {"direct", "dataload"}
            and routes["direct"] is not None
            and routes["direct"] == routes["dataload"],
        }
        for key, routes in sorted(groups.items())
    ]


if __name__ == "__main__":
    raise SystemExit(main())
