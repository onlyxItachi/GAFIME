#!/usr/bin/env python3
"""Four-version screen, one runner, immutable GAFIME and original fixture bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from urllib.request import urlopen

VERSIONS = ("1.3.0", "1.28.0", "1.39.0", "1.44.2")
SOURCE = "509d68a99a70182d117dad84eb905dae04764ec1"
CORE_HASH = "a15feea6a510c7ceb95d88e3aa5dead325bf431a38de6ad917e5c670260dcbb1"


def load(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(prior, output, h):
    core = prior / "wheels/gafime-1.0.0rc2-cp311-cp311-linux_x86_64.whl"
    if h.file_identity(core)["sha256"] != CORE_HASH:
        raise ValueError("prior Core bytes changed")
    requirements = Path(__file__).with_name("diag_15_requirements.txt").read_text()
    pinned = output / "common-requirements.txt"
    pinned.write_text(
        "\n".join(
            line
            for line in requirements.splitlines()
            if not line.startswith(("polars==", "polars-runtime-32=="))
        )
        + "\n"
    )
    variants, metadata = [], {}
    for version in VERSIONS:
        env = output.parent / ("env-polars-" + version)
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        python = str(env / "bin/python")
        subprocess.run([python, "-m", "pip", "install", "-r", str(pinned)], check=True)
        wheels = output.parent / ("polars-wheels-" + version)
        packages = ["polars"] + (
            ["polars-runtime-32"] if version in {"1.39.0", "1.44.2"} else []
        )
        for package in packages:
            subprocess.run(
                [
                    python,
                    "-m",
                    "pip",
                    "download",
                    "--no-deps",
                    "--only-binary=:all:",
                    package + "==" + version,
                    "--dest",
                    str(wheels),
                ],
                check=True,
            )
            url = f"https://pypi.org/pypi/{package}/{version}/json"
            with urlopen(url, timeout=30) as response:
                data = json.load(response)
            metadata[package + "==" + version] = {"url": url, "metadata": data}
        for wheel in sorted(wheels.glob("*.whl")):
            matches = [
                item
                for data in metadata.values()
                for item in data["metadata"]["urls"]
                if item["filename"] == wheel.name
            ]
            if (
                len(matches) != 1
                or matches[0]["digests"]["sha256"] != h.file_identity(wheel)["sha256"]
            ):
                raise ValueError("Polars wheel not authenticated by public metadata")
            subprocess.run(
                [python, "-m", "pip", "install", "--no-deps", str(wheel)], check=True
            )
        subprocess.run(
            [python, "-m", "pip", "install", "--no-deps", str(core)], check=True
        )
        variants.append(
            {
                "label": "polars-" + version,
                "python": python,
                "wheel": str(core),
                "expected_version": "1.0.0rc2",
                "source_sha": SOURCE,
            }
        )
        subprocess.run(
            [python, "-m", "pip", "freeze"],
            check=True,
            stdout=(output / ("dependencies-" + version + ".txt")).open("w"),
        )
    (output / "variants.json").write_text(json.dumps(variants, indent=2))
    (output / "polars-public-metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "core-wheel-identity.json").write_text(
        json.dumps(h.file_identity(core), indent=2)
    )


def collect(prior, output, h):
    original = json.loads((prior / "evidence/reference/report.json").read_text())
    variants = json.loads((output / "variants.json").read_text())
    fixtures = {}
    for record in original["records"]:
        fixture = deepcopy(record["result"]["fixture"])
        for key in ("file", "features", "target"):
            fixture[key]["path"] = str(
                prior / "evidence/reference/fixtures" / Path(fixture[key]["path"]).name
            )
            if h.file_identity(Path(fixture[key]["path"])) != fixture[key]:
                raise ValueError("original fixture identity changed")
        case = record["case"]
        fixtures[(case["rows"], case["cols"], case["format"])] = fixture
    public_dir = output / "public"
    public_dir.mkdir()
    cases = [
        {
            "rows": 100000,
            "cols": cols,
            "format": "parquet",
            "precision": p,
            "workflow": "light",
            "cache_state": "miss",
            "backend": "auto-core",
            "route": route,
        }
        for cols in (20, 100)
        for p in ("fp32", "fp64")
        for route in ("direct", "dataload")
    ]
    jobs = [(v, c, rep, False) for v in variants for c in cases for rep in range(3)]
    random.Random(17).shuffle(jobs)
    records, specs = [], []
    for index, (variant, case, repeat, instrument) in enumerate(jobs):
        stem = public_dir / f"sample-{index:05d}"
        fixture = fixtures[(case["rows"], case["cols"], case["format"])]
        # These miss-only trials never read changed_target, but keep the complete
        # collector schema. It names the same unchanged file, not fabricated data.
        spec = {
            "variant": variant,
            "case": case,
            "repeat": repeat,
            "instrument": instrument,
            "fixture": fixture,
            "changed_fixture": fixture,
            "directory": str(public_dir),
        }
        path = stem.with_suffix(".spec.json")
        path.write_text(json.dumps(spec, indent=2))
        record = h.run_child(variant["python"], "--worker", path, stem, 120)
        record.update(case=case, variant=variant["label"], repeat=repeat, sample=index)
        records.append(record)
        specs.append(spec)
        stem.with_suffix(".result.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"sample": index, "status": record["status"]}), flush=True)
    parity = h.public_numeric_parity(records, jobs)
    passed = (
        all(x["status"] == "passed" for x in records) and parity["status"] == "passed"
    )
    public = {
        "schema": h.SCHEMA,
        "status": "collected" if passed else "failed",
        "collector_status": "passed" if passed else "failed",
        "release_ready": False,
        "harness": h.file_identity(Path(h.__file__)),
        "variants": variants,
        "records": records,
        "requested_jobs": len(jobs),
        "public_numeric_parity": parity,
    }
    (public_dir / "report.json").write_text(json.dumps(public, indent=2))
    if not passed:
        raise ValueError(
            "Polars version screen has missing workers or unequal numeric results"
        )
    identities = {
        (r["result"]["identity"]["wheel"]["sha256"], r["result"]["native"]["sha256"])
        for r in records
    }
    contexts = {
        json.dumps(r["result"]["measurement_context"], sort_keys=True) for r in records
    }
    numpy_ids = {
        (
            r["result"]["dependencies"]["numpy"],
            r["result"]["dependencies"]["numpy_native_sha256"],
        )
        for r in records
    }
    if len(identities) != 1 or len(contexts) != 1 or len(numpy_ids) != 1:
        raise ValueError("comparison did not hold GAFIME/runtime/host/NumPy constant")
    snapshots = {}
    for record in records:
        case = record["case"]
        key = (case["rows"], case["cols"], case["precision"])
        value = record["result"]["snapshot"]["numeric_sha256"]
        if key in snapshots and snapshots[key] != value:
            raise ValueError("cross-Polars numeric snapshot differs")
        snapshots[key] = value
    # Add independent producer cases not in the Parquet-only execution screen.
    parser_dir = output / "producer"
    parser_dir.mkdir()
    parser = load("perf_14_parser_control")
    producer_jobs = [
        (v, k, rep) for v in variants for k in sorted(fixtures) for rep in range(3)
    ]
    random.Random(19).shuffle(producer_jobs)
    producers = []
    for index, (variant, key, repeat) in enumerate(producer_jobs):
        case = dict(zip(("rows", "cols", "format"), key))
        stem = parser_dir / f"parser-{index:05d}"
        spec = {
            "variant": variant,
            "case": case,
            "fixture": fixtures[key],
            "directory": str(public_dir),
        }
        path = stem.with_suffix(".spec.json")
        path.write_text(json.dumps(spec, indent=2))
        run = subprocess.run(
            [
                variant["python"],
                "-I",
                str(Path(parser.__file__)),
                "--worker",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stem.with_suffix(".stdout.log").write_text(run.stdout)
        stem.with_suffix(".stderr.log").write_text(run.stderr)
        if run.returncode != 0:
            raise ValueError("independent parser worker failed: " + stem.name)
        producers.append(
            {
                "status": "passed",
                "variant": variant["label"],
                "case": case,
                "repeat": repeat,
                "result": json.loads(run.stdout),
            }
        )
    (parser_dir / "report.json").write_text(
        json.dumps(
            {
                "schema": parser.SCHEMA,
                "status": "collected",
                "collector_status": "passed",
                "release_ready": False,
                "requested_jobs": len(producers),
                "records": producers,
                "variants": variants,
                "harness": h.file_identity(Path(parser.__file__)),
                "public_harness": public["harness"],
            },
            indent=2,
        )
    )
    # Four normal phase workers per version, explicitly excluded from controls.
    phases = output / "phases"
    phases.mkdir()
    diagnostic = load("diag_15_ingest_memory")
    for index, spec in enumerate(
        [
            s
            for s in specs
            if s["case"]["cols"] == 20
            and s["case"]["route"] == "dataload"
            and s["repeat"] in (0, 1)
        ]
    ):
        stem = phases / f"phase-{index:03d}"
        path = stem.with_suffix(".spec.json")
        path.write_text(json.dumps(spec, indent=2))
        run = diagnostic.sampled_run(
            [
                spec["variant"]["python"],
                "-I",
                str(Path(__file__).with_name("diag_15_ingest_memory_worker.py")),
                "--source-root",
                str(Path(__file__).parents[2]),
                "--spec",
                str(path),
                "--markers",
                str(stem.with_suffix(".markers.jsonl")),
            ],
            stem,
            timeout=120,
        )
        stem.with_suffix(".result.json").write_text(json.dumps(run, indent=2))
        if run["returncode"] != 0 or run.get("invalid_json"):
            raise ValueError("phase diagnostic worker failed")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "source_sha": SOURCE,
                "gafime_wheel_sha256": CORE_HASH,
                "versions": VERSIONS,
                "release_ready": False,
                "uninstrumented_public_workers": len(records),
                "numeric_pairs": len(parity["comparisons"]),
                "uninstrumented_parser_workers": len(producers),
                "instrumented_phase_workers": 16,
                "scope": "One matched runner; no production allocator overrides or source/runtime rebuild",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harness = load("perf_14_public_ingest")
    (prepare if args.prepare else collect)(
        args.prior.resolve(), args.output.resolve(), harness
    )


if __name__ == "__main__":
    main()
