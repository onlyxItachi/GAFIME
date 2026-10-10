#!/usr/bin/env python3
r"""Installed-wheel public file-ingestion evidence, separate from kernel timing.

The standard-library-only driver generates synthetic files in a separate
interpreter, then measures each trial in a fresh isolated subprocess. Example::

    python tests/release_measure/perf_14_public_ingest.py \
      --python rc2=/env/rc2/bin/python --wheel rc2=/artifacts/rc2.whl \
      --expected-version rc2=1.0.0rc2 --source-sha rc2=<verified-source-sha> \
      --python candidate=/env/candidate/bin/python \
      --wheel candidate=/artifacts/candidate.whl \
      --expected-version candidate=1.0.0rc3 \
      --shape 100000x20 --shape 100000x100 --format csv,parquet,ipc \
      --precision fp32,mixed,fp64 --workflow light \
      --cache-state miss,hit,disabled,target-change --repeats 3 \
      --output /scratch/ingestion-ab

Use small shapes for ``default``, ``configured``, ``time-series`` and
``decision-path`` workflows. ``--instrument`` is a separate, explicitly
instrumented component sample; it does not replace uninstrumented trials.
No absolute performance threshold or universal speedup is inferred. Native
call boundaries combine phases unless the native implementation exposes them.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from email.parser import Parser
import functools
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import statistics
import struct
import subprocess
import sys
import time
import traceback
from typing import Any
import zipfile

SCHEMA = "gafime.public-ingestion-performance.v1"
FORMATS = {"csv", "parquet", "ipc"}
PRECISIONS = {"fp32", "mixed", "fp64"}
WORKFLOWS = {"light", "configured", "default", "time-series", "decision-path"}
CACHE_STATES = {"miss", "hit", "disabled", "target-change"}
MAX_CELLS = 20_000_000
SNAPSHOT_SCOPE = {
    "included": [
        "feature_names",
        "ordered_interactions",
        "candidate_ids",
        "metric_bits",
        "metric_rankings",
        "candidate_diagnostics",
        "stability",
        "permutations",
        "decision.signal_detected",
    ],
    "excluded": {
        "config": "Recorded separately; versions and transport can differ.",
        "warnings": "Recorded separately; historical transport diagnostics differ.",
        "decision.message": "Recorded separately; transport-specific text is not numeric parity.",
        "backend": "Recorded separately, including placement and precision; live memory is not a numeric result.",
    },
    "independent_numerical_oracle": False,
}
COMPONENT_SCOPE = {
    "wall": "Public call, including file reading for dataload; direct input is preloaded.",
    "file_read": "Combined source reading and parsing; OS page-cache state is not forced cold.",
    "arrow_acquisition": "Combined Arrow import, validation, ownership copy and fingerprint when exposed.",
    "native_compile": "Combined resident acquisition/allocation/upload/planning; no fabricated internal split.",
    "artifact_analyze": "Combined numerical execution, significance and report construction.",
    "inclusive_stage_times": "Nested component times are inclusive, non-additive and instrumented.",
    "stage_memory": "Instrumented process RSS/HWM at entry/exit; cumulative, non-additive, not isolated allocation peaks.",
    "copy_counts": "Source-backed ledger, not allocator-measured byte totals.",
}
COPY_LEDGER = {
    "legacy_row_fallback": [
        "Polars rows -> Python row/scalar objects",
        "whole iterable may become a list",
        "flat numeric array -> transport bytes -> Rust-owned vectors",
    ],
    "numpy_direct": [
        "content-keyed ingest can snapshot caller bytes",
        "validation/layout/dtype conversion can allocate",
        "native ownership is not resident zero-copy",
    ],
    "native_arrow": [
        "foreign chunks retained until native import/copy completes",
        "loader-only foreign frames can then be released before resident construction; allocators may retain freed pages",
        "checked row-major resident acquisition requires owned storage",
        "known frame height permits one checked capacity reservation, not per-batch exact reallocations",
        "additional family expansion/device storage is execution-dependent",
    ],
    "resident_execution": [
        "Core converts owned row-major acquisition into a second owned column-major allocation/transpose",
        "GPU upload/device storage is additional to host acquisition, not zero-copy",
        "host originals for GPU significance or target-dependent generated families can remain retained where required",
    ],
    "classification": "Expected stages from current implementation; observations identify which route ran.",
}


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def linux_memory(text: str) -> dict[str, int]:
    """Parse this exec image's RSS/HWM, not an inherited wait4/ru_maxrss value."""
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in {"VmRSS:", "VmHWM:"}:
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                raise ValueError(f"invalid process memory field: {line!r}")
            result[fields[0][:-1] + "_kib"] = int(fields[1])
    if set(result) != {"VmRSS_kib", "VmHWM_kib"}:
        raise ValueError("process status does not contain both VmRSS and VmHWM")
    if result["VmHWM_kib"] < result["VmRSS_kib"]:
        raise ValueError("process peak RSS is below current RSS")
    return result


def memory_now() -> dict[str, Any]:
    if sys.platform != "linux":
        return {"source": "unavailable", "VmRSS_kib": None, "VmHWM_kib": None}
    return {
        "source": "linux_proc_self_status",
        **linux_memory(Path("/proc/self/status").read_text()),
    }


def bits_snapshot(value: Any) -> Any:
    """Preserve IEEE result bits, including signed zero and NaN representation."""
    if is_dataclass(value):
        return bits_snapshot(asdict(value))
    if isinstance(value, float):
        return {"float64_le_hex": struct.pack("<d", value).hex()}
    if isinstance(value, dict):
        return {str(key): bits_snapshot(item) for key, item in value.items()}
    if isinstance(value, array):
        # Decision-path public metadata uses compact typed arrays. Export only
        # after measured wall/RSS are frozen, retaining dtype and every value.
        return {
            "array_typecode": value.typecode,
            "values": [bits_snapshot(item) for item in value],
        }
    if isinstance(value, (tuple, list)):
        return [bits_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"unsupported report snapshot value: {type(value).__name__}")


def report_snapshot(report: Any) -> dict[str, Any]:
    ranked = getattr(report.interactions, "ranked", None)
    rankings = (
        {
            metric: [item.candidate_id for item in ranked(metric_name=metric)]
            for metric in getattr(getattr(report, "config", None), "metric_names", ())
        }
        if callable(ranked)
        else None
    )
    numeric = {
        "feature_names": list(report.feature_names),
        "interactions": [bits_snapshot(item) for item in report.interactions],
        "stability": [bits_snapshot(item) for item in report.stability],
        "permutations": [bits_snapshot(item) for item in report.permutations],
        "metric_rankings": rankings,
        "signal_detected": getattr(report.decision, "signal_detected", None),
    }
    encoded = json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()
    return {
        "numeric": numeric,
        "numeric_sha256": hashlib.sha256(encoded).hexdigest(),
        "warnings": list(report.warnings),
        "decision": bits_snapshot(report.decision),
        "backend": bits_snapshot(report.backend),
        "scope": SNAPSHOT_SCOPE,
    }


def verify_wheel(
    wheel: Path, expected_version: str, package_root: Path
) -> dict[str, Any]:
    """Authenticate installed package members against the supplied immutable wheel."""
    matched = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate archive names")
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("wheel must contain exactly one package METADATA")
        identity = Parser().parsestr(archive.read(metadata[0]).decode())
        if identity["Name"] != "gafime" or identity["Version"] != expected_version:
            raise ValueError(
                "wheel distribution/version does not match the requested Core identity"
            )
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("noncanonical wheel member path")
            if not name.startswith("gafime/") or name.endswith("/"):
                continue
            if Path(name).suffix not in {".py", ".so", ".pyd", ".dylib", ".metallib"}:
                continue
            installed = package_root.joinpath(*relative.parts[1:])
            if not installed.is_file():
                raise ValueError(f"installed package differs from frozen wheel: {name}")
            # Authenticate with bounded reads, rather than making a full native
            # library copy just to measure the product's memory amplification.
            with installed.open("rb") as local, archive.open(name) as frozen:
                while True:
                    local_chunk, frozen_chunk = (
                        local.read(1024 * 1024),
                        frozen.read(1024 * 1024),
                    )
                    if local_chunk != frozen_chunk:
                        raise ValueError(
                            f"installed package differs from frozen wheel: {name}"
                        )
                    if not local_chunk:
                        break
            matched.append(file_identity(installed))
    if not any(item["path"].endswith((".so", ".pyd")) for item in matched):
        raise ValueError("Core wheel has no verified native extension")
    return {
        "wheel": file_identity(wheel),
        "verified_members": matched,
        "expected_version": expected_version,
        "byte_identical_installed_package": True,
    }


def verify_loaded_native(native_path: Path, authenticated: dict[str, Any]) -> None:
    matched_paths = {
        item["path"]
        for item in authenticated["verified_members"]
        if item["path"].endswith((".so", ".pyd"))
    }
    if str(native_path.resolve()) not in matched_paths:
        raise ValueError(
            "loaded native extension is not a byte-verified frozen wheel member"
        )


def check_bounds(rows: int, cols: int, workflow: str) -> None:
    if rows <= 0 or cols <= 0 or rows * cols > MAX_CELLS:
        raise ValueError(
            f"shape must be positive and contain at most {MAX_CELLS} cells"
        )
    if workflow not in WORKFLOWS:
        raise ValueError(f"unsupported workflow: {workflow}")
    if workflow == "default" and (rows > 512 or cols > 8):
        raise ValueError("literal default significance is bounded to 512 x 8")
    if workflow not in {"default", "light"} and (rows > 4096 or cols > 16):
        raise ValueError(
            "full-metric/significance/generated-family cells are bounded to 4096 x 16"
        )


def check_cache_states(workflow: str, states: list[str]) -> None:
    if workflow in {"time-series", "decision-path"} and not set(states) <= {
        "miss",
        "disabled",
    }:
        raise ValueError(
            "generated-family eager paths do not use the continuous resident cache; use miss/disabled"
        )


def generate_fixtures(spec: dict[str, Any]) -> dict[str, Any]:
    # This process is distinct from all measured workers and imports no GAFIME.
    import numpy as np
    import polars as pl

    destination = Path(spec["directory"])
    records = {}
    for rows, cols in spec["shapes"]:
        rng = np.random.default_rng(spec["seed"])
        x = rng.normal(size=(rows, cols)).astype(spec["source_dtype"])
        y = (0.7 * x[:, 0] + rng.normal(0, 0.1, rows)).astype(spec["source_dtype"])
        names = [f"f{index}" for index in range(cols)]
        for target_kind, target in (("original", y), ("changed", y[::-1].copy())):
            frame = pl.DataFrame(
                {
                    **{name: x[:, index] for index, name in enumerate(names)},
                    "target": target,
                }
            )
            for fmt in spec["formats"]:
                stem = destination / f"{rows}x{cols}-{target_kind}-{fmt}"
                path = stem.with_suffix("." + fmt)
                getattr(
                    frame,
                    {
                        "csv": "write_csv",
                        "parquet": "write_parquet",
                        "ipc": "write_ipc",
                    }[fmt],
                )(path)
                # CSV parsing is part of the input semantics. The direct oracle
                # therefore comes from this exact written-and-parsed file.
                parsed = getattr(
                    pl,
                    {"csv": "read_csv", "parquet": "read_parquet", "ipc": "read_ipc"}[
                        fmt
                    ],
                )(path)
                features_path = Path(str(stem) + "-X.npy")
                target_path = Path(str(stem) + "-y.npy")
                np.save(features_path, parsed.select(names).to_numpy(order="c"))
                np.save(target_path, parsed["target"].to_numpy())
                key = f"{rows}x{cols}/{fmt}/{target_kind}"
                records[key] = {
                    "file": file_identity(path),
                    "features": file_identity(features_path),
                    "target": file_identity(target_path),
                    "feature_names": names,
                    "parser_version": pl.__version__,
                    "source_dtype": spec["source_dtype"],
                }
    return {
        "fixtures": records,
        "numpy_version": np.__version__,
        "polars_version": pl.__version__,
    }


def build_config(case: dict[str, Any]) -> Any:
    from gafime import ComputeBudget, EngineConfig

    workflow = case["workflow"]
    backend = "auto" if case["backend"] == "auto-core" else "core"
    if workflow == "default":
        if case["precision"] != "mixed" or backend != "auto":
            raise ValueError(
                "literal default workflow requires mixed precision and auto-core"
            )
        return EngineConfig()
    shared = {"backend": backend, "precision": case["precision"], "random_seed": 7}
    if workflow == "light":
        return EngineConfig(
            **shared,
            metric_names=("pearson",),
            permutation_tests=0,
            num_repeats=1,
            budget=ComputeBudget(max_comb_size=1),
        )
    shared.update(
        metric_names=("pearson", "spearman", "mutual_info", "r2"),
        permutation_tests=3,
        num_repeats=2,
        significance_top_n=8,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=24,
            top_features_for_higher_k=8,
            top_k_features_for_time_series=2,
            max_time_series_candidates=24,
        ),
    )
    if workflow == "time-series":
        shared.update(
            enable_time_series_functions=True,
            time_series_lags=(1, 2),
            time_series_windows=(4,),
        )
    elif workflow == "decision-path":
        shared.update(
            enable_decision_path_functions=True,
            decision_path_max_depth=2,
            decision_path_rounds=1,
            decision_path_max_paths=8,
            decision_path_max_bins=8,
            decision_path_min_leaf=2,
            decision_path_top_k_features=4,
        )
    return EngineConfig(**shared)


def observe_boundary(
    owner: Any, name: str, events: dict[str, Any], label: str | None = None
) -> None:
    """Instrument a coarse call boundary without retaining inputs/outputs."""
    function = getattr(owner, name, None)
    if not callable(function):
        return
    key = label or name

    @functools.wraps(function)
    def observed(*args: Any, **kwargs: Any) -> Any:
        entry_memory = memory_now()
        start = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter_ns() - start
            exit_memory = memory_now()
            event = events.setdefault(
                key,
                {"calls": 0, "inclusive_nanoseconds": 0, "memory_observations": []},
            )
            event["calls"] += 1
            event["inclusive_nanoseconds"] += elapsed
            event["memory_observations"].append(
                {"entry": entry_memory, "exit": exit_memory}
            )

    setattr(owner, name, observed)


def install_observers(case: dict[str, Any]) -> dict[str, Any]:
    """Optional component observation only; no per-scalar instrumentation."""
    import gafime.dataloader as loader
    import gafime.v1_adapter as adapter
    import polars as pl

    events: dict[str, Any] = {}

    def observe(owner: Any, name: str, label: str | None = None) -> None:
        observe_boundary(owner, name, events, label)

    observe(loader, "_read_frame", "read_parse")
    observe(pl.DataFrame, "__getitem__", "polars_named_projection")
    for name in ("select", "cast", "rechunk"):
        observe(pl.DataFrame, name, "polars_" + name)
    for name in (
        "_coerce_row_major_f32_for_cache",
        "_numeric_storage_to_le_bytes",
        "_numeric_array_digest",
        "_numeric_buffer_digest",
        "_diagnostic_from_native_report",
    ):
        observe(adapter, name)
    observe(adapter.NativeCompiledGafime, "analyze", "artifact_analyze_combined")
    native = adapter._load_boundary_for_backend("core")
    for name in (
        "_acquire_arrow_input",
        "_acquire_rows_input",
        "_compile_acquired_continuous",
        "_compile_acquired_time_series",
        "_compile_acquired_decision_path",
        "compile_continuous_buffers",
        "analyze_continuous_arrow",
        "analyze_continuous_buffers",
    ):
        observe(native, name)
    return events


def cache_identities() -> dict[str, Any]:
    """Observe TLS cache metadata without retaining artifacts or native owners."""
    import gafime.v1_adapter as adapter

    current = getattr(adapter, "_current_analyze_cache", None)
    if not callable(current):
        return {"available": False, "entries": []}
    entries = []
    for entry in current().values():
        artifact = entry.artifact
        digest = entry.target_digest
        entries.append(
            {
                "entry_id": id(entry),
                "artifact_id": id(artifact),
                "native_handle_id": id(artifact.native_handle),
                "target_digest": digest.hex()
                if isinstance(digest, bytes)
                else str(digest),
                "closed": bool(getattr(artifact, "_closed", False)),
            }
        )
    return {"available": True, "entries": entries}


def observed_cache_state(
    requested: str, before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    if not before["available"] or not after["available"]:
        state = "observation_unavailable"
    elif not after["entries"]:
        state = (
            "disabled_no_resident_state"
            if requested == "disabled"
            else "no_resident_state"
        )
    else:

        def key(entry: dict[str, Any]) -> tuple[int, int, int]:
            return entry["entry_id"], entry["artifact_id"], entry["native_handle_id"]

        previous = {
            key(entry): entry for entry in before["entries"] if not entry["closed"]
        }
        reused = [
            entry
            for entry in after["entries"]
            if key(entry) in previous and not entry["closed"]
        ]
        if reused:
            state = (
                "resident_target_update"
                if any(
                    entry["target_digest"] != previous[key(entry)]["target_digest"]
                    for entry in reused
                )
                else "resident_identity_reuse"
            )
        else:
            state = "new_resident_state"
    return {
        "requested": requested,
        "observed": state,
        "before": before,
        "after": after,
        "scope": "same-process TLS entry/artifact/native-object identity and target digest; no extra retained owners",
    }


def worker(spec: dict[str, Any]) -> dict[str, Any]:
    # CPU-only control is declared explicitly, not a silent fallback. Both
    # versions get identical process-local overrides for ranked auto selection.
    for name in ("GAFIME_CUDA_V1_LIB", "GAFIME_ROCM_V1_LIB", "GAFIME_METAL_V1_LIB"):
        os.environ[name] = str(Path(spec["directory"]) / "unavailable-payload")
    os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = (
        "0" if spec["case"]["cache_state"] == "disabled" else "2"
    )
    import importlib.metadata as metadata
    import numpy as np
    import gafime
    import gafime.gafime_py as native

    expected = spec["variant"]["expected_version"]
    if gafime.__version__ != expected or metadata.version("gafime") != expected:
        raise ValueError(
            "installed runtime/distribution version differs from requested version"
        )
    package = Path(gafime.__file__).resolve().parent
    authenticated = verify_wheel(Path(spec["variant"]["wheel"]), expected, package)
    verify_loaded_native(Path(native.__file__), authenticated)
    imported_memory = memory_now()
    config = build_config(spec["case"])
    names = spec["fixture"]["feature_names"]
    events = install_observers(spec["case"]) if spec["instrument"] else {}
    direct_original = direct_changed = None
    if spec["case"]["route"] == "direct":
        direct_original = (
            np.load(spec["fixture"]["features"]["path"]),
            np.load(spec["fixture"]["target"]["path"]),
        )
        direct_changed = (
            direct_original[0],
            np.load(spec["changed_fixture"]["target"]["path"]),
        )

    def call(changed: bool = False) -> Any:
        selected = spec["changed_fixture"] if changed else spec["fixture"]
        if spec["case"]["route"] == "dataload":
            return gafime.dataload(selected["file"]["path"], "target", config=config)
        x, y = direct_changed if changed else direct_original
        return gafime.GafimeEngine(config).analyze(x, y, names)

    if spec["case"]["cache_state"] in {"hit", "target-change"}:
        warmup = call()
        if warmup.backend.selected_backend != "core":
            raise ValueError("CPU control resolved to a non-Core backend")
        del warmup
    events.clear()
    cache_before = cache_identities()
    before = memory_now()
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    report = call(spec["case"]["cache_state"] == "target-change")
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.process_time_ns() - cpu_start
    # Freeze memory/time evidence before a report snapshot expands lazy rows
    # into Python objects. That later export is not the measured public call.
    after = memory_now()
    cache_observation = observed_cache_state(
        spec["case"]["cache_state"], cache_before, cache_identities()
    )
    if report.backend.selected_backend != "core":
        raise ValueError("CPU control resolved to a non-Core backend")
    export_start = time.perf_counter_ns()
    snapshot = report_snapshot(report)
    export_wall_ns = time.perf_counter_ns() - export_start
    after_export = memory_now()
    return {
        "schema": SCHEMA + ".worker",
        "case": spec["case"],
        "variant": spec["variant"],
        "identity": authenticated,
        "native": file_identity(Path(native.__file__)),
        "python": {"executable": sys.executable, "version": sys.version},
        "dependencies": {name: metadata.version(name) for name in ("numpy", "polars")},
        "harness": file_identity(Path(__file__)),
        "fixture": spec["fixture"],
        "changed_fixture": spec["changed_fixture"],
        "config": asdict(config),
        "timing": {
            "wall_nanoseconds": wall_ns,
            "process_cpu_nanoseconds": cpu_ns,
            "cpu_core_equivalents": cpu_ns / wall_ns,
        },
        "memory": {
            "after_import": imported_memory,
            "before_call": before,
            "after_call_before_snapshot": after,
            "peak_scope": "fresh exec, imports and optional cache warmup included",
            "inherited_resource_ru_maxrss_used": False,
        },
        "instrumented": spec["instrument"],
        "cache_observation": cache_observation,
        "components_inclusive_nonadditive": dict(events),
        "execution_component_observation": (
            "observed_inclusive_python_artifact_analyze"
            if "artifact_analyze_combined" in events
            else "not_separated_direct_native_handle_or_uninstrumented_public_wall"
        ),
        "component_scope": COMPONENT_SCOPE,
        "copy_ledger": COPY_LEDGER,
        "snapshot": snapshot,
        "snapshot_generated_after_measurement": True,
        "post_call_report_export": {
            "wall_nanoseconds": export_wall_ns,
            "memory_after_export": after_export,
            "scope": "separate full lazy-row/ranking snapshot export; excluded from public call timing and its RSS",
        },
        "environment": {
            "platform": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "cpu_count": os.cpu_count(),
            "RAYON_NUM_THREADS": os.getenv("RAYON_NUM_THREADS"),
            "POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS"),
            "auto_gpu_payloads_explicitly_disabled": True,
            "load_average": list(os.getloadavg())
            if hasattr(os, "getloadavg")
            else None,
        },
        "source_sha_binding": "supplied source SHA is an attestation; frozen provenance is independently required",
    }


def run_child(
    python: str, mode: str, spec_path: Path, log_prefix: Path, timeout: float
) -> dict[str, Any]:
    command = [python, "-I", str(Path(__file__).resolve()), mode, str(spec_path)]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        stdout = (
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        )
        stderr = (
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        )
        code = None
    log_prefix.with_suffix(".stdout.log").write_text(stdout)
    log_prefix.with_suffix(".stderr.log").write_text(stderr)
    if code != 0:
        return {
            "status": "failed",
            "returncode": code,
            "reason": "timeout" if code is None else "worker_failed",
            "command": command,
        }
    try:
        result = json.loads(stdout)
    except (ValueError, TypeError):
        return {
            "status": "failed",
            "returncode": code,
            "reason": "invalid_worker_json",
            "command": command,
        }
    if not isinstance(result, dict):
        return {
            "status": "failed",
            "returncode": code,
            "reason": "invalid_worker_structure",
            "command": command,
        }
    return {"status": "passed", "result": result}


def summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["status"] != "passed" or record["result"]["instrumented"]:
            continue
        result = record["result"]
        key = json.dumps(
            {
                "variant": result["variant"]["label"],
                **result["case"],
                "observed_cache_state": result.get("cache_observation", {}).get(
                    "observed"
                ),
            },
            sort_keys=True,
        )
        groups.setdefault(key, []).append(result)
    output = []
    for key, group in groups.items():
        wall = [item["timing"]["wall_nanoseconds"] for item in group]
        peaks = [
            item["memory"]["after_call_before_snapshot"]["VmHWM_kib"] for item in group
        ]
        output.append(
            {
                "cell": json.loads(key),
                "samples": len(group),
                "wall_nanoseconds": wall,
                "median_wall_nanoseconds": statistics.median(wall),
                "peak_rss_kib": peaks,
                "median_peak_rss_kib": statistics.median(peaks)
                if all(value is not None for value in peaks)
                else None,
            }
        )
    return output


def parity_key(
    variant: str, case: dict[str, Any], repeat: int, instrumented: bool
) -> str:
    return json.dumps(
        {
            "variant": variant,
            "case": {key: value for key, value in case.items() if key != "route"},
            "repeat": repeat,
            "instrumented": instrumented,
        },
        sort_keys=True,
    )


def public_numeric_parity(
    records: list[dict[str, Any]],
    jobs: list[tuple[dict[str, Any], dict[str, Any], int, bool]],
) -> dict[str, Any]:
    """Fail closed on missing/unequal direct-vs-file cells within each variant.

    RC2-vs-candidate equality is deliberately not asserted: earlier correctness
    fixes can change released RC2 semantics. This gate is also not an independent
    numerical oracle; those installed-package correctness tests remain required.
    """
    expected = {
        parity_key(variant["label"], case, repeat, instrumented)
        for variant, case, repeat, instrumented in jobs
    }
    observed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        if record["status"] != "passed":
            continue
        result = record["result"]
        key = parity_key(
            record["variant"], record["case"], record["repeat"], result["instrumented"]
        )
        observed.setdefault(key, {}).setdefault(record["case"]["route"], []).append(
            result["snapshot"]
        )
    comparisons = []
    for key in sorted(expected):
        routes = observed.get(key, {})
        comparison: dict[str, Any] = {"cell": json.loads(key), "status": "failed"}
        if set(routes) != {"direct", "dataload"} or any(
            len(items) != 1 for items in routes.values()
        ):
            comparison["reason"] = "missing_or_duplicate_same_cell_samples"
        else:
            direct, dataload = routes["direct"][0], routes["dataload"][0]
            comparison["direct_sha256"] = direct["numeric_sha256"]
            comparison["dataload_sha256"] = dataload["numeric_sha256"]
            if direct["numeric"] != dataload["numeric"]:
                comparison["reason"] = "same_variant_numeric_snapshot_mismatch"
            elif direct["numeric_sha256"] != dataload["numeric_sha256"]:
                comparison["reason"] = "contradictory_numeric_snapshot_digest"
            else:
                comparison["status"] = "passed"
        comparisons.append(comparison)
    return {
        "status": "passed"
        if comparisons and all(item["status"] == "passed" for item in comparisons)
        else "failed",
        "scope": "same variant/config/cache state/fixture/repeat; not cross-version equality or an independent oracle",
        "exclusions": SNAPSHOT_SCOPE["excluded"],
        "comparisons": comparisons,
        "numeric_mismatch_count": sum(
            item.get("reason") == "same_variant_numeric_snapshot_mismatch"
            for item in comparisons
        ),
        "missing_or_duplicate_count": sum(
            item.get("reason") == "missing_or_duplicate_same_cell_samples"
            for item in comparisons
        ),
    }


def assignments(values: list[str] | None) -> dict[str, str]:
    result = {}
    for value in values or []:
        label, separator, item = value.partition("=")
        if not separator or not label or not item or label in result:
            raise ValueError("assignments require unique nonempty LABEL=VALUE entries")
        result[label] = item
    return result


def interpreter_path(value: str) -> str:
    """Keep venv entry-point symlinks: resolving them drops venv identity."""
    selected = value if os.path.isabs(value) or os.sep in value else shutil.which(value)
    if not selected:
        raise ValueError(f"Python interpreter not found: {value}")
    path = Path(selected).absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python interpreter is not executable: {value}")
    return str(path)


def choices(value: str, supported: set[str]) -> list[str]:
    selected = value.split(",")
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected) <= supported
    ):
        raise ValueError(
            f"expected unique comma-separated choices from {sorted(supported)}"
        )
    return selected


def driver(args: argparse.Namespace) -> int:
    interpreters, wheels, versions, sources = (
        assignments(getattr(args, name))
        for name in ("python", "wheel", "expected_version", "source_sha")
    )
    if (
        not interpreters
        or set(interpreters) != set(wheels)
        or set(interpreters) != set(versions)
        or not set(sources) <= set(interpreters)
    ):
        raise ValueError(
            "each interpreter requires one matching wheel and expected version"
        )
    for source in sources.values():
        if len(source) != 40 or any(char not in "0123456789abcdef" for char in source):
            raise ValueError("source SHA must be a full lowercase Git commit SHA")
    actual_harness = file_identity(Path(__file__))["sha256"]
    if args.harness_sha and args.harness_sha != actual_harness:
        raise ValueError("harness source digest does not match expected digest")
    interpreters = {
        label: interpreter_path(value) for label, value in interpreters.items()
    }
    shapes = []
    for text in args.shape or ["256x8"]:
        fields = text.split("x")
        if len(fields) != 2:
            raise ValueError("shape syntax is ROWSxCOLS")
        shapes.append(tuple(int(field) for field in fields))
    if len(shapes) != len(set(shapes)):
        raise ValueError("duplicate shapes")
    formats = choices(args.format, FORMATS)
    precisions = choices(args.precision, PRECISIONS)
    workflows = choices(args.workflow, WORKFLOWS)
    caches = choices(args.cache_state, CACHE_STATES)
    if args.repeats <= 0 or not all(
        math.isfinite(value) and value > 0
        for value in (args.timeout, args.max_time_seconds)
    ):
        raise ValueError("repeat count and time limits must be positive")
    for rows, cols in shapes:
        for workflow in workflows:
            check_bounds(rows, cols, workflow)
            check_cache_states(workflow, caches)
            if workflow == "default" and (
                precisions != ["mixed"] or args.backend != "auto-core"
            ):
                raise ValueError(
                    "literal default requires only mixed precision and auto-core"
                )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(
            "output directory must be empty; prior evidence will not be overwritten"
        )
    fixtures_dir = output / "fixtures"
    fixtures_dir.mkdir()
    generator_spec = {
        "directory": str(fixtures_dir),
        "shapes": shapes,
        "formats": formats,
        "source_dtype": args.source_dtype,
        "seed": args.seed,
    }
    fixture_spec_path = output / "fixture-spec.json"
    fixture_spec_path.write_text(json.dumps(generator_spec, indent=2))
    generated = run_child(
        next(iter(interpreters.values())),
        "--generate",
        fixture_spec_path,
        output / "fixture-generator",
        args.timeout,
    )
    if generated["status"] != "passed":
        (output / "report.json").write_text(
            json.dumps(
                {"schema": SCHEMA, "status": "failed", "fixture_generation": generated},
                indent=2,
            )
        )
        return 1
    fixture_records = generated["result"]["fixtures"]
    variants = [
        {
            "label": label,
            "python": python,
            "wheel": str(Path(wheels[label]).resolve()),
            "expected_version": versions[label],
            "source_sha": sources.get(label),
        }
        for label, python in interpreters.items()
    ]
    matrix = []
    for rows, cols in shapes:
        for fmt in formats:
            for precision in precisions:
                for workflow in workflows:
                    for cache in caches:
                        for route in ("direct", "dataload"):
                            matrix.append(
                                {
                                    "rows": rows,
                                    "cols": cols,
                                    "format": fmt,
                                    "precision": precision,
                                    "workflow": workflow,
                                    "cache_state": cache,
                                    "route": route,
                                    "backend": args.backend,
                                }
                            )
    jobs = [
        (variant, case, repeat, False)
        for repeat in range(args.repeats)
        for case in matrix
        for variant in variants
    ]
    random.Random(args.seed).shuffle(jobs)
    if args.instrument:
        jobs.extend((variant, case, 0, True) for case in matrix for variant in variants)
    records = []
    start = time.monotonic()
    for index, (variant, case, repeat, instrument) in enumerate(jobs):
        if time.monotonic() - start >= args.max_time_seconds:
            records.append(
                {
                    "status": "failed",
                    "reason": "campaign_budget_exhausted",
                    "jobs_remaining": len(jobs) - index,
                }
            )
            break
        stem = output / f"sample-{index:05d}"
        fixture_key = f"{case['rows']}x{case['cols']}/{case['format']}"
        spec = {
            "variant": variant,
            "case": case,
            "repeat": repeat,
            "instrument": instrument,
            "directory": str(output),
            "fixture": fixture_records[fixture_key + "/original"],
            "changed_fixture": fixture_records[fixture_key + "/changed"],
        }
        spec_path = stem.with_suffix(".spec.json")
        spec_path.write_text(json.dumps(spec, indent=2))
        record = run_child(
            variant["python"],
            "--worker",
            spec_path,
            stem,
            min(
                args.timeout,
                max(0.1, args.max_time_seconds - (time.monotonic() - start)),
            ),
        )
        record.update(sample=index, repeat=repeat, case=case, variant=variant["label"])
        stem.with_suffix(".result.json").write_text(json.dumps(record, indent=2))
        records.append(record)
        print(
            json.dumps(
                {
                    "sample": index,
                    "variant": variant["label"],
                    "case": case,
                    "status": record["status"],
                }
            ),
            flush=True,
        )
    numeric_parity = public_numeric_parity(records, jobs)
    collection_complete = len(records) == len(jobs) and all(
        record["status"] == "passed" for record in records
    )
    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "collected"
        if collection_complete and numeric_parity["status"] == "passed"
        else "failed",
        "collector_status": "passed" if collection_complete else "failed",
        "public_numeric_parity": numeric_parity,
        "performance_budget_status": "not_assessed",
        "release_ready": False,
        "harness": file_identity(Path(__file__)),
        "variants": variants,
        "fixture_generation": generated,
        "requested_jobs": len(jobs),
        "records": records,
        "summaries": summaries(records),
        "snapshot_scope": SNAPSHOT_SCOPE,
        "component_scope": COMPONENT_SCOPE,
        "claim_boundary": "Raw public workflow evidence; no kernel/device-time or universal-performance claim.",
        "comparison_policy": "Same-variant direct-vs-dataload snapshots must match; cross-version differences are retained, not automatically rejected. Independent numerical gates and evidence-derived cost budgets remain required.",
        "elapsed_seconds": time.monotonic() - start,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return 0 if report["status"] == "collected" else 1


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] in {"--generate", "--worker"}:
        try:
            spec = json.loads(Path(sys.argv[2]).read_text())
            result = (
                generate_fixtures(spec) if sys.argv[1] == "--generate" else worker(spec)
            )
            print(json.dumps(result, allow_nan=False))
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("python", "wheel", "expected-version", "source-sha"):
        parser.add_argument("--" + name, action="append", metavar="LABEL=VALUE")
    parser.add_argument("--harness-sha")
    parser.add_argument("--shape", action="append", metavar="ROWSxCOLS")
    parser.add_argument("--format", default="parquet")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--workflow", default="light")
    parser.add_argument("--cache-state", default="miss")
    parser.add_argument("--backend", choices=("core", "auto-core"), default="auto-core")
    parser.add_argument(
        "--source-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-time-seconds", type=float, default=600)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        return driver(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
