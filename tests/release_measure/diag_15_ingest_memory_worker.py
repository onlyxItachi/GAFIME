#!/usr/bin/env python3
"""Diagnostic-only process snapshots around unchanged installed-wheel calls.

No sample from this instrumented worker replaces the uninstrumented perf_14
gate. Native compile remains combined: input retirement at its successful exit
is source-inferred, not an independently observed native allocation event.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import functools
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

TREATMENTS = ("normal", "glibc-trim", "decay-zero", "thp-never", "thp-data", "thp-all")
ALLOCATOR_ENV = (
    "MALLOC_CONF",
    "_RJEM_MALLOC_CONF",
    "JEMALLOC_CONF",
    "MALLOC_ARENA_MAX",
    "MALLOC_TRIM_THRESHOLD_",
    "MALLOC_MMAP_THRESHOLD_",
    "LD_PRELOAD",
    "POLARS_MAX_THREADS",
    "POLARS_THP",
    "RAYON_NUM_THREADS",
)


class Mallinfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


def allocator_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in ALLOCATOR_ENV}


def proc_values(path: str) -> dict[str, Any]:
    try:
        values = {}
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(":")
            fields = value.split()
            if fields and fields[0].isdigit():
                values[key + ("_kib" if fields[1:] == ["kB"] else "")] = int(fields[0])
        return {"available": True, "fields": values}
    except OSError as error:
        return {"available": False, "reason": str(error)}


class Markers:
    def __init__(self, path: Path, treatment: str) -> None:
        self.fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        self.treatment = treatment
        self.sequence = 0
        self.trim_calls = 0
        self.libc = ctypes.CDLL(None)
        self.mallinfo = getattr(self.libc, "mallinfo2", None)
        if self.mallinfo is not None:
            self.mallinfo.argtypes = []
            self.mallinfo.restype = Mallinfo2
        self.trim = getattr(self.libc, "malloc_trim", None)
        if self.trim is not None:
            self.trim.argtypes = [ctypes.c_size_t]
            self.trim.restype = ctypes.c_int

    def emit(self, phase: str, edge: str, **detail: Any) -> None:
        start = time.monotonic_ns()
        info = self.mallinfo() if self.mallinfo is not None else None
        payload = {
            "schema": "gafime.ingest-memory-diagnostic.marker.v1",
            "pid": os.getpid(),
            "sequence": self.sequence,
            "monotonic_ns": start,
            "phase": phase,
            "edge": edge,
            "diagnostic": True,
            "instrumented": True,
            "status": proc_values("/proc/self/status"),
            "smaps_rollup": proc_values("/proc/self/smaps_rollup"),
            "memory_scope": "process aggregate; anonymous/file RSS is not allocator ownership",
            "mallinfo2": {
                "available": info is not None,
                "fields": {name: getattr(info, name) for name, _ in Mallinfo2._fields_}
                if info is not None
                else None,
                "count_fields": ["ordblks", "smblks", "hblks"],
                "scope": "glibc only; excludes private jemalloc and is not Vec capacity",
            },
            **detail,
        }
        payload["snapshot_end_monotonic_ns"] = time.monotonic_ns()
        encoded = memoryview(
            (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        )
        while encoded:
            encoded = encoded[os.write(self.fd, encoded) :]
        self.sequence += 1

    def hook(self, owner: Any, name: str, phase: str) -> None:
        function = getattr(owner, name, None)
        if not callable(function):
            self.emit(phase, "unavailable", name=name)
            return

        @functools.wraps(function)
        def observed(*args: Any, **kwargs: Any) -> Any:
            self.emit(phase, "entry")
            if (
                phase == "prepared_after_foreign_release"
                and self.treatment == "glibc-trim"
            ):
                if self.trim is None:
                    raise RuntimeError(
                        "glibc-trim requested but malloc_trim is unavailable"
                    )
                result = int(self.trim(0))
                self.trim_calls += 1
                self.emit(phase, "after_glibc_trim", malloc_trim_result=result)
            failed = False
            try:
                return function(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                self.emit(
                    phase,
                    "exit",
                    failed=failed,
                    input_retirement=(
                        "source_inferred_after_successful_compile_not_isolated_native_event"
                        if phase == "native_compile_combined" and not failed
                        else "not_observed"
                    ),
                )

        setattr(owner, name, observed)

    def close(self) -> None:
        os.close(self.fd)


def install_hooks(markers: Markers) -> dict[str, Any]:
    import gafime.dataloader as loader
    import gafime.v1_adapter as adapter
    import polars as pl

    native = adapter._load_boundary_for_backend("core")
    for owner, name, phase in (
        (loader, "_read_frame", "read_parse"),
        (pl.DataFrame, "__getitem__", "literal_projection"),
        (native, "_acquire_arrow_input", "native_acquisition_combined"),
        (native, "_acquire_rows_input", "native_scalar_acquisition_combined"),
        (
            adapter,
            "_analyze_prepared_frame_acquisition",
            "prepared_after_foreign_release",
        ),
        (native, "_compile_acquired_continuous", "native_compile_combined"),
        (adapter.NativeCompiledGafime, "analyze", "artifact_analyze_combined"),
        (adapter, "_diagnostic_from_native_report", "report_construction"),
    ):
        markers.hook(owner, name, phase)
    return {"polars": pl, "loader": loader}


def environment(harness: Any, pl: Any, before: dict[str, Any]) -> dict[str, Any]:
    native_paths = {
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name.startswith("polars")
        and getattr(module, "__file__", None)
        and Path(module.__file__).suffix in {".so", ".pyd"}
    }
    thp = Path("/sys/kernel/mm/transparent_hugepage")
    return {
        "allocator_environment_pre_import": before,
        "allocator_environment_post_import": allocator_environment(),
        "polars_thread_pool_size": pl.thread_pool_size(),
        "polars_init": harness.file_identity(Path(pl.__file__)),
        "polars_native": [harness.file_identity(path) for path in sorted(native_paths)],
        "polars_native_identity_available": bool(native_paths),
        "thp_system_configuration": {
            name: (thp / name).read_text().strip()
            for name in ("enabled", "defrag", "shmem_enabled", "hpage_pmd_size")
            if (thp / name).is_file()
        },
        "cpu_info": Path("/proc/cpuinfo").read_text(),
        "system_memory": Path("/proc/meminfo").read_text(),
        "kernel": platform.uname()._asdict(),
        "observer_setup": "Polars imported/pool inspected before calls; diagnostic overhead included",
    }


def parser_only(harness: Any, spec: dict[str, Any]) -> dict[str, Any]:
    for name in ("GAFIME_CUDA_V1_LIB", "GAFIME_ROCM_V1_LIB", "GAFIME_METAL_V1_LIB"):
        os.environ[name] = str(Path(spec["directory"]) / "unavailable-payload")
    os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = (
        "0" if spec["case"]["cache_state"] == "disabled" else "2"
    )
    import importlib.metadata as metadata
    import numpy  # noqa: F401 -- match the original worker's imports
    import gafime
    import gafime.gafime_py as native
    import gafime.dataloader as loader

    expected = spec["variant"]["expected_version"]
    if gafime.__version__ != expected or metadata.version("gafime") != expected:
        raise ValueError(
            "installed runtime/distribution version differs from requested version"
        )
    identity = harness.verify_wheel(
        Path(spec["variant"]["wheel"]), expected, Path(gafime.__file__).resolve().parent
    )
    harness.verify_loaded_native(Path(native.__file__), identity)
    imported_memory = harness.memory_now()
    config = harness.build_config(spec["case"])
    before = harness.memory_now()
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    frame = loader._read_frame(Path(spec["fixture"]["file"]["path"]))
    names = loader._resolve_feature_columns(frame.columns, "target", None)
    features, target = frame[names], frame[["target"]]
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.process_time_ns() - cpu_start
    # Freeze the parser-only interval before validation, identities, or the
    # deferred environment probes. No numerical acquisition occurs here.
    after = harness.memory_now()
    if (frame.height, features.width) != (spec["case"]["rows"], spec["case"]["cols"]):
        raise ValueError("parser-only fixture shape differs from the requested case")
    if names != spec["fixture"]["feature_names"]:
        raise ValueError("parser-only projected names differ from the fixture identity")
    return {
        "identity": identity,
        "native": harness.file_identity(Path(native.__file__)),
        "config": asdict(config),
        "fixture": spec["fixture"],
        "parsed_rows": frame.height,
        "projected_feature_columns": features.width,
        "projected_target_columns": target.width,
        "numerical_acquisition": False,
        "timing": {
            "wall_nanoseconds": wall_ns,
            "process_cpu_nanoseconds": cpu_ns,
            "cpu_core_equivalents": cpu_ns / wall_ns,
            "scope": "file read, parse, and literal projection only; not public dataload",
        },
        "memory": {
            "after_import": imported_memory,
            "before_call": before,
            "after_call_before_snapshot": after,
            "peak_scope": "fresh exec, installed imports and authenticated wheel included",
            "inherited_resource_ru_maxrss_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--treatment", choices=TREATMENTS, default="normal")
    parser.add_argument("--parser-only", action="store_true")
    parser.add_argument("--uninstrumented-parser", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this process-memory diagnostic is Linux-only")
    if args.uninstrumented_parser and not args.parser_only:
        parser.error("uninstrumented-parser requires parser-only")
    harness_path = (
        args.source_root.resolve() / "tests/release_measure/perf_14_public_ingest.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "_perf14_memory_diagnostic", harness_path
    )
    if module_spec is None or module_spec.loader is None:
        parser.error("source-root does not provide the existing perf_14 harness")
    harness = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(harness)
    spec = json.loads(args.spec.read_text())
    case = spec["case"]
    for field in ("rows", "cols"):
        if type(case[field]) is not int or case[field] <= 0:
            parser.error(f"case {field} must be a positive integer")
    harness.check_bounds(case["rows"], case["cols"], case["workflow"])
    for field, allowed in (
        ("format", harness.FORMATS),
        ("precision", harness.PRECISIONS),
        ("cache_state", harness.CACHE_STATES),
        ("route", {"direct", "dataload"}),
        ("backend", {"core", "auto-core"}),
    ):
        if case[field] not in allowed:
            parser.error(f"unsupported case {field}: {case[field]!r}")
    if args.parser_only and case["route"] != "dataload":
        parser.error("parser-only requires the dataload route")
    pre_import = allocator_environment()
    markers = None
    observed_environment = {}
    if not args.uninstrumented_parser:
        markers = Markers(args.markers, args.treatment)
        original_memory = harness.memory_now

        def memory_now() -> dict[str, Any]:
            if not observed_environment:
                markers.emit(
                    "after_installed_imports_before_observer_setup", "snapshot"
                )
                hooks = install_hooks(markers)
                observed_environment.update(
                    environment(harness, hooks["polars"], pre_import)
                )
                markers.emit("after_imports", "snapshot", observer_setup=True)
            return original_memory()

        harness.memory_now = memory_now
    spec["instrument"] = (
        False  # Custom diagnostic hooks replace—not stack—the perf_14 probes.
    )
    try:
        result = (
            parser_only(harness, spec) if args.parser_only else harness.worker(spec)
        )
        if args.uninstrumented_parser:
            import polars as pl

            # This control uses the unchanged harness memory reader around the
            # timed parse. Do not initialize observers or hash/probe until now.
            observed_environment.update(environment(harness, pl, pre_import))
            observed_environment["observer_setup"] = (
                "no hooks or diagnostic probes during parse; environment collected after measurement"
            )
            markers = Markers(args.markers, args.treatment)
        markers.emit("worker_complete", "snapshot")
        result.update(
            schema="gafime.ingest-memory-diagnostic.worker.v1",
            diagnostic=True,
            instrumented=not args.uninstrumented_parser,
            treatment=args.treatment,
            parser_only=args.parser_only,
            uninstrumented_parser=args.uninstrumented_parser,
            diagnostic_environment=observed_environment,
            diagnostic_worker=harness.file_identity(Path(__file__)),
            source_harness=harness.file_identity(harness_path),
            markers=str(args.markers.resolve()),
            malloc_trim_calls=markers.trim_calls,
            execution_component_observation=(
                "uninstrumented parser-only interval; no numerical acquisition"
                if args.uninstrumented_parser
                else "coarse markers only; no isolated native timing"
            ),
            scope="coarse call snapshots; cumulative HWM, no exact Vec capacity or copy counters",
        )
        print(json.dumps(result, separators=(",", ":")))
    finally:
        if markers is not None:
            markers.close()


if __name__ == "__main__":
    main()
