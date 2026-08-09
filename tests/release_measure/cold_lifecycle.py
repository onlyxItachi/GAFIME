#!/usr/bin/env python3
"""Measure the observable cold lifecycle of a GAFIME ABI 1.1 payload.

This benchmark is intentionally separate from the public precision benchmark.
Each profile sample runs in a new process and calls the canonical generic ABI
directly.  That gives the report honest boundaries for Python import, payload
discovery, dynamic loading, route negotiation, allocation, upload, execution,
and teardown.  The ABI has no separate planning, result-materialisation, or
module-registration hooks; those fields are therefore explicitly marked as
``observed_combined`` or ``not_observable`` instead of being inferred from a
public timer.

Example::

    python tests/release_measure/cold_lifecycle.py \
      --backend cuda --payload /path/to/libgafime_cuda.so \
      --profile fp32,mixed,fp64 --repetitions 30 \
      --output cold-cuda.json

The driver imports only the Python standard library.  The worker imports the
installed GAFIME package, discovers the requested payload, loads its exact
bytes, and performs a four-row canonical ABI execution.  ``--wheel`` and
``--source-root`` bind artifact and source provenance to every sample.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "gafime.cold-lifecycle.v1"
WORKER_SCHEMA = "gafime.cold-lifecycle.worker.v1"
ABI_VERSION = (1 << 16) | 1
BACKEND_KINDS = {"cuda": 2, "rocm": 3, "metal": 4}
PROFILE_IDS = {"fp32": 1, "mixed": 2, "fp64": 3}
PROFILE_ORDER = ("fp32", "mixed", "fp64")
STATUS_OK = 0
STATUS_UNSUPPORTED_BACKEND = -2
STATUS_DEVICE_ERROR = -4
DTYPE_F32 = 1
DTYPE_F64 = 2
METRIC_PEARSON = 1
FAMILY_CONTINUOUS = 1
MATRIX_ROW_MAJOR = 1
BUFFER_HOST = 1
BUFFER_CONTIGUOUS = 2

REQUIRED_PHASES = (
    "python_import",
    "payload_discovery",
    "dynamic_library_load",
    "symbol_resolution",
    "runtime_context_initialization",
    "code_object_or_module_registration",
    "first_capability_query",
    "first_allocation",
    "first_upload",
    "target_update",
    "planning",
    "execution_memory_forecast",
    "first_execution",
    "first_result_materialization",
    "explicit_cleanup",
    "process_exit_cleanup",
)


class _Route(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("route_id", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("storage_dtype", ctypes.c_uint32),
        ("pointwise_dtype", ctypes.c_uint32),
        ("reduction_dtype", ctypes.c_uint32),
        ("result_dtype", ctypes.c_uint32),
        ("overflow_policy", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _ConstView(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("element_count", ctypes.c_uint64),
        ("byte_length", ctypes.c_uint64),
        ("byte_stride", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _MutableView(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("element_capacity", ctypes.c_uint64),
        ("byte_length", ctypes.c_uint64),
        ("byte_stride", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _MatrixDesc(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("route", _Route),
        ("layout", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("rows", ctypes.c_uint64),
        ("cols", ctypes.c_uint32),
        ("row_stride", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _SliceU32(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint32)), ("len", ctypes.c_uint64)]


class _SliceU64(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint64)), ("len", ctypes.c_uint64)]


class _ArityChunk(ctypes.Structure):
    _fields_ = [
        ("arity", ctypes.c_uint32),
        ("family", ctypes.c_uint32),
        ("metric_mask", ctypes.c_uint32),
        ("shape_hint_index", ctypes.c_uint32),
        ("combo_row_offset", ctypes.c_uint64),
        ("combo_count", ctypes.c_uint64),
        ("local_chunk_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("descriptor_offset", ctypes.c_uint64),
        ("descriptor_count", ctypes.c_uint64),
    ]


class _RankSpec(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_uint32),
        ("primary_metric", ctypes.c_uint32),
        ("descending", ctypes.c_uint32),
        ("include_ties", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _PermutationSchedule(ctypes.Structure):
    _fields_ = [
        ("permutation_count", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved32", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("target_offsets", _SliceU64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _LaunchProtocol(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("backend_kind", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("max_arity", ctypes.c_uint32),
        ("n_samples", ctypes.c_uint64),
        ("n_features", ctypes.c_uint32),
        ("family_count", ctypes.c_uint32),
        ("combo_indices", _SliceU32),
        ("metric_ids", _SliceU32),
        ("chunks", ctypes.POINTER(_ArityChunk)),
        ("chunk_count", ctypes.c_uint32),
        ("reserved32_a", ctypes.c_uint32),
        ("shape_hints", ctypes.c_void_p),
        ("shape_hint_count", ctypes.c_uint32),
        ("reserved32_b", ctypes.c_uint32),
        ("rank", _RankSpec),
        ("permutations", _PermutationSchedule),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _NumericLaunch(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("route", _Route),
        ("base", ctypes.POINTER(_LaunchProtocol)),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _NumericResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("max_arity", ctypes.c_uint32),
        ("metric_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved32", ctypes.c_uint32),
        ("capacity", ctypes.c_uint64),
        ("row_count", ctypes.c_uint64),
        ("combo_indices", ctypes.POINTER(ctypes.c_uint32)),
        ("metric_values", _MutableView),
        ("ranks", ctypes.POINTER(ctypes.c_uint32)),
        ("families", ctypes.POINTER(ctypes.c_uint32)),
        ("candidate_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("row_flags", ctypes.POINTER(ctypes.c_uint32)),
        ("reserved", ctypes.c_uint64 * 8),
    ]


def _abi_layout_self_check() -> None:
    expected = {
        _Route: (104, 8),
        _ConstView: (80, 8),
        _MutableView: (80, 8),
        _MatrixDesc: (208, 8),
        _NumericLaunch: (184, 8),
        _NumericResult: (224, 8),
    }
    for structure, (size, alignment) in expected.items():
        if ctypes.sizeof(structure) != size or ctypes.alignment(structure) != alignment:
            raise RuntimeError(
                f"ABI ctypes layout drift for {structure.__name__}: "
                f"size={ctypes.sizeof(structure)} align={ctypes.alignment(structure)}"
            )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _command(command: Sequence[str], timeout: int = 20) -> dict[str, object]:
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "command": list(command), "detail": str(exc)}
    return {
        "status": "pass" if result.returncode == 0 else "error",
        "command": list(command),
        "returncode": result.returncode,
        "output": (result.stdout + result.stderr).strip()[:16_384],
    }


def _source_provenance(source_root: str | None) -> dict[str, object]:
    if not source_root:
        return {"status": "not_supplied"}
    root = Path(source_root).expanduser().resolve(strict=True)
    commit = _command(("git", "-C", str(root), "rev-parse", "HEAD"))
    state = _command(
        ("git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all")
    )
    return {
        "root": str(root),
        "commit": str(commit.get("output", "")).splitlines()[0]
        if commit.get("status") == "pass"
        else None,
        "git_status": {
            "status": "clean"
            if state.get("status") == "pass" and not state.get("output")
            else "dirty"
            if state.get("status") == "pass"
            else "unavailable",
            "entries": str(state.get("output", "")).splitlines(),
        },
    }


def _phase(
    status: str,
    duration_ns: int | None,
    detail: str,
    *,
    included_in: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "duration_ns": duration_ns,
        "detail": detail,
        "included_in": included_in,
    }


def _dtype_size(dtype: int) -> int:
    if dtype == DTYPE_F32:
        return 4
    if dtype == DTYPE_F64:
        return 8
    raise ValueError(f"unsupported dtype {dtype}")


def _route_ok(route: _Route, profile: str) -> bool:
    expected = PROFILE_IDS[profile]
    if route.route_id != expected or route.profile != expected:
        return False
    if route.overflow_policy != 1 or route.flags != 0:
        return False
    if profile == "fp32":
        expected_domains = (DTYPE_F32,) * 4
    elif profile == "mixed":
        expected_domains = (DTYPE_F32, DTYPE_F32, DTYPE_F64, DTYPE_F64)
    else:
        expected_domains = (DTYPE_F64,) * 4
    return (
        route.storage_dtype,
        route.pointwise_dtype,
        route.reduction_dtype,
        route.result_dtype,
    ) == expected_domains


def _const_view(dtype: int, array: Any, count: int) -> _ConstView:
    view = _ConstView()
    view.abi_version = ABI_VERSION
    view.struct_size = ctypes.sizeof(_ConstView)
    view.dtype = dtype
    view.flags = BUFFER_HOST | BUFFER_CONTIGUOUS
    view.data = ctypes.cast(array, ctypes.c_void_p)
    view.element_count = count
    view.byte_length = count * _dtype_size(dtype)
    view.byte_stride = _dtype_size(dtype)
    return view


def _mutable_view(dtype: int, array: Any, count: int) -> _MutableView:
    view = _MutableView()
    view.abi_version = ABI_VERSION
    view.struct_size = ctypes.sizeof(_MutableView)
    view.dtype = dtype
    view.flags = BUFFER_HOST | BUFFER_CONTIGUOUS
    view.data = ctypes.cast(array, ctypes.c_void_p)
    view.element_capacity = count
    view.byte_length = count * _dtype_size(dtype)
    view.byte_stride = _dtype_size(dtype)
    return view


def _initialize_runtime_context(backend: str) -> tuple[dict[str, object], Any | None]:
    """Force the vendor runtime/context boundary after loading the payload.

    Opening the runtime SONAME again returns the loader's existing object. The
    timed ``cudaFree(0)``/``hipFree(0)`` call is the first operation in this
    worker that deliberately requires a vendor runtime context. Metal exposes
    no equivalent C runtime operation at the canonical ABI boundary.
    """

    runtime_specs = {
        "cuda": (
            ("nvcudart_hybrid64.dll",)
            if os.name == "nt"
            else ("libcudart.so.13",),
            "cudaFree",
        ),
        "rocm": (("libamdhip64.so.7",), "hipFree"),
    }
    if backend not in runtime_specs:
        return (
            _phase(
                "not_observable",
                None,
                "Metal exposes no separate runtime/context initialization operation at the canonical ABI boundary",
            ),
            None,
        )
    library_names, function_name = runtime_specs[backend]
    runtime = None
    last_error: OSError | None = None
    start = time.perf_counter_ns()
    for library_name in library_names:
        try:
            runtime = ctypes.CDLL(library_name)
            break
        except OSError as exc:
            last_error = exc
    if runtime is None:
        return (
            _phase(
                "not_observable",
                None,
                f"vendor runtime handle unavailable for an isolated context timer: {last_error}",
            ),
            None,
        )
    initialize = getattr(runtime, function_name)
    initialize.argtypes = [ctypes.c_void_p]
    initialize.restype = ctypes.c_int
    status = int(initialize(None))
    duration = time.perf_counter_ns() - start
    if status != STATUS_OK:
        raise RuntimeError(
            f"{backend} runtime/context initialization failed: "
            f"{function_name}(0)={status}"
        )
    return (
        _phase(
            "observed",
            duration,
            f"first explicit {function_name}(0), including any remaining vendor runtime/context initialization",
        ),
        runtime,
    )


def _set_prototypes(lib: ctypes.CDLL) -> dict[str, Any]:
    c_uint = ctypes.c_uint32
    c_u64 = ctypes.c_uint64
    c_int = ctypes.c_int
    funcs: dict[str, Any] = {}
    funcs["routes"] = lib.gafime_gpu_numeric_routes_v2
    funcs["routes"].argtypes = [c_uint, c_uint, c_uint, ctypes.POINTER(_Route), c_uint, ctypes.POINTER(c_uint)]
    funcs["routes"].restype = c_int
    funcs["alloc"] = lib.gafime_gpu_matrix_alloc_v2
    funcs["alloc"].argtypes = [c_uint, ctypes.POINTER(_MatrixDesc), ctypes.POINTER(ctypes.c_void_p)]
    funcs["alloc"].restype = c_int
    funcs["upload"] = lib.gafime_gpu_matrix_upload_v2
    funcs["upload"].argtypes = [ctypes.c_void_p, ctypes.POINTER(_Route), ctypes.POINTER(_ConstView), ctypes.POINTER(_ConstView), c_u64, c_uint]
    funcs["upload"].restype = c_int
    funcs["update_target"] = lib.gafime_gpu_matrix_update_target_v2
    funcs["update_target"].argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Route),
        ctypes.POINTER(_ConstView),
        c_u64,
    ]
    funcs["update_target"].restype = c_int
    funcs["execute"] = lib.gafime_gpu_execute_v2
    funcs["execute"].argtypes = [ctypes.c_void_p, ctypes.POINTER(_NumericLaunch), ctypes.POINTER(_NumericResult)]
    funcs["execute"].restype = c_int
    funcs["execution_memory"] = lib.gafime_gpu_execution_memory_peak_v2
    funcs["execution_memory"].argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NumericLaunch),
        ctypes.POINTER(c_u64),
    ]
    funcs["execution_memory"].restype = c_int
    funcs["free"] = lib.gafime_gpu_matrix_free_v2
    funcs["free"].argtypes = [ctypes.c_void_p]
    funcs["free"].restype = c_int
    return funcs


def _run_abi_sample(
    *, payload: Path, backend: str, profile: str, source_root: str | None, wheel: str | None
) -> dict[str, object]:
    _abi_layout_self_check()
    phases = {
        name: _phase("not_observable", None, "not reached") for name in REQUIRED_PHASES
    }
    worker_start = time.perf_counter_ns()

    import_start = time.perf_counter_ns()
    gafime = importlib.import_module("gafime")
    import_duration = time.perf_counter_ns() - import_start
    phases["python_import"] = _phase(
        "observed", import_duration, "import gafime and its Python/native extension dependencies"
    )

    discovery_start = time.perf_counter_ns()
    payload_module = importlib.import_module("gafime._payloads")
    discovered = payload_module.discover_payloads(backend)
    discovery_duration = time.perf_counter_ns() - discovery_start
    phases["payload_discovery"] = _phase(
        "observed",
        discovery_duration,
        "installed-payload discovery before dynamic loading",
    )

    payload = payload.expanduser().resolve(strict=True)
    expected_kind = BACKEND_KINDS[backend]
    mode = 0
    if os.name != "nt":
        mode = int(getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_LOCAL", 0))
    load_start = time.perf_counter_ns()
    lib = ctypes.CDLL(str(payload), mode=mode) if mode else ctypes.CDLL(str(payload))
    load_duration = time.perf_counter_ns() - load_start
    phases["dynamic_library_load"] = _phase(
        "observed", load_duration, "ctypes CDLL/LoadLibrary of the exact payload path"
    )
    phases["code_object_or_module_registration"] = _phase(
        "observed_combined",
        load_duration,
        "host code-object/fatbinary registration runs in payload loader constructors and cannot be split from dlopen/LoadLibrary",
        included_in="dynamic_library_load",
    )

    resolve_start = time.perf_counter_ns()
    funcs = _set_prototypes(lib)
    resolve_duration = time.perf_counter_ns() - resolve_start
    phases["symbol_resolution"] = _phase(
        "observed", resolve_duration, "resolution of canonical ABI 1.1 generic symbols"
    )

    phases["runtime_context_initialization"], runtime_lib = (
        _initialize_runtime_context(backend)
    )

    count = ctypes.c_uint32()
    capability_start = time.perf_counter_ns()
    status = funcs["routes"](0, ABI_VERSION, ctypes.sizeof(_Route), None, 0, ctypes.byref(count))
    if status in (STATUS_UNSUPPORTED_BACKEND, STATUS_DEVICE_ERROR):
        return {
            "schema": WORKER_SCHEMA,
            "status": "unavailable",
            "backend": backend,
            "profile": profile,
            "reason": "payload route capability unavailable on this device",
            "status_code": int(status),
            "phases": phases,
            "provenance": _provenance(payload, source_root, wheel),
        }
    if status != STATUS_OK or count.value < PROFILE_IDS[profile]:
        raise RuntimeError(f"numeric route count failed: status={status}, count={count.value}")
    routes = (_Route * count.value)()
    status = funcs["routes"](
        0, ABI_VERSION, ctypes.sizeof(_Route), routes, count.value, ctypes.byref(count)
    )
    if status != STATUS_OK:
        raise RuntimeError(f"numeric route enumeration failed: status={status}")
    route = next((record for record in routes if record.profile == PROFILE_IDS[profile]), None)
    if route is None or not _route_ok(route, profile):
        raise RuntimeError(f"payload did not advertise the canonical {profile} route")
    capability_duration = time.perf_counter_ns() - capability_start
    phases["first_capability_query"] = _phase(
        "observed",
        capability_duration,
        "ABI 1.1 route count, enumeration, validation, and selection on a fresh process",
    )

    rows, cols = 4, 2
    if route.storage_dtype == DTYPE_F32:
        feature_array = (ctypes.c_float * (rows * cols))(1.0, 7.0, 2.0, 5.0, 3.0, 3.0, 4.0, 1.0)
        target_array = (ctypes.c_float * rows)(1.0, 2.0, 3.0, 4.0)
    else:
        epsilon = 1.0 / 1073741824.0
        feature_array = (ctypes.c_double * (rows * cols))(
            1.0 + epsilon, 7.0, 2.0 + epsilon, 5.0,
            3.0 + epsilon, 3.0, 4.0 + epsilon, 1.0,
        )
        target_array = (ctypes.c_double * rows)(1.0 + epsilon, 2.0 + epsilon, 3.0 + epsilon, 4.0 + epsilon)
    result_dtype = int(route.result_dtype)
    metric_array = (
        (ctypes.c_float * 1)() if result_dtype == DTYPE_F32 else (ctypes.c_double * 1)()
    )
    feature_view = _const_view(route.storage_dtype, feature_array, rows * cols)
    target_view = _const_view(route.storage_dtype, target_array, rows)
    desc = _MatrixDesc()
    desc.abi_version = ABI_VERSION
    desc.struct_size = ctypes.sizeof(_MatrixDesc)
    desc.route = route
    desc.layout = MATRIX_ROW_MAJOR
    desc.rows = rows
    desc.cols = cols
    desc.row_stride = cols
    desc.bytes = rows * cols * _dtype_size(route.storage_dtype)
    matrix = ctypes.c_void_p()
    alloc_start = time.perf_counter_ns()
    status = funcs["alloc"](0, ctypes.byref(desc), ctypes.byref(matrix))
    alloc_duration = time.perf_counter_ns() - alloc_start
    if status != STATUS_OK or not matrix.value:
        raise RuntimeError(f"matrix allocation failed: status={status}")
    phases["first_allocation"] = _phase(
        "observed", alloc_duration, "first generic matrix allocation"
    )
    try:
        upload_start = time.perf_counter_ns()
        status = funcs["upload"](
            matrix, ctypes.byref(route), ctypes.byref(feature_view), ctypes.byref(target_view), rows, cols
        )
        upload_duration = time.perf_counter_ns() - upload_start
        if status != STATUS_OK:
            raise RuntimeError(f"matrix upload failed: status={status}")
        phases["first_upload"] = _phase(
            "observed", upload_duration, "first typed f32/f64 host upload"
        )
        target_start = time.perf_counter_ns()
        status = funcs["update_target"](
            matrix, ctypes.byref(route), ctypes.byref(target_view), rows
        )
        target_duration = time.perf_counter_ns() - target_start
        if status != STATUS_OK:
            raise RuntimeError(f"target update failed: status={status}")
        phases["target_update"] = _phase(
            "observed", target_duration, "first canonical typed target replacement"
        )
        planning_start = time.perf_counter_ns()
        combo = ctypes.c_uint32(0)
        metric_id = ctypes.c_uint32(METRIC_PEARSON)
        chunk = _ArityChunk()
        chunk.arity = 1
        chunk.family = FAMILY_CONTINUOUS
        chunk.combo_count = 1
        chunk.descriptor_count = 1
        base = _LaunchProtocol()
        base.abi_version = (1 << 16) | 0
        base.backend_kind = expected_kind
        base.max_arity = 1
        base.n_samples = rows
        base.n_features = cols
        base.family_count = 1
        base.combo_indices = _SliceU32(ctypes.pointer(combo), 1)
        base.metric_ids = _SliceU32(ctypes.pointer(metric_id), 1)
        base.chunks = ctypes.pointer(chunk)
        base.chunk_count = 1
        base.rank.top_k = 1
        base.rank.primary_metric = METRIC_PEARSON
        protocol = _NumericLaunch()
        protocol.abi_version = ABI_VERSION
        protocol.struct_size = ctypes.sizeof(_NumericLaunch)
        protocol.route = route
        protocol.base = ctypes.pointer(base)
        combo_out = ctypes.c_uint32(0)
        ranks = ctypes.c_uint32(0)
        families = ctypes.c_uint32(0)
        candidate_ids = ctypes.c_uint64(0)
        row_flags = ctypes.c_uint32(0)
        result = _NumericResult()
        result.abi_version = ABI_VERSION
        result.struct_size = ctypes.sizeof(_NumericResult)
        result.max_arity = 1
        result.metric_count = 1
        result.capacity = 1
        result.combo_indices = ctypes.pointer(combo_out)
        result.metric_values = _mutable_view(result_dtype, metric_array, 1)
        result.ranks = ctypes.pointer(ranks)
        result.families = ctypes.pointer(families)
        result.candidate_ids = ctypes.pointer(candidate_ids)
        result.row_flags = ctypes.pointer(row_flags)
        planning_duration = time.perf_counter_ns() - planning_start
        phases["planning"] = _phase(
            "observed",
            planning_duration,
            "caller-side candidate, protocol, ranking, and typed result-buffer construction before execution",
        )
        execution_peak = ctypes.c_uint64()
        forecast_start = time.perf_counter_ns()
        status = funcs["execution_memory"](
            matrix, ctypes.byref(protocol), ctypes.byref(execution_peak)
        )
        forecast_duration = time.perf_counter_ns() - forecast_start
        if status != STATUS_OK or execution_peak.value == 0:
            raise RuntimeError(
                f"execution memory forecast failed: status={status}, peak={execution_peak.value}"
            )
        phases["execution_memory_forecast"] = _phase(
            "observed", forecast_duration, "canonical state-aware execution-memory forecast"
        )
        execution_start = time.perf_counter_ns()
        status = funcs["execute"](matrix, ctypes.byref(protocol), ctypes.byref(result))
        execution_duration = time.perf_counter_ns() - execution_start
        if status != STATUS_OK or result.row_count != 1:
            raise RuntimeError(
                "canonical execute failed: "
                f"status={status}, rows={result.row_count}, "
                f"route={route.route_id}/{route.profile}, "
                f"result_dtype={result.metric_values.dtype}, "
                f"capacity={result.capacity}, metric_count={result.metric_count}, "
                f"base_metrics={base.metric_ids.len}, base_chunks={base.chunk_count}"
            )
        phases["first_execution"] = _phase(
            "observed", execution_duration, "first canonical ABI execute call"
        )
        materialization_start = time.perf_counter_ns()
        visible_metric = float(metric_array[0])
        visible_result = (
            int(result.row_count),
            int(combo_out.value),
            int(ranks.value),
            int(families.value),
            int(candidate_ids.value),
            int(row_flags.value),
            visible_metric,
        )
        materialization_duration = time.perf_counter_ns() - materialization_start
        if visible_result[0] != 1 or not math.isfinite(visible_metric):
            raise RuntimeError(f"invalid materialized result: {visible_result!r}")
        phases["first_result_materialization"] = _phase(
            "observed",
            materialization_duration,
            "first typed host read of caller-owned structural and metric result buffers",
        )
    finally:
        cleanup_start = time.perf_counter_ns()
        free_status = funcs["free"](matrix)
        cleanup_duration = time.perf_counter_ns() - cleanup_start
        if free_status != STATUS_OK:
            raise RuntimeError(f"matrix free failed: status={free_status}")
        phases["explicit_cleanup"] = _phase(
            "observed", cleanup_duration, "canonical ABI matrix_free_v2"
        )

    del runtime_lib
    del lib
    gc.collect()
    phases["process_exit_cleanup"] = _phase(
        "not_observable",
        None,
        "process exit and dynamic-loader teardown are outside the worker timing boundary",
    )
    return {
        "schema": WORKER_SCHEMA,
        "status": "pass",
        "backend": backend,
        "profile": profile,
        "backend_kind": expected_kind,
        "route": {
            "route_id": int(route.route_id),
            "profile": int(route.profile),
            "storage_dtype": int(route.storage_dtype),
            "pointwise_dtype": int(route.pointwise_dtype),
            "reduction_dtype": int(route.reduction_dtype),
            "result_dtype": int(route.result_dtype),
        },
        "workload": {"rows": rows, "cols": cols, "candidate_count": 1, "metric": "pearson"},
        "phases": phases,
        "worker_duration_ns": time.perf_counter_ns() - worker_start,
        "payload_discovered": {str(key): str(value) for key, value in discovered.items()},
        "provenance": _provenance(payload, source_root, wheel),
    }


def _provenance(payload: Path, source_root: str | None, wheel: str | None) -> dict[str, object]:
    result: dict[str, object] = {
        "payload": _identity(payload),
        "benchmark_script": _identity(Path(__file__).resolve()),
        "source": _source_provenance(source_root),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "HSA_OVERRIDE_GFX_VERSION",
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
            )
            if key in os.environ
        },
    }
    if wheel:
        result["wheel"] = _identity(wheel)
    else:
        result["wheel"] = {"status": "not_supplied"}
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty distribution")
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), int(position) + (0 if position.is_integer() else 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Sequence[int], seed: int, bootstrap_resamples: int) -> dict[str, object]:
    numeric = [float(value) for value in values]
    center = float(statistics.median(numeric))
    mad = float(statistics.median(abs(value - center) for value in numeric))
    rng = random.Random(seed)
    boot = [
        statistics.median(rng.choice(numeric) for _ in numeric)
        for _ in range(bootstrap_resamples)
    ]
    return {
        "count": len(values),
        "raw_duration_ns": [int(value) for value in values],
        "median_ns": center,
        "mad_ns": mad,
        "p05_ns": _percentile(numeric, 0.05),
        "p95_ns": _percentile(numeric, 0.95),
        "bootstrap_median_95_ci_ns": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "bootstrap_resamples": bootstrap_resamples,
    }


def _stable_seed(seed: int, *parts: object) -> int:
    """Derive a reproducible bootstrap seed without Python's randomized hash."""

    material = "/".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "little")


def _parse_named(raw: Sequence[str], label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in raw:
        name, sep, value = entry.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"--{label} requires NAME=VALUE")
        if name in values:
            raise ValueError(f"duplicate {label} {name}")
        values[name] = value
    return values


def _driver(args: argparse.Namespace) -> dict[str, object]:
    profiles = tuple(dict.fromkeys(part.strip().lower() for raw in args.profile for part in raw.split(",") if part.strip()))
    if not profiles:
        profiles = PROFILE_ORDER
    unknown = set(profiles) - set(PROFILE_ORDER)
    if unknown:
        raise ValueError(f"unknown profile(s): {sorted(unknown)}")
    backend = args.backend.lower()
    if backend not in BACKEND_KINDS:
        raise ValueError(f"unsupported backend {backend!r}")
    if backend == "metal" and any(profile != "fp32" for profile in profiles):
        raise ValueError("Metal cold lifecycle accepts fp32 only")
    payload = Path(args.payload).expanduser().resolve(strict=True)
    source_root = str(Path(args.source_root).expanduser().resolve()) if args.source_root else None
    wheel = str(Path(args.wheel).expanduser().resolve(strict=True)) if args.wheel else None
    all_orders = tuple(itertools.permutations(profiles))
    order_schedule = _profile_order_schedule(profiles, args.repetitions, args.seed)
    records: list[dict[str, object]] = []
    driver_start = time.perf_counter_ns()
    for index, order in enumerate(order_schedule):
        for position, profile in enumerate(order):
            worker_command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_worker",
                "--backend",
                backend,
                "--profile",
                profile,
                "--payload",
                str(payload),
            ]
            if source_root:
                worker_command += ["--source-root", source_root]
            if wheel:
                worker_command += ["--wheel", wheel]
            launched = time.perf_counter_ns()
            completed = subprocess.run(
                worker_command,
                capture_output=True,
                text=True,
                check=False,
                timeout=args.timeout_seconds,
            )
            parent_duration = time.perf_counter_ns() - launched
            if not completed.stdout.strip():
                raise RuntimeError(
                    f"cold worker emitted no JSON for {backend}/{profile}: "
                    f"rc={completed.returncode} stderr={completed.stderr.strip()}"
                )
            try:
                record = json.loads(completed.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"cold worker emitted invalid JSON: {completed.stdout[-1000:]}"
                ) from exc
            worker_duration = record.get("worker_duration_ns")
            phases = record.get("phases")
            if isinstance(worker_duration, int) and isinstance(phases, dict):
                residual = max(0, parent_duration - worker_duration)
                phases["process_exit_cleanup"] = _phase(
                    "observed_combined",
                    residual,
                    "parent subprocess duration minus the worker cold region; combines interpreter startup, provenance/JSON output, and process-exit teardown",
                    included_in="parent_process_duration",
                )
            record.update(
                {
                    "sample_index": index,
                    "profile_order": list(order),
                    "profile_position": position,
                    "worker_command": worker_command,
                    "worker_returncode": completed.returncode,
                    "parent_process_duration_ns": parent_duration,
                    "worker_stderr": completed.stderr[-16_384:],
                }
            )
            if completed.returncode != 0 or record.get("status") not in {"pass", "unavailable"}:
                raise RuntimeError(
                    f"cold worker failed for {backend}/{profile}: "
                    f"rc={completed.returncode}, record={record}"
                )
            records.append(record)
    phase_groups: dict[tuple[str, str], list[int]] = {}
    combined_groups: dict[tuple[str, str], list[int]] = {}
    status_counts: dict[tuple[str, str, str], int] = {}
    for record in records:
        profile = str(record.get("profile"))
        phases = record.get("phases", {})
        if not isinstance(phases, Mapping):
            continue
        for phase_name, value in phases.items():
            if not isinstance(value, Mapping):
                continue
            status = str(value.get("status"))
            status_counts[(profile, str(phase_name), status)] = status_counts.get((profile, str(phase_name), status), 0) + 1
            duration = value.get("duration_ns")
            if not isinstance(duration, (int, float)) or duration <= 0:
                continue
            key = (profile, str(phase_name))
            if status == "observed":
                phase_groups.setdefault(key, []).append(int(duration))
            elif status == "observed_combined":
                combined_groups.setdefault(key, []).append(int(duration))
    summaries: dict[str, object] = {}
    for profile in profiles:
        summaries[profile] = {}
        for phase in REQUIRED_PHASES + ("symbol_resolution",):
            key = (profile, phase)
            entry: dict[str, object] = {
                "status_counts": {
                    status: count
                    for (p, name, status), count in status_counts.items()
                    if p == profile and name == phase
                },
                "observed": _stats(
                    phase_groups[key],
                    _stable_seed(args.seed, "observed", *key),
                    args.bootstrap_resamples,
                )
                if key in phase_groups
                else None,
                "observed_combined": _stats(
                    combined_groups[key],
                    _stable_seed(args.seed, "combined", *key),
                    args.bootstrap_resamples,
                )
                if key in combined_groups
                else None,
            }
            summaries[profile][phase] = entry
    return {
        "schema": SCHEMA,
        "status": "pass" if all(record.get("status") == "pass" for record in records) else "partial",
        "backend": backend,
        "profiles": list(profiles),
        "repetitions_per_profile": args.repetitions,
        "fresh_subprocess_per_sample": True,
        "profile_orders": [list(order) for order in all_orders],
        "profile_order_counts": {
            "/".join(order): sum(1 for record in records if tuple(record.get("profile_order", ())) == order) // len(profiles)
            for order in all_orders
        },
        "workload": {"rows": 4, "cols": 2, "candidate_count": 1, "metric": "pearson"},
        "driver_command": sys.argv,
        "driver_duration_ns": time.perf_counter_ns() - driver_start,
        "provenance": {
            "payload": _identity(payload),
            "benchmark_script": _identity(Path(__file__).resolve()),
            "source": _source_provenance(source_root),
            "wheel": _identity(wheel) if wheel else {"status": "not_supplied"},
            "python": {"executable": sys.executable, "version": platform.python_version()},
        },
        "records": records,
        "phase_summaries": summaries,
        "phase_boundary_policy": {
            "runtime_context_initialization": "first explicit cudaFree(0) or hipFree(0) is timed after payload loading; Metal has no separate runtime C boundary",
            "code_object_or_module_registration": "observed_combined with dynamic-library load because registration runs in loader constructors",
            "planning": "caller-side canonical protocol and result-buffer construction is timed separately",
            "first_result_materialization": "first typed host read of caller-owned result buffers is timed after execute",
            "process_exit_cleanup": "parent-minus-worker residual is retained as a combined startup, provenance/serialization, and process-exit boundary; it is not labeled pure teardown",
        },
    }


def _profile_order_schedule(
    profiles: Sequence[str], repetitions: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Return randomized complete permutation blocks for isolated workers."""

    if repetitions < 0:
        raise ValueError("repetitions must be non-negative")
    permutations = list(itertools.permutations(profiles))
    if not permutations and repetitions:
        raise ValueError("at least one profile is required")
    rng = random.Random(seed)
    schedule: list[tuple[str, ...]] = []
    while len(schedule) < repetitions:
        rng.shuffle(permutations)
        schedule.extend(permutations)
    return tuple(schedule[:repetitions])


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKEND_KINDS))
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("-"))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_check:
        return args
    if args._worker:
        if not args.backend or not args.payload or len(args.profile) != 1:
            parser.error("worker requires --backend, --payload and one --profile")
        return args
    if not args.backend or not args.payload:
        parser.error("driver requires --backend and --payload")
    if args.repetitions < 30:
        parser.error("--repetitions must be at least 30")
    if args.bootstrap_resamples < 500:
        parser.error("--bootstrap-resamples must be at least 500")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.self_check:
            _abi_layout_self_check()
            print(json.dumps({"schema": SCHEMA, "status": "pass", "layout": "stable"}))
            return 0
        if args._worker:
            record = _run_abi_sample(
                payload=args.payload,
                backend=args.backend,
                profile=args.profile[0].lower(),
                source_root=str(args.source_root) if args.source_root else None,
                wheel=str(args.wheel) if args.wheel else None,
            )
        else:
            record = _driver(args)
        encoded = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            if args._worker
            else json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        if args.output == Path("-"):
            print(encoded, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
    except (OSError, RuntimeError, ValueError, ctypes.ArgumentError, subprocess.SubprocessError) as exc:
        print(f"cold lifecycle benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
