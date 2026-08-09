#!/usr/bin/env python3
"""Measure the observable cold lifecycle of a GAFIME ABI 1.1 payload.

This benchmark is intentionally separate from the public precision benchmark.
Each profile sample runs in a new process and calls the canonical ABI directly.
Current payloads use the generic numeric-route ABI; frozen pre-route ABI 1.1
payloads use the typed precision fallback.  Both paths retain honest
boundaries for Python import, payload discovery, dynamic loading, capability
negotiation, allocation, upload, execution, and teardown.  The ABI has no
separate planning, result-materialisation, or module-registration hooks; those
fields are therefore explicitly marked as combined, host-only, or
not-comparable instead of being inferred from a public timer.

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

# These labels are deliberately about comparing the generic numeric-route
# surface with the frozen typed precision surface.  A phase can still be
# observed on each payload while being unsuitable for a cross-surface delta.
PHASE_COMPARABILITY_LIMITS = {
    "symbol_resolution": {
        "status": "not_comparable",
        "detail": "generic and typed payloads resolve different symbol sets",
    },
    "first_capability_query": {
        "status": "not_comparable",
        "detail": "generic route enumeration is not the typed capability-mask query",
    },
    "planning": {
        "status": "not_comparable",
        "detail": "generic route/result-view structures differ from typed protocol/result structures",
    },
    "first_result_materialization": {
        "status": "host_only_d2h_unobservable",
        "detail": "this phase reads caller-owned host buffers; execute already includes vendor D2H and synchronization",
    },
    "first_allocation": {
        "status": "semantic_only",
        "detail": "same profile allocation boundary, but generic and typed descriptor validation differs",
    },
    "first_upload": {
        "status": "semantic_only",
        "detail": "same typed host upload operation, with different ABI wrapper validation",
    },
    "target_update": {
        "status": "semantic_only",
        "detail": "same typed target-update operation, with different ABI wrapper validation",
    },
    "execution_memory_forecast": {
        "status": "semantic_only",
        "detail": "same forecast concept, but the protocol wrapper layout differs",
    },
    "first_execution": {
        "status": "semantic_only",
        "detail": "same profile/metric/ranking execution boundary, with different ABI wrapper and result validation",
    },
    "explicit_cleanup": {
        "status": "semantic_only",
        "detail": "same matrix teardown, but frozen typed free is void while generic free returns status",
    },
    "code_object_or_module_registration": {
        "status": "combined_not_separately_observable",
        "detail": "loader constructors combine registration with dynamic-library load",
    },
    "process_exit_cleanup": {
        "status": "combined_not_separately_observable",
        "detail": "process-exit residual also contains startup, provenance, and serialization work",
    },
}

# Publish a value for every declared phase.  Unlisted phases are directly
# observable; the explicit entries above are the only cross-surface caveats.
PHASE_COMPARABILITY = {
    phase: PHASE_COMPARABILITY_LIMITS.get(
        phase,
        {"status": "direct", "detail": "directly observable at this ABI boundary"},
    )
    for phase in REQUIRED_PHASES
}


def _phase_comparability(phase: str) -> str:
    return str(PHASE_COMPARABILITY[phase]["status"])


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


class _PrecisionMatrixDesc(ctypes.Structure):
    """Frozen pre-route ABI 1.1 typed matrix descriptor."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("layout", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved32", ctypes.c_uint32),
        ("rows", ctypes.c_uint64),
        ("cols", ctypes.c_uint32),
        ("row_stride", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _PrecisionCapabilities(ctypes.Structure):
    """Frozen pre-route ABI 1.1 profile capability record."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("backend_kind", ctypes.c_uint32),
        ("profile_mask", ctypes.c_uint32),
        ("storage_dtype_mask", ctypes.c_uint32),
        ("result_dtype_mask", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _PrecisionLaunch(ctypes.Structure):
    """Frozen pre-route ABI 1.1 typed protocol wrapper."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("base", ctypes.POINTER(_LaunchProtocol)),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _PrecisionResult(ctypes.Structure):
    """ABI 1.0 structural result table with f32 metric values."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("max_arity", ctypes.c_uint32),
        ("metric_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("capacity", ctypes.c_uint64),
        ("row_count", ctypes.c_uint64),
        ("combo_indices", ctypes.POINTER(ctypes.c_uint32)),
        ("metric_values", ctypes.POINTER(ctypes.c_float)),
        ("ranks", ctypes.POINTER(ctypes.c_uint32)),
        ("families", ctypes.POINTER(ctypes.c_uint32)),
        ("candidate_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("row_flags", ctypes.POINTER(ctypes.c_uint32)),
        ("backend_private", ctypes.c_void_p),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _PrecisionResultF64(ctypes.Structure):
    """ABI 1.1 structural result table with f64 metric values."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("max_arity", ctypes.c_uint32),
        ("metric_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("capacity", ctypes.c_uint64),
        ("row_count", ctypes.c_uint64),
        ("combo_indices", ctypes.POINTER(ctypes.c_uint32)),
        ("metric_values", ctypes.POINTER(ctypes.c_double)),
        ("ranks", ctypes.POINTER(ctypes.c_uint32)),
        ("families", ctypes.POINTER(ctypes.c_uint32)),
        ("candidate_ids", ctypes.POINTER(ctypes.c_uint64)),
        ("row_flags", ctypes.POINTER(ctypes.c_uint32)),
        ("backend_private", ctypes.c_void_p),
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
        _PrecisionMatrixDesc: (112, 8),
        _PrecisionCapabilities: (88, 8),
        _PrecisionLaunch: (80, 8),
        _PrecisionResult: (152, 8),
        _PrecisionResultF64: (152, 8),
    }
    for structure, (size, alignment) in expected.items():
        if ctypes.sizeof(structure) != size or ctypes.alignment(structure) != alignment:
            raise RuntimeError(
                f"ABI ctypes layout drift for {structure.__name__}: "
                f"size={ctypes.sizeof(structure)} align={ctypes.alignment(structure)}"
            )
    offsets = {
        (_Route, "route_id"): 8,
        (_Route, "result_dtype"): 28,
        (_Route, "reserved"): 40,
        (_PrecisionMatrixDesc, "rows"): 24,
        (_PrecisionMatrixDesc, "reserved"): 48,
        (_PrecisionLaunch, "base"): 8,
        (_PrecisionResult, "metric_values"): 40,
        (_PrecisionResultF64, "metric_values"): 40,
    }
    for (structure, field), expected_offset in offsets.items():
        actual_offset = int(getattr(structure, field).offset)
        if actual_offset != expected_offset:
            raise RuntimeError(
                f"ABI ctypes offset drift for {structure.__name__}.{field}: "
                f"expected={expected_offset} actual={actual_offset}"
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
    comparability: str = "direct",
) -> dict[str, object]:
    return {
        "status": status,
        "duration_ns": duration_ns,
        "detail": detail,
        "included_in": included_in,
        "comparability": comparability,
    }


def _validate_phase_comparability(phases: Mapping[str, Any]) -> None:
    """Reject a record whose emitted phase labels drift from the contract."""

    missing = set(REQUIRED_PHASES) - set(phases)
    if missing:
        raise RuntimeError(f"cold lifecycle phase record is missing {sorted(missing)}")
    for phase in REQUIRED_PHASES:
        value = phases[phase]
        if not isinstance(value, Mapping):
            raise RuntimeError(f"cold lifecycle phase {phase} is not a mapping")
        actual = value.get("comparability")
        expected = _phase_comparability(phase)
        if actual != expected:
            raise RuntimeError(
                f"cold lifecycle phase {phase} comparability drift: "
                f"expected={expected!r}, actual={actual!r}"
            )


def _dtype_size(dtype: int) -> int:
    if dtype == DTYPE_F32:
        return 4
    if dtype == DTYPE_F64:
        return 8
    raise ValueError(f"unsupported dtype {dtype}")


def _route_ok(route: _Route, profile: str) -> bool:
    expected = PROFILE_IDS[profile]
    if route.abi_version >> 16 != ABI_VERSION >> 16:
        return False
    if (route.abi_version & 0xFFFF) < (ABI_VERSION & 0xFFFF):
        return False
    if route.struct_size < int(_Route.reserved.offset):
        return False
    if route.route_id != expected or route.profile != expected:
        return False
    if route.overflow_policy != 1 or route.flags != 0:
        return False
    if any(int(value) != 0 for value in route.reserved):
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


def _bind_symbol(
    lib: Any,
    name: str,
    argtypes: Sequence[Any],
    restype: Any,
    *,
    required: bool = True,
) -> Any | None:
    """Bind a C symbol while keeping fake-library tests dependency-free."""

    try:
        function = getattr(lib, name)
    except AttributeError as exc:
        if required:
            raise RuntimeError(f"payload is missing required ABI symbol {name}") from exc
        return None
    # ctypes symbols expose these attributes.  Small Python fakes used by the
    # contract tests intentionally do not need to emulate ctypes internals.
    try:
        function.argtypes = list(argtypes)
        function.restype = restype
    except (AttributeError, TypeError):
        pass
    return function


def _set_prototypes(lib: ctypes.CDLL) -> dict[str, Any]:
    """Select generic numeric-route ABI or frozen typed precision ABI.

    The baseline release predates the generic route records but already
    exposes the ABI 1.1 typed profile surface.  Detection is based solely on
    the symbol surface; the selected value is recorded in every worker JSON.
    """

    c_uint = ctypes.c_uint32
    c_u64 = ctypes.c_uint64
    c_int = ctypes.c_int
    void_p = ctypes.c_void_p
    funcs: dict[str, Any] = {}

    try:
        generic_routes = getattr(lib, "gafime_gpu_numeric_routes_v2")
    except AttributeError:
        generic_routes = None

    if generic_routes is not None:
        funcs["abi_surface"] = "numeric-route-v2"
        funcs["route_source"] = "numeric_routes_v2"
        funcs["routes"] = generic_routes
        try:
            generic_routes.argtypes = [
                c_uint,
                c_uint,
                c_uint,
                ctypes.POINTER(_Route),
                c_uint,
                ctypes.POINTER(c_uint),
            ]
            generic_routes.restype = c_int
        except (AttributeError, TypeError):
            pass
        funcs["alloc"] = _bind_symbol(
            lib,
            "gafime_gpu_matrix_alloc_v2",
            [c_uint, ctypes.POINTER(_MatrixDesc), ctypes.POINTER(void_p)],
            c_int,
        )
        funcs["upload"] = _bind_symbol(
            lib,
            "gafime_gpu_matrix_upload_v2",
            [
                void_p,
                ctypes.POINTER(_Route),
                ctypes.POINTER(_ConstView),
                ctypes.POINTER(_ConstView),
                c_u64,
                c_uint,
            ],
            c_int,
        )
        funcs["update_target"] = _bind_symbol(
            lib,
            "gafime_gpu_matrix_update_target_v2",
            [void_p, ctypes.POINTER(_Route), ctypes.POINTER(_ConstView), c_u64],
            c_int,
        )
        funcs["execute"] = _bind_symbol(
            lib,
            "gafime_gpu_execute_v2",
            [void_p, ctypes.POINTER(_NumericLaunch), ctypes.POINTER(_NumericResult)],
            c_int,
        )
        funcs["execution_memory"] = _bind_symbol(
            lib,
            "gafime_gpu_execution_memory_peak_v2",
            [void_p, ctypes.POINTER(_NumericLaunch), ctypes.POINTER(c_u64)],
            c_int,
        )
        funcs["free"] = _bind_symbol(
            lib, "gafime_gpu_matrix_free_v2", [void_p], c_int
        )
        funcs["free_returns_status"] = True
        return funcs

    funcs["abi_surface"] = "precision-typed-v1.1"
    funcs["route_source"] = "precision_capabilities"
    funcs["capabilities"] = _bind_symbol(
        lib,
        "gafime_gpu_precision_capabilities",
        [c_uint, ctypes.POINTER(_PrecisionCapabilities)],
        c_int,
    )
    funcs["alloc"] = _bind_symbol(
        lib,
        "gafime_gpu_matrix_alloc_v2",
        [c_uint, ctypes.POINTER(_PrecisionMatrixDesc), ctypes.POINTER(void_p)],
        c_int,
    )
    funcs["upload_f32"] = _bind_symbol(
        lib,
        "gafime_gpu_matrix_upload_f32_v2",
        [void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), c_u64, c_uint],
        c_int,
        required=False,
    )
    funcs["upload_f64"] = _bind_symbol(
        lib,
        "gafime_gpu_matrix_upload_f64_v2",
        [void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), c_u64, c_uint],
        c_int,
        required=False,
    )
    funcs["update_target_f32"] = _bind_symbol(
        lib,
        "gafime_gpu_matrix_update_target_f32_v2",
        [void_p, ctypes.POINTER(ctypes.c_float), c_u64],
        c_int,
        required=False,
    )
    funcs["update_target_f64"] = _bind_symbol(
        lib,
        "gafime_gpu_matrix_update_target_f64_v2",
        [void_p, ctypes.POINTER(ctypes.c_double), c_u64],
        c_int,
        required=False,
    )
    funcs["execute_f32"] = _bind_symbol(
        lib,
        "gafime_gpu_execute_f32_v2",
        [void_p, ctypes.POINTER(_PrecisionLaunch), ctypes.POINTER(_PrecisionResult)],
        c_int,
        required=False,
    )
    funcs["execute_f64"] = _bind_symbol(
        lib,
        "gafime_gpu_execute_f64_v2",
        [void_p, ctypes.POINTER(_PrecisionLaunch), ctypes.POINTER(_PrecisionResultF64)],
        c_int,
        required=False,
    )
    funcs["execution_memory"] = _bind_symbol(
        lib,
        "gafime_gpu_execution_memory_peak_v2",
        [void_p, ctypes.POINTER(_PrecisionLaunch), ctypes.POINTER(c_u64)],
        c_int,
    )
    # The frozen ABI 1.1 free entry point is void and generation-neutral.
    funcs["free"] = _bind_symbol(lib, "gafime_gpu_matrix_free", [void_p], None)
    funcs["free_returns_status"] = False
    return funcs


def _route_for_profile(profile: str) -> _Route:
    """Synthesize the route record implied by a typed capability mask."""

    route = _Route()
    profile_id = PROFILE_IDS[profile]
    route.abi_version = ABI_VERSION
    route.struct_size = ctypes.sizeof(_Route)
    route.route_id = profile_id
    route.profile = profile_id
    if profile == "fp32":
        domains = (DTYPE_F32,) * 4
    elif profile == "mixed":
        domains = (DTYPE_F32, DTYPE_F32, DTYPE_F64, DTYPE_F64)
    else:
        domains = (DTYPE_F64,) * 4
    (
        route.storage_dtype,
        route.pointwise_dtype,
        route.reduction_dtype,
        route.result_dtype,
    ) = domains
    route.overflow_policy = 1
    return route


def _select_route(
    funcs: Mapping[str, Any], backend: str, profile: str
) -> tuple[_Route | None, int, str, dict[str, object]]:
    """Return a validated generic route or a typed-profile compatibility route."""

    if funcs["abi_surface"] == "numeric-route-v2":
        count = ctypes.c_uint32()
        status = int(
            funcs["routes"](
                0, ABI_VERSION, ctypes.sizeof(_Route), None, 0, ctypes.byref(count)
            )
        )
        if status in (STATUS_UNSUPPORTED_BACKEND, STATUS_DEVICE_ERROR):
            return None, status, "payload route capability unavailable on this device", {}
        if status != STATUS_OK:
            raise RuntimeError(
                f"numeric route count failed: status={status}, count={count.value}"
            )
        routes = (_Route * count.value)()
        status = int(
            funcs["routes"](
                0,
                ABI_VERSION,
                ctypes.sizeof(_Route),
                routes,
                count.value,
                ctypes.byref(count),
            )
        )
        if status != STATUS_OK:
            raise RuntimeError(f"numeric route enumeration failed: status={status}")
        route_ids = [int(record.route_id) for record in routes]
        if len(route_ids) != len(set(route_ids)):
            raise RuntimeError("payload advertised duplicate numeric route ids")
        known_profiles = [
            int(record.profile)
            for record in routes
            if int(record.profile) in PROFILE_IDS.values()
        ]
        if len(known_profiles) != len(set(known_profiles)):
            raise RuntimeError("payload advertised duplicate known precision profiles")
        names_by_profile = {value: name for name, value in PROFILE_IDS.items()}
        for record in routes:
            known_name = names_by_profile.get(int(record.profile))
            if known_name is not None and not _route_ok(record, known_name):
                raise RuntimeError(
                    f"payload advertised a malformed {known_name} numeric route"
                )
        route = next((record for record in routes if record.profile == PROFILE_IDS[profile]), None)
        if route is None or not _route_ok(route, profile):
            raise RuntimeError(f"payload did not advertise the canonical {profile} route")
        return route, STATUS_OK, "ABI 1.1 numeric-route count, enumeration, validation, and selection", {
            "profile_mask": sum(1 << (int(record.profile) - 1) for record in routes if record.profile in PROFILE_IDS.values()),
        }

    capabilities = _PrecisionCapabilities()
    status = int(funcs["capabilities"](0, ctypes.byref(capabilities)))
    if status in (STATUS_UNSUPPORTED_BACKEND, STATUS_DEVICE_ERROR):
        return (
            None,
            status,
            "typed precision capability query unavailable on this device",
            {},
        )
    if status != STATUS_OK:
        raise RuntimeError(f"typed precision capability query failed: status={status}")
    expected_kind = BACKEND_KINDS[backend]
    if capabilities.abi_version != ABI_VERSION:
        raise RuntimeError(
            "typed precision capability ABI mismatch: "
            f"expected={ABI_VERSION}, actual={capabilities.abi_version}"
        )
    if capabilities.backend_kind != expected_kind:
        raise RuntimeError(
            "typed precision capability backend mismatch: "
            f"expected={expected_kind}, actual={capabilities.backend_kind}"
        )
    capability_metadata = {
        "profile_mask": int(capabilities.profile_mask),
        "storage_dtype_mask": int(capabilities.storage_dtype_mask),
        "result_dtype_mask": int(capabilities.result_dtype_mask),
        "flags": int(capabilities.flags),
    }
    profile_bit = 1 << (PROFILE_IDS[profile] - 1)
    if capabilities.profile_mask & profile_bit == 0:
        return (
            None,
            STATUS_UNSUPPORTED_BACKEND,
            f"typed payload does not advertise {profile}",
            capability_metadata,
        )
    route = _route_for_profile(profile)
    storage_bit = 1 << (int(route.storage_dtype) - 1)
    result_bit = 1 << (int(route.result_dtype) - 1)
    if not capabilities.storage_dtype_mask & storage_bit:
        raise RuntimeError(
            f"typed payload advertises {profile} but omits storage dtype mask bit {storage_bit}"
        )
    if not capabilities.result_dtype_mask & result_bit:
        raise RuntimeError(
            f"typed payload advertises {profile} but omits result dtype mask bit {result_bit}"
        )
    capability_metadata["route_synthesized"] = True
    return (
        route,
        STATUS_OK,
        "ABI 1.1 typed precision capability-mask query and profile selection",
        capability_metadata,
    )


def _run_abi_sample(
    *, payload: Path, backend: str, profile: str, source_root: str | None, wheel: str | None
) -> dict[str, object]:
    _abi_layout_self_check()
    phases = {
        name: _phase(
            "not_observable",
            None,
            "not reached",
            comparability=_phase_comparability(name),
        )
        for name in REQUIRED_PHASES
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
        comparability=_phase_comparability("code_object_or_module_registration"),
    )

    resolve_start = time.perf_counter_ns()
    funcs = _set_prototypes(lib)
    resolve_duration = time.perf_counter_ns() - resolve_start
    phases["symbol_resolution"] = _phase(
        "observed",
        resolve_duration,
        f"resolution of canonical ABI 1.1 {funcs['abi_surface']} symbols",
        comparability=_phase_comparability("symbol_resolution"),
    )

    phases["runtime_context_initialization"], runtime_lib = (
        _initialize_runtime_context(backend)
    )

    capability_start = time.perf_counter_ns()
    route, status, capability_detail, capability_metadata = _select_route(
        funcs, backend, profile
    )
    capability_duration = time.perf_counter_ns() - capability_start
    phases["first_capability_query"] = _phase(
        "observed" if route is not None else "unavailable",
        capability_duration,
        capability_detail,
        comparability=_phase_comparability("first_capability_query"),
    )
    if route is None:
        _validate_phase_comparability(phases)
        return {
            "schema": WORKER_SCHEMA,
            "status": "unavailable",
            "backend": backend,
            "profile": profile,
            "backend_kind": expected_kind,
            "abi_surface": funcs["abi_surface"],
            "route_source": funcs["route_source"],
            "reason": capability_detail,
            "status_code": int(status),
            "capability": capability_metadata,
            "phase_comparability": PHASE_COMPARABILITY,
            "phases": phases,
            "provenance": _provenance(payload, source_root, wheel),
        }

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
    typed_surface = funcs["abi_surface"] == "precision-typed-v1.1"
    if typed_surface:
        desc = _PrecisionMatrixDesc()
        desc.abi_version = ABI_VERSION
        desc.profile = PROFILE_IDS[profile]
        desc.dtype = route.storage_dtype
        desc.layout = MATRIX_ROW_MAJOR
        desc.rows = rows
        desc.cols = cols
        desc.row_stride = cols
        desc.bytes = rows * cols * _dtype_size(route.storage_dtype)
    else:
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
        "observed",
        alloc_duration,
        f"first {funcs['abi_surface']} matrix allocation",
        comparability=_phase_comparability("first_allocation"),
    )
    try:
        upload_start = time.perf_counter_ns()
        if typed_surface:
            upload = funcs["upload_f32"] if route.storage_dtype == DTYPE_F32 else funcs["upload_f64"]
            if upload is None:
                raise RuntimeError(
                    f"typed payload is missing the upload symbol for {profile}"
                )
            status = upload(matrix, feature_array, target_array, rows, cols)
        else:
            status = funcs["upload"](
                matrix,
                ctypes.byref(route),
                ctypes.byref(feature_view),
                ctypes.byref(target_view),
                rows,
                cols,
            )
        upload_duration = time.perf_counter_ns() - upload_start
        if status != STATUS_OK:
            raise RuntimeError(f"matrix upload failed: status={status}")
        phases["first_upload"] = _phase(
            "observed",
            upload_duration,
            f"first typed f32/f64 host upload through {funcs['abi_surface']}",
            comparability=_phase_comparability("first_upload"),
        )
        target_start = time.perf_counter_ns()
        if typed_surface:
            update_target = (
                funcs["update_target_f32"]
                if route.storage_dtype == DTYPE_F32
                else funcs["update_target_f64"]
            )
            if update_target is None:
                raise RuntimeError(
                    f"typed payload is missing the target-update symbol for {profile}"
                )
            status = update_target(matrix, target_array, rows)
        else:
            status = funcs["update_target"](
                matrix, ctypes.byref(route), ctypes.byref(target_view), rows
            )
        target_duration = time.perf_counter_ns() - target_start
        if status != STATUS_OK:
            raise RuntimeError(f"target update failed: status={status}")
        phases["target_update"] = _phase(
            "observed",
            target_duration,
            f"first canonical typed target replacement through {funcs['abi_surface']}",
            comparability=_phase_comparability("target_update"),
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
        combo_out = ctypes.c_uint32(0)
        ranks = ctypes.c_uint32(0)
        families = ctypes.c_uint32(0)
        candidate_ids = ctypes.c_uint64(0)
        row_flags = ctypes.c_uint32(0)
        if typed_surface:
            protocol = _PrecisionLaunch()
            protocol.abi_version = ABI_VERSION
            protocol.profile = PROFILE_IDS[profile]
            protocol.base = ctypes.pointer(base)
            if result_dtype == DTYPE_F32:
                result = _PrecisionResult()
                result.abi_version = (1 << 16) | 0
                result_metric_type = ctypes.c_float
            else:
                result = _PrecisionResultF64()
                result.abi_version = ABI_VERSION
                result_metric_type = ctypes.c_double
            result.max_arity = 1
            result.metric_count = 1
            result.capacity = 1
            result.combo_indices = ctypes.pointer(combo_out)
            result.metric_values = ctypes.cast(
                metric_array, ctypes.POINTER(result_metric_type)
            )
            result.ranks = ctypes.pointer(ranks)
            result.families = ctypes.pointer(families)
            result.candidate_ids = ctypes.pointer(candidate_ids)
            result.row_flags = ctypes.pointer(row_flags)
        else:
            protocol = _NumericLaunch()
            protocol.abi_version = ABI_VERSION
            protocol.struct_size = ctypes.sizeof(_NumericLaunch)
            protocol.route = route
            protocol.base = ctypes.pointer(base)
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
            f"caller-side candidate, protocol, ranking, and typed result-buffer construction before {funcs['abi_surface']} execution",
            comparability=_phase_comparability("planning"),
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
            "observed",
            forecast_duration,
            f"canonical state-aware execution-memory forecast through {funcs['abi_surface']}",
            comparability=_phase_comparability("execution_memory_forecast"),
        )
        execution_start = time.perf_counter_ns()
        if typed_surface:
            execute = funcs["execute_f32"] if result_dtype == DTYPE_F32 else funcs["execute_f64"]
            if execute is None:
                raise RuntimeError(
                    f"typed payload is missing the execute symbol for {profile}"
                )
            status = execute(matrix, ctypes.byref(protocol), ctypes.byref(result))
        else:
            status = funcs["execute"](matrix, ctypes.byref(protocol), ctypes.byref(result))
        execution_duration = time.perf_counter_ns() - execution_start
        if status != STATUS_OK or result.row_count != 1:
            raise RuntimeError(
                "canonical execute failed: "
                f"status={status}, rows={result.row_count}, "
                f"route={route.route_id}/{route.profile}, "
                f"result_dtype={result_dtype}, "
                f"capacity={result.capacity}, metric_count={result.metric_count}, "
                f"base_metrics={base.metric_ids.len}, base_chunks={base.chunk_count}"
            )
        phases["first_execution"] = _phase(
            "observed",
            execution_duration,
            f"first canonical {funcs['abi_surface']} execute call",
            comparability=_phase_comparability("first_execution"),
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
            comparability=_phase_comparability("first_result_materialization"),
        )
    finally:
        cleanup_start = time.perf_counter_ns()
        free_status = funcs["free"](matrix)
        cleanup_duration = time.perf_counter_ns() - cleanup_start
        if funcs["free_returns_status"] and free_status != STATUS_OK:
            raise RuntimeError(f"matrix free failed: status={free_status}")
        phases["explicit_cleanup"] = _phase(
            "observed",
            cleanup_duration,
            "canonical ABI matrix teardown through " + funcs["abi_surface"],
            comparability=_phase_comparability("explicit_cleanup"),
        )

    del runtime_lib
    del lib
    gc.collect()
    phases["process_exit_cleanup"] = _phase(
        "not_observable",
        None,
        "process exit and dynamic-loader teardown are outside the worker timing boundary",
        comparability=_phase_comparability("process_exit_cleanup"),
    )
    _validate_phase_comparability(phases)
    return {
        "schema": WORKER_SCHEMA,
        "status": "pass",
        "backend": backend,
        "profile": profile,
        "backend_kind": expected_kind,
        "abi_surface": funcs["abi_surface"],
        "route_source": funcs["route_source"],
        "route": {
            "route_id": int(route.route_id),
            "profile": int(route.profile),
            "storage_dtype": int(route.storage_dtype),
            "pointwise_dtype": int(route.pointwise_dtype),
            "reduction_dtype": int(route.reduction_dtype),
            "result_dtype": int(route.result_dtype),
            "route_synthesized": bool(capability_metadata.get("route_synthesized", False)),
        },
        "capability": capability_metadata,
        "workload": {"rows": rows, "cols": cols, "candidate_count": 1, "metric": "pearson"},
        "phases": phases,
        "phase_comparability": PHASE_COMPARABILITY,
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
                    comparability=_phase_comparability("process_exit_cleanup"),
                )
                _validate_phase_comparability(phases)
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
                "comparability": PHASE_COMPARABILITY[phase],
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
            "abi_surface_selection": "generic numeric-route symbols are preferred; frozen typed ABI 1.1 payloads are selected only when numeric_routes_v2 is absent",
            "first_capability_query": "generic route enumeration and typed capability-mask selection are recorded but not cross-surface comparable",
            "planning": "caller-side canonical protocol and result-buffer construction is timed separately",
            "first_result_materialization": "first typed host read of caller-owned result buffers is timed after execute; vendor D2H and synchronization remain inside execute and are not separately observable",
            "process_exit_cleanup": "parent-minus-worker residual is retained as a combined startup, provenance/serialization, and process-exit boundary; it is not labeled pure teardown",
        },
        "phase_comparability": PHASE_COMPARABILITY,
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
