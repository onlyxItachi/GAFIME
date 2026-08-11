#!/usr/bin/env python3
"""Measure the observable cold lifecycle of a GAFIME ABI 1.1 payload.

This benchmark is intentionally separate from the public precision benchmark.
Each profile sample runs in a new process and calls the canonical ABI directly.
Current payloads use the generic numeric-route ABI; the exact historical
pre-freeze ABI 1.1 baseline uses the typed precision fallback.  Both paths
retain honest boundaries for Python import, payload discovery, dynamic loading, capability
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
import zipfile


SCHEMA = "gafime.cold-lifecycle.v1"
WORKER_SCHEMA = "gafime.cold-lifecycle.worker.v1"
COMPARISON_MANIFEST_SCHEMA = "gafime.cold-lifecycle-comparison-manifest.v1"
COMPARISON_SCHEMA = "gafime.cold-lifecycle-comparison.v1"
ABI_VERSION = (1 << 16) | 1
ABI_IGNORABLE_FLAG_MASK = 0xFFFF0000
ABI_REQUIRED_FLAG_MASK = 0x0000FFFF
BACKEND_KINDS = {"cuda": 2, "rocm": 3, "metal": 4}
WHEEL_PAYLOAD_MEMBERS = {
    "cuda": (
        "gafime_cuda/libgafime_cuda.so",
        "gafime_cuda/gafime_cuda.dll",
    ),
    "rocm": ("gafime_rocm/libgafime_rocm.so",),
    "metal": ("gafime/_metal/libgafime_metal_v1.dylib",),
}
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

# These phases have an honest duration, but that duration is necessarily
# shared with a wider observable boundary.  They still require one raw sample
# for every fresh worker; ``observed_combined`` is not permission to omit the
# measurement.
COMBINED_SAMPLE_PHASES = frozenset(
    {
        "code_object_or_module_registration",
        "process_exit_cleanup",
    }
)

# Metal exposes no separate public C runtime/context initialization operation.
# Keep that one backend limitation explicit and counted rather than silently
# accepting a missing distribution.  CUDA and ROCm must measure this phase.
OPTIONAL_UNOBSERVED_PHASES_BY_BACKEND = {
    "metal": frozenset({"runtime_context_initialization"}),
}

# These labels are deliberately about comparing the generic numeric-route
# surface with the historical pre-freeze typed baseline.  A phase can still be
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
        "detail": "same matrix teardown, but the historical typed free is void while generic free returns status",
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


class _RouteRecord(ctypes.Structure):
    """Caller-owned route record with room for a future ABI 1.2 tail."""

    _fields_ = [
        ("known", _Route),
        ("future_fields", ctypes.c_uint64 * 2),
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
    """Historical pre-freeze pre-route ABI 1.1 typed matrix descriptor."""

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
    """Historical pre-freeze pre-route ABI 1.1 profile capability record."""

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
    """Historical pre-freeze pre-route ABI 1.1 typed protocol wrapper."""

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
        _RouteRecord: (120, 8),
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


def _validated_source_provenance(source_root: str | None) -> dict[str, object]:
    provenance = _source_provenance(source_root)
    commit = provenance.get("commit")
    git_status = provenance.get("git_status")
    if (
        provenance.get("status") == "not_supplied"
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(git_status, Mapping)
        or git_status.get("status") != "clean"
    ):
        raise ValueError(
            "cold lifecycle evidence requires a clean source tree at a full lowercase commit"
        )
    return provenance


def _wheel_payload_binding(
    backend: str, payload: str | Path, wheel: str | Path | None
) -> dict[str, object]:
    if wheel is None:
        raise ValueError("cold lifecycle evidence requires the exact wheel")
    payload_identity = _identity(payload)
    wheel_identity = _identity(wheel)
    allowed_members = WHEEL_PAYLOAD_MEMBERS[backend]
    try:
        with zipfile.ZipFile(wheel_identity["path"]) as archive:
            present = [
                member for member in allowed_members if member in archive.namelist()
            ]
            if len(present) != 1:
                raise ValueError(
                    "wheel must contain exactly one expected backend payload member: "
                    f"observed={present}"
                )
            member = present[0]
            member_bytes = archive.read(member)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"cannot inspect exact wheel payload: {exc}") from exc
    member_sha256 = hashlib.sha256(member_bytes).hexdigest()
    if (
        member_sha256 != payload_identity["sha256"]
        or len(member_bytes) != payload_identity["size_bytes"]
    ):
        raise ValueError("payload bytes do not match the exact wheel member")
    return {
        "status": "verified",
        "member": member,
        "member_size_bytes": len(member_bytes),
        "member_sha256": member_sha256,
        "payload": payload_identity,
        "wheel": wheel_identity,
    }


def _process_affinity() -> dict[str, object]:
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        return {"status": "observed", "cpus": cpus}
    return {
        "status": "unavailable",
        "detail": "the platform does not expose os.sched_getaffinity",
    }


def _amd_sysfs_identity(root: Path | None = None) -> dict[str, object]:
    """Read a stable AMD GPU/driver identity when ROCm SMI is unavailable."""

    system_root = root is None
    root = Path("/sys/class/drm") if root is None else root
    if not root.is_dir():
        return {"status": "unavailable", "output": "Linux DRM sysfs is unavailable"}
    cards: list[dict[str, object]] = []
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device"
        try:
            if (device / "vendor").read_text().strip().lower() != "0x1002":
                continue
            record: dict[str, object] = {
                "card": card.name,
                "device": (device / "device").read_text().strip(),
                "uevent": sorted((device / "uevent").read_text().splitlines()),
            }
            for name in ("subsystem_vendor", "subsystem_device", "unique_id"):
                path = device / name
                if path.is_file():
                    value = path.read_text().strip()
                    if value:
                        record[name] = value
            cards.append(record)
        except OSError:
            continue
    if not cards:
        return {"status": "unavailable", "output": "no AMD DRM device was readable"}
    driver: dict[str, str] = {
        "name": "amdgpu",
        "kernel_release": platform.release(),
    }
    if system_root:
        module_root = Path("/sys/module/amdgpu")
        for name in ("version", "srcversion"):
            path = module_root / name
            try:
                value = path.read_text().strip()
            except OSError:
                continue
            if value:
                driver[name] = value
    return {
        "status": "pass",
        "output": json.dumps(
            {"devices": cards, "driver": driver},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "source": "linux_drm_sysfs",
    }


def _usable_rocm_smi_identity(result: Mapping[str, object]) -> bool:
    """Require both a concrete GPU record and driver identity from ROCm SMI."""

    if result.get("status") != "pass":
        return False
    output = str(result.get("output", "")).strip().lower()
    if not output:
        return False
    lines = tuple(line.strip() for line in output.splitlines() if line.strip())
    has_device = any("gpu[" in line for line in lines)
    has_driver = any("driver" in line and "version" in line for line in lines)
    return has_device and has_driver


def _device_identity(backend: str) -> dict[str, object]:
    if backend == "cuda":
        return _command(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ),
            timeout=60,
        )
    if backend == "rocm":
        result = _command(
            (
                "rocm-smi",
                "--showproductname",
                "--showuniqueid",
                "--showdriverversion",
            ),
            timeout=60,
        )
        if _usable_rocm_smi_identity(result):
            return result
        return _amd_sysfs_identity()
    return _command(
        (
            "system_profiler",
            "-json",
            "SPHardwareDataType",
            "SPDisplaysDataType",
        ),
        timeout=120,
    )


def _toolchain_identity(backend: str) -> dict[str, object]:
    if backend == "cuda":
        compiler = _command(("nvcc", "--version"), timeout=60)
    elif backend == "rocm":
        compiler = _command(("hipcc", "--version"), timeout=60)
    else:
        compiler = _command(("xcrun", "metal", "-v"), timeout=60)
    linker_command = (
        ("ld", "-v") if platform.system() == "Darwin" else ("ld", "--version")
    )
    return {
        "compiler": compiler,
        "linker": _command(linker_command, timeout=60),
    }


def _clock_and_power_state(backend: str) -> dict[str, object]:
    state: dict[str, object] = {
        "cpu_governor": _command(
            (
                "sh",
                "-c",
                "for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; "
                'do test -r "$f" && printf \'%s=%s\\n\' "$f" "$(cat "$f")"; done',
            )
        )
    }
    if backend == "cuda":
        state["accelerator"] = _command(
            (
                "nvidia-smi",
                "--query-gpu=pstate,clocks.current.graphics,clocks.current.memory,power.draw",
                "--format=csv,noheader",
            ),
            timeout=60,
        )
    elif backend == "rocm":
        state["accelerator"] = _command(
            ("rocm-smi", "--showclocks", "--showpower", "--showperflevel"),
            timeout=60,
        )
    else:
        state["accelerator"] = _command(("pmset", "-g", "therm"), timeout=60)
    return state


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
    if route.overflow_policy != 1 or route.flags & ABI_REQUIRED_FLAG_MASK:
        return False
    if route.struct_size >= ctypes.sizeof(_Route) and any(
        int(value) != 0 for value in route.reserved
    ):
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


def _known_route_id(route_id: int) -> bool:
    return route_id in PROFILE_IDS.values()


def _route_record_error(
    record: _RouteRecord,
    route_stride: int,
    seen_ids: set[int],
) -> tuple[_Route | None, str | None]:
    """Validate one enumerated record and normalize recognized routes.

    Only the stable route prefix is interpreted before the route ID is known.
    Unknown IDs are retained in the duplicate/structural checks but their
    profile, dtype, and overflow values are never dispatched.
    """

    route = record.known
    route_size = int(route.struct_size)
    if (
        route.abi_version >> 16 != ABI_VERSION >> 16
        or (route.abi_version & 0xFFFF) < (ABI_VERSION & 0xFFFF)
        or route_stride < int(_Route.reserved.offset)
        or route_size < int(_Route.reserved.offset)
        or int(route.route_id) == 0
        or int(route.flags) & ABI_REQUIRED_FLAG_MASK
    ):
        return None, f"invalid route prefix/id {int(route.route_id)}"
    if route_size >= ctypes.sizeof(_Route) and any(
        int(value) != 0 for value in route.reserved
    ):
        return None, f"nonzero reserved fields in route {int(route.route_id)}"
    route_id = int(route.route_id)
    if route_id in seen_ids:
        return None, f"duplicate route record {route_id}"
    seen_ids.add(route_id)
    if not _known_route_id(route_id):
        return None, None
    profile = next(name for name, value in PROFILE_IDS.items() if value == route_id)
    if not _route_ok(route, profile):
        return None, f"malformed {profile} numeric route"
    normalized = _Route()
    ctypes.memmove(ctypes.byref(normalized), ctypes.byref(route), ctypes.sizeof(_Route))
    normalized.struct_size = ctypes.sizeof(_Route)
    return normalized, None


def _collect_generic_routes(
    records: Sequence[_RouteRecord], expected_profiles: set[str]
) -> tuple[dict[int, _Route], int]:
    route_stride = ctypes.sizeof(_RouteRecord)
    seen_ids: set[int] = set()
    known: dict[int, _Route] = {}
    for record in records:
        route, error = _route_record_error(record, route_stride, seen_ids)
        if error is not None:
            raise RuntimeError(f"payload advertised {error}")
        if route is not None:
            known[int(route.route_id)] = route
    expected_ids = {PROFILE_IDS[profile] for profile in expected_profiles}
    if set(known) != expected_ids:
        raise RuntimeError(
            "payload did not advertise the expected known route set: "
            f"actual={sorted(known)} expected={sorted(expected_ids)}"
        )
    return known, sum(1 << (route_id - 1) for route_id in known)


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
            ("nvcudart_hybrid64.dll",) if os.name == "nt" else ("libcudart.so.13",),
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
            raise RuntimeError(
                f"payload is missing required ABI symbol {name}"
            ) from exc
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
    """Select the generic ABI or the historical pre-freeze typed baseline.

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
        funcs["permutation_memory"] = _bind_symbol(
            lib,
            "gafime_gpu_permutation_memory_peak_v2",
            [void_p, ctypes.POINTER(_NumericLaunch), c_u64, ctypes.POINTER(c_u64)],
            c_int,
        )
        # These operations are required for a canonical generic payload even
        # though the cold lifecycle itself does not invoke significance or
        # diagnostics. Bind them here so a partial surface fails during
        # symbol resolution, before any allocation is attempted.
        funcs["permutation"] = _bind_symbol(
            lib,
            "gafime_gpu_permutation_pvalues_v2",
            [void_p, ctypes.POINTER(_NumericLaunch), void_p],
            c_int,
        )
        funcs["diagnostics"] = _bind_symbol(
            lib,
            "gafime_gpu_interaction_diagnostics_v2",
            [void_p, void_p],
            c_int,
        )
        funcs["free"] = _bind_symbol(lib, "gafime_gpu_matrix_free_v2", [void_p], c_int)
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
        [
            void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            c_u64,
            c_uint,
        ],
        c_int,
        required=False,
    )
    funcs["upload_f64"] = _bind_symbol(
        lib,
        "gafime_gpu_matrix_upload_f64_v2",
        [
            void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            c_u64,
            c_uint,
        ],
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
    # The historical pre-freeze typed free is void and generation-neutral.
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
        route_stride = ctypes.sizeof(_RouteRecord)
        count = ctypes.c_uint32()
        status = int(
            funcs["routes"](0, ABI_VERSION, route_stride, None, 0, ctypes.byref(count))
        )
        if status in (STATUS_UNSUPPORTED_BACKEND, STATUS_DEVICE_ERROR):
            return (
                None,
                status,
                "payload route capability unavailable on this device",
                {},
            )
        if status != STATUS_OK:
            raise RuntimeError(
                f"numeric route count failed: status={status}, count={count.value}"
            )
        if count.value == 0 or count.value > 1024:
            raise RuntimeError(
                f"numeric route count is outside the supported bound: {count.value}"
            )
        routes = (_RouteRecord * count.value)()
        capacity = ctypes.c_uint32(count.value)
        status = int(
            funcs["routes"](
                0,
                ABI_VERSION,
                route_stride,
                ctypes.cast(routes, ctypes.POINTER(_Route)),
                capacity.value,
                ctypes.byref(capacity),
            )
        )
        if status != STATUS_OK:
            raise RuntimeError(f"numeric route enumeration failed: status={status}")
        if capacity.value > count.value:
            raise RuntimeError(
                f"numeric route enumeration returned {capacity.value} records above capacity {count.value}"
            )
        expected_profiles = {"fp32"} if backend == "metal" else set(PROFILE_ORDER)
        known, profile_mask = _collect_generic_routes(
            routes[: capacity.value], expected_profiles
        )
        route = known.get(PROFILE_IDS[profile])
        if route is None:
            raise RuntimeError(
                f"payload did not advertise the canonical {profile} route"
            )
        return (
            route,
            STATUS_OK,
            "ABI 1.1 numeric-route count, enumeration, validation, and selection",
            {
                "profile_mask": profile_mask,
            },
        )

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
    *,
    payload: Path,
    backend: str,
    profile: str,
    source_root: str | None,
    wheel: str | None,
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
    importlib.import_module("gafime")
    import_duration = time.perf_counter_ns() - import_start
    phases["python_import"] = _phase(
        "observed",
        import_duration,
        "import gafime and its Python/native extension dependencies",
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

    phases["runtime_context_initialization"], runtime_lib = _initialize_runtime_context(
        backend
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
            "provenance": _provenance(payload, backend, source_root, wheel),
        }

    rows, cols = 4, 2
    if route.storage_dtype == DTYPE_F32:
        feature_array = (ctypes.c_float * (rows * cols))(
            1.0, 7.0, 2.0, 5.0, 3.0, 3.0, 4.0, 1.0
        )
        target_array = (ctypes.c_float * rows)(1.0, 2.0, 3.0, 4.0)
    else:
        epsilon = 1.0 / 1073741824.0
        feature_array = (ctypes.c_double * (rows * cols))(
            1.0 + epsilon,
            7.0,
            2.0 + epsilon,
            5.0,
            3.0 + epsilon,
            3.0,
            4.0 + epsilon,
            1.0,
        )
        target_array = (ctypes.c_double * rows)(
            1.0 + epsilon, 2.0 + epsilon, 3.0 + epsilon, 4.0 + epsilon
        )
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
            upload = (
                funcs["upload_f32"]
                if route.storage_dtype == DTYPE_F32
                else funcs["upload_f64"]
            )
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
            execute = (
                funcs["execute_f32"]
                if result_dtype == DTYPE_F32
                else funcs["execute_f64"]
            )
            if execute is None:
                raise RuntimeError(
                    f"typed payload is missing the execute symbol for {profile}"
                )
            status = execute(matrix, ctypes.byref(protocol), ctypes.byref(result))
        else:
            status = funcs["execute"](
                matrix, ctypes.byref(protocol), ctypes.byref(result)
            )
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
            "route_synthesized": bool(
                capability_metadata.get("route_synthesized", False)
            ),
        },
        "capability": capability_metadata,
        "workload": {
            "rows": rows,
            "cols": cols,
            "candidate_count": 1,
            "metric": "pearson",
        },
        "phases": phases,
        "phase_comparability": PHASE_COMPARABILITY,
        "worker_duration_ns": time.perf_counter_ns() - worker_start,
        "payload_discovered": {
            str(key): str(value) for key, value in discovered.items()
        },
        "provenance": _provenance(payload, backend, source_root, wheel),
    }


def _provenance(
    payload: Path,
    backend: str,
    source_root: str | None,
    wheel: str | None,
) -> dict[str, object]:
    binding = _wheel_payload_binding(backend, payload, wheel)
    result: dict[str, object] = {
        "payload": binding["payload"],
        "benchmark_script": _identity(Path(__file__).resolve()),
        "source": _validated_source_provenance(source_root),
        "payload_wheel_binding": binding,
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
    result["wheel"] = binding["wheel"]
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


def _stats(
    values: Sequence[int], seed: int, bootstrap_resamples: int
) -> dict[str, object]:
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
        "bootstrap_median_95_ci_ns": [
            _percentile(boot, 0.025),
            _percentile(boot, 0.975),
        ],
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
    profiles = tuple(
        dict.fromkeys(
            part.strip().lower()
            for raw in args.profile
            for part in raw.split(",")
            if part.strip()
        )
    )
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
    source_root = (
        str(Path(args.source_root).expanduser().resolve()) if args.source_root else None
    )
    wheel = (
        str(Path(args.wheel).expanduser().resolve(strict=True)) if args.wheel else None
    )
    driver_provenance = _provenance(payload, backend, source_root, wheel)
    driver_provenance.update(
        {
            "device": _device_identity(backend),
            "toolchain": _toolchain_identity(backend),
            "process_affinity": _process_affinity(),
            "clock_and_power_state": {"before": _clock_and_power_state(backend)},
        }
    )
    variant = str(args.variant or "").strip()
    variant_sequence = tuple(
        part.strip()
        for part in str(args.variant_sequence or "").split(",")
        if part.strip()
    )
    schedule_requested = bool(variant or variant_sequence or args.ab_block is not None)
    if schedule_requested and (
        variant not in {"baseline", "candidate"}
        or set(variant_sequence) != {"baseline", "candidate"}
        or len(variant_sequence) != 2
        or variant not in variant_sequence
        or args.ab_block not in {0, 1}
    ):
        raise ValueError(
            "comparative evidence requires --variant baseline|candidate, "
            "--ab-block 0|1, and a two-variant --variant-sequence"
        )
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
            if completed.returncode != 0 or record.get("status") not in {
                "pass",
                "unavailable",
            }:
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
            status_counts[(profile, str(phase_name), status)] = (
                status_counts.get((profile, str(phase_name), status), 0) + 1
            )
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
    driver_provenance["clock_and_power_state"]["after"] = _clock_and_power_state(
        backend
    )
    return {
        "schema": SCHEMA,
        "status": "pass"
        if all(record.get("status") == "pass" for record in records)
        else "partial",
        "backend": backend,
        "variant": variant or None,
        "ab_block": args.ab_block,
        "variant_sequence": list(variant_sequence),
        "process_isolation": "fresh_worker_process_per_profile_sample",
        "profiles": list(profiles),
        "repetitions_per_profile": args.repetitions,
        "fresh_subprocess_per_sample": True,
        "profile_orders": [list(order) for order in all_orders],
        "profile_order_counts": {
            "/".join(order): sum(
                1
                for record in records
                if tuple(record.get("profile_order", ())) == order
            )
            // len(profiles)
            for order in all_orders
        },
        "workload": {"rows": 4, "cols": 2, "candidate_count": 1, "metric": "pearson"},
        "driver_command": sys.argv,
        "driver_duration_ns": time.perf_counter_ns() - driver_start,
        "seed": args.seed,
        "provenance": driver_provenance,
        "records": records,
        "phase_summaries": summaries,
        "phase_boundary_policy": {
            "runtime_context_initialization": "first explicit cudaFree(0) or hipFree(0) is timed after payload loading; Metal has no separate runtime C boundary",
            "code_object_or_module_registration": "observed_combined with dynamic-library load because registration runs in loader constructors",
            "abi_surface_selection": "generic numeric-route symbols are preferred; the historical pre-freeze typed ABI 1.1 baseline is selected only when numeric_routes_v2 is absent",
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


def _bootstrap_delta_percent(
    baseline: Sequence[int],
    candidate: Sequence[int],
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    if len(baseline) < 30 or len(candidate) < 30:
        raise ValueError("cold A/B comparisons require at least 30 samples per variant")
    baseline_values = [float(value) for value in baseline]
    candidate_values = [float(value) for value in candidate]
    baseline_median = float(statistics.median(baseline_values))
    candidate_median = float(statistics.median(candidate_values))
    if baseline_median <= 0.0:
        raise ValueError("cold A/B baseline median must be positive")
    delta = (candidate_median / baseline_median - 1.0) * 100.0
    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(resamples):
        baseline_sample = [rng.choice(baseline_values) for _ in baseline_values]
        candidate_sample = [rng.choice(candidate_values) for _ in candidate_values]
        baseline_center = float(statistics.median(baseline_sample))
        candidate_center = float(statistics.median(candidate_sample))
        bootstrap.append((candidate_center / baseline_center - 1.0) * 100.0)
    interval = [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
    if interval[0] > 3.0:
        classification = "confirmed_regression_above_three_percent"
    elif interval[0] > 1.0:
        classification = "confirmed_regression_above_one_percent"
    elif interval[1] < 0.0:
        classification = "confirmed_improvement"
    elif interval[0] > 0.0:
        classification = "confirmed_regression_within_one_percent"
    else:
        classification = "inconclusive_ci_crosses_zero"
    return {
        "baseline_median_ns": baseline_median,
        "candidate_median_ns": candidate_median,
        "candidate_latency_delta_percent": delta,
        "bootstrap_candidate_latency_delta_95_ci_percent": interval,
        "review_status": classification,
        "baseline_samples_ns": list(map(int, baseline)),
        "candidate_samples_ns": list(map(int, candidate)),
    }


def _summary_samples(summary: object) -> list[int]:
    if not isinstance(summary, Mapping):
        return []
    for key in ("observed", "observed_combined"):
        distribution = summary.get(key)
        if not isinstance(distribution, Mapping):
            continue
        raw = distribution.get("raw_duration_ns")
        if isinstance(raw, list) and all(
            isinstance(value, int) and value > 0 for value in raw
        ):
            return list(raw)
    return []


def _phase_completeness_failures(
    artifacts: Mapping[tuple[int, str], Mapping[str, object]],
    *,
    backend: str,
    profiles: Sequence[str],
) -> list[dict[str, object]]:
    """Require the declared fresh-process count for every canonical phase.

    Comparison artifacts are untrusted inputs.  A top-level
    ``repetitions_per_profile=30`` declaration cannot stand in for raw phase
    samples.  Every required phase therefore needs exactly the declared count
    in its expected bucket.  The sole current backend exception is Metal's
    separately unobservable runtime/context boundary; even that exception must
    carry one explicit diagnostic status per worker.
    """

    failures: list[dict[str, object]] = []
    optional = OPTIONAL_UNOBSERVED_PHASES_BY_BACKEND.get(backend, frozenset())
    for (block, variant), payload in sorted(artifacts.items()):
        repetitions = int(payload.get("repetitions_per_profile", 0))
        summaries = payload.get("phase_summaries")
        if not isinstance(summaries, Mapping):
            failures.append(
                {
                    "ab_block": block,
                    "variant": variant,
                    "reason": "canonical_phase_summaries_missing",
                }
            )
            continue
        for profile in profiles:
            profile_summaries = summaries.get(profile)
            if not isinstance(profile_summaries, Mapping):
                failures.append(
                    {
                        "ab_block": block,
                        "variant": variant,
                        "profile": profile,
                        "reason": "canonical_profile_phase_summaries_missing",
                    }
                )
                continue
            for phase in REQUIRED_PHASES:
                summary = profile_summaries.get(phase)
                if not isinstance(summary, Mapping):
                    failures.append(
                        {
                            "ab_block": block,
                            "variant": variant,
                            "profile": profile,
                            "phase": phase,
                            "reason": "canonical_phase_summary_missing",
                        }
                    )
                    continue
                if summary.get("comparability") != PHASE_COMPARABILITY[phase]:
                    failures.append(
                        {
                            "ab_block": block,
                            "variant": variant,
                            "profile": profile,
                            "phase": phase,
                            "reason": "canonical_phase_comparability_mismatch",
                        }
                    )
                status_counts = summary.get("status_counts")
                if not isinstance(status_counts, Mapping):
                    status_counts = {}
                normalized_status_counts = {
                    str(status): int(count)
                    for status, count in status_counts.items()
                    if isinstance(count, int) and not isinstance(count, bool)
                }
                if phase in optional:
                    expected_status = "not_observable"
                    expected_bucket = None
                elif phase in COMBINED_SAMPLE_PHASES:
                    expected_status = "observed_combined"
                    expected_bucket = "observed_combined"
                else:
                    expected_status = "observed"
                    expected_bucket = "observed"
                if (
                    normalized_status_counts.get(expected_status) != repetitions
                    or sum(normalized_status_counts.values()) != repetitions
                ):
                    failures.append(
                        {
                            "ab_block": block,
                            "variant": variant,
                            "profile": profile,
                            "phase": phase,
                            "reason": "canonical_phase_status_count_mismatch",
                            "expected_status": expected_status,
                            "expected_count": repetitions,
                            "observed": normalized_status_counts,
                        }
                    )
                if expected_bucket is None:
                    if _summary_samples(summary):
                        failures.append(
                            {
                                "ab_block": block,
                                "variant": variant,
                                "profile": profile,
                                "phase": phase,
                                "reason": "optional_phase_must_remain_diagnostic_only",
                            }
                        )
                    continue
                distribution = summary.get(expected_bucket)
                raw = (
                    distribution.get("raw_duration_ns")
                    if isinstance(distribution, Mapping)
                    else None
                )
                declared_count = (
                    distribution.get("count")
                    if isinstance(distribution, Mapping)
                    else None
                )
                if (
                    not isinstance(raw, list)
                    or len(raw) != repetitions
                    or declared_count != repetitions
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value <= 0
                        for value in raw
                    )
                ):
                    failures.append(
                        {
                            "ab_block": block,
                            "variant": variant,
                            "profile": profile,
                            "phase": phase,
                            "reason": "canonical_phase_sample_count_mismatch",
                            "expected_count": repetitions,
                            "observed_count": len(raw)
                            if isinstance(raw, list)
                            else None,
                            "declared_count": declared_count,
                            "sample_bucket": expected_bucket,
                        }
                    )
    return failures


def _load_comparison_artifact(
    manifest_root: Path, item: object
) -> tuple[Path, dict[str, object]]:
    if not isinstance(item, Mapping):
        raise ValueError("cold comparison artifact entries must be objects")
    raw_path = item.get("path")
    digest = item.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("cold comparison artifact path is required")
    artifact_path = Path(raw_path)
    if not artifact_path.is_absolute():
        artifact_path = manifest_root / artifact_path
    artifact_path = artifact_path.resolve(strict=True)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or _sha256(artifact_path) != digest.lower()
    ):
        raise ValueError(f"cold comparison artifact hash mismatch: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cold comparison artifact must contain a JSON object")
    for key in ("variant", "ab_block", "variant_sequence"):
        if payload.get(key) != item.get(key):
            raise ValueError(f"cold comparison manifest/artifact {key} mismatch")
    return artifact_path, payload


def _cold_comparison(
    manifest_path: Path, *, seed: int, resamples: int
) -> dict[str, object]:
    resolved_manifest = manifest_path.expanduser().resolve(strict=True)
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != COMPARISON_MANIFEST_SCHEMA
    ):
        raise ValueError("invalid cold lifecycle comparison manifest schema")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 4:
        raise ValueError("cold comparison requires exactly four A/B plus B/A artifacts")
    artifacts: dict[tuple[int, str], dict[str, object]] = {}
    artifact_identities: list[dict[str, object]] = []
    for item in raw_artifacts:
        path, payload = _load_comparison_artifact(resolved_manifest.parent, item)
        if payload.get("schema") != SCHEMA or payload.get("status") != "pass":
            raise ValueError(f"cold lifecycle artifact is not passing: {path}")
        variant = str(payload.get("variant"))
        block = payload.get("ab_block")
        sequence = payload.get("variant_sequence")
        if (
            variant not in {"baseline", "candidate"}
            or block not in {0, 1}
            or not isinstance(sequence, list)
            or len(sequence) != 2
            or set(map(str, sequence)) != {"baseline", "candidate"}
            or variant not in sequence
            or payload.get("process_isolation")
            != "fresh_worker_process_per_profile_sample"
            or payload.get("fresh_subprocess_per_sample") is not True
            or int(payload.get("repetitions_per_profile", 0)) < 30
        ):
            raise ValueError(f"invalid cold lifecycle A/B schedule metadata: {path}")
        key = (int(block), variant)
        if key in artifacts:
            raise ValueError(f"duplicate cold lifecycle A/B cell: {key}")
        artifacts[key] = payload
        artifact_identities.append(_identity(path))
    if set(artifacts) != {
        (0, "baseline"),
        (0, "candidate"),
        (1, "baseline"),
        (1, "candidate"),
    }:
        raise ValueError("cold comparison is missing an A/B or B/A artifact")
    sequences = {
        block: tuple(map(str, artifacts[(block, "baseline")]["variant_sequence"]))
        for block in (0, 1)
    }
    for block in (0, 1):
        if (
            tuple(artifacts[(block, "candidate")]["variant_sequence"])
            != sequences[block]
        ):
            raise ValueError("variants in one cold A/B block disagree on sequence")
    if sequences[1] != tuple(reversed(sequences[0])):
        raise ValueError("cold lifecycle block 1 must reverse block 0")

    reference = artifacts[(0, "baseline")]
    backend = str(reference.get("backend"))
    profiles = tuple(map(str, reference.get("profiles", ())))
    stable_fields = ("backend", "profiles", "workload", "phase_comparability")
    for payload in artifacts.values():
        if any(payload.get(field) != reference.get(field) for field in stable_fields):
            raise ValueError("cold A/B artifacts disagree on backend/profile/workload")
    benchmark_hashes: set[str] = set()
    device_identities: set[str] = set()
    toolchain_identities: set[str] = set()
    product_commits: dict[str, set[str]] = {"baseline": set(), "candidate": set()}
    variant_payloads: dict[str, set[str]] = {"baseline": set(), "candidate": set()}
    variant_wheels: dict[str, set[str]] = {"baseline": set(), "candidate": set()}
    for (_block, variant), payload in artifacts.items():
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("cold artifact provenance is required")
        source = provenance.get("source")
        binding = provenance.get("payload_wheel_binding")
        benchmark_identity = provenance.get("benchmark_script")
        payload_identity = provenance.get("payload")
        wheel_identity = provenance.get("wheel")
        device_identity = provenance.get("device")
        toolchain_identity = provenance.get("toolchain")
        affinity = provenance.get("process_affinity")
        clock_and_power = provenance.get("clock_and_power_state")
        source_commit = source.get("commit") if isinstance(source, Mapping) else None
        if (
            not isinstance(source, Mapping)
            or not isinstance(source.get("git_status"), Mapping)
            or source["git_status"].get("status") != "clean"
            or not isinstance(source_commit, str)
            or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
            or not isinstance(binding, Mapping)
            or binding.get("status") != "verified"
            or not isinstance(benchmark_identity, Mapping)
            or not isinstance(benchmark_identity.get("sha256"), str)
            or not isinstance(payload_identity, Mapping)
            or not isinstance(payload_identity.get("sha256"), str)
            or not isinstance(wheel_identity, Mapping)
            or not isinstance(wheel_identity.get("sha256"), str)
            or not isinstance(device_identity, Mapping)
            or device_identity.get("status") != "pass"
            or not device_identity.get("output")
            or not isinstance(toolchain_identity, Mapping)
            or not isinstance(toolchain_identity.get("compiler"), Mapping)
            or toolchain_identity["compiler"].get("status") != "pass"
            or not isinstance(toolchain_identity.get("linker"), Mapping)
            or toolchain_identity["linker"].get("status") != "pass"
            or not isinstance(affinity, Mapping)
            or not isinstance(clock_and_power, Mapping)
            or not isinstance(clock_and_power.get("before"), Mapping)
            or not isinstance(clock_and_power.get("after"), Mapping)
        ):
            raise ValueError(
                "cold artifact lacks clean source, wheel binding, or hardware controls"
            )
        benchmark_hashes.add(str(benchmark_identity["sha256"]))
        device_identities.add(json.dumps(device_identity, sort_keys=True))
        toolchain_identities.add(json.dumps(toolchain_identity, sort_keys=True))
        product_commits[variant].add(source_commit)
        variant_payloads[variant].add(str(payload_identity["sha256"]))
        variant_wheels[variant].add(str(wheel_identity["sha256"]))
    if (
        len(benchmark_hashes) != 1
        or len(device_identities) != 1
        or len(toolchain_identities) != 1
    ):
        raise ValueError(
            "cold A/B variants must share one benchmark, toolchain, and physical device"
        )
    if any(len(values) != 1 for values in product_commits.values()):
        raise ValueError("cold A/B product commit changed between blocks")
    if product_commits["baseline"] == product_commits["candidate"]:
        raise ValueError("cold A/B requires distinct baseline and candidate commits")
    if any(len(values) != 1 for values in variant_payloads.values()) or any(
        len(values) != 1 for values in variant_wheels.values()
    ):
        raise ValueError("cold A/B payload or wheel changed between blocks")

    comparisons: list[dict[str, object]] = []
    failures = _phase_completeness_failures(
        artifacts,
        backend=backend,
        profiles=profiles,
    )
    for profile in profiles:
        for phase in REQUIRED_PHASES:
            block_results: list[dict[str, object]] = []
            aggregate_baseline: list[int] = []
            aggregate_candidate: list[int] = []
            for block in (0, 1):
                try:
                    baseline_summary = artifacts[(block, "baseline")][
                        "phase_summaries"
                    ][profile][phase]
                    candidate_summary = artifacts[(block, "candidate")][
                        "phase_summaries"
                    ][profile][phase]
                except (KeyError, TypeError):
                    # The completeness gate above records the precise artifact,
                    # profile, and phase.  Keep producing a fail-closed report
                    # instead of aborting before the caller can inspect it.
                    continue
                baseline_samples = _summary_samples(baseline_summary)
                candidate_samples = _summary_samples(candidate_summary)
                if not baseline_samples and not candidate_samples:
                    continue
                if not baseline_samples or not candidate_samples:
                    failures.append(
                        {
                            "profile": profile,
                            "phase": phase,
                            "reason": "one_variant_phase_missing",
                            "ab_block": block,
                        }
                    )
                    continue
                baseline_expected = int(
                    artifacts[(block, "baseline")]["repetitions_per_profile"]
                )
                candidate_expected = int(
                    artifacts[(block, "candidate")]["repetitions_per_profile"]
                )
                if (
                    len(baseline_samples) != baseline_expected
                    or len(candidate_samples) != candidate_expected
                ):
                    # Completeness failures already identify the malformed
                    # artifact.  Do not feed a partial distribution into the
                    # release comparison or let the bootstrap helper abort the
                    # fail-closed report.
                    continue
                result = _bootstrap_delta_percent(
                    baseline_samples,
                    candidate_samples,
                    seed=_stable_seed(seed, backend, profile, phase, block),
                    resamples=resamples,
                )
                result["ab_block"] = block
                result["variant_sequence"] = list(sequences[block])
                block_results.append(result)
                aggregate_baseline.extend(baseline_samples)
                aggregate_candidate.extend(candidate_samples)
            comparability = str(PHASE_COMPARABILITY[phase]["status"])
            if not block_results:
                comparisons.append(
                    {
                        "profile": profile,
                        "phase": phase,
                        "comparability": comparability,
                        "status": "not_observed_on_either_variant",
                    }
                )
                continue
            aggregate = _bootstrap_delta_percent(
                aggregate_baseline,
                aggregate_candidate,
                seed=_stable_seed(seed, backend, profile, phase, "aggregate"),
                resamples=resamples,
            )
            deltas = [
                float(result["candidate_latency_delta_percent"])
                for result in block_results
            ]
            repeated_above_three = len(block_results) == 2 and all(
                result["bootstrap_candidate_latency_delta_95_ci_percent"][0] > 3.0
                for result in block_results
            )
            repeated_above_one = len(block_results) == 2 and all(
                result["bootstrap_candidate_latency_delta_95_ci_percent"][0] > 1.0
                for result in block_results
            )
            order_sensitive = len(deltas) == 2 and (
                max(deltas) - min(deltas) > 1.0
                or (deltas[0] > 1.0 and deltas[1] < -1.0)
                or (deltas[1] > 1.0 and deltas[0] < -1.0)
            )
            gate_eligible = comparability in {"direct", "semantic_only"}
            if gate_eligible and (repeated_above_one or order_sensitive):
                failures.append(
                    {
                        "profile": profile,
                        "phase": phase,
                        "reason": (
                            "repeatable_regression_above_three_percent"
                            if repeated_above_three
                            else "repeatable_regression_above_one_percent"
                            if repeated_above_one
                            else "ab_ba_order_sensitivity_above_one_percent"
                        ),
                        "block_deltas_percent": deltas,
                    }
                )
            comparisons.append(
                {
                    "profile": profile,
                    "phase": phase,
                    "comparability": comparability,
                    "block_comparisons": block_results,
                    "aggregate": aggregate,
                    "repeatable_regression_above_one_percent": repeated_above_one,
                    "repeatable_regression_above_three_percent": repeated_above_three,
                    "ab_ba_order_sensitive_above_one_percent": order_sensitive,
                }
            )
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "pass" if not failures else "regression_or_contamination_detected",
        "valid_for_canonical_cold_lifecycle_claims": not failures,
        "backend": backend,
        "profiles": list(profiles),
        "schedule": {
            "blocks": [
                {"ab_block": block, "variant_sequence": list(sequences[block])}
                for block in (0, 1)
            ],
            "process_isolation": "fresh_worker_process_per_profile_sample",
        },
        "provenance": {
            "manifest": _identity(resolved_manifest),
            "benchmark_script_sha256": next(iter(benchmark_hashes)),
            "baseline_commit": next(iter(product_commits["baseline"])),
            "candidate_commit": next(iter(product_commits["candidate"])),
            "artifacts": artifact_identities,
        },
        "comparisons": comparisons,
        "failures": failures,
        "policy": (
            "A/B and reversed B/A blocks require the declared fresh-process sample "
            "count in every canonical phase for both variants; Metal runtime/context "
            "initialization is the only current diagnostic-only phase. Repeatable "
            "regressions above one percent and order sensitivity above one percent "
            "block claims, while repeatable regressions above three percent are "
            "release-ineligible"
        ),
    }


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
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--ab-block", type=int)
    parser.add_argument("--variant-sequence")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("-"))
    parser.add_argument("--compare-manifest", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_check:
        return args
    if args.compare_manifest:
        if args._worker or args.backend or args.payload:
            parser.error("--compare-manifest is a standalone comparison mode")
        if args.bootstrap_resamples < 500:
            parser.error("--bootstrap-resamples must be at least 500")
        return args
    if args._worker:
        if (
            not args.backend
            or not args.payload
            or len(args.profile) != 1
            or not args.source_root
            or not args.wheel
        ):
            parser.error(
                "worker requires --backend, --payload, --source-root, --wheel and one --profile"
            )
        return args
    if not args.backend or not args.payload or not args.source_root or not args.wheel:
        parser.error("driver requires --backend, --payload, --source-root and --wheel")
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
        if args.compare_manifest:
            record = _cold_comparison(
                args.compare_manifest,
                seed=args.seed,
                resamples=args.bootstrap_resamples,
            )
        elif args._worker:
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
    except (
        OSError,
        RuntimeError,
        ValueError,
        ctypes.ArgumentError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"cold lifecycle benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
