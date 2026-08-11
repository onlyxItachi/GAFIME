#!/usr/bin/env python3
"""Emit hash-bound ABI 1.1 payload lifecycle evidence.

The producer executes a surface-specific standalone C consumer against the
payload bytes embedded in an exact wheel. It does not import GAFIME or rely on
the repository's Rust loader.  The preserved typed surface is historical A/B
evidence; only the generic numeric-route surface is the candidate canonical
ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile


BACKENDS = {
    "cuda": {
        "kind": 2,
        "profiles": ("fp32", "mixed", "fp64"),
        "route_count": 3,
        "members": (
            "gafime_cuda/libgafime_cuda.so",
            "gafime_cuda/gafime_cuda.dll",
        ),
    },
    "rocm": {
        "kind": 3,
        "profiles": ("fp32", "mixed", "fp64"),
        "route_count": 3,
        "members": ("gafime_rocm/libgafime_rocm.so",),
    },
    "metal": {
        "kind": 4,
        "profiles": ("fp32",),
        "route_count": 1,
        "members": ("gafime/_metal/libgafime_metal_v1.dylib",),
    },
}

REQUIRED_OPERATIONS = (
    "numeric_routes",
    "matrix_alloc",
    "matrix_upload",
    "matrix_update_target",
    "execute",
    "execution_memory_peak",
    "permutation_memory_peak",
    "permutation_pvalues",
    "interaction_diagnostics",
    "matrix_free",
)

RESULT_SCHEMA = "gafime.abi-1.1-consumer-result.v1"
TYPED_RESULT_SCHEMA = "gafime.abi-1.1-typed-consumer-result.v1"
EVIDENCE_SCHEMA = "gafime.native-decomposition.v1"

# ABI 1.1 has two deliberately different public surfaces while the generic
# numeric-route ABI is still being introduced.  Keep the surface explicit in
# the evidence rather than inferring it from a missing symbol.  The typed
# surface is the historical pre-freeze contract used by the exact PR-70 baseline;
# it is not a fourth execution profile.
ABI_SURFACES = {
    "numeric-route-v2": {
        "contract_role": "candidate_canonical_numeric_route",
        "result_schema": RESULT_SCHEMA,
        "execution_layer": "independent_abi_1_1_c_consumer",
        "consumer_source": "tests/gpu/abi_consumers/abi_1_1_c_consumer.c",
        "operations": REQUIRED_OPERATIONS,
        "symbols": (
            "gafime_gpu_numeric_routes_v2",
            "gafime_gpu_matrix_alloc_v2",
            "gafime_gpu_matrix_upload_v2",
            "gafime_gpu_matrix_update_target_v2",
            "gafime_gpu_execute_v2",
            "gafime_gpu_execution_memory_peak_v2",
            "gafime_gpu_permutation_memory_peak_v2",
            "gafime_gpu_permutation_pvalues_v2",
            "gafime_gpu_interaction_diagnostics_v2",
            "gafime_gpu_matrix_free_v2",
        ),
    },
    "precision-typed-v1.1": {
        "contract_role": "historical_pre_freeze_typed_baseline",
        "result_schema": TYPED_RESULT_SCHEMA,
        "execution_layer": "independent_abi_1_1_typed_c_consumer",
        "consumer_source": "tests/gpu/abi_consumers/abi_1_1_typed_c_consumer.c",
        # The historical typed baseline has separate f32/f64 convenience entry points.
        # These names are semantic lifecycle operations, not generic-route
        # symbols, so cross-surface comparisons never treat their wrapper cost
        # as arithmetic cost.
        "operations": (
            "precision_capabilities",
            "matrix_alloc",
            "matrix_upload",
            "matrix_update_target",
            "execute",
            "execution_memory_peak",
            "interaction_diagnostics",
            "matrix_free",
        ),
        "symbols": (
            "gafime_gpu_precision_capabilities",
            "gafime_gpu_matrix_alloc_v2",
            "gafime_gpu_matrix_upload_f32_v2",
            "gafime_gpu_matrix_upload_f64_v2",
            "gafime_gpu_matrix_update_target_f32_v2",
            "gafime_gpu_matrix_update_target_f64_v2",
            "gafime_gpu_execute_f32_v2",
            "gafime_gpu_execute_f64_v2",
            "gafime_gpu_execution_memory_peak_v2",
            "gafime_gpu_interaction_diagnostics",
            "gafime_gpu_matrix_free",
        ),
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_bytes(resolved.read_bytes()),
    }


def _git_output(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(source_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _consumer_result(stdout: str, *, schema: str) -> dict[str, object]:
    objects: list[dict[str, object]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == schema:
            objects.append(value)
    if len(objects) != 1:
        raise RuntimeError(
            "ABI 1.1 consumer must emit exactly one structured success record"
        )
    return objects[0]


def _wheel_payload(wheel: Path, allowed_members: tuple[str, ...]) -> tuple[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        observed = [member for member in allowed_members if member in archive.namelist()]
        if len(observed) != 1:
            raise RuntimeError(
                "wheel must contain exactly one expected backend payload; "
                f"observed={observed!r}"
            )
        member = observed[0]
        return member, archive.read(member)


def produce(
    *,
    backend: str,
    consumer: Path,
    payload: Path,
    wheel: Path,
    source_root: Path,
    abi_surface: str = "numeric-route-v2",
    consumer_source: Path | None = None,
    harness_source_root: Path | None = None,
) -> dict[str, object]:
    contract = BACKENDS[backend]
    surface = ABI_SURFACES.get(abi_surface)
    if surface is None:
        raise RuntimeError(f"unknown ABI 1.1 surface: {abi_surface!r}")
    source_root = source_root.expanduser().resolve(strict=True)
    consumer = consumer.expanduser().resolve(strict=True)
    payload = payload.expanduser().resolve(strict=True)
    wheel = wheel.expanduser().resolve(strict=True)
    harness_root = (
        harness_source_root.expanduser().resolve(strict=True)
        if harness_source_root is not None
        else source_root
    )

    source_commit = _git_output(source_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError(f"source commit is not a full lowercase SHA: {source_commit!r}")
    dirty = _git_output(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError("canonical lifecycle evidence requires a clean source tree")
    harness_commit = _git_output(harness_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", harness_commit) is None:
        raise RuntimeError(
            f"harness source commit is not a full lowercase SHA: {harness_commit!r}"
        )
    harness_dirty = _git_output(
        harness_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if harness_dirty:
        raise RuntimeError("canonical lifecycle evidence requires a clean harness tree")

    wheel_member, embedded_payload = _wheel_payload(
        wheel, tuple(str(value) for value in contract["members"])
    )
    payload_bytes = payload.read_bytes()
    if payload_bytes != embedded_payload:
        raise RuntimeError(
            "payload bytes differ from the native member embedded in the exact wheel"
        )

    command = (
        str(consumer),
        str(payload),
        str(contract["kind"]),
        str(contract["route_count"]),
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ABI 1.1 consumer lifecycle failed: "
            f"returncode={result.returncode} stderr={result.stderr.strip()}"
        )
    marker = _consumer_result(result.stdout, schema=str(surface["result_schema"]))
    if marker.get("status") != "pass":
        raise RuntimeError("ABI 1.1 consumer did not report pass")
    if marker.get("backend_kind") != contract["kind"]:
        raise RuntimeError("ABI 1.1 consumer backend marker mismatch")
    observed_count = marker.get("route_count", marker.get("profile_count"))
    if observed_count != contract["route_count"]:
        raise RuntimeError("ABI 1.1 consumer route/profile-count marker mismatch")
    if tuple(marker.get("operations", ())) != tuple(surface["operations"]):
        raise RuntimeError("ABI 1.1 consumer operation marker is incomplete or reordered")
    marker_surface = marker.get("abi_surface")
    if marker_surface != abi_surface:
        raise RuntimeError("ABI 1.1 consumer surface marker mismatch")

    consumer_source = (
        consumer_source.expanduser().resolve(strict=True)
        if consumer_source is not None
        else harness_root / str(surface["consumer_source"])
    )
    if not consumer_source.is_file():
        raise RuntimeError("standalone ABI 1.1 consumer source is missing")
    try:
        consumer_source_relative = consumer_source.relative_to(harness_root)
    except ValueError as error:
        raise RuntimeError(
            "standalone ABI 1.1 consumer source must be inside the harness tree"
        ) from error
    tracked_source = _git_output(
        harness_root,
        "ls-files",
        "--error-unmatch",
        "--",
        consumer_source_relative.as_posix(),
    )
    if tracked_source != consumer_source_relative.as_posix():
        raise RuntimeError("standalone ABI 1.1 consumer source is not tracked")

    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "execution_mode": "canonical_payload",
        "execution_layer": str(surface["execution_layer"]),
        "abi": "1.1",
        "abi_surface": abi_surface,
        "contract_role": str(surface["contract_role"]),
        "backend": backend,
        "profiles": list(contract["profiles"]),
        "source_commit": source_commit,
        "product_source_commit": source_commit,
        "source_tree_state": {"status": "clean", "entries": []},
        "harness_source_commit": harness_commit,
        "harness_source_tree_state": {"status": "clean", "entries": []},
        "wheel_member": wheel_member,
        "wheel_member_sha256": _sha256_bytes(embedded_payload),
        "route_count": int(contract["route_count"]),
        "operations": list(surface["operations"]),
        "symbols": list(surface["symbols"]),
        "consumer_result": {
            "schema": str(surface["result_schema"]),
            "status": "pass",
            "returncode": result.returncode,
            "marker": marker,
            "stderr": result.stderr,
        },
        "provenance": {
            "payload": _file_identity(payload),
            "wheel": _file_identity(wheel),
            "consumer_binary": _file_identity(consumer),
            "consumer_source": _file_identity(consumer_source),
            "harness_source": _file_identity(consumer_source),
        },
        "command": list(command),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=tuple(BACKENDS))
    parser.add_argument("--consumer", required=True, type=Path)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--abi-surface",
        choices=tuple(ABI_SURFACES),
        default="numeric-route-v2",
        help="ABI 1.1 public surface exercised by the standalone consumer",
    )
    parser.add_argument(
        "--consumer-source",
        type=Path,
        help="external common consumer source; defaults to the selected surface path",
    )
    parser.add_argument(
        "--harness-source-root",
        type=Path,
        help="clean Git tree containing the external common consumer harness",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        evidence = produce(
            backend=args.backend,
            consumer=args.consumer,
            payload=args.payload,
            wheel=args.wheel,
            source_root=args.source_root,
            abi_surface=args.abi_surface,
            consumer_source=args.consumer_source,
            harness_source_root=args.harness_source_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"canonical ABI lifecycle evidence failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
