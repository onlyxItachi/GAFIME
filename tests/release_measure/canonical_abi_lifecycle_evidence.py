#!/usr/bin/env python3
"""Emit hash-bound canonical ABI 1.1 lifecycle evidence.

The producer executes the standalone public-header C consumer against the
payload bytes embedded in an exact wheel. It does not import GAFIME or rely on
the repository's Rust loader.
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
EVIDENCE_SCHEMA = "gafime.native-decomposition.v1"


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


def _consumer_result(stdout: str) -> dict[str, object]:
    objects: list[dict[str, object]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == RESULT_SCHEMA:
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
) -> dict[str, object]:
    contract = BACKENDS[backend]
    source_root = source_root.expanduser().resolve(strict=True)
    consumer = consumer.expanduser().resolve(strict=True)
    payload = payload.expanduser().resolve(strict=True)
    wheel = wheel.expanduser().resolve(strict=True)

    source_commit = _git_output(source_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError(f"source commit is not a full lowercase SHA: {source_commit!r}")
    dirty = _git_output(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError("canonical lifecycle evidence requires a clean source tree")

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
    marker = _consumer_result(result.stdout)
    if marker.get("status") != "pass":
        raise RuntimeError("ABI 1.1 consumer did not report pass")
    if marker.get("backend_kind") != contract["kind"]:
        raise RuntimeError("ABI 1.1 consumer backend marker mismatch")
    if marker.get("route_count") != contract["route_count"]:
        raise RuntimeError("ABI 1.1 consumer route-count marker mismatch")
    if tuple(marker.get("operations", ())) != REQUIRED_OPERATIONS:
        raise RuntimeError("ABI 1.1 consumer operation marker is incomplete or reordered")

    consumer_source = source_root / "tests/gpu/abi_consumers/abi_1_1_c_consumer.c"
    if not consumer_source.is_file():
        raise RuntimeError("standalone ABI 1.1 consumer source is missing")

    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "execution_mode": "canonical_payload",
        "execution_layer": "independent_abi_1_1_c_consumer",
        "abi": "canonical_1.1",
        "backend": backend,
        "profiles": list(contract["profiles"]),
        "source_commit": source_commit,
        "source_tree_state": {"status": "clean", "entries": []},
        "wheel_member": wheel_member,
        "wheel_member_sha256": _sha256_bytes(embedded_payload),
        "operations": list(REQUIRED_OPERATIONS),
        "consumer_result": {
            "schema": RESULT_SCHEMA,
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
