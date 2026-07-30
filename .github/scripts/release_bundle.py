#!/usr/bin/env python3
"""Create or verify the immutable handoff between build and publish workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "release_measure"))

from release_manifest import load_release_manifest  # noqa: E402


PROVENANCE_NAME = "release-bundle-provenance.json"
CHECKSUMS_NAME = "SHA256SUMS"
PACKAGE_SUFFIXES = (".whl", ".tar.gz")
SOURCE_SHA_RE = re.compile(r"[0-9a-f]{40,64}")
RUN_ID_RE = re.compile(r"[1-9][0-9]*")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"release bundle directory does not exist: {directory}")
    symlinks = sorted(path.name for path in directory.iterdir() if path.is_symlink())
    if symlinks:
        raise ValueError(f"frozen bundle contains symbolic links: {symlinks}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.endswith(PACKAGE_SUFFIXES)
    )
    return files


def _manifest_digest(root: Path) -> str:
    return _sha256(root / ".github" / "release-artifacts.json")


def _record(
    directory: Path,
    source_sha: str,
    run_id: str,
    repository: str,
) -> dict[str, object]:
    if SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("source SHA must be 40-64 lowercase hexadecimal characters")
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("GitHub Actions run ID must be a positive decimal integer")
    if REPOSITORY_RE.fullmatch(repository) is None:
        raise ValueError("repository must use the owner/name form")
    manifest = load_release_manifest(ROOT)
    files = _package_files(directory)
    if len(files) != manifest.standard_artifact_count:
        raise ValueError(
            f"frozen bundle has {len(files)} package files; manifest derives "
            f"{manifest.standard_artifact_count}"
        )
    return {
        "schema_version": 1,
        "workflow": "build_wheels.yml",
        "repository": repository,
        "run_id": run_id,
        "source_sha": source_sha,
        "release_manifest_sha256": _manifest_digest(ROOT),
        "package_artifact_count": manifest.standard_artifact_count,
        "files": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in files
        ],
    }


def create(
    directory: Path,
    source_sha: str,
    run_id: str,
    repository: str,
) -> None:
    if not directory.is_dir():
        raise ValueError(f"release bundle directory does not exist: {directory}")
    unexpected = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_symlink()
        or not path.is_file()
        or not path.name.endswith(PACKAGE_SUFFIXES)
    )
    if unexpected:
        raise ValueError(
            f"release freeze contains non-package files before provenance: {unexpected}"
        )
    provenance = _record(directory, source_sha, run_id, repository)
    provenance_path = directory / PROVENANCE_NAME
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_paths = [*_package_files(directory), provenance_path]
    (directory / CHECKSUMS_NAME).write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )


def verify(
    directory: Path,
    source_sha: str,
    run_id: str,
    repository: str,
) -> None:
    provenance_path = directory / PROVENANCE_NAME
    checksums_path = directory / CHECKSUMS_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = _record(directory, source_sha, run_id, repository)
    if provenance != expected:
        raise ValueError("frozen bundle provenance does not match its package files")

    expected_lines = {
        f"{entry['sha256']}  {entry['filename']}" for entry in expected["files"]
    }
    expected_lines.add(f"{_sha256(provenance_path)}  {PROVENANCE_NAME}")
    actual_lines = {
        line for line in checksums_path.read_text(encoding="utf-8").splitlines() if line
    }
    if actual_lines != expected_lines:
        raise ValueError("SHA256SUMS does not exactly cover the frozen bundle")
    allowed = {
        *(entry["filename"] for entry in expected["files"]),
        PROVENANCE_NAME,
        CHECKSUMS_NAME,
    }
    actual = {path.name for path in directory.iterdir()}
    if actual != allowed:
        raise ValueError(
            f"frozen bundle files differ from provenance: "
            f"missing={sorted(allowed - actual)} extra={sorted(actual - allowed)}"
        )


def self_test() -> None:
    manifest = load_release_manifest(ROOT)
    with tempfile.TemporaryDirectory(prefix="gafime-release-bundle-") as temporary:
        directory = Path(temporary)
        for index in range(manifest.standard_artifact_count):
            (directory / f"artifact_{index:03d}-0-cp310-cp310-any.whl").write_bytes(
                f"artifact-{index}".encode("ascii")
            )
        kwargs = {
            "source_sha": "a" * 40,
            "run_id": "12345",
            "repository": "onlyxItachi/GAFIME",
        }
        create(directory, **kwargs)
        verify(directory, **kwargs)
        package = _package_files(directory)[0]
        package.write_bytes(b"tampered")
        try:
            verify(directory, **kwargs)
        except ValueError as error:
            if "provenance" not in str(error):
                raise
        else:
            raise AssertionError("tampered frozen bundle unexpectedly verified")
    with tempfile.TemporaryDirectory(prefix="gafime-release-bundle-link-") as temporary:
        directory = Path(temporary)
        target = directory / "target.whl"
        target.write_bytes(b"target")
        (directory / "linked.whl").symlink_to(target)
        try:
            _package_files(directory)
        except ValueError as error:
            if "symbolic links" not in str(error):
                raise
        else:
            raise AssertionError("symbolic link unexpectedly entered frozen bundle")
    print("RELEASE BUNDLE SELF-TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "verify"), nargs="?")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--run-id")
    parser.add_argument("--repository")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if (
        args.mode is None
        or args.artifacts is None
        or args.source_sha is None
        or args.run_id is None
        or args.repository is None
    ):
        parser.error(
            "mode, --artifacts, --source-sha, --run-id, and --repository are required"
        )
    directory = args.artifacts.resolve()
    if args.mode == "create":
        create(directory, args.source_sha, args.run_id, args.repository)
    else:
        verify(directory, args.source_sha, args.run_id, args.repository)
    print(f"RELEASE BUNDLE {args.mode.upper()}: PASS")


if __name__ == "__main__":
    main()
