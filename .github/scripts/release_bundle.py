#!/usr/bin/env python3
"""Create or verify the immutable handoff between build and publish workflows."""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "release_measure"))

from release_manifest import load_release_manifest, python_tag  # noqa: E402


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


def _package_precision_contract(filename: str) -> dict[str, list[str]]:
    """Return the manifest-owned execution profiles carried by one package."""

    manifest = load_release_manifest(ROOT)
    matches: list[dict[str, list[str]]] = []
    for distribution in manifest.standard_distributions:
        contract = {
            distribution.execution_backend: list(distribution.precision_profiles)
        }
        if filename.endswith(".whl"):
            for wheel in distribution.wheels:
                if any(
                    fnmatchcase(filename, pattern)
                    for pattern in wheel.filename_patterns
                ):
                    wheel_contract = dict(contract)
                    for backend, profiles in wheel.embedded_backend_profiles:
                        wheel_contract[backend] = list(profiles)
                    matches.append(wheel_contract)
        elif filename.startswith(f"{distribution.wheel_prefix}-") and filename.endswith(
            ".tar.gz"
        ):
            matches.append(contract)
    if len(matches) != 1:
        raise ValueError(
            f"package filename {filename!r} matched {len(matches)} release profile contracts"
        )
    return matches[0]


def _validate_source_sha(label: str, source_sha: object) -> str:
    if not isinstance(source_sha, str) or SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError(
            f"{label} must be 40-64 lowercase hexadecimal characters"
        )
    return source_sha


def _resolve_source_shas(
    source_sha: object,
    *,
    built_source_sha: object,
    authoritative_source_sha: object,
) -> tuple[str, str]:
    """Resolve the legacy source alias and the two explicit source identities.

    ``source_sha`` remains an input compatibility alias for the authoritative
    source identity.  A PR build is checked out at GitHub's synthetic merge
    commit, so its built and authoritative identities are allowed to differ.
    For non-PR builds the explicit built identity defaults to the authoritative
    identity.
    """

    if source_sha is not None:
        source_sha = _validate_source_sha("source SHA", source_sha)
        if authoritative_source_sha is not None:
            authoritative_source_sha = _validate_source_sha(
                "authoritative source SHA", authoritative_source_sha
            )
            if source_sha != authoritative_source_sha:
                raise ValueError(
                    "source SHA and authoritative source SHA must match when both "
                    "are provided"
                )
        authoritative = source_sha
    else:
        authoritative = _validate_source_sha(
            "authoritative source SHA", authoritative_source_sha
        )

    if built_source_sha is None:
        built = authoritative
    else:
        built = _validate_source_sha("built source SHA", built_source_sha)
    return built, authoritative


def _record(
    directory: Path,
    source_sha: str | None,
    run_id: str,
    repository: str,
    *,
    built_source_sha: str | None = None,
    authoritative_source_sha: str | None = None,
) -> dict[str, object]:
    built_source_sha, authoritative_source_sha = _resolve_source_shas(
        source_sha,
        built_source_sha=built_source_sha,
        authoritative_source_sha=authoritative_source_sha,
    )
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
        "schema_version": 3,
        "workflow": "build_wheels.yml",
        "repository": repository,
        "run_id": run_id,
        # ``source_sha`` is retained as the historical CLI/JSON alias for the
        # authoritative source identity.  The explicit fields below are the
        # canonical representation for PR merge builds.
        "source_sha": authoritative_source_sha,
        "built_source_sha": built_source_sha,
        "authoritative_source_sha": authoritative_source_sha,
        "release_manifest_sha256": _manifest_digest(ROOT),
        "package_artifact_count": manifest.standard_artifact_count,
        "files": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "precision_profiles": _package_precision_contract(path.name),
            }
            for path in files
        ],
    }


def create(
    directory: Path,
    source_sha: str | None,
    run_id: str,
    repository: str,
    *,
    built_source_sha: str | None = None,
    authoritative_source_sha: str | None = None,
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
    provenance = _record(
        directory,
        source_sha,
        run_id,
        repository,
        built_source_sha=built_source_sha,
        authoritative_source_sha=authoritative_source_sha,
    )
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
    source_sha: str | None,
    run_id: str,
    repository: str,
    *,
    built_source_sha: str | None = None,
    authoritative_source_sha: str | None = None,
) -> None:
    provenance_path = directory / PROVENANCE_NAME
    checksums_path = directory / CHECKSUMS_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("frozen bundle provenance must be a JSON object")
    if provenance.get("schema_version") != 3:
        raise ValueError("frozen bundle provenance schema_version must be 3")
    stored_built_source_sha, stored_authoritative_source_sha = _resolve_source_shas(
        provenance.get("source_sha"),
        built_source_sha=provenance.get("built_source_sha"),
        authoritative_source_sha=provenance.get("authoritative_source_sha"),
    )
    if source_sha is not None and authoritative_source_sha is not None:
        _resolve_source_shas(
            source_sha,
            built_source_sha=built_source_sha,
            authoritative_source_sha=authoritative_source_sha,
        )
    requested_authoritative_source_sha = (
        _validate_source_sha("source SHA", source_sha)
        if source_sha is not None
        else _validate_source_sha(
            "authoritative source SHA", authoritative_source_sha
        )
    )
    if requested_authoritative_source_sha != stored_authoritative_source_sha:
        raise ValueError(
            "frozen bundle authoritative source SHA does not match the requested "
            "source SHA"
        )
    if built_source_sha is not None:
        requested_built_source_sha = _validate_source_sha(
            "built source SHA", built_source_sha
        )
        if requested_built_source_sha != stored_built_source_sha:
            raise ValueError(
                "frozen bundle built source SHA does not match the requested "
                "built source SHA"
            )
    expected = _record(
        directory,
        stored_authoritative_source_sha,
        run_id,
        repository,
        built_source_sha=stored_built_source_sha,
        authoritative_source_sha=stored_authoritative_source_sha,
    )
    if provenance != expected:
        raise ValueError("frozen bundle provenance does not match its package files")

    expected_lines = [
        f"{entry['sha256']}  {entry['filename']}" for entry in expected["files"]
    ]
    expected_lines.append(f"{_sha256(provenance_path)}  {PROVENANCE_NAME}")
    expected_contents = "".join(f"{line}\n" for line in expected_lines).encode("utf-8")
    if checksums_path.read_bytes() != expected_contents:
        raise ValueError(
            "SHA256SUMS is not the exact canonical ordered frozen-bundle manifest"
        )
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
    package_count = manifest.standard_artifact_count
    checksum_count = package_count + 1
    frozen_file_count = package_count + 2
    with tempfile.TemporaryDirectory(prefix="gafime-release-bundle-") as temporary:
        directory = Path(temporary)
        filenames = []
        for distribution in manifest.standard_distributions:
            filenames.append(f"{distribution.wheel_prefix}-1.0.0.tar.gz")
            for wheel in distribution.wheels:
                for version in wheel.python_versions:
                    filename = wheel.filename_template.format(
                        python_tag=python_tag(version)
                    ).replace("*", "1.0.0")
                    filenames.append(filename)
        if len(filenames) != package_count or len(filenames) != len(set(filenames)):
            raise AssertionError("release-bundle self-test filename matrix is invalid")
        for index, filename in enumerate(sorted(filenames)):
            (directory / filename).write_bytes(f"artifact-{index}".encode("ascii"))
        kwargs = {
            "source_sha": None,
            "built_source_sha": "a" * 40,
            "authoritative_source_sha": "b" * 40,
            "run_id": "12345",
            "repository": "onlyxItachi/GAFIME",
        }
        create(directory, **kwargs)
        verify(directory, **kwargs)
        verify(
            directory,
            source_sha="b" * 40,
            run_id="12345",
            repository=kwargs["repository"],
        )
        provenance = json.loads(
            (directory / PROVENANCE_NAME).read_text(encoding="utf-8")
        )
        checksum_lines = (
            (directory / CHECKSUMS_NAME).read_text(encoding="utf-8").splitlines()
        )
        if provenance["package_artifact_count"] != package_count:
            raise AssertionError("provenance package count differs from the manifest")
        if provenance["source_sha"] != "b" * 40:
            raise AssertionError("source_sha compatibility alias is not authoritative")
        if provenance["built_source_sha"] != "a" * 40:
            raise AssertionError("built source SHA was not retained")
        if provenance["authoritative_source_sha"] != "b" * 40:
            raise AssertionError("authoritative source SHA was not retained")
        profile_contracts = {
            entry["filename"]: entry["precision_profiles"]
            for entry in provenance["files"]
        }
        if len(profile_contracts) != package_count:
            raise AssertionError("provenance precision contracts are incomplete")
        if not any(
            contract.get("metal") == ["fp32"] for contract in profile_contracts.values()
        ):
            raise AssertionError("provenance omitted the Metal fp32-only contract")
        if any(
            contract.get("metal") not in (None, ["fp32"])
            for contract in profile_contracts.values()
        ):
            raise AssertionError("provenance advertised unsupported Metal profiles")
        if len(checksum_lines) != checksum_count:
            raise AssertionError("checksum count differs from packages plus provenance")
        if len(tuple(directory.iterdir())) != frozen_file_count:
            raise AssertionError(
                "frozen bundle count differs from packages, provenance, and checksums"
            )
        checksums_path = directory / CHECKSUMS_NAME
        canonical_checksums = checksums_path.read_bytes()
        checksum_lines_with_endings = canonical_checksums.splitlines(keepends=True)
        invalid_checksums = {
            "duplicate line": canonical_checksums + checksum_lines_with_endings[0],
            "blank line": (
                checksum_lines_with_endings[0]
                + b"\n"
                + b"".join(checksum_lines_with_endings[1:])
            ),
            "reordered lines": (
                checksum_lines_with_endings[1]
                + checksum_lines_with_endings[0]
                + b"".join(checksum_lines_with_endings[2:])
            ),
            "missing final newline": canonical_checksums.removesuffix(b"\n"),
        }
        for label, invalid_contents in invalid_checksums.items():
            checksums_path.write_bytes(invalid_contents)
            try:
                verify(directory, **kwargs)
            except ValueError as error:
                if "SHA256SUMS" not in str(error):
                    raise
            else:
                raise AssertionError(f"{label} unexpectedly verified")
        checksums_path.write_bytes(canonical_checksums)
        verify(directory, **kwargs)
        try:
            verify(
                directory,
                source_sha="c" * 40,
                run_id="12345",
                repository=kwargs["repository"],
            )
        except ValueError as error:
            if "authoritative source SHA" not in str(error):
                raise
        else:
            raise AssertionError("wrong authoritative source unexpectedly verified")
        try:
            verify(
                directory,
                source_sha="b" * 40,
                built_source_sha="c" * 40,
                run_id="12345",
                repository=kwargs["repository"],
            )
        except ValueError as error:
            if "built source SHA" not in str(error):
                raise
        else:
            raise AssertionError("wrong built source unexpectedly verified")
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
    print(
        "RELEASE BUNDLE SELF-TEST: PASS "
        f"packages={package_count} checksums={checksum_count} "
        f"files={frozen_file_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "verify"), nargs="?")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument(
        "--source-sha",
        help="compatibility alias for --authoritative-source-sha",
    )
    parser.add_argument(
        "--built-source-sha",
        help="exact checkout commit used to build the package files",
    )
    parser.add_argument(
        "--authoritative-source-sha",
        help="PR head SHA, or the built SHA for non-PR workflows",
    )
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
        or (args.source_sha is None and args.authoritative_source_sha is None)
        or args.run_id is None
        or args.repository is None
    ):
        parser.error(
            "mode, --artifacts, --source-sha or --authoritative-source-sha, "
            "--run-id, and --repository are required"
        )
    directory = args.artifacts.resolve()
    if args.mode == "create":
        create(
            directory,
            args.source_sha,
            args.run_id,
            args.repository,
            built_source_sha=args.built_source_sha,
            authoritative_source_sha=args.authoritative_source_sha,
        )
    else:
        verify(
            directory,
            args.source_sha,
            args.run_id,
            args.repository,
            built_source_sha=args.built_source_sha,
            authoritative_source_sha=args.authoritative_source_sha,
        )
    print(f"RELEASE BUNDLE {args.mode.upper()}: PASS")


if __name__ == "__main__":
    main()
