#!/usr/bin/env python3
"""Fail closed on PyPI filename collisions before publishing a frozen bundle."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "release_measure"))
from release_manifest import load_release_manifest  # noqa: E402


RELEASE_MANIFEST = load_release_manifest(ROOT)
PROJECT_PREFIXES = {
    distribution.wheel_prefix: distribution.name
    for distribution in RELEASE_MANIFEST.standard_distributions
}
PYPI_POLICY = {
    distribution.name: (distribution.pypi_wheels, distribution.pypi_sdist)
    for distribution in RELEASE_MANIFEST.standard_distributions
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROJECT_NORMALIZATION_RE = re.compile(r"[-_.]+")


class CollisionError(RuntimeError):
    """A remote filename exists without an explicitly accepted identical hash."""


def _canonical_project_name(value: str) -> str:
    return PROJECT_NORMALIZATION_RE.sub("-", value).lower()


def validate_projects(
    projects: Iterable[str],
    metadata_loader: Callable[[str], dict[str, object] | None],
) -> tuple[str, ...]:
    """Require every manifest-selected PyPI project to exist with the right name."""
    expected = tuple(sorted(set(projects)))
    if not expected:
        raise CollisionError("the release manifest selects no PyPI projects")
    for project in expected:
        metadata = metadata_loader(project)
        if metadata is None:
            raise CollisionError(f"PyPI project does not exist: {project}")
        info = metadata.get("info")
        remote_name = info.get("name") if isinstance(info, dict) else None
        if not isinstance(remote_name, str):
            raise CollisionError(f"PyPI project metadata lacks a name: {project}")
        if _canonical_project_name(remote_name) != _canonical_project_name(project):
            raise CollisionError(
                f"PyPI project identity mismatch: expected {project}, got {remote_name}"
            )
    return expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_project(path: Path, version: str) -> str:
    for prefix, project in PROJECT_PREFIXES.items():
        versioned_prefix = f"{prefix}-{version}"
        if path.name.endswith(".tar.gz"):
            if path.name == f"{versioned_prefix}.tar.gz":
                return project
            continue
        if not path.name.endswith(".whl"):
            continue
        try:
            filename_prefix, python_tag, abi_tag, platform_tag = path.name[:-4].rsplit(
                "-", 3
            )
        except ValueError:
            continue
        if not all((python_tag, abi_tag, platform_tag)):
            continue
        if filename_prefix == versioned_prefix:
            return project
        build_prefix = f"{versioned_prefix}-"
        if filename_prefix.startswith(build_prefix):
            build_tag = filename_prefix.removeprefix(build_prefix)
            if re.fullmatch(r"[0-9][A-Za-z0-9_]*", build_tag) is not None:
                return project
    raise CollisionError(
        f"unexpected release artifact filename for version {version}: {path.name}"
    )


def _remote_digests(metadata: dict[str, object], project: str) -> dict[str, str]:
    urls = metadata.get("urls", [])
    if not isinstance(urls, list):
        raise CollisionError(f"PyPI metadata for {project} has no URL list")
    digests: dict[str, str] = {}
    for entry in urls:
        if not isinstance(entry, dict):
            raise CollisionError(
                f"PyPI metadata for {project} has a malformed URL entry"
            )
        filename = entry.get("filename")
        raw_digests = entry.get("digests")
        sha256 = raw_digests.get("sha256") if isinstance(raw_digests, dict) else None
        if not isinstance(filename, str) or not isinstance(sha256, str):
            raise CollisionError(f"PyPI metadata for {project} lacks a filename digest")
        sha256 = sha256.lower()
        if SHA256_RE.fullmatch(sha256) is None:
            raise CollisionError(
                f"PyPI metadata for {project}/{filename} has an invalid SHA-256"
            )
        previous = digests.setdefault(filename, sha256)
        if previous != sha256:
            raise CollisionError(
                f"PyPI metadata for {project}/{filename} has conflicting SHA-256 values"
            )
    return digests


def _expected_pypi_artifacts(
    artifact_dir: Path, version: str
) -> dict[str, dict[str, Path]]:
    artifacts = sorted(
        path
        for path in artifact_dir.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not artifacts:
        raise CollisionError(f"no release artifacts found in {artifact_dir}")

    by_project: dict[str, dict[str, Path]] = {}
    for artifact in artifacts:
        project = _artifact_project(artifact, version)
        publish_wheels, publish_sdist = PYPI_POLICY[project]
        if artifact.suffix == ".whl" and not publish_wheels:
            continue
        if artifact.name.endswith(".tar.gz") and not publish_sdist:
            continue
        project_artifacts = by_project.setdefault(project, {})
        if artifact.name in project_artifacts:
            raise CollisionError(
                f"duplicate frozen PyPI artifact for {project}: {artifact.name}"
            )
        project_artifacts[artifact.name] = artifact
    if not by_project:
        raise CollisionError("the release bundle has no manifest-selected PyPI files")
    return by_project


def validate_artifacts(
    artifact_dir: Path,
    version: str,
    metadata_loader: Callable[[str, str], dict[str, object] | None],
    *,
    allow_matching_existing: bool,
    require_complete_published: bool = False,
) -> tuple[int, int]:
    by_project = _expected_pypi_artifacts(artifact_dir, version)
    remote_by_project: dict[str, dict[str, str]] = {}
    for project in sorted(by_project):
        metadata = metadata_loader(project, version)
        remote_by_project[project] = (
            {} if metadata is None else _remote_digests(metadata, project)
        )
    for project, project_artifacts in sorted(by_project.items()):
        remote = remote_by_project[project]
        unexpected = sorted(set(remote) - set(project_artifacts))
        if unexpected:
            raise CollisionError(
                f"PyPI published inventory contains files outside the frozen bundle "
                f"for {project}: {unexpected}"
            )
        if require_complete_published:
            missing = sorted(set(project_artifacts) - set(remote))
            if missing:
                raise CollisionError(
                    f"PyPI published inventory is missing frozen files for "
                    f"{project}: {missing}"
                )

    matching = 0
    new = 0
    for project, project_artifacts in sorted(by_project.items()):
        remote = remote_by_project[project]
        for artifact in project_artifacts.values():
            remote_sha256 = remote.get(artifact.name)
            if remote_sha256 is None:
                new += 1
                continue
            local_sha256 = _sha256(artifact)
            if not allow_matching_existing:
                raise CollisionError(
                    f"PyPI filename already exists: {project}/{artifact.name}; "
                    "use explicit matching-hash recovery only after a partial publication"
                )
            if local_sha256 != remote_sha256:
                raise CollisionError(
                    f"PyPI collision hash mismatch for {project}/{artifact.name}: "
                    f"local={local_sha256} remote={remote_sha256}"
                )
            matching += 1
    return new, matching


def validate_resolver_receipt(
    artifact_dir: Path,
    version: str,
    receipt_dir: Path,
    projects: Iterable[str],
) -> tuple[str, ...]:
    """Bind resolver-selected files to names and bytes in the frozen bundle."""
    expected = _expected_pypi_artifacts(artifact_dir, version)
    requested = tuple(sorted(set(projects)))
    if not requested:
        raise CollisionError("resolver receipt selects no PyPI projects")
    unexpected_projects = sorted(set(requested) - set(expected))
    if unexpected_projects:
        raise CollisionError(
            f"resolver receipt requests projects outside the frozen bundle: "
            f"{unexpected_projects}"
        )
    if not receipt_dir.is_dir() or receipt_dir.is_symlink():
        raise CollisionError(f"resolver receipt directory is invalid: {receipt_dir}")

    receipt_paths = sorted(receipt_dir.iterdir())
    if not receipt_paths:
        raise CollisionError(f"resolver receipt is empty: {receipt_dir}")
    observed: set[str] = set()
    for receipt in receipt_paths:
        if receipt.is_symlink() or not receipt.is_file():
            raise CollisionError(f"resolver receipt has a non-regular entry: {receipt}")
        project = _artifact_project(receipt, version)
        if project not in requested:
            raise CollisionError(
                f"resolver receipt selected unexpected project artifact: {receipt.name}"
            )
        expected_artifact = expected[project].get(receipt.name)
        if expected_artifact is None:
            raise CollisionError(
                f"resolver receipt selected a file outside the frozen bundle: "
                f"{project}/{receipt.name}"
            )
        if project in observed:
            raise CollisionError(
                f"resolver receipt selected multiple files for {project}: {receipt.name}"
            )
        if receipt.stat().st_size != expected_artifact.stat().st_size:
            raise CollisionError(
                f"resolver receipt size mismatch for {project}/{receipt.name}"
            )
        expected_sha256 = _sha256(expected_artifact)
        receipt_sha256 = _sha256(receipt)
        if receipt_sha256 != expected_sha256:
            raise CollisionError(
                f"resolver receipt hash mismatch for {project}/{receipt.name}: "
                f"frozen={expected_sha256} receipt={receipt_sha256}"
            )
        observed.add(project)
    missing = sorted(set(requested) - observed)
    if missing:
        raise CollisionError(f"resolver receipt is missing projects: {missing}")
    return tuple(sorted(observed))


def _load_pypi_metadata(project: str, version: str) -> dict[str, object] | None:
    url = f"https://pypi.org/pypi/{quote(project, safe='')}/{quote(version, safe='')}/json"
    request = Request(url, headers={"User-Agent": "gafime-release-preflight/1"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise CollisionError(
            f"PyPI metadata request failed for {project}: HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise CollisionError(
            f"PyPI metadata request failed for {project}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CollisionError(f"PyPI metadata for {project} is not an object")
    return payload


def _pypi_download_url(
    metadata: dict[str, object],
    project: str,
    filename: str,
    expected_sha256: str,
) -> str:
    remote = _remote_digests(metadata, project)
    if remote.get(filename) != expected_sha256:
        raise CollisionError(
            f"PyPI metadata does not bind {project}/{filename} to the frozen SHA-256"
        )
    urls = metadata.get("urls")
    assert isinstance(urls, list)  # Checked by _remote_digests above.
    matching = [
        entry
        for entry in urls
        if isinstance(entry, dict) and entry.get("filename") == filename
    ]
    if len(matching) != 1:
        raise CollisionError(
            f"PyPI metadata has an ambiguous download URL for {project}/{filename}"
        )
    url = matching[0].get("url")
    if not isinstance(url, str):
        raise CollisionError(
            f"PyPI metadata has no download URL for {project}/{filename}"
        )
    _require_trusted_pypi_file_url(url, project, filename)
    return url


def _require_trusted_pypi_file_url(url: str, project: str, filename: str) -> None:
    """Reject redirects or metadata URLs that leave the PyPI file origin."""
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise CollisionError(
            f"PyPI metadata has an invalid download URL for {project}/{filename}"
        ) from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise CollisionError(
            f"PyPI metadata has an untrusted download URL for {project}/{filename}"
        )


def download_verified_sdist_receipt(
    artifact_dir: Path,
    version: str,
    project: str,
    receipt_dir: Path,
    metadata_loader: Callable[[str, str], dict[str, object] | None],
) -> Path:
    """Download the unique public sdist as raw bytes before build tooling sees it."""
    expected = _expected_pypi_artifacts(artifact_dir, version)
    project_artifacts = expected.get(project)
    if project_artifacts is None or len(project_artifacts) != 1:
        raise CollisionError(
            f"frozen bundle does not have one public artifact for raw {project} receipt"
        )
    filename, frozen = next(iter(project_artifacts.items()))
    if not filename.endswith(".tar.gz"):
        raise CollisionError(
            f"raw receipt is reserved for a unique source distribution, not {filename}"
        )
    if not receipt_dir.is_dir() or receipt_dir.is_symlink():
        raise CollisionError(f"resolver receipt directory is invalid: {receipt_dir}")
    destination = receipt_dir / filename
    if destination.exists() or destination.is_symlink():
        raise CollisionError(f"resolver receipt already contains {destination.name}")
    metadata = metadata_loader(project, version)
    if metadata is None:
        raise CollisionError(
            f"PyPI project/version does not exist: {project}/{version}"
        )
    frozen_sha256 = _sha256(frozen)
    url = _pypi_download_url(metadata, project, filename, frozen_sha256)
    expected_size = frozen.stat().st_size
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".receipt-",
            suffix=".part",
            dir=receipt_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            downloaded = 0
            request = Request(url, headers={"User-Agent": "gafime-release-preflight/1"})
            with urlopen(request, timeout=30) as response:
                response_url = response.geturl()
                _require_trusted_pypi_file_url(response_url, project, filename)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    downloaded += len(chunk)
                    if downloaded > expected_size:
                        raise CollisionError(
                            f"raw receipt exceeds frozen size for {project}/{filename}"
                        )
                    digest.update(chunk)
                    temporary.write(chunk)
        if downloaded != expected_size or digest.hexdigest() != frozen_sha256:
            raise CollisionError(f"raw receipt hash mismatch for {project}/{filename}")
        try:
            os.link(temporary_path, destination)
        except FileExistsError as error:
            raise CollisionError(
                f"resolver receipt already contains {destination.name}"
            ) from error
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise CollisionError(
            f"PyPI raw receipt download failed for {project}/{filename}: {error}"
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _load_pypi_project_metadata(project: str) -> dict[str, object] | None:
    url = f"https://pypi.org/pypi/{quote(project, safe='')}/json"
    request = Request(url, headers={"User-Agent": "gafime-release-preflight/1"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise CollisionError(
            f"PyPI project request failed for {project}: HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise CollisionError(
            f"PyPI project request failed for {project}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CollisionError(f"PyPI project metadata for {project} is not an object")
    return payload


def run_preflight(
    artifact_dir: Path,
    version: str,
    release_metadata_loader: Callable[[str, str], dict[str, object] | None],
    project_metadata_loader: Callable[[str], dict[str, object] | None],
    *,
    allow_matching_existing: bool,
    require_complete_published: bool = False,
) -> tuple[int, int, int]:
    projects = validate_projects(PYPI_POLICY, project_metadata_loader)
    new, matching = validate_artifacts(
        artifact_dir,
        version,
        release_metadata_loader,
        allow_matching_existing=allow_matching_existing,
        require_complete_published=require_complete_published,
    )
    return len(projects), new, matching


def _expect_collision(operation: Callable[[], object], expected: str) -> None:
    try:
        operation()
    except CollisionError as error:
        if expected not in str(error):
            raise AssertionError(f"expected {expected!r} in {error!r}") from error
    else:
        raise AssertionError(f"expected collision containing {expected!r}")


def _self_test() -> None:
    version = "1.0.0b2"
    expected_projects = set(PROJECT_PREFIXES.values())

    def existing_project(project: str) -> dict[str, object]:
        return {"info": {"name": project}}

    assert validate_projects(expected_projects, existing_project) == tuple(
        sorted(expected_projects)
    )
    _expect_collision(
        lambda: validate_projects({"gafime"}, lambda _project: None),
        "does not exist",
    )
    _expect_collision(
        lambda: validate_projects(
            {"gafime"}, lambda _project: {"info": {"name": "different-project"}}
        ),
        "identity mismatch",
    )

    with tempfile.TemporaryDirectory(prefix="gafime-pypi-collision-") as temp_dir:
        artifact_dir = Path(temp_dir)
        wheel = artifact_dir / f"gafime-{version}-cp310-cp310-manylinux_2_28_x86_64.whl"
        sdist = artifact_dir / f"gafime-{version}.tar.gz"
        wheel.write_bytes(b"wheel")
        sdist.write_bytes(b"sdist")

        def absent(_project: str, _version: str) -> None:
            return None

        assert validate_artifacts(
            artifact_dir, version, absent, allow_matching_existing=False
        ) == (2, 0)

        matching_metadata = {
            "urls": [
                {"filename": wheel.name, "digests": {"sha256": _sha256(wheel)}},
                {"filename": sdist.name, "digests": {"sha256": _sha256(sdist)}},
            ]
        }

        def matching(_project: str, _version: str) -> dict[str, object]:
            return matching_metadata

        _expect_collision(
            lambda: validate_artifacts(
                artifact_dir, version, matching, allow_matching_existing=False
            ),
            "already exists",
        )
        assert validate_artifacts(
            artifact_dir, version, matching, allow_matching_existing=True
        ) == (0, 2)

        partial_metadata = {
            "urls": [
                {"filename": sdist.name, "digests": {"sha256": _sha256(sdist)}},
            ]
        }

        def partial(_project: str, _version: str) -> dict[str, object]:
            return partial_metadata

        assert validate_artifacts(
            artifact_dir, version, partial, allow_matching_existing=True
        ) == (1, 1)

        def mismatched(_project: str, _version: str) -> dict[str, object]:
            return {
                "urls": [
                    {"filename": wheel.name, "digests": {"sha256": "0" * 64}},
                ]
            }

        _expect_collision(
            lambda: validate_artifacts(
                artifact_dir, version, mismatched, allow_matching_existing=True
            ),
            "hash mismatch",
        )

        all_projects = artifact_dir / "all-projects"
        all_projects.mkdir()
        for prefix in PROJECT_PREFIXES:
            (all_projects / f"{prefix}-{version}.tar.gz").write_bytes(prefix.encode())
        raw_rocm_wheel = (
            all_projects / f"gafime_rocm-{version}-cp310-cp310-linux_x86_64.whl"
        )
        raw_rocm_wheel.write_bytes(b"GitHub Release only")
        requested_projects: set[str] = set()

        def all_absent(project: str, _version: str) -> None:
            requested_projects.add(project)
            return None

        assert run_preflight(
            all_projects,
            version,
            all_absent,
            existing_project,
            allow_matching_existing=False,
        ) == (len(expected_projects), len(expected_projects), 0)
        assert requested_projects == expected_projects
    print("PYPI COLLISION SELF-TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--allow-matching-existing", action="store_true")
    parser.add_argument("--require-complete-published", action="store_true")
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--receipt-project", action="append", default=[])
    parser.add_argument("--download-sdist-project", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.artifacts is None or args.version is None:
        parser.error(
            "--artifacts and --version are required unless --self-test is used"
        )
    if args.receipt_dir is None and (
        args.receipt_project or args.download_sdist_project
    ):
        parser.error("--receipt-dir is required for resolver receipt options")
    if args.receipt_dir is not None and not args.receipt_project:
        parser.error("--receipt-project is required with --receipt-dir")
    if set(args.download_sdist_project) - set(args.receipt_project):
        parser.error("every --download-sdist-project must be a --receipt-project")

    try:
        require_complete = bool(
            args.require_complete_published or args.download_sdist_project
        )
        projects, new, matching = run_preflight(
            args.artifacts,
            args.version,
            _load_pypi_metadata,
            _load_pypi_project_metadata,
            allow_matching_existing=(args.allow_matching_existing or require_complete),
            require_complete_published=require_complete,
        )
        downloaded = []
        for project in args.download_sdist_project:
            assert args.receipt_dir is not None
            downloaded.append(
                download_verified_sdist_receipt(
                    args.artifacts,
                    args.version,
                    project,
                    args.receipt_dir,
                    _load_pypi_metadata,
                ).name
            )
        receipt_projects: tuple[str, ...] = ()
        if args.receipt_dir is not None:
            receipt_projects = validate_resolver_receipt(
                args.artifacts,
                args.version,
                args.receipt_dir,
                args.receipt_project,
            )
    except (CollisionError, OSError) as error:
        parser.exit(1, f"PYPI COLLISION PREFLIGHT: FAIL: {error}\n")
    print(
        "PYPI COLLISION PREFLIGHT: PASS "
        f"projects={projects} new={new} matching_existing={matching} "
        f"receipt_projects={list(receipt_projects)} downloaded={downloaded}"
    )


if __name__ == "__main__":
    main()
