#!/usr/bin/env python3
"""Fail closed on PyPI filename collisions before publishing a frozen bundle."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
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


def validate_artifacts(
    artifact_dir: Path,
    version: str,
    metadata_loader: Callable[[str, str], dict[str, object] | None],
    *,
    allow_matching_existing: bool,
) -> tuple[int, int]:
    artifacts = sorted(
        path
        for path in artifact_dir.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not artifacts:
        raise CollisionError(f"no release artifacts found in {artifact_dir}")

    by_project: dict[str, list[Path]] = {}
    for artifact in artifacts:
        project = _artifact_project(artifact, version)
        publish_wheels, publish_sdist = PYPI_POLICY[project]
        if artifact.suffix == ".whl" and not publish_wheels:
            continue
        if artifact.name.endswith(".tar.gz") and not publish_sdist:
            continue
        by_project.setdefault(project, []).append(artifact)
    if not by_project:
        raise CollisionError("the release bundle has no manifest-selected PyPI files")

    matching = 0
    new = 0
    for project, project_artifacts in sorted(by_project.items()):
        metadata = metadata_loader(project, version)
        remote = {} if metadata is None else _remote_digests(metadata, project)
        for artifact in project_artifacts:
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
) -> tuple[int, int, int]:
    projects = validate_projects(PYPI_POLICY, project_metadata_loader)
    new, matching = validate_artifacts(
        artifact_dir,
        version,
        release_metadata_loader,
        allow_matching_existing=allow_matching_existing,
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
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.artifacts is None or args.version is None:
        parser.error(
            "--artifacts and --version are required unless --self-test is used"
        )

    try:
        projects, new, matching = run_preflight(
            args.artifacts,
            args.version,
            _load_pypi_metadata,
            _load_pypi_project_metadata,
            allow_matching_existing=args.allow_matching_existing,
        )
    except (CollisionError, OSError) as error:
        parser.exit(1, f"PYPI COLLISION PREFLIGHT: FAIL: {error}\n")
    print(
        "PYPI COLLISION PREFLIGHT: PASS "
        f"projects={projects} new={new} matching_existing={matching}"
    )


if __name__ == "__main__":
    main()
