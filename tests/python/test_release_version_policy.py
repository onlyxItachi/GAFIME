from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / ".github" / "scripts" / "release_version.py"
SPEC = importlib.util.spec_from_file_location("gafime_release_version", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release_version = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_version
SPEC.loader.exec_module(release_version)
ReleaseVersion = release_version.ReleaseVersion
VersionPolicyError = release_version.VersionPolicyError

COLLISION_MODULE_PATH = (
    ROOT / ".github" / "scripts" / "check_pypi_artifact_collisions.py"
)
COLLISION_SPEC = importlib.util.spec_from_file_location(
    "gafime_pypi_collision", COLLISION_MODULE_PATH
)
assert COLLISION_SPEC is not None and COLLISION_SPEC.loader is not None
collision = importlib.util.module_from_spec(COLLISION_SPEC)
sys.modules[COLLISION_SPEC.name] = collision
COLLISION_SPEC.loader.exec_module(collision)


@pytest.mark.parametrize(
    ("semver", "pep440", "prerelease"),
    (
        ("1.0.0-alpha.2", "1.0.0a2", True),
        ("1.0.0-beta.2", "1.0.0b2", True),
        ("1.0.0-rc.2", "1.0.0rc2", True),
        ("1.0.0-beta.12", "1.0.0b12", True),
        ("1.0.0", "1.0.0", False),
        ("1.0.1", "1.0.1", False),
        ("1.1.0", "1.1.0", False),
        ("2.0.0", "2.0.0", False),
    ),
)
def test_release_mapping_roundtrips(
    semver: str, pep440: str, prerelease: bool
) -> None:
    from_semver = ReleaseVersion.from_semver(semver)
    from_pep440 = ReleaseVersion.from_pep440(pep440)

    assert from_semver == from_pep440
    assert from_semver.semver == semver
    assert from_semver.pep440 == pep440
    assert from_semver.tag == f"v{semver}"
    assert from_semver.release_note == f"docs/releases/v{semver}.md"
    assert from_semver.prerelease is prerelease
    assert ReleaseVersion.from_tag(from_semver.tag) == from_semver


@pytest.mark.parametrize(
    "value",
    (
        "",
        "v1.0.0",
        "1.0",
        "1.0.0-beta",
        "1.0.0-beta.01",
        "01.0.0",
        "1.0.0-dev.1",
        "1.0.0-preview.1",
        "1.0.0-beta.1+build.7",
        "1.0.0+build.7",
    ),
)
def test_unsupported_semver_identifiers_fail_closed(value: str) -> None:
    with pytest.raises(VersionPolicyError):
        ReleaseVersion.from_semver(value)


@pytest.mark.parametrize(
    "value",
    (
        "",
        "v1.0.0b2",
        "1.0",
        "1.0.0b",
        "1.0.0b01",
        "01.0.0",
        "1.0.0.dev1",
        "1.0.0.post1",
        "1.0.0+local",
        "1.0.0beta2",
    ),
)
def test_unsupported_pep440_identifiers_fail_closed(value: str) -> None:
    with pytest.raises(VersionPolicyError):
        ReleaseVersion.from_pep440(value)


def test_current_project_metadata_agrees_through_authoritative_parser() -> None:
    release = release_version.validate_project_versions(ROOT)

    assert release.semver == "1.0.0-beta.2"
    assert release.pep440 == "1.0.0b2"
    assert release.tag == "v1.0.0-beta.2"
    release_version.validate_github_ref(release, f"refs/tags/{release.tag}")
    release_version.validate_github_ref(release, "refs/heads/release-candidate")
    with pytest.raises(VersionPolicyError, match="canonical SemVer tag"):
        release_version.validate_github_ref(release, "refs/tags/v1.0.0b2")


def test_cli_exports_parsed_release_outputs(tmp_path: Path) -> None:
    output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--check-project",
            "--project-root",
            str(ROOT),
            "--github-ref",
            "refs/tags/v1.0.0-beta.2",
            "--github-output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "pep440": "1.0.0b2",
        "prerelease": "true",
        "release_note": "docs/releases/v1.0.0-beta.2.md",
        "semver": "1.0.0-beta.2",
        "tag": "v1.0.0-beta.2",
    }
    assert dict(
        line.split("=", 1)
        for line in output.read_text(encoding="utf-8").splitlines()
    ) == json.loads(result.stdout)


@pytest.mark.parametrize(
    ("tag", "pep440"),
    (
        ("v1.0.0-alpha.12", "1.0.0a12"),
        ("v1.0.0-beta.2", "1.0.0b2"),
        ("v1.0.0-rc.3", "1.0.0rc3"),
        ("v1.2.3", "1.2.3"),
    ),
)
def test_semver_tag_selects_only_mapped_pep440_artifacts(
    tag: str, pep440: str, tmp_path: Path
) -> None:
    release = ReleaseVersion.from_tag(tag)
    good = tmp_path / f"gafime-{pep440}.tar.gz"
    good.write_bytes(b"release")
    observed: list[tuple[str, str]] = []

    def absent(project: str, version: str) -> None:
        observed.append((project, version))
        return None

    assert collision.validate_artifacts(
        tmp_path,
        release.pep440,
        absent,
        allow_matching_existing=False,
    ) == (1, 0)
    assert observed == [("gafime", pep440)]

    good.unlink()
    wrong_version = (
        release.semver if release.prerelease else f"{release.semver}+local"
    )
    (tmp_path / f"gafime-{wrong_version}.tar.gz").write_bytes(b"wrong spelling")
    with pytest.raises(collision.CollisionError, match="unexpected release artifact"):
        collision.validate_artifacts(
            tmp_path,
            release.pep440,
            absent,
            allow_matching_existing=False,
        )
