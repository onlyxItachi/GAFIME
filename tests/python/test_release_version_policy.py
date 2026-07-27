from __future__ import annotations

import importlib.util
from pathlib import Path
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
