#!/usr/bin/env python3
"""Strict GAFIME release-version mapping and source metadata validation."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
_SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:-(?P<channel>alpha|beta|rc)\.(?P<serial>0|[1-9][0-9]*))?$"
)
_PEP440_PATTERN = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:(?P<channel>a|b|rc)(?P<serial>0|[1-9][0-9]*))?$"
)
_SEMVER_TO_PEP440 = {"alpha": "a", "beta": "b", "rc": "rc"}
_PEP440_TO_SEMVER = {value: key for key, value in _SEMVER_TO_PEP440.items()}


class VersionPolicyError(ValueError):
    """A release identifier is unsupported or project metadata has drifted."""


@dataclass(frozen=True)
class ReleaseVersion:
    major: int
    minor: int
    patch: int
    channel: str | None = None
    serial: int | None = None

    def __post_init__(self) -> None:
        if min(self.major, self.minor, self.patch) < 0:
            raise VersionPolicyError("release components must be non-negative")
        if (self.channel is None) != (self.serial is None):
            raise VersionPolicyError(
                "prerelease channel and serial must either both exist or both be absent"
            )
        if self.channel is not None and self.channel not in _SEMVER_TO_PEP440:
            raise VersionPolicyError(
                f"unsupported prerelease channel {self.channel!r}"
            )
        if self.serial is not None and self.serial < 0:
            raise VersionPolicyError("prerelease serial must be non-negative")

    @classmethod
    def from_semver(cls, value: str) -> "ReleaseVersion":
        match = _SEMVER_PATTERN.fullmatch(value)
        if match is None:
            raise VersionPolicyError(
                f"unsupported SemVer release identifier {value!r}; expected "
                "MAJOR.MINOR.PATCH[-alpha.N|-beta.N|-rc.N]"
            )
        return cls(
            major=int(match["major"]),
            minor=int(match["minor"]),
            patch=int(match["patch"]),
            channel=match["channel"],
            serial=int(match["serial"]) if match["serial"] is not None else None,
        )

    @classmethod
    def from_pep440(cls, value: str) -> "ReleaseVersion":
        match = _PEP440_PATTERN.fullmatch(value)
        if match is None:
            raise VersionPolicyError(
                f"unsupported PEP 440 release identifier {value!r}; expected "
                "MAJOR.MINOR.PATCH[aN|bN|rcN]"
            )
        pep440_channel = match["channel"]
        return cls(
            major=int(match["major"]),
            minor=int(match["minor"]),
            patch=int(match["patch"]),
            channel=(
                _PEP440_TO_SEMVER[pep440_channel]
                if pep440_channel is not None
                else None
            ),
            serial=int(match["serial"]) if match["serial"] is not None else None,
        )

    @classmethod
    def from_tag(cls, value: str) -> "ReleaseVersion":
        if not value.startswith("v"):
            raise VersionPolicyError(
                f"release tag {value!r} must use the v<SemVer> form"
            )
        return cls.from_semver(value[1:])

    @property
    def semver(self) -> str:
        release = f"{self.major}.{self.minor}.{self.patch}"
        if self.channel is None:
            return release
        return f"{release}-{self.channel}.{self.serial}"

    @property
    def pep440(self) -> str:
        release = f"{self.major}.{self.minor}.{self.patch}"
        if self.channel is None:
            return release
        return f"{release}{_SEMVER_TO_PEP440[self.channel]}{self.serial}"

    @property
    def tag(self) -> str:
        return f"v{self.semver}"

    @property
    def release_note(self) -> str:
        return f"docs/releases/{self.tag}.md"

    @property
    def prerelease(self) -> bool:
        return self.channel is not None

    def as_dict(self) -> dict[str, str]:
        return {
            "semver": self.semver,
            "pep440": self.pep440,
            "tag": self.tag,
            "release_note": self.release_note,
            "prerelease": str(self.prerelease).lower(),
        }


def _read_python_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(values) != 1:
        raise VersionPolicyError(
            f"{path} must contain exactly one literal __version__ assignment"
        )
    return values[0]


def validate_project_versions(root: Path = ROOT) -> ReleaseVersion:
    """Validate every source metadata identity and return the canonical release."""

    root = root.resolve()
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    release = ReleaseVersion.from_semver(
        str(cargo["workspace"]["package"]["version"])
    )
    pyproject = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    python_version = str(pyproject["project"]["version"])
    if python_version != release.pep440:
        raise VersionPolicyError(
            f"pyproject version {python_version!r} must equal mapped PEP 440 "
            f"version {release.pep440!r}"
        )
    runtime_version = _read_python_version(root / "python" / "gafime" / "_version.py")
    if runtime_version != release.pep440:
        raise VersionPolicyError(
            f"Python runtime version {runtime_version!r} must equal "
            f"{release.pep440!r}"
        )

    member_names: set[str] = set()
    for member in cargo["workspace"]["members"]:
        manifest_path = root / str(member) / "Cargo.toml"
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        package = manifest["package"]
        if package.get("version", {}).get("workspace") is not True:
            raise VersionPolicyError(
                f"{manifest_path.relative_to(root)} must inherit workspace version"
            )
        member_names.add(str(package["name"]))

    lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    locked_versions = {
        str(package["name"]): str(package["version"])
        for package in lock["package"]
        if str(package["name"]) in member_names
    }
    if set(locked_versions) != member_names:
        missing = sorted(member_names - set(locked_versions))
        raise VersionPolicyError(f"Cargo.lock is missing workspace crates: {missing}")
    drifted = {
        name: version
        for name, version in locked_versions.items()
        if version != release.semver
    }
    if drifted:
        raise VersionPolicyError(
            f"Cargo.lock workspace versions must equal {release.semver!r}: {drifted}"
        )

    optional = pyproject["project"].get("optional-dependencies", {})
    mismatched_requirements = [
        requirement
        for requirements in optional.values()
        for requirement in requirements
        if requirement.startswith("gafime-")
        and f"=={release.pep440}" not in requirement
    ]
    if mismatched_requirements:
        raise VersionPolicyError(
            "same-release payload requirements must use the mapped PEP 440 "
            f"version {release.pep440!r}: {mismatched_requirements}"
        )

    release_note = root / release.release_note
    if not release_note.is_file():
        raise VersionPolicyError(
            f"SemVer release note is missing: {release_note.relative_to(root)}"
        )
    note_text = release_note.read_text(encoding="utf-8")
    for identity in (release.tag, release.semver, release.pep440):
        if identity not in note_text:
            raise VersionPolicyError(
                f"{release_note.relative_to(root)} must expose release identity "
                f"{identity!r}"
            )
    return release


def validate_github_ref(release: ReleaseVersion, github_ref: str) -> None:
    if not github_ref.startswith("refs/tags/"):
        return
    actual = github_ref.removeprefix("refs/tags/")
    if actual != release.tag:
        raise VersionPolicyError(
            f"release tag {actual!r} must equal canonical SemVer tag {release.tag!r}"
        )


def _write_github_output(path: Path, release: ReleaseVersion) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in release.as_dict().items():
            output.write(f"{key}={value}\n")


def _parse_explicit(args: argparse.Namespace) -> ReleaseVersion | None:
    if args.semver is not None:
        return ReleaseVersion.from_semver(args.semver)
    if args.pep440 is not None:
        return ReleaseVersion.from_pep440(args.pep440)
    if args.tag is not None:
        return ReleaseVersion.from_tag(args.tag)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--semver")
    source.add_argument("--pep440")
    source.add_argument("--tag")
    parser.add_argument("--check-project", action="store_true")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--github-ref", default="")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    explicit = _parse_explicit(args)
    if explicit is not None and args.check_project:
        parser.error("explicit version input and --check-project are mutually exclusive")
    release = (
        validate_project_versions(args.project_root)
        if args.check_project or explicit is None
        else explicit
    )
    validate_github_ref(release, args.github_ref)
    if args.github_output is not None:
        _write_github_output(args.github_output, release)
    print(json.dumps(release.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
