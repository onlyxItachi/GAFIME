#!/usr/bin/env python3
"""Validate source-build policy and release archive composition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Iterable
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
LICENSE_EXPRESSION = "Apache-2.0"
PAYLOAD_IDENTITIES = {
    "gafime-cuda": ("cuda", "gafime_cuda", "off"),
    "gafime-cuda-rt": ("cuda", "gafime_cuda_rt", "on"),
    "gafime-rocm": ("rocm", "gafime_rocm", None),
}
DISTRIBUTIONS = ("gafime", *PAYLOAD_IDENTITIES)
CORE_WHEEL_PLATFORMS = {
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_11_0_arm64",
    "win_amd64",
    "win_arm64",
}
CUDA_WHEEL_PLATFORMS = {"manylinux_2_28_x86_64", "win_amd64"}
ROCM_WHEEL_PLATFORMS = {"manylinux_2_28_x86_64"}
CUDA_RT_WHEEL_PLATFORMS = {"manylinux_2_28_x86_64"}
CORE_CMAKE_TEST_SOURCES = {
    "tests/gpu/cuda_rt_same_device_concurrency.cpp",
    "tests/gpu/cuda_rt_state_policy_test.cpp",
    "tests/gpu/cuda_v1_abi_smoke.cpp",
}
CUDA_SDIST_SOURCES = {
    "src/common/gafime_gpu_abi.hpp",
    "src/common/gpu_abi_impl.hpp",
    "src/cuda/cuda_api.hpp",
    "src/cuda/kernels.cu",
    "src/cuda/kernels.cuh",
    "src/cuda/launcher.cu",
    "src/cuda/rt_kernels.cu",
    "src/cuda/rt_kernels.cuh",
    "src/cuda/rt_launcher.cu",
    "src/cuda/rt_launcher.cuh",
}
ROCM_SDIST_SOURCES = {
    "src/common/gafime_gpu_abi.hpp",
    "src/common/gpu_abi_impl.hpp",
    "src/rocm/kernels.hip",
    "src/rocm/kernels.hpp",
    "src/rocm/launcher.hip",
    "src/rocm/rocm_api.hpp",
}


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    distribution: str
    version: str
    metadata: object
    members: frozenset[str]
    platforms: frozenset[str] = frozenset()
    build_policy: dict[str, object] | None = None
    build_provenance: dict[str, object] | None = None


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _project_version(root: Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _metadata_from_text(text: str, path: Path) -> tuple[str, str, object]:
    metadata = Parser().parsestr(text)
    name = metadata.get("Name")
    version = metadata.get("Version")
    _require(bool(name), f"{path.name} metadata has no Name")
    _require(bool(version), f"{path.name} metadata has no Version")
    return _canonical_name(str(name)), str(version), metadata


def _read_wheel(path: Path) -> Artifact:
    with zipfile.ZipFile(path) as archive:
        members = frozenset(name.rstrip("/") for name in archive.namelist())
        candidates = sorted(
            name for name in members if name.endswith(".dist-info/METADATA")
        )
        _require(len(candidates) == 1, f"{path.name} must contain one METADATA file")
        metadata_text = archive.read(candidates[0]).decode("utf-8")
        wheel_candidates = sorted(
            name for name in members if name.endswith(".dist-info/WHEEL")
        )
        _require(len(wheel_candidates) == 1, f"{path.name} must contain one WHEEL file")
        wheel_metadata = Parser().parsestr(
            archive.read(wheel_candidates[0]).decode("utf-8")
        )
        policy_names = sorted(
            f"{package}/build_policy.json"
            for backend, package, _ in PAYLOAD_IDENTITIES.values()
            if backend == "cuda" and f"{package}/build_policy.json" in members
        )
        _require(
            len(policy_names) <= 1,
            f"{path.name} contains multiple CUDA build policies: {policy_names}",
        )
        build_policy = (
            json.loads(archive.read(policy_names[0]).decode("utf-8"))
            if policy_names
            else None
        )
        provenance_names = sorted(
            f"{package}/build_provenance.json"
            for backend, package, _ in PAYLOAD_IDENTITIES.values()
            if backend == "cuda" and f"{package}/build_provenance.json" in members
        )
        _require(
            len(provenance_names) <= 1,
            f"{path.name} contains multiple CUDA build provenance records: "
            f"{provenance_names}",
        )
        build_provenance = (
            json.loads(archive.read(provenance_names[0]).decode("utf-8"))
            if provenance_names
            else None
        )
    distribution, version, metadata = _metadata_from_text(metadata_text, path)
    try:
        _, python_tag, abi_tag, platform_tag = path.name[:-4].rsplit("-", 3)
    except ValueError as exc:
        raise AssertionError(f"invalid wheel filename: {path.name}") from exc
    python_tags = frozenset(python_tag.split("."))
    abi_tags = frozenset(abi_tag.split("."))
    platform_tags = frozenset(platform_tag.split("."))
    _require(
        python_tags == {"cp310"} and abi_tags == {"abi3"},
        f"{path.name} must use the cp310-abi3 compatibility contract",
    )
    filename_tags = {
        f"{python}-{abi}-{platform}"
        for python in python_tags
        for abi in abi_tags
        for platform in platform_tags
    }
    wheel_tags = set(wheel_metadata.get_all("Tag", []))
    _require(
        wheel_tags == filename_tags,
        f"{path.name} filename tags {sorted(filename_tags)} do not match "
        f"internal WHEEL tags {sorted(wheel_tags)}",
    )
    return Artifact(
        path=path,
        kind="wheel",
        distribution=distribution,
        version=version,
        metadata=metadata,
        members=members,
        platforms=platform_tags,
        build_policy=build_policy,
        build_provenance=build_provenance,
    )


def _read_sdist(path: Path) -> Artifact:
    with tarfile.open(path, "r:gz") as archive:
        raw_members = frozenset(
            member.name.rstrip("/") for member in archive.getmembers()
        )
        roots = {name.split("/", 1)[0] for name in raw_members if name}
        _require(len(roots) == 1, f"{path.name} must contain one top-level directory")
        root = next(iter(roots))
        members = frozenset(
            name[len(root) + 1 :]
            for name in raw_members
            if name.startswith(f"{root}/") and len(name) > len(root) + 1
        )
        metadata_name = f"{root}/PKG-INFO"
        _require(metadata_name in raw_members, f"{path.name} has no root PKG-INFO")
        extracted = archive.extractfile(metadata_name)
        _require(extracted is not None, f"unable to read {metadata_name}")
        metadata_text = extracted.read().decode("utf-8")
        policy_names = sorted(
            f"{root}/{package}/build_policy.json"
            for backend, package, _ in PAYLOAD_IDENTITIES.values()
            if backend == "cuda"
            and f"{root}/{package}/build_policy.json" in raw_members
        )
        _require(
            len(policy_names) <= 1,
            f"{path.name} contains multiple CUDA build policies: {policy_names}",
        )
        policy_file = archive.extractfile(policy_names[0]) if policy_names else None
        build_policy = (
            json.loads(policy_file.read().decode("utf-8")) if policy_file else None
        )
        provenance_names = sorted(
            f"{root}/{package}/build_provenance.json"
            for backend, package, _ in PAYLOAD_IDENTITIES.values()
            if backend == "cuda"
            and f"{root}/{package}/build_provenance.json" in raw_members
        )
        _require(
            len(provenance_names) <= 1,
            f"{path.name} contains multiple CUDA build provenance records: "
            f"{provenance_names}",
        )
        provenance_file = (
            archive.extractfile(provenance_names[0]) if provenance_names else None
        )
        build_provenance = (
            json.loads(provenance_file.read().decode("utf-8"))
            if provenance_file
            else None
        )
    distribution, version, metadata = _metadata_from_text(metadata_text, path)
    return Artifact(
        path=path,
        kind="sdist",
        distribution=distribution,
        version=version,
        metadata=metadata,
        members=members,
        build_policy=build_policy,
        build_provenance=build_provenance,
    )


def _discover_artifacts(path: Path) -> list[Artifact]:
    paths: list[Path]
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and (candidate.name.endswith(".whl") or candidate.name.endswith(".tar.gz"))
        )
    _require(bool(paths), f"no wheels or source distributions found under {path}")
    names = [candidate.name for candidate in paths]
    _require(
        len(names) == len(set(names)), f"duplicate artifact filenames found: {names}"
    )
    artifacts = [
        _read_wheel(candidate)
        if candidate.name.endswith(".whl")
        else _read_sdist(candidate)
        for candidate in paths
    ]
    unknown = sorted(
        {artifact.distribution for artifact in artifacts} - set(DISTRIBUTIONS)
    )
    _require(not unknown, f"unexpected distributions in release artifacts: {unknown}")
    return artifacts


def _assert_license(artifact: Artifact) -> None:
    metadata = artifact.metadata
    expression = metadata.get("License-Expression")
    _require(
        expression == LICENSE_EXPRESSION,
        f"{artifact.path.name} License-Expression is {expression!r}, expected {LICENSE_EXPRESSION!r}",
    )
    license_files = metadata.get_all("License-File", [])
    _require(
        any(PurePosixPath(value).name == "LICENSE" for value in license_files),
        f"{artifact.path.name} metadata does not declare LICENSE",
    )
    _require(
        any(PurePosixPath(member).name == "LICENSE" for member in artifact.members),
        f"{artifact.path.name} does not contain LICENSE",
    )


def _assert_common_metadata(artifact: Artifact, version: str) -> None:
    _require(
        artifact.version == version,
        f"{artifact.path.name} metadata version {artifact.version!r} != {version!r}",
    )
    _assert_license(artifact)
    if artifact.distribution == "gafime":
        return
    requirements = {
        requirement.split(";", 1)[0].strip()
        for requirement in artifact.metadata.get_all("Requires-Dist", [])
    }
    _require(
        f"gafime=={version}" in requirements,
        f"{artifact.path.name} must require the exact base gafime version {version}",
    )


def _assert_core_sdist(artifact: Artifact, root: Path) -> None:
    _require(
        artifact.kind == "sdist" and artifact.distribution == "gafime",
        f"expected core sdist, found {artifact.path.name}",
    )
    expected_native = {
        path.relative_to(root).as_posix()
        for path in (root / "src").rglob("*")
        if path.is_file()
    }
    expected = (
        expected_native
        | CORE_CMAKE_TEST_SOURCES
        | {
            "Cargo.lock",
            "Cargo.toml",
            "LICENSE",
            "README.md",
            "pyproject.toml",
        }
    )
    missing = sorted(expected - artifact.members)
    _require(not missing, f"core sdist is missing reproducibility sources: {missing}")
    packaged_gpu_tests = {
        member
        for member in artifact.members
        if member.startswith("tests/gpu/") and member.endswith(".cpp")
    }
    _require(
        packaged_gpu_tests == CORE_CMAKE_TEST_SOURCES,
        f"core sdist GPU test sources {sorted(packaged_gpu_tests)} != "
        f"{sorted(CORE_CMAKE_TEST_SOURCES)}",
    )


def _payload_identity(distribution: str) -> tuple[str, str, str | None]:
    _require(
        distribution in PAYLOAD_IDENTITIES,
        f"unknown payload distribution identity: {distribution}",
    )
    return PAYLOAD_IDENTITIES[distribution]


def _native_library_names(package: str) -> set[str]:
    return {
        f"{package}.dll",
        f"lib{package}.so",
        f"{package}.so",
        f"{package}.pyd",
    }


def _assert_payload_sdist(artifact: Artifact, expected_distribution: str) -> None:
    backend, package, _ = _payload_identity(expected_distribution)
    expected_sources = CUDA_SDIST_SOURCES if backend == "cuda" else ROCM_SDIST_SOURCES
    _require(
        artifact.kind == "sdist" and artifact.distribution == expected_distribution,
        f"expected {expected_distribution} sdist, found {artifact.path.name}",
    )
    expected = expected_sources | {
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "setup.py",
        f"{package}/__init__.py",
    }
    missing = sorted(expected - artifact.members)
    _require(not missing, f"{expected_distribution} sdist is missing files: {missing}")
    other_backend = "rocm" if backend == "cuda" else "cuda"
    other_packages = {
        f"{other_package}/"
        for distribution, (_, other_package, _) in PAYLOAD_IDENTITIES.items()
        if distribution != expected_distribution
    }
    leaked = sorted(
        member
        for member in artifact.members
        if member.startswith(f"src/{other_backend}/")
        or any(member.startswith(prefix) for prefix in other_packages)
    )
    _require(
        not leaked,
        f"{expected_distribution} sdist contains another payload variant: {leaked}",
    )


def _assert_cuda_build_policy(artifact: Artifact, expected_rt_mode: str) -> None:
    expected = {
        "cuda_architectures": ["75", "80", "86", "89", "90", "100", "120"],
        "cuda_tuning_policy": "package-wide-sm89",
        "cuda_tuning_sm": 89,
        "optix_rt": expected_rt_mode,
        "per_architecture_tuning": False,
    }
    _require(
        artifact.build_policy == expected,
        f"{artifact.path.name} CUDA build policy {artifact.build_policy!r} != {expected!r}",
    )
    bundled_optix = sorted(
        member
        for member in artifact.members
        if PurePosixPath(member).name == "optix.h" or ".optix-sdk/" in member
    )
    _require(
        not bundled_optix,
        f"CUDA artifacts must not redistribute OptiX SDK content: {bundled_optix}",
    )
    if expected_rt_mode == "off":
        _require(
            artifact.build_provenance is None,
            f"{artifact.path.name} standard CUDA artifact contains RT provenance",
        )
        return

    provenance = artifact.build_provenance
    _require(
        isinstance(provenance, dict),
        f"{artifact.path.name} RT artifact has no build provenance",
    )
    _require(
        set(provenance) == {"cuda_image", "optix_sdk_archive_sha256"},
        f"{artifact.path.name} RT provenance fields are invalid: {provenance!r}",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(provenance["optix_sdk_archive_sha256"]))
        is not None,
        f"{artifact.path.name} OptiX SDK digest is not canonical SHA-256",
    )
    _require(
        re.fullmatch(
            r"[^@\s]+@sha256:[0-9a-f]{64}", str(provenance["cuda_image"])
        )
        is not None,
        f"{artifact.path.name} CUDA image is not digest pinned",
    )


def _assert_core_wheel(artifact: Artifact) -> None:
    _require(
        artifact.kind == "wheel" and artifact.distribution == "gafime",
        f"expected core wheel, found {artifact.path.name}",
    )
    payload_packages = tuple(
        f"{package}/" for _, package, _ in PAYLOAD_IDENTITIES.values()
    )
    payload_libraries = {
        name
        for _, package, _ in PAYLOAD_IDENTITIES.values()
        for name in _native_library_names(package)
    }
    leaked = sorted(
        member
        for member in artifact.members
        if member.startswith(payload_packages)
        or PurePosixPath(member).name in payload_libraries
    )
    _require(not leaked, f"core wheel contains vendor payload files: {leaked}")
    if "macosx_11_0_arm64" in artifact.platforms:
        expected_metal = {
            "gafime/_metal/gafime_metal_v1.metallib",
            "gafime/_metal/libgafime_metal_v1.dylib",
        }
        missing = sorted(expected_metal - artifact.members)
        _require(
            not missing, f"macOS core wheel is missing Metal payload files: {missing}"
        )


def _assert_payload_wheel(artifact: Artifact, expected_distribution: str) -> None:
    _, package, _ = _payload_identity(expected_distribution)
    _require(
        artifact.kind == "wheel" and artifact.distribution == expected_distribution,
        f"expected {expected_distribution} wheel, found {artifact.path.name}",
    )
    _require(
        any(member.startswith(f"{package}/") for member in artifact.members),
        f"{artifact.path.name} does not contain {package}",
    )
    native_names = _native_library_names(package)
    native_members = sorted(
        member
        for member in artifact.members
        if PurePosixPath(member).name in native_names
    )
    _require(
        len(native_members) == 1,
        f"{artifact.path.name} must contain exactly one native payload library; "
        f"found {native_members}",
    )
    other_packages = {
        f"{other_package}/"
        for distribution, (_, other_package, _) in PAYLOAD_IDENTITIES.items()
        if distribution != expected_distribution
    }
    other_native_names = {
        name
        for distribution, (_, other_package, _) in PAYLOAD_IDENTITIES.items()
        if distribution != expected_distribution
        for name in _native_library_names(other_package)
    }
    leaked = sorted(
        member
        for member in artifact.members
        if any(member.startswith(prefix) for prefix in other_packages)
        or PurePosixPath(member).name in other_native_names
    )
    _require(
        not leaked, f"{artifact.path.name} contains another payload variant: {leaked}"
    )


def _select(
    artifacts: Iterable[Artifact], distribution: str, kind: str
) -> list[Artifact]:
    return [
        artifact
        for artifact in artifacts
        if artifact.distribution == distribution and artifact.kind == kind
    ]


def _assert_one(items: list[Artifact], label: str) -> Artifact:
    _require(len(items) == 1, f"expected exactly one {label}, found {len(items)}")
    return items[0]


def _assert_wheel_platforms(
    artifacts: list[Artifact], distribution: str, expected: set[str]
) -> None:
    wheels = _select(artifacts, distribution, "wheel")
    matched_platforms = []
    for artifact in wheels:
        matches = artifact.platforms & expected
        _require(
            len(matches) == 1,
            f"{artifact.path.name} platform tags {sorted(artifact.platforms)} must "
            f"match exactly one release platform from {sorted(expected)}",
        )
        matched_platforms.append(next(iter(matches)))
    actual = set(matched_platforms)
    _require(
        len(matched_platforms) == len(actual),
        f"{distribution} has duplicate release wheel platforms",
    )
    _require(
        actual == expected,
        f"{distribution} wheel platforms {sorted(actual)} != {sorted(expected)}",
    )


def _assert_scope(
    artifacts: list[Artifact], scope: str, root: Path, version: str
) -> None:
    for artifact in artifacts:
        _assert_common_metadata(artifact, version)
        if artifact.kind == "wheel" and artifact.distribution == "gafime":
            _assert_core_wheel(artifact)
        elif artifact.kind == "wheel":
            _assert_payload_wheel(artifact, artifact.distribution)

    expected_count: int
    if scope == "core-sdist":
        expected_count = 1
        _assert_core_sdist(_assert_one(artifacts, "core sdist"), root)
    elif scope == "core-wheel":
        expected_count = len(artifacts)
        _require(expected_count > 0, "no core wheels found")
        for artifact in artifacts:
            _assert_core_wheel(artifact)
    elif scope in {"cuda-sdist", "rocm-sdist"}:
        expected_count = 1
        backend = scope.removesuffix("-sdist")
        _assert_payload_sdist(
            _assert_one(artifacts, f"{backend} sdist"), f"gafime-{backend}"
        )
        if backend == "cuda":
            _assert_cuda_build_policy(artifacts[0], "off")
    elif scope == "cuda-rt-sdist":
        expected_count = 1
        _assert_payload_sdist(_assert_one(artifacts, "CUDA RT sdist"), "gafime-cuda-rt")
        _assert_cuda_build_policy(artifacts[0], "on")
    elif scope in {"cuda-wheel", "rocm-wheel"}:
        backend = scope.removesuffix("-wheel")
        expected_count = len(artifacts)
        _require(expected_count > 0, f"no {backend} wheels found")
        for artifact in artifacts:
            _assert_payload_wheel(artifact, f"gafime-{backend}")
            if backend == "cuda":
                _assert_cuda_build_policy(artifact, "off")
    elif scope == "cuda-rt-wheel":
        expected_count = len(artifacts)
        _require(expected_count > 0, "no CUDA RT wheels found")
        for artifact in artifacts:
            _assert_payload_wheel(artifact, "gafime-cuda-rt")
            _assert_cuda_build_policy(artifact, "on")
    elif scope == "sdists":
        expected_count = 3
        _assert_core_sdist(
            _assert_one(_select(artifacts, "gafime", "sdist"), "core sdist"), root
        )
        for backend in ("cuda", "rocm"):
            payload_sdist = _assert_one(
                _select(artifacts, f"gafime-{backend}", "sdist"),
                f"{backend} sdist",
            )
            _assert_payload_sdist(payload_sdist, f"gafime-{backend}")
            if backend == "cuda":
                _assert_cuda_build_policy(payload_sdist, "off")
    elif scope == "core-release":
        expected_count = 6
        _assert_wheel_platforms(artifacts, "gafime", CORE_WHEEL_PLATFORMS)
        _assert_core_sdist(
            _assert_one(_select(artifacts, "gafime", "sdist"), "core sdist"), root
        )
    elif scope == "cuda-release":
        expected_count = 3
        _assert_wheel_platforms(artifacts, "gafime-cuda", CUDA_WHEEL_PLATFORMS)
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-cuda", "sdist"), "CUDA sdist"),
            "gafime-cuda",
        )
        for artifact in artifacts:
            _assert_cuda_build_policy(artifact, "off")
    elif scope == "cuda-rt-release":
        expected_count = 2
        _assert_wheel_platforms(artifacts, "gafime-cuda-rt", CUDA_RT_WHEEL_PLATFORMS)
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-cuda-rt", "sdist"), "CUDA RT sdist"),
            "gafime-cuda-rt",
        )
        for artifact in artifacts:
            _assert_cuda_build_policy(artifact, "on")
    elif scope == "rocm-release":
        expected_count = 2
        _assert_wheel_platforms(artifacts, "gafime-rocm", ROCM_WHEEL_PLATFORMS)
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-rocm", "sdist"), "ROCm sdist"),
            "gafime-rocm",
        )
    elif scope == "full-release":
        expected_count = 11
        _assert_scope(
            _select_distribution(artifacts, "gafime"), "core-release", root, version
        )
        _assert_scope(
            _select_distribution(artifacts, "gafime-cuda"),
            "cuda-release",
            root,
            version,
        )
        _assert_scope(
            _select_distribution(artifacts, "gafime-rocm"),
            "rocm-release",
            root,
            version,
        )
    else:
        raise AssertionError(f"unsupported artifact scope: {scope}")
    _require(
        len(artifacts) == expected_count,
        f"{scope} expected {expected_count} artifacts, found {len(artifacts)}",
    )


def _select_distribution(
    artifacts: Iterable[Artifact], distribution: str
) -> list[Artifact]:
    return [artifact for artifact in artifacts if artifact.distribution == distribution]


def _assert_release_tag(
    root: Path, version: str, github_ref: str, git_sha: str | None, required: bool
) -> None:
    expected = f"v{version}"
    if github_ref.startswith("refs/tags/"):
        actual = github_ref.removeprefix("refs/tags/")
        _require(actual == expected, f"release tag {actual!r} must equal {expected!r}")
    elif required:
        command = ["git", "tag", "--points-at", git_sha or "HEAD", "--list", "v*"]
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        tags = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        _require(
            expected in tags,
            f"publishing requires commit {git_sha or 'HEAD'} to carry tag {expected}; found {sorted(tags)}",
        )
    if github_ref.startswith("refs/tags/") or required:
        release_note = root / "docs" / "releases" / f"{expected}.md"
        _require(release_note.is_file(), f"release note is missing: {release_note}")


def _write_checksums(artifacts: list[Artifact], output: Path) -> None:
    lines = []
    for artifact in sorted(artifacts, key=lambda item: item.path.name):
        digest = hashlib.sha256(artifact.path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {artifact.path.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def _assert_source_tree(root: Path) -> None:
    dockerfiles = (root / "Dockerfile", root / "Dockerfile.smoketest")
    for path in dockerfiles:
        text = path.read_text(encoding="utf-8")
        _require(
            "--no-build-isolation" in text,
            f"{path.name} must exercise no-build-isolation",
        )
        _require('"maturin>=1.7,<2"' in text, f"{path.name} must install Maturin")
        _require("1.89.0" in text, f"{path.name} must pin Rust 1.89.0")
        _require(
            'CMD ["gafime", "--check"' in text, f"{path.name} must use the gafime CLI"
        )
        _require(
            "python -m gafime --check" not in text,
            f"{path.name} uses missing gafime.__main__",
        )

    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    _require(
        "recursive-include src *" in manifest, "MANIFEST.in must include native src"
    )
    _require("prune src" not in manifest, "MANIFEST.in must not prune native src")
    manifest_lines = {line.strip() for line in manifest.splitlines()}
    missing_manifest_tests = sorted(
        f"include {source}"
        for source in CORE_CMAKE_TEST_SOURCES
        if f"include {source}" not in manifest_lines
    )
    _require(
        not missing_manifest_tests,
        f"MANIFEST.in is missing CMake GPU test entries: {missing_manifest_tests}",
    )
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    _require(
        '{ path = "src/**/*", format = "sdist" }' in pyproject,
        "Maturin sdist policy must include all native src files",
    )
    pyproject_data = tomllib.loads(pyproject)
    sdist_patterns = {
        str(entry["path"])
        for entry in pyproject_data["tool"]["maturin"].get("include", [])
        if entry.get("format") == "sdist"
    }
    available_gpu_tests = {
        path.relative_to(root).as_posix()
        for path in (root / "tests" / "gpu").glob("*.cpp")
    }
    selected_gpu_tests = {
        source
        for source in available_gpu_tests
        if any(PurePosixPath(source).match(pattern) for pattern in sdist_patterns)
    }
    _require(
        selected_gpu_tests == CORE_CMAKE_TEST_SOURCES,
        f"Maturin sdist GPU test sources {sorted(selected_gpu_tests)} != "
        f"{sorted(CORE_CMAKE_TEST_SOURCES)}",
    )

    dockerignore_lines = {
        line.strip()
        for line in (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    }
    _require(
        "Cargo.lock" not in dockerignore_lines,
        ".dockerignore must retain the pinned Rust lockfile in Docker build contexts",
    )

    stage_path = root / ".github" / "scripts" / "stage_gpu_payload.py"
    stage_script = stage_path.read_text(encoding="utf-8")
    for token in (
        'license = "Apache-2.0"',
        'license-files = ["LICENSE"]',
        'output / "LICENSE"',
        'CUDA_RT_BUILD_MODE = "{cuda_rt_mode}"',
        'CUDA_TUNING_POLICY = "package-wide-sm89"',
        "PER_ARCHITECTURE_TUNING = False",
        'f"-DGAFIME_CUDA_TUNING_SM={{CUDA_TUNING_SM}}"',
        'package_name = "gafime_cuda_rt" if cuda_rt else f"gafime_{kind}"',
        'dist_name = "gafime-cuda-rt" if cuda_rt else f"gafime-{kind}"',
    ):
        _require(token in stage_script, f"GPU payload staging is missing {token}")
    _require(
        'choices=("off", "on")' in stage_script,
        "GPU payload staging must expose separate immutable RT-off/RT-on selection",
    )
    with tempfile.TemporaryDirectory(prefix="gafime-invalid-rt-policy-") as temp_dir:
        rejected = subprocess.run(
            [
                sys.executable,
                str(stage_path),
                "cuda",
                str(Path(temp_dir) / "payload"),
                "--cuda-rt",
                "both",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    _require(
        rejected.returncode == 2 and "invalid choice: 'both'" in rejected.stderr,
        "GPU payload staging must reject the ambiguous CUDA RT build mode 'both'",
    )

    build_workflow = (root / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )
    for token in (
        "windows-2025-vs2026",
        "macos-26",
        "RUST_VERSION: '1.89.0'",
        "release_preflight:",
        "name: release-bundle",
        "needs: release_preflight",
        "build_cuda_rt_linux_payload:",
        "GAFIME_OPTIX_SDK_ARCHIVE_URL",
        "--scope cuda-rt-release",
        "gafime_cuda_rt-*-cp310-abi3-*.whl",
        "name: cuda-rt-linux-artifacts",
    ):
        _require(token in build_workflow, f"release workflow is missing {token}")
    _require(
        "publish_pypi_cuda_rt:" not in build_workflow,
        "optional gafime-cuda-rt artifacts must not have a PyPI publishing job",
    )

    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / ".github" / "workflows").glob("*.yml"))
    )
    _require(
        "rustup default stable" not in workflow_text,
        "workflows must not select floating Rust stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=(
            "source-tree",
            "core-sdist",
            "core-wheel",
            "cuda-sdist",
            "cuda-rt-sdist",
            "rocm-sdist",
            "cuda-wheel",
            "cuda-rt-wheel",
            "rocm-wheel",
            "sdists",
            "core-release",
            "cuda-release",
            "cuda-rt-release",
            "rocm-release",
            "full-release",
        ),
        required=True,
    )
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--github-ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--git-sha", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--require-release-tag", action="store_true")
    parser.add_argument("--write-checksums", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    version = _project_version(root)
    if args.scope == "source-tree":
        _assert_source_tree(root)
        print(f"RELEASE SOURCE POLICY: PASS version={version}")
        return

    if args.artifacts is None:
        parser.error("--artifacts is required for archive scopes")
    artifacts = _discover_artifacts(args.artifacts.resolve())
    _assert_scope(artifacts, args.scope, root, version)
    if args.scope == "full-release":
        _assert_release_tag(
            root,
            version,
            args.github_ref,
            args.git_sha,
            args.require_release_tag,
        )
    if args.write_checksums is not None:
        _write_checksums(artifacts, args.write_checksums.resolve())
    print(
        f"RELEASE ARTIFACT COMPOSITION: PASS scope={args.scope} "
        f"version={version} artifacts={len(artifacts)}"
    )


if __name__ == "__main__":
    main()
