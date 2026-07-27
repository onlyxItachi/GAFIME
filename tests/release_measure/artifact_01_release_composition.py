#!/usr/bin/env python3
"""Validate source-build policy and release archive composition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.parser import Parser
from fnmatch import fnmatchcase
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_manifest import load_release_manifest, render_release_matrix

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
RELEASE_MANIFEST = load_release_manifest(ROOT)
LICENSE_EXPRESSION = "Apache-2.0"
CUDA_RT_FIXTURE_IMAGE = (
    "docker.io/nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:"
    "69e9e39eb8fe2cda271654a0f5eac2f1bb946b2fb9c460eb19c7c3c155f4e64e"
)
CUDA_RT_WHEEL_BUILDER_IMAGE = (
    "quay.io/pypa/manylinux_2_28_x86_64@sha256:"
    "a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca"
)
CUDA_RT_RPM_BASE_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64"
)
CUDA_RT_RPM_MANIFEST = ROOT / ".github" / "scripts" / "cuda_13_3_rpms.sha256"
ROCM_BUNDLED_POLICY = ROOT / ".github" / "scripts" / "rocm_7_2_3_bundled_policy.json"
ROCM_SYSTEM_POLICY = ROOT / ".github" / "scripts" / "rocm_7_2_3_system_policy.json"
ROCM_MANYLINUX_IMAGE = (
    "quay.io/pypa/manylinux_2_28_x86_64@sha256:"
    "a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca"
)
ROCM_REPOSITORY = "https://repo.radeon.com/rocm/el8/7.2.3/main"
ROCM_GPG_KEY_URL = "https://repo.radeon.com/rocm/rocm.gpg.key"
ROCM_GPG_KEY_SHA256 = "2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b"
ROCM_BUILD_PACKAGES = (
    "hip-devel7.2.3-7.2.53211.70203-90.el8.x86_64",
    "rocm-device-libs7.2.3-1.0.0.70203-90.el8.x86_64",
    "libstdc++-devel-8.5.0-28.el8_10.alma.1.x86_64",
)
PAYLOAD_IDENTITIES = {
    distribution.name: (
        distribution.backend,
        distribution.package,
        "off" if distribution.policy == "rt-off" else None,
    )
    for distribution in RELEASE_MANIFEST.distributions
    if distribution.backend is not None
}
PAYLOAD_IDENTITIES.update(
    {
        distribution.name: (
            distribution.backend,
            distribution.package,
            "on" if distribution.policy == "rt-on" else None,
        )
        for distribution in RELEASE_MANIFEST.excluded_distributions
    }
)
DISTRIBUTIONS = RELEASE_MANIFEST.all_distribution_names
CORE_GPU_TEST_SOURCES = {
    "tests/gpu/cuda_launch_policy_test.cu",
    "tests/gpu/cuda_rt_decision_path_optix_smoke.cu",
    "tests/gpu/cuda_rt_membership_scale_bench.cpp",
    "tests/gpu/cuda_rt_same_device_concurrency.cpp",
    "tests/gpu/cuda_rt_state_policy_test.cpp",
    "tests/gpu/cuda_spearman_target_cache_bench.cpp",
    "tests/gpu/cuda_v1_abi_smoke.cpp",
    "tests/gpu/rocm_v1_abi_smoke.cpp",
    "tests/gpu/spearman_cache_boundaries.hpp",
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
    try:
        filename_prefix, python_tag, abi_tag, platform_tag = path.name[:-4].rsplit(
            "-", 3
        )
        prefix_parts = filename_prefix.split("-")
        if not path.name.endswith(".whl") or len(prefix_parts) not in (2, 3):
            raise ValueError
        build_tag = prefix_parts[2] if len(prefix_parts) == 3 else ""
    except ValueError as exc:
        raise AssertionError(f"invalid wheel filename: {path.name}") from exc
    _require(
        not build_tag or re.fullmatch(r"[0-9][A-Za-z0-9_]*", build_tag) is not None,
        f"invalid wheel filename: {path.name}: non-canonical build tag {build_tag!r}",
    )
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
            for _, package, _ in PAYLOAD_IDENTITIES.values()
            if f"{package}/build_policy.json" in members
        )
        _require(
            len(policy_names) <= 1,
            f"{path.name} contains multiple payload build policies: {policy_names}",
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
    filename_distribution = _canonical_name(prefix_parts[0])
    filename_version = prefix_parts[1]
    _require(
        filename_distribution == distribution,
        f"{path.name} filename distribution {filename_distribution!r} does not match "
        f"METADATA Name {distribution!r}",
    )
    _require(
        filename_version == version,
        f"{path.name} filename version {filename_version!r} does not match "
        f"METADATA Version {version!r}",
    )
    python_tags = frozenset(python_tag.split("."))
    abi_tags = frozenset(abi_tag.split("."))
    platform_tags = frozenset(platform_tag.split("."))
    expected_python_tags = {RELEASE_MANIFEST.python_tag}
    expected_abi_tags = {RELEASE_MANIFEST.abi_tag}
    _require(
        python_tags == expected_python_tags and abi_tags == expected_abi_tags,
        f"{path.name} must use the {RELEASE_MANIFEST.python_tag}-"
        f"{RELEASE_MANIFEST.abi_tag} compatibility contract",
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
    internal_build = wheel_metadata.get("Build", "")
    _require(
        internal_build == build_tag,
        f"{path.name} filename build tag {build_tag!r} does not match "
        f"internal WHEEL Build {internal_build!r}",
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
            for _, package, _ in PAYLOAD_IDENTITIES.values()
            if f"{root}/{package}/build_policy.json" in raw_members
        )
        _require(
            len(policy_names) <= 1,
            f"{path.name} contains multiple payload build policies: {policy_names}",
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
        | CORE_GPU_TEST_SOURCES
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
        member for member in artifact.members if member.startswith("tests/gpu/")
    }
    _require(
        packaged_gpu_tests == CORE_GPU_TEST_SOURCES,
        f"core sdist GPU test sources {sorted(packaged_gpu_tests)} != "
        f"{sorted(CORE_GPU_TEST_SOURCES)}",
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
    expected_sources = {
        "cuda": CUDA_SDIST_SOURCES,
        "rocm": ROCM_SDIST_SOURCES,
    }[backend]
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
        f"{package}/build_policy.json",
    }
    missing = sorted(expected - artifact.members)
    _require(not missing, f"{expected_distribution} sdist is missing files: {missing}")
    other_packages = {
        f"{other_package}/"
        for distribution, (_, other_package, _) in PAYLOAD_IDENTITIES.items()
        if distribution != expected_distribution
    }
    leaked = sorted(
        member
        for member in artifact.members
        if any(
            member.startswith(f"src/{other_backend}/")
            for other_backend in {"cuda", "rocm", "metal"} - {backend}
        )
        or any(member.startswith(prefix) for prefix in other_packages)
    )
    _require(
        not leaked,
        f"{expected_distribution} sdist contains another payload variant: {leaked}",
    )


def _assert_cuda_build_policy(artifact: Artifact, expected_rt_mode: str) -> None:
    expected = {
        "cuda_architectures": ["75", "80", "86", "89", "90", "100", "120"],
        "cuda_tuning_policy": "runtime-device-class",
        "cuda_tuning_sm": None,
        "optix_rt": expected_rt_mode,
        "per_architecture_tuning": False,
        "runtime_architecture_dispatch": True,
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
    expected_fields = {
        "cuda_fixture_image",
        "cuda_rpm_base_url",
        "cuda_toolkit_rpms",
        "optix_sdk_archive_sha256",
        "wheel_builder_image",
    }
    _require(
        set(provenance) == expected_fields,
        f"{artifact.path.name} RT provenance fields are invalid: {provenance!r}",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(provenance["optix_sdk_archive_sha256"]))
        is not None,
        f"{artifact.path.name} OptiX SDK digest is not canonical SHA-256",
    )
    for image_field in ("cuda_fixture_image", "wheel_builder_image"):
        _require(
            re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", str(provenance[image_field]))
            is not None,
            f"{artifact.path.name} {image_field} is not digest pinned",
        )
    _require(
        provenance["cuda_fixture_image"] == CUDA_RT_FIXTURE_IMAGE,
        f"{artifact.path.name} RT fixture image differs from release policy",
    )
    _require(
        provenance["wheel_builder_image"] == CUDA_RT_WHEEL_BUILDER_IMAGE,
        f"{artifact.path.name} RT wheel-builder image differs from release policy",
    )
    _require(
        re.fullmatch(r"https://[^\s]+", str(provenance["cuda_rpm_base_url"]))
        is not None,
        f"{artifact.path.name} CUDA RPM base URL is not canonical HTTPS",
    )
    _require(
        provenance["cuda_rpm_base_url"] == CUDA_RT_RPM_BASE_URL,
        f"{artifact.path.name} CUDA RPM repository differs from release policy",
    )
    rpm_entries = provenance["cuda_toolkit_rpms"]
    _require(
        isinstance(rpm_entries, list) and bool(rpm_entries),
        f"{artifact.path.name} CUDA RPM provenance must be a non-empty list",
    )
    rpm_names: list[str] = []
    for entry in rpm_entries:
        _require(
            isinstance(entry, dict) and set(entry) == {"filename", "sha256"},
            f"{artifact.path.name} CUDA RPM provenance entry is invalid: {entry!r}",
        )
        filename = str(entry["filename"])
        rpm_names.append(filename)
        _require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.rpm", filename) is not None,
            f"{artifact.path.name} CUDA RPM filename is invalid: {filename!r}",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])) is not None,
            f"{artifact.path.name} CUDA RPM digest is invalid for {filename}",
        )
    _require(
        len(rpm_names) == len(set(rpm_names)),
        f"{artifact.path.name} CUDA RPM provenance contains duplicate filenames",
    )
    expected_rpms = [
        {"filename": fields[1], "sha256": fields[0]}
        for line in CUDA_RT_RPM_MANIFEST.read_text(encoding="utf-8").splitlines()
        if (fields := line.split())
    ]
    _require(
        rpm_entries == expected_rpms,
        f"{artifact.path.name} CUDA RPM provenance differs from release policy",
    )


def _load_rocm_bundled_policy(root: Path) -> dict[str, object]:
    policy_path = root / ".github" / "scripts" / ROCM_BUNDLED_POLICY.name
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    _require(
        policy.get("schema_version") == 1
        and policy.get("backend") == "rocm"
        and policy.get("distribution_identity") == "gafime-rocm-bundled"
        and policy.get("wheel_policy") == "bundled"
        and policy.get("rocm_version") == "7.2.3"
        and policy.get("manylinux_platform") == "manylinux_2_28_x86_64"
        and policy.get("userspace_bundled") is True
        and policy.get("sbom_required") is True
        and policy.get("mixed_runtime_coexistence") == "unsupported",
        "checked-in ROCm bundled-wheel policy identity is invalid",
    )
    expected_build_inputs = {
        "manylinux_image": ROCM_MANYLINUX_IMAGE,
        "packages": list(ROCM_BUILD_PACKAGES),
        "rocm_gpg_key_sha256": ROCM_GPG_KEY_SHA256,
        "rocm_gpg_key_url": ROCM_GPG_KEY_URL,
        "rocm_repository": ROCM_REPOSITORY,
    }
    _require(
        policy.get("build_inputs") == expected_build_inputs,
        "checked-in ROCm bundled-wheel build inputs are not fully pinned",
    )
    components = policy.get("bundled_components")
    _require(
        isinstance(components, list) and bool(components),
        "checked-in ROCm bundled-wheel policy has no component manifest",
    )
    packages: list[str] = []
    prefixes: list[str] = []
    for component in components:
        _require(
            isinstance(component, dict)
            and isinstance(component.get("package"), str)
            and bool(component["package"])
            and isinstance(component.get("version"), str)
            and bool(component["version"])
            and isinstance(component.get("license"), str)
            and bool(component["license"])
            and isinstance(component.get("max_uncompressed_bytes"), int)
            and component["max_uncompressed_bytes"] > 0,
            f"invalid ROCm bundled component policy: {component!r}",
        )
        library_prefixes = component.get("library_prefixes")
        _require(
            isinstance(library_prefixes, list)
            and bool(library_prefixes)
            and all(isinstance(prefix, str) and prefix for prefix in library_prefixes),
            f"invalid ROCm library prefixes for {component['package']}",
        )
        packages.append(str(component["package"]))
        prefixes.extend(str(prefix) for prefix in library_prefixes)
    _require(
        len(packages) == len(set(packages)),
        "ROCm bundled component policy contains duplicate package names",
    )
    _require(
        len(prefixes) == len(set(prefixes)),
        "ROCm bundled component policy contains duplicate library prefixes",
    )
    limits = policy.get("artifact_limits")
    _require(
        isinstance(limits, dict)
        and all(
            isinstance(limits.get(name), int) and limits[name] > 0
            for name in (
                "wheel_bytes",
                "wheel_uncompressed_bytes",
                "native_payload_uncompressed_bytes",
            )
        ),
        "checked-in ROCm bundled-wheel artifact limits are invalid",
    )
    return policy


def _load_rocm_system_policy(root: Path) -> dict[str, object]:
    policy_path = root / ".github" / "scripts" / ROCM_SYSTEM_POLICY.name
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    _require(
        policy.get("schema_version") == 1
        and policy.get("backend") == "rocm"
        and policy.get("distribution_identity") == "gafime-rocm"
        and policy.get("wheel_policy") == "system"
        and policy.get("rocm_version") == "7.2.3"
        and policy.get("platform_tag") == "linux_x86_64"
        and policy.get("glibc_minimum") == "2.28"
        and policy.get("userspace_bundled") is False
        and policy.get("sbom_required") is False
        and policy.get("mixed_runtime_coexistence") == "host-managed-single-runtime",
        "checked-in ROCm system-wheel policy identity is invalid",
    )
    expected_build_inputs = {
        "image": ROCM_MANYLINUX_IMAGE,
        "packages": list(ROCM_BUILD_PACKAGES),
        "rocm_gpg_key_sha256": ROCM_GPG_KEY_SHA256,
        "rocm_gpg_key_url": ROCM_GPG_KEY_URL,
        "rocm_repository": ROCM_REPOSITORY,
    }
    _require(
        policy.get("build_inputs") == expected_build_inputs,
        "checked-in ROCm system-wheel build inputs are not fully pinned",
    )
    external_runtime = policy.get("external_runtime")
    _require(
        isinstance(external_runtime, dict)
        and external_runtime.get("required_sonames") == ["libamdhip64.so.7"]
        and isinstance(external_runtime.get("requirement"), str)
        and bool(external_runtime["requirement"])
        and "system dynamic loader only"
        in str(external_runtime.get("search_policy", "")),
        "checked-in ROCm system runtime prerequisite is invalid",
    )
    limits = policy.get("artifact_limits")
    _require(
        isinstance(limits, dict)
        and all(
            isinstance(limits.get(name), int) and limits[name] > 0
            for name in (
                "wheel_bytes",
                "wheel_uncompressed_bytes",
                "native_payload_uncompressed_bytes",
            )
        ),
        "checked-in ROCm system-wheel artifact limits are invalid",
    )
    return policy


def _assert_rocm_build_policy(artifact: Artifact, root: Path) -> dict[str, object]:
    expected = (
        _load_rocm_bundled_policy(root)
        if artifact.distribution == "gafime-rocm-bundled"
        else _load_rocm_system_policy(root)
    )
    _require(
        artifact.build_policy == expected,
        f"{artifact.path.name} ROCm build policy differs from the reviewed manifest",
    )
    return expected


def _readelf_dynamic(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        result = subprocess.run(
            ["readelf", "-d", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AssertionError(
            "readelf is required for ROCm wheel closure checks"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"readelf could not inspect {path.name}: {exc.stderr.strip()}"
        ) from exc
    values: dict[str, list[str]] = {
        "NEEDED": [],
        "SONAME": [],
        "RPATH": [],
        "RUNPATH": [],
    }
    for line in result.stdout.splitlines():
        match = re.search(r"\((NEEDED|SONAME|RPATH|RUNPATH)\).*\[([^]]*)\]", line)
        if match:
            values[match.group(1)].append(match.group(2))
    return {name: tuple(items) for name, items in values.items()}


def _assert_rocm_system_wheel(artifact: Artifact, root: Path) -> dict[str, object]:
    policy = _assert_rocm_build_policy(artifact, root)
    _require(
        artifact.kind == "wheel"
        and artifact.distribution == "gafime-rocm"
        and artifact.platforms == {"linux_x86_64"},
        f"{artifact.path.name} is not the reviewed ROCm system-wheel shape",
    )
    limits = policy["artifact_limits"]
    _require(isinstance(limits, dict), "ROCm artifact limits must be a mapping")
    wheel_bytes = artifact.path.stat().st_size
    _require(
        wheel_bytes <= limits["wheel_bytes"],
        f"{artifact.path.name} size {wheel_bytes} exceeds policy limit "
        f"{limits['wheel_bytes']}",
    )

    with zipfile.ZipFile(artifact.path) as archive:
        file_infos = [info for info in archive.infolist() if not info.is_dir()]
        uncompressed_bytes = sum(info.file_size for info in file_infos)
        _require(
            uncompressed_bytes <= limits["wheel_uncompressed_bytes"],
            f"{artifact.path.name} uncompressed size {uncompressed_bytes} exceeds "
            f"policy limit {limits['wheel_uncompressed_bytes']}",
        )
        native_infos = [
            info
            for info in file_infos
            if info.filename == "gafime_rocm/libgafime_rocm.so"
        ]
        _require(
            len(native_infos) == 1,
            f"{artifact.path.name} must contain one Linux ROCm native payload",
        )
        native_info = native_infos[0]
        _require(
            native_info.file_size <= limits["native_payload_uncompressed_bytes"],
            f"{artifact.path.name} native payload size {native_info.file_size} exceeds "
            f"policy limit {limits['native_payload_uncompressed_bytes']}",
        )
        forbidden = sorted(
            info.filename
            for info in file_infos
            if info.filename.startswith("gafime_rocm.libs/")
            or PurePosixPath(info.filename).name.startswith(
                (
                    "libamd",
                    "libhsa",
                    "librocprofiler",
                    "libdrm",
                )
            )
        )
        _require(
            not forbidden,
            f"{artifact.path.name} system policy contains vendored ROCm userspace: "
            f"{forbidden}",
        )
        sboms = [
            info.filename for info in file_infos if ".dist-info/sboms/" in info.filename
        ]
        _require(
            not sboms,
            f"{artifact.path.name} system policy unexpectedly carries a repair SBOM: "
            f"{sboms}",
        )
        with tempfile.TemporaryDirectory(prefix="gafime-rocm-system-") as temporary:
            native_path = Path(temporary) / "libgafime_rocm.so"
            native_path.write_bytes(archive.read(native_info))
            dynamic = _readelf_dynamic(native_path)

    _require(
        not dynamic["RPATH"] and not dynamic["RUNPATH"],
        f"{artifact.path.name} system payload embeds a runtime search path: {dynamic}",
    )
    required_sonames = policy["external_runtime"]["required_sonames"]
    _require(
        sorted(name for name in dynamic["NEEDED"] if name.startswith("libamdhip64"))
        == required_sonames,
        f"{artifact.path.name} ROCm runtime dependency differs from policy: "
        f"{dynamic['NEEDED']}",
    )
    allowed_needed = {
        "ld-linux-x86-64.so.2",
        "libamdhip64.so.7",
        "libc.so.6",
        "libgcc_s.so.1",
        "libm.so.6",
        "libstdc++.so.6",
    }
    unexpected_needed = sorted(set(dynamic["NEEDED"]) - allowed_needed)
    _require(
        not unexpected_needed,
        f"{artifact.path.name} has undeclared direct dependencies: {unexpected_needed}",
    )
    return {
        "schema_version": 1,
        "artifact": artifact.path.name,
        "wheel_bytes": wheel_bytes,
        "wheel_uncompressed_bytes": uncompressed_bytes,
        "native_payload_uncompressed_bytes": native_info.file_size,
        "policy_sha256": hashlib.sha256(
            (root / ".github" / "scripts" / ROCM_SYSTEM_POLICY.name).read_bytes()
        ).hexdigest(),
        "wheel_policy": policy["wheel_policy"],
        "rocm_version": policy["rocm_version"],
        "platform_tag": policy["platform_tag"],
        "userspace_bundled": False,
        "required_sonames": required_sonames,
    }


def _assert_rocm_bundled_wheel(artifact: Artifact, root: Path) -> dict[str, object]:
    policy = _assert_rocm_build_policy(artifact, root)
    expected_platforms = set(
        RELEASE_MANIFEST.excluded_distribution("gafime-rocm-bundled").wheel_platforms
    )
    _require(
        artifact.kind == "wheel"
        and artifact.distribution == "gafime-rocm-bundled"
        and artifact.platforms == expected_platforms,
        f"{artifact.path.name} is not the reviewed ROCm manylinux wheel shape",
    )
    limits = policy["artifact_limits"]
    _require(isinstance(limits, dict), "ROCm artifact limits must be a mapping")
    wheel_bytes = artifact.path.stat().st_size
    _require(
        wheel_bytes <= limits["wheel_bytes"],
        f"{artifact.path.name} size {wheel_bytes} exceeds policy limit "
        f"{limits['wheel_bytes']}",
    )

    with zipfile.ZipFile(artifact.path) as archive:
        file_infos = [info for info in archive.infolist() if not info.is_dir()]
        uncompressed_bytes = sum(info.file_size for info in file_infos)
        _require(
            uncompressed_bytes <= limits["wheel_uncompressed_bytes"],
            f"{artifact.path.name} uncompressed size {uncompressed_bytes} exceeds "
            f"policy limit {limits['wheel_uncompressed_bytes']}",
        )
        native_infos = [
            info
            for info in file_infos
            if info.filename == "gafime_rocm_bundled/libgafime_rocm_bundled.so"
        ]
        _require(
            len(native_infos) == 1,
            f"{artifact.path.name} must contain one Linux ROCm native payload",
        )
        native_info = native_infos[0]
        _require(
            native_info.file_size <= limits["native_payload_uncompressed_bytes"],
            f"{artifact.path.name} native payload size {native_info.file_size} exceeds "
            f"policy limit {limits['native_payload_uncompressed_bytes']}",
        )
        library_infos = [
            info
            for info in file_infos
            if info.filename.startswith("gafime_rocm_bundled.libs/")
        ]
        _require(
            bool(library_infos),
            f"{artifact.path.name} bundled policy has no private userspace libraries",
        )
        library_names = {PurePosixPath(info.filename).name for info in library_infos}
        _require(
            len(library_names) == len(library_infos),
            f"{artifact.path.name} contains duplicate private library basenames",
        )

        component_reports: list[dict[str, object]] = []
        matched_libraries: set[str] = set()
        components = policy["bundled_components"]
        _require(isinstance(components, list), "ROCm component manifest must be a list")
        for component in components:
            _require(
                isinstance(component, dict), "ROCm component entry must be a mapping"
            )
            component_names: set[str] = set()
            for prefix in component["library_prefixes"]:
                matches = {name for name in library_names if name.startswith(prefix)}
                _require(
                    len(matches) == 1,
                    f"{artifact.path.name} expected one library matching {prefix!r}; "
                    f"found {sorted(matches)}",
                )
                component_names.update(matches)
            overlap = matched_libraries & component_names
            _require(
                not overlap,
                f"{artifact.path.name} maps libraries to multiple components: "
                f"{sorted(overlap)}",
            )
            matched_libraries.update(component_names)
            component_bytes = sum(
                info.file_size
                for info in library_infos
                if PurePosixPath(info.filename).name in component_names
            )
            _require(
                component_bytes <= component["max_uncompressed_bytes"],
                f"{artifact.path.name} component {component['package']} size "
                f"{component_bytes} exceeds policy limit "
                f"{component['max_uncompressed_bytes']}",
            )
            component_reports.append(
                {
                    "package": component["package"],
                    "version": component["version"],
                    "license": component["license"],
                    "libraries": sorted(component_names),
                    "uncompressed_bytes": component_bytes,
                }
            )
        _require(
            matched_libraries == library_names,
            f"{artifact.path.name} has unowned private libraries: "
            f"{sorted(library_names - matched_libraries)}",
        )

        sbom_infos = [
            info
            for info in file_infos
            if ".dist-info/sboms/" in info.filename
            and info.filename.endswith("auditwheel.cdx.json")
        ]
        _require(
            len(sbom_infos) == 1,
            f"{artifact.path.name} must contain one auditwheel CycloneDX SBOM",
        )
        sbom = json.loads(archive.read(sbom_infos[0]).decode("utf-8"))
        sbom_components = sbom.get("components")
        _require(
            isinstance(sbom_components, list),
            f"{artifact.path.name} auditwheel SBOM has no component list",
        )
        sbom_identities = {
            (component.get("name"), component.get("version"))
            for component in sbom_components
            if isinstance(component, dict)
        }
        metadata_component = sbom.get("metadata", {}).get("component", {})
        root_ref = (
            metadata_component.get("bom-ref")
            if isinstance(metadata_component, dict)
            else None
        )
        root_dependencies = [
            dependency
            for dependency in sbom.get("dependencies", [])
            if isinstance(dependency, dict) and dependency.get("ref") == root_ref
        ]
        _require(
            len(root_dependencies) == 1
            and isinstance(root_dependencies[0].get("dependsOn"), list),
            f"{artifact.path.name} auditwheel SBOM has no root dependency closure",
        )
        root_dependency_refs = {
            str(reference).split("#", 1)[0]
            for reference in root_dependencies[0]["dependsOn"]
        }
        for component in components:
            identity = (component["package"], component["version"])
            _require(
                identity in sbom_identities,
                f"{artifact.path.name} SBOM does not identify {identity!r}",
            )
            rpm_purl = (
                f"pkg:rpm/almalinux/{component['package']}@{component['version']}"
            )
            _require(
                rpm_purl in root_dependency_refs,
                f"{artifact.path.name} SBOM root does not depend on {rpm_purl}",
            )

        with tempfile.TemporaryDirectory(prefix="gafime-rocm-wheel-") as temp_dir:
            temp_root = Path(temp_dir)
            extracted: dict[str, Path] = {}
            for info in (native_info, *library_infos):
                basename = PurePosixPath(info.filename).name
                output = temp_root / basename
                output.write_bytes(archive.read(info))
                extracted[basename] = output

            allowed_external = {
                "ld-linux-x86-64.so.2",
                "libc.so.6",
                "libdl.so.2",
                "libgcc_s.so.1",
                "libm.so.6",
                "libpthread.so.0",
                "librt.so.1",
                "libstdc++.so.6",
                "libz.so.1",
            }
            external_needed: set[str] = set()
            for basename, path in extracted.items():
                dynamic = _readelf_dynamic(path)
                _require(
                    not dynamic["RUNPATH"],
                    f"{artifact.path.name} {basename} must not carry RUNPATH",
                )
                expected_rpath = (
                    ("$ORIGIN/../gafime_rocm_bundled.libs",)
                    if basename == "libgafime_rocm_bundled.so"
                    else ()
                )
                if basename != "libgafime_rocm_bundled.so" and dynamic["RPATH"]:
                    expected_rpath = ("$ORIGIN",)
                _require(
                    dynamic["RPATH"] == expected_rpath,
                    f"{artifact.path.name} {basename} RPATH {dynamic['RPATH']!r} "
                    f"!= {expected_rpath!r}",
                )
                if basename != "libgafime_rocm_bundled.so":
                    _require(
                        dynamic["SONAME"] == (basename,),
                        f"{artifact.path.name} {basename} has unexpected SONAME "
                        f"{dynamic['SONAME']!r}",
                    )
                for needed in dynamic["NEEDED"]:
                    if needed in library_names:
                        continue
                    _require(
                        needed in allowed_external,
                        f"{artifact.path.name} {basename} has unresolved or unapproved "
                        f"dependency {needed!r}",
                    )
                    external_needed.add(needed)

    return {
        "schema_version": 1,
        "artifact": artifact.path.name,
        "wheel_bytes": wheel_bytes,
        "wheel_uncompressed_bytes": uncompressed_bytes,
        "native_payload_uncompressed_bytes": native_info.file_size,
        "policy_sha256": hashlib.sha256(
            (root / ".github" / "scripts" / ROCM_BUNDLED_POLICY.name).read_bytes()
        ).hexdigest(),
        "wheel_policy": policy["wheel_policy"],
        "rocm_version": policy["rocm_version"],
        "manylinux_platform": policy["manylinux_platform"],
        "mixed_runtime_coexistence": policy["mixed_runtime_coexistence"],
        "sbom": sbom_infos[0].filename,
        "components": component_reports,
        "external_needed": sorted(external_needed),
    }


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
    leaked_payloads = sorted(
        member
        for member in artifact.members
        if member.startswith(payload_packages)
        or PurePosixPath(member).name in payload_libraries
    )
    _require(
        not leaked_payloads,
        f"core wheel contains external payload files: {leaked_payloads}",
    )
    core = RELEASE_MANIFEST.distribution("gafime")
    matching_specs = [
        wheel for wheel in core.wheels if wheel.platform in artifact.platforms
    ]
    _require(
        len(matching_specs) <= 1,
        f"{artifact.path.name} matches multiple core platform policies",
    )
    embedded = set(matching_specs[0].embedded_backends) if matching_specs else set()
    metal_members = {
        "gafime/_metal/libgafime_metal_v1.dylib",
        "gafime/_metal/gafime_metal_v1.metallib",
    }
    packaged_metal = {
        member for member in artifact.members if member.startswith("gafime/_metal/")
    }
    if "metal" in embedded:
        _require(
            packaged_metal == metal_members,
            f"{artifact.path.name} bundled Metal files {sorted(packaged_metal)} != "
            f"{sorted(metal_members)}",
        )
    else:
        _require(
            not packaged_metal,
            f"{artifact.path.name} unexpectedly embeds Metal: {sorted(packaged_metal)}",
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
    expected_native_count = 1
    _require(
        len(native_members) == expected_native_count,
        f"{artifact.path.name} must contain exactly {expected_native_count} native "
        "payload artifacts; "
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
        elif backend == "rocm":
            _assert_rocm_build_policy(artifacts[0], root)
    elif scope == "rocm-bundled-sdist":
        expected_count = 1
        _assert_payload_sdist(
            _assert_one(artifacts, "bundled ROCm sdist"),
            "gafime-rocm-bundled",
        )
        _assert_rocm_build_policy(artifacts[0], root)
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
            elif backend == "rocm":
                _assert_rocm_system_wheel(artifact, root)
    elif scope == "rocm-bundled-wheel":
        expected_count = len(artifacts)
        _require(expected_count > 0, "no bundled ROCm wheels found")
        for artifact in artifacts:
            _assert_payload_wheel(artifact, "gafime-rocm-bundled")
            _assert_rocm_bundled_wheel(artifact, root)
    elif scope == "cuda-rt-wheel":
        expected_count = len(artifacts)
        _require(expected_count > 0, "no CUDA RT wheels found")
        for artifact in artifacts:
            _assert_payload_wheel(artifact, "gafime-cuda-rt")
            _assert_cuda_build_policy(artifact, "on")
    elif scope == "sdists":
        expected_count = len(RELEASE_MANIFEST.standard_distributions)
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
            elif backend == "rocm":
                _assert_rocm_build_policy(payload_sdist, root)
    elif scope == "core-release":
        distribution = RELEASE_MANIFEST.distribution("gafime")
        expected_count = distribution.artifact_count
        _assert_wheel_platforms(
            artifacts,
            distribution.name,
            {wheel.platform for wheel in distribution.wheels},
        )
        _assert_core_sdist(
            _assert_one(_select(artifacts, "gafime", "sdist"), "core sdist"), root
        )
    elif scope == "cuda-release":
        distribution = RELEASE_MANIFEST.distribution("gafime-cuda")
        expected_count = distribution.artifact_count
        _assert_wheel_platforms(
            artifacts,
            distribution.name,
            {wheel.platform for wheel in distribution.wheels},
        )
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-cuda", "sdist"), "CUDA sdist"),
            "gafime-cuda",
        )
        for artifact in artifacts:
            _assert_cuda_build_policy(artifact, "off")
    elif scope == "cuda-rt-release":
        distribution = RELEASE_MANIFEST.excluded_distribution("gafime-cuda-rt")
        expected_count = len(distribution.wheel_platforms) + int(distribution.sdist)
        _assert_wheel_platforms(
            artifacts,
            distribution.name,
            set(distribution.wheel_platforms),
        )
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-cuda-rt", "sdist"), "CUDA RT sdist"),
            "gafime-cuda-rt",
        )
        for artifact in artifacts:
            _assert_cuda_build_policy(artifact, "on")
    elif scope == "rocm-release":
        distribution = RELEASE_MANIFEST.distribution("gafime-rocm")
        expected_count = distribution.artifact_count
        _assert_wheel_platforms(
            artifacts,
            distribution.name,
            {wheel.platform for wheel in distribution.wheels},
        )
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-rocm", "sdist"), "ROCm sdist"),
            "gafime-rocm",
        )
        for artifact in artifacts:
            if artifact.kind == "wheel":
                _assert_rocm_system_wheel(artifact, root)
            else:
                _assert_rocm_build_policy(artifact, root)
    elif scope == "rocm-bundled-release":
        distribution = RELEASE_MANIFEST.excluded_distribution("gafime-rocm-bundled")
        expected_count = len(distribution.wheel_platforms) + int(distribution.sdist)
        _assert_wheel_platforms(
            artifacts,
            distribution.name,
            set(distribution.wheel_platforms),
        )
        _assert_payload_sdist(
            _assert_one(
                _select(artifacts, "gafime-rocm-bundled", "sdist"),
                "bundled ROCm sdist",
            ),
            "gafime-rocm-bundled",
        )
        for artifact in artifacts:
            if artifact.kind == "wheel":
                _assert_rocm_bundled_wheel(artifact, root)
            else:
                _assert_rocm_build_policy(artifact, root)
    elif scope == "full-release":
        expected_count = RELEASE_MANIFEST.standard_artifact_count
        scope_by_distribution = {
            "gafime": "core-release",
            "gafime-cuda": "cuda-release",
            "gafime-rocm": "rocm-release",
        }
        for distribution in RELEASE_MANIFEST.standard_distributions:
            _assert_scope(
                _select_distribution(artifacts, distribution.name),
                scope_by_distribution[distribution.name],
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


def _workflow_job_block(workflow: str, job_name: str) -> str:
    marker = f"\n  {job_name}:\n"
    _require(marker in workflow, f"release workflow is missing job {job_name}")
    tail = workflow.split(marker, 1)[1]
    next_job = re.search(r"\n  [A-Za-z0-9_]+:\n", tail)
    return tail if next_job is None else tail[: next_job.start()]


def _assert_release_manifest_pyproject(
    project: dict[str, object], version: str
) -> None:
    project_metadata = project["project"]
    optional = project_metadata["optional-dependencies"]
    expected_python = set(RELEASE_MANIFEST.supported_python)
    actual_python = {
        classifier.removeprefix("Programming Language :: Python :: ")
        for classifier in project_metadata["classifiers"]
        if classifier.startswith("Programming Language :: Python :: ")
    }
    _require(
        actual_python == expected_python,
        f"release manifest supported Python versions {sorted(expected_python)} != "
        f"pyproject classifiers {sorted(actual_python)}",
    )
    _require(
        project_metadata["requires-python"]
        == f">={RELEASE_MANIFEST.supported_python[0]}",
        "pyproject requires-python must start at the release manifest minimum",
    )
    expected_backend_extras = {
        distribution.extra_name
        for distribution in RELEASE_MANIFEST.standard_distributions
        if distribution.extra_name is not None
    }
    actual_backend_extras = set(optional) & {"cuda", "rocm", "metal"}
    _require(
        actual_backend_extras == expected_backend_extras,
        "release manifest backend extras "
        f"{sorted(expected_backend_extras)} != pyproject backend extras "
        f"{sorted(actual_backend_extras)}",
    )
    for distribution in RELEASE_MANIFEST.standard_distributions:
        if distribution.extra_name is None:
            continue
        expected = [f"{distribution.name}=={version}; {distribution.extra_marker}"]
        actual = optional.get(distribution.extra_name)
        _require(
            actual == expected,
            f"release manifest {distribution.name} extra {distribution.extra_name!r} "
            f"expects {expected!r}, found {actual!r}",
        )


def _assert_release_manifest_workflow(workflow: str) -> None:
    selector = f'CIBW_BUILD: "{RELEASE_MANIFEST.build_selector}"'
    checked_build_jobs: set[str] = set()
    for distribution in RELEASE_MANIFEST.standard_distributions:
        sdist_job = _workflow_job_block(workflow, distribution.sdist_build_job)
        _require(
            f"name: {distribution.sdist_artifact}" in sdist_job,
            f"release manifest {distribution.name} sdist artifact "
            f"{distribution.sdist_artifact!r} is absent from job "
            f"{distribution.sdist_build_job}",
        )
        for wheel in distribution.wheels:
            build_job = _workflow_job_block(workflow, wheel.build_job)
            _require(
                wheel.artifact in build_job,
                f"release manifest {distribution.name}/{wheel.platform} build artifact "
                f"{wheel.artifact!r} is absent from job {wheel.build_job}",
            )
            if wheel.build_job not in checked_build_jobs:
                _require(
                    selector in build_job,
                    f"release manifest job {wheel.build_job} must build the ABI3 wheel "
                    f"once with {RELEASE_MANIFEST.build_selector!r}",
                )
                selectors = set(re.findall(r'CIBW_BUILD:\s*"([^"]+)"', build_job))
                _require(
                    selectors == {RELEASE_MANIFEST.build_selector},
                    f"release manifest job {wheel.build_job} has CIBW_BUILD selectors "
                    f"{sorted(selectors)}, expected only "
                    f"{RELEASE_MANIFEST.build_selector!r}",
                )
                checked_build_jobs.add(wheel.build_job)
            validation_job = _workflow_job_block(workflow, wheel.validation_job)
            for value, field in (
                (wheel.validation_label, "validation label"),
                (wheel.artifact, "artifact"),
                (wheel.filename_pattern, "wheel pattern"),
            ):
                _require(
                    value in validation_job,
                    f"release manifest {distribution.name}/{wheel.platform} {field} "
                    f"{value!r} is absent from job {wheel.validation_job}",
                )
            for version in wheel.validation_python:
                compact_tag = (
                    f"cp{version.replace('.', '')}-cp{version.replace('.', '')}"
                )
                _require(
                    version in validation_job or compact_tag in validation_job,
                    f"release manifest validation job {wheel.validation_job} does not "
                    f"install {distribution.name}/{wheel.platform} on Python {version}",
                )
            omitted_versions = set(RELEASE_MANIFEST.supported_python) - set(
                wheel.validation_python
            )
            for version in omitted_versions:
                _require(
                    f'"{version}"' not in validation_job
                    and f"'{version}'" not in validation_job,
                    f"release manifest validation job {wheel.validation_job} includes "
                    f"undeclared Python {version} for "
                    f"{distribution.name}/{wheel.platform}",
                )
            if wheel.embedded_backends:
                _require(
                    wheel.embedded_backends == ("metal",),
                    f"unsupported embedded backend set for "
                    f"{distribution.name}/{wheel.platform}: "
                    f"{wheel.embedded_backends}",
                )
                _require(
                    "stage_metal_payload.py" in build_job,
                    f"{distribution.name}/{wheel.platform} must stage bundled Metal",
                )
                _require(
                    "--backend metal" in validation_job
                    and "--execute-metal" in validation_job,
                    f"{distribution.name}/{wheel.platform} must execute bundled Metal "
                    "on every declared Python version",
                )

    preflight = _workflow_job_block(workflow, "release_preflight")
    pattern = RELEASE_MANIFEST.bundle_download_pattern
    _require(
        f"pattern: {pattern}" in preflight,
        f"release manifest bundle download pattern {pattern!r} is absent from "
        "release_preflight",
    )
    standard_artifacts = {
        distribution.sdist_artifact
        for distribution in RELEASE_MANIFEST.standard_distributions
    } | {
        wheel.artifact
        for distribution in RELEASE_MANIFEST.standard_distributions
        for wheel in distribution.wheels
    }
    for artifact in sorted(standard_artifacts):
        _require(
            fnmatchcase(artifact, pattern),
            f"release manifest standard artifact {artifact!r} is not selected by "
            f"release_preflight pattern {pattern!r}",
        )
    for excluded in RELEASE_MANIFEST.excluded_distributions:
        if excluded.artifact is None:
            continue
        _require(
            not fnmatchcase(excluded.artifact, pattern),
            f"release manifest excluded artifact {excluded.artifact!r} is selected by "
            f"standard bundle pattern {pattern!r}",
        )
    _require(
        f"name: {RELEASE_MANIFEST.bundle_artifact}" in preflight,
        f"release manifest frozen bundle name {RELEASE_MANIFEST.bundle_artifact!r} "
        "is absent from release_preflight",
    )


def _assert_release_manifest_documentation(root: Path) -> None:
    matrix_path = root / "docs" / "releases" / "release-artifact-matrix.md"
    _require(matrix_path.is_file(), "generated release artifact matrix is missing")
    expected = render_release_matrix(RELEASE_MANIFEST)
    actual = matrix_path.read_text(encoding="utf-8")
    _require(
        actual == expected,
        "docs/releases/release-artifact-matrix.md differs from "
        ".github/release-artifacts.json; regenerate it with "
        "tests/release_measure/release_manifest.py --write",
    )
    runbook = (root / "docs" / "releases" / "release-operations.md").read_text(
        encoding="utf-8"
    )
    _require(
        "release-artifact-matrix.md" in runbook,
        "release operations must link the manifest-derived artifact matrix",
    )


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
    _require(
        "recursive-include tests/gpu *" in manifest_lines,
        "MANIFEST.in must include the complete GPU test fixture directory",
    )
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    _require(
        '{ path = "src/**/*", format = "sdist" }' in pyproject,
        "Maturin sdist policy must include all native src files",
    )
    pyproject_data = tomllib.loads(pyproject)
    _assert_release_manifest_pyproject(pyproject_data, _project_version(root))
    sdist_patterns = {
        str(entry["path"])
        for entry in pyproject_data["tool"]["maturin"].get("include", [])
        if entry.get("format") == "sdist"
    }
    available_gpu_tests = {
        path.relative_to(root).as_posix()
        for path in (root / "tests" / "gpu").iterdir()
        if path.is_file()
    }
    selected_gpu_tests = {
        source
        for source in available_gpu_tests
        if any(PurePosixPath(source).match(pattern) for pattern in sdist_patterns)
    }
    _require(
        available_gpu_tests == CORE_GPU_TEST_SOURCES,
        f"source-tree GPU test fixtures {sorted(available_gpu_tests)} != "
        f"{sorted(CORE_GPU_TEST_SOURCES)}",
    )
    _require(
        selected_gpu_tests == CORE_GPU_TEST_SOURCES,
        f"Maturin sdist GPU test sources {sorted(selected_gpu_tests)} != "
        f"{sorted(CORE_GPU_TEST_SOURCES)}",
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
        'CUDA_TUNING_POLICY = "runtime-device-class"',
        "RUNTIME_ARCHITECTURE_DISPATCH = True",
        "PER_ARCHITECTURE_TUNING = False",
        'package_name = "gafime_cuda_rt"',
        'dist_name = "gafime-cuda-rt"',
        'package_name = "gafime_rocm_bundled"',
        'dist_name = "gafime-rocm-bundled"',
        '"cuda_toolkit_rpms": rpm_entries',
        '"wheel_builder_image": builder_image',
        'ROCM_WHEEL_POLICY = "{rocm_wheel_policy}"',
        "GAFIME_ROCM_WHEEL_POLICY",
        "--rocm-wheel-policy",
        "the reviewed policies are 'system' and 'bundled'",
    ):
        _require(token in stage_script, f"GPU payload staging is missing {token}")
    _require(
        "GAFIME_CUDA_TUNING_SM" not in stage_script,
        "GPU payload staging must not inject one package-wide CUDA tuning SM",
    )
    _require(
        'choices=("off", "on")' in stage_script,
        "GPU payload staging must expose separate immutable RT-off/RT-on selection",
    )
    for label, rocm_policy in (
        ("system", _load_rocm_system_policy(root)),
        ("bundled", _load_rocm_bundled_policy(root)),
    ):
        _require(
            len(rocm_policy["gfx_targets"]) == 13,
            f"ROCm {label}-wheel policy must declare all 13 release code-object targets",
        )
    metal_stage_script = (
        root / ".github" / "scripts" / "stage_metal_payload.py"
    ).read_text(encoding="utf-8")
    for token in (
        'METAL_LIBRARY = "libgafime_metal_v1.dylib"',
        'METALLIB = "gafime_metal_v1.metallib"',
        'default=REPO_ROOT / "python" / "gafime" / "_metal"',
        '"-DCMAKE_OSX_ARCHITECTURES=arm64"',
    ):
        _require(
            token in metal_stage_script, f"bundled Metal staging is missing {token}"
        )
    _require(
        not (root / ".github" / "scripts" / "stage_metal_distribution.py").exists(),
        "a separate gafime-metal staging path must not exist",
    )
    rpm_manifest_path = root / ".github" / "scripts" / "cuda_13_3_rpms.sha256"
    rpm_manifest_entries = [
        line.split()
        for line in rpm_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        len(rpm_manifest_entries) == 11,
        "CUDA 13.3 wheel-builder manifest must pin all 11 toolkit RPMs",
    )
    _require(
        all(
            len(fields) == 2
            and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.rpm", fields[1]) is not None
            for fields in rpm_manifest_entries
        ),
        "CUDA 13.3 wheel-builder manifest has an invalid entry",
    )
    rpm_filenames = [fields[1] for fields in rpm_manifest_entries]
    _require(
        len(rpm_filenames) == len(set(rpm_filenames)),
        "CUDA 13.3 wheel-builder manifest has duplicate packages",
    )
    _require(
        any(name.startswith("cuda-nvcc-13-3-") for name in rpm_filenames)
        and any(name.startswith("cuda-cudart-devel-13-3-") for name in rpm_filenames),
        "CUDA 13.3 wheel-builder manifest must pin nvcc and cudart-devel",
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
    with tempfile.TemporaryDirectory(prefix="gafime-missing-rocm-policy-") as temp_dir:
        rejected = subprocess.run(
            [
                sys.executable,
                str(stage_path),
                "rocm",
                str(Path(temp_dir) / "payload"),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    _require(
        rejected.returncode == 2
        and "requires explicit --rocm-wheel-policy system|bundled" in rejected.stderr,
        "ROCm payload staging must fail closed without an explicit wheel policy",
    )

    build_workflow = (root / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )
    _assert_release_manifest_workflow(build_workflow)
    _assert_release_manifest_documentation(root)
    for token in (
        "windows-2025-vs2026",
        "macos-26",
        "RUST_VERSION: '1.89.0'",
        "release_preflight:",
        "name: release-bundle",
        "build_cuda_rt_linux_payload:",
        "GAFIME_OPTIX_SDK_ARCHIVE_URL",
        "CUDA_RT_WHEEL_BUILDER_IMAGE",
        "cuda_13_3_rpms.sha256",
        "/project/payload-src/gafime-cuda-rt/.cuda-rpms/*.rpm",
        "/project/payload-src/gafime-cuda-rt/.optix-sdk/include",
        "/opt/rh/gcc-toolset-14/root/usr/bin/g++",
        "rpm -Uvh --nodeps",
        "PUBLISH_REQUESTED: ${{ (github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')) ||",
        'git merge-base --is-ancestor "$GITHUB_SHA" origin/main',
        "if: (github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')) ||",
        "--scope cuda-rt-release",
        f"gafime_cuda_rt-*-{RELEASE_MANIFEST.python_tag}-"
        f"{RELEASE_MANIFEST.abi_tag}-*.whl",
        "name: cuda-rt-linux-artifacts",
        "python .github/scripts/stage_metal_payload.py",
        "gafime/_metal/libgafime_metal_v1",
        "gafime/_metal/gafime_metal_v1",
        "--rocm-wheel-policy system",
        "GAFIME_ROCM_WHEEL_POLICY=system",
        'CIBW_REPAIR_WHEEL_COMMAND_LINUX: "cp {wheel} {dest_dir}/"',
        RELEASE_MANIFEST.distribution("gafime-rocm").wheels[0].filename_pattern,
        "rocm-wheel-policy-report.json",
        "install_local_core_wheel.py",
        ROCM_MANYLINUX_IMAGE,
        ROCM_GPG_KEY_SHA256,
        *ROCM_BUILD_PACKAGES,
    ):
        _require(token in build_workflow, f"release workflow is missing {token}")
    _require(
        "publish_pypi_cuda_rt:" not in build_workflow,
        "optional gafime-cuda-rt artifacts must not have a PyPI publishing job",
    )
    for forbidden in (
        "publish_pypi_metal",
        "build_metal_payload_wheels",
        "build_metal_payload_sdist",
        "gafime_metal-*.whl",
        "gafime_metal-*.tar.gz",
    ):
        _require(
            forbidden not in build_workflow,
            f"separate Metal release path remains in workflow: {forbidden}",
        )
    _require(
        "skip-existing: true" not in build_workflow,
        "release publishing must not blindly skip an existing PyPI filename",
    )

    rocm_build_job = _workflow_job_block(
        build_workflow, "build_rocm_linux_payload_wheels"
    )
    _require(
        'CIBW_BUILD: "cp310-*"' in rocm_build_job and "cp311-*" not in rocm_build_job,
        "the raw Linux ROCm ABI3 wheel must be built once to avoid duplicate filenames",
    )
    rocm_validation_job = _workflow_job_block(
        build_workflow, "validate_rocm_payload_wheels"
    )
    for version in RELEASE_MANIFEST.supported_python:
        python = version.replace(".", "")
        python_tag = f"cp{python}-cp{python}"
        _require(
            python_tag in rocm_validation_job,
            f"ROCm installed validation is missing {python_tag}",
        )

    cuda_publish_job = _workflow_job_block(build_workflow, "publish_pypi_cuda")
    rocm_publish_job = _workflow_job_block(build_workflow, "publish_pypi_rocm")
    core_publish_job = _workflow_job_block(build_workflow, "publish_pypi_core")
    github_release_job = _workflow_job_block(build_workflow, "release")
    release_preflight_job = _workflow_job_block(build_workflow, "release_preflight")
    for name, job in (
        ("CUDA", cuda_publish_job),
        ("ROCm", rocm_publish_job),
    ):
        _require(
            "needs: release_preflight" in job,
            f"{name} payload publishing must depend directly on release preflight",
        )
    for group, job in (
        ("gafime-pypi-cuda-publication", cuda_publish_job),
        ("gafime-pypi-rocm-publication", rocm_publish_job),
        ("gafime-pypi-core-publication", core_publish_job),
        ("gafime-github-release-publication", github_release_job),
    ):
        _require(
            f"group: {group}" in job and "cancel-in-progress: false" in job,
            f"publication job must serialize through {group}",
        )
        _require(
            "timeout-minutes: 30" in job,
            f"publication job {group} must have a bounded timeout",
        )
    recovery_expression = (
        "skip-existing: ${{ github.event_name == 'workflow_dispatch' && "
        "inputs.allow_matching_existing_pypi_files == true }}"
    )
    for name, job in (
        ("CUDA", cuda_publish_job),
        ("ROCm", rocm_publish_job),
        ("Core", core_publish_job),
    ):
        _require(
            recovery_expression in job,
            f"{name} publishing may skip files only in explicit recovery mode",
        )
        _require(
            "check_pypi_artifact_collisions.py" in job
            and "--allow-matching-existing" in job,
            f"{name} recovery must verify matching PyPI hashes before upload",
        )
    for dependency in (
        "publish_pypi_cuda",
        "publish_pypi_rocm",
    ):
        _require(
            f"- {dependency}" in core_publish_job,
            f"Core publishing must wait for {dependency}",
        )
        _require(
            f"needs.{dependency}.result == 'success'" in core_publish_job,
            f"Core publishing must require successful {dependency}",
        )
    for dependency in (
        "publish_pypi_cuda",
        "publish_pypi_rocm",
        "publish_pypi_core",
    ):
        _require(
            f"- {dependency}" in github_release_job,
            f"GitHub Release publishing must wait for {dependency}",
        )
        _require(
            f"needs.{dependency}.result == 'success'" in github_release_job,
            f"GitHub Release publishing must require successful {dependency}",
        )
    _require(
        "always()" in core_publish_job and "always()" in github_release_job,
        "ordered publication jobs must inspect failed or skipped dependencies explicitly",
    )
    _require(
        "prerelease: ${{" in github_release_job
        and "contains(github.ref_name, 'rc')" in github_release_job,
        "GitHub Release publishing must classify prerelease version tags",
    )
    _require(
        "inputs.publish_github_release == true" in github_release_job
        and github_release_job.count("inputs.publish_pypi_") >= 3
        and "startsWith(github.ref, 'refs/tags/v')" in github_release_job,
        "manual GitHub Release recovery must require the version tag and every PyPI lane",
    )
    _require(
        "find dist -type f -name 'gafime_rocm-*.tar.gz'" in rocm_publish_job
        and "--scope rocm-sdist" in rocm_publish_job
        and "gafime_rocm-*.whl" not in rocm_publish_job,
        "PyPI ROCm publication must ship only the system-policy sdist; "
        "the truthful linux wheel belongs to the GitHub Release",
    )
    _require(
        "inputs.check_pypi_collisions == true" in release_preflight_job
        and "check_pypi_artifact_collisions.py" in release_preflight_job
        and "--artifacts dist" in release_preflight_job,
        "release preflight must support a live full-bundle PyPI collision dry run",
    )

    collision_script = (
        root / ".github" / "scripts" / "check_pypi_artifact_collisions.py"
    )
    collision_self_test = subprocess.run(
        [sys.executable, str(collision_script), "--self-test"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        collision_self_test.returncode == 0
        and "PYPI COLLISION SELF-TEST: PASS" in collision_self_test.stdout,
        "PyPI collision preflight self-test failed: "
        f"stdout={collision_self_test.stdout!r} stderr={collision_self_test.stderr!r}",
    )

    rt_job = build_workflow.split("\n  build_cuda_rt_linux_payload:\n", 1)[1].split(
        "\n  build_rocm_linux_payload_wheels:\n", 1
    )[0]
    _require(
        "dnf install" not in rt_job,
        "CUDA RT wheel construction must not install unpinned live-repository RPMs",
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
            "rocm-bundled-sdist",
            "cuda-wheel",
            "cuda-rt-wheel",
            "rocm-wheel",
            "rocm-bundled-wheel",
            "sdists",
            "core-release",
            "cuda-release",
            "cuda-rt-release",
            "rocm-release",
            "rocm-bundled-release",
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
    parser.add_argument("--write-rocm-report", type=Path)
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
    if args.write_rocm_report is not None:
        rocm_wheels = [
            *_select(artifacts, "gafime-rocm", "wheel"),
            *_select(artifacts, "gafime-rocm-bundled", "wheel"),
        ]
        rocm_wheel = _assert_one(rocm_wheels, "ROCm wheel for policy report")
        report = (
            _assert_rocm_bundled_wheel(rocm_wheel, root)
            if rocm_wheel.distribution == "gafime-rocm-bundled"
            else _assert_rocm_system_wheel(rocm_wheel, root)
        )
        report_path = args.write_rocm_report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"RELEASE ARTIFACT COMPOSITION: PASS scope={args.scope} "
        f"version={version} artifacts={len(artifacts)}"
    )


if __name__ == "__main__":
    main()
