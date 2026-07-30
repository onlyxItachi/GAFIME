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
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Iterable
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
from release_manifest import (  # noqa: E402
    load_release_manifest,
    python_tag,
    render_release_matrix,
)
from release_version import ReleaseVersion, validate_project_versions  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


RELEASE_MANIFEST = load_release_manifest(ROOT)
LICENSE_EXPRESSION = "Apache-2.0"
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
        "off" if distribution.backend == "cuda" else None,
    )
    for distribution in RELEASE_MANIFEST.distributions
    if distribution.backend is not None
}
DISTRIBUTIONS = RELEASE_MANIFEST.all_distribution_names
CUDA_SDIST_SOURCES = {
    "src/common/gafime_gpu_abi.hpp",
    "src/common/gpu_abi_impl.hpp",
    "src/cuda/cuda_api.hpp",
    "src/cuda/kernels.cu",
    "src/cuda/kernels.cuh",
    "src/cuda/launcher.cu",
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
    python_tags: frozenset[str] = frozenset()
    build_policy: dict[str, object] | None = None
    build_provenance: dict[str, object] | None = None


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _project_version(root: Path) -> str:
    return validate_project_versions(root).pep440


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
        filename_prefix, wheel_python_tag, abi_tag, platform_tag = path.name[:-4].rsplit(
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
    python_tags = frozenset(wheel_python_tag.split("."))
    abi_tags = frozenset(abi_tag.split("."))
    platform_tags = frozenset(platform_tag.split("."))
    expected_python_tags = {
        python_tag(version) for version in RELEASE_MANIFEST.supported_python
    }
    _require(
        len(python_tags) == 1
        and python_tags == abi_tags
        and python_tags <= expected_python_tags,
        f"{path.name} must use one matching per-CPython interpreter/ABI tag from "
        f"{sorted(expected_python_tags)}",
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
        python_tags=python_tags,
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
    requirements = {
        requirement.split(";", 1)[0].strip()
        for requirement in artifact.metadata.get_all("Requires-Dist", [])
    }
    if artifact.distribution == "gafime":
        payload_dependencies = sorted(
            requirement
            for requirement in requirements
            if re.match(r"(?i)^gafime[-_.](cuda|rocm|metal)\b", requirement)
        )
        _require(
            not payload_dependencies,
            f"{artifact.path.name} Core metadata depends on payload distributions: "
            f"{payload_dependencies}",
        )
        return
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
        for path in (root / "src" / "common").rglob("*")
        if path.is_file()
    }
    expected = (
        expected_native
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
    backend_sources = sorted(
        member
        for member in artifact.members
        if member.startswith(("src/cuda/", "src/rocm/", "src/metal/", "tests/gpu/"))
    )
    _require(
        not backend_sources,
        f"core sdist must not carry backend or experimental test sources: "
        f"{backend_sources}",
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
    experimental_rt = sorted(
        member
        for member in artifact.members
        if PurePosixPath(member).name
        in {"rt_kernels.cu", "rt_kernels.cuh", "rt_launcher.cu", "rt_launcher.cuh"}
        or ".optix-sdk/" in member
    )
    _require(
        not experimental_rt,
        f"{expected_distribution} sdist contains experimental RT sources: "
        f"{experimental_rt}",
    )


def _assert_cuda_build_policy(artifact: Artifact, expected_rt_mode: str) -> None:
    _require(expected_rt_mode == "off", "distributed CUDA artifacts cannot select RT")
    expected = {
        "cuda_architectures": ["75", "80", "86", "89", "90", "100", "120"],
        "cuda_tuning_policy": "runtime-device-class",
        "cuda_tuning_sm": None,
        "cuda_runtime": "system",
        "cuda_runtime_libraries": {
            "linux": "libcudart.so.13",
            "windows": "nvcudart_hybrid64.dll",
        },
        "optix_rt": "off",
        "rt_sources_included": False,
        "per_architecture_tuning": False,
        "runtime_architecture_dispatch": True,
    }
    _require(
        artifact.build_policy == expected,
        f"{artifact.path.name} CUDA build policy {artifact.build_policy!r} != {expected!r}",
    )
    forbidden = sorted(
        member for member in artifact.members
        if PurePosixPath(member).name in {
            "rt_kernels.cu", "rt_kernels.cuh", "rt_launcher.cu", "rt_launcher.cuh",
        } or ".optix-sdk/" in member or PurePosixPath(member).name == "optix.h"
    )
    _require(not forbidden, f"CUDA distribution contains RT or OptiX sources: {forbidden}")
    _require(
        artifact.build_provenance is None,
        f"{artifact.path.name} standard CUDA artifact contains RT provenance",
    )
    bundled_runtime = sorted(
        member
        for member in artifact.members
        if PurePosixPath(member).name.lower().startswith(
            ("libcudart", "cudart64_", "nvcudart")
        )
    )
    _require(
        not bundled_runtime,
        f"{artifact.path.name} contains bundled CUDA runtime files: "
        f"{bundled_runtime}",
    )


def _assert_cuda_system_wheel(artifact: Artifact) -> None:
    _assert_cuda_build_policy(artifact, "off")
    linux_platforms = {
        platform
        for platform in artifact.platforms
        if re.fullmatch(r"manylinux_[0-9]+_[0-9]+_x86_64", platform)
    }
    valid_linux = (
        "manylinux_2_28_x86_64" in linux_platforms
        and linux_platforms == set(artifact.platforms)
    )
    valid_windows = artifact.platforms == {"win_amd64"}
    _require(
        artifact.kind == "wheel"
        and artifact.distribution == "gafime-cuda"
        and (valid_linux or valid_windows),
        f"{artifact.path.name} is not a pinned CUDA system-runtime wheel",
    )
    native_name = (
        "gafime_cuda/libgafime_cuda.so"
        if valid_linux
        else "gafime_cuda/gafime_cuda.dll"
    )
    with zipfile.ZipFile(artifact.path) as archive:
        _require(
            native_name in artifact.members,
            f"{artifact.path.name} is missing {native_name}",
        )
        with tempfile.TemporaryDirectory(prefix="gafime-cuda-system-") as temporary:
            native_path = Path(temporary) / PurePosixPath(native_name).name
            native_path.write_bytes(archive.read(native_name))
            if native_path.suffix == ".so":
                dynamic = _readelf_dynamic(native_path)
                _require(
                    "libcudart.so.13" in dynamic["NEEDED"],
                    f"{artifact.path.name} must dynamically require system "
                    f"libcudart.so.13: {dynamic['NEEDED']}",
                )
                _require(
                    not dynamic["RPATH"] and not dynamic["RUNPATH"],
                    f"{artifact.path.name} embeds a CUDA runtime search path: "
                    f"{dynamic}",
                )
            elif os.name == "nt":
                dumpbin = shutil.which("dumpbin")
                _require(
                    dumpbin is not None,
                    "dumpbin is required for Windows CUDA dependency validation",
                )
                result = subprocess.run(
                    [dumpbin, "/DEPENDENTS", str(native_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                binary = native_path.read_bytes().lower()
                loader_identities = (b"nvcuda.dll", b"nvcudart_hybrid64.dll")
                _require(
                    all(identity in binary for identity in loader_identities),
                    f"{artifact.path.name} must retain the CUDA 13.3 system "
                    "hybrid-loader identities nvcuda.dll and "
                    "nvcudart_hybrid64.dll; direct PE dependencies were: "
                    f"{result.stdout.strip()}",
                )





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
    expected = _load_rocm_system_policy(root)
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


def _rocm_system_policy_report(
    artifacts: list[Artifact], root: Path
) -> dict[str, object]:
    _require(artifacts, "ROCm policy report requires at least one wheel")
    wheel_reports = [
        _assert_rocm_system_wheel(artifact, root)
        for artifact in sorted(artifacts, key=lambda item: item.path.name)
    ]
    common_fields = (
        "policy_sha256",
        "wheel_policy",
        "rocm_version",
        "platform_tag",
        "userspace_bundled",
        "required_sonames",
    )
    first = wheel_reports[0]
    for report in wheel_reports[1:]:
        for field in common_fields:
            _require(
                report[field] == first[field],
                f"ROCm wheel policy report disagrees on {field}: {report['artifact']}",
            )

    artifact_fields = (
        "artifact",
        "wheel_bytes",
        "wheel_uncompressed_bytes",
        "native_payload_uncompressed_bytes",
    )
    return {
        "schema_version": 2,
        "distribution": "gafime-rocm",
        "wheel_count": len(wheel_reports),
        **{field: first[field] for field in common_fields},
        "wheels": [
            {field: report[field] for field in artifact_fields}
            for report in wheel_reports
        ],
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


def _assert_distribution_wheels(
    artifacts: list[Artifact], distribution_name: str
) -> None:
    distribution = RELEASE_MANIFEST.distribution(distribution_name)
    wheels = _select(artifacts, distribution_name, "wheel")
    expected_patterns = [
        pattern
        for wheel in distribution.wheels
        for pattern in wheel.filename_patterns
    ]
    for pattern in expected_patterns:
        matches = [
            artifact for artifact in wheels if fnmatchcase(artifact.path.name, pattern)
        ]
        _require(
            len(matches) == 1,
            f"{distribution_name} expected one wheel matching {pattern!r}, "
            f"found {[artifact.path.name for artifact in matches]}",
        )
    unmatched = sorted(
        artifact.path.name
        for artifact in wheels
        if not any(
            fnmatchcase(artifact.path.name, pattern) for pattern in expected_patterns
        )
    )
    _require(
        not unmatched,
        f"{distribution_name} contains wheels outside the manifest matrix: {unmatched}",
    )
    _require(
        len(wheels) == len(expected_patterns),
        f"{distribution_name} wheel count {len(wheels)} != manifest-derived "
        f"{len(expected_patterns)}",
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
    elif scope in {"cuda-wheel", "rocm-wheel"}:
        backend = scope.removesuffix("-wheel")
        expected_count = len(artifacts)
        _require(expected_count > 0, f"no {backend} wheels found")
        if backend == "rocm":
            _assert_distribution_wheels(artifacts, "gafime-rocm")
        for artifact in artifacts:
            _assert_payload_wheel(artifact, f"gafime-{backend}")
            if backend == "cuda":
                _assert_cuda_system_wheel(artifact)
            elif backend == "rocm":
                _assert_rocm_system_wheel(artifact, root)
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
        _assert_distribution_wheels(artifacts, distribution.name)
        _assert_core_sdist(
            _assert_one(_select(artifacts, "gafime", "sdist"), "core sdist"), root
        )
    elif scope == "cuda-release":
        distribution = RELEASE_MANIFEST.distribution("gafime-cuda")
        expected_count = distribution.artifact_count
        _assert_distribution_wheels(artifacts, distribution.name)
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-cuda", "sdist"), "CUDA sdist"),
            "gafime-cuda",
        )
        for artifact in artifacts:
            if artifact.kind == "wheel":
                _assert_cuda_system_wheel(artifact)
            else:
                _assert_cuda_build_policy(artifact, "off")
    elif scope == "rocm-release":
        distribution = RELEASE_MANIFEST.distribution("gafime-rocm")
        expected_count = distribution.artifact_count
        _assert_distribution_wheels(artifacts, distribution.name)
        _assert_payload_sdist(
            _assert_one(_select(artifacts, "gafime-rocm", "sdist"), "ROCm sdist"),
            "gafime-rocm",
        )
        for artifact in artifacts:
            if artifact.kind == "wheel":
                _assert_rocm_system_wheel(artifact, root)
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
    root: Path,
    release: ReleaseVersion,
    github_ref: str,
    git_sha: str | None,
    required: bool,
) -> None:
    expected = release.tag
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
        release_note = root / release.release_note
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
    global_selector = f'CIBW_BUILD: "{RELEASE_MANIFEST.build_selector}"'
    _require(
        global_selector in workflow,
        "build workflow CPython selector must match the release manifest",
    )

    checked_build_jobs: set[str] = set()
    checked_validation_jobs: set[tuple[str, tuple[str, ...]]] = set()
    required_freeze_dependencies: set[str] = set()
    for distribution in RELEASE_MANIFEST.standard_distributions:
        sdist_job = _workflow_job_block(workflow, distribution.sdist_build_job)
        _require(
            f"name: {distribution.sdist_artifact}" in sdist_job,
            f"{distribution.name} sdist artifact {distribution.sdist_artifact!r} "
            f"is absent from {distribution.sdist_build_job}",
        )
        required_freeze_dependencies.add(distribution.sdist_build_job)

        for wheel in distribution.wheels:
            build_job = _workflow_job_block(workflow, wheel.build_job)
            _require(
                wheel.artifact in build_job,
                f"{distribution.name}/{wheel.platform} artifact {wheel.artifact!r} "
                f"is absent from {wheel.build_job}",
            )
            required_freeze_dependencies.add(wheel.build_job)
            if wheel.build_job not in checked_build_jobs:
                selector = (
                    "CIBW_BUILD: ${{ env.CIBW_BUILD }}"
                    if wheel.python_versions == RELEASE_MANIFEST.supported_python
                    else f'CIBW_BUILD: "{wheel.build_selector}"'
                )
                _require(
                    selector in build_job,
                    f"{wheel.build_job} does not build its manifest-declared "
                    f"CPython matrix {wheel.build_selector!r}",
                )
                checked_build_jobs.add(wheel.build_job)

            validation_job = _workflow_job_block(workflow, wheel.validation_job)
            _require(
                wheel.validation_label in validation_job
                and wheel.artifact in validation_job,
                f"{wheel.validation_job} does not validate "
                f"{distribution.name}/{wheel.platform}",
            )
            required_freeze_dependencies.add(wheel.validation_job)
            validation_key = (wheel.validation_job, wheel.python_versions)
            if validation_key not in checked_validation_jobs:
                for version in wheel.python_versions:
                    _require(
                        f'"{version}"' in validation_job
                        or python_tag(version) in validation_job,
                        f"{wheel.validation_job} does not validate Python {version}",
                    )
                checked_validation_jobs.add(validation_key)

            if wheel.embedded_backends:
                _require(
                    wheel.embedded_backends == ("metal",),
                    f"unsupported embedded backend set for {wheel.platform}: "
                    f"{wheel.embedded_backends}",
                )
                _require(
                    "stage_metal_payload.py" in build_job,
                    "Apple Silicon Core wheels must stage Metal in the Core package",
                )
                _require(
                    "--backend metal" in validation_job
                    and "--execute-metal" in validation_job,
                    "every Apple Silicon Core wheel must execute its embedded Metal "
                    "payload during validation",
                )

    freeze_job = _workflow_job_block(workflow, "freeze_release_bundle")
    for dependency in sorted(required_freeze_dependencies):
        _require(
            f"- {dependency}" in freeze_job,
            f"frozen release bundle does not wait for {dependency}",
        )
    _require(
        f"pattern: {RELEASE_MANIFEST.bundle_download_pattern}" in freeze_job
        and f"name: {RELEASE_MANIFEST.bundle_artifact}" in freeze_job,
        "freeze job does not consume and publish the manifest-declared bundle",
    )
    _require(
        "--scope full-release" in freeze_job
        and ".github/scripts/release_bundle.py create" in freeze_job,
        "freeze job must validate complete composition before writing provenance",
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

def _assert_build_workflow(workflow: str) -> None:
    _assert_release_manifest_workflow(workflow)
    _require(
        "pull_request:" in workflow
        and "push:" in workflow
        and "workflow_dispatch:" in workflow,
        "build workflow must support review, mainline, and explicit frozen builds",
    )
    for forbidden in (
        "refs/tags/",
        "gh-action-pypi-publish",
        "softprops/action-gh-release",
        "build_cuda_rt",
        "gafime_cuda_rt",
        "gafime-cuda-rt",
        "gafime_rocm_bundled",
        "gafime-rocm-bundled",
        "rt_kernels.cu",
        "rt_launcher.cu",
        "GAFIME_OPTIX",
        "OPTIX_SDK",
    ):
        _require(
            forbidden not in workflow,
            f"build workflow contains forbidden release path {forbidden!r}",
        )
    _require(
        re.search(r"(?m)^\s+target\s*$", workflow) is None,
        "workflow caches must not archive build target directories",
    )
    _require(
        "auditwheel repair --plat manylinux_2_28_x86_64 "
        "--exclude libcudart.so.13" in workflow
        and (
            'delvewheel repair --exclude "cudart64_13.dll;'
            'nvcudart_hybrid64.dll"'
        )
        in workflow
        and "cudart_static.lib" not in workflow,
        "CUDA wheel repair must preserve the system-runtime boundary",
    )
    _require(
        "CUDA_CUDART_VERSION: '13.3.29'" in workflow
        and (
            "CUDA_CUDART_LINUX_URL: "
            "'https://developer.download.nvidia.com/compute/cuda/redist/"
            "cuda_cudart/linux-x86_64/"
            "cuda_cudart-linux-x86_64-13.3.29-archive.tar.xz'"
        )
        in workflow
        and (
            "CUDA_CUDART_LINUX_SHA256: "
            "'1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d'"
        )
        in workflow
        and (
            "CUDA_CUDART_WINDOWS_URL: "
            "'https://developer.download.nvidia.com/compute/cuda/redist/"
            "cuda_cudart/windows-x86_64/"
            "cuda_cudart-windows-x86_64-13.3.29-archive.zip'"
        )
        in workflow
        and (
            "CUDA_CUDART_WINDOWS_SHA256: "
            "'1feb7dd266813ffe8dbc24e115183a5ac35a4795c8d34aca0df85ab616b64d9c'"
        )
        in workflow
        and "Get-FileHash -Algorithm SHA256" in workflow
        and 'Copy-Item -Path $runtimeDll -Destination "$cudaRoot\\bin\\cudart64_13.dll"'  # noqa: E501
        in workflow,
        "Windows CUDA builds must use the pinned, verified NVIDIA runtime archive",
    )
    cuda_validator = _workflow_job_block(workflow, "validate_cuda_payload_wheels")
    _require(
        "Provision pinned CUDA runtime prerequisite (Linux)" in cuda_validator
        and '"$CUDA_CUDART_LINUX_URL"' in cuda_validator
        and '"$CUDA_CUDART_LINUX_SHA256"' in cuda_validator
        and "sha256sum --check --strict" in cuda_validator
        and 'test -f "$runtime_root/lib/libcudart.so.13"' in cuda_validator
        and '>> "$GITHUB_ENV"' in cuda_validator,
        "Linux CUDA validation must provision the pinned external runtime "
        "without modifying wheel artifacts",
    )
    _require(
        '"cudart_$componentVersion"' not in workflow,
        "the Windows network installer must not provide an unpinned CUDA runtime",
    )
    validator_conditions = {
        "validate_wheels": (
            "needs.build_wheels.result == 'success' && "
            "needs.build_arm_linux_wheels.result == 'success'"
        ),
        "validate_windows_arm_wheel": (
            "needs.build_arm_windows_wheels.result == 'success'"
        ),
        "validate_cuda_payload_wheels": (
            "needs.build_wheels.result == 'success' && "
            "needs.build_cuda_payload_wheels.result == 'success'"
        ),
        "validate_rocm_payload_wheels": (
            "needs.build_wheels.result == 'success' && "
            "needs.build_rocm_linux_payload_wheels.result == 'success'"
        ),
    }
    for job_name, condition in validator_conditions.items():
        job = _workflow_job_block(workflow, job_name)
        _require(
            condition in job and "!cancelled()" not in job,
            f"{job_name} must run only after successful artifact producers",
        )
    rocm_validator = _workflow_job_block(workflow, "validate_rocm_payload_wheels")
    _require(
        'python_abi_tag="${python_tag}-${python_tag}"' in rocm_validator
        and 'python="/opt/python/${python_abi_tag}/bin/python"' in rocm_validator
        and 'gafime-*-"${python_abi_tag}"-manylinux_2_28_x86_64.whl'
        in rocm_validator
        and 'gafime_rocm-*-"${python_abi_tag}"-linux_x86_64.whl'
        in rocm_validator
        and '"${python_tag}"-"${python_tag}"' not in rocm_validator,
        "ROCm validation must select one matching Core/payload pair per CPython ABI",
    )


def _assert_publish_workflow(workflow: str) -> None:
    _require(
        "workflow_dispatch:" in workflow
        and "\n  push:" not in workflow
        and "\n  pull_request:" not in workflow,
        "publisher must be an explicit manual workflow",
    )
    for forbidden in (
        "cibuildwheel",
        "python -m build",
        "maturin build",
        "auditwheel",
        "delvewheel",
        "retag_wheel",
        "repair-wheel",
        "wheel tags",
        "gafime_cuda_rt",
        "gafime-cuda-rt",
        "gafime_rocm_bundled",
        "gafime-rocm-bundled",
        "rt_kernels.cu",
        "rt_launcher.cu",
        "GAFIME_OPTIX",
        "OPTIX_SDK",
    ):
        _require(
            forbidden not in workflow,
            f"publisher contains forbidden build or distribution path {forbidden!r}",
        )

    preflight = _workflow_job_block(workflow, "publication_preflight")
    _require(
        ".github/workflows/build_wheels.yml" in preflight
        and "run-id: ${{ inputs.build_run_id }}" in preflight
        and "release_bundle.py verify" in preflight
        and "--scope full-release" in preflight,
        "publisher must bind a successful build run and revalidate its frozen bundle",
    )
    _require(
        "check_pypi_artifact_collisions.py" in preflight
        and "--allow-matching-existing" in preflight,
        "publisher must fail closed on PyPI collisions unless hashes match",
    )

    core = _workflow_job_block(workflow, "publish_pypi_core")
    cuda = _workflow_job_block(workflow, "publish_pypi_cuda")
    rocm = _workflow_job_block(workflow, "publish_pypi_rocm")
    _require(
        "needs: publication_preflight" in core,
        "Core publication must follow frozen-bundle preflight",
    )
    for name, job in (("CUDA", cuda), ("ROCm", rocm)):
        _require(
            "publish_pypi_core" in job,
            f"{name} payload must never publish before matching Core",
        )
    for name, job in (("Core", core), ("CUDA", cuda), ("ROCm", rocm)):
        _require(
            "release_bundle.py verify" in job
            and "pypa/gh-action-pypi-publish" in job,
            f"{name} publisher must verify then upload the frozen bytes",
        )
        _require(
            "skip-existing: ${{ inputs.allow_matching_existing_pypi_files }}" in job,
            f"{name} publisher may recover only through the hash-matched input",
        )
    _require(
        "gafime_rocm-*.tar.gz" in rocm and "gafime_rocm-*.whl" not in rocm,
        "PyPI ROCm publication must contain only the system-runtime sdist",
    )

    public_jobs = (
        "verify_public_core_and_cuda",
        "verify_public_windows_arm_core",
        "verify_public_rocm_install",
    )
    for job_name in public_jobs:
        job = _workflow_job_block(workflow, job_name)
        _require(
            "publish_pypi_cuda" in job and "publish_pypi_rocm" in job,
            f"{job_name} must run only after both payload publication lanes",
        )
    public_matrix = _workflow_job_block(workflow, "verify_public_core_and_cuda")
    for version in RELEASE_MANIFEST.supported_python:
        _require(
            f'"{version}"' in public_matrix,
            f"public install matrix is missing Python {version}",
        )
    _require(
        "--backend metal" in public_matrix and "--execute-metal" in public_matrix,
        "public Apple Silicon Core installation must execute bundled Metal",
    )
    _require(
        '"gafime==$PYPI_VERSION"' in public_matrix
        and '"gafime-cuda==$PYPI_VERSION"' in public_matrix,
        "public installation must pin Core and CUDA to the same release identity",
    )

    github_release = _workflow_job_block(workflow, "publish_github_release")
    for job_name in public_jobs:
        _require(
            f"- {job_name}" in github_release,
            f"GitHub Release must wait for {job_name}",
        )
    _require(
        "release_bundle.py verify" in github_release
        and "softprops/action-gh-release" in github_release
        and "files: dist/*" in github_release,
        "GitHub Release must publish the verified frozen bundle after public installs",
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
            'CMD ["gafime", "--check"' in text,
            f"{path.name} must execute the public CLI",
        )

    manifest_lines = {
        line.strip()
        for line in (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    }
    _require(
        "recursive-include src/common *" in manifest_lines,
        "Core source manifest must retain shared ABI headers",
    )
    _require(
        "recursive-include src *" not in manifest_lines
        and "recursive-include tests/gpu *" not in manifest_lines,
        "Core source manifest must exclude backend and GPU test sources",
    )

    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    _assert_release_manifest_pyproject(pyproject, _project_version(root))
    sdist_paths = {
        str(entry["path"])
        for entry in pyproject["tool"]["maturin"].get("include", [])
        if entry.get("format") == "sdist"
    }
    _require(
        "src/common/**/*" in sdist_paths
        and "src/**/*" not in sdist_paths
        and not any(path.startswith("tests/gpu") for path in sdist_paths),
        "Maturin Core sdist policy must include only shared native headers",
    )
    cargo_py = (root / "crates" / "gafime-py" / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    _require(
        "abi3" not in cargo_py and 'pyo3 = "0.27.2"' in cargo_py,
        "Core Python extension must use dedicated CPython wheels, not Stable ABI",
    )

    stage_path = root / ".github" / "scripts" / "stage_gpu_payload.py"
    stage_script = stage_path.read_text(encoding="utf-8")
    for token in (
        'license = "Apache-2.0"',
        'dependencies = ["gafime=={version}"]',
        "-DGAFIME_CUDA_DISTRIBUTION_NO_RT=1",
        '"cuda_runtime": "system"',
        '"linux": "libcudart.so.13"',
        '"windows": "nvcudart_hybrid64.dll"',
        '"rt_sources_included": False',
        'choices=("system",)',
    ):
        _require(token in stage_script, f"payload staging is missing {token!r}")
    for forbidden in (
        "py_limited_api",
        "Py_LIMITED_API",
        "abi3",
        "gafime_cuda_rt",
        "gafime-cuda-rt",
        "gafime_rocm_bundled",
        "gafime-rocm-bundled",
        '"rt_kernels.cu"',
        '"rt_launcher.cu"',
    ):
        _require(
            forbidden not in stage_script,
            f"payload staging contains forbidden policy {forbidden!r}",
        )
    _require(
        '"-cudart",\n            "shared"' in stage_script
        and '"-cudart",\n            "static"' not in stage_script,
        "distributed CUDA payload must dynamically link the system runtime",
    )
    _require(
        not (root / ".github" / "scripts" / "rocm_7_2_3_bundled_policy.json").exists(),
        "obsolete bundled-ROCm distribution policy must not remain active",
    )
    rocm_policy = _load_rocm_system_policy(root)
    _require(
        len(rocm_policy["gfx_targets"]) == 13,
        "system ROCm policy must retain every release code-object target",
    )

    with tempfile.TemporaryDirectory(prefix="gafime-payload-stage-") as temporary:
        temporary_root = Path(temporary)
        for backend in ("cuda", "rocm"):
            output = temporary_root / backend
            command = [sys.executable, str(stage_path), backend, str(output)]
            if backend == "rocm":
                command.extend(("--rocm-wheel-policy", "system"))
            subprocess.run(command, cwd=root, check=True)
            forbidden_files = sorted(
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
                and (
                    path.name
                    in {
                        "rt_kernels.cu",
                        "rt_kernels.cuh",
                        "rt_launcher.cu",
                        "rt_launcher.cuh",
                    }
                    or "optix" in path.as_posix().lower()
                )
            )
            _require(
                not forbidden_files,
                f"{backend} staged distribution contains RT sources: "
                f"{forbidden_files}",
            )

    metal_stage = (
        root / ".github" / "scripts" / "stage_metal_payload.py"
    ).read_text(encoding="utf-8")
    _require(
        'default=REPO_ROOT / "python" / "gafime" / "_metal"' in metal_stage
        and '"-DCMAKE_OSX_ARCHITECTURES=arm64"' in metal_stage,
        "Metal must stage only into the Apple Silicon Core package",
    )
    _require(
        not (root / ".github" / "scripts" / "stage_metal_distribution.py").exists(),
        "a separate Metal distribution path must not exist",
    )

    cuda_cmake = (root / "src" / "cuda" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    _require(
        "GAFIME_CUDA_RT_BUILD_MODE" in cuda_cmake
        and 'PROPERTY STRINGS off on both' in cuda_cmake,
        "experimental RT must remain available only through local CMake selection",
    )

    build_workflow = (root / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )
    publish_workflow = (
        root / ".github" / "workflows" / "publish_release.yml"
    ).read_text(encoding="utf-8")
    _assert_build_workflow(build_workflow)
    _assert_publish_workflow(publish_workflow)
    _assert_release_manifest_documentation(root)

    collision_script = (
        root / ".github" / "scripts" / "check_pypi_artifact_collisions.py"
    )
    collision_test = subprocess.run(
        [sys.executable, str(collision_script), "--self-test"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        collision_test.returncode == 0
        and "PYPI COLLISION SELF-TEST: PASS" in collision_test.stdout,
        "PyPI collision preflight self-test failed: "
        f"stdout={collision_test.stdout!r} stderr={collision_test.stderr!r}",
    )

    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / ".github" / "workflows").glob("*.yml"))
    )
    _require(
        "rustup default stable" not in workflow_text,
        "workflows must not select floating Rust stable",
    )
    for retired_staging_option in (
        "--cuda-rt",
        "--rocm-wheel-policy bundled",
    ):
        _require(
            retired_staging_option not in workflow_text,
            f"workflow still invokes retired payload option "
            f"{retired_staging_option!r}",
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
            "rocm-sdist",
            "cuda-wheel",
            "rocm-wheel",
            "sdists",
            "core-release",
            "cuda-release",
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
    parser.add_argument("--write-rocm-report", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    release = validate_project_versions(root)
    version = release.pep440
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
            release,
            args.github_ref,
            args.git_sha,
            args.require_release_tag,
        )
    if args.write_checksums is not None:
        _write_checksums(artifacts, args.write_checksums.resolve())
    if args.write_rocm_report is not None:
        rocm_wheels = _select(artifacts, "gafime-rocm", "wheel")
        report = _rocm_system_policy_report(rocm_wheels, root)
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
