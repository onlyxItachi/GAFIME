#!/usr/bin/env python3
"""Load and render the authoritative GAFIME release artifact manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


CANONICAL_PRECISION_PROFILES = ("fp32", "mixed", "fp64")


@dataclass(frozen=True)
class WheelSpec:
    platform: str
    artifact: str
    build_job: str
    validation_job: str
    validation_label: str
    python_versions: tuple[str, ...]
    embedded_backends: tuple[str, ...]
    embedded_backend_profiles: tuple[tuple[str, tuple[str, ...]], ...]
    filename_template: str

    @property
    def build_selector(self) -> str:
        return " ".join(f"{python_tag(version)}-*" for version in self.python_versions)

    def filename_pattern(self, version: str) -> str:
        _require(
            version in self.python_versions,
            f"{self.platform} does not build CPython {version}",
        )
        return self.filename_template.format(python_tag=python_tag(version))

    @property
    def filename_patterns(self) -> tuple[str, ...]:
        return tuple(self.filename_pattern(version) for version in self.python_versions)

    def profiles_for_backend(self, backend: str) -> tuple[str, ...]:
        for candidate, profiles in self.embedded_backend_profiles:
            if candidate == backend:
                return profiles
        raise AssertionError(
            f"{self.platform} has no embedded precision contract for {backend!r}"
        )


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    package: str
    wheel_prefix: str
    kind: str
    backend: str | None
    policy: str
    precision_profiles: tuple[str, ...]
    standard_bundle: bool
    pypi_wheels: bool
    pypi_sdist: bool
    extra_name: str | None
    extra_marker: str | None
    sdist_artifact: str
    sdist_build_job: str
    wheels: tuple[WheelSpec, ...]

    @property
    def artifact_count(self) -> int:
        return sum(len(wheel.python_versions) for wheel in self.wheels) + 1

    @property
    def execution_backend(self) -> str:
        return self.backend or "core"


@dataclass(frozen=True)
class ReleaseManifest:
    path: Path
    schema_version: int
    bundle_artifact: str
    bundle_download_pattern: str
    abi_policy: str
    supported_python: tuple[str, ...]
    distributions: tuple[DistributionSpec, ...]

    @property
    def build_selector(self) -> str:
        return " ".join(f"{python_tag(version)}-*" for version in self.supported_python)

    @property
    def standard_distributions(self) -> tuple[DistributionSpec, ...]:
        return tuple(item for item in self.distributions if item.standard_bundle)

    @property
    def standard_artifact_count(self) -> int:
        return sum(item.artifact_count for item in self.standard_distributions)

    @property
    def all_distribution_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.distributions)

    def distribution(self, name: str) -> DistributionSpec:
        for item in self.distributions:
            if item.name == name:
                return item
        raise AssertionError(f"release manifest has no standard distribution {name!r}")


def python_tag(version: str) -> str:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", version)
    _require(match is not None, f"invalid Python version {version!r}")
    assert match is not None
    return f"cp{match.group(1)}{match.group(2)}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    _require(
        isinstance(value, str) and bool(value), f"{context} must be a non-empty string"
    )
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{context} fields {sorted(actual)} != {sorted(expected)}",
    )


def _parse_wheel(
    value: Any, context: str, supported_python: tuple[str, ...]
) -> WheelSpec:
    data = _object(value, context)
    fields = {
        "platform",
        "artifact",
        "build_job",
        "validation_job",
        "validation_label",
        "python_versions",
        "embedded_backends",
        "embedded_backend_profiles",
        "filename_template",
    }
    _exact_keys(data, fields, context)
    configured_python = data["python_versions"]
    if configured_python == "all":
        resolved_python = supported_python
    else:
        _require(
            isinstance(configured_python, list)
            and bool(configured_python)
            and all(
                isinstance(version, str) and version in supported_python
                for version in configured_python
            ),
            f"{context}.python_versions must be 'all' or a supported version list",
        )
        resolved_python = tuple(configured_python)
    _require(
        resolved_python == supported_python,
        f"{context} must cover the complete supported CPython matrix",
    )
    embedded_backends = data["embedded_backends"]
    _require(
        isinstance(embedded_backends, list)
        and all(isinstance(backend, str) and backend for backend in embedded_backends),
        f"{context}.embedded_backends must be a string list",
    )
    _require(
        len(embedded_backends) == len(set(embedded_backends)),
        f"{context}.embedded_backends must be unique",
    )
    embedded_profiles_data = _object(
        data["embedded_backend_profiles"],
        f"{context}.embedded_backend_profiles",
    )
    _require(
        set(embedded_profiles_data) == set(embedded_backends),
        f"{context}.embedded_backend_profiles must cover exactly embedded_backends",
    )
    embedded_backend_profiles = []
    for backend in embedded_backends:
        profiles = embedded_profiles_data[backend]
        _require(
            isinstance(profiles, list)
            and bool(profiles)
            and all(profile in CANONICAL_PRECISION_PROFILES for profile in profiles)
            and len(profiles) == len(set(profiles)),
            f"{context}.embedded_backend_profiles.{backend} is invalid",
        )
        embedded_backend_profiles.append((backend, tuple(profiles)))
    string_fields = fields - {
        "python_versions",
        "embedded_backends",
        "embedded_backend_profiles",
    }
    wheel = WheelSpec(
        **{name: _string(data[name], f"{context}.{name}") for name in string_fields},
        python_versions=resolved_python,
        embedded_backends=tuple(embedded_backends),
        embedded_backend_profiles=tuple(embedded_backend_profiles),
    )
    _require(
        wheel.filename_template.count("{python_tag}") == 2,
        f"{context}.filename_template must contain two {{python_tag}} placeholders",
    )
    return wheel


def _parse_distribution(
    value: Any, index: int, supported_python: tuple[str, ...]
) -> DistributionSpec:
    context = f"release manifest distributions[{index}]"
    data = _object(value, context)
    _exact_keys(
        data,
        {
            "name",
            "package",
            "wheel_prefix",
            "kind",
            "backend",
            "policy",
            "precision_profiles",
            "standard_bundle",
            "pypi",
            "extra",
            "sdist",
            "wheels",
        },
        context,
    )
    backend = data["backend"]
    _require(
        backend is None or isinstance(backend, str), f"{context}.backend is invalid"
    )
    _require(
        isinstance(data["standard_bundle"], bool),
        f"{context}.standard_bundle is invalid",
    )
    precision_profiles = data["precision_profiles"]
    _require(
        isinstance(precision_profiles, list)
        and tuple(precision_profiles) == CANONICAL_PRECISION_PROFILES,
        f"{context}.precision_profiles must be {list(CANONICAL_PRECISION_PROFILES)!r}",
    )
    pypi = _object(data["pypi"], f"{context}.pypi")
    _exact_keys(pypi, {"wheels", "sdist"}, f"{context}.pypi")
    _require(
        isinstance(pypi["wheels"], bool) and isinstance(pypi["sdist"], bool),
        f"{context}.pypi values must be booleans",
    )
    extra = data["extra"]
    if extra is None:
        extra_name = None
        extra_marker = None
    else:
        extra_data = _object(extra, f"{context}.extra")
        _exact_keys(extra_data, {"name", "marker"}, f"{context}.extra")
        extra_name = _string(extra_data["name"], f"{context}.extra.name")
        extra_marker = _string(extra_data["marker"], f"{context}.extra.marker")
    sdist = _object(data["sdist"], f"{context}.sdist")
    _exact_keys(sdist, {"artifact", "build_job"}, f"{context}.sdist")
    wheels_data = data["wheels"]
    _require(
        isinstance(wheels_data, list) and bool(wheels_data),
        f"{context}.wheels is empty",
    )
    wheels = tuple(
        _parse_wheel(
            item,
            f"{context}.wheels[{wheel_index}]",
            supported_python,
        )
        for wheel_index, item in enumerate(wheels_data)
    )
    platforms = [wheel.platform for wheel in wheels]
    _require(
        len(platforms) == len(set(platforms)),
        f"{context} has duplicate wheel platforms: {platforms}",
    )
    distribution = DistributionSpec(
        name=_string(data["name"], f"{context}.name"),
        package=_string(data["package"], f"{context}.package"),
        wheel_prefix=_string(data["wheel_prefix"], f"{context}.wheel_prefix"),
        kind=_string(data["kind"], f"{context}.kind"),
        backend=backend,
        policy=_string(data["policy"], f"{context}.policy"),
        precision_profiles=tuple(precision_profiles),
        standard_bundle=data["standard_bundle"],
        pypi_wheels=pypi["wheels"],
        pypi_sdist=pypi["sdist"],
        extra_name=extra_name,
        extra_marker=extra_marker,
        sdist_artifact=_string(sdist["artifact"], f"{context}.sdist.artifact"),
        sdist_build_job=_string(sdist["build_job"], f"{context}.sdist.build_job"),
        wheels=wheels,
    )
    for wheel in distribution.wheels:
        _require(
            wheel.filename_template.startswith(f"{distribution.wheel_prefix}-*-")
            and "abi3" not in wheel.filename_template,
            f"{context}/{wheel.platform} filename template must use the "
            "distribution prefix and matching per-CPython ABI",
        )
    return distribution


def load_release_manifest(root: Path) -> ReleaseManifest:
    path = root / ".github" / "release-artifacts.json"
    data = _object(json.loads(path.read_text(encoding="utf-8")), "release manifest")
    _exact_keys(
        data,
        {
            "schema_version",
            "bundle",
            "python",
            "distributions",
        },
        "release manifest",
    )
    _require(data["schema_version"] == 3, "release manifest schema_version must be 3")
    bundle = _object(data["bundle"], "release manifest bundle")
    _exact_keys(
        bundle, {"artifact_name", "download_pattern"}, "release manifest bundle"
    )
    python = _object(data["python"], "release manifest python")
    _exact_keys(
        python,
        {"abi_policy", "supported_versions"},
        "release manifest python",
    )
    supported = python["supported_versions"]
    _require(
        isinstance(supported, list)
        and bool(supported)
        and all(
            isinstance(version, str)
            and re.fullmatch(r"[0-9]+\.[0-9]+", version) is not None
            for version in supported
        ),
        "release manifest supported Python versions must be major.minor strings",
    )
    _require(
        len(supported) == len(set(supported)),
        "release manifest supported Python versions must be unique",
    )
    _require(
        supported == ["3.10", "3.11", "3.12", "3.13", "3.14"],
        "release Python support must remain CPython 3.10 through 3.14",
    )
    distributions_data = data["distributions"]
    _require(
        isinstance(distributions_data, list),
        "release manifest distributions must be a list",
    )
    distributions = tuple(
        _parse_distribution(item, index, tuple(supported))
        for index, item in enumerate(distributions_data)
    )
    names = [item.name for item in distributions]
    packages = [item.package for item in distributions]
    _require(
        len(names) == len(set(names)), f"release manifest has duplicate names: {names}"
    )
    _require(
        len(packages) == len(set(packages)),
        f"release manifest has duplicate package identities: {packages}",
    )
    _require(
        [item.name for item in distributions]
        == ["gafime", "gafime-cuda", "gafime-rocm"],
        "standard release distributions must remain core, CUDA, then ROCm",
    )
    _require(
        all(item.standard_bundle for item in distributions),
        "all primary distributions must belong to the standard bundle",
    )
    expected_distribution_policy = {
        "gafime": {
            "package": "gafime",
            "wheel_prefix": "gafime",
            "kind": "core",
            "backend": None,
            "platforms": {
                "manylinux_2_28_x86_64",
                "manylinux_2_28_aarch64",
                "macosx_11_0_arm64",
                "win_amd64",
                "win_arm64",
            },
            "pypi": (True, True),
            "policy": "core-with-metal-on-apple-silicon",
            "precision_profiles": CANONICAL_PRECISION_PROFILES,
        },
        "gafime-cuda": {
            "package": "gafime_cuda",
            "wheel_prefix": "gafime_cuda",
            "kind": "payload",
            "backend": "cuda",
            "platforms": {"manylinux_2_28_x86_64", "win_amd64"},
            "pypi": (True, True),
            "policy": "system",
            "precision_profiles": CANONICAL_PRECISION_PROFILES,
        },
        "gafime-rocm": {
            "package": "gafime_rocm",
            "wheel_prefix": "gafime_rocm",
            "kind": "payload",
            "backend": "rocm",
            "platforms": {"linux_x86_64"},
            "pypi": (False, True),
            "policy": "system",
            "precision_profiles": CANONICAL_PRECISION_PROFILES,
        },
    }
    for distribution in distributions:
        expected = expected_distribution_policy[distribution.name]
        _require(
            (
                distribution.package,
                distribution.wheel_prefix,
                distribution.kind,
                distribution.backend,
            )
            == (
                expected["package"],
                expected["wheel_prefix"],
                expected["kind"],
                expected["backend"],
            ),
            f"{distribution.name} identity violates pinned distribution policy",
        )
        _require(
            {wheel.platform for wheel in distribution.wheels} == expected["platforms"],
            f"{distribution.name} wheel platforms violate pinned distribution policy",
        )
        _require(
            (distribution.pypi_wheels, distribution.pypi_sdist) == expected["pypi"],
            f"{distribution.name} PyPI policy violates pinned distribution policy",
        )
        _require(
            distribution.policy == expected["policy"],
            f"{distribution.name} build policy violates pinned distribution policy",
        )
        _require(
            distribution.precision_profiles == expected["precision_profiles"],
            f"{distribution.name} precision profiles violate pinned distribution policy",
        )
        _require(
            distribution.extra_name is None and distribution.extra_marker is None,
            f"{distribution.name} must not be exposed through Core extras",
        )
    manifest = ReleaseManifest(
        path=path,
        schema_version=3,
        bundle_artifact=_string(
            bundle["artifact_name"], "release manifest bundle.artifact_name"
        ),
        bundle_download_pattern=_string(
            bundle["download_pattern"], "release manifest bundle.download_pattern"
        ),
        abi_policy=_string(python["abi_policy"], "release manifest python.abi_policy"),
        supported_python=tuple(supported),
        distributions=distributions,
    )
    _require(
        manifest.abi_policy == "per-cpython",
        "release manifest must build dedicated per-CPython wheels",
    )
    embedded = [
        (
            distribution.name,
            wheel.platform,
            backend,
            wheel.profiles_for_backend(backend),
        )
        for distribution in manifest.standard_distributions
        for wheel in distribution.wheels
        for backend in wheel.embedded_backends
    ]
    _require(
        embedded == [("gafime", "macosx_11_0_arm64", "metal", ("fp32",))],
        "Metal fp32 must be embedded only in the Apple Silicon core wheel",
    )
    for distribution in manifest.standard_distributions:
        _require(
            not any(
                token in distribution.name.lower()
                for token in CANONICAL_PRECISION_PROFILES
            ),
            f"precision-specific distribution identity is forbidden: {distribution.name}",
        )
    return manifest


def render_release_matrix(manifest: ReleaseManifest) -> str:
    rows = []
    wheel_count = sum(
        len(wheel.python_versions)
        for distribution in manifest.standard_distributions
        for wheel in distribution.wheels
    )
    sdist_count = len(manifest.standard_distributions)
    checksum_count = manifest.standard_artifact_count + 1
    frozen_file_count = manifest.standard_artifact_count + 2
    for distribution in manifest.standard_distributions:
        policy_label = (
            "Core; Metal embedded on Apple Silicon"
            if distribution.policy == "core-with-metal-on-apple-silicon"
            else f"system {distribution.backend.upper()} runtime"
        )
        wheel_platforms = ", ".join(
            f"`{wheel.platform}`" for wheel in distribution.wheels
        )
        pypi = []
        if distribution.pypi_wheels:
            pypi.append("wheels")
        if distribution.pypi_sdist:
            pypi.append("sdist")
        profiles = ", ".join(
            f"`{profile}`" for profile in distribution.precision_profiles
        )
        embedded = ", ".join(
            f"`{backend}` ({', '.join(f'`{profile}`' for profile in wheel.profiles_for_backend(backend))}) "
            f"in `{wheel.platform}`"
            for wheel in distribution.wheels
            for backend in wheel.embedded_backends
        )
        rows.append(
            f"| `{distribution.name}` | {distribution.kind} | "
            f"{policy_label} | {wheel_platforms} | "
            f"{profiles} | {embedded or 'none'} | yes | {', '.join(pypi) or 'none'} | "
            f"{distribution.artifact_count} |"
        )
    versions = ", ".join(f"`{version}`" for version in manifest.supported_python)
    return (
        "# GAFIME Release Artifact Matrix\n\n"
        "<!-- Generated from .github/release-artifacts.json; do not edit by hand. -->\n\n"
        f"The standard package set contains **{manifest.standard_artifact_count} package "
        f"artifacts**: **{wheel_count} wheels** and **{sdist_count} sdists**, derived "
        "from the manifest's per-CPython/platform matrix. The frozen bundle contains "
        f"**{frozen_file_count} files** after adding provenance and `SHA256SUMS`; "
        f"`SHA256SUMS` covers **{checksum_count} entries** (the packages plus "
        "provenance). "
        f"Dedicated wheels are built and tested for CPython {versions}; "
        "every declared platform covers this complete matrix. "
        "Python's Stable ABI is not used.\n\n"
        "Every listed profile is compiled into each wheel of its distribution; "
        "profiles do not create additional distributions or wheel families.\n\n"
        "| Backend | `fp32` | `mixed` | `fp64` |\n"
        "|---|---:|---:|---:|\n"
        "| Core | yes | yes | yes |\n"
        "| CUDA | yes | yes | yes |\n"
        "| ROCm | yes | yes | yes |\n"
        "| Metal | yes | no | no |\n\n"
        "| Distribution | Kind | Runtime policy | Wheel platforms | "
        "Primary profiles | Embedded backends and profiles | Sdist | PyPI publication | Count |\n"
        "|---|---|---|---|---|---|---:|---|---:|\n" + "\n".join(rows) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in generated Markdown differs from the manifest",
    )
    output.add_argument(
        "--write",
        action="store_true",
        help="rewrite the checked-in generated Markdown from the manifest",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = load_release_manifest(root)
    rendered = render_release_matrix(manifest)
    output_path = root / "docs" / "releases" / "release-artifact-matrix.md"
    if args.check:
        _require(
            output_path.read_text(encoding="utf-8") == rendered,
            f"{output_path.relative_to(root)} differs from {manifest.path.relative_to(root)}",
        )
        print(
            f"RELEASE MANIFEST DOC: PASS artifacts={manifest.standard_artifact_count}"
        )
    elif args.write:
        output_path.write_text(rendered, encoding="utf-8")
        print(f"wrote {output_path.relative_to(root)}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
