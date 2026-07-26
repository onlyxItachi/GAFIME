#!/usr/bin/env python3
"""Load and render the authoritative GAFIME release artifact manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class WheelSpec:
    platform: str
    artifact: str
    build_job: str
    validation_job: str
    validation_label: str
    filename_pattern: str


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    package: str
    wheel_prefix: str
    kind: str
    backend: str | None
    policy: str
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
        return len(self.wheels) + 1


@dataclass(frozen=True)
class ExcludedDistributionSpec:
    name: str
    package: str
    backend: str
    policy: str
    artifact: str | None
    wheel_platforms: tuple[str, ...]
    sdist: bool
    pypi: bool
    reason: str


@dataclass(frozen=True)
class ReleaseManifest:
    path: Path
    schema_version: int
    bundle_artifact: str
    bundle_download_pattern: str
    python_tag: str
    abi_tag: str
    build_selector: str
    supported_python: tuple[str, ...]
    distributions: tuple[DistributionSpec, ...]
    excluded_distributions: tuple[ExcludedDistributionSpec, ...]

    @property
    def standard_distributions(self) -> tuple[DistributionSpec, ...]:
        return tuple(item for item in self.distributions if item.standard_bundle)

    @property
    def standard_artifact_count(self) -> int:
        return sum(item.artifact_count for item in self.standard_distributions)

    @property
    def all_distribution_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.distributions) + tuple(
            item.name for item in self.excluded_distributions
        )

    def distribution(self, name: str) -> DistributionSpec:
        for item in self.distributions:
            if item.name == name:
                return item
        raise AssertionError(f"release manifest has no standard distribution {name!r}")

    def excluded_distribution(self, name: str) -> ExcludedDistributionSpec:
        for item in self.excluded_distributions:
            if item.name == name:
                return item
        raise AssertionError(f"release manifest has no excluded distribution {name!r}")


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


def _parse_wheel(value: Any, context: str) -> WheelSpec:
    data = _object(value, context)
    fields = {
        "platform",
        "artifact",
        "build_job",
        "validation_job",
        "validation_label",
        "filename_pattern",
    }
    _exact_keys(data, fields, context)
    return WheelSpec(
        **{name: _string(data[name], f"{context}.{name}") for name in fields}
    )


def _parse_distribution(value: Any, index: int) -> DistributionSpec:
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
        _parse_wheel(item, f"{context}.wheels[{wheel_index}]")
        for wheel_index, item in enumerate(wheels_data)
    )
    platforms = [wheel.platform for wheel in wheels]
    _require(
        len(platforms) == len(set(platforms)),
        f"{context} has duplicate wheel platforms: {platforms}",
    )
    return DistributionSpec(
        name=_string(data["name"], f"{context}.name"),
        package=_string(data["package"], f"{context}.package"),
        wheel_prefix=_string(data["wheel_prefix"], f"{context}.wheel_prefix"),
        kind=_string(data["kind"], f"{context}.kind"),
        backend=backend,
        policy=_string(data["policy"], f"{context}.policy"),
        standard_bundle=data["standard_bundle"],
        pypi_wheels=pypi["wheels"],
        pypi_sdist=pypi["sdist"],
        extra_name=extra_name,
        extra_marker=extra_marker,
        sdist_artifact=_string(sdist["artifact"], f"{context}.sdist.artifact"),
        sdist_build_job=_string(sdist["build_job"], f"{context}.sdist.build_job"),
        wheels=wheels,
    )


def _parse_excluded(value: Any, index: int) -> ExcludedDistributionSpec:
    context = f"release manifest excluded_distributions[{index}]"
    data = _object(value, context)
    fields = {
        "name",
        "package",
        "backend",
        "policy",
        "artifact",
        "wheel_platforms",
        "sdist",
        "pypi",
        "reason",
    }
    _exact_keys(data, fields, context)
    artifact = data["artifact"]
    _require(
        artifact is None or isinstance(artifact, str), f"{context}.artifact is invalid"
    )
    platforms = data["wheel_platforms"]
    _require(
        isinstance(platforms, list)
        and bool(platforms)
        and all(isinstance(item, str) and item for item in platforms),
        f"{context}.wheel_platforms is invalid",
    )
    _require(
        isinstance(data["sdist"], bool) and isinstance(data["pypi"], bool),
        f"{context} publication fields must be booleans",
    )
    return ExcludedDistributionSpec(
        name=_string(data["name"], f"{context}.name"),
        package=_string(data["package"], f"{context}.package"),
        backend=_string(data["backend"], f"{context}.backend"),
        policy=_string(data["policy"], f"{context}.policy"),
        artifact=artifact,
        wheel_platforms=tuple(platforms),
        sdist=data["sdist"],
        pypi=data["pypi"],
        reason=_string(data["reason"], f"{context}.reason"),
    )


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
            "excluded_distributions",
        },
        "release manifest",
    )
    _require(data["schema_version"] == 1, "release manifest schema_version must be 1")
    bundle = _object(data["bundle"], "release manifest bundle")
    _exact_keys(
        bundle, {"artifact_name", "download_pattern"}, "release manifest bundle"
    )
    python = _object(data["python"], "release manifest python")
    _exact_keys(
        python,
        {"python_tag", "abi_tag", "build_selector", "supported_versions"},
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
    distributions_data = data["distributions"]
    excluded_data = data["excluded_distributions"]
    _require(
        isinstance(distributions_data, list),
        "release manifest distributions must be a list",
    )
    _require(
        isinstance(excluded_data, list),
        "release manifest excluded_distributions must be a list",
    )
    distributions = tuple(
        _parse_distribution(item, index)
        for index, item in enumerate(distributions_data)
    )
    excluded = tuple(
        _parse_excluded(item, index) for index, item in enumerate(excluded_data)
    )
    names = [item.name for item in distributions] + [item.name for item in excluded]
    packages = [item.package for item in distributions] + [
        item.package for item in excluded
    ]
    _require(
        len(names) == len(set(names)), f"release manifest has duplicate names: {names}"
    )
    _require(
        len(packages) == len(set(packages)),
        f"release manifest has duplicate package identities: {packages}",
    )
    _require(
        [item.name for item in distributions]
        == ["gafime", "gafime-cuda", "gafime-rocm", "gafime-metal"],
        "standard release distributions must remain core, CUDA, ROCm, then Metal",
    )
    _require(
        all(item.standard_bundle for item in distributions),
        "all primary distributions must belong to the standard bundle",
    )
    _require(
        all(not item.pypi for item in excluded),
        "excluded distributions must not be publishable to PyPI",
    )
    manifest = ReleaseManifest(
        path=path,
        schema_version=1,
        bundle_artifact=_string(
            bundle["artifact_name"], "release manifest bundle.artifact_name"
        ),
        bundle_download_pattern=_string(
            bundle["download_pattern"], "release manifest bundle.download_pattern"
        ),
        python_tag=_string(python["python_tag"], "release manifest python.python_tag"),
        abi_tag=_string(python["abi_tag"], "release manifest python.abi_tag"),
        build_selector=_string(
            python["build_selector"], "release manifest python.build_selector"
        ),
        supported_python=tuple(supported),
        distributions=distributions,
        excluded_distributions=excluded,
    )
    python_tag_match = re.fullmatch(r"cp([0-9])([0-9]+)", manifest.python_tag)
    _require(
        python_tag_match is not None,
        "release manifest python_tag must use a canonical cp<major><minor> tag",
    )
    minimum_python = (
        f"{python_tag_match.group(1)}.{python_tag_match.group(2)}"
        if python_tag_match is not None
        else ""
    )
    _require(
        manifest.supported_python[0] == minimum_python,
        f"release manifest supported Python range must start at {minimum_python}",
    )
    _require(
        manifest.build_selector == f"{manifest.python_tag}-*",
        "release manifest build_selector must build only the minimum Stable ABI "
        "interpreter",
    )
    _require(
        manifest.abi_tag == "abi3",
        "release manifest ABI must remain Python's Stable ABI",
    )
    return manifest


def render_release_matrix(manifest: ReleaseManifest) -> str:
    rows = []
    for distribution in manifest.standard_distributions:
        wheel_platforms = ", ".join(
            f"`{wheel.platform}`" for wheel in distribution.wheels
        )
        pypi = []
        if distribution.pypi_wheels:
            pypi.append("wheels")
        if distribution.pypi_sdist:
            pypi.append("sdist")
        rows.append(
            f"| `{distribution.name}` | {distribution.kind} | {wheel_platforms} | "
            f"yes | {', '.join(pypi) or 'none'} | {distribution.artifact_count} |"
        )
    exclusions = "\n".join(
        f"- `{item.name}` (`{item.policy}`): {item.reason}"
        for item in manifest.excluded_distributions
    )
    versions = ", ".join(f"`{version}`" for version in manifest.supported_python)
    return (
        "# GAFIME Release Artifact Matrix\n\n"
        "<!-- Generated from .github/release-artifacts.json; do not edit by hand. -->\n\n"
        f"The standard GitHub release bundle contains **{manifest.standard_artifact_count} "
        "artifacts**. Every wheel is built once with "
        f"`{manifest.python_tag}-{manifest.abi_tag}` and the same frozen wheel is "
        f"installed and tested on CPython {versions}.\n\n"
        "| Distribution | Kind | Wheel platforms | Sdist | PyPI publication | Count |\n"
        "|---|---|---|---:|---|---:|\n" + "\n".join(rows) + "\n\n"
        "## Excluded Identities\n\n" + exclusions + "\n"
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
