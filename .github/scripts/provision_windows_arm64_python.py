#!/usr/bin/env python3
"""Provision a Windows ARM64 target interpreter through cibuildwheel's NuGet path."""

from __future__ import annotations

import argparse
from importlib.metadata import version as distribution_version
import json
import os
from pathlib import Path
import platform
import re
import subprocess


CIBUILDWHEEL_VERSION = "3.4.1"
IDENTIFIER_PATTERN = re.compile(r"cp(?P<major>[0-9])(?P<minor>[0-9]{1,2})-win_arm64")


def resolve_configuration(identifier: str):
    match = IDENTIFIER_PATTERN.fullmatch(identifier)
    if match is None:
        raise ValueError(
            f"unsupported Windows ARM64 CPython identifier: {identifier!r}"
        )
    if distribution_version("cibuildwheel") != CIBUILDWHEEL_VERSION:
        raise ValueError(
            f"cibuildwheel {CIBUILDWHEEL_VERSION} is required by the release contract"
        )

    from cibuildwheel.architecture import Architecture
    from cibuildwheel.platforms.windows import (
        get_nuget_args,
        get_python_configurations,
    )
    from cibuildwheel.selector import BuildSelector

    selector = BuildSelector(build_config=identifier, skip_config="")
    configurations = get_python_configurations(selector, {Architecture.ARM64})
    if len(configurations) != 1:
        raise ValueError(
            f"expected one cibuildwheel configuration for {identifier}, "
            f"found {len(configurations)}"
        )
    configuration = configurations[0]
    nuget_args = get_nuget_args(
        configuration.version,
        "ARM64",
        False,
        Path("nuget-cpython"),
    )
    if nuget_args[0] != "pythonarm64":
        raise ValueError(
            f"{identifier} resolved to unexpected NuGet package {nuget_args[0]!r}"
        )
    expected_minor = f"{match.group('major')}.{match.group('minor')}"
    if not configuration.version.startswith(f"{expected_minor}."):
        raise ValueError(
            f"{identifier} resolved to unexpected Python {configuration.version}"
        )
    return configuration


def provision(configuration, venv_path: Path) -> Path:
    if os.name != "nt" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "Windows ARM64 target provisioning requires a native Windows ARM64 runner"
        )
    if not venv_path.is_absolute():
        raise ValueError("target virtual environment path must be absolute")

    from cibuildwheel.platforms.windows import install_cpython
    from cibuildwheel.venv import virtualenv

    base_python = install_cpython(configuration)
    virtualenv(
        configuration.version,
        base_python,
        venv_path,
        dependency_constraint=None,
        use_uv=False,
    )
    target_python = venv_path / "Scripts" / "python.exe"
    expected_minor = ".".join(configuration.version.split(".")[:2])
    subprocess.run(
        [
            str(target_python),
            "-c",
            (
                "import platform, struct, sys; "
                "expected=tuple(map(int, sys.argv[1].split('.'))); "
                "assert sys.version_info[:2] == expected; "
                "assert struct.calcsize('P') == 8; "
                "assert platform.machine().lower() in {'arm64', 'aarch64'}"
            ),
            expected_minor,
        ],
        check=True,
    )
    return target_python


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--github-output")
    parser.add_argument("--github-env")
    parser.add_argument("--venv")
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()

    configuration = resolve_configuration(args.identifier)
    result = {
        "identifier": configuration.identifier,
        "nuget_package": "pythonarm64",
        "python_version": configuration.version,
    }
    if not args.resolve_only:
        if not args.venv:
            parser.error("--venv is required when provisioning")
        target_python = provision(configuration, Path(args.venv))
        output_path = args.github_output or os.environ.get("GITHUB_OUTPUT")
        if not output_path:
            parser.error(
                "--github-output or GITHUB_OUTPUT is required when provisioning"
            )
        if "\n" in str(target_python) or "\r" in str(target_python):
            raise ValueError("target interpreter path contains a newline")
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"target_python={target_python}\n")
        environment_path = args.github_env or os.environ.get("GITHUB_ENV")
        if environment_path:
            with Path(environment_path).open("a", encoding="utf-8") as environment:
                environment.write(f"TARGET_PYTHON={target_python}\n")
        result["target_python"] = str(target_python)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
