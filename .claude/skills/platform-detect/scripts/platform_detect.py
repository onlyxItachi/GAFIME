#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess


RELEASE_STATUS = "see_docs_releases_status"
CURRENT_INSTALL_GUIDANCE = (
    "Consult docs/releases/STATUS.md, GitHub Releases, and PyPI for mutable "
    "publication state."
)
PRERELEASE_CORE_INSTALL = 'pip install --pre gafime "polars>=1.3,<2"'
PRERELEASE_CUDA_INSTALL = 'pip install --pre gafime gafime-cuda "polars>=1.3,<2"'
PRERELEASE_ROCM_INSTALL = 'pip install --pre gafime gafime-rocm "polars>=1.3,<2"'


def _release_install_fields(command: str) -> dict[str, str]:
    return {
        "release_status": RELEASE_STATUS,
        "current_install_guidance": CURRENT_INSTALL_GUIDANCE,
        "prerelease_install": command,
    }


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> list[str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return [line.strip() for line in output.splitlines() if line.strip()] or None


def _amd_rocm_hint() -> list[str] | None:
    env_hints = [
        name
        for name in ("ROCM_PATH", "HIP_PATH", "HIPSDK_PATH")
        if os.environ.get(name)
    ]
    if env_hints:
        return [f"{name}=configured" for name in env_hints]
    if shutil.which("rocm_agent_enumerator"):
        try:
            output = subprocess.check_output(
                ["rocm_agent_enumerator"], text=True, timeout=5
            )
            rows = [
                line.strip()
                for line in output.splitlines()
                if line.strip().startswith("gfx")
            ]
            return rows or ["rocm_agent_enumerator available"]
        except Exception:
            return ["rocm_agent_enumerator available"]
    if shutil.which("rocminfo"):
        return ["rocminfo available"]
    return None


def _payload_version_warnings(payloads: dict[str, str | None]) -> list[str]:
    core = payloads.get("gafime")
    if core is None:
        return []
    return [
        f"{name} {version} does not match gafime {core}; payloads require exact Core alignment."
        for name, version in sorted(payloads.items())
        if name != "gafime" and version is not None and version != core
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect the supported GAFIME platform")
    parser.add_argument(
        "--precision", default="mixed", choices=["fp32", "mixed", "fp64"]
    )
    args = parser.parse_args()

    system = platform.system()
    machine = platform.machine()
    system_l = system.lower()
    machine_l = machine.lower()
    nvidia = _nvidia_smi()
    amd_rocm = _amd_rocm_hint()
    payloads = {
        name: _dist_version(name) for name in ("gafime", "gafime-cuda", "gafime-rocm")
    }
    result: dict[str, object] = {
        "os": system,
        "machine": machine,
        "python": platform.python_version(),
        "requested_precision": args.precision,
        "nvidia": nvidia,
        "amd_rocm": amd_rocm,
        "payload_distributions": payloads,
        "recommended_backend": "core",
        **_release_install_fields(PRERELEASE_CORE_INSTALL),
        "capability_probe": None,
        "notes": [
            "backend='auto' ranks validated GPU payloads above Rust Core.",
            "Explicit cuda, rocm, and metal requests never fall back to another backend.",
            "Family generation and scoring placement are separate capability facts.",
            "GAFIME v1 uses dedicated wheels for CPython 3.10 through 3.14.",
        ],
    }
    result["notes"].extend(_payload_version_warnings(payloads))

    try:
        from gafime import backend_capabilities

        caps = backend_capabilities("auto", probe=True, precision=args.precision)
        result["capability_probe"] = caps.to_dict()
        if caps.selected_backend:
            result["recommended_backend"] = caps.selected_backend
    except Exception as exc:
        result["notes"].append(
            f"installed capability probe unavailable: {type(exc).__name__}: {exc}"
        )

    if system_l == "darwin" and machine_l in {"arm64", "aarch64"}:
        result["prerelease_install"] = PRERELEASE_CORE_INSTALL
        result["notes"].append(
            "Metal is bundled in the macOS arm64 Core wheel, supports fp32 only, "
            "and is selected only after a successful runtime probe."
        )
        if args.precision != "fp32":
            result["notes"].append(
                "The requested precision excludes Metal; auto may select Rust Core."
            )
    elif nvidia and machine_l in {"x86_64", "amd64", "x64"}:
        result["prerelease_install"] = PRERELEASE_CUDA_INSTALL
    elif amd_rocm and system_l == "linux" and machine_l in {"x86_64", "amd64", "x64"}:
        result["prerelease_install"] = PRERELEASE_ROCM_INSTALL
        result["notes"].append(
            "PyPI provides the buildable ROCm sdist; the matching "
            "GitHub Release is the prebuilt thin raw-Linux wheel channel. Both use the "
            "system ROCm runtime."
        )
    elif amd_rocm and system_l == "windows":
        result["notes"].append(
            "ROCm payload wheels are not distributed for Windows; use backend='core'."
        )

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
