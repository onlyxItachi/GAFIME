#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess


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
        name for name in ("ROCM_PATH", "HIP_PATH", "HIPSDK_PATH") if os.environ.get(name)
    ]
    if env_hints:
        return [f"{name}=configured" for name in env_hints]
    if shutil.which("rocm_agent_enumerator"):
        try:
            output = subprocess.check_output(
                ["rocm_agent_enumerator"], text=True, timeout=5
            )
            rows = [line.strip() for line in output.splitlines() if line.strip().startswith("gfx")]
            return rows or ["rocm_agent_enumerator available"]
        except Exception:
            return ["rocm_agent_enumerator available"]
    if shutil.which("rocminfo"):
        return ["rocminfo available"]
    return None


def main() -> int:
    system = platform.system()
    machine = platform.machine()
    system_l = system.lower()
    machine_l = machine.lower()
    nvidia = _nvidia_smi()
    amd_rocm = _amd_rocm_hint()
    payloads = {
        name: _dist_version(name)
        for name in ("gafime", "gafime-cuda", "gafime-rocm")
    }
    result: dict[str, object] = {
        "os": system,
        "machine": machine,
        "python": platform.python_version(),
        "nvidia": nvidia,
        "amd_rocm": amd_rocm,
        "payload_distributions": payloads,
        "recommended_backend": "core",
        "recommended_install": "pip install gafime",
        "capability_probe": None,
        "notes": [
            "backend='auto' ranks validated GPU payloads above Rust Core.",
            "Explicit cuda, rocm, and metal requests never fall back to another backend.",
            "Family generation and scoring placement are separate capability facts.",
        ],
    }

    try:
        from gafime import backend_capabilities

        caps = backend_capabilities("auto", probe=True)
        result["capability_probe"] = caps.to_dict()
        if caps.selected_backend:
            result["recommended_backend"] = caps.selected_backend
    except Exception as exc:
        result["notes"].append(f"installed capability probe unavailable: {type(exc).__name__}: {exc}")

    if system_l == "darwin" and machine_l in {"arm64", "aarch64"}:
        result["recommended_install"] = "pip install gafime"
        result["notes"].append("Metal is bundled in the macOS arm64 Core wheel and is selected only after a successful runtime probe.")
    elif nvidia and machine_l in {"x86_64", "amd64", "x64"}:
        result["recommended_install"] = "pip install gafime gafime-cuda"
    elif amd_rocm and system_l == "linux" and machine_l in {"x86_64", "amd64", "x64"}:
        result["recommended_install"] = "pip install gafime gafime-rocm"
    elif amd_rocm and system_l == "windows":
        result["notes"].append("ROCm payload wheels are not distributed for Windows; use backend='core'.")

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
