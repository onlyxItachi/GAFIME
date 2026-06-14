#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.metadata as metadata
import os
import platform
import shutil
import subprocess


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    rows = [line.strip() for line in out.splitlines() if line.strip()]
    return rows


def _amd_rocm_hint():
    env_hints = [name for name in ("ROCM_PATH", "HIP_PATH", "HIPSDK_PATH") if os.environ.get(name)]
    if env_hints:
        return [f"{name}={os.environ[name]}" for name in env_hints]
    for path in (r"C:\Program Files\AMD\ROCm", r"C:\Program Files\AMD\ROCm SDK"):
        if os.path.isdir(path):
            return [path]
    if shutil.which("rocm_agent_enumerator"):
        try:
            out = subprocess.check_output(["rocm_agent_enumerator"], text=True, timeout=5)
            rows = [line.strip() for line in out.splitlines() if line.strip().startswith("gfx")]
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
    result = {
        "os": system,
        "machine": machine,
        "python": platform.python_version(),
        "nvidia": _nvidia_smi(),
        "amd_rocm": _amd_rocm_hint(),
        "payload_distributions": {
            "gafime": _dist_version("gafime"),
            "gafime-cuda": _dist_version("gafime-cuda"),
            "gafime-rocm": _dist_version("gafime-rocm"),
        },
        "recommended_backend": "core",
        "recommended_install": "pip install gafime",
        "notes": [
            "GAFIME v0.4.5 has no NumPy backend.",
            "GPU runtime payloads are explicit: gafime[cuda] for NVIDIA and gafime[rocm] for AMD ROCm/HIP.",
            "backend='auto' selects an installed vendor payload before core.",
            "backend='gpu' is deprecated; use auto, cuda, metal, or core.",
        ],
    }
    if system_l == "darwin" and machine_l in {"arm64", "aarch64"}:
        result["recommended_backend"] = "metal"
        result["recommended_metric_names"] = ["pearson", "r2"]
    elif result["nvidia"] and result["payload_distributions"]["gafime-cuda"] and machine_l in {"x86_64", "amd64", "x64"}:
        result["recommended_backend"] = "cuda"
        result["recommended_install"] = 'pip install "gafime[cuda]"'
        result["recommended_metric_names"] = ["pearson", "r2"]
    elif result["amd_rocm"] and result["payload_distributions"]["gafime-rocm"] and system_l in {"linux", "windows"} and machine_l in {"x86_64", "amd64", "x64"}:
        result["recommended_backend"] = "rocm"
        result["recommended_install"] = 'pip install "gafime[rocm]"'
        result["recommended_metric_names"] = ["pearson", "r2"]
    else:
        result["recommended_metric_names"] = ["pearson", "spearman", "mutual_info", "r2"]
        if result["nvidia"] and machine_l in {"x86_64", "amd64", "x64"}:
            result["recommended_install"] = 'pip install "gafime[cuda]"'
        elif result["amd_rocm"] and system_l in {"linux", "windows"} and machine_l in {"x86_64", "amd64", "x64"}:
            result["recommended_install"] = 'pip install "gafime[rocm]"'
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
