#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import subprocess


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
        "recommended_backend": "core",
        "notes": [
            "GAFIME v0.4.5 has no NumPy backend.",
            "backend='auto' is platform-aware: macOS uses metal->core, x86 Linux/Windows uses cuda->core, ARM Linux/Windows uses core.",
            "backend='gpu' is deprecated; use auto, cuda, metal, or core.",
        ],
    }
    if system_l == "darwin" and machine_l in {"arm64", "aarch64"}:
        result["recommended_backend"] = "metal"
        result["recommended_metric_names"] = ["pearson", "r2"]
    elif result["nvidia"] and machine_l in {"x86_64", "amd64", "x64"}:
        result["recommended_backend"] = "cuda"
        result["recommended_metric_names"] = ["pearson", "r2"]
    else:
        result["recommended_metric_names"] = ["pearson", "spearman", "mutual_info", "r2"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
