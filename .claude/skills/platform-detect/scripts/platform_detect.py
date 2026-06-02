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
    result = {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "nvidia": _nvidia_smi(),
        "recommended_backend": "core",
        "notes": [
            "GAFIME v0.4.5 has no NumPy backend.",
            "Use backend='core' unless CUDA is available and report metrics are pearson/r2.",
            "Metal is disabled in v0.4.5.",
        ],
    }
    if result["nvidia"]:
        result["recommended_backend"] = "cuda"
        result["recommended_metric_names"] = ["pearson", "r2"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
