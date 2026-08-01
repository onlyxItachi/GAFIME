#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    import gafime
    from gafime import backend_capabilities

    package_dir = Path(gafime.__file__).parent
    result: dict[str, object] = {
        "version": gafime.__version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "package_dir": str(package_dir),
        "core_artifacts": sorted(path.name for path in package_dir.glob("*gafime*")),
        "payload_distributions": {
            name: _dist_version(name)
            for name in ("gafime", "gafime-cuda", "gafime-rocm")
        },
        "environment_overrides_present": {
            name: bool(os.environ.get(name))
            for name in (
                "GAFIME_CUDA_V1_LIB",
                "GAFIME_ROCM_V1_LIB",
                "GAFIME_METAL_V1_LIB",
                "GAFIME_METAL_V1_METALLIB",
            )
        },
        "backends": {},
    }

    backends = result["backends"]
    assert isinstance(backends, dict)
    for backend in ("core", "cuda", "rocm", "metal", "auto"):
        try:
            caps = backend_capabilities(backend, probe=True)
            backends[backend] = caps.to_dict()
        except Exception as exc:
            backends[backend] = {
                "configured_backend": backend,
                "selection_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
