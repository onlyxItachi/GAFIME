#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.metadata as metadata
from pathlib import Path


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    import gafime
    from gafime import EngineConfig
    from gafime.backends import resolve_backend
    from gafime.utils.arrays import coerce_inputs

    package_dir = Path(gafime.__file__).parent
    X, y, _ = coerce_inputs([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1.0, 2.0, 3.0])
    result = {
        "version": gafime.__version__,
        "package_dir": str(package_dir),
        "artifacts": sorted(path.name for path in package_dir.glob("*gafime*")),
        "payload_distributions": {
            "gafime": _dist_version("gafime"),
            "gafime-cuda": _dist_version("gafime-cuda"),
            "gafime-rocm": _dist_version("gafime-rocm"),
        },
        "backends": {},
    }

    for backend in ("core", "cuda", "rocm", "metal", "auto"):
        try:
            resolved, warnings = resolve_backend(
                EngineConfig(backend=backend, metric_names=("pearson", "r2")),
                X,
                y,
            )
            info = resolved.info()
            result["backends"][backend] = {
                "ok": True,
                "name": info.name,
                "device": info.device,
                "warnings": warnings,
            }
        except Exception as exc:
            result["backends"][backend] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
