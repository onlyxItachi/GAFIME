#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path


RELEASE_STATUS = "see_docs_releases_status"
CURRENT_INSTALL_GUIDANCE = (
    "Consult docs/releases/STATUS.md, GitHub Releases, and PyPI for mutable "
    "publication state."
)
PRERELEASE_INSTALLS = {
    "core": 'pip install --pre gafime "polars>=1.3,<2"',
    "cuda": 'pip install --pre gafime gafime-cuda "polars>=1.3,<2"',
    "rocm": 'pip install --pre gafime gafime-rocm "polars>=1.3,<2"',
}


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
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


def _diagnostic_precision(backend: str, requested_precision: str) -> str:
    """Use fp32 to test Metal payload health without misreporting mixed/fp64."""

    return "fp32" if backend == "metal" else requested_precision


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose GAFIME backend payloads")
    parser.add_argument(
        "--precision", default="mixed", choices=["fp32", "mixed", "fp64"]
    )
    args = parser.parse_args()

    import gafime
    from gafime import backend_capabilities

    package_dir = Path(gafime.__file__).parent
    payloads = {
        name: _dist_version(name) for name in ("gafime", "gafime-cuda", "gafime-rocm")
    }
    result: dict[str, object] = {
        "version": gafime.__version__,
        "release_status": RELEASE_STATUS,
        "current_install_guidance": CURRENT_INSTALL_GUIDANCE,
        "prerelease_install": PRERELEASE_INSTALLS,
        "requested_precision": args.precision,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "package_dir": str(package_dir),
        "core_artifacts": sorted(path.name for path in package_dir.glob("*gafime*")),
        "payload_distributions": payloads,
        "version_alignment_warnings": _payload_version_warnings(payloads),
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
        probe_precision = _diagnostic_precision(backend, args.precision)
        try:
            caps = backend_capabilities(backend, probe=True, precision=probe_precision)
            record = caps.to_dict()
            record["diagnostic_probe_precision"] = probe_precision
            record["requested_precision"] = args.precision
            if backend == "metal" and args.precision != "fp32":
                record["requested_precision_supported"] = False
                record["requested_precision_note"] = (
                    "Metal supports fp32 only; the fp32 payload-health probe does not "
                    "make the requested profile eligible."
                )
            backends[backend] = record
        except Exception as exc:
            record = {
                "configured_backend": backend,
                "diagnostic_probe_precision": probe_precision,
                "requested_precision": args.precision,
                "selection_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if backend == "metal" and args.precision != "fp32":
                record["requested_precision_supported"] = False
                record["requested_precision_note"] = (
                    "Metal supports fp32 only; payload-health failure is separate "
                    "from the requested profile being unsupported."
                )
            backends[backend] = record

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
