#!/usr/bin/env python3
from __future__ import annotations

import importlib
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


def _command_available(name: str) -> bool:
    return shutil.which(name) is not None


def _has_nvidia_gpu() -> bool:
    if not _command_available("nvidia-smi"):
        return False
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


def _has_amd_gpu_hint() -> bool:
    # Keep this lightweight: do not import HIP or initialize a ROCm runtime here.
    if any(os.environ.get(name) for name in ("ROCM_PATH", "HIP_PATH", "HIPSDK_PATH")):
        return True
    for path in (r"C:\Program Files\AMD\ROCm", r"C:\Program Files\AMD\ROCm SDK"):
        if os.path.isdir(path):
            return True
    if _command_available("rocm_agent_enumerator"):
        try:
            result = subprocess.run(
                ["rocm_agent_enumerator"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            return any(line.strip().startswith("gfx") for line in result.stdout.splitlines())
        except Exception:
            return True
    return _command_available("rocminfo")


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn):
        try:
            detail = fn()
            checks.append((name, True, detail))
        except Exception as exc:
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    check("Python", lambda: platform.python_version())

    def import_gafime() -> str:
        import gafime

        return f"gafime {gafime.__version__}"

    check("GAFIME", import_gafime)

    def subfunctions() -> str:
        from gafime import subfunctions

        assert hasattr(subfunctions, "BatchScheduler")
        return f"subfunctions {getattr(subfunctions, '__version__', 'unknown')}"

    check("Rust subfunctions", subfunctions)

    def payloads() -> str:
        installed = {
            "gafime": _dist_version("gafime"),
            "gafime-cuda": _dist_version("gafime-cuda"),
            "gafime-rocm": _dist_version("gafime-rocm"),
        }
        notes = [f"{name}={version}" for name, version in installed.items() if version]
        if _has_nvidia_gpu() and not installed["gafime-cuda"]:
            notes.append('recommend: pip install "gafime[cuda]"')
        if _has_amd_gpu_hint() and not installed["gafime-rocm"]:
            notes.append('recommend: pip install "gafime[rocm]"')
        return ", ".join(notes) if notes else "no GAFIME distributions found"

    check("Vendor payload packages", payloads)

    def core_backend() -> str:
        from gafime import ComputeBudget, EngineConfig, GafimeEngine

        X = [[float(i), float(i % 3), float((i * 5) % 7)] for i in range(80)]
        y = [row[0] * row[1] for row in X]
        report = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
                permutation_tests=1,
                num_repeats=1,
            )
        ).analyze(X, y)
        return f"{report.backend.name}, {len(report.interactions)} interactions"

    check("C++ Core engine", core_backend)

    def auto_backend() -> str:
        from gafime import EngineConfig
        from gafime.backends import resolve_backend
        from gafime.utils.arrays import coerce_inputs

        X, y, _ = coerce_inputs([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1.0, 2.0, 3.0])
        backend, warnings = resolve_backend(EngineConfig(backend="auto", metric_names=("pearson", "r2")), X, y)
        info = backend.info()
        detail = f"{info.name} ({info.device})"
        if warnings:
            detail += f"; warnings={len(warnings)}"
        return detail

    check("Auto backend resolver", auto_backend)

    def polars() -> str:
        mod = importlib.import_module("polars")
        return getattr(mod, "__version__", "available")

    check("Polars", polars)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
