from __future__ import annotations

from pathlib import Path

__version__ = "0.4.7"


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def library_candidates() -> list[Path]:
    base = package_dir()
    return [
        base / "gafime_cuda.dll",
        base / "libgafime_cuda.so",
        base / "gafime_cuda.so",
        base / "gafime_cuda.pyd",
    ]
