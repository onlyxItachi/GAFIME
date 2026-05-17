from __future__ import annotations

from importlib import import_module


def _load_rust_helpers():
    try:
        return import_module("gafime.gafime_cpu")
    except ImportError:
        return import_module("gafime_cpu")


_rust_helpers = _load_rust_helpers()

__all__ = [name for name in dir(_rust_helpers) if not name.startswith("_")]


def __getattr__(name: str):
    return getattr(_rust_helpers, name)


def __dir__():
    return sorted(__all__)

