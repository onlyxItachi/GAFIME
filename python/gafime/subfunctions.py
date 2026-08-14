"""Lazy compatibility proxy for published native helper surfaces.

Ordinary analysis should use :class:`gafime.GafimeEngine`.  This module keeps
v0.4.7 helper classes and advanced native-boundary functions importable without
loading the extension merely by importing :mod:`gafime`.  Its dynamic
``__all__`` reflects public names exported by the installed compatible native
module, so availability may differ for an older third-party boundary.  The
low-level ``analyze_*``/``compile_*`` functions accept native protocol-shaped
inputs and are not an alternative Python-owned production data plane.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_HELPER_MODULE_NAMES = (
    "gafime.gafime_py",
    "gafime.gafime_cpu",
    "gafime_cpu",
)
_rust_helpers: ModuleType | None = None
_rust_helper_modules: tuple[ModuleType, ...] | None = None


def _load_rust_helpers() -> tuple[ModuleType, ...]:
    global _rust_helper_modules, _rust_helpers
    if _rust_helper_modules is not None:
        return _rust_helper_modules

    import_errors: list[ImportError] = []
    modules: list[ModuleType] = []
    for module_name in _HELPER_MODULE_NAMES:
        try:
            modules.append(import_module(module_name))
        except ImportError as exc:
            import_errors.append(exc)

    if not modules:
        raise ImportError(
            "Unable to load GAFIME's native helper module; tried "
            + ", ".join(_HELPER_MODULE_NAMES)
        ) from import_errors[0]

    _rust_helper_modules = tuple(modules)
    _rust_helpers = modules[0]
    return _rust_helper_modules


def _public_names() -> list[str]:
    return sorted(
        {
            name
            for module in _load_rust_helpers()
            for name in dir(module)
            if not name.startswith("_")
        }
    )


def __getattr__(name: str):
    if name == "__all__":
        return _public_names()
    for module in _load_rust_helpers():
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_public_names())
