"""Explicit, lazy access to selected native helper and boundary exports.

``DataQualityAnalyzer`` is deprecated on construction; its existing behavior
is retained and it never cleans or removes data without the caller requesting
it. ``OTSEncoder`` remains available. The listed report/artifact types and
low-level functions preserve the advanced native boundary; their inputs are
native protocol-shaped, not an alternative Python-owned production data plane.
Ordinary analysis should use :class:`gafime.GafimeEngine` instead.

Importing this module, inspecting ``__all__``, or calling ``dir`` does not
load the extension. Only the names listed below are forwarded; other native
implementation details are not implicitly part of this namespace.
"""

from __future__ import annotations

# These exports are resolved lazily by the module-level __getattr__.
# ruff: noqa: F822
from importlib import import_module
from types import ModuleType


__all__ = [
    "BOUNDARY_NAME",
    "CompiledContinuousArtifact",
    "ContinuousRecord",
    "ContinuousReport",
    "DataQualityAnalyzer",
    "OTSEncoder",
    "analyze_continuous",
    "analyze_continuous_arrow",
    "analyze_continuous_buffers",
    "analyze_continuous_cpu",
    "analyze_continuous_rows",
    "analyze_decision_path",
    "analyze_time_series",
    "compile_continuous",
    "compile_continuous_buffers",
    "compile_continuous_rows",
    "compile_decision_path",
    "compile_time_series",
    "native_version",
    "runtime_capabilities",
]

_rust_helpers: ModuleType | None = None


def _load_rust_helpers() -> ModuleType:
    global _rust_helpers
    if _rust_helpers is None:
        _rust_helpers = import_module("gafime.gafime_py")
    return _rust_helpers


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_load_rust_helpers(), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
