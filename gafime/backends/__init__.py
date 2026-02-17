from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import EngineConfig
from .base import Backend
from .core_backend import CoreBackend

__all__ = ["Backend", "CoreBackend", "resolve_backend"]


def resolve_backend(
    config: EngineConfig,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Backend, List[str]]:
    """Resolve the compute backend for GAFIME analysis.
    
    Priority order:
    1. Native CUDA backend (if available)
    2. Native Metal backend (Apple Silicon only)
    3. C++ core backend (gafime_core)
    4. Pure NumPy fallback
    """
    warnings: List[str] = []
    requested = (config.backend or "auto").lower()
    backend: Backend | None = None

    # Try native CUDA backend first
    if requested in ("auto", "cuda", "gpu"):
        backend = _try_native_cuda(config, warnings, emit_warning=(requested != "auto"))

    # Try native Metal backend (Apple Silicon)
    if backend is None and requested in ("auto", "metal", "gpu"):
        backend = _try_native_metal(warnings, emit_warning=(requested not in ("auto",)))

    # Try C++ core backend
    if backend is None and requested in ("auto", "cpu", "numpy", "core", "cpp"):
        emit_warning = requested not in ("auto", "cpu", "numpy")
        backend = _try_core(warnings, emit_warning=emit_warning)

    if backend is None and requested == "auto":
        warnings.append("No accelerated backend available; using NumPy CPU fallback.")

    # Final fallback to pure NumPy
    if backend is None:
        backend = Backend()

    ok, budget_warnings = backend.check_budget(X, y, config.budget)
    warnings.extend(budget_warnings)
    if not ok:
        backend = Backend()

    return backend, warnings


def _try_native_cuda(
    config: EngineConfig, warnings: List[str], emit_warning: bool
) -> Backend | None:
    """Try to load native CUDA backend."""
    try:
        from .native_cuda_backend import NativeCudaBackend
        return NativeCudaBackend(device_id=config.device_id)
    except ImportError:
        if emit_warning:
            warnings.append("Native CUDA backend not compiled; GPU unavailable.")
    except Exception as exc:
        if emit_warning:
            warnings.append(f"Native CUDA backend unavailable: {exc}")
    return None


def _try_native_metal(warnings: List[str], emit_warning: bool) -> Backend | None:
    """Try to load native Metal backend (Apple Silicon only)."""
    try:
        from .native_metal_backend import NativeMetalBackend
        return NativeMetalBackend()
    except ImportError:
        if emit_warning:
            warnings.append("Native Metal backend not compiled; Metal unavailable.")
    except Exception as exc:
        if emit_warning:
            warnings.append(f"Native Metal backend unavailable: {exc}")
    return None


def _try_core(warnings: List[str], emit_warning: bool) -> Backend | None:
    """Try to load C++ core backend."""
    try:
        return CoreBackend()
    except ModuleNotFoundError:
        if emit_warning:
            warnings.append("gafime_core not installed; core backend unavailable.")
    except Exception as exc:
        if emit_warning:
            warnings.append(f"Core backend unavailable: {exc}")
    return None
