from __future__ import annotations

import warnings as _warnings
from typing import List, Tuple

from ..config import EngineConfig
from ..native_data import NativeMatrix, NativeVector
from .base import Backend
from .core_backend import CoreBackend
from .policy import backend_priority

__all__ = ["Backend", "CoreBackend", "resolve_backend"]


def resolve_backend(
    config: EngineConfig,
    X: NativeMatrix,
    y: NativeVector,
) -> Tuple[Backend, List[str]]:
    """Resolve the compute backend for GAFIME analysis.
    
    Priority order is platform-aware:
    - macOS: Metal -> C++ Core
    - Linux/Windows x86_64: CUDA -> C++ Core
    - Explicit ROCm/HIP requests: ROCm/HIP only
    - Linux/Windows ARM64: C++ Core
    """
    warnings: List[str] = []
    requested = (config.backend or "auto").lower()
    allowed = {"auto", "cuda", "gpu", "rocm", "hip", "metal", "cpu", "core", "cpp"}
    if requested not in allowed:
        raise ValueError(
            f"Unknown backend '{requested}'. "
            "Allowed backends are auto, cuda/gpu, rocm/hip, metal, cpu/core/cpp."
        )

    if requested == "gpu":
        _warnings.warn(
            "backend='gpu' is deprecated because GPU backend selection is platform-specific. "
            "Use backend='auto', backend='cuda', or backend='metal'.",
            DeprecationWarning,
            stacklevel=2,
        )

    priority = backend_priority(requested)
    backend: Backend | None = None
    for candidate in priority:
        if candidate == "cuda":
            backend = _try_native_cuda(config, warnings, emit_warning=True)
        elif candidate == "rocm":
            backend = _try_native_rocm(config, warnings, emit_warning=True)
        elif candidate == "metal":
            backend = _try_native_metal(warnings, emit_warning=True)
        elif candidate == "core":
            emit_warning = requested not in ("auto", "cpu", "core", "cpp")
            backend = _try_core(warnings, emit_warning=emit_warning)
        else:
            raise RuntimeError(f"Internal backend policy produced unknown backend '{candidate}'.")
        if backend is not None:
            break

    if backend is None:
        if requested == "auto":
            detail = "; ".join(warnings) if warnings else "no native backend could be loaded"
            raise RuntimeError(f"No native GAFIME backend is available in auto mode for this platform: {detail}.")
        detail = "; ".join(warnings) if warnings else "no native backend could be loaded"
        raise RuntimeError(f"Requested backend '{requested}' is unavailable: {detail}.")

    ok, budget_warnings = backend.check_budget(X, y, config.budget)
    warnings.extend(budget_warnings)
    if not ok:
        raise RuntimeError("Selected native backend rejected the compute budget.")

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
            warnings.append("CUDA payload not installed or native CUDA library not found; install gafime[cuda].")
    except Exception as exc:
        if emit_warning:
            warnings.append(f"Native CUDA backend unavailable: {exc}")
    return None


def _try_native_rocm(
    config: EngineConfig, warnings: List[str], emit_warning: bool
) -> Backend | None:
    """Try to load native ROCm/HIP backend."""
    try:
        from .native_rocm_backend import NativeRocmBackend
        return NativeRocmBackend(device_id=config.device_id)
    except ImportError:
        if emit_warning:
            warnings.append("ROCm/HIP payload not installed or native HIP library not found; install gafime[rocm].")
    except Exception as exc:
        if emit_warning:
            warnings.append(f"Native ROCm/HIP backend unavailable: {exc}")
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
