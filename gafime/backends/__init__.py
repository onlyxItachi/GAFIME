from __future__ import annotations

from typing import List, Tuple

from ..config import EngineConfig
from ..native_data import NativeMatrix, NativeVector
from .base import Backend
from .core_backend import CoreBackend

__all__ = ["Backend", "CoreBackend", "resolve_backend"]


def resolve_backend(
    config: EngineConfig,
    X: NativeMatrix,
    y: NativeVector,
) -> Tuple[Backend, List[str]]:
    """Resolve the compute backend for GAFIME analysis.
    
    Priority order:
    1. Native CUDA backend (if available)
    2. Native Metal backend (Apple Silicon only)
    3. C++ core backend (gafime_core)
    """
    warnings: List[str] = []
    requested = (config.backend or "auto").lower()
    allowed = {"auto", "cuda", "gpu", "metal", "cpu", "core", "cpp"}
    if requested not in allowed:
        raise ValueError(
            f"Unknown backend '{requested}'. "
            "Allowed backends are auto, cuda/gpu, metal, cpu/core/cpp."
        )

    backend: Backend | None = None

    gpu_report_metrics = {"pearson", "r2"}
    unsupported_gpu_metrics = [
        name for name in config.metric_names if name not in gpu_report_metrics
    ]

    # Try native CUDA backend first. CUDA v0.4.5 intentionally fails fast for
    # report metrics that are not implemented in the arity-template batch path.
    if requested in ("auto", "cuda", "gpu") and not unsupported_gpu_metrics:
        backend = _try_native_cuda(config, warnings, emit_warning=(requested != "auto"))
    elif requested in ("cuda", "gpu") and unsupported_gpu_metrics:
        raise ValueError(
            "CUDA backend supports report metrics ('pearson', 'r2') in v0.4.5; "
            f"unsupported metrics requested: {tuple(unsupported_gpu_metrics)}."
        )
    elif requested == "auto" and unsupported_gpu_metrics:
        warnings.append(
            "Skipping CUDA auto-selection because requested report metrics require the C++ core backend."
        )

    # Try native Metal backend (Apple Silicon)
    if backend is None and requested in ("auto", "metal", "gpu"):
        backend = _try_native_metal(warnings, emit_warning=(requested not in ("auto",)))

    # Try C++ core backend
    if backend is None and requested in ("auto", "cpu", "core", "cpp"):
        emit_warning = requested not in ("auto", "cpu")
        backend = _try_core(warnings, emit_warning=emit_warning)

    if backend is None:
        if requested == "auto":
            detail = "; ".join(warnings) if warnings else "no native backend could be loaded"
            raise RuntimeError(f"No native GAFIME backend is available in auto mode: {detail}.")
        raise RuntimeError(f"Requested backend '{requested}' is unavailable.")

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
