from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import EngineConfig
from .base import Backend
from .cupy_backend import CupyBackend
from .torch_backend import TorchBackend

__all__ = ["Backend", "resolve_backend"]


def resolve_backend(
    config: EngineConfig,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Backend, List[str]]:
    warnings: List[str] = []
    requested = (config.backend or "auto").lower()
    backend: Backend | None = None

    if requested in ("auto", "cuda", "cupy"):
        emit_warning = requested != "auto"
        backend = _try_cupy(config, warnings, emit_warning=emit_warning)
        if backend is None and requested in ("auto", "cuda"):
            backend = _try_torch("cuda", config, warnings, emit_warning=emit_warning)

    if backend is None and requested in ("rocm",):
        backend = _try_torch("rocm", config, warnings, emit_warning=True)

    if backend is None and requested in ("mlx",):
        warnings.append("MLX backend requested but not available on this platform.")

    if backend is None and requested == "auto":
        warnings.append("No GPU backend available; using CPU.")

    if backend is None and requested in ("cpu", "numpy", "auto", "mlx", "rocm", "cuda", "cupy"):
        backend = Backend()

    if backend is None:
        raise ValueError(f"Unknown backend selection: {config.backend}.")

    ok, budget_warnings = backend.check_budget(X, y, config.budget)
    warnings.extend(budget_warnings)
    if not ok:
        backend = Backend()

    return backend, warnings


def _try_cupy(config: EngineConfig, warnings: List[str], emit_warning: bool) -> Backend | None:
    try:
        return CupyBackend(device_id=config.device_id)
    except ModuleNotFoundError:
        if emit_warning:
            warnings.append("CuPy is not installed; CUDA backend unavailable.")
    except Exception as exc:  # pragma: no cover - device availability varies
        if emit_warning:
            warnings.append(f"CUDA backend unavailable: {exc}")
    return None


def _try_torch(
    mode: str,
    config: EngineConfig,
    warnings: List[str],
    emit_warning: bool,
) -> Backend | None:
    try:
        return TorchBackend(mode=mode, device_id=config.device_id)
    except ModuleNotFoundError:
        if emit_warning:
            warnings.append("Torch is not installed; Torch backend unavailable.")
    except Exception as exc:  # pragma: no cover - device availability varies
        if emit_warning:
            warnings.append(f"Torch backend unavailable: {exc}")
    return None
