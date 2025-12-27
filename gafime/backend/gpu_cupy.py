"""CuPy backend for GPU acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import cupy as cp
    from cupy.cuda.runtime import CUDARuntimeError
except Exception:  # pragma: no cover - optional dependency
    cp = None
    CUDARuntimeError = Exception

from gafime.backend.metrics import MetricResult


@dataclass(frozen=True)
class GPUStatus:
    available: bool
    reason: str | None = None


def gpu_available() -> GPUStatus:
    if cp is None:
        return GPUStatus(available=False, reason="CuPy not installed")
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        return GPUStatus(available=True)
    except CUDARuntimeError as exc:
        return GPUStatus(available=False, reason=str(exc))


def score_features_gpu(x: cp.ndarray, y: cp.ndarray, bins: int = 16) -> list[MetricResult]:
    results: list[MetricResult] = []
    for i in range(x.shape[1]):
        feature = x[:, i]
        pearson = _pearson_corr_gpu(feature, y)
        mi = _mutual_information_binned_gpu(feature, y, bins=bins)
        results.append(MetricResult(pearson=float(pearson), mutual_information=float(mi)))
    return results


def _pearson_corr_gpu(x: cp.ndarray, y: cp.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    std_x = float(cp.std(x))
    std_y = float(cp.std(y))
    if std_x <= 0.0 or std_y <= 0.0:
        return 0.0
    cov = float(cp.mean((x - cp.mean(x)) * (y - cp.mean(y))))
    return cov / (std_x * std_y)


def _mutual_information_binned_gpu(x: cp.ndarray, y: cp.ndarray, bins: int = 16) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if bins <= 1:
        raise ValueError("bins must be > 1")
    hist_2d, _, _ = cp.histogram2d(x, y, bins=bins)
    total = float(cp.sum(hist_2d))
    if total == 0.0:
        return 0.0
    pxy = hist_2d / total
    px = cp.sum(pxy, axis=1)
    py = cp.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nonzero = pxy > 0
    mi = cp.sum(pxy[nonzero] * cp.log(pxy[nonzero] / px_py[nonzero]))
    return float(mi)
