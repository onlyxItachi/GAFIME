"""GPU backend (CuPy)."""

from __future__ import annotations

from typing import List, Sequence

from gafime.backend import metrics
from gafime.backend.metrics import MetricResult
from gafime.report.schema import FeatureScore

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import
    cp = None
    CUPY_AVAILABLE = False


def is_available() -> bool:
    if not CUPY_AVAILABLE:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _to_gpu(array):
    return cp.asarray(array, dtype=cp.float32)


def pearson_corr(x, y) -> MetricResult:
    if not is_available():
        return MetricResult(value=float("nan"), is_valid=False, reason="gpu_unavailable")
    x_gpu = _to_gpu(x)
    y_gpu = _to_gpu(y)
    if x_gpu.size != y_gpu.size:
        return MetricResult(value=float("nan"), is_valid=False, reason="length_mismatch")
    if x_gpu.size < 2:
        return MetricResult(value=float("nan"), is_valid=False, reason="insufficient_samples")
    if cp.allclose(x_gpu, x_gpu[0]) or cp.allclose(y_gpu, y_gpu[0]):
        return MetricResult(value=0.0, is_valid=False, reason="constant_input")
    corr = cp.corrcoef(x_gpu, y_gpu)[0, 1]
    corr_val = float(cp.asnumpy(corr))
    if corr_val != corr_val:
        return MetricResult(value=float("nan"), is_valid=False, reason="nan_result")
    return MetricResult(value=corr_val, is_valid=True)


def mutual_information(x, y, bins: int = 10) -> MetricResult:
    if not is_available():
        return MetricResult(value=float("nan"), is_valid=False, reason="gpu_unavailable")
    x_gpu = _to_gpu(x)
    y_gpu = _to_gpu(y)
    if x_gpu.size != y_gpu.size:
        return MetricResult(value=float("nan"), is_valid=False, reason="length_mismatch")
    if x_gpu.size < 2:
        return MetricResult(value=float("nan"), is_valid=False, reason="insufficient_samples")
    if cp.allclose(x_gpu, x_gpu[0]) or cp.allclose(y_gpu, y_gpu[0]):
        return MetricResult(value=0.0, is_valid=False, reason="constant_input")

    bins = max(2, int(bins))
    x_edges = cp.linspace(cp.min(x_gpu), cp.max(x_gpu), bins + 1)
    y_edges = cp.linspace(cp.min(y_gpu), cp.max(y_gpu), bins + 1)
    hist, _, _ = cp.histogram2d(x_gpu, y_gpu, bins=(x_edges, y_edges))
    total = cp.sum(hist)
    if total <= 0:
        return MetricResult(value=0.0, is_valid=False, reason="empty_histogram")

    joint = hist / total
    px = cp.sum(joint, axis=1, keepdims=True)
    py = cp.sum(joint, axis=0, keepdims=True)
    ratio = joint / (px @ py)
    log_term = cp.where(joint > 0, cp.log(ratio), 0.0)
    mi = float(cp.asnumpy(cp.sum(joint * log_term)))
    if mi != mi:
        return MetricResult(value=float("nan"), is_valid=False, reason="nan_result")
    return MetricResult(value=mi, is_valid=True)


def score_unary(
    features,
    target,
    names: Sequence[str],
    bins: int,
) -> List[FeatureScore]:
    scores: List[FeatureScore] = []
    for idx, name in enumerate(names):
        feature = features[:, idx]
        pearson = pearson_corr(feature, target)
        mi = mutual_information(feature, target, bins=bins)
        pearson_value, mi_value = metrics.summarize_metrics(pearson, mi)
        scores.append(
            FeatureScore(
                name=name,
                pearson=pearson_value,
                mutual_information=mi_value,
            )
        )
    return scores
