from __future__ import annotations

import numpy as np

ADAPTIVE_MI_BIN_LEVELS = (2, 4, 8, 16, 32, 64, 96)


def _safe_pearson(x, y, xp=np) -> float:
    x_centered = x - xp.mean(x)
    y_centered = y - xp.mean(y)
    denom = xp.sqrt(xp.sum(x_centered ** 2) * xp.sum(y_centered ** 2))
    if float(denom) == 0.0:
        return 0.0
    return float(xp.sum(x_centered * y_centered) / denom)


def _rankdata(values, xp=np):
    if values.size == 0:
        return values
    sorter = xp.argsort(values)
    sorted_values = values[sorter]
    ranks = xp.empty_like(sorted_values, dtype=float)

    diff = sorted_values[1:] != sorted_values[:-1]
    change_idx = xp.flatnonzero(diff) + 1
    starts = xp.concatenate([xp.asarray([0]), change_idx])
    ends = xp.concatenate([starts[1:], xp.asarray([values.size])])

    for start, end in zip(starts.tolist(), ends.tolist()):
        avg_rank = 0.5 * (start + end - 1)
        ranks[start:end] = avg_rank

    inv = xp.empty_like(sorter)
    inv[sorter] = xp.arange(values.size)
    return ranks[inv]


def pearson_corr(x, y, xp=np) -> float:
    return _safe_pearson(x, y, xp=xp)


def spearman_corr(x, y, xp=np) -> float:
    x_rank = _rankdata(x, xp=xp)
    y_rank = _rankdata(y, xp=xp)
    return _safe_pearson(x_rank, y_rank, xp=xp)


def mutual_info(x, y, bins: int = 96, xp=np) -> float:
    if bins < 2:
        return 0.0
    if xp is not np:
        x_np = np.asarray(xp.asnumpy(x) if hasattr(xp, "asnumpy") else x, dtype=np.float64)
        y_np = np.asarray(xp.asnumpy(y) if hasattr(xp, "asnumpy") else y, dtype=np.float64)
        return dense_mutual_info(x_np, y_np, max_bins=bins)
    return dense_mutual_info(x, y, max_bins=bins)


def dense_mutual_info(
    x,
    y,
    *,
    max_bins: int = 96,
    min_samples_per_joint_bin: int = 8,
) -> float:
    """Adaptive dense MI for report metrics.

    ``max_bins`` is a cap, not a fixed histogram shape. The actual bin count is
    selected from powers-of-two-style levels so small datasets avoid sparse
    noisy histograms while larger datasets can use up to 96 bins.
    """
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.shape[0] == 0 or y_arr.shape[0] != x_arr.shape[0]:
        return 0.0

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.all(finite):
        x_arr = x_arr[finite]
        y_arr = y_arr[finite]
    n = int(x_arr.shape[0])
    if n <= 1 or np.min(x_arr) == np.max(x_arr) or np.min(y_arr) == np.max(y_arr):
        return 0.0

    actual_bins = select_adaptive_mi_bins(
        n,
        max_bins=max_bins,
        samples_per_bin=min_samples_per_joint_bin,
        dimensions=2,
    )
    if actual_bins < 2:
        return 0.0

    x_bins, x_count = adaptive_bin_indices(x_arr, actual_bins, exact_low_cardinality=True)
    y_bins, y_count = adaptive_bin_indices(y_arr, actual_bins, exact_low_cardinality=True)
    if x_count < 2 or y_count < 2:
        return 0.0

    joint_size = x_count * y_count
    joint = np.bincount(
        x_bins * y_count + y_bins,
        minlength=joint_size,
    ).astype(np.float64).reshape(x_count, y_count)
    return _corrected_mi_from_joint(joint)


def soft_binary_mutual_info(
    mask,
    y,
    *,
    max_bins: int = 96,
    min_samples_per_target_bin: int = 25,
    min_effective_support: float | None = None,
) -> float:
    """MI between a soft region mask and the target.

    The mask is treated as weighted membership in two states: inside and
    outside. This matches threshold/interval/rectangle semantics and avoids the
    sparse ``B x B`` histogram that made v0.4.0 discrete rankings noisy.
    """
    w = np.clip(np.asarray(mask, dtype=np.float64).reshape(-1), 0.0, 1.0)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if w.shape[0] == 0 or y_arr.shape[0] != w.shape[0]:
        return 0.0

    finite = np.isfinite(w) & np.isfinite(y_arr)
    if not np.all(finite):
        w = w[finite]
        y_arr = y_arr[finite]
    n = int(w.shape[0])
    if n <= 1 or np.min(y_arr) == np.max(y_arr):
        return 0.0

    support_floor = (
        adaptive_min_effective_support(n)
        if min_effective_support is None
        else float(min_effective_support)
    )
    support = soft_mask_support(w)
    if (
        support["effective_in"] < support_floor
        or support["effective_out"] < support_floor
    ):
        return 0.0

    target_bins = select_adaptive_mi_bins(
        n,
        max_bins=max_bins,
        samples_per_bin=min_samples_per_target_bin,
        dimensions=1,
    )
    y_bins, y_count = adaptive_bin_indices(y_arr, target_bins, exact_low_cardinality=True)
    if y_count < 2:
        return 0.0

    hist_in = np.bincount(y_bins, weights=w, minlength=y_count).astype(np.float64)
    hist_out = np.bincount(y_bins, weights=1.0 - w, minlength=y_count).astype(np.float64)
    joint = np.vstack([hist_in, hist_out])
    return _corrected_mi_from_joint(joint)


def soft_mask_support(mask) -> dict[str, float]:
    w = np.clip(np.asarray(mask, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if w.shape[0] == 0:
        return {
            "sum_in": 0.0,
            "sum_out": 0.0,
            "effective_in": 0.0,
            "effective_out": 0.0,
        }
    sum_in = float(np.sum(w))
    sum_in_sq = float(np.sum(w * w))
    out = 1.0 - w
    sum_out = float(np.sum(out))
    sum_out_sq = float(np.sum(out * out))
    return {
        "sum_in": sum_in,
        "sum_out": sum_out,
        "effective_in": (sum_in * sum_in / sum_in_sq) if sum_in_sq > 1e-12 else 0.0,
        "effective_out": (sum_out * sum_out / sum_out_sq) if sum_out_sq > 1e-12 else 0.0,
    }


def adaptive_min_effective_support(n_samples: int) -> float:
    """Support floor for soft split scores.

    A fixed floor of 8 protects medium/large datasets, but it over-prunes true
    narrow regions in small datasets. Use a capped sample-size-aware floor so
    small-n region signals are not zeroed before ranking.
    """
    n_samples = int(max(0, n_samples))
    if n_samples <= 0:
        return 8.0
    return float(min(8.0, max(3.0, 0.02 * n_samples)))


def select_adaptive_mi_bins(
    n_samples: int,
    *,
    max_bins: int = 96,
    samples_per_bin: int = 8,
    dimensions: int = 2,
) -> int:
    max_bins = int(max(2, min(int(max_bins), ADAPTIVE_MI_BIN_LEVELS[-1])))
    n_samples = int(max(0, n_samples))
    samples_per_bin = int(max(1, samples_per_bin))
    dimensions = int(max(1, dimensions))
    best = 2
    for level in ADAPTIVE_MI_BIN_LEVELS:
        if level > max_bins:
            break
        required = samples_per_bin * (level ** dimensions)
        if n_samples >= required:
            best = level
    return best


def mi_bin_template_capacity(bin_count: int) -> int:
    """Return the compile-time MI histogram capacity for a runtime bin count."""
    bin_count = int(max(1, bin_count))
    for level in ADAPTIVE_MI_BIN_LEVELS:
        if bin_count <= level:
            return int(level)
    return int(ADAPTIVE_MI_BIN_LEVELS[-1])


def adaptive_bin_indices(
    values,
    max_bins: int,
    *,
    exact_low_cardinality: bool = True,
) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.shape[0])
    max_bins = int(max(2, min(int(max_bins), ADAPTIVE_MI_BIN_LEVELS[-1])))
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0

    unique = np.unique(arr)
    if unique.size <= 1:
        return np.zeros(n, dtype=np.int32), 1
    if exact_low_cardinality and unique.size <= max_bins:
        return np.searchsorted(unique, arr).astype(np.int32), int(unique.size)

    bins = int(min(max_bins, n))
    order = np.argsort(arr, kind="mergesort")
    out = np.empty(n, dtype=np.int32)
    out[order] = np.minimum((np.arange(n, dtype=np.int64) * bins) // n, bins - 1)
    return out, bins


def _corrected_mi_from_joint(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=np.float64)
    total = float(np.sum(joint))
    if total <= 0.0:
        return 0.0

    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)
    nonzero_rows = int(np.count_nonzero(px > 0.0))
    nonzero_cols = int(np.count_nonzero(py > 0.0))
    if nonzero_rows < 2 or nonzero_cols < 2:
        return 0.0

    pxy = joint / total
    px_prob = px / total
    py_prob = py / total
    expected = px_prob[:, None] * py_prob[None, :]
    valid = (pxy > 0.0) & (expected > 0.0)
    if not np.any(valid):
        return 0.0

    mi = float(np.sum(pxy[valid] * np.log(pxy[valid] / expected[valid])))
    bias = ((nonzero_rows - 1) * (nonzero_cols - 1)) / (2.0 * total)
    return float(max(mi - bias, 0.0))


def linear_r2(x, y, xp=np) -> float:
    corr = _safe_pearson(x, y, xp=xp)
    return float(corr * corr)
