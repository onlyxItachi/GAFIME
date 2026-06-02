from __future__ import annotations

import math
from collections.abc import Sequence
from typing import List, Tuple

ADAPTIVE_MI_BIN_LEVELS = (2, 4, 8, 16, 32, 64, 96)


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    return _safe_pearson(_to_floats(x), _to_floats(y))


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    return _safe_pearson(_rankdata(_to_floats(x)), _rankdata(_to_floats(y)))


def mutual_info(x: Sequence[float], y: Sequence[float], bins: int = 96) -> float:
    return dense_mutual_info(x, y, max_bins=bins)


def dense_mutual_info(
    x: Sequence[float],
    y: Sequence[float],
    *,
    max_bins: int = 96,
    min_samples_per_joint_bin: int = 8,
) -> float:
    x_arr, y_arr = _finite_pair_values(x, y)
    n = len(x_arr)
    if n <= 1 or _constant(x_arr) or _constant(y_arr):
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
    joint = [[0.0 for _ in range(y_count)] for _ in range(x_count)]
    for xb, yb in zip(x_bins, y_bins):
        joint[xb][yb] += 1.0
    return _corrected_mi_from_joint(joint)


def soft_binary_mutual_info(
    mask: Sequence[float],
    y: Sequence[float],
    *,
    max_bins: int = 96,
    min_samples_per_target_bin: int = 25,
    min_effective_support: float | None = None,
) -> float:
    weights, y_arr = _finite_pair_values(mask, y)
    weights = [_clip(value, 0.0, 1.0) for value in weights]
    n = len(weights)
    if n <= 1 or _constant(y_arr):
        return 0.0

    support_floor = (
        adaptive_min_effective_support(n)
        if min_effective_support is None
        else float(min_effective_support)
    )
    support = soft_mask_support(weights)
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

    hist_in = [0.0] * y_count
    hist_out = [0.0] * y_count
    for weight, y_bin in zip(weights, y_bins):
        hist_in[y_bin] += weight
        hist_out[y_bin] += 1.0 - weight
    return _corrected_mi_from_joint([hist_in, hist_out])


def soft_mask_support(mask: Sequence[float]) -> dict[str, float]:
    weights = [_clip(float(value), 0.0, 1.0) for value in mask]
    if not weights:
        return {
            "sum_in": 0.0,
            "sum_out": 0.0,
            "effective_in": 0.0,
            "effective_out": 0.0,
        }
    sum_in = math.fsum(weights)
    sum_in_sq = math.fsum(value * value for value in weights)
    out = [1.0 - value for value in weights]
    sum_out = math.fsum(out)
    sum_out_sq = math.fsum(value * value for value in out)
    return {
        "sum_in": sum_in,
        "sum_out": sum_out,
        "effective_in": (sum_in * sum_in / sum_in_sq) if sum_in_sq > 1e-12 else 0.0,
        "effective_out": (sum_out * sum_out / sum_out_sq) if sum_out_sq > 1e-12 else 0.0,
    }


def adaptive_min_effective_support(n_samples: int) -> float:
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
        required = samples_per_bin * (level**dimensions)
        if n_samples >= required:
            best = level
    return best


def mi_bin_template_capacity(bin_count: int) -> int:
    bin_count = int(max(1, bin_count))
    for level in ADAPTIVE_MI_BIN_LEVELS:
        if bin_count <= level:
            return int(level)
    return int(ADAPTIVE_MI_BIN_LEVELS[-1])


def adaptive_bin_indices(
    values: Sequence[float],
    max_bins: int,
    *,
    exact_low_cardinality: bool = True,
) -> Tuple[List[int], int]:
    arr = _to_floats(values)
    n = len(arr)
    max_bins = int(max(2, min(int(max_bins), ADAPTIVE_MI_BIN_LEVELS[-1])))
    if n == 0:
        return [], 0

    unique = sorted(set(arr))
    if len(unique) <= 1:
        return [0] * n, 1
    if exact_low_cardinality and len(unique) <= max_bins:
        index = {value: idx for idx, value in enumerate(unique)}
        return [index[value] for value in arr], len(unique)

    bins = int(min(max_bins, n))
    order = sorted(range(n), key=lambda idx: (arr[idx], idx))
    out = [0] * n
    for pos, idx in enumerate(order):
        bin_id = (pos * bins) // n
        out[idx] = min(bin_id, bins - 1)
    return out, bins


def linear_r2(x: Sequence[float], y: Sequence[float]) -> float:
    corr = pearson_corr(x, y)
    return float(corr * corr)


def _safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr, y_arr = _finite_pair_values(x, y)
    n = len(x_arr)
    if n == 0:
        return 0.0
    mean_x = math.fsum(x_arr) / float(n)
    mean_y = math.fsum(y_arr) / float(n)
    x_centered = [value - mean_x for value in x_arr]
    y_centered = [value - mean_y for value in y_arr]
    var_x = math.fsum(value * value for value in x_centered)
    var_y = math.fsum(value * value for value in y_centered)
    denom = math.sqrt(var_x * var_y)
    if denom <= 0.0:
        return 0.0
    return float(math.fsum(a * b for a, b in zip(x_centered, y_centered)) / denom)


def _rankdata(values: Sequence[float]) -> List[float]:
    arr = _to_floats(values)
    n = len(arr)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: (arr[idx], idx))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * float(i + j - 1)
        for pos in range(i, j):
            ranks[order[pos]] = avg_rank
        i = j
    return ranks


def _corrected_mi_from_joint(joint: Sequence[Sequence[float]]) -> float:
    if not joint:
        return 0.0
    row_count = len(joint)
    col_count = len(joint[0])
    if row_count == 0 or col_count == 0:
        return 0.0
    total = math.fsum(math.fsum(row) for row in joint)
    if total <= 0.0:
        return 0.0
    px = [math.fsum(row) for row in joint]
    py = [math.fsum(joint[row][col] for row in range(row_count)) for col in range(col_count)]
    nonzero_rows = sum(1 for value in px if value > 0.0)
    nonzero_cols = sum(1 for value in py if value > 0.0)
    if nonzero_rows < 2 or nonzero_cols < 2:
        return 0.0
    mi = 0.0
    inv_total = 1.0 / total
    for row in range(row_count):
        for col in range(col_count):
            count = float(joint[row][col])
            if count <= 0.0:
                continue
            pxy = count * inv_total
            expected = (px[row] * inv_total) * (py[col] * inv_total)
            if expected > 0.0:
                mi += pxy * math.log(pxy / expected)
    bias = ((nonzero_rows - 1) * (nonzero_cols - 1)) / (2.0 * total)
    return float(max(mi - bias, 0.0))


def _finite_pair_values(x: Sequence[float], y: Sequence[float]) -> Tuple[List[float], List[float]]:
    x_arr = _to_floats(x)
    y_arr = _to_floats(y)
    if len(x_arr) != len(y_arr):
        return [], []
    out_x: List[float] = []
    out_y: List[float] = []
    for a, b in zip(x_arr, y_arr):
        if math.isfinite(a) and math.isfinite(b):
            out_x.append(a)
            out_y.append(b)
    return out_x, out_y


def _to_floats(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def _constant(values: Sequence[float]) -> bool:
    return not values or min(values) == max(values)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)
