from __future__ import annotations

import numpy as np


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


def mutual_info(x, y, bins: int = 16, xp=np) -> float:
    if bins < 2:
        return 0.0
    hist, _, _ = xp.histogram2d(x, y, bins=bins)
    total = xp.sum(hist)
    if total == 0:
        return 0.0

    p_xy = hist / total
    p_x = xp.sum(p_xy, axis=1, keepdims=True)
    p_y = xp.sum(p_xy, axis=0, keepdims=True)
    expected = p_x * p_y

    nonzero = p_xy > 0
    mi = xp.sum(p_xy[nonzero] * xp.log(p_xy[nonzero] / expected[nonzero]))
    return float(mi)


def linear_r2(x, y, xp=np) -> float:
    corr = _safe_pearson(x, y, xp=xp)
    return float(corr * corr)
