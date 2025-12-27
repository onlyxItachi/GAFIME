"""VRAM safety and batch control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VRAMInfo:
    free_bytes: int
    total_bytes: int


def get_vram_info() -> VRAMInfo | None:
    try:
        import cupy as cp
    except Exception:
        return None

    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return VRAMInfo(free_bytes=int(free_bytes), total_bytes=int(total_bytes))


def estimate_bytes_for_array(shape: Tuple[int, ...], itemsize: int) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size * itemsize)


def estimate_required_bytes(features_shape: Tuple[int, ...], target_shape: Tuple[int, ...], itemsize: int) -> int:
    return estimate_bytes_for_array(features_shape, itemsize) + estimate_bytes_for_array(target_shape, itemsize)


def enforce_keep_in_vram(
    *,
    keep_in_vram: bool,
    required_bytes: int,
    safety_margin: float = 0.1,
) -> tuple[bool, str | None]:
    """Return adjusted keep_in_vram and reason if overridden."""
    if not keep_in_vram:
        return False, None

    vram = get_vram_info()
    if vram is None:
        return False, "gpu_unavailable"

    safe_free = int(vram.free_bytes * (1.0 - safety_margin))
    if required_bytes > safe_free:
        return False, "insufficient_vram"

    return True, None


def adapt_batch_size(
    *,
    max_batch_size: int,
    bytes_per_item: int,
    safety_margin: float = 0.1,
) -> int:
    """Return a batch size adjusted to available VRAM."""
    vram = get_vram_info()
    if vram is None:
        return max_batch_size

    safe_free = int(vram.free_bytes * (1.0 - safety_margin))
    if bytes_per_item <= 0:
        return max_batch_size
    cap = max(1, safe_free // bytes_per_item)
    return max(1, min(max_batch_size, cap))
