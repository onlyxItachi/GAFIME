"""VRAM safety and batch control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VramInfo:
    free_bytes: int
    total_bytes: int


def get_vram_info() -> Optional[VramInfo]:
    try:
        import cupy
    except Exception:
        return None

    try:
        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return VramInfo(free_bytes=int(free_bytes), total_bytes=int(total_bytes))


def estimate_array_bytes(rows: int, cols: int, dtype_bytes: int = 8) -> int:
    return int(rows) * int(cols) * int(dtype_bytes)


def plan_batch_size(
    total_items: int,
    bytes_per_item: int,
    vram_info: Optional[VramInfo],
    safety_ratio: float = 0.8,
    min_batch: int = 1,
) -> int:
    if vram_info is None:
        return max(min_batch, total_items)
    usable = int(vram_info.free_bytes * safety_ratio)
    if bytes_per_item <= 0:
        return max(min_batch, total_items)
    max_items = max(min_batch, usable // bytes_per_item)
    return min(total_items, max_items)


def should_keep_in_vram(
    requested: bool,
    required_bytes: int,
    vram_info: Optional[VramInfo],
    safety_ratio: float = 0.8,
) -> bool:
    if not requested:
        return False
    if vram_info is None:
        return False
    usable = int(vram_info.free_bytes * safety_ratio)
    return required_bytes <= usable
