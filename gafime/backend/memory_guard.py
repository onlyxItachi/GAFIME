"""GPU memory guard utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


@dataclass
class VramInfo:
    free_bytes: int
    total_bytes: int


def get_vram_info() -> VramInfo | None:
    """Return current VRAM info if CuPy is available."""
    if cp is None:
        return None
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return VramInfo(free_bytes=int(free_bytes), total_bytes=int(total_bytes))
    except Exception:
        return None


def estimate_batch_size(
    total_elements: int,
    element_size: int,
    free_bytes: int,
    safety_factor: float = 0.8,
) -> int:
    """Estimate a safe batch size given free VRAM."""
    if total_elements <= 0 or element_size <= 0 or free_bytes <= 0:
        return 0
    usable = int(free_bytes * safety_factor)
    per_element = element_size
    max_elements = max(1, usable // per_element)
    return min(total_elements, max_elements)


def enforce_keep_in_vram(keep_in_vram: bool) -> bool:
    """Soft override keep_in_vram when VRAM is unavailable."""
    if keep_in_vram and get_vram_info() is None:
        return False
    return keep_in_vram
