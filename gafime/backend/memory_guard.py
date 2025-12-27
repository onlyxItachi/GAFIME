"""VRAM safety and batch control."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import cupy as cp
    from cupy.cuda.runtime import CUDARuntimeError
except Exception:  # pragma: no cover - optional dependency
    cp = None
    CUDARuntimeError = Exception


@dataclass(frozen=True)
class VRAMStatus:
    free_bytes: int | None
    total_bytes: int | None
    error: str | None = None


@dataclass(frozen=True)
class GuardDecision:
    use_gpu: bool
    batch_rows: int
    keep_in_vram: bool
    reason: str | None = None


def query_vram() -> VRAMStatus:
    if cp is None:
        return VRAMStatus(None, None, error="CuPy not installed")
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return VRAMStatus(int(free_bytes), int(total_bytes))
    except CUDARuntimeError as exc:
        return VRAMStatus(None, None, error=str(exc))


def plan_gpu_batches(
    *,
    total_rows: int,
    bytes_per_row: int,
    keep_in_vram: bool,
    safety_margin: float = 0.8,
) -> GuardDecision:
    status = query_vram()
    if status.free_bytes is None:
        return GuardDecision(
            use_gpu=False,
            batch_rows=0,
            keep_in_vram=False,
            reason=status.error or "Unable to query VRAM",
        )

    if total_rows <= 0:
        return GuardDecision(use_gpu=False, batch_rows=0, keep_in_vram=False, reason="No data")

    required_bytes = total_rows * bytes_per_row
    safe_limit = int(status.free_bytes * safety_margin)

    if required_bytes <= safe_limit:
        return GuardDecision(use_gpu=True, batch_rows=total_rows, keep_in_vram=keep_in_vram)

    if safe_limit <= 0:
        return GuardDecision(
            use_gpu=False,
            batch_rows=0,
            keep_in_vram=False,
            reason="Insufficient VRAM available",
        )

    batch_rows = max(1, int(safe_limit / bytes_per_row))
    if batch_rows < 1:
        return GuardDecision(
            use_gpu=False,
            batch_rows=0,
            keep_in_vram=False,
            reason="Insufficient VRAM for even a single batch",
        )

    return GuardDecision(
        use_gpu=True,
        batch_rows=batch_rows,
        keep_in_vram=False,
        reason="Batching enabled; keep_in_vram overridden",
    )
