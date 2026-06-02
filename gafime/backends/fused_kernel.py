from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


class UnaryOp:
    IDENTITY = 0
    LOG = 1
    EXP = 2
    SQRT = 3
    TANH = 4
    SIGMOID = 5
    SQUARE = 6
    NEGATE = 7
    ABS = 8
    INVERSE = 9
    CUBE = 10

    _names = {
        IDENTITY: "IDENTITY",
        LOG: "LOG",
        EXP: "EXP",
        SQRT: "SQRT",
        TANH: "TANH",
        SIGMOID: "SIGMOID",
        SQUARE: "SQUARE",
        NEGATE: "NEGATE",
        ABS: "ABS",
        INVERSE: "INVERSE",
        CUBE: "CUBE",
    }


class InteractionType:
    MULT = 0
    ADD = 1
    SUB = 2
    DIV = 3
    MAX = 4
    MIN = 5

    _names = {
        MULT: "MULT",
        ADD: "ADD",
        SUB: "SUB",
        DIV: "DIV",
        MAX: "MAX",
        MIN: "MIN",
    }


class CandidateKind:
    CONTINUOUS = 0
    TS_LAG = 1
    TS_DELTA = 2
    TS_VELOCITY = 3
    TS_ACCELERATION = 4
    TS_ROLLING_MEAN = 5
    TS_ROLLING_STD = 6
    TS_ROLLING_SUM = 7


@dataclass(frozen=True)
class GpuConfig:
    gpu_name: str
    block_size: int
    max_blocks: int
    sm_count: int
    compute_major: int
    compute_minor: int
    l2_cache_bytes: int


def get_gpu_config() -> GpuConfig:
    from .native_cuda_backend import NativeCudaBackend

    backend = NativeCudaBackend()
    info = backend.info()
    return GpuConfig(
        gpu_name=info.device,
        block_size=256,
        max_blocks=256,
        sm_count=0,
        compute_major=0,
        compute_minor=0,
        l2_cache_bytes=0,
    )


def pearson_from_stats(
    n: float,
    sx: float,
    sy: float,
    sxx: float,
    syy: float,
    sxy: float,
) -> float:
    n = float(n)
    if n <= 0.0:
        return 0.0
    numerator = n * float(sxy) - float(sx) * float(sy)
    denom_x = n * float(sxx) - float(sx) * float(sx)
    denom_y = n * float(syy) - float(sy) * float(sy)
    denom = math.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0))
    if denom <= 0.0:
        return 0.0
    return numerator / denom


def unpack_stats(stats: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    values = [float(value) for value in stats]
    if len(values) != 12:
        raise ValueError("stats must contain exactly 12 values.")
    train = {
        "n": values[0],
        "sum_x": values[1],
        "sum_y": values[2],
        "sum_x2": values[3],
        "sum_y2": values[4],
        "sum_xy": values[5],
    }
    val = {
        "n": values[6],
        "sum_x": values[7],
        "sum_y": values[8],
        "sum_x2": values[9],
        "sum_y2": values[10],
        "sum_xy": values[11],
    }
    return train, val


def compute_pearson_from_stats(stats: Sequence[float]) -> Tuple[float, float]:
    train, val = unpack_stats(stats)
    return (
        pearson_from_stats(
            train["n"], train["sum_x"], train["sum_y"],
            train["sum_x2"], train["sum_y2"], train["sum_xy"],
        ),
        pearson_from_stats(
            val["n"], val["sum_x"], val["sum_y"],
            val["sum_x2"], val["sum_y2"], val["sum_xy"],
        ),
    )


class FusedKernelWrapper:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "FusedKernelWrapper was removed in GAFIME v0.4.5. "
            "Use NativeCudaBackend through GafimeEngine; it routes through the "
            "arity-template batch spine."
        )


class StaticBucket:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "StaticBucket direct debug wrapper was removed in GAFIME v0.4.5. "
            "Use GafimeEngine/NativeCudaBackend arity-template batches."
        )
