from __future__ import annotations

import logging
from typing import Any, Protocol, Type, Union

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

logger = logging.getLogger(__name__)


class ArrayBackend(Protocol):
    """Protocol for array backends (numpy or cupy)."""
    
    @property
    def name(self) -> str: ...
    
    def asarray(self, obj: Any, dtype: Any = None) -> Any: ...
    def asnumpy(self, arr: Any) -> np.ndarray: ...
    def to_device(self, arr: np.ndarray) -> Any: ...
    
    # Mathematical operations
    def mean(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any: ...
    def std(self, a: Any, axis: Any = None, ddof: int = 0) -> Any: ...
    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any: ...
    def sqrt(self, a: Any) -> Any: ...
    def log(self, a: Any) -> Any: ...
    def abs(self, a: Any) -> Any: ...
    def dot(self, a: Any, b: Any) -> Any: ...
    def prod(self, a: Any, axis: Any = None) -> Any: ...
    def argsort(self, a: Any, axis: Any = -1) -> Any: ...
    def transpose(self, a: Any) -> Any: ...
    def expand_dims(self, a: Any, axis: Any) -> Any: ...
    def isnan(self, a: Any) -> Any: ...
    def isinf(self, a: Any) -> Any: ...


class NumpyBackend:
    name = "cpu"
    xp = np

    def asarray(self, obj: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(obj, dtype=dtype)

    def asnumpy(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


class CupyBackend:
    name = "gpu"
    xp = cp

    def asarray(self, obj: Any, dtype: Any = None) -> Any:
        return cp.asarray(obj, dtype=dtype)

    def asnumpy(self, arr: Any) -> np.ndarray:
        return cp.asnumpy(arr)
    
    def to_device(self, arr: np.ndarray) -> Any:
        return cp.asarray(arr)

    def __getattr__(self, name: str) -> Any:
        return getattr(cp, name)


def get_backend(prefer_gpu: bool = True) -> ArrayBackend:
    if prefer_gpu and HAS_CUPY:
        logger.info("GAFIME: GPU backend (CuPy) activated.")
        return CupyBackend()
    
    if prefer_gpu and not HAS_CUPY:
        logger.warning("GAFIME: GPU requested but CuPy not installed. Falling back to CPU.")
    
    return NumpyBackend()
