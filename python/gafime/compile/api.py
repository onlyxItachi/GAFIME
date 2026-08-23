from __future__ import annotations

from typing import Iterable

from ..api import GafimeEngine
from ..config import EngineConfig
from ..v1_adapter import NativeCompiledGafime
from .flags import CompileFlags


def compile(
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
    *,
    config: EngineConfig | None = None,
    flags: CompileFlags | None = None,
) -> NativeCompiledGafime:
    """Create a thread-affine native artifact from a validated matrix.

    This submodule entry point is equivalent to :func:`gafime.compile`.
    ``flags`` controls compatibility-plan metadata, backend graph replay, and
    Arrow result export.  Unsupported graph/backend combinations fail closed;
    the caller must close the returned artifact on its creation thread.
    """

    return GafimeEngine(config).compile(X, y, feature_names=feature_names, flags=flags)
