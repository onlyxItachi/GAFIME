from __future__ import annotations

from typing import Iterable

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
    from ..api import GafimeEngine

    engine = GafimeEngine(config)
    return engine.compile(X, y, feature_names=feature_names, flags=flags)
