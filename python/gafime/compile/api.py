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
    return GafimeEngine(config).compile(X, y, feature_names=feature_names, flags=flags)
