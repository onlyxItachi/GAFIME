from __future__ import annotations

from typing import Iterable

from .config import EngineConfig
from .reporting import DiagnosticReport
from .v1_adapter import (
    NativeCompiledGafime,
    analyze_time_series_with_v1_boundary,
    analyze_with_v1_boundary,
    compile_with_v1_boundary,
)


class GafimeEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
    ) -> DiagnosticReport:
        if self.config.enable_time_series_functions:
            return analyze_time_series_with_v1_boundary(self.config, X, y, feature_names)
        return analyze_with_v1_boundary(self.config, X, y, feature_names)

    def compile(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
        *,
        flags=None,
    ) -> NativeCompiledGafime:
        return compile_with_v1_boundary(
            self.config,
            X,
            y,
            feature_names,
            flags=flags,
        )


def compile(
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
    *,
    config: EngineConfig | None = None,
    flags=None,
) -> NativeCompiledGafime:
    return GafimeEngine(config).compile(X, y, feature_names=feature_names, flags=flags)
