from __future__ import annotations

from typing import Iterable

from .config import EngineConfig
from .reporting import DiagnosticReport
from .v1_adapter import (
    NativeCompiledGafime,
    analyze_decision_path_with_v1_boundary,
    analyze_time_series_with_v1_boundary,
    analyze_with_v1_boundary,
    compile_with_v1_boundary,
)


class GafimeEngine:
    """Configure and run native GAFIME analysis.

    ``config`` defaults to :class:`EngineConfig`.  Python owns this declaration
    boundary; planning, candidate scoring, significance, and backend execution
    remain native.  Use :meth:`analyze` for an eager report or :meth:`compile`
    when the same matrix will be analyzed repeatedly.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
    ) -> DiagnosticReport:
        """Analyze a non-empty numeric feature matrix against one target.

        ``X`` must be rectangular, ``y`` must have the same row count, and
        ``feature_names`` must match the column count when supplied.  NumPy,
        ordinary nested sequences, and supported dataframe-like inputs are
        copied into storage selected by ``config.precision``.  Explicit backend
        requests fail closed if their payload, device, family, or precision is
        unsupported; only ``backend="auto"`` may select another backend.

        Returns a :class:`DiagnosticReport`.  Generated time-series or
        decision-path execution is selected by the corresponding mutually
        exclusive ``EngineConfig`` switch.
        """

        if self.config.enable_time_series_functions:
            return analyze_time_series_with_v1_boundary(
                self.config, X, y, feature_names
            )
        if self.config.enable_decision_path_functions:
            return analyze_decision_path_with_v1_boundary(
                self.config, X, y, feature_names
            )
        return analyze_with_v1_boundary(self.config, X, y, feature_names)

    def compile(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
        *,
        flags=None,
    ) -> NativeCompiledGafime:
        """Create a caller-owned resident artifact for repeated analysis.

        ``flags`` accepts :class:`gafime.CompileFlags`.  Graph execution is
        available only when the selected CUDA/ROCm payload proves support;
        result export requires ``CompileFlags(export=True)``.  The returned
        artifact is thread-affine and must be closed on its creation thread.
        Input validation and explicit-backend failure semantics match
        :meth:`analyze`.
        """

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
    """Compile ``X`` and ``y`` into a resident native analysis artifact.

    This is the functional form of :meth:`GafimeEngine.compile`.  The returned
    :class:`NativeCompiledGafime` owns native matrix/session state until
    :meth:`NativeCompiledGafime.close` is called and must be used on the thread
    that created it.
    """

    return GafimeEngine(config).compile(X, y, feature_names=feature_names, flags=flags)
