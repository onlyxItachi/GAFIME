from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, List

from ..backends import resolve_backend
from ..config import EngineConfig
from ..utils.arrays import coerce_inputs
from ..utils.safety import validate_budget
from .exports import ExportHandles, unsupported_export
from .flags import CompileFlags
from .scenario import build_scenario_plan

if TYPE_CHECKING:
    from ..backends.base import Backend, BackendInfo
    from ..engine import GafimeEngine
    from ..native_data import NativeMatrix, NativeVector
    from .sessions import BackendSession


@dataclass
class CompiledGafime:
    config: EngineConfig
    flags: CompileFlags
    feature_names: List[str]
    warnings: List[str] = field(default_factory=list)
    scenario_plan: Any | None = None
    _engine: "GafimeEngine | None" = None
    _X: "NativeMatrix | None" = None
    _y: "NativeVector | None" = None
    _backend: "Backend | None" = None
    _session: "BackendSession | None" = None
    _exports: ExportHandles | None = None
    _closed: bool = False

    @classmethod
    def from_engine(
        cls,
        engine: "GafimeEngine",
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        feature_names: Iterable[str] | None = None,
        *,
        flags: CompileFlags | None = None,
    ) -> "CompiledGafime":
        compile_flags = flags or CompileFlags()
        X_array, y_array, names = coerce_inputs(X, y, feature_names)
        warnings = validate_budget(X_array.shape[1], engine.config.budget)
        backend, backend_warnings = resolve_backend(engine.config, X_array, y_array)
        warnings.extend(backend_warnings)
        metric_suite = backend.metric_suite(engine.config)
        scenario_plan = build_scenario_plan(X_array, engine.config, compile_flags)
        warnings.extend(scenario_plan.warnings)
        session = backend.compile_session(
            X_array,
            y_array,
            scenario_plan,
            metric_suite,
            compile_flags,
        )
        warnings.extend(getattr(session, "warnings", []))
        exports = _prepare_exports(backend, session, compile_flags)
        return cls(
            config=engine.config,
            flags=compile_flags,
            feature_names=names,
            warnings=warnings,
            scenario_plan=scenario_plan,
            _engine=engine,
            _X=X_array,
            _y=y_array,
            _backend=backend,
            _session=session,
            _exports=exports,
        )

    @property
    def backend(self) -> "BackendInfo":
        self._ensure_open()
        if self._backend is None:
            raise RuntimeError("CompiledGafime has no backend.")
        return self._backend.info()

    @property
    def exports(self) -> ExportHandles:
        self._ensure_open()
        if self._exports is None:
            backend_name = self.backend.name if self._backend is not None else "unknown"
            raise unsupported_export(backend_name)
        return self._exports

    def analyze(self):
        self._ensure_open()
        if self._engine is None or self._X is None or self._y is None:
            raise RuntimeError("CompiledGafime is missing analysis inputs.")
        return self._engine._analyze_native(
            self._X,
            self._y,
            self.feature_names,
            initial_warnings=self.warnings,
            backend=self._backend,
            executor=self._session,
            prevalidated=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._session is not None:
            self._session.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CompiledGafime is closed.")


def _prepare_exports(backend: Any, session: Any, flags: CompileFlags) -> ExportHandles | None:
    if not flags.export:
        return None
    return ExportHandles(
        backend_name=backend.info().name,
        feature_matrix_handle=getattr(session, "feature_matrix_handle", None),
        result_table_handle=getattr(session, "result_table_handle", None),
        candidate_table_handle=getattr(session, "candidate_table_handle", None),
    )
