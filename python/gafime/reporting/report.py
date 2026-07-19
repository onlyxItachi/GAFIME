from __future__ import annotations

from collections.abc import Iterator, Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Tuple
import warnings as _warnings

from ..config import EngineConfig


@dataclass(frozen=True)
class BackendInfo:
    name: str
    device: str
    is_gpu: bool
    memory_total_mb: int | None
    memory_free_mb: int | None
    selected_backend: str | None = None
    execution_placement: str | None = None

    def __post_init__(self) -> None:
        selected = self.selected_backend or {
            "cpu": "core",
            "cuda": "cuda",
            "rocm": "rocm",
            "hip": "rocm",
            "metal": "metal",
        }.get(self.device)
        if selected is not None:
            object.__setattr__(self, "selected_backend", selected)
        if self.execution_placement is None and selected is not None:
            object.__setattr__(
                self,
                "execution_placement",
                "gafime_cpu" if selected == "core" else selected,
            )


@dataclass(frozen=True)
class InteractionResult:
    combo: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    metrics: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class StabilityResult:
    combo: Tuple[int, ...]
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class PermutationResult:
    combo: Tuple[int, ...]
    p_values: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class Decision:
    signal_detected: bool
    message: str


@dataclass
class DiagnosticReport:
    config: EngineConfig
    feature_names: List[str]
    interactions: SequenceABC[InteractionResult] = field(default_factory=list)
    stability: SequenceABC[StabilityResult] = field(default_factory=list)
    permutations: SequenceABC[PermutationResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    decision: Decision | None = None
    backend: BackendInfo | None = None

    @property
    def configured_backend(self) -> str:
        """The caller-requested backend; ``backend`` records the selection used."""

        return self.config.backend

    def to_dict(self) -> Dict[str, object]:
        _warnings.warn(
            "DiagnosticReport.to_dict() materializes the native report for export.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "config": _jsonable(self.config),
            "configured_backend": self.configured_backend,
            "feature_names": list(self.feature_names),
            "interactions": [_jsonable(item) for item in list(self.interactions)],
            "stability": [_jsonable(item) for item in list(self.stability)],
            "permutations": [_jsonable(item) for item in list(self.permutations)],
            "warnings": list(self.warnings),
            "decision": _jsonable(self.decision),
            "backend": _jsonable(self.backend),
        }


class NativeContinuousInteractions(SequenceABC):
    _ITER_BATCH_SIZE = 1024

    def __init__(
        self,
        native_report: Any,
        feature_names: SequenceABC[str],
        metric_names: SequenceABC[str],
        indices: Tuple[int, ...] | None = None,
    ) -> None:
        self.kind = "interaction"
        self._native_report = native_report
        self._feature_names = tuple(str(name) for name in feature_names)
        self._feature_families = tuple(
            _family_for_feature_names((name,)) for name in self._feature_names
        )
        self._metric_names = tuple(str(name) for name in metric_names)
        self._indices = indices

    @property
    def is_native_backed(self) -> bool:
        return True

    @property
    def native_handle(self) -> Any:
        return self._native_report

    @property
    def native_indices(self) -> Tuple[int, ...] | None:
        return self._indices

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return int(len(self._native_report))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = self._indices[index] if self._indices is not None else index
        components = getattr(self._native_report, "interaction_components", None)
        if callable(components):
            combo_values, metric_values, native_candidate_id = components(source_index)
        else:
            combo_values = self._native_report.combo(source_index)
            metric_values = self._native_report.metric_values(source_index)
            native_candidate_id = self._native_report.candidate_id(source_index)
        return self._result_from_components(
            combo_values, metric_values, native_candidate_id, coerce=True
        )

    def _result_from_components(
        self, combo_values, metric_values, native_candidate_id, *, coerce: bool
    ):
        if coerce:
            combo = tuple(int(value) for value in combo_values)
            metrics = {
                name: float(value)
                for name, value in zip(self._metric_names, metric_values)
            }
        else:
            combo = tuple(combo_values)
            metrics = dict(zip(self._metric_names, metric_values))
        feature_names = tuple(self._feature_names[idx] for idx in combo)
        family = "interaction"
        for feature_index in combo:
            feature_family = self._feature_families[feature_index]
            if feature_family == "decision_path":
                family = feature_family
                break
            if feature_family == "time_series":
                family = feature_family
        return InteractionResult(
            combo=combo,
            feature_names=feature_names,
            metrics=metrics,
            family=family,
            expression="*".join(feature_names),
            candidate_id=f"{family}:{native_candidate_id}",
        )

    def __iter__(self) -> Iterator[InteractionResult]:
        batch = getattr(self._native_report, "interaction_components_batch", None)
        if self._indices is not None or not callable(batch):
            for index in range(len(self)):
                yield self[index]
            return
        for start in range(0, len(self), self._ITER_BATCH_SIZE):
            for combo_values, metric_values, candidate_id in batch(
                start, self._ITER_BATCH_SIZE
            ):
                yield self._result_from_components(
                    combo_values, metric_values, candidate_id, coerce=False
                )

    def ranked(
        self,
        metric_name: str | None = None,
        *,
        descending: bool = True,
        limit: int | None = None,
    ) -> "NativeContinuousInteractions":
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0.")
        metric_index = None if metric_name is None else self._metric_names.index(str(metric_name))
        indices = tuple(
            int(value)
            for value in self._native_report.ranked_indices(
                metric_index=metric_index,
                descending=bool(descending),
                limit=limit,
            )
        )
        return NativeContinuousInteractions(
            self._native_report,
            self._feature_names,
            self._metric_names,
            indices,
        )

    def top_k(
        self,
        k: int,
        metric_name: str | None = None,
        *,
        descending: bool = True,
    ) -> "NativeContinuousInteractions":
        if k < 0:
            raise ValueError("k must be >= 0.")
        return self.ranked(metric_name=metric_name, descending=descending, limit=k)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _family_for_feature_names(feature_names: SequenceABC[str]) -> str:
    if any(name.startswith("path[") for name in feature_names):
        return "decision_path"
    if any(
        "_lag" in name
        or "_delta" in name
        or "_velocity" in name
        or "_acceleration" in name
        or "_rollmean" in name
        or "_rollstd" in name
        or "_rollsum" in name
        for name in feature_names
    ):
        return "time_series"
    return "interaction"
