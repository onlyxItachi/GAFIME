from __future__ import annotations

from collections.abc import Iterator, Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
import importlib
import warnings as _warnings
from typing import Any, Dict, List, Tuple

from ..config import EngineConfig


@dataclass(frozen=True)
class BackendInfo:
    name: str
    device: str
    is_gpu: bool
    memory_total_mb: int | None
    memory_free_mb: int | None


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

    def __post_init__(self) -> None:
        self.interactions = _NativeResultSequence.from_items("interaction", self.interactions)
        self.stability = _NativeResultSequence.from_items("stability", self.stability)
        self.permutations = _NativeResultSequence.from_items("permutation", self.permutations)

    def to_dict(self) -> Dict[str, object]:
        _warnings.warn(
            "DiagnosticReport.to_dict() is deprecated and materializes a JSON-style export. "
            "Use native report properties such as report.interactions, report.decision, "
            "and report.backend for framework integration.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "config": _jsonable(self.config),
            "feature_names": list(self.feature_names),
            "interactions": [_jsonable(item) for item in list(self.interactions)],
            "stability": [_jsonable(item) for item in list(self.stability)],
            "permutations": [_jsonable(item) for item in list(self.permutations)],
            "warnings": list(self.warnings),
            "decision": _jsonable(self.decision),
            "backend": _jsonable(self.backend),
        }


class NativeReportBuilder:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        core = _core_module()
        self._table = core.NativeReportTable() if core is not None and hasattr(core, "NativeReportTable") else None
        self._fallback: List[Any] = []

    @property
    def is_native_backed(self) -> bool:
        return self._table is not None

    @property
    def native_handle(self) -> Any | None:
        return self._table

    def append(self, item: Any) -> None:
        if self._table is None:
            self._fallback.append(item)
            return
        self._table.append(*_native_payload(self.kind, item))

    def append_interaction(
        self,
        *,
        combo: Tuple[int, ...],
        feature_names: Tuple[str, ...],
        metrics: Dict[str, float],
        family: str = "interaction",
        expression: str = "",
        params: Dict[str, object] | None = None,
        candidate_id: str = "",
    ) -> None:
        item = InteractionResult(
            combo=combo,
            feature_names=feature_names,
            metrics=metrics,
            family=family,
            expression=expression,
            params=params or {},
            candidate_id=candidate_id,
        )
        self.append(item)

    def sequence(self) -> "_NativeResultSequence":
        if self._table is not None:
            return _NativeResultSequence(self.kind, table=self._table)
        return _NativeResultSequence(self.kind, fallback=list(self._fallback))


class NativeContinuousInteractions(SequenceABC):
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
        combo = tuple(self._combo(source_index))
        metrics = {
            name: float(value)
            for name, value in zip(self._metric_names, self._metric_values(source_index))
        }
        return InteractionResult(
            combo=combo,
            feature_names=tuple(self._feature_names[idx] for idx in combo),
            metrics=metrics,
            family="interaction",
            expression="*".join(self._feature_names[idx] for idx in combo),
            candidate_id=f"continuous:{self._candidate_id(source_index)}",
        )

    def __iter__(self) -> Iterator[InteractionResult]:
        for index in range(len(self)):
            yield self[index]

    def ranked(
        self,
        metric_name: str | None = None,
        *,
        descending: bool = True,
        limit: int | None = None,
    ) -> "NativeContinuousInteractions":
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0.")
        metric_index = self._metric_index(metric_name)
        base_indices = self._indices
        ranked_indices = None
        native_ranked = getattr(self._native_report, "ranked_indices", None)
        if native_ranked is not None and base_indices is None:
            ranked_indices = tuple(
                int(value)
                for value in native_ranked(
                    metric_index=metric_index,
                    descending=bool(descending),
                    limit=limit,
                )
            )
        else:
            indices = list(base_indices) if base_indices is not None else list(range(len(self._native_report)))
            indices.sort(
                key=lambda item: _rank_key(
                    self._rank_value(item, metric_index),
                    descending,
                    str(self._candidate_id(item)),
                )
            )
            if limit is not None:
                indices = indices[:limit]
            ranked_indices = tuple(indices)
        return NativeContinuousInteractions(
            self._native_report,
            self._feature_names,
            self._metric_names,
            ranked_indices,
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

    def _metric_index(self, metric_name: str | None) -> int | None:
        if metric_name is None:
            return None
        try:
            return self._metric_names.index(str(metric_name))
        except ValueError as exc:
            raise ValueError(f"Unknown metric: {metric_name!r}") from exc

    def _combo(self, index: int) -> Tuple[int, ...]:
        combo = getattr(self._native_report, "combo", None)
        if combo is not None:
            return tuple(int(value) for value in combo(index))
        record = self._record(index)
        return tuple(int(value) for value in record.combo)

    def _metric_values(self, index: int) -> Tuple[float, ...]:
        metric_values = getattr(self._native_report, "metric_values", None)
        if metric_values is not None:
            return tuple(float(value) for value in metric_values(index))
        record = self._record(index)
        return tuple(float(value) for value in record.metrics)

    def _candidate_id(self, index: int) -> int:
        candidate_id = getattr(self._native_report, "candidate_id", None)
        if candidate_id is not None:
            return int(candidate_id(index))
        record = self._record(index)
        return int(getattr(record, "candidate_id", index))

    def _record(self, index: int) -> Any:
        record = getattr(self._native_report, "record", None)
        if record is not None:
            return record(index)
        return self._native_report.records()[index]

    def _rank_value(self, index: int, metric_index: int | None) -> float | None:
        values = self._metric_values(index)
        if metric_index is not None:
            if metric_index >= len(values):
                return None
            return values[metric_index]
        return _rank_value(self.kind, self._metric_names, values, None)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind='interaction', len={len(self)}, backing='native-v1')"


class _NativeResultSequence(SequenceABC):
    def __init__(
        self,
        kind: str,
        table: Any | None = None,
        fallback: List[Any] | None = None,
        indices: Tuple[int, ...] | None = None,
    ) -> None:
        self.kind = kind
        self._table = table
        self._fallback = fallback
        self._indices = indices

    @classmethod
    def from_items(cls, kind: str, items: SequenceABC[Any]) -> "_NativeResultSequence":
        if isinstance(items, (_NativeResultSequence, NativeContinuousInteractions)):
            return items
        items_list = list(items)
        table = _build_native_result_table(kind, items_list)
        if table is None:
            return cls(kind, fallback=items_list)
        return cls(kind, table=table)

    @property
    def is_native_backed(self) -> bool:
        return self._table is not None

    @property
    def native_handle(self) -> Any | None:
        return self._table

    @property
    def native_indices(self) -> Tuple[int, ...] | None:
        return self._indices

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        if self._table is not None:
            return len(self._table)
        return len(self._fallback or [])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = self._indices[index] if self._indices is not None else index
        if self._table is None:
            return (self._fallback or [])[source_index]
        return _record_from_native_table(self.kind, self._table, source_index)

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def ranked(
        self,
        metric_name: str | None = None,
        *,
        descending: bool = True,
        limit: int | None = None,
    ) -> "_NativeResultSequence":
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0.")
        if self._table is not None:
            indices = list(self._indices) if self._indices is not None else list(range(len(self._table)))
            indices.sort(
                key=lambda index: _native_rank_key(
                    self.kind,
                    self._table,
                    index,
                    metric_name,
                    descending=descending,
                )
            )
            if limit is not None:
                indices = indices[:limit]
            return _NativeResultSequence(self.kind, table=self._table, indices=tuple(indices))

        items = list(self._fallback or [])
        items.sort(
            key=lambda item: _fallback_rank_key(
                self.kind,
                item,
                metric_name,
                descending=descending,
            )
        )
        if limit is not None:
            items = items[:limit]
        return _NativeResultSequence(self.kind, fallback=items)

    def top_k(
        self,
        k: int,
        metric_name: str | None = None,
        *,
        descending: bool = True,
    ) -> "_NativeResultSequence":
        if k < 0:
            raise ValueError("k must be >= 0.")
        return self.ranked(metric_name=metric_name, descending=descending, limit=k)

    def __repr__(self) -> str:
        backing = "native" if self.is_native_backed else "python"
        return f"{type(self).__name__}(kind={self.kind!r}, len={len(self)}, backing={backing})"


def _core_module():
    try:
        return importlib.import_module("gafime.gafime_core")
    except ImportError:
        try:
            return importlib.import_module("gafime_core")
        except ImportError:
            return None


def _build_native_result_table(kind: str, items: List[Any]):
    core = _core_module()
    if core is None or not hasattr(core, "NativeReportTable"):
        return None
    table = core.NativeReportTable()
    for item in items:
        (
            combo,
            feature_names,
            metric_names,
            metric_values,
            secondary_metric_names,
            secondary_metric_values,
            family,
            expression,
            params,
            candidate_id,
        ) = _native_payload(kind, item)
        table.append(
            combo,
            feature_names,
            metric_names,
            metric_values,
            secondary_metric_names,
            secondary_metric_values,
            family,
            expression,
            params,
            candidate_id,
        )
    return table


def _native_payload(kind: str, item: Any):
    combo = [int(value) for value in getattr(item, "combo", ())]
    feature_names = [str(value) for value in getattr(item, "feature_names", ())]
    secondary_metric_names: List[str] = []
    secondary_metric_values: List[float] = []
    if kind == "interaction":
        primary = dict(getattr(item, "metrics", {}))
    elif kind == "stability":
        primary = dict(getattr(item, "metrics_mean", {}))
        secondary = dict(getattr(item, "metrics_std", {}))
        secondary_metric_names = [str(name) for name in secondary]
        secondary_metric_values = [float(value) for value in secondary.values()]
    elif kind == "permutation":
        primary = dict(getattr(item, "p_values", {}))
    else:
        raise ValueError(f"Unknown report result kind: {kind}")

    metric_names = [str(name) for name in primary]
    metric_values = [float(value) for value in primary.values()]
    raw_params = getattr(item, "params", {})
    params = dict(raw_params) if raw_params else None
    return (
        combo,
        feature_names,
        metric_names,
        metric_values,
        secondary_metric_names,
        secondary_metric_values,
        str(getattr(item, "family", "interaction")),
        str(getattr(item, "expression", "")),
        params,
        str(getattr(item, "candidate_id", "")),
    )


def _record_from_native_table(kind: str, table: Any, index: int):
    combo = tuple(int(value) for value in table.combo(index))
    family = str(table.family(index))
    expression = str(table.expression(index))
    params = dict(table.params(index))
    candidate_id = str(table.candidate_id(index))
    primary = _metric_map(table.metric_names(index), table.metric_values(index))
    if kind == "interaction":
        return InteractionResult(
            combo=combo,
            feature_names=tuple(str(value) for value in table.feature_names(index)),
            metrics=primary,
            family=family,
            expression=expression,
            params=params,
            candidate_id=candidate_id,
        )
    if kind == "stability":
        return StabilityResult(
            combo=combo,
            metrics_mean=primary,
            metrics_std=_metric_map(table.secondary_metric_names(index), table.secondary_metric_values(index)),
            family=family,
            expression=expression,
            params=params,
            candidate_id=candidate_id,
        )
    if kind == "permutation":
        return PermutationResult(
            combo=combo,
            p_values=primary,
            family=family,
            expression=expression,
            params=params,
            candidate_id=candidate_id,
        )
    raise ValueError(f"Unknown report result kind: {kind}")


def _metric_map(names: Any, values: Any) -> Dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, values)}


def _native_rank_key(
    kind: str,
    table: Any,
    index: int,
    metric_name: str | None,
    *,
    descending: bool,
) -> tuple[int, float, str]:
    value = _rank_value(
        kind,
        table.metric_names(index),
        table.metric_values(index),
        metric_name,
    )
    return _rank_key(value, descending, _native_tie_breaker(table, index))


def _fallback_rank_key(
    kind: str,
    item: Any,
    metric_name: str | None,
    *,
    descending: bool,
) -> tuple[int, float, str]:
    if kind == "interaction":
        metrics = dict(getattr(item, "metrics", {}))
    elif kind == "stability":
        metrics = dict(getattr(item, "metrics_mean", {}))
    elif kind == "permutation":
        metrics = dict(getattr(item, "p_values", {}))
    else:
        metrics = {}
    value = _rank_value(kind, metrics.keys(), metrics.values(), metric_name)
    candidate_id = str(getattr(item, "candidate_id", ""))
    tie_breaker = candidate_id or ",".join(str(value) for value in getattr(item, "combo", ()))
    return _rank_key(value, descending, tie_breaker)


def _rank_value(
    kind: str,
    names: Any,
    values: Any,
    metric_name: str | None,
) -> float | None:
    metric_values = {str(name): float(value) for name, value in zip(names, values)}
    if metric_name is not None:
        return metric_values.get(str(metric_name))
    if not metric_values:
        return None
    if kind == "permutation":
        return -min(metric_values.values())
    strengths = [
        abs(value) if name in {"pearson", "spearman"} else value
        for name, value in metric_values.items()
    ]
    return max(strengths)


def _rank_key(value: float | None, descending: bool, tie_breaker: str) -> tuple[int, float, str]:
    if value is None:
        return (1, 0.0, tie_breaker)
    rank_value = -float(value) if descending else float(value)
    return (0, rank_value, tie_breaker)


def _native_tie_breaker(table: Any, index: int) -> str:
    candidate_id = str(table.candidate_id(index))
    if candidate_id:
        return candidate_id
    return ",".join(str(value) for value in table.combo(index))


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
