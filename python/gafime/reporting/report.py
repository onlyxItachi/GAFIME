from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
from numbers import Integral, Real
from typing import Any, Dict, List, Tuple
import warnings as _warnings

from .._precision import normalize_precision
from ..config import EngineConfig


@dataclass(frozen=True)
class BackendInfo:
    """Backend and numeric-domain facts for one completed analysis.

    ``name`` is the native boundary identity, while ``selected_backend`` and
    ``execution_placement`` describe where the request actually ran.
    ``requested_precision`` and ``effective_precision`` are kept separate so a
    report never implies an unobserved conversion.  Memory fields may be
    ``None`` when the boundary cannot report device capacity.
    """

    name: str
    device: str
    is_gpu: bool
    memory_total_mb: int | None
    memory_free_mb: int | None
    selected_backend: str | None = None
    execution_placement: str | None = None
    requested_precision: str = "mixed"
    effective_precision: str = "mixed"
    storage_dtype: str = "float32"
    interaction_arithmetic: str = "float32"
    reduction_dtype: str = "float64"
    result_dtype: str = "float64"
    metric_accumulators: Dict[str, str] = field(default_factory=dict)
    scale_normalization: str | None = None
    compensated_summation: bool = False
    interaction_diagnostics_available: bool = False

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
    """One surfaced candidate and its metric values.

    ``combo`` indexes ``DiagnosticReport.feature_names``; ``candidate_id`` is
    the stable native identity and ``family`` distinguishes continuous,
    time-series, and decision-path results.  Precision diagnostics separate
    finite-input pointwise overflow from source non-finite values and are
    meaningful only when ``precision_diagnostics_available`` is true.
    """

    combo: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    metrics: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""
    interaction_overflow_rows: int = 0
    interaction_overflow_ratio: float = 0.0
    source_nonfinite: bool = False
    precision_diagnostics_available: bool = False


@dataclass(frozen=True)
class StabilityResult:
    """Conditional bootstrap summary for one already-selected candidate.

    ``metrics_mean`` and ``metrics_std`` describe resampling on the same rows
    that selected the candidate.  They are not out-of-sample evidence and do
    not correct selection bias.
    """

    combo: Tuple[int, ...]
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class PermutationResult:
    """Family-wise maxT p-values for one surfaced candidate.

    Generated adaptive families rediscover their target-dependent candidates
    for every permutation.  ``p_values`` therefore follows the configured
    current-v1 significance contract rather than the historical candidate-wise
    v0.x procedure.
    """

    combo: Tuple[int, ...]
    p_values: Dict[str, float]
    family: str = "interaction"
    expression: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class Decision:
    """Threshold-based summary of whether the configured report found signal."""

    signal_detected: bool
    message: str


@dataclass
class DiagnosticReport:
    """Structured result of one eager or compiled GAFIME analysis.

    ``interactions``, ``stability``, and ``permutations`` are sequence-like and
    support slicing; native interaction sequences additionally provide
    ``ranked`` and ``top_k`` without first materializing every result.
    ``config.backend`` is the configured request, whereas ``backend`` records
    the selected placement and effective numeric domains.  Warnings disclose
    bounded candidate caps or generated-family details without changing the
    request silently.
    """

    config: EngineConfig
    feature_names: List[str]
    interactions: SequenceABC[InteractionResult] = field(default_factory=list)
    stability: SequenceABC[StabilityResult] = field(default_factory=list)
    permutations: SequenceABC[PermutationResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    decision: Decision | None = None
    backend: BackendInfo | None = None

    def __post_init__(self) -> None:
        self.interactions = _result_sequence("interaction", self.interactions)
        self.stability = _result_sequence("stability", self.stability)
        self.permutations = _result_sequence("permutation", self.permutations)

    @property
    def configured_backend(self) -> str:
        """The caller-requested backend; ``backend`` records the selection used."""

        return self.config.backend

    def to_dict(self) -> Dict[str, object]:
        """Materialize the full report into JSON-like Python values.

        This deprecated convenience can copy every native result row and emits
        :class:`DeprecationWarning`; prefer sequence access/ranking or compiled
        Arrow export for non-trivial data flow.
        """

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


class NativeReportBuilder:
    """Python-backed compatibility facade for the v0.5 report builder API.

    The builder is intended for compatibility tests and small custom adapters,
    not native production scoring.  ``kind`` determines default ranking
    semantics for the frozen sequence returned by :meth:`sequence`.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: List[Any] = []

    @property
    def is_native_backed(self) -> bool:
        """Always ``False`` for this Python compatibility builder."""

        return False

    @property
    def native_handle(self) -> Any | None:
        """Always ``None`` because this builder owns Python items only."""

        return None

    def append(self, item: Any) -> None:
        """Append one result-like Python object to the builder."""

        self._items.append(item)

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
        """Construct and append one :class:`InteractionResult`."""

        self.append(
            InteractionResult(
                combo=combo,
                feature_names=feature_names,
                metrics=metrics,
                family=family,
                expression=expression,
                params=params or {},
                candidate_id=candidate_id,
            )
        )

    def sequence(self) -> "_PythonResultSequence":
        """Return a frozen Python-backed snapshot of current builder items."""

        return _PythonResultSequence(self.kind, list(self._items))


class _PythonResultSequence(SequenceABC):
    def __init__(self, kind: str, items: List[Any]) -> None:
        self.kind = kind
        self._items = items

    @property
    def is_native_backed(self) -> bool:
        return False

    @property
    def native_handle(self) -> Any | None:
        return None

    @property
    def native_indices(self) -> Tuple[int, ...] | None:
        return None

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._items[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def ranked(
        self,
        metric_name: str | None = None,
        *,
        descending: bool = True,
        limit: int | None = None,
    ) -> "_PythonResultSequence":
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0.")
        items = sorted(
            self._items,
            key=lambda item: _python_result_rank_key(
                self.kind,
                item,
                metric_name,
                descending=descending,
            ),
        )
        if limit is not None:
            items = items[:limit]
        return _PythonResultSequence(self.kind, items)

    def top_k(
        self,
        k: int,
        metric_name: str | None = None,
        *,
        descending: bool = True,
    ) -> "_PythonResultSequence":
        if k < 0:
            raise ValueError("k must be >= 0.")
        return self.ranked(metric_name=metric_name, descending=descending, limit=k)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind!r}, len={len(self)}, backing=python)"


class NativeContinuousInteractions(SequenceABC):
    """Lazy sequence view over a native compact interaction report.

    Users normally obtain this object as ``DiagnosticReport.interactions``
    rather than constructing it.  Indexing materializes only the requested
    row, iteration uses native batches when available, and ranking remains in
    the native result dtype.  ``generated_feature_start`` and
    ``generated_family`` are an all-or-none advanced adapter contract.
    """

    _ITER_BATCH_SIZE = 1024

    def __init__(
        self,
        native_report: Any,
        feature_names: SequenceABC[str],
        metric_names: SequenceABC[str],
        indices: Tuple[int, ...] | None = None,
        *,
        precision: str | None = None,
        generated_feature_start: int | None = None,
        generated_family: str | None = None,
    ) -> None:
        self.kind = "interaction"
        self._native_report = native_report
        self._feature_names = tuple(str(name) for name in feature_names)
        if generated_family not in (None, "time_series", "decision_path"):
            raise ValueError("generated_family must be time_series or decision_path.")
        if (generated_feature_start is None) != (generated_family is None):
            raise ValueError(
                "generated_feature_start and generated_family must be provided together."
            )
        if generated_feature_start is not None and not (
            0 <= generated_feature_start <= len(self._feature_names)
        ):
            raise ValueError("generated_feature_start is outside feature_names.")
        self._generated_feature_start = generated_feature_start
        self._generated_family = generated_family
        self._feature_families = tuple(
            generated_family
            if generated_feature_start is not None and index >= generated_feature_start
            else "interaction"
            for index in range(len(self._feature_names))
        )
        self._metric_names = tuple(str(name) for name in metric_names)
        self._indices = indices
        self._precision = normalize_precision(
            precision
            if precision is not None
            else getattr(native_report, "precision", "mixed")
        )
        self._precision_diagnostics_available = bool(
            getattr(native_report, "interaction_diagnostics_available", False)
        )
        self._interaction_diagnostic = getattr(
            native_report, "interaction_diagnostic", None
        )
        self._interaction_diagnostics_batch = getattr(
            native_report, "interaction_diagnostics_batch", None
        )
        native_diagnostics = (
            None
            if callable(self._interaction_diagnostic)
            else getattr(native_report, "interaction_diagnostics", None)
        )
        self._interaction_diagnostics = (
            tuple(native_diagnostics)
            if self._precision_diagnostics_available and native_diagnostics is not None
            else ()
        )
        self._sample_rows = int(getattr(native_report, "rows", 0))

    @property
    def is_native_backed(self) -> bool:
        """Always ``True`` for this compact native view."""

        return True

    @property
    def native_handle(self) -> Any:
        """Return the advanced native report handle backing this view."""

        return self._native_report

    @property
    def native_indices(self) -> Tuple[int, ...] | None:
        """Return ranked/subset native indices, or ``None`` for the full view."""

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
            combo_values,
            metric_values,
            native_candidate_id,
            source_index=source_index,
            coerce=True,
        )

    def _result_from_components(
        self,
        combo_values,
        metric_values,
        native_candidate_id,
        *,
        source_index: int,
        coerce: bool,
        diagnostic=None,
    ):
        if coerce:
            combo = tuple(int(value) for value in combo_values)
            metrics = dict(zip(self._metric_names, metric_values))
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
        overflow_rows = 0
        source_nonfinite = False
        if self._precision_diagnostics_available:
            if diagnostic is None:
                if callable(self._interaction_diagnostic):
                    diagnostic = self._interaction_diagnostic(source_index)
                else:
                    diagnostic = self._interaction_diagnostics[source_index]
            if diagnostic is None:
                raise RuntimeError(
                    "native report advertised diagnostics but returned no diagnostic row"
                )
            overflow_rows, source_nonfinite = diagnostic
            overflow_rows = int(overflow_rows)
            source_nonfinite = bool(source_nonfinite)
        return InteractionResult(
            combo=combo,
            feature_names=feature_names,
            metrics=metrics,
            family=family,
            expression="*".join(feature_names),
            candidate_id=f"{family}:{native_candidate_id}",
            interaction_overflow_rows=overflow_rows,
            interaction_overflow_ratio=self._overflow_ratio(overflow_rows),
            source_nonfinite=source_nonfinite,
            precision_diagnostics_available=self._precision_diagnostics_available,
        )

    def _overflow_ratio(self, overflow_rows: int) -> float:
        ratio = overflow_rows / self._sample_rows if self._sample_rows else 0.0
        if self._precision == "fp32":
            return array("f", (ratio,))[0]
        return ratio

    def __iter__(self) -> Iterator[InteractionResult]:
        batch = getattr(self._native_report, "interaction_components_batch", None)
        if self._indices is not None or not callable(batch):
            for index in range(len(self)):
                yield self[index]
            return
        for start in range(0, len(self), self._ITER_BATCH_SIZE):
            components = batch(start, self._ITER_BATCH_SIZE)
            diagnostics = None
            if self._precision_diagnostics_available and callable(
                self._interaction_diagnostics_batch
            ):
                diagnostics = self._interaction_diagnostics_batch(
                    start, len(components)
                )
                if diagnostics is None or len(diagnostics) != len(components):
                    raise RuntimeError(
                        "native diagnostic batch does not align with interaction rows"
                    )
            for offset, (combo_values, metric_values, candidate_id) in enumerate(
                components
            ):
                yield self._result_from_components(
                    combo_values,
                    metric_values,
                    candidate_id,
                    source_index=start + offset,
                    coerce=False,
                    diagnostic=None if diagnostics is None else diagnostics[offset],
                )

    def ranked(
        self,
        metric_name: str | None = None,
        *,
        descending: bool = True,
        limit: int | None = None,
    ) -> "NativeContinuousInteractions":
        """Return a lazy deterministically ranked view.

        ``metric_name=None`` uses the report's established primary metric.
        Unknown metric names and negative limits raise :class:`ValueError`.
        Non-finite scores sort behind finite scores; native candidate identity
        supplies the deterministic tie order.
        """

        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0.")
        requested_metric = None if metric_name is None else str(metric_name)
        metric_index = (
            None
            if requested_metric is None
            else self._metric_names.index(requested_metric)
        )
        if self._indices is None:
            indices = tuple(
                int(value)
                for value in self._native_report.ranked_indices(
                    metric_index=metric_index,
                    descending=bool(descending),
                    limit=limit,
                )
            )
        else:
            rank_subset = getattr(self._native_report, "ranked_indices_subset", None)
            if callable(rank_subset):
                indices = tuple(
                    int(value)
                    for value in rank_subset(
                        list(self._indices),
                        metric_index=metric_index,
                        descending=bool(descending),
                        limit=limit,
                    )
                )
            else:
                # Compatibility for third-party/older boundaries that do not
                # expose typed subset ranking. Shipped ABI 1.1 reports always
                # take the native path above, preserving fp32 comparisons.
                indices = tuple(
                    sorted(
                        self._indices,
                        key=lambda index: _native_continuous_rank_key(
                            self._native_report,
                            self._metric_names,
                            index,
                            requested_metric,
                            descending=descending,
                        ),
                    )[:limit]
                )
        return type(self)(
            self._native_report,
            self._feature_names,
            self._metric_names,
            indices,
            precision=self._precision,
            generated_feature_start=self._generated_feature_start,
            generated_family=self._generated_family,
        )

    def top_k(
        self,
        k: int,
        metric_name: str | None = None,
        *,
        descending: bool = True,
    ) -> "NativeContinuousInteractions":
        """Return at most ``k`` rows from :meth:`ranked`; ``k`` must be non-negative."""

        if k < 0:
            raise ValueError("k must be >= 0.")
        return self.ranked(metric_name=metric_name, descending=descending, limit=k)


def _result_sequence(kind: str, items: SequenceABC[Any]) -> SequenceABC[Any]:
    if isinstance(items, (_PythonResultSequence, NativeContinuousInteractions)):
        return items
    return _PythonResultSequence(kind, list(items))


def _native_continuous_rank_key(
    native_report: Any,
    metric_names: Tuple[str, ...],
    index: int,
    metric_name: str | None,
    *,
    descending: bool,
) -> tuple[int, float, int]:
    metrics = dict(zip(metric_names, native_report.metric_values(index)))
    if metric_name is not None:
        value = metrics.get(metric_name)
    elif not metrics:
        value = None
    else:
        value = max(
            abs(metric_value) if name in {"pearson", "spearman"} else metric_value
            for name, metric_value in metrics.items()
        )
    candidate_id = int(native_report.candidate_id(index))
    if value is None:
        return (1, 0.0, candidate_id)
    rank_value = -value if descending else value
    return (0, rank_value, candidate_id)


def _python_result_rank_key(
    kind: str,
    item: Any,
    metric_name: str | None,
    *,
    descending: bool,
) -> tuple[int, float, str]:
    if kind == "interaction":
        raw_metrics = dict(getattr(item, "metrics", {}))
    elif kind == "stability":
        raw_metrics = dict(getattr(item, "metrics_mean", {}))
    elif kind == "permutation":
        raw_metrics = dict(getattr(item, "p_values", {}))
    else:
        raw_metrics = {}
    metrics = {str(name): float(value) for name, value in raw_metrics.items()}

    if metric_name is not None:
        value = metrics.get(str(metric_name))
    elif not metrics:
        value = None
    elif kind == "permutation":
        value = -min(metrics.values())
    else:
        value = max(
            abs(metric_value) if name in {"pearson", "spearman"} else metric_value
            for name, metric_value in metrics.items()
        )

    candidate_id = str(getattr(item, "candidate_id", ""))
    tie_breaker = candidate_id or ",".join(
        str(feature) for feature in getattr(item, "combo", ())
    )
    if value is None:
        return (1, 0.0, tie_breaker)
    rank_value = -float(value) if descending else float(value)
    return (0, rank_value, tie_breaker)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
