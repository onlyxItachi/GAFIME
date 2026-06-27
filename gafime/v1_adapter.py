from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Iterable, List

from .backends.base import BackendInfo
from .config import EngineConfig
from .reporting import Decision, DiagnosticReport, InteractionResult
from .utils.arrays import coerce_inputs


_BOUNDARY_MODULE_ENV = "GAFIME_V1_BOUNDARY_MODULE"
_BOUNDARY_MODULES = ("gafime.gafime_py", "gafime_py")
_METRIC_IDS = {
    "pearson": 1,
    "spearman": 2,
    "mutual_info": 3,
    "r2": 4,
}


def analyze_with_v1_boundary(
    engine,
    X: Iterable[Iterable[float]],
    y: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> DiagnosticReport:
    boundary = _load_boundary()
    X_array, y_array, names = coerce_inputs(X, y, feature_names)
    config: EngineConfig = engine.config
    metric_ids = [_metric_id(name) for name in config.metric_names]

    native_report = boundary.analyze_continuous_cpu(
        X_array.rows(),
        y_array.to_list(),
        max_arity=int(config.budget.max_comb_size),
        max_combinations_per_k=int(config.budget.max_combinations_per_k),
        metric_ids=metric_ids,
    )
    interactions = _interaction_results(native_report, names, config.metric_names)
    return DiagnosticReport(
        config=config,
        feature_names=names,
        interactions=interactions,
        stability=[],
        permutations=[],
        warnings=["GAFIME_V1_ENGINE=1 used experimental Rust v1 continuous boundary."],
        decision=Decision(bool(interactions), "v1 continuous opt-in path executed."),
        backend=BackendInfo(
            name="v1-rust-cpu",
            device="cpu",
            is_gpu=False,
            memory_total_mb=None,
            memory_free_mb=None,
        ),
    )


def _load_boundary() -> ModuleType:
    explicit = os.environ.get(_BOUNDARY_MODULE_ENV)
    names = (explicit,) if explicit else _BOUNDARY_MODULES
    failures: List[str] = []
    for name in names:
        if not name:
            continue
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            failures.append(f"{name}: {exc}")
            continue
        if not hasattr(module, "analyze_continuous_cpu"):
            failures.append(f"{name}: missing analyze_continuous_cpu")
            continue
        return module
    detail = "; ".join(failures) if failures else "no module names configured"
    raise RuntimeError(f"GAFIME v1 Python boundary is unavailable: {detail}")


def _metric_id(metric_name: str) -> int:
    try:
        return _METRIC_IDS[str(metric_name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported v1 metric: {metric_name!r}") from exc


def _interaction_results(native_report, feature_names: List[str], metric_names: Iterable[str]):
    metric_names = tuple(str(name) for name in metric_names)
    out = []
    for record in native_report.records():
        combo = tuple(int(value) for value in record.combo)
        metrics = {
            metric_names[idx]: float(value)
            for idx, value in enumerate(record.metrics)
            if idx < len(metric_names)
        }
        out.append(
            InteractionResult(
                combo=combo,
                feature_names=tuple(feature_names[idx] for idx in combo),
                metrics=metrics,
                family="interaction",
                expression="*".join(feature_names[idx] for idx in combo),
                candidate_id=f"continuous:{','.join(str(idx) for idx in combo)}",
            )
        )
    return out
