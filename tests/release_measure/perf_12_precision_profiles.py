#!/usr/bin/env python3
"""Historical provisional benchmark for the three precision profiles.

.. warning::

   Output from this script is methodologically invalid for before/after or
   cross-profile performance claims.  It is retained only so old PR evidence
   remains reproducible.  Use ``perf_13_precision_profiles.py`` for release
   evidence: that harness isolates cold samples, exercises every profile order,
   separates source-dtype policies, and emits raw distributions and provenance.

The benchmark intentionally uses only the public ``EngineConfig`` precision
surface and continuous-family execution.  It reports raw measurements and
parity evidence; it does not infer or promise a universal speedup.

Examples::

    python perf_12_precision_profiles.py --backend core
    python perf_12_precision_profiles.py \
        --backend core,cuda --backend rocm \
        --profile fp32,mixed --samples 4096 --features 12 --repeats 5 \
        --output precision-profiles.json

An explicitly requested backend is mandatory.  Missing payloads, unavailable
devices, unsupported backend/profile pairs, and missing CUDA/ROCm graph support
are errors rather than skips.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
from typing import Callable, Mapping, Sequence

import numpy as np

import gafime
from gafime import CompileFlags, ComputeBudget, EngineConfig, GafimeEngine


ALL_METRICS = ("pearson", "spearman", "mutual_info", "r2")
BACKEND_ORDER = ("core", "cuda", "rocm", "metal")
PROFILE_ORDER = ("fp32", "mixed", "fp64")
PROFILE_MATRIX = {
    "core": PROFILE_ORDER,
    "cuda": PROFILE_ORDER,
    "rocm": PROFILE_ORDER,
    "metal": ("fp32",),
}
GRAPH_BACKENDS = frozenset(("cuda", "rocm"))
MAX_SAMPLES = 65_536
MAX_FEATURES = 32
MAX_REPEATS = 10
MAX_ESTIMATED_CANDIDATE_SAMPLE_EVALUATIONS = 500_000_000
ANALYZE_CACHE_ENV = "GAFIME_V1_ANALYZE_CACHE_SIZE"


@dataclass(frozen=True)
class ReportSnapshot:
    identities: tuple[tuple[str, str, tuple[int, ...]], ...]
    values: Mapping[str, Mapping[str, float]]
    rankings: Mapping[str, tuple[str, ...]]
    backend: Mapping[str, object]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(identity[0] for identity in self.identities)


class ParityAccumulator:
    def __init__(self, precision: str) -> None:
        self.precision = precision
        self.absolute_tolerance, self.relative_tolerance = _tolerances(precision)
        self.comparisons = 0
        self.value_comparisons = 0
        self.max_absolute_delta = 0.0
        self.max_relative_delta = 0.0
        self.worst_absolute: dict[str, object] | None = None
        self.worst_relative: dict[str, object] | None = None

    def compare(
        self,
        actual: ReportSnapshot,
        expected: ReportSnapshot,
        metrics: Sequence[str],
        label: str,
    ) -> None:
        if actual.identities != expected.identities:
            raise AssertionError(f"{label}: candidate identities changed")
        for metric in metrics:
            if actual.rankings[metric] != expected.rankings[metric]:
                raise AssertionError(f"{label}/{metric}: visible-score ranking changed")
        self.comparisons += 1

        for candidate_id in expected.candidate_ids:
            for metric in metrics:
                expected_value = expected.values[candidate_id][metric]
                actual_value = actual.values[candidate_id][metric]
                absolute_delta = abs(actual_value - expected_value)
                scale = max(abs(actual_value), abs(expected_value), sys.float_info.min)
                relative_delta = absolute_delta / scale
                detail = {
                    "label": label,
                    "candidate_id": candidate_id,
                    "metric": metric,
                    "expected": expected_value,
                    "actual": actual_value,
                }
                if absolute_delta > self.max_absolute_delta:
                    self.max_absolute_delta = absolute_delta
                    self.worst_absolute = detail
                if relative_delta > self.max_relative_delta:
                    self.max_relative_delta = relative_delta
                    self.worst_relative = detail
                self.value_comparisons += 1
                if not math.isclose(
                    actual_value,
                    expected_value,
                    rel_tol=self.relative_tolerance,
                    abs_tol=self.absolute_tolerance,
                ):
                    raise AssertionError(
                        f"{label}/{candidate_id}/{metric}: {actual_value!r} != "
                        f"{expected_value!r} (abs_tol={self.absolute_tolerance}, "
                        f"rel_tol={self.relative_tolerance})"
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id_and_structure_parity": True,
            "visible_ranking_parity": True,
            "numeric_parity_within_tolerance": True,
            "report_comparisons": self.comparisons,
            "value_comparisons": self.value_comparisons,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "max_absolute_delta": self.max_absolute_delta,
            "max_relative_delta": self.max_relative_delta,
            "worst_absolute_delta": self.worst_absolute,
            "worst_relative_delta": self.worst_relative,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        metavar="NAME[,NAME...]",
        help=(
            "required backend(s): core, cuda, rocm, metal, or all; repeat or "
            "use comma-separated values (default: core)"
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        metavar="NAME[,NAME...]",
        help=(
            "precision profile(s): fp32, mixed, fp64, or all; repeat or use "
            "comma-separated values (default: every profile supported by each backend)"
        ),
    )
    parser.add_argument("--samples", type=int, default=2_048)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--output",
        default="-",
        metavar="PATH",
        help="JSON destination, or '-' for stdout (default: '-')",
    )
    arguments = parser.parse_args()

    if not 64 <= arguments.samples <= MAX_SAMPLES:
        parser.error(f"--samples must be between 64 and {MAX_SAMPLES}")
    if not 2 <= arguments.features <= MAX_FEATURES:
        parser.error(f"--features must be between 2 and {MAX_FEATURES}")
    if not 1 <= arguments.repeats <= MAX_REPEATS:
        parser.error(f"--repeats must be between 1 and {MAX_REPEATS}")
    if arguments.device_id < 0:
        parser.error("--device-id must be non-negative")

    arguments.backends = _selection(
        arguments.backends,
        allowed=BACKEND_ORDER,
        order=BACKEND_ORDER,
        default=("core",),
        label="backend",
        all_expansion=BACKEND_ORDER,
        parser=parser,
    )
    arguments.profiles = _profile_selection(arguments.profiles, parser)
    arguments.cases = _case_matrix(arguments.backends, arguments.profiles, parser)

    candidates = _expected_candidate_count(arguments.features)
    execution_calls = sum(
        (7 + int(backend in GRAPH_BACKENDS)) * (1 + arguments.repeats)
        for backend, _ in arguments.cases
    )
    estimated_work = candidates * arguments.samples * execution_calls
    if estimated_work > MAX_ESTIMATED_CANDIDATE_SAMPLE_EVALUATIONS:
        parser.error(
            "requested matrix exceeds the bounded-work limit: estimated "
            f"{estimated_work:,} candidate-sample evaluations > "
            f"{MAX_ESTIMATED_CANDIDATE_SAMPLE_EVALUATIONS:,}; reduce samples, "
            "features, repeats, backends, or profiles"
        )
    arguments.estimated_candidate_sample_evaluations = estimated_work
    return arguments


def _selection(
    raw_values: Sequence[str] | None,
    *,
    allowed: Sequence[str],
    order: Sequence[str],
    default: Sequence[str],
    label: str,
    all_expansion: Sequence[str],
    parser: argparse.ArgumentParser,
) -> tuple[str, ...]:
    if not raw_values:
        return tuple(default)
    parsed: list[str] = []
    for raw in raw_values:
        parsed.extend(part.strip().lower() for part in raw.split(",") if part.strip())
    if not parsed:
        parser.error(f"--{label} must contain at least one value")
    unknown = sorted(set(parsed) - set(allowed) - {"all"})
    if unknown:
        parser.error(
            f"unknown --{label} value(s): {', '.join(unknown)}; expected "
            f"{', '.join(allowed)} or all"
        )
    expanded = set(all_expansion if "all" in parsed else parsed)
    return tuple(value for value in order if value in expanded)


def _profile_selection(
    raw_values: Sequence[str] | None,
    parser: argparse.ArgumentParser,
) -> tuple[str, ...] | None:
    if not raw_values:
        return None
    parsed: list[str] = []
    for raw in raw_values:
        parsed.extend(part.strip().lower() for part in raw.split(",") if part.strip())
    if not parsed:
        parser.error("--profile must contain at least one value")
    unknown = sorted(set(parsed) - set(PROFILE_ORDER) - {"all"})
    if unknown:
        parser.error(
            f"unknown --profile value(s): {', '.join(unknown)}; expected "
            f"{', '.join(PROFILE_ORDER)} or all"
        )
    if "all" in parsed:
        if len(set(parsed)) != 1:
            parser.error("--profile all cannot be combined with named profiles")
        return None
    selected = set(parsed)
    return tuple(profile for profile in PROFILE_ORDER if profile in selected)


def _case_matrix(
    backends: Sequence[str],
    profiles: Sequence[str] | None,
    parser: argparse.ArgumentParser,
) -> tuple[tuple[str, str], ...]:
    cases: list[tuple[str, str]] = []
    for backend in backends:
        supported = PROFILE_MATRIX[backend]
        selected = supported if profiles is None else profiles
        unsupported = tuple(profile for profile in selected if profile not in supported)
        if unsupported:
            parser.error(
                f"backend={backend} does not support explicitly requested profile(s) "
                f"{', '.join(unsupported)}; Metal supports fp32 only"
            )
        cases.extend((backend, profile) for profile in selected)
    return tuple(cases)


def _expected_candidate_count(features: int) -> int:
    return features + math.comb(features, 2)


def _make_dataset(
    samples: int,
    features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((samples, features)).astype(np.float64)
    rows = np.arange(samples, dtype=np.float64)
    for column in range(features):
        matrix[:, column] += 0.20 * np.sin(
            rows * (column + 1) * 0.0031
        ) + 0.07 * np.cos(rows * (column + 2) * 0.0017)
    weights = np.linspace(0.8, -0.25, features, dtype=np.float64)
    target = matrix @ weights
    target += 0.35 * matrix[:, 0] * matrix[:, 1]
    target += 0.08 * np.sin(matrix[:, 0] * 1.7)
    target += rng.standard_normal(samples) * 0.03
    return (
        np.ascontiguousarray(matrix),
        np.ascontiguousarray(target),
        tuple(f"x{column}" for column in range(features)),
    )


def _config(
    backend: str,
    precision: str,
    metrics: Sequence[str],
    *,
    features: int,
    seed: int,
    device_id: int,
) -> EngineConfig:
    return EngineConfig(
        backend=backend,
        device_id=device_id,
        precision=precision,
        metric_names=tuple(metrics),
        num_repeats=1,
        permutation_tests=0,
        random_seed=seed,
        mi_bins=8,
        mi_approximate=True,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=max(features, math.comb(features, 2)),
            top_features_for_higher_k=features,
            max_generated_features=0,
            keep_in_vram=True,
            vram_budget_mb=2_048,
            max_feature_candidate=features,
        ),
    )


def _probe_backend(backend: str, precision: str, device_id: int):
    try:
        capabilities = gafime.backend_capabilities(
            backend,
            device_id,
            probe=True,
            precision=precision,
            mi_bins=8,
            mi_approximate=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"requested backend={backend!r} precision={precision!r} is unavailable: {exc}"
        ) from exc
    if capabilities.selection_status != "available":
        raise RuntimeError(
            f"requested backend={backend!r} precision={precision!r} is unavailable: "
            f"status={capabilities.selection_status!r}, "
            f"detail={capabilities.selection_detail!r}"
        )
    if capabilities.selected_backend != backend:
        raise RuntimeError(
            f"requested backend={backend!r} resolved to "
            f"{capabilities.selected_backend!r}; explicit fallback is forbidden"
        )
    contract = capabilities.precision_contract.value
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            f"{backend}/{precision}: precision capability is not structured"
        )
    if contract.get("request_supported") is not True:
        raise RuntimeError(f"{backend}/{precision}: capability rejected the profile")
    if contract.get("effective") != precision:
        raise RuntimeError(
            f"{backend}/{precision}: effective precision is {contract.get('effective')!r}"
        )
    advertised = tuple(contract.get("supported_profiles", ()))
    if advertised != PROFILE_MATRIX[backend]:
        raise RuntimeError(
            f"{backend}/{precision}: advertised profiles {advertised!r} do not match "
            f"the distributed contract {PROFILE_MATRIX[backend]!r}"
        )
    graph_support = capabilities.graph_support.value
    graph_supported = (
        graph_support
        if isinstance(graph_support, bool)
        else (
            graph_support.get("supported") is True
            if isinstance(graph_support, Mapping)
            else False
        )
    )
    if backend in GRAPH_BACKENDS and graph_supported is not True:
        raise RuntimeError(
            f"requested backend={backend!r} does not advertise required graph support: "
            f"{graph_support!r}"
        )
    return capabilities


def _snapshot_report(
    report: object,
    *,
    backend: str,
    precision: str,
    metrics: Sequence[str],
    expected_candidates: int,
) -> ReportSnapshot:
    backend_info = getattr(report, "backend", None)
    if backend_info is None:
        raise AssertionError(
            f"{backend}/{precision}: report omitted backend diagnostics"
        )
    if getattr(backend_info, "selected_backend", None) != backend:
        raise AssertionError(
            f"{backend}/{precision}: report selected "
            f"{getattr(backend_info, 'selected_backend', None)!r}"
        )
    if getattr(backend_info, "requested_precision", None) != precision:
        raise AssertionError(f"{backend}/{precision}: report lost requested precision")
    if getattr(backend_info, "effective_precision", None) != precision:
        raise AssertionError(
            f"{backend}/{precision}: report changed effective precision"
        )

    interactions = list(getattr(report, "interactions"))
    if len(interactions) != expected_candidates:
        raise AssertionError(
            f"{backend}/{precision}: expected {expected_candidates} candidates, "
            f"received {len(interactions)}"
        )
    identities: list[tuple[str, str, tuple[int, ...]]] = []
    values: dict[str, dict[str, float]] = {}
    for item in interactions:
        candidate_id = str(item.candidate_id)
        identity = (
            candidate_id,
            str(item.family),
            tuple(int(index) for index in item.combo),
        )
        if candidate_id in values:
            raise AssertionError(f"duplicate candidate id {candidate_id!r}")
        item_values: dict[str, float] = {}
        for metric in metrics:
            value = float(item.metrics[metric])
            if not math.isfinite(value):
                raise AssertionError(
                    f"{backend}/{precision}/{candidate_id}/{metric}: non-finite result"
                )
            item_values[metric] = value
        identities.append(identity)
        values[candidate_id] = item_values

    rankings = {
        metric: tuple(
            str(item.candidate_id)
            for item in report.interactions.ranked(
                metric_name=metric,
                descending=True,
            )
        )
        for metric in metrics
    }
    backend_dict = (
        asdict(backend_info)
        if is_dataclass(backend_info)
        else {
            "selected_backend": getattr(backend_info, "selected_backend", None),
            "device": getattr(backend_info, "device", None),
            "memory_total_mb": getattr(backend_info, "memory_total_mb", None),
            "memory_free_mb": getattr(backend_info, "memory_free_mb", None),
            "requested_precision": getattr(backend_info, "requested_precision", None),
            "effective_precision": getattr(backend_info, "effective_precision", None),
        }
    )
    return ReportSnapshot(tuple(identities), values, rankings, backend_dict)


def _tolerances(precision: str) -> tuple[float, float]:
    if precision == "fp32":
        return 2.0e-5, 2.0e-6
    return 2.0e-10, 2.0e-11


def _timed(function: Callable[[], object]) -> tuple[object, int]:
    start = perf_counter_ns()
    result = function()
    return result, perf_counter_ns() - start


def _timing_point(duration_ns: int, samples: int, candidates: int) -> dict[str, object]:
    return {
        "duration_ns": duration_ns,
        "duration_ms": duration_ns / 1.0e6,
        "sample_rows_per_second": samples * 1.0e9 / duration_ns,
        "candidate_results_per_second": candidates * 1.0e9 / duration_ns,
        "candidate_sample_evaluations_per_second": (
            samples * candidates * 1.0e9 / duration_ns
        ),
    }


def _timing_distribution(
    durations_ns: Sequence[int],
    samples: int,
    candidates: int,
) -> dict[str, object]:
    median_ns = float(statistics.median(durations_ns))
    return {
        "repeats": len(durations_ns),
        "raw_duration_ns": list(durations_ns),
        "median_ns": median_ns,
        "median_ms": median_ns / 1.0e6,
        "min_ns": min(durations_ns),
        "max_ns": max(durations_ns),
        "mean_ns": statistics.fmean(durations_ns),
        "median_sample_rows_per_second": samples * 1.0e9 / median_ns,
        "median_candidate_results_per_second": candidates * 1.0e9 / median_ns,
        "median_candidate_sample_evaluations_per_second": (
            samples * candidates * 1.0e9 / median_ns
        ),
    }


def _rss_snapshot() -> dict[str, object]:
    if sys.platform.startswith("linux"):
        try:
            fields: dict[str, int] = {}
            for line in Path("/proc/self/status").read_text().splitlines():
                name, separator, raw = line.partition(":")
                if separator and name in {"VmRSS", "VmHWM"}:
                    fields[name] = int(raw.strip().split()[0]) * 1_024
            return {
                "source": "/proc/self/status",
                "current_rss_bytes": fields.get("VmRSS"),
                "peak_rss_bytes": fields.get("VmHWM"),
                "scope": "current process; peak is cumulative for the process lifetime",
            }
        except (OSError, ValueError, IndexError):
            pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            success = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if success:
                return {
                    "source": "GetProcessMemoryInfo",
                    "current_rss_bytes": int(counters.WorkingSetSize),
                    "peak_rss_bytes": int(counters.PeakWorkingSetSize),
                    "scope": "current process; peak is cumulative for the process lifetime",
                }
        except (AttributeError, OSError, ValueError):
            pass
    try:
        import resource

        high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_bytes = high_water if sys.platform == "darwin" else high_water * 1_024
        return {
            "source": "getrusage(RUSAGE_SELF).ru_maxrss",
            "current_rss_bytes": None,
            "peak_rss_bytes": peak_bytes,
            "scope": "current process; peak is cumulative for the process lifetime",
        }
    except (ImportError, OSError, ValueError):
        return {
            "source": "unavailable",
            "current_rss_bytes": None,
            "peak_rss_bytes": None,
            "scope": "no portable process RSS source was available",
        }


def _memory_evidence(
    before: Mapping[str, object],
    after_cold: Mapping[str, object],
    after_warm: Mapping[str, object],
    cold_report: ReportSnapshot,
    warm_report: ReportSnapshot,
) -> dict[str, object]:
    before_peak = before.get("peak_rss_bytes")
    after_peak = after_warm.get("peak_rss_bytes")
    peak_delta = None
    if isinstance(before_peak, int) and isinstance(after_peak, int):
        peak_delta = max(0, after_peak - before_peak)
    return {
        "process_rss": {
            "before": dict(before),
            "after_cold": dict(after_cold),
            "after_warm": dict(after_warm),
            "cumulative_peak_increase_bytes": peak_delta,
            "interpretation": (
                "The high-water mark is process-lifetime cumulative and the increase is "
                "a lower-bound observation, not an isolated allocation peak."
            ),
        },
        "device_memory_snapshots": {
            "source": "DiagnosticReport.backend",
            "cold": {
                "total_mb": cold_report.backend.get("memory_total_mb"),
                "free_mb": cold_report.backend.get("memory_free_mb"),
            },
            "warm": {
                "total_mb": warm_report.backend.get("memory_total_mb"),
                "free_mb": warm_report.backend.get("memory_free_mb"),
            },
            "interpretation": (
                "These are runtime snapshots reported by the backend, not a device "
                "allocation high-water mark."
            ),
        },
    }


@contextmanager
def _analyze_cache(capacity: int):
    from gafime import v1_adapter

    previous = os.environ.get(ANALYZE_CACHE_ENV)
    os.environ[ANALYZE_CACHE_ENV] = str(capacity)
    v1_adapter._clear_analyze_cache_for_tests()
    try:
        yield
    finally:
        v1_adapter._clear_analyze_cache_for_tests()
        if previous is None:
            os.environ.pop(ANALYZE_CACHE_ENV, None)
        else:
            os.environ[ANALYZE_CACHE_ENV] = previous


def _measure_eager(
    *,
    label: str,
    cache_capacity: int,
    config: EngineConfig,
    matrix: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    repeats: int,
    backend: str,
    precision: str,
    expected_candidates: int,
    parity: ParityAccumulator,
    baseline: ReportSnapshot | None,
) -> tuple[dict[str, object], ReportSnapshot]:
    engine = GafimeEngine(config)
    gc.collect()
    with _analyze_cache(cache_capacity):
        memory_before = _rss_snapshot()
        cold_report, cold_ns = _timed(
            lambda: engine.analyze(matrix, target, feature_names=names)
        )
        cold_snapshot = _snapshot_report(
            cold_report,
            backend=backend,
            precision=precision,
            metrics=ALL_METRICS,
            expected_candidates=expected_candidates,
        )
        if baseline is None:
            baseline = cold_snapshot
        else:
            parity.compare(cold_snapshot, baseline, ALL_METRICS, f"{label}/cold")
        memory_after_cold = _rss_snapshot()

        warm_durations: list[int] = []
        warm_snapshot = cold_snapshot
        for repeat in range(repeats):
            warm_report, duration = _timed(
                lambda: engine.analyze(matrix, target, feature_names=names)
            )
            warm_snapshot = _snapshot_report(
                warm_report,
                backend=backend,
                precision=precision,
                metrics=ALL_METRICS,
                expected_candidates=expected_candidates,
            )
            parity.compare(
                warm_snapshot,
                baseline,
                ALL_METRICS,
                f"{label}/warm/{repeat}",
            )
            warm_durations.append(duration)
        memory_after_warm = _rss_snapshot()

    return (
        {
            "cache_capacity": cache_capacity,
            "cold": _timing_point(cold_ns, len(target), expected_candidates),
            "warm": _timing_distribution(
                warm_durations, len(target), expected_candidates
            ),
            "memory": _memory_evidence(
                memory_before,
                memory_after_cold,
                memory_after_warm,
                cold_snapshot,
                warm_snapshot,
            ),
        },
        baseline,
    )


def _measure_compiled(
    *,
    label: str,
    config: EngineConfig,
    metrics: Sequence[str],
    matrix: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    repeats: int,
    backend: str,
    precision: str,
    expected_candidates: int,
    parity: ParityAccumulator,
    baseline: ReportSnapshot,
    graph: bool,
) -> dict[str, object]:
    gc.collect()
    memory_before = _rss_snapshot()
    compile_start = perf_counter_ns()
    artifact = gafime.compile(
        matrix,
        target,
        names,
        config=config,
        flags=CompileFlags(plan=True, graph=graph),
    )
    compile_ns = perf_counter_ns() - compile_start
    memory_after_compile = _rss_snapshot()
    try:
        cold_report, cold_ns = _timed(artifact.analyze)
        cold_snapshot = _snapshot_report(
            cold_report,
            backend=backend,
            precision=precision,
            metrics=metrics,
            expected_candidates=expected_candidates,
        )
        parity.compare(cold_snapshot, baseline, metrics, f"{label}/cold")
        if graph and artifact.graph_replayed is not True:
            raise AssertionError(f"{label}: graph replay was not confirmed")
        memory_after_cold = _rss_snapshot()

        warm_durations: list[int] = []
        warm_snapshot = cold_snapshot
        for repeat in range(repeats):
            warm_report, duration = _timed(artifact.analyze)
            warm_snapshot = _snapshot_report(
                warm_report,
                backend=backend,
                precision=precision,
                metrics=metrics,
                expected_candidates=expected_candidates,
            )
            parity.compare(
                warm_snapshot,
                baseline,
                metrics,
                f"{label}/warm/{repeat}",
            )
            if graph and artifact.graph_replayed is not True:
                raise AssertionError(
                    f"{label}/warm/{repeat}: graph replay was not confirmed"
                )
            warm_durations.append(duration)
        memory_after_warm = _rss_snapshot()
    finally:
        artifact.close()
    memory_after_close = _rss_snapshot()

    return {
        "graph_requested": graph,
        "graph_replay_confirmed": graph,
        "compile_ns": compile_ns,
        "compile_ms": compile_ns / 1.0e6,
        "compile_plus_cold_ns": compile_ns + cold_ns,
        "cold": _timing_point(cold_ns, len(target), expected_candidates),
        "warm": _timing_distribution(warm_durations, len(target), expected_candidates),
        "memory": {
            **_memory_evidence(
                memory_before,
                memory_after_cold,
                memory_after_warm,
                cold_snapshot,
                warm_snapshot,
            ),
            "process_rss_after_compile": memory_after_compile,
            "process_rss_after_close": memory_after_close,
        },
    }


def _candidate_identity(snapshot: ReportSnapshot) -> dict[str, object]:
    encoded = json.dumps(snapshot.identities, separators=(",", ":")).encode("utf-8")
    return {
        "count": len(snapshot.identities),
        "ids": list(snapshot.candidate_ids),
        "identity_sha256": hashlib.sha256(encoded).hexdigest(),
        "identity_fields": ["candidate_id", "family", "combo"],
    }


def _benchmark_case(
    backend: str,
    precision: str,
    arguments: argparse.Namespace,
    matrix: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
) -> dict[str, object]:
    print(f"benchmarking {backend}/{precision}", file=sys.stderr, flush=True)
    capabilities = _probe_backend(backend, precision, arguments.device_id)
    expected_candidates = _expected_candidate_count(arguments.features)
    parity = ParityAccumulator(precision)
    config = _config(
        backend,
        precision,
        ALL_METRICS,
        features=arguments.features,
        seed=arguments.seed,
        device_id=arguments.device_id,
    )

    one_shot, baseline = _measure_eager(
        label="one_shot_uncached",
        cache_capacity=0,
        config=config,
        matrix=matrix,
        target=target,
        names=names,
        repeats=arguments.repeats,
        backend=backend,
        precision=precision,
        expected_candidates=expected_candidates,
        parity=parity,
        baseline=None,
    )
    one_shot["warm_interpretation"] = (
        "Repeated stateless calls after the first process-warm execution; resident "
        "analyze caching is disabled, so every call owns a complete eager execution."
    )
    eager_resident, _ = _measure_eager(
        label="eager_resident",
        cache_capacity=1,
        config=config,
        matrix=matrix,
        target=target,
        names=names,
        repeats=arguments.repeats,
        backend=backend,
        precision=precision,
        expected_candidates=expected_candidates,
        parity=parity,
        baseline=baseline,
    )
    eager_resident["warm_interpretation"] = (
        "Repeated public GafimeEngine.analyze calls reuse the resident analyze-cache "
        "entry created by the cold call."
    )

    compiled_plain = _measure_compiled(
        label="compiled_plain",
        config=config,
        metrics=ALL_METRICS,
        matrix=matrix,
        target=target,
        names=names,
        repeats=arguments.repeats,
        backend=backend,
        precision=precision,
        expected_candidates=expected_candidates,
        parity=parity,
        baseline=baseline,
        graph=False,
    )
    graph_result = None
    if backend in GRAPH_BACKENDS:
        graph_result = _measure_compiled(
            label="compiled_graph",
            config=config,
            metrics=ALL_METRICS,
            matrix=matrix,
            target=target,
            names=names,
            repeats=arguments.repeats,
            backend=backend,
            precision=precision,
            expected_candidates=expected_candidates,
            parity=parity,
            baseline=baseline,
            graph=True,
        )

    metric_paths: dict[str, object] = {}
    for metric in ALL_METRICS:
        metric_config = _config(
            backend,
            precision,
            (metric,),
            features=arguments.features,
            seed=arguments.seed,
            device_id=arguments.device_id,
        )
        metric_paths[metric] = _measure_compiled(
            label=f"metric_critical_path/{metric}",
            config=metric_config,
            metrics=(metric,),
            matrix=matrix,
            target=target,
            names=names,
            repeats=arguments.repeats,
            backend=backend,
            precision=precision,
            expected_candidates=expected_candidates,
            parity=parity,
            baseline=baseline,
            graph=False,
        )

    capability_dict = capabilities.to_dict()
    return {
        "backend": backend,
        "precision": precision,
        "capability_probe": {
            "selection_status": capabilities.selection_status,
            "selected_backend": capabilities.selected_backend,
            "graph_support": capability_dict["graph_support"],
            "precision_contract": capability_dict["precision_contract"],
            "device": capability_dict["device"],
            "native_boundary": capability_dict["native_boundary"],
            "native_version": capability_dict["native_version"],
        },
        "report_backend": dict(baseline.backend),
        "candidate_identity": _candidate_identity(baseline),
        "parity": parity.to_dict(),
        "end_to_end": {
            "one_shot_uncached": one_shot,
            "eager_resident": eager_resident,
            "compiled_plain": compiled_plain,
            "compiled_graph": graph_result,
        },
        "metric_critical_paths": metric_paths,
    }


def _host_metadata() -> dict[str, object]:
    cpu_model = platform.processor() or None
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    distributions: dict[str, str | None] = {}
    for distribution in ("gafime", "gafime-cuda", "gafime-rocm"):
        try:
            distributions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            distributions[distribution] = None
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "gafime_module": str(Path(gafime.__file__).resolve()),
        "gafime_version": gafime.__version__,
        "installed_distributions": distributions,
    }


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write_payload(payload: Mapping[str, object], output: str) -> None:
    rendered = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(rendered)
        return
    destination = Path(output)
    destination.write_text(rendered)
    print(f"wrote {destination}", file=sys.stderr)


def main() -> int:
    arguments = _parse_args()
    matrix, target, names = _make_dataset(
        arguments.samples,
        arguments.features,
        arguments.seed,
    )
    results = [
        _benchmark_case(backend, precision, arguments, matrix, target, names)
        for backend, precision in arguments.cases
    ]
    payload = {
        "schema": "gafime.precision-profile-performance.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_invalid_for_comparison",
        "valid_for_performance_claims": False,
        "methodology": {
            "superseded_by": "tests/release_measure/perf_13_precision_profiles.py",
            "known_hazards": [
                "fixed fp32, mixed, fp64 order in one process",
                "capability probing and payload loading precede the cold timer",
                "three repeats are insufficient for comparative evidence",
                "one small float64-source workload conflates conversion and arithmetic",
                "public resident and compiled timings include boundary overhead",
            ],
            "claim_policy": (
                "Raw per-profile measurements only; this artifact makes no universal "
                "speedup claim."
            ),
            "capability_probe_timing": (
                "Each backend/profile is probed before timing. Cold timings include "
                "conversion, allocation/planning as appropriate, and execution, but not "
                "the initial native-library load performed by the probe."
            ),
            "workload": (
                "continuous family, unary and pairwise candidates, all four metrics, "
                "fixed 8-bin approximate MI, no bootstrap or permutation significance"
            ),
            "input_source_dtype": "float64",
            "memory": (
                "Process RSS uses an OS high-water source and is cumulative. Device "
                "memory values are report snapshots, not allocation peaks."
            ),
            "graph_scope": "compiled graph replay is measured for CUDA and ROCm only",
        },
        "arguments": {
            "backends": list(arguments.backends),
            "profiles": (
                "all_supported_per_backend"
                if arguments.profiles is None
                else list(arguments.profiles)
            ),
            "cases": [list(case) for case in arguments.cases],
            "samples": arguments.samples,
            "features": arguments.features,
            "repeats": arguments.repeats,
            "seed": arguments.seed,
            "device_id": arguments.device_id,
            "estimated_candidate_sample_evaluations": (
                arguments.estimated_candidate_sample_evaluations
            ),
            "bounded_work_limit": MAX_ESTIMATED_CANDIDATE_SAMPLE_EVALUATIONS,
        },
        "host": _host_metadata(),
        "results": results,
    }
    _write_payload(payload, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
