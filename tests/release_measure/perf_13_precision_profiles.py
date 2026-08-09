#!/usr/bin/env python3
"""Release-grade precision-profile benchmark driver and isolated worker.

This replaces ``perf_12_precision_profiles.py`` for comparative evidence.  The
driver itself imports only the Python standard library.  Every cold sample and
every public benchmark trial runs in a fresh worker subprocess so importing this
file cannot pre-load GAFIME, NumPy, a vendor payload, or a GPU runtime.

Canonical full run (long-running)::

    python tests/release_measure/perf_13_precision_profiles.py \
        --backend core --backend cuda --backend rocm \
        --profile fp32,mixed,fp64 --workload release \
        --input-policy common-f64,native \
        --native-evidence /artifacts/native-evidence.json \
        --output precision-profile-perf-v2.json

Two-build randomized A/B and B/A run::

    python tests/release_measure/perf_13_precision_profiles.py \
        --variant baseline=/opt/gafime-baseline/bin/python \
        --variant candidate=/opt/gafime-candidate/bin/python \
        --wheel baseline=/artifacts/baseline.whl \
        --wheel candidate=/artifacts/candidate.whl \
        --source-root baseline=/src/baseline \
        --source-root candidate=/src/candidate \
        --native-evidence baseline=/artifacts/native-evidence-baseline.json \
        --native-evidence candidate=/artifacts/native-evidence-candidate.json \
        --backend cuda --workload release --output cuda-ab.json

The public layer deliberately measures public wall-clock surfaces.  It does not
mislabel report construction as device-kernel time.  Native arithmetic phase
timing requires the backend event/microbenchmark evidence collected separately;
this artifact records which cold sub-phases are combined by the public API.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from email.parser import Parser
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter_ns
from typing import Callable, Mapping, Sequence
import zipfile


SCHEMA = "gafime.precision-profile-performance.v2"
WORKER_SCHEMA = "gafime.precision-profile-performance.worker.v2"
NATIVE_EVIDENCE_SCHEMA = "gafime.precision-profile-native-evidence.v1"
NATIVE_EVIDENCE_KINDS = frozenset(
    {
        "native_decomposition",
        "device_events",
        "core_microbenchmark",
        "cuda_events",
        "rocm_events",
        "metal_events",
    }
)
# Native evidence is deliberately stricter than a file/hash allow-list.  Each
# backend owns a machine-readable schema and a complete decomposition contract;
# a manifest entry that merely points at a file containing a plausible SHA-256
# is not timing evidence.
NATIVE_ARTIFACT_SCHEMAS = {
    "core_microbenchmark": {
        "core": frozenset(("gafime.core-native-arithmetic.v2",)),
    },
    "cuda_events": {
        "cuda": frozenset(("gafime.cuda.native_timing.v2",)),
    },
    "rocm_events": {
        "rocm": frozenset(
            ("gafime.rocm.native_timing.v2", "gafime-rocm-native-timing-v2")
        ),
    },
    # Metal v1 was published with the precision-profile branch and remains an
    # accepted backend-specific schema until the next Metal helper revision.
    "metal_events": {
        "metal": frozenset(("gafime.metal.native_timing.v1",)),
    },
    # A canonical lifecycle artifact may be emitted by a separate stable-ABI
    # consumer.  It is backend tagged in the manifest and still must declare
    # the same complete operation/profile coverage below.
    "native_decomposition": {
        "core": frozenset(("gafime.native-decomposition.v1",)),
        "cuda": frozenset(("gafime.native-decomposition.v1",)),
        "rocm": frozenset(("gafime.native-decomposition.v1",)),
        "metal": frozenset(("gafime.native-decomposition.v1",)),
    },
    "device_events": {
        "cuda": frozenset(("gafime.cuda.native_timing.v2",)),
        "rocm": frozenset(
            ("gafime.rocm.native_timing.v2", "gafime-rocm-native-timing-v2")
        ),
        "metal": frozenset(("gafime.metal.native_timing.v1",)),
    },
}
NATIVE_REQUIRED_OPERATIONS = frozenset(
    {
        "ingest_conversion",
        "planning",
        "allocation",
        "h2d_upload",
        "candidate_materialization",
        "metric:pearson",
        "metric:spearman",
        "metric:mutual_info",
        "metric:r2",
        "ranking_target_ranks",
        "ranking_topk",
        "selected_row_gather",
        "d2h_transfer",
        "report_construction",
    }
)
NATIVE_REQUIRED_OPERATIONS_BY_BACKEND = {
    # Core's native artifact is an arithmetic microbenchmark, not a GPU
    # transfer/lifecycle or public-report trace. It covers every metric/profile;
    # report construction remains explicitly outside this native boundary.
    "core": frozenset(
        {
            "metric:pearson",
            "metric:spearman",
            "metric:mutual_info",
            "metric:r2",
        }
    ),
    "cuda": NATIVE_REQUIRED_OPERATIONS,
    "rocm": NATIVE_REQUIRED_OPERATIONS,
    # Metal records native fp32 source input conversion as not present, while
    # its helper records the remaining host/device lifecycle explicitly.
    "metal": NATIVE_REQUIRED_OPERATIONS - frozenset(("ingest_conversion",)),
}
NATIVE_REQUIRED_PROVENANCE = frozenset(
    {
        "benchmark_source",
        "benchmark_binary",
        "payload",
        "python_executable",
        "wheel",
    }
)
CANONICAL_ABI_LIFECYCLE_OPERATIONS = frozenset(
    {
        "numeric_routes",
        "matrix_alloc",
        "matrix_upload",
        "matrix_update_target",
        "execute",
        "execution_memory_peak",
        "permutation_memory_peak",
        "permutation_pvalues",
        "interaction_diagnostics",
        "matrix_free",
    }
)
CANONICAL_ABI_BACKEND_KINDS = {"cuda": 2, "rocm": 3, "metal": 4}
CANONICAL_ABI_WHEEL_MEMBERS = {
    "cuda": frozenset(
        ("gafime_cuda/libgafime_cuda.so", "gafime_cuda/gafime_cuda.dll")
    ),
    "rocm": frozenset(("gafime_rocm/libgafime_rocm.so",)),
    "metal": frozenset(("gafime/_metal/libgafime_metal_v1.dylib",)),
}
# These are the generic ABI 1.1 entry points exercised by the cold lifecycle
# and canonical native lanes.  Older PR-70 payloads exported only
# dtype-suffixed helpers (for example ``*_f32_v2``/``*_f64_v2``); those
# payloads cannot be used as the baseline for a generic-route native A/B.
CANONICAL_ABI_GENERIC_SYMBOLS = frozenset(
    {
        "gafime_gpu_numeric_routes_v2",
        "gafime_gpu_matrix_alloc_v2",
        "gafime_gpu_matrix_upload_v2",
        "gafime_gpu_matrix_update_target_v2",
        "gafime_gpu_execute_v2",
        "gafime_gpu_execution_memory_peak_v2",
        "gafime_gpu_permutation_memory_peak_v2",
        "gafime_gpu_permutation_pvalues_v2",
        "gafime_gpu_interaction_diagnostics_v2",
        "gafime_gpu_matrix_free_v2",
    }
)
NATIVE_REQUIRED_PROVENANCE_BY_BACKEND = {
    "core": frozenset(
        ("benchmark_source", "benchmark_binary", "python_executable", "wheel")
    ),
    "cuda": NATIVE_REQUIRED_PROVENANCE,
    "rocm": NATIVE_REQUIRED_PROVENANCE,
    "metal": frozenset(
        {
            "benchmark_source",
            "benchmark_binary",
            "payload",
            "python_executable",
            "wheel",
        }
    ),
}
NATIVE_DEVICE_TIMED_OPERATIONS_BY_BACKEND = {
    "cuda": frozenset(
        {
            "candidate_materialization",
            "ranking_target_ranks",
            "ranking_topk",
            "selected_row_gather",
            "metric:pearson",
            "metric:spearman",
            "metric:mutual_info",
            "metric:r2",
        }
    ),
    "rocm": frozenset(
        {
            "ranking_target_ranks",
            "ranking_topk",
            "selected_row_gather",
            "metric:pearson",
            "metric:spearman",
            "metric:mutual_info",
            "metric:r2",
        }
    ),
}
PROFILE_ORDER = ("fp32", "mixed", "fp64")
BACKEND_ORDER = ("core", "cuda", "rocm", "metal")
BACKEND_PROFILES = {
    "core": PROFILE_ORDER,
    "cuda": PROFILE_ORDER,
    "rocm": PROFILE_ORDER,
    "metal": ("fp32",),
}
GRAPH_BACKENDS = frozenset(("cuda", "rocm"))
SURFACES = ("one_shot", "resident", "compiled", "graph")
INPUT_POLICIES = ("common-f64", "native")
ALL_METRICS = ("pearson", "spearman", "mutual_info", "r2")
MIN_WARMUPS = 10
MIN_REPETITIONS = 30
DEFAULT_MIN_SAMPLE_NS = 100_000_000
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
BENCHMARK_RUNTIME_DISTRIBUTIONS = ("numpy", "polars")
RELEVANT_ENV_KEYS = (
    "GAFIME_CUDA_V1_LIB",
    "GAFIME_ROCM_V1_LIB",
    "GAFIME_METAL_V1_LIB",
    "GAFIME_METAL_V1_METALLIB",
    "GAFIME_V1_PY_MODULE",
    "GAFIME_V1_ANALYZE_CACHE_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
)
NATIVE_DIRECT_PATH_ENV_KEYS = frozenset(
    {
        "GAFIME_WHEEL_PATH",
        "GAFIME_NATIVE_BENCH_WHEEL",
        "GAFIME_PAYLOAD_PATH",
        "GAFIME_CUDA_V1_LIB",
        "GAFIME_ROCM_V1_LIB",
        "GAFIME_METAL_V1_LIB",
        "GAFIME_METAL_V1_METALLIB",
        "GAFIME_V1_PY_MODULE",
        "VIRTUAL_ENV",
    }
)
NATIVE_SEARCH_PATH_ENV_KEYS = frozenset(
    {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}
)
COLD_PHASES = (
    "python_import",
    "payload_discovery",
    "dynamic_library_load",
    "runtime_context_initialization",
    "code_object_or_module_registration",
    "first_capability_query",
    "first_allocation",
    "first_upload",
    "planning",
    "first_execution",
    "first_result_materialization",
    "explicit_cleanup",
    "process_exit_cleanup",
)


@dataclass(frozen=True)
class Workload:
    name: str
    workload_class: str
    samples: int
    features: int
    max_arity: int
    metrics: tuple[str, ...]
    mi_bins: int = 16

    @property
    def expected_candidates(self) -> int:
        return sum(
            math.comb(self.features, arity) for arity in range(1, self.max_arity + 1)
        )


WORKLOADS = {
    "small-latency": Workload("small-latency", "latency", 2_048, 8, 2, ALL_METRICS, 8),
    "medium-mixed": Workload(
        "medium-mixed", "mixed-overhead", 16_384, 12, 2, ALL_METRICS, 16
    ),
    "large-kernel": Workload(
        "large-kernel", "kernel-dominant", 65_536, 16, 2, ALL_METRICS, 32
    ),
    "metric-pearson": Workload(
        "metric-pearson", "metric-specific", 131_072, 8, 1, ("pearson",), 8
    ),
    "metric-spearman": Workload(
        "metric-spearman", "metric-specific", 65_536, 8, 1, ("spearman",), 8
    ),
    "metric-mi": Workload(
        "metric-mi", "metric-specific", 65_536, 8, 1, ("mutual_info",), 32
    ),
    "metric-r2": Workload("metric-r2", "metric-specific", 131_072, 8, 1, ("r2",), 8),
    "all-metrics": Workload(
        "all-metrics", "all-metrics", 32_768, 12, 2, ALL_METRICS, 24
    ),
    # The public continuous-family planner accepts arities through five.  Keep
    # these small and explicit so the release matrix exercises that supported
    # surface without pretending that generated families have a native GPU
    # precision benchmark (generated-family timing remains out of scope here).
    "arity-3": Workload("arity-3", "continuous-arity-3", 8_192, 6, 3, ALL_METRICS, 16),
    "arity-4": Workload("arity-4", "continuous-arity-4", 8_192, 6, 4, ALL_METRICS, 16),
    "arity-5": Workload("arity-5", "continuous-arity-5", 8_192, 6, 5, ALL_METRICS, 16),
}
RELEASE_WORKLOADS = tuple(WORKLOADS)


@dataclass(frozen=True)
class Variant:
    name: str
    python: str
    source_root: str | None
    wheels: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", action="append", dest="backends")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--surface", action="append", dest="surfaces")
    parser.add_argument("--input-policy", action="append", dest="input_policies")
    parser.add_argument("--workload", action="append", dest="workloads")
    parser.add_argument("--layer", choices=("all", "cold", "public"), default="all")
    parser.add_argument("--warmups", type=int, default=MIN_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=MIN_REPETITIONS)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--min-sample-ms", type=float, default=100.0)
    parser.add_argument("--max-loop-count", type=int, default=1_024)
    parser.add_argument(
        "--no-interleaved-control",
        action="store_false",
        dest="interleaved_control",
        help="disable the additional warmed, balanced profile-order control",
    )
    parser.set_defaults(interleaved_control=True)
    parser.add_argument(
        "--order-repetitions",
        type=int,
        default=1,
        help="repeat the six profile-order blocks this many times",
    )
    parser.add_argument(
        "--ab-blocks",
        type=int,
        default=2,
        help="with two variants, run both randomized A/B and B/A blocks",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    parser.add_argument(
        "--variant",
        action="append",
        metavar="NAME=PYTHON",
        help="isolated Python environment; repeat twice for A/B (default: current)",
    )
    parser.add_argument("--source-root", action="append", metavar="NAME=PATH")
    parser.add_argument(
        "--wheel",
        action="append",
        metavar="NAME=PATH",
        help="wheel bound to a variant; repeat the same name for Core plus payload",
    )
    parser.add_argument(
        "--native-evidence",
        action="append",
        metavar="PATH|NAME=PATH",
        help=(
            "required JSON manifest binding validated native arithmetic/device "
            "artifacts; with two variants provide one NAME=PATH manifest per "
            "variant; use status=not_collected for an E2E-only run"
        ),
    )
    parser.add_argument("--output", default="-", metavar="PATH")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        return args
    if args.self_check:
        return args
    if not args.native_evidence:
        parser.error("--native-evidence is required for a benchmark run")
    if args.warmups < MIN_WARMUPS:
        parser.error(f"--warmups must be at least {MIN_WARMUPS}")
    if args.repetitions < MIN_REPETITIONS:
        parser.error(f"--repetitions must be at least {MIN_REPETITIONS}")
    if args.bootstrap_resamples < 500:
        parser.error("--bootstrap-resamples must be at least 500")
    if not math.isfinite(args.min_sample_ms) or args.min_sample_ms <= 0.0:
        parser.error("--min-sample-ms must be positive")
    configured_min_sample_ns = int(args.min_sample_ms * 1.0e6)
    if configured_min_sample_ns < DEFAULT_MIN_SAMPLE_NS:
        parser.error(
            "--min-sample-ms must be at least "
            f"{DEFAULT_MIN_SAMPLE_NS / 1.0e6:g} ms for release claims"
        )
    if args.max_loop_count < 1:
        parser.error("--max-loop-count must be positive")
    if args.order_repetitions < 1 or args.ab_blocks < 1:
        parser.error("--order-repetitions and --ab-blocks must be positive")
    if args.device_id < 0 or args.timeout_seconds < 1:
        parser.error("--device-id must be non-negative and timeout positive")

    args.backends = _ordered_selection(
        args.backends, BACKEND_ORDER, ("core",), "backend", parser
    )
    args.profiles = _ordered_selection(
        args.profiles, PROFILE_ORDER, PROFILE_ORDER, "profile", parser
    )
    args.surfaces = _ordered_selection(
        args.surfaces, SURFACES, SURFACES, "surface", parser
    )
    args.input_policies = _ordered_selection(
        args.input_policies,
        INPUT_POLICIES,
        INPUT_POLICIES,
        "input-policy",
        parser,
    )
    args.workloads = _workload_selection(args.workloads, parser)
    source_roots = _named_values(args.source_root, "source-root", parser)
    wheels = _named_multi_values(args.wheel, "wheel", parser)
    raw_variants = args.variant or [f"current={sys.executable}"]
    parsed_variants = _named_values(raw_variants, "variant", parser)
    if args.variant is None:
        source_roots.setdefault("current", str(Path(__file__).resolve().parents[2]))
    if len(parsed_variants) > 2:
        parser.error("at most two --variant values are supported")
    unknown_sources = set(source_roots) - set(parsed_variants)
    unknown_wheels = set(wheels) - set(parsed_variants)
    if unknown_sources or unknown_wheels:
        parser.error("--source-root/--wheel names must match a --variant name")
    args.variants = tuple(
        Variant(
            name,
            os.path.abspath(os.path.expanduser(python)),
            (
                str(Path(source_roots[name]).expanduser().resolve())
                if name in source_roots
                else None
            ),
            (
                tuple(
                    str(Path(path).expanduser().resolve())
                    for path in wheels.get(name, ())
                )
            ),
        )
        for name, python in parsed_variants.items()
    )
    for variant in args.variants:
        if not Path(variant.python).is_file():
            parser.error(
                f"variant {variant.name!r} Python does not exist: {variant.python}"
            )
        if variant.source_root and not Path(variant.source_root).is_dir():
            parser.error(
                f"variant {variant.name!r} source root does not exist: {variant.source_root}"
            )
        for wheel in variant.wheels:
            if not Path(wheel).is_file():
                parser.error(f"variant {variant.name!r} wheel does not exist: {wheel}")
    native_evidence_specs: list[tuple[str | None, str]] = []
    for raw_spec in args.native_evidence:
        name, separator, raw_path = raw_spec.partition("=")
        if separator:
            name = name.strip()
            raw_path = raw_path.strip()
            if not name or not raw_path or name not in {v.name for v in args.variants}:
                parser.error(
                    "--native-evidence NAME=PATH must name one of the configured variants"
                )
            native_evidence_specs.append(
                (name, str(Path(raw_path).expanduser().resolve()))
            )
        else:
            raw_path = raw_spec.strip()
            if not raw_path:
                parser.error("--native-evidence PATH must not be empty")
            native_evidence_specs.append(
                (None, str(Path(raw_path).expanduser().resolve()))
            )
    variant_names = {variant.name for variant in args.variants}
    named_evidence_names = {name for name, _ in native_evidence_specs if name is not None}
    if len(args.variants) == 2:
        if len(native_evidence_specs) != 2 or any(
            name is None for name, _ in native_evidence_specs
        ) or named_evidence_names != variant_names:
            parser.error(
                "two-variant runs require exactly one --native-evidence NAME=PATH "
                "manifest for each variant"
            )
    if len(native_evidence_specs) > 1 and any(
        name is None for name, _ in native_evidence_specs
    ):
        parser.error("multiple --native-evidence manifests must all use NAME=PATH")
    for name, path in native_evidence_specs:
        if not Path(path).is_file():
            parser.error(f"native-evidence manifest does not exist: {path}")
    args.native_evidence = (
        native_evidence_specs[0][1]
        if len(native_evidence_specs) == 1 and native_evidence_specs[0][0] is None
        else native_evidence_specs
    )
    for backend in args.backends:
        unsupported = [
            profile
            for profile in args.profiles
            if profile not in BACKEND_PROFILES[backend]
        ]
        if unsupported:
            parser.error(
                f"backend={backend} does not support requested profile(s): "
                f"{', '.join(unsupported)}"
            )
    return args


def _split_values(raw_values: Sequence[str] | None) -> list[str]:
    values: list[str] = []
    for raw in raw_values or ():
        values.extend(part.strip().lower() for part in raw.split(",") if part.strip())
    return values


def _ordered_selection(
    raw_values: Sequence[str] | None,
    allowed: Sequence[str],
    default: Sequence[str],
    label: str,
    parser: argparse.ArgumentParser,
) -> tuple[str, ...]:
    values = _split_values(raw_values)
    if not values:
        return tuple(default)
    if "all" in values:
        if len(values) != 1:
            parser.error(f"--{label} all cannot be combined with named values")
        return tuple(allowed)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        parser.error(
            f"unknown --{label} value(s): {', '.join(unknown)}; expected "
            f"{', '.join(allowed)} or all"
        )
    if len(set(values)) != len(values):
        parser.error(f"--{label} contains duplicate values")
    return tuple(values)


def _workload_selection(
    raw_values: Sequence[str] | None, parser: argparse.ArgumentParser
) -> tuple[str, ...]:
    values = _split_values(raw_values)
    if not values or values == ["release"] or values == ["all"]:
        return RELEASE_WORKLOADS
    if "release" in values or "all" in values:
        parser.error("--workload release/all cannot be combined with named workloads")
    unknown = [value for value in values if value not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workload(s): {', '.join(unknown)}")
    if len(set(values)) != len(values):
        parser.error("--workload contains duplicate values")
    return tuple(values)


def _named_values(
    raw_values: Sequence[str] | None, label: str, parser: argparse.ArgumentParser
) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in raw_values or ():
        name, separator, value = raw.partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            parser.error(f"--{label} must use NAME=VALUE")
        if name in result:
            parser.error(f"duplicate --{label} name {name!r}")
        result[name] = value
    return result


def _named_multi_values(
    raw_values: Sequence[str] | None, label: str, parser: argparse.ArgumentParser
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw in raw_values or ():
        name, separator, value = raw.partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            parser.error(f"--{label} must use NAME=VALUE")
        result.setdefault(name, []).append(value)
    return result


def _profile_orders(profiles: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    # itertools preserves the caller's requested ordering; no set round-trip is
    # permitted here.  For all profiles this is exactly the six possible orders.
    if len(profiles) <= 1:
        return (tuple(profiles),)
    return tuple(itertools.permutations(tuple(profiles)))


def _workload_payload(workload: Workload) -> dict[str, object]:
    return {
        **asdict(workload),
        "expected_candidate_count": workload.expected_candidates,
    }


def _decode_workload(payload: Mapping[str, object]) -> Workload:
    return Workload(
        name=str(payload["name"]),
        workload_class=str(payload["workload_class"]),
        samples=int(payload["samples"]),
        features=int(payload["features"]),
        max_arity=int(payload["max_arity"]),
        metrics=tuple(str(metric) for metric in payload["metrics"]),
        mi_bins=int(payload["mi_bins"]),
    )


def _surface_order(block_index: int, surfaces: Sequence[str]) -> tuple[str, ...]:
    if not surfaces:
        return ()
    offset = block_index % len(surfaces)
    return tuple(surfaces[offset:]) + tuple(surfaces[:offset])


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(
    raw_region_ns: Sequence[int],
    loop_count: int,
    *,
    bootstrap_resamples: int,
    seed: int,
    samples: int,
    candidates: int,
    metrics: Sequence[str],
    sample_region_target_ns: int,
) -> dict[str, object]:
    if loop_count < 1 or not raw_region_ns or sample_region_target_ns < 1:
        raise ValueError(
            "timing distribution requires samples, a positive loop count, and target"
        )
    per_call = [float(value) / loop_count for value in raw_region_ns]
    median_ns = float(statistics.median(per_call))
    mad_ns = float(statistics.median(abs(value - median_ns) for value in per_call))
    rng = random.Random(seed)
    bootstrap_medians = [
        statistics.median(rng.choice(per_call) for _ in per_call)
        for _ in range(bootstrap_resamples)
    ]
    pair_count = samples * candidates
    return {
        "measured_repetitions": len(raw_region_ns),
        "loop_count_per_repetition": loop_count,
        "sample_region_target_ns": sample_region_target_ns,
        "sample_region_min_observed_ns": min(int(value) for value in raw_region_ns),
        "sample_region_target_met": all(
            int(value) >= sample_region_target_ns for value in raw_region_ns
        ),
        "raw_region_duration_ns": list(raw_region_ns),
        "raw_per_call_duration_ns": per_call,
        "median_ns": median_ns,
        "mad_ns": mad_ns,
        "p05_ns": _percentile(per_call, 0.05),
        "p95_ns": _percentile(per_call, 0.95),
        "bootstrap_median_95_ci_ns": [
            _percentile(bootstrap_medians, 0.025),
            _percentile(bootstrap_medians, 0.975),
        ],
        "primary_quantity": "public wall-clock latency per call",
        "metric_set": list(metrics),
        "metric_count": len(metrics),
        "candidate_count": candidates,
        "sample_count": samples,
        "candidate_sample_pairs_per_second_for_configured_metric_set": (
            pair_count * 1.0e9 / median_ns
        ),
        "rate_definition": (
            "samples times surfaced candidates per public call; the configured "
            "metric set is named explicitly and this is not generic GEval/s"
        ),
    }


def _native_record_statistics(
    values: Sequence[float],
    *,
    raw_values: Sequence[float] | None = None,
    seed: int,
    unit: str,
    auto_scaling: Mapping[str, object] | None = None,
) -> dict[str, object]:
    numeric = [float(value) for value in values]
    raw_numeric = [
        float(value) for value in (raw_values if raw_values is not None else values)
    ]
    center = float(statistics.median(numeric))
    mad = float(statistics.median(abs(value - center) for value in numeric))
    rng = random.Random(seed)
    bootstrap = [
        statistics.median(rng.choice(numeric) for _ in numeric)
        for _ in range(DEFAULT_BOOTSTRAP_RESAMPLES)
    ]
    return {
        "unit": unit,
        "statistics_scope": "normalized_per_call",
        "normalized_durations": list(numeric),
        "raw_durations": list(raw_numeric),
        "median": center,
        "mad": mad,
        "p05": _percentile(numeric, 0.05),
        "p95": _percentile(numeric, 0.95),
        "bootstrap_median_95_ci": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ],
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "auto_scaling": dict(
            auto_scaling
            or {"status": "not_observed_in_native_artifact"}
        ),
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _command_output(command: Sequence[str], timeout: int = 10) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"status": "unavailable", "command": list(command), "output": None}
    try:
        result = subprocess.run(
            list(command), text=True, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "command": list(command), "output": str(exc)}
    output = (result.stdout + result.stderr).strip()
    return {
        "status": "pass" if result.returncode == 0 else "error",
        "command": list(command),
        "returncode": result.returncode,
        "output": output[:16_384],
    }


def _git_commit(source_root: str | None) -> str | None:
    if source_root is None:
        return None
    result = _command_output(("git", "-C", source_root, "rev-parse", "HEAD"))
    if result.get("status") != "pass":
        return None
    return str(result["output"]).splitlines()[0].strip()


def _git_state(source_root: str | None) -> dict[str, object]:
    if source_root is None:
        return {"status": "not_supplied", "entries": []}
    result = _command_output(
        ("git", "-C", source_root, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if result.get("status") != "pass":
        return {
            "status": "unavailable",
            "entries": [],
            "detail": result.get("output"),
        }
    output = str(result.get("output") or "")
    entries = [line for line in output.splitlines() if line.strip()]
    return {
        "status": "clean" if not entries else "dirty",
        "entries": entries[:512],
        "entry_count": len(entries),
    }


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _file_identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    try:
        return {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }
    except OSError as exc:
        return {"path": str(resolved), "status": "error", "detail": str(exc)}


def _runtime_dependency_identities() -> dict[str, object]:
    from importlib import metadata

    result: dict[str, object] = {}
    for name in BENCHMARK_RUNTIME_DISTRIBUTIONS:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            result[name] = {"status": "missing"}
            continue
        record = next(
            (
                entry
                for entry in distribution.files or ()
                if str(entry).endswith(".dist-info/RECORD")
            ),
            None,
        )
        result[name] = {
            "status": "observed" if record is not None else "record_missing",
            "version": distribution.version,
            "record": (
                _file_identity(distribution.locate_file(record))
                if record is not None
                else None
            ),
        }
    return result


def _wheel_identity(path: str | Path) -> dict[str, object]:
    identity = _file_identity(path)
    identity["format"] = "wheel"
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_paths = sorted(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            if len(metadata_paths) != 1:
                raise ValueError(
                    f"expected one dist-info/METADATA, found {len(metadata_paths)}"
                )
            metadata = Parser().parsestr(
                archive.read(metadata_paths[0]).decode("utf-8", errors="strict")
            )
            name = metadata.get("Name")
            version = metadata.get("Version")
            if not name or not version:
                raise ValueError("wheel metadata is missing Name or Version")
            record_paths = sorted(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/RECORD")
            )
            record_hashes: dict[str, str] = {}
            if len(record_paths) == 1:
                for row in csv.reader(
                    archive.read(record_paths[0]).decode("utf-8", errors="strict").splitlines()
                ):
                    if len(row) < 2 or not row[1].startswith("sha256="):
                        continue
                    encoded = row[1].partition("=")[2]
                    padding = "=" * (-len(encoded) % 4)
                    record_hashes[row[0]] = base64.urlsafe_b64decode(
                        encoded + padding
                    ).hex()
            identity.update(
                {
                    "distribution": str(name),
                    "canonical_distribution": _canonical_distribution_name(str(name)),
                    "version": str(version),
                    "metadata_path": metadata_paths[0],
                    "record_hashes": record_hashes,
                }
            )
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile) as exc:
        identity.update({"status": "invalid", "detail": str(exc)})
    return identity


def _wheel_identity_summary(identity: Mapping[str, object]) -> dict[str, object]:
    summary = {
        str(key): value for key, value in identity.items() if key != "record_hashes"
    }
    record_hashes = identity.get("record_hashes")
    if isinstance(record_hashes, Mapping):
        summary["record_file_count"] = len(record_hashes)
    return summary


def _loaded_module_inventory() -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for module_name, module in tuple(sys.modules.items()):
        if not (
            module_name == "gafime"
            or module_name.startswith("gafime.")
            or module_name == "gafime_py"
        ):
            continue
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(str(raw)).resolve()
        if not path.is_file():
            continue
        identity = _file_identity(path)
        suffix = path.suffix.lower()
        identity.update(
            {
                "module": module_name,
                "kind": "native"
                if suffix in (".so", ".dylib", ".dll", ".pyd")
                else "python",
            }
        )
        records[str(path)] = identity
    return [records[path] for path in sorted(records)]


def _path_is_below(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _wheel_runtime_binding(
    wheels: Sequence[str],
    distributions: Mapping[str, object],
    loaded_modules: Sequence[Mapping[str, object]],
    native_binaries: Sequence[Mapping[str, object]],
    source_root: str | None,
    backend: str,
) -> dict[str, object]:
    identities = [_wheel_identity(path) for path in wheels]
    failures: list[str] = []
    artifact_by_name: dict[str, Mapping[str, object]] = {}
    for identity in identities:
        name = identity.get("canonical_distribution")
        if identity.get("status") == "invalid" or not name:
            failures.append(f"invalid_wheel:{identity.get('path')}")
            continue
        if name in artifact_by_name:
            failures.append(f"duplicate_wheel:{name}")
        artifact_by_name[name] = identity
        runtime = distributions.get(name)
        if not isinstance(runtime, Mapping):
            failures.append(f"wheel_not_installed:{name}")
        elif str(runtime.get("version")) != str(identity.get("version")):
            failures.append(
                f"wheel_version_mismatch:{name}:{identity.get('version')}!={runtime.get('version')}"
            )
    required_name = "gafime-" + backend if backend in ("cuda", "rocm") else "gafime"
    if required_name not in artifact_by_name:
        failures.append(f"required_wheel_not_declared:{required_name}")
    runtime_roots = [
        str(value.get("root"))
        for value in distributions.values()
        if isinstance(value, Mapping) and value.get("root")
    ]
    module_paths = [
        str(value.get("path"))
        for value in loaded_modules
        if value.get("path")
    ]
    module_under_installed_distribution = bool(module_paths) and all(
        any(_path_is_below(path, root) for root in runtime_roots)
        for path in module_paths
    )
    if not module_paths:
        failures.append("no_loaded_gafime_modules")
    elif not module_under_installed_distribution:
        failures.append("loaded_module_origin_not_installed_wheel")
    if source_root and any(_path_is_below(path, source_root) for path in module_paths):
        failures.append("loaded_module_origin_is_source_tree")
    module_wheel_bindings: list[dict[str, object]] = []
    for module in loaded_modules:
        module_path = module.get("path")
        if not module_path or not runtime_roots:
            continue
        relative = None
        for root in runtime_roots:
            try:
                relative = Path(module_path).resolve().relative_to(Path(root).resolve()).as_posix()
                break
            except ValueError:
                continue
        if relative is None:
            continue
        expected_hashes = {
            str(identity.get("record_hashes", {}).get(relative))
            for identity in identities
            if isinstance(identity.get("record_hashes"), Mapping)
            and relative in identity.get("record_hashes", {})
        }
        expected_hashes.discard("None")
        observed_hash = str(module.get("sha256"))
        binding = {
            "module": module.get("module"),
            "path": str(module_path),
            "wheel_relative_path": relative,
            "observed_sha256": observed_hash,
            "record_sha256": sorted(expected_hashes),
        }
        module_wheel_bindings.append(binding)
        if not expected_hashes:
            failures.append(f"loaded_module_missing_wheel_record:{relative}")
        elif observed_hash not in expected_hashes:
            failures.append(f"loaded_module_wheel_hash_mismatch:{relative}")
        if source_root and module.get("kind") == "python":
            source_candidates = (
                Path(source_root) / "python" / relative,
                Path(source_root) / relative,
            )
            source_file = next(
                (candidate for candidate in source_candidates if candidate.is_file()),
                None,
            )
            if source_file is None:
                failures.append(f"source_tree_module_missing:{relative}")
            elif _sha256(source_file) not in expected_hashes:
                failures.append(f"source_tree_wheel_module_mismatch:{relative}")
    for binary in native_binaries:
        binary_path = binary.get("path")
        observed_hash = str(binary.get("sha256"))
        if not binary_path:
            failures.append("native_binary_missing_path")
            continue
        relative = None
        for root in runtime_roots:
            try:
                relative = Path(str(binary_path)).resolve().relative_to(
                    Path(root).resolve()
                ).as_posix()
                break
            except ValueError:
                continue
        expected_hashes = {
            str(identity.get("record_hashes", {}).get(relative))
            for identity in identities
            if relative
            and isinstance(identity.get("record_hashes"), Mapping)
            and relative in identity.get("record_hashes", {})
        }
        if not relative:
            expected_hashes = {
                str(expected)
                for identity in identities
                for expected in (
                    identity.get("record_hashes", {}).values()
                    if isinstance(identity.get("record_hashes"), Mapping)
                    else ()
                )
            }
        expected_hashes.discard("None")
        if not expected_hashes or observed_hash not in expected_hashes:
            failures.append(f"native_binary_wheel_hash_mismatch:{binary_path}")
    return {
        "complete": not failures,
        "required_distribution": required_name,
        "wheel_identities": [_wheel_identity_summary(identity) for identity in identities],
        "loaded_module_under_installed_distribution": module_under_installed_distribution,
        "loaded_module_wheel_bindings": module_wheel_bindings,
        "failures": failures,
    }


def _cpu_governors() -> dict[str, object]:
    root = Path("/sys/devices/system/cpu/cpufreq")
    if not root.is_dir():
        return {"status": "unavailable", "values": []}
    try:
        values = sorted(
            {
                path.read_text().strip()
                for path in root.glob("policy*/scaling_governor")
                if path.is_file()
            }
        )
    except OSError as exc:
        return {"status": "error", "values": [], "detail": str(exc)}
    return {"status": "observed", "values": values}


def _clock_power_snapshot(backend: str) -> dict[str, object]:
    snapshot: dict[str, object] = {"cpu_governor": _cpu_governors()}
    if backend == "cuda":
        snapshot["nvidia_smi"] = _command_output(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            )
        )
    elif backend == "rocm":
        rocm_smi = _command_output(
            (
                "rocm-smi",
                "--showproductname",
                "--showdriverversion",
                "--showclocks",
                "--showpower",
                "--json",
            )
        )
        if (
            rocm_smi.get("status") == "pass"
            and _rocm_dynamic_telemetry_fields(rocm_smi)
        ):
            rocm_smi = dict(rocm_smi)
            rocm_smi["source"] = "rocm-smi"
            snapshot["rocm_smi"] = rocm_smi
        else:
            fallback = _amd_sysfs_snapshot(dynamic=True)
            if fallback.get("status") == "pass":
                fallback = dict(fallback)
                fallback["detail"] = (
                    "rocm-smi did not provide a nonempty dynamic clock or power "
                    "field; dynamic DRM sysfs fallback used"
                )
            snapshot["rocm_smi"] = fallback
    elif backend == "metal" and platform.system() == "Darwin":
        snapshot["system_profiler"] = _command_output(
            ("system_profiler", "SPDisplaysDataType", "-json")
        )
    return snapshot


def _amd_sysfs_snapshot(
    *, dynamic: bool, root: Path | None = None
) -> dict[str, object]:
    """Read stable AMD identity or dynamic clocks when ROCm SMI is absent."""

    root = Path("/sys/class/drm") if root is None else root
    cards: list[dict[str, object]] = []
    if not root.is_dir():
        return {"status": "unavailable", "output": "Linux DRM sysfs is unavailable"}
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device"
        try:
            if (device / "vendor").read_text().strip().lower() != "0x1002":
                continue
            record: dict[str, object] = {
                "card": card.name,
                "device": (device / "device").read_text().strip(),
                "uevent": sorted((device / "uevent").read_text().splitlines()),
            }
            if dynamic:
                for name in (
                    "pp_dpm_sclk",
                    "pp_dpm_mclk",
                    "power_dpm_state",
                    "gpu_busy_percent",
                ):
                    path = device / name
                    if path.is_file():
                        record[name] = path.read_text().strip()
                hwmon = device / "hwmon"
                if hwmon.is_dir():
                    power_values: dict[str, str] = {}
                    for path in sorted(hwmon.glob("hwmon*/power*_average")):
                        if path.is_file():
                            value = path.read_text().strip()
                            if value:
                                power_values[path.name] = value
                    if power_values:
                        record["power_average"] = power_values
            cards.append(record)
        except OSError:
            continue
    if not cards:
        return {"status": "unavailable", "output": "no AMD DRM device was readable"}
    output = json.dumps(cards, sort_keys=True, separators=(",", ":"))
    if dynamic and not _rocm_dynamic_telemetry_fields({"output": output}):
        return {
            "status": "unavailable",
            "output": output,
            "source": "linux_drm_sysfs",
            "detail": "AMD DRM identity was readable but no dynamic clock or power field was exposed",
        }
    return {
        "status": "pass",
        "output": output,
        "source": "linux_drm_sysfs",
    }


def _rocm_dynamic_telemetry_fields(observation: object) -> tuple[str, ...]:
    """Return nonempty dynamic clock/power fields from a ROCm observation.

    A successful command exit is not evidence by itself: some ``rocm-smi``
    versions return only product and driver identity even when clock/power
    queries are unsupported.  Both command JSON and the DRM fallback are
    inspected semantically before the snapshot can support a performance claim.
    """

    if not isinstance(observation, Mapping):
        return ()
    output = observation.get("output")
    if not isinstance(output, str) or not output.strip():
        return ()
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return ()

    found: set[str] = set()

    def observed_value(value: object) -> bool:
        if isinstance(value, bool) or value is None:
            return False
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if not isinstance(value, str):
            return False
        normalized = re.sub(r"\s+", " ", value).strip().lower()
        if not normalized:
            return False
        placeholders = (
            "n/a",
            "na",
            "unknown",
            "unsupported",
            "not supported",
            "not available",
            "unavailable",
            "none",
            "null",
            "nan",
            "--",
        )
        if normalized in placeholders or any(
            marker in normalized
            for marker in ("not supported", "not available", "unsupported")
        ):
            return False
        return re.search(r"(?:^|[^a-z])[-+]?\d", normalized) is not None

    def visit(value: object, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                visit(child, f"{prefix}.{key}" if prefix else key)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")
            return
        if not observed_value(value):
            return
        leaf = prefix.rsplit(".", 1)[-1].lower().replace("_", " ")
        is_clock = any(
            token in leaf
            for token in (
                "sclk",
                "mclk",
                "fclk",
                "socclk",
                "dcefclk",
                "vclk",
                "dclk",
                "clock speed",
                "current clock",
            )
        )
        is_power = "power" in leaf and any(
            token in leaf
            for token in ("average", "current", "draw", "consumption")
        )
        if is_clock or is_power:
            found.add(prefix)

    visit(parsed)
    return tuple(sorted(found))


def _device_identity_snapshot(backend: str) -> dict[str, object]:
    """Capture a stable device/driver identity separately from live clocks."""

    if backend == "cuda":
        return _command_output(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            )
        )
    if backend == "rocm":
        result = _command_output(
            (
                "rocm-smi",
                "--showproductname",
                "--showdriverversion",
                "--showuniqueid",
                "--json",
            )
        )
        return (
            result
            if result.get("status") == "pass"
            else _amd_sysfs_snapshot(dynamic=False)
        )
    if backend == "metal" and platform.system() == "Darwin":
        return _command_output(("system_profiler", "SPDisplaysDataType", "-json"))
    return {"status": "not_applicable", "output": "CPU backend"}


def _toolchain_snapshot() -> dict[str, object]:
    linker = _command_output(("ld", "--version"))
    if platform.system() == "Darwin" and linker.get("status") != "pass":
        linker = _command_output(("ld", "-v"))
    return {
        "rustc": _command_output(("rustc", "--version", "--verbose")),
        "cc": _command_output(("cc", "--version")),
        "cxx": _command_output(("c++", "--version")),
        "nvcc": _command_output(("nvcc", "--version")),
        "hipcc": _command_output(("hipcc", "--version")),
        "linker": linker,
    }


def _cpu_identity() -> str | None:
    identity = platform.processor().strip()
    if identity:
        return identity
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        result = _command_output(("sysctl", "-n", "machdep.cpu.brand_string"))
        if result.get("status") == "pass":
            return str(result["output"]).strip()
    return None


def _base_provenance(
    job: Mapping[str, object],
    *,
    clock_power_before: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from importlib import metadata

    source_root = job.get("source_root")
    wheels = tuple(str(path) for path in job.get("wheels", ()))
    affinity: dict[str, object] = {
        "status": "unavailable",
        "cpus": None,
        "detail": "os.sched_getaffinity is unavailable on this platform",
    }
    if hasattr(os, "sched_getaffinity"):
        affinity = {"status": "observed", "cpus": sorted(os.sched_getaffinity(0))}
    distributions: dict[str, object] = {}
    for name in ("gafime", "gafime-cuda", "gafime-rocm"):
        try:
            distribution = metadata.distribution(name)
            distributions[_canonical_distribution_name(name)] = {
                "version": distribution.version,
                "root": str(Path(distribution.locate_file("")).resolve()),
            }
        except metadata.PackageNotFoundError:
            distributions[_canonical_distribution_name(name)] = None
    loaded_modules = _loaded_module_inventory()
    native_binaries = _native_binary_inventory()
    source_tree_state = _git_state(str(source_root) if source_root else None)
    benchmark_script = _file_identity(__file__)
    # The worker is deliberately the one frozen perf13 harness script used by
    # both A/B environments.  ``source_root`` identifies the build under test;
    # it is not allowed to substitute a different benchmark implementation.
    benchmark_script_canonical = bool(
        benchmark_script.get("path") == str(Path(__file__).resolve())
        and benchmark_script.get("sha256") == _sha256(__file__)
    )
    wheel_runtime_binding = _wheel_runtime_binding(
        wheels,
        distributions,
        loaded_modules,
        native_binaries,
        str(source_root) if source_root else None,
        str(job["backend"]),
    )
    clock_power_after = _clock_power_snapshot(str(job["backend"]))
    return {
        "variant": job["variant"],
        "source_root": source_root,
        "source_commit": _git_commit(str(source_root)) if source_root else None,
        "source_tree_state": source_tree_state,
        "benchmark_script": benchmark_script,
        "benchmark_script_canonical": benchmark_script_canonical,
        "wheel_artifacts": [
            _wheel_identity_summary(_wheel_identity(wheel))
            for wheel in wheels
        ],
        "wheel_hash_status": "observed" if wheels else "not_supplied",
        "wheel_runtime_binding": wheel_runtime_binding,
        "loaded_module_files": loaded_modules,
        "native_binaries": native_binaries,
        "python_executable": sys.executable,
        "python_executable_identity": _file_identity(sys.executable),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_identity(),
        "process_affinity": affinity,
        "driver_command": list(job.get("driver_command", ())),
        "worker_command": list(job.get("worker_command", ())),
        "environment": {
            key: os.environ[key] for key in RELEVANT_ENV_KEYS if key in os.environ
        },
        "installed_distributions": distributions,
        "runtime_dependencies": _runtime_dependency_identities(),
        "toolchains": _toolchain_snapshot(),
        "device_identity": _device_identity_snapshot(str(job["backend"])),
        "clock_and_power_capture_point": "before and after all timed benchmark regions",
        "clock_and_power_state": {
            "before": dict(clock_power_before or {}),
            "after": clock_power_after,
        },
    }


def _phase(
    status: str,
    duration_ns: int | None,
    *,
    detail: str,
    combined_in: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "duration_ns": duration_ns,
        "detail": detail,
        "combined_in": combined_in,
    }


def _time_call(function: Callable[[], object]) -> tuple[object, int]:
    start = perf_counter_ns()
    result = function()
    return result, perf_counter_ns() - start


def _make_dataset(np, workload: Workload, input_policy: str, precision: str, seed: int):
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((workload.samples, workload.features)).astype(
        np.float64
    )
    rows = np.arange(workload.samples, dtype=np.float64)
    for column in range(workload.features):
        source[:, column] += 0.20 * np.sin(rows * (column + 1) * 0.0031)
        source[:, column] += 0.07 * np.cos(rows * (column + 2) * 0.0017)
    weights = np.linspace(0.8, -0.25, workload.features, dtype=np.float64)
    target = source @ weights
    if workload.features > 1:
        target += 0.35 * source[:, 0] * source[:, 1]
    target += rng.standard_normal(workload.samples) * 0.03
    source_dtype = "float64"
    if input_policy == "native" and precision in ("fp32", "mixed"):
        matrix = np.ascontiguousarray(source, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        source_dtype = "float32"
    else:
        matrix = np.ascontiguousarray(source, dtype=np.float64)
        target = np.ascontiguousarray(target, dtype=np.float64)
    names = tuple(f"x{column}" for column in range(workload.features))
    return matrix, target, names, source_dtype


def _dataset_identity(matrix, target, names: Sequence[str]) -> dict[str, object]:
    return {
        "matrix_sha256": hashlib.sha256(memoryview(matrix).cast("B")).hexdigest(),
        "target_sha256": hashlib.sha256(memoryview(target).cast("B")).hexdigest(),
        "feature_names_sha256": hashlib.sha256(
            "\0".join(str(name) for name in names).encode("utf-8")
        ).hexdigest(),
        "matrix_shape": list(getattr(matrix, "shape", ())),
        "target_shape": list(getattr(target, "shape", ())),
        "matrix_dtype": str(getattr(matrix, "dtype", "")),
        "target_dtype": str(getattr(target, "dtype", "")),
    }


def _config(
    gafime, workload: Workload, backend: str, precision: str, device_id: int, seed: int
):
    max_per_k = max(
        math.comb(workload.features, arity)
        for arity in range(1, workload.max_arity + 1)
    )
    return gafime.EngineConfig(
        backend=backend,
        device_id=device_id,
        precision=precision,
        metric_names=workload.metrics,
        num_repeats=1,
        permutation_tests=0,
        random_seed=seed,
        mi_bins=workload.mi_bins,
        mi_approximate=True,
        budget=gafime.ComputeBudget(
            max_comb_size=workload.max_arity,
            max_combinations_per_k=max_per_k,
            top_features_for_higher_k=workload.features,
            max_generated_features=0,
            keep_in_vram=True,
            vram_budget_mb=8_192,
            max_feature_candidate=workload.features,
        ),
    )


def _validate_report(
    report,
    backend: str,
    precision: str,
    metrics: Sequence[str],
    expected_candidates: int,
) -> int:
    info = getattr(report, "backend", None)
    if info is None or getattr(info, "selected_backend", None) != backend:
        raise AssertionError(
            f"{backend}/{precision}: explicit backend was not preserved"
        )
    if getattr(info, "effective_precision", None) != precision:
        raise AssertionError(f"{backend}/{precision}: effective precision changed")
    rows = list(getattr(report, "interactions"))
    if len(rows) != expected_candidates:
        raise AssertionError(
            f"{backend}/{precision}: expected {expected_candidates} candidates, "
            f"received {len(rows)}"
        )
    for row in rows:
        for metric in metrics:
            if not math.isfinite(float(row.metrics[metric])):
                raise AssertionError(
                    f"{backend}/{precision}/{metric}: non-finite value"
                )
    return len(rows)


@contextmanager
def _analyze_cache(gafime, capacity: int):
    from gafime import v1_adapter

    key = "GAFIME_V1_ANALYZE_CACHE_SIZE"
    previous = os.environ.get(key)
    os.environ[key] = str(capacity)
    v1_adapter._clear_analyze_cache_for_tests()
    try:
        yield
    finally:
        v1_adapter._clear_analyze_cache_for_tests()
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _calibrated_samples(
    operation: Callable[[], object],
    validate: Callable[[object], int],
    *,
    warmups: int,
    repetitions: int,
    min_sample_ns: int,
    max_loop_count: int,
) -> tuple[list[int], int, int]:
    candidate_count = 0
    for _ in range(warmups):
        candidate_count = validate(operation())
    loop_count = 1
    while True:
        start = perf_counter_ns()
        last_report = None
        for _ in range(loop_count):
            last_report = operation()
        elapsed = perf_counter_ns() - start
        candidate_count = validate(last_report)
        if elapsed >= min_sample_ns or loop_count >= max_loop_count:
            break
        loop_count = min(max_loop_count, loop_count * 2)
    raw: list[int] = []
    for _ in range(repetitions):
        start = perf_counter_ns()
        last_report = None
        for _ in range(loop_count):
            last_report = operation()
        raw.append(perf_counter_ns() - start)
        # Validation is deliberately outside the measured region.
        candidate_count = validate(last_report)
    return raw, loop_count, candidate_count


def _calibrate_operation(
    operation: Callable[[], object],
    validate: Callable[[object], int],
    *,
    min_sample_ns: int,
    max_loop_count: int,
) -> tuple[int, int]:
    loop_count = 1
    candidate_count = 0
    while True:
        start = perf_counter_ns()
        last_report = None
        for _ in range(loop_count):
            last_report = operation()
        elapsed = perf_counter_ns() - start
        candidate_count = validate(last_report)
        if elapsed >= min_sample_ns or loop_count >= max_loop_count:
            return loop_count, candidate_count
        loop_count = min(max_loop_count, loop_count * 2)


def _make_analyze_operation(engine, matrix, target, names):
    def operation():
        return engine.analyze(matrix, target, feature_names=names)

    return operation


def _make_artifact_operation(artifact):
    def operation():
        return artifact.analyze()

    return operation


@contextmanager
def _interleaved_operations(
    gafime,
    np,
    *,
    job: Mapping[str, object],
    workload: Workload,
    surface: str,
):
    backend = str(job["backend"])
    profiles = tuple(str(profile) for profile in job["profile_order"])
    artifacts = []
    operations: dict[
        str, tuple[Callable[[], object], Callable[[object], int], dict[str, object]]
    ] = {}
    cache_capacity = len(profiles) if surface == "resident" else 0
    try:
        with _analyze_cache(gafime, cache_capacity):
            for precision in profiles:
                matrix, target, names, source_dtype = _make_dataset(
                    np,
                    workload,
                    str(job["input_policy"]),
                    precision,
                    int(job["seed"]),
                )
                config = _config(
                    gafime,
                    workload,
                    backend,
                    precision,
                    int(job["device_id"]),
                    int(job["seed"]),
                )

                artifact_for_validation = [None]

                def validate(
                    report,
                    selected_precision=precision,
                    artifact_holder=artifact_for_validation,
                ) -> int:
                    count = _validate_report(
                        report,
                        backend,
                        selected_precision,
                        workload.metrics,
                        workload.expected_candidates,
                    )
                    if (
                        surface == "graph"
                        and getattr(artifact_holder[0], "graph_replayed", None) is not True
                    ):
                        raise AssertionError("graph execution did not confirm replay")
                    return count

                if surface in ("one_shot", "resident"):
                    engine = gafime.GafimeEngine(config)
                    operation = _make_analyze_operation(engine, matrix, target, names)
                else:
                    artifact = gafime.compile(
                        matrix,
                        target,
                        names,
                        config=config,
                        flags=gafime.CompileFlags(plan=True, graph=surface == "graph"),
                    )
                    artifacts.append(artifact)
                    artifact_for_validation[0] = artifact
                    operation = _make_artifact_operation(artifact)
                operations[precision] = (
                    operation,
                    validate,
                    {
                        "source_dtype": source_dtype,
                        "input_bytes": int(matrix.nbytes + target.nbytes),
                        "input_identity": _dataset_identity(matrix, target, names),
                    },
                )
            if surface == "resident":
                for precision in profiles:
                    operation, validate, _ = operations[precision]
                    validate(operation())
            yield operations
    finally:
        for artifact in reversed(artifacts):
            artifact.close()


def _measure_interleaved_control(
    gafime,
    np,
    *,
    job: Mapping[str, object],
    workload: Workload,
    surface: str,
) -> dict[str, object]:
    backend = str(job["backend"])
    profiles = tuple(str(profile) for profile in job["profile_order"])
    if len(profiles) < 2:
        return {
            "status": "not_applicable",
            "surface": surface,
            "detail": "an interleaved order control requires at least two profiles",
        }
    if surface == "graph" and backend not in GRAPH_BACKENDS:
        return {
            "status": "unsupported_by_contract",
            "surface": surface,
            "detail": "graph replay is a CUDA/ROCm public surface",
        }
    rng = random.Random(
        int(job["seed"])
        ^ int.from_bytes(
            hashlib.sha256(
                f"interleaved/{backend}/{workload.name}/{surface}".encode("utf-8")
            ).digest()[:8],
            "little",
        )
    )
    possible_orders = list(_profile_orders(profiles))
    measured_orders: list[tuple[str, ...]] = []
    while len(measured_orders) < int(job["repetitions"]):
        cycle = list(possible_orders)
        rng.shuffle(cycle)
        measured_orders.extend(cycle)
    measured_orders = measured_orders[: int(job["repetitions"])]
    raw_by_profile: dict[str, list[int]] = {profile: [] for profile in profiles}
    loop_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    metadata: dict[str, dict[str, object]] = {}
    with _interleaved_operations(
        gafime, np, job=job, workload=workload, surface=surface
    ) as operations:
        warmup_orders = list(possible_orders)
        rng.shuffle(warmup_orders)
        for warmup in range(int(job["warmups"])):
            for precision in warmup_orders[warmup % len(warmup_orders)]:
                operation, validate, _ = operations[precision]
                validate(operation())
        for precision in profiles:
            operation, validate, details = operations[precision]
            loop_count, candidate_count = _calibrate_operation(
                operation,
                validate,
                min_sample_ns=int(job["min_sample_ns"]),
                max_loop_count=int(job["max_loop_count"]),
            )
            loop_counts[precision] = loop_count
            candidate_counts[precision] = candidate_count
            metadata[precision] = details
        for order in measured_orders:
            for precision in order:
                operation, validate, _ = operations[precision]
                start = perf_counter_ns()
                last_report = None
                for _ in range(loop_counts[precision]):
                    last_report = operation()
                elapsed = perf_counter_ns() - start
                # Report validation is intentionally outside the timed public
                # call region; only the repeated public operation is timed.
                validate(last_report)
                raw_by_profile[precision].append(elapsed)
    profiles_result = {}
    for precision in profiles:
        position_samples: dict[int, list[float]] = {}
        for sample_index, order in enumerate(measured_orders):
            position = order.index(precision)
            position_samples.setdefault(position, []).append(
                raw_by_profile[precision][sample_index] / loop_counts[precision]
            )
        position_medians = {
            str(position): statistics.median(values)
            for position, values in sorted(position_samples.items())
        }
        median_values = list(position_medians.values())
        position_center = statistics.median(median_values)
        position_spread_percent = (
            (max(median_values) - min(median_values)) * 100.0 / position_center
            if len(median_values) > 1 and position_center > 0.0
            else 0.0
        )
        profiles_result[precision] = {
            **metadata[precision],
            "order_position_median_ns": position_medians,
            "max_order_position_spread_percent": position_spread_percent,
            "order_sensitivity_status": (
                "unacceptable_until_investigated"
                if position_spread_percent > 3.0
                else "investigate_possible_order_contamination"
                if position_spread_percent > 1.0
                else "no_order_effect_above_one_percent_observed"
            ),
            "distribution": _distribution(
                raw_by_profile[precision],
                loop_counts[precision],
                bootstrap_resamples=int(job["bootstrap_resamples"]),
                seed=int(job["seed"])
                ^ int.from_bytes(
                    hashlib.sha256(
                        f"interleaved/{precision}/{surface}/{workload.name}".encode(
                            "utf-8"
                        )
                    ).digest()[:8],
                    "little",
                ),
                samples=workload.samples,
                candidates=candidate_counts[precision],
                metrics=workload.metrics,
                sample_region_target_ns=int(job["min_sample_ns"]),
            ),
        }
    return {
        "status": "pass",
        "surface": surface,
        "profile_block_orders": [list(order) for order in measured_orders],
        "all_possible_orders_covered": {tuple(order) for order in measured_orders}
        == set(possible_orders),
        "timing_scope": (
            "already-warmed public backend; each recorded block interleaves profiles "
            "in a balanced randomized order to expose thermal or clock contamination"
        ),
        "profiles": profiles_result,
    }


def _measure_surface(
    gafime,
    np,
    *,
    job: Mapping[str, object],
    workload: Workload,
    precision: str,
    surface: str,
) -> dict[str, object]:
    backend = str(job["backend"])
    input_policy = str(job["input_policy"])
    matrix, target, names, source_dtype = _make_dataset(
        np, workload, input_policy, precision, int(job["seed"])
    )
    config = _config(
        gafime, workload, backend, precision, int(job["device_id"]), int(job["seed"])
    )

    def validate(report) -> int:
        count = _validate_report(
            report,
            backend,
            precision,
            workload.metrics,
            workload.expected_candidates,
        )
        if surface == "graph" and getattr(artifact, "graph_replayed", None) is not True:
            raise AssertionError("graph execution did not confirm replay")
        return count

    def analyze_operation():
        return engine.analyze(matrix, target, feature_names=names)

    artifact = None
    engine = gafime.GafimeEngine(config)
    if surface == "graph" and backend not in GRAPH_BACKENDS:
        return {
            "status": "unsupported_by_contract",
            "surface": surface,
            "detail": "graph replay is a CUDA/ROCm public surface",
        }
    if surface == "one_shot":
        cache_context = _analyze_cache(gafime, 0)
        operation = analyze_operation
    elif surface == "resident":
        cache_context = _analyze_cache(gafime, 1)
        operation = analyze_operation
    else:
        cache_context = _analyze_cache(gafime, 0)
        artifact = gafime.compile(
            matrix,
            target,
            names,
            config=config,
            flags=gafime.CompileFlags(plan=True, graph=surface == "graph"),
        )
        operation = artifact.analyze
    try:
        with cache_context:
            if surface == "resident":
                validate(operation())
            raw, loop_count, candidate_count = _calibrated_samples(
                operation,
                validate,
                warmups=int(job["warmups"]),
                repetitions=int(job["repetitions"]),
                min_sample_ns=int(job["min_sample_ns"]),
                max_loop_count=int(job["max_loop_count"]),
            )
            if (
                surface == "graph"
                and getattr(artifact, "graph_replayed", None) is not True
            ):
                raise AssertionError("graph execution did not confirm replay")
    finally:
        if artifact is not None:
            artifact.close()
    return {
        "status": "pass",
        "surface": surface,
        "source_dtype": source_dtype,
        "input_policy": input_policy,
        "input_bytes": int(matrix.nbytes + target.nbytes),
        "input_identity": _dataset_identity(matrix, target, names),
        "output_row_count": candidate_count,
        "timing_scope": (
            "public wall clock including the report materialization performed by "
            "this public surface; not a device-kernel timer"
        ),
        "distribution": _distribution(
            raw,
            loop_count,
            bootstrap_resamples=int(job["bootstrap_resamples"]),
            seed=int(job["seed"])
            ^ int.from_bytes(
                hashlib.sha256(
                    f"{precision}/{surface}/{workload.name}".encode("utf-8")
                ).digest()[:8],
                "little",
            ),
            samples=workload.samples,
            candidates=candidate_count,
            metrics=workload.metrics,
            sample_region_target_ns=int(job["min_sample_ns"]),
        ),
    }


def _native_binary_inventory() -> list[dict[str, object]]:
    suffixes = (".so", ".dylib", ".dll", ".pyd")
    paths: set[Path] = set()
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "gafime_py" and not module_name.startswith("gafime."):
            continue
        raw = getattr(module, "__file__", None)
        if raw and str(raw).lower().endswith(suffixes):
            path = Path(raw).resolve()
            if path.is_file():
                paths.add(path)
    for key in RELEVANT_ENV_KEYS[:4]:
        raw = os.environ.get(key)
        if raw:
            path = Path(raw).resolve()
            if path.is_file():
                paths.add(path)
    return [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(paths)
    ]


def _cold_worker(job: Mapping[str, object], process_start_ns: int) -> dict[str, object]:
    backend = str(job["backend"])
    clock_power_before = _clock_power_snapshot(backend)
    cold_region_start_ns = perf_counter_ns()
    phases = {
        name: _phase("not_observed", None, detail="not reached") for name in COLD_PHASES
    }
    import importlib

    dependency_import_start = perf_counter_ns()
    np = importlib.import_module("numpy")
    dependency_import_ns = perf_counter_ns() - dependency_import_start
    import_start = perf_counter_ns()
    gafime = importlib.import_module("gafime")
    phases["python_import"] = _phase(
        "observed",
        perf_counter_ns() - import_start,
        detail="GAFIME import after a separately recorded NumPy dependency import",
    )
    workload = _decode_workload(job["workload"])
    precision = str(job["precision"])
    matrix, target, names, source_dtype = _make_dataset(
        np, workload, str(job["input_policy"]), precision, int(job["seed"])
    )

    discovery_start = perf_counter_ns()
    payloads = importlib.import_module("gafime._payloads").discover_payloads(backend)
    phases["payload_discovery"] = _phase(
        "observed",
        perf_counter_ns() - discovery_start,
        detail="side-effect-free installed-payload path discovery; no capability probe",
    )
    capability_start = perf_counter_ns()
    capabilities = gafime.backend_capabilities(
        backend,
        int(job["device_id"]),
        probe=True,
        precision=precision,
        mi_bins=workload.mi_bins,
        mi_approximate=True,
    )
    capability_ns = perf_counter_ns() - capability_start
    if (
        capabilities.selection_status != "available"
        or capabilities.selected_backend != backend
    ):
        raise RuntimeError(
            f"{backend}/{precision}: capability probe failed closed: "
            f"{capabilities.selection_status} {capabilities.selection_detail}"
        )
    phases["first_capability_query"] = _phase(
        "observed_combined",
        capability_ns,
        detail="public capability query includes native load/runtime work",
    )
    for name in (
        "dynamic_library_load",
        "runtime_context_initialization",
        "code_object_or_module_registration",
    ):
        phases[name] = _phase(
            "combined_not_separately_observable",
            None,
            detail="the public capability boundary exposes no safe separate timer",
            combined_in="first_capability_query and/or first_execution",
        )

    config = _config(
        gafime, workload, backend, precision, int(job["device_id"]), int(job["seed"])
    )
    compile_start = perf_counter_ns()
    artifact = gafime.compile(
        matrix,
        target,
        names,
        config=config,
        flags=gafime.CompileFlags(plan=True, graph=False),
    )
    compile_ns = perf_counter_ns() - compile_start
    for name in ("first_allocation", "first_upload", "planning"):
        phases[name] = _phase(
            "combined_not_separately_observable",
            None,
            detail="canonical public compile combines conversion, allocation, upload, and planning",
            combined_in="first_compile",
        )
    execution_start = perf_counter_ns()
    report = artifact.analyze()
    execution_ns = perf_counter_ns() - execution_start
    for name in ("first_execution", "first_result_materialization"):
        phases[name] = _phase(
            "combined_not_separately_observable",
            None,
            detail="public artifact.analyze combines synchronized execution and report materialization",
            combined_in="first_analyze",
        )
    cleanup_start = perf_counter_ns()
    artifact.close()
    phases["explicit_cleanup"] = _phase(
        "observed",
        perf_counter_ns() - cleanup_start,
        detail="artifact close and Python reference release before process exit",
    )
    cold_region_duration_ns = perf_counter_ns() - cold_region_start_ns
    # Validate the result after the clean cold interval has stopped.  The
    # validation is an integrity check, not part of the cold timing boundary.
    candidate_count = _validate_report(
        report,
        backend,
        precision,
        workload.metrics,
        workload.expected_candidates,
    )
    del report
    phases["process_exit_cleanup"] = _phase(
        "parent_residual",
        None,
        detail="driver records subprocess wall time; exit-only residual is not a kernel metric",
    )
    provenance = _base_provenance(job, clock_power_before=clock_power_before)
    native_binaries = _native_binary_inventory()
    worker_elapsed_ns = perf_counter_ns() - process_start_ns
    return {
        "schema": WORKER_SCHEMA,
        "kind": "cold",
        "status": "pass",
        "profile": precision,
        "backend": backend,
        "profile_order_context": list(job["profile_order_context"]),
        "order_repeat": job["order_repeat"],
        "order_index": job["order_index"],
        "ab_block": job["ab_block"],
        "variant_sequence": list(job.get("variant_sequence", ())),
        "input_policy": job["input_policy"],
        "source_dtype": source_dtype,
        "input_identity": _dataset_identity(matrix, target, names),
        "workload": _workload_payload(workload),
        "candidate_count": candidate_count,
        "phases": phases,
        "dependency_import_duration_ns": dependency_import_ns,
        "combined_phase_durations_ns": {
            "first_compile": compile_ns,
            "first_analyze": execution_ns,
        },
        "cold_region_duration_ns": cold_region_duration_ns,
        "cold_region_timing_scope": (
            "fresh worker interval from worker entry through explicit artifact cleanup; "
            "excludes provenance capture, JSON serialization, process startup, and exit residual"
        ),
        "payload_discovery": {key: str(value) for key, value in payloads.items()},
        "capability": capabilities.to_dict(),
        "native_binaries": native_binaries,
        "worker_elapsed_ns": worker_elapsed_ns,
        "provenance": provenance,
    }


def _public_worker(
    job: Mapping[str, object], process_start_ns: int
) -> dict[str, object]:
    import importlib

    clock_power_before = _clock_power_snapshot(str(job["backend"]))
    np = importlib.import_module("numpy")
    gafime = importlib.import_module("gafime")
    workload = _decode_workload(job["workload"])
    cells: list[dict[str, object]] = []
    capability_records: dict[str, object] = {}
    for ordinal, precision in enumerate(job["profile_order"]):
        capabilities = gafime.backend_capabilities(
            str(job["backend"]),
            int(job["device_id"]),
            probe=True,
            precision=precision,
            mi_bins=workload.mi_bins,
            mi_approximate=True,
        )
        if capabilities.selection_status != "available":
            raise RuntimeError(
                f"{job['backend']}/{precision}: {capabilities.selection_detail}"
            )
        capability_records[precision] = capabilities.to_dict()
        for surface in job["surface_order"]:
            result = _measure_surface(
                gafime,
                np,
                job=job,
                workload=workload,
                precision=precision,
                surface=surface,
            )
            result.update(
                {
                    "profile": precision,
                    "profile_order_ordinal": ordinal,
                    "workload": _workload_payload(workload),
                }
            )
            cells.append(result)
    interleaved_controls: list[dict[str, object]] = []
    if job.get("interleaved_control") is True:
        for surface in job["surface_order"]:
            interleaved_controls.append(
                _measure_interleaved_control(
                    gafime,
                    np,
                    job=job,
                    workload=workload,
                    surface=surface,
                )
            )
    provenance = _base_provenance(job, clock_power_before=clock_power_before)
    native_binaries = _native_binary_inventory()
    worker_elapsed_ns = perf_counter_ns() - process_start_ns
    return {
        "schema": WORKER_SCHEMA,
        "kind": "public",
        "status": "pass",
        "backend": job["backend"],
        "input_policy": job["input_policy"],
        "workload": _workload_payload(workload),
        "ab_block": job["ab_block"],
        "variant_sequence": list(job.get("variant_sequence", ())),
        "order_repeat": job["order_repeat"],
        "order_index": job["order_index"],
        "profile_order": list(job["profile_order"]),
        "surface_order": list(job["surface_order"]),
        "capabilities": capability_records,
        "cells": cells,
        "interleaved_controls": interleaved_controls,
        "native_binaries": native_binaries,
        "worker_elapsed_ns": worker_elapsed_ns,
        "provenance": provenance,
    }


def _worker_main() -> int:
    process_start_ns = perf_counter_ns()
    job = json.load(sys.stdin)
    if job["kind"] == "cold":
        result = _cold_worker(job, process_start_ns)
    elif job["kind"] == "public":
        result = _public_worker(job, process_start_ns)
    else:
        raise ValueError(f"unknown worker kind {job['kind']!r}")
    sys.stdout.write(json.dumps(_json_safe(result), sort_keys=True) + "\n")
    return 0


def _variant_sequences(
    variants: Sequence[Variant], blocks: int, rng: random.Random
) -> tuple[tuple[Variant, ...], ...]:
    if len(variants) == 1:
        return ((variants[0],),)
    first = list(variants)
    if rng.randrange(2):
        first.reverse()
    second = list(reversed(first))
    return tuple(tuple(first if block % 2 == 0 else second) for block in range(blocks))


def _variant_worker_script(variant: Variant) -> str:
    # Baseline 8df predates perf13.  Keep one frozen driver/worker identity for
    # every isolated Python environment and bind each build independently via
    # its source root, wheels, native evidence, and loaded modules.
    return str(Path(__file__).resolve())


def _run_worker(
    variant: Variant,
    job: dict[str, object],
    *,
    timeout: int,
) -> dict[str, object]:
    script = _variant_worker_script(variant)
    if not Path(script).is_file():
        raise RuntimeError(
            f"variant {variant.name!r} benchmark script does not exist: {script}"
        )
    command = [variant.python, script, "--_worker"]
    job.update(
        {
            "variant": variant.name,
            "source_root": variant.source_root,
            "wheels": list(variant.wheels),
            "driver_command": list(sys.argv),
            "worker_command": command,
        }
    )
    print(
        f"[{job['kind']}] variant={variant.name} backend={job['backend']} "
        f"order={job.get('profile_order', [job.get('precision')])} "
        f"workload={job['workload']['name']}",
        file=sys.stderr,
        flush=True,
    )
    start = perf_counter_ns()
    worker_environment = dict(os.environ)
    python_path = Path(variant.python).expanduser().absolute()
    python_bin = python_path.parent
    candidate_virtual_env = python_bin.parent
    inherited_virtual_env = worker_environment.get("VIRTUAL_ENV")
    filtered_path = []
    for entry in worker_environment.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        if inherited_virtual_env and Path(entry) in {
            Path(inherited_virtual_env) / "bin",
            Path(inherited_virtual_env) / "Scripts",
        }:
            continue
        if Path(entry) == python_bin:
            continue
        filtered_path.append(entry)
    worker_environment["PATH"] = os.pathsep.join(
        (str(python_bin), *filtered_path)
    )
    if (candidate_virtual_env / "pyvenv.cfg").is_file():
        worker_environment["VIRTUAL_ENV"] = str(candidate_virtual_env)
    else:
        worker_environment.pop("VIRTUAL_ENV", None)
    # The worker must import the independently installed wheel. Inherited
    # source/module overrides would invalidate that isolation.
    worker_environment.pop("PYTHONPATH", None)
    for key in (
        "GAFIME_CUDA_V1_LIB",
        "GAFIME_ROCM_V1_LIB",
        "GAFIME_METAL_V1_LIB",
        "GAFIME_METAL_V1_METALLIB",
        "GAFIME_V1_PY_MODULE",
    ):
        worker_environment.pop(key, None)
    completed = subprocess.run(
        command,
        input=json.dumps(job),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=worker_environment,
    )
    wall_ns = perf_counter_ns() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed ({' '.join(command)}):\n{completed.stderr[-16_384:]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"worker returned invalid JSON: {completed.stdout[-4096:]}"
        ) from exc
    result["subprocess_wall_ns"] = wall_ns
    worker_elapsed = result.get("worker_elapsed_ns")
    result["subprocess_start_and_exit_residual_ns"] = (
        max(0, wall_ns - int(worker_elapsed))
        if isinstance(worker_elapsed, int)
        else None
    )
    if result.get("kind") == "cold" and isinstance(result.get("phases"), dict):
        result["phases"]["process_exit_cleanup"] = _phase(
            "combined_not_separately_observable",
            None,
            detail=(
                "the parent measured process startup through exit; interpreter "
                "startup and exit cleanup cannot be separated safely"
            ),
            combined_in="subprocess_start_and_exit_residual_ns",
        )
    result["worker_stderr"] = completed.stderr
    return result


def _backend_profile_order(
    backend: str, requested: Sequence[str], parser_error: Callable[[str], None]
) -> tuple[str, ...]:
    supported = BACKEND_PROFILES[backend]
    unsupported = [profile for profile in requested if profile not in supported]
    if unsupported:
        parser_error(
            f"backend={backend} does not support requested profile(s): {', '.join(unsupported)}"
        )
    return tuple(requested)


def _order_sensitivity(
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[tuple[int, float, tuple[str, ...]]]] = {}
    for result in results:
        if result.get("kind") != "public" or result.get("status") != "pass":
            continue
        provenance = result.get("provenance", {})
        variant = (
            str(provenance.get("variant"))
            if isinstance(provenance, Mapping)
            else "unknown"
        )
        workload = result.get("workload", {})
        workload_name = (
            str(workload.get("name")) if isinstance(workload, Mapping) else "unknown"
        )
        profile_order = tuple(str(value) for value in result.get("profile_order", ()))
        for cell in result.get("cells", ()):
            if not isinstance(cell, Mapping) or cell.get("status") != "pass":
                continue
            distribution = cell.get("distribution", {})
            if not isinstance(distribution, Mapping):
                continue
            key = (
                variant,
                str(result.get("backend")),
                workload_name,
                str(result.get("input_policy")),
                str(cell.get("surface")),
                str(cell.get("profile")),
            )
            groups.setdefault(key, []).append(
                (
                    int(cell["profile_order_ordinal"]),
                    float(distribution["median_ns"]),
                    profile_order,
                )
            )
    summaries: list[dict[str, object]] = []
    for key, observations in sorted(groups.items()):
        by_position: dict[int, list[float]] = {}
        for position, median_ns, _ in observations:
            by_position.setdefault(position, []).append(median_ns)
        position_medians = {
            str(position): statistics.median(values)
            for position, values in sorted(by_position.items())
        }
        values = list(position_medians.values())
        center = statistics.median(values)
        spread_percent = (
            (max(values) - min(values)) * 100.0 / center
            if len(values) > 1 and center > 0.0
            else 0.0
        )
        if spread_percent > 3.0:
            status = "unacceptable_until_investigated"
        elif spread_percent > 1.0:
            status = "investigate_possible_order_contamination"
        else:
            status = "no_order_effect_above_one_percent_observed"
        summaries.append(
            {
                "variant": key[0],
                "backend": key[1],
                "workload": key[2],
                "input_policy": key[3],
                "surface": key[4],
                "profile": key[5],
                "observation_count": len(observations),
                "position_median_ns": position_medians,
                "max_position_median_spread_percent": spread_percent,
                "status": status,
                "orders": [list(order) for _, _, order in observations],
                "interpretation": (
                    "An order-dependent spread above one percent requires investigation; "
                    "the raw distributions and confidence intervals remain authoritative."
                ),
            }
        )
    return summaries


def _interleaved_order_sensitivity(
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for result_index, result in enumerate(results):
        if result.get("kind") != "public":
            continue
        for control_index, control in enumerate(result.get("interleaved_controls", ())):
            if not isinstance(control, Mapping) or control.get("status") != "pass":
                continue
            profiles = control.get("profiles")
            if not isinstance(profiles, Mapping):
                continue
            for profile, profile_result in profiles.items():
                if not isinstance(profile_result, Mapping):
                    continue
                summaries.append(
                    {
                        "result_index": result_index,
                        "control_index": control_index,
                        "backend": result.get("backend"),
                        "workload": result.get("workload", {}).get("name")
                        if isinstance(result.get("workload"), Mapping)
                        else None,
                        "input_policy": result.get("input_policy"),
                        "surface": control.get("surface"),
                        "profile": profile,
                        "status": profile_result.get("order_sensitivity_status"),
                        "max_order_position_spread_percent": profile_result.get(
                            "max_order_position_spread_percent"
                        ),
                    }
                )
    return summaries


def _ab_comparisons(
    results: Sequence[Mapping[str, object]],
    variants: Sequence[Variant],
    *,
    bootstrap_resamples: int = 500,
    seed: int = 0,
) -> list[dict[str, object]]:
    if len(variants) != 2:
        return []
    baseline, candidate = variants
    groups: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = {}
    for result in results:
        if result.get("kind") != "public" or result.get("status") != "pass":
            continue
        provenance = result.get("provenance", {})
        if not isinstance(provenance, Mapping):
            continue
        variant = str(provenance.get("variant"))
        workload = result.get("workload", {})
        workload_name = (
            str(workload.get("name")) if isinstance(workload, Mapping) else "unknown"
        )
        order = tuple(str(item) for item in result.get("profile_order", ()))
        for cell in result.get("cells", ()):
            if not isinstance(cell, Mapping) or cell.get("status") != "pass":
                continue
            distribution = cell.get("distribution", {})
            if not isinstance(distribution, Mapping):
                continue
            key = (
                result.get("backend"),
                workload_name,
                result.get("input_policy"),
                order,
                result.get("ab_block"),
                result.get("order_repeat"),
                cell.get("surface"),
                cell.get("profile"),
            )
            groups.setdefault(key, {})[variant] = distribution
    comparisons: list[dict[str, object]] = []
    for key, distributions in groups.items():
        if baseline.name not in distributions or candidate.name not in distributions:
            continue
        baseline_distribution = distributions[baseline.name]
        candidate_distribution = distributions[candidate.name]
        baseline_ns = float(baseline_distribution["median_ns"])
        candidate_ns = float(candidate_distribution["median_ns"])
        delta_percent = (candidate_ns - baseline_ns) * 100.0 / baseline_ns
        baseline_raw = baseline_distribution.get("raw_per_call_duration_ns", ())
        candidate_raw = candidate_distribution.get("raw_per_call_duration_ns", ())
        delta_ci_ns: list[float] | None = None
        delta_ci_percent: list[float] | None = None
        effective_comparison_sample_count = 0
        if (
            isinstance(baseline_raw, (list, tuple))
            and isinstance(candidate_raw, (list, tuple))
            and baseline_raw
            and candidate_raw
        ):
            baseline_values = [float(value) for value in baseline_raw]
            candidate_values = [float(value) for value in candidate_raw]
            baseline_sample_count = len(baseline_values)
            candidate_sample_count = len(candidate_values)
            effective_comparison_sample_count = min(
                baseline_sample_count, candidate_sample_count
            )
            # Baseline and candidate execute in separate fresh workers.  Matching
            # raw index ``i`` would invent a pair and understate uncertainty, so
            # resample each worker distribution independently.
            local_rng = random.Random(
                seed
                ^ int.from_bytes(
                    hashlib.sha256(repr(key).encode("utf-8")).digest()[:8],
                    "little",
                )
            )
            bootstrap_deltas = []
            for _ in range(bootstrap_resamples):
                baseline_indices = [
                    local_rng.randrange(baseline_sample_count)
                    for _ in range(baseline_sample_count)
                ]
                candidate_indices = [
                    local_rng.randrange(candidate_sample_count)
                    for _ in range(candidate_sample_count)
                ]
                baseline_sample = [
                    baseline_values[index] for index in baseline_indices
                ]
                candidate_sample = [
                    candidate_values[index] for index in candidate_indices
                ]
                bootstrap_deltas.append(
                    float(statistics.median(candidate_sample))
                    - float(statistics.median(baseline_sample))
                )
            delta_ci_ns = [
                _percentile(bootstrap_deltas, 0.025),
                _percentile(bootstrap_deltas, 0.975),
            ]
            delta_ci_percent = [
                value * 100.0 / baseline_ns for value in delta_ci_ns
            ]
        comparisons.append(
            {
                "backend": key[0],
                "workload": key[1],
                "input_policy": key[2],
                "profile_order": list(key[3]),
                "ab_block": key[4],
                "order_repeat": key[5],
                "surface": key[6],
                "profile": key[7],
                "baseline_variant": baseline.name,
                "candidate_variant": candidate.name,
                "baseline_median_ns": baseline_ns,
                "candidate_median_ns": candidate_ns,
                "candidate_latency_delta_percent": delta_percent,
                "sample_count_baseline": len(baseline_raw) if isinstance(baseline_raw, (list, tuple)) else 0,
                "sample_count_candidate": len(candidate_raw) if isinstance(candidate_raw, (list, tuple)) else 0,
                "effective_comparison_sample_count": effective_comparison_sample_count,
                "comparison_sample_count": effective_comparison_sample_count,
                "pairing": "independent_worker_distributions",
                "bootstrap_delta_median_95_ci_ns": delta_ci_ns,
                "bootstrap_candidate_latency_delta_95_ci_percent": delta_ci_percent,
                "review_status": (
                    "maintainer_approval_required"
                    if delta_percent > 3.0
                    else "investigate"
                    if delta_percent > 1.0
                    else "within_one_percent"
                ),
            }
        )
    return comparisons


def _cold_summaries(
    results: Sequence[Mapping[str, object]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[float]] = {}
    contexts: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for result in results:
        if result.get("kind") != "cold" or result.get("status") != "pass":
            continue
        duration = result.get("cold_region_duration_ns")
        workload = result.get("workload", {})
        provenance = result.get("provenance", {})
        if not isinstance(duration, int) or not isinstance(workload, Mapping):
            continue
        variant = (
            str(provenance.get("variant"))
            if isinstance(provenance, Mapping)
            else "unknown"
        )
        key = (
            variant,
            str(result.get("backend")),
            str(result.get("profile")),
            str(result.get("input_policy")),
            str(workload.get("name")),
        )
        context = {
            "profile_order_context": list(result.get("profile_order_context", ())),
            "order_repeat": result.get("order_repeat"),
            "order_index": result.get("order_index"),
            "ab_block": result.get("ab_block"),
        }
        phase_samples: list[tuple[str, float]] = [
            ("overall_cold_interval", float(duration))
        ]
        phases = result.get("phases")
        if isinstance(phases, Mapping):
            for phase_name, phase in phases.items():
                if not isinstance(phase, Mapping):
                    continue
                phase_status = phase.get("status")
                phase_duration = phase.get("duration_ns")
                if phase_status not in {"observed", "observed_combined"}:
                    continue
                if not isinstance(phase_duration, (int, float)):
                    continue
                if not math.isfinite(float(phase_duration)) or float(phase_duration) <= 0:
                    continue
                phase_samples.append((str(phase_name), float(phase_duration)))
        for phase_name, phase_duration in phase_samples:
            phase_key = key + (phase_name,)
            groups.setdefault(phase_key, []).append(phase_duration)
            contexts.setdefault(phase_key, []).append(context)
    summaries: list[dict[str, object]] = []
    for key, raw in sorted(groups.items()):
        values = [float(value) for value in raw]
        median_ns = float(statistics.median(values))
        mad_ns = float(statistics.median(abs(value - median_ns) for value in values))
        local_seed = seed ^ int.from_bytes(
            hashlib.sha256("/".join(key).encode("utf-8")).digest()[:8], "little"
        )
        rng = random.Random(local_seed)
        bootstrap = [
            statistics.median(rng.choice(values) for _ in values)
            for _ in range(bootstrap_resamples)
        ]
        summaries.append(
            {
                "variant": key[0],
                "backend": key[1],
                "profile": key[2],
                "input_policy": key[3],
                "workload": key[4],
                "comparison_scope": (
                    "overall" if key[5] == "overall_cold_interval" else "phase"
                ),
                "phase": key[5],
                "measured_repetitions": len(raw),
                "raw_durations_ns": raw,
                "raw_clean_cold_interval_ns": (
                    raw if key[5] == "overall_cold_interval" else []
                ),
                "raw_phase_duration_ns": (
                    raw if key[5] != "overall_cold_interval" else []
                ),
                "median_ns": median_ns,
                "mad_ns": mad_ns,
                "p05_ns": _percentile(values, 0.05),
                "p95_ns": _percentile(values, 0.95),
                "bootstrap_median_95_ci_ns": [
                    _percentile(bootstrap, 0.025),
                    _percentile(bootstrap, 0.975),
                ],
                "timing_scope": (
                    "fresh-worker interval through explicit cleanup; provenance, JSON, "
                    "process startup, and exit residual are excluded"
                ),
                "sample_contexts": contexts[key],
            }
        )
    return summaries


def _independent_delta_summary(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    baseline = [float(value) for value in baseline_values]
    candidate = [float(value) for value in candidate_values]
    baseline_median = float(statistics.median(baseline))
    candidate_median = float(statistics.median(candidate))
    delta_percent = (candidate_median - baseline_median) * 100.0 / baseline_median
    rng = random.Random(seed)
    deltas = []
    for _ in range(bootstrap_resamples):
        baseline_sample = [rng.choice(baseline) for _ in baseline]
        candidate_sample = [rng.choice(candidate) for _ in candidate]
        deltas.append(
            float(statistics.median(candidate_sample))
            - float(statistics.median(baseline_sample))
        )
    delta_ci = [_percentile(deltas, 0.025), _percentile(deltas, 0.975)]
    return {
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "candidate_latency_delta_percent": delta_percent,
        "sample_count_baseline": len(baseline),
        "sample_count_candidate": len(candidate),
        "effective_comparison_sample_count": min(len(baseline), len(candidate)),
        "comparison_sample_count": min(len(baseline), len(candidate)),
        "bootstrap_delta_median_95_ci": delta_ci,
        "bootstrap_candidate_latency_delta_95_ci_percent": [
            value * 100.0 / baseline_median for value in delta_ci
        ],
        "pairing": "independent_worker_distributions",
        "review_status": (
            "maintainer_approval_required"
            if delta_percent > 3.0
            else "investigate"
            if delta_percent > 1.0
            else "within_one_percent"
        ),
    }


def _cold_comparisons(
    cold_summaries: Sequence[Mapping[str, object]],
    variants: Sequence[Variant],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    if len(variants) != 2:
        return []
    baseline, candidate = variants
    groups: dict[tuple[str, ...], dict[str, Mapping[str, object]]] = {}
    for summary in cold_summaries:
        variant = str(summary.get("variant"))
        key = (
            str(summary.get("backend")),
            str(summary.get("profile")),
            str(summary.get("input_policy")),
            str(summary.get("workload")),
            str(summary.get("comparison_scope", "overall")),
            str(summary.get("phase", "overall_cold_interval")),
        )
        groups.setdefault(key, {})[variant] = summary
    comparisons = []
    for key, values in groups.items():
        if baseline.name not in values or candidate.name not in values:
            continue
        baseline_raw = values[baseline.name].get(
            "raw_durations_ns", values[baseline.name].get("raw_clean_cold_interval_ns", ())
        )
        candidate_raw = values[candidate.name].get(
            "raw_durations_ns", values[candidate.name].get("raw_clean_cold_interval_ns", ())
        )
        if not isinstance(baseline_raw, (list, tuple)) or not isinstance(candidate_raw, (list, tuple)):
            continue
        comparison = _independent_delta_summary(
            baseline_raw,
            candidate_raw,
            seed=seed ^ int.from_bytes(hashlib.sha256(repr(key).encode("utf-8")).digest()[:8], "little"),
            bootstrap_resamples=bootstrap_resamples,
        )
        comparisons.append(
            {
                "comparison_kind": "cold",
                "backend": key[0],
                "profile": key[1],
                "input_policy": key[2],
                "workload": key[3],
                "comparison_scope": key[4],
                "phase": key[5],
                "baseline_variant": baseline.name,
                "candidate_variant": candidate.name,
                **comparison,
            }
        )
    return comparisons


def _native_ab_comparisons(
    native_evidence: Mapping[str, object],
    variants: Sequence[Variant],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    # Never manufacture a native A/B from an invalid/unsupported manifest.
    # In particular, a supplemental helper may emit timings even when its
    # baseline payload lacks the generic ABI route symbols; those records are
    # diagnostics, not a comparable release distribution.
    if (
        len(variants) != 2
        or native_evidence.get("valid") is not True
        or native_evidence.get("arithmetic_claims_valid") is not True
    ):
        return []
    baseline, candidate = variants
    groups: dict[tuple[object, ...], dict[str, list[float]]] = {}
    for artifact in native_evidence.get("artifacts", ()):
        if not isinstance(artifact, Mapping):
            continue
        validation = artifact.get("validation")
        if not isinstance(validation, Mapping) or validation.get("complete") is not True:
            continue
        variant = str(artifact.get("variant"))
        backend = str(artifact.get("backend"))
        path = artifact.get("path")
        if not isinstance(path, str):
            continue
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        records = payload.get("records") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            continue
        declared_orders = payload.get("profile_orders", ())
        declared_orders = (
            [tuple(str(item) for item in order) for order in declared_orders]
            if isinstance(declared_orders, (list, tuple))
            else []
        )
        raw_workload = payload.get("workload")
        workload_descriptor = (
            dict(raw_workload) if isinstance(raw_workload, Mapping) else {}
        )
        for workload_field in (
            "rows",
            "features",
            "candidates",
            "arity",
            "mi_bins",
            "top_k",
            "dataset_seed",
            "order_seed",
        ):
            if workload_field in payload:
                workload_descriptor[workload_field] = payload.get(workload_field)
        workload_identity = json.dumps(
            _json_safe(workload_descriptor), sort_keys=True
        )
        raw_input_policy = payload.get(
            "input_policy", payload.get("input_policy_name")
        )
        raw_input_identity = payload.get(
            "input_identity", payload.get("dataset_identity")
        )
        # Native records must carry an explicit policy and dataset identity.  A
        # missing value is not a comparable common/native bucket: silently
        # grouping it under JSON null can combine incompatible measurements.
        if (
            not isinstance(raw_input_policy, str)
            or raw_input_policy not in INPUT_POLICIES
            or not isinstance(raw_input_identity, Mapping)
            or not raw_input_identity
        ):
            continue
        input_policy = raw_input_policy
        input_identity = json.dumps(
            _json_safe(raw_input_identity), sort_keys=True
        )
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            # ``samples_us``/``samples_ns`` are normalized to one operation.
            # Raw calibrated regions are retained for audit/target checks and
            # must never be the native latency comparison quantity.
            samples = record.get("samples_us", record.get("samples_ns"))
            if not isinstance(samples, list) or not samples:
                continue
            order_index = record.get("order_index")
            try:
                normalized_order_index: object = int(order_index)
            except (TypeError, ValueError):
                normalized_order_index = None
            raw_order = record.get("profile_order")
            if isinstance(raw_order, (list, tuple)) and raw_order:
                profile_order: object = tuple(str(item) for item in raw_order)
            elif (
                normalized_order_index is not None
                and declared_orders
            ):
                profile_order = declared_orders[
                    normalized_order_index % len(declared_orders)
                ]
            else:
                profile_order = None
            clock = str(record.get("clock", payload.get("timing_clock", "")))
            timing_boundary = str(
                record.get(
                    "timing_scope",
                    record.get("synchronization", payload.get("timing_scope", "")),
                )
            )
            key = (
                backend,
                workload_identity,
                input_policy,
                input_identity,
                str(record.get("profile")),
                str(record.get("operation")),
                str(record.get("metric")),
                normalized_order_index,
                profile_order,
                clock,
                timing_boundary,
                record.get("unit", "us" if "samples_us" in record else "ns"),
            )
            # Keep every record if a helper emits duplicate operation/order
            # rows; assignment would silently discard one distribution.
            groups.setdefault(key, {}).setdefault(variant, []).extend(
                float(value) for value in samples
            )
    comparisons = []
    for key, values in groups.items():
        if baseline.name not in values or candidate.name not in values:
            continue
        comparison = _independent_delta_summary(
            values[baseline.name],
            values[candidate.name],
            seed=seed ^ int.from_bytes(hashlib.sha256(repr(key).encode("utf-8")).digest()[:8], "little"),
            bootstrap_resamples=bootstrap_resamples,
        )
        comparisons.append(
            {
                "comparison_kind": "native",
                "backend": key[0],
                "workload": key[1],
                "input_policy": key[2],
                "input_identity": key[3],
                "profile": key[4],
                "operation": key[5],
                "metric": key[6],
                "order_index": key[7],
                "profile_order": list(key[8]) if isinstance(key[8], tuple) else key[8],
                "clock": key[9],
                "timing_boundary": key[10],
                "duration_unit": key[11],
                "baseline_variant": baseline.name,
                "candidate_variant": candidate.name,
                **comparison,
            }
        )
    return comparisons


def _ab_schedule_readiness(
    schedule: Sequence[Mapping[str, object]], variants: Sequence[Variant], blocks: int
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if len(variants) != 2:
        return {
            "complete": False,
            "failures": [{"reason": "exactly_two_variants_required"}],
            "policy": "comparative release claims require two independent variants",
        }
    if blocks < 2:
        failures.append({"reason": "at_least_two_ab_blocks_required", "blocks": blocks})
    grouped: dict[tuple[object, ...], dict[int, set[tuple[str, ...]]]] = {}
    for item in schedule:
        if item.get("kind") != "public":
            continue
        raw_sequence = item.get("variant_sequence")
        if not isinstance(raw_sequence, list):
            continue
        key = (
            item.get("backend"),
            tuple(item.get("profile_order", ())),
            item.get("order_repeat"),
        )
        grouped.setdefault(key, {}).setdefault(int(item.get("ab_block", -1)), set()).add(
            tuple(str(value) for value in raw_sequence)
        )
    for key, observed_sets in sorted(grouped.items(), key=lambda pair: repr(pair[0])):
        if set(observed_sets) != set(range(blocks)):
            failures.append(
                {"group": list(key), "reason": "missing_ab_block", "observed": sorted(observed_sets)}
            )
            continue
        if any(len(sequences) != 1 for sequences in observed_sets.values()):
            failures.append({"group": list(key), "reason": "inconsistent_variant_sequence"})
            continue
        observed = {block: next(iter(sequences)) for block, sequences in observed_sets.items()}
        first = observed[0]
        reverse = tuple(reversed(first))
        expected_names = {variants[0].name, variants[1].name}
        if set(first) != expected_names or len(first) != 2:
            failures.append({"group": list(key), "reason": "invalid_variant_sequence"})
            continue
        for block in range(blocks):
            expected = first if block % 2 == 0 else reverse
            if observed[block] != expected:
                failures.append(
                    {
                        "group": list(key),
                        "reason": "ab_blocks_are_not_alternating_ab_ba",
                        "block": block,
                        "observed": list(observed[block]),
                        "expected": list(expected),
                    }
                )
    if not grouped:
        failures.append({"reason": "no_public_ab_schedule_observed"})
    return {
        "complete": not failures,
        "failures": failures,
        "policy": "each profile-order block alternates randomized A/B and reversed B/A",
    }


def _coverage_readiness(
    args: argparse.Namespace, schedule: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if args.layer != "all":
        failures.append({"reason": "all_cold_and_public_layers_required", "layer": args.layer})
    backend_profile_sets = {
        tuple(BACKEND_PROFILES[backend]) for backend in args.backends
    }
    if len(backend_profile_sets) != 1:
        failures.append(
            {
                "reason": "heterogeneous_backend_profile_matrices_require_separate_runs",
                "backend_profiles": {
                    backend: list(BACKEND_PROFILES[backend])
                    for backend in args.backends
                },
            }
        )
    required_profiles = set(next(iter(backend_profile_sets), ()))
    if set(args.profiles) != required_profiles or len(args.profiles) != len(required_profiles):
        failures.append(
            {
                "reason": "all_distributed_profiles_for_requested_backends_required",
                "profiles": list(args.profiles),
                "required_profiles": sorted(required_profiles),
            }
        )
    if set(args.surfaces) != set(SURFACES) or len(args.surfaces) != len(SURFACES):
        failures.append({"reason": "all_public_surfaces_required", "surfaces": list(args.surfaces)})
    if set(args.input_policies) != set(INPUT_POLICIES) or len(args.input_policies) != len(INPUT_POLICIES):
        failures.append(
            {"reason": "both_input_policies_required", "input_policies": list(args.input_policies)}
        )
    if set(args.workloads) != set(RELEASE_WORKLOADS) or len(args.workloads) != len(RELEASE_WORKLOADS):
        failures.append({"reason": "release_workload_matrix_required", "workloads": list(args.workloads)})
    if args.warmups < MIN_WARMUPS:
        failures.append({"reason": "warmup_threshold_not_met", "observed": args.warmups})
    if args.repetitions < MIN_REPETITIONS:
        failures.append({"reason": "repetition_threshold_not_met", "observed": args.repetitions})
    if args.bootstrap_resamples < 500:
        failures.append(
            {"reason": "bootstrap_threshold_not_met", "observed": args.bootstrap_resamples}
        )
    configured_min_sample_ns = int(float(args.min_sample_ms) * 1.0e6)
    if configured_min_sample_ns < DEFAULT_MIN_SAMPLE_NS:
        failures.append(
            {
                "reason": "sample_region_floor_not_met",
                "observed_ns": configured_min_sample_ns,
                "required_ns": DEFAULT_MIN_SAMPLE_NS,
            }
        )
    if not args.interleaved_control:
        failures.append({"reason": "interleaved_control_disabled"})
    observed_by_backend: dict[str, set[tuple[str, ...]]] = {}
    observed_surfaces: set[str] = set()
    observed_workloads: set[str] = set()
    observed_policies: set[str] = set()
    observed_control_keys: set[tuple[str, str, str]] = set()
    controls = 0
    for item in schedule:
        if item.get("kind") != "public":
            continue
        backend = str(item.get("backend"))
        observed_by_backend.setdefault(backend, set()).add(
            tuple(str(value) for value in item.get("profile_order", ()))
        )
        observed_surfaces.update(str(value) for value in item.get("surface_order", ()))
        if item.get("workload"):
            observed_workloads.add(str(item["workload"]))
        if item.get("input_policy"):
            observed_policies.add(str(item["input_policy"]))
        if item.get("interleaved_control") is True:
            controls += 1
            observed_control_keys.add(
                (backend, str(item.get("workload")), str(item.get("input_policy")))
            )
    for backend in args.backends:
        expected = set(_profile_orders(BACKEND_PROFILES[backend]))
        observed = observed_by_backend.get(backend, set())
        if observed != expected:
            failures.append(
                {
                    "reason": "profile_order_coverage_incomplete",
                    "backend": backend,
                    "expected": [list(order) for order in sorted(expected)],
                    "observed": [list(order) for order in sorted(observed)],
                }
            )
    if observed_surfaces != set(args.surfaces):
        failures.append({"reason": "scheduled_surface_coverage_incomplete"})
    if observed_workloads != set(args.workloads):
        failures.append({"reason": "scheduled_workload_coverage_incomplete"})
    if observed_policies != set(args.input_policies):
        failures.append({"reason": "scheduled_input_policy_coverage_incomplete"})
    if args.interleaved_control and not controls:
        failures.append({"reason": "no_interleaved_control_schedule_observed"})
    expected_control_keys = {
        (backend, workload, policy)
        for backend in args.backends
        for workload in args.workloads
        for policy in args.input_policies
    }
    if args.interleaved_control and observed_control_keys != expected_control_keys:
        failures.append(
            {
                "reason": "interleaved_control_coverage_incomplete",
                "missing": [
                    list(key) for key in sorted(expected_control_keys - observed_control_keys)
                ],
            }
        )
    return {
        "complete": not failures,
        "failures": failures,
        "observed_profile_orders_by_backend": {
            backend: [list(order) for order in sorted(orders)]
            for backend, orders in sorted(observed_by_backend.items())
        },
        "observed_surfaces": sorted(observed_surfaces),
        "observed_workloads": sorted(observed_workloads),
        "observed_input_policies": sorted(observed_policies),
        "interleaved_control_schedule_count": controls,
        "configured_min_sample_ns": configured_min_sample_ns,
        "observed_interleaved_control_keys": [
            list(key) for key in sorted(observed_control_keys)
        ],
        "policy": (
            "release coverage requires all supported requested profiles, all public "
            "surfaces, six orders where three profiles are supported, every release "
            "workload, both input policies, and an enabled interleaved control"
        ),
    }


def _sample_target_readiness(
    results: Sequence[Mapping[str, object]], expected_repetitions: int
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    observed = 0
    for result_index, result in enumerate(results):
        if result.get("kind") != "public" or result.get("status") != "pass":
            continue
        for cell_index, cell in enumerate(result.get("cells", ())):
            if not isinstance(cell, Mapping) or cell.get("status") != "pass":
                continue
            distribution = cell.get("distribution", {})
            if not isinstance(distribution, Mapping):
                continue
            observed += 1
            if distribution.get("measured_repetitions") != expected_repetitions:
                failures.append(
                    {
                        "result_index": result_index,
                        "cell_index": cell_index,
                        "reason": "recorded_repetition_count_mismatch",
                        "observed": distribution.get("measured_repetitions"),
                        "expected": expected_repetitions,
                    }
                )
            if distribution.get("sample_region_target_met") is not True:
                failures.append(
                    {
                        "result_index": result_index,
                        "cell_index": cell_index,
                        "reason": "sample_region_target_not_met",
                        "target_ns": distribution.get("sample_region_target_ns"),
                        "minimum_observed_ns": distribution.get("sample_region_min_observed_ns"),
                    }
                )
        for control in result.get("interleaved_controls", ()):
            if not isinstance(control, Mapping) or control.get("status") != "pass":
                continue
            profiles = control.get("profiles", {})
            if not isinstance(profiles, Mapping):
                continue
            for profile, profile_result in profiles.items():
                if not isinstance(profile_result, Mapping):
                    continue
                distribution = profile_result.get("distribution", {})
                if not isinstance(distribution, Mapping):
                    continue
                if distribution.get("sample_region_target_met") is not True:
                    failures.append(
                        {
                            "result_index": result_index,
                            "control_profile": str(profile),
                            "reason": "control_sample_region_target_not_met",
                            "target_ns": distribution.get("sample_region_target_ns"),
                            "minimum_observed_ns": distribution.get("sample_region_min_observed_ns"),
                        }
                    )
    if observed == 0:
        failures.append({"reason": "no_public_distributions_observed"})
    return {
        "complete": not failures,
        "observed_distribution_count": observed,
        "failures": failures,
        "policy": "every recorded public region must meet the configured auto-scaling target",
    }


def _result_matrix_readiness(
    results: Sequence[Mapping[str, object]], expected_public_result_count: int | None = None
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    public_result_count = 0
    observed_orders_by_backend: dict[str, set[tuple[str, ...]]] = {}
    for result_index, result in enumerate(results):
        if result.get("kind") != "public":
            continue
        public_result_count += 1
        backend = str(result.get("backend"))
        observed_orders_by_backend.setdefault(backend, set()).add(
            tuple(str(value) for value in result.get("profile_order", ()))
        )
        if result.get("status") != "pass":
            failures.append(
                {"result_index": result_index, "reason": "public_worker_not_pass", "status": result.get("status")}
            )
            continue
        expected_cells = len(result.get("profile_order", ())) * len(
            result.get("surface_order", ())
        )
        cells = result.get("cells", ())
        if len(cells) != expected_cells:
            failures.append(
                {
                    "result_index": result_index,
                    "reason": "public_cell_count_mismatch",
                    "observed": len(cells),
                    "expected": expected_cells,
                }
            )
        for cell_index, cell in enumerate(cells):
            if not isinstance(cell, Mapping):
                failures.append({"result_index": result_index, "cell_index": cell_index, "reason": "cell_not_object"})
                continue
            status = cell.get("status")
            allowed_unsupported = (
                status == "unsupported_by_contract"
                and cell.get("surface") == "graph"
                and backend not in GRAPH_BACKENDS
            )
            if status != "pass" and not allowed_unsupported:
                failures.append(
                    {
                        "result_index": result_index,
                        "cell_index": cell_index,
                        "reason": "public_surface_not_pass",
                        "status": status,
                        "surface": cell.get("surface"),
                        "profile": cell.get("profile"),
                    }
                )
        for control in result.get("interleaved_controls", ()):
            if not isinstance(control, Mapping):
                failures.append({"result_index": result_index, "reason": "control_not_object"})
                continue
            status = control.get("status")
            allowed_unsupported = (
                status == "unsupported_by_contract"
                and control.get("surface") == "graph"
                and backend not in GRAPH_BACKENDS
            )
            if status not in ("pass", "not_applicable") and not allowed_unsupported:
                failures.append(
                    {
                        "result_index": result_index,
                        "reason": "interleaved_control_not_pass",
                        "status": status,
                        "surface": control.get("surface"),
                    }
                )
    if public_result_count == 0:
        failures.append({"reason": "no_public_worker_results"})
    for backend, observed_orders in sorted(observed_orders_by_backend.items()):
        expected_orders = set(_profile_orders(BACKEND_PROFILES.get(backend, ())))
        if observed_orders != expected_orders:
            failures.append(
                {
                    "reason": "result_profile_order_coverage_incomplete",
                    "backend": backend,
                    "expected": [list(order) for order in sorted(expected_orders)],
                    "observed": [list(order) for order in sorted(observed_orders)],
                }
            )
    if (
        expected_public_result_count is not None
        and public_result_count != expected_public_result_count
    ):
        failures.append(
            {
                "reason": "public_worker_schedule_count_mismatch",
                "observed": public_result_count,
                "expected": expected_public_result_count,
            }
        )
    return {
        "complete": not failures,
        "public_result_count": public_result_count,
        "observed_profile_orders_by_backend": {
            backend: [list(order) for order in sorted(orders)]
            for backend, orders in sorted(observed_orders_by_backend.items())
        },
        "expected_public_result_count": expected_public_result_count,
        "failures": failures,
        "policy": "every scheduled public cell must pass, except graph unsupported by the backend contract",
    }


def _threshold_readiness(
    order_sensitivity: Sequence[Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
    *,
    require_comparison: bool = False,
    interleaved_order_sensitivity: Sequence[Mapping[str, object]] = (),
    cold_comparisons: Sequence[Mapping[str, object]] = (),
    native_comparisons: Sequence[Mapping[str, object]] = (),
    require_cold_comparison: bool = False,
    require_native_comparison: bool = False,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index, summary in enumerate(order_sensitivity):
        if summary.get("status") != "no_order_effect_above_one_percent_observed":
            failures.append(
                {
                    "kind": "order_sensitivity",
                    "index": index,
                    "status": summary.get("status"),
                    "threshold_percent": 1.0,
                }
            )
    for index, summary in enumerate(interleaved_order_sensitivity):
        if summary.get("status") != "no_order_effect_above_one_percent_observed":
            failures.append(
                {
                    "kind": "interleaved_order_sensitivity",
                    "index": index,
                    "status": summary.get("status"),
                    "threshold_percent": 1.0,
                }
            )
    all_comparisons = [
        ("public", comparison) for comparison in comparisons
    ] + [("cold", comparison) for comparison in cold_comparisons] + [
        ("native", comparison) for comparison in native_comparisons
    ]
    for index, (comparison_kind, comparison) in enumerate(all_comparisons):
        baseline_count = comparison.get(
            "sample_count_baseline",
            comparison.get(
                "comparison_sample_count",
                comparison.get("effective_comparison_sample_count", 0),
            ),
        )
        candidate_count = comparison.get(
            "sample_count_candidate",
            comparison.get(
                "comparison_sample_count",
                comparison.get("effective_comparison_sample_count", 0),
            ),
        )
        minimum_comparison_samples = (
            2 if comparison_kind == "cold" else MIN_REPETITIONS
        )
        if (
            baseline_count < minimum_comparison_samples
            or candidate_count < minimum_comparison_samples
        ):
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": "bootstrap_sample_count_insufficient",
                    "observed": {
                        "baseline": baseline_count,
                        "candidate": candidate_count,
                    },
                    "required": minimum_comparison_samples,
                }
            )
        if not isinstance(
            comparison.get("bootstrap_candidate_latency_delta_95_ci_percent"),
            list,
        ):
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": "independent_bootstrap_delta_ci_missing",
                }
            )
        if comparison.get("review_status") != "within_one_percent":
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": comparison.get("review_status"),
                    "delta_percent": comparison.get("candidate_latency_delta_percent"),
                    "threshold_percent": 1.0,
                }
            )
    if require_comparison and not comparisons:
        failures.append(
            {
                "kind": "ab_regression",
                "status": "ab_comparison_required",
            }
        )
    if require_cold_comparison and not cold_comparisons:
        failures.append(
            {
                "kind": "cold_regression",
                "status": "cold_comparison_required",
            }
        )
    if require_native_comparison and not native_comparisons:
        failures.append(
            {
                "kind": "native_regression",
                "status": "native_comparison_required",
            }
        )
    return {
        "complete": not failures,
        "failures": failures,
        "policy": (
            "order effects and candidate median latency deltas above one percent "
            "require investigation; above three percent requires maintainer approval; "
            "public/native deltas require 30 raw observations per variant, while "
            "cold deltas require both fresh-worker distributions and use independent "
            "bootstrap intervals"
        ),
    }


def _provenance_readiness(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    observed_variants: set[str] = set()
    for index, result in enumerate(results):
        provenance = result.get("provenance", {})
        missing: list[str] = []
        if not isinstance(provenance, Mapping):
            missing.append("provenance")
            variant = "unknown"
        else:
            variant = str(provenance.get("variant"))
            observed_variants.add(variant)
            if not provenance.get("source_commit"):
                missing.append("source_commit")
            source_tree_state = provenance.get("source_tree_state", {})
            if not isinstance(source_tree_state, Mapping):
                missing.append("source_tree_state")
            elif source_tree_state.get("status") != "clean":
                missing.append("clean_source_tree")
            if not provenance.get("wheel_artifacts"):
                missing.append("wheel_artifact_sha256")
            benchmark_script = provenance.get("benchmark_script", {})
            if not isinstance(benchmark_script, Mapping) or not benchmark_script.get(
                "sha256"
            ):
                missing.append("benchmark_script_sha256")
            if provenance.get("benchmark_script_canonical") is not True:
                missing.append("benchmark_script_canonical")
            wheel_binding = provenance.get("wheel_runtime_binding", {})
            if not isinstance(wheel_binding, Mapping) or not wheel_binding.get("complete"):
                missing.append("wheel_runtime_identity")
            if not provenance.get("loaded_module_files"):
                missing.append("loaded_module_identity")
            affinity = provenance.get("process_affinity")
            affinity_unobservable = bool(
                isinstance(affinity, Mapping)
                and affinity.get("status") == "unavailable"
                and isinstance(affinity.get("detail"), str)
                and affinity.get("detail")
                and str(provenance.get("platform", "")).startswith("macOS-")
            )
            if not isinstance(affinity, Mapping) or (
                affinity.get("status") != "observed" and not affinity_unobservable
            ):
                missing.append("observed_process_affinity")
            elif affinity.get("status") == "observed" and (
                not isinstance(affinity.get("cpus"), list) or not affinity.get("cpus")
            ):
                missing.append("nonempty_process_affinity")
            if not provenance.get("machine"):
                missing.append("machine_identity")
            if not provenance.get("processor"):
                missing.append("processor_identity")
            if not provenance.get("platform"):
                missing.append("platform_identity")
            if not provenance.get("python_executable"):
                missing.append("python_executable")
            interpreter_identity = provenance.get("python_executable_identity")
            if _native_identity_failures(
                interpreter_identity, "python_executable", verify_files=True
            ):
                missing.append("python_executable_identity")
            if not provenance.get("python_version"):
                missing.append("python_version")
            if not isinstance(provenance.get("environment"), Mapping):
                missing.append("environment")
            runtime_dependencies = provenance.get("runtime_dependencies")
            if not isinstance(runtime_dependencies, Mapping):
                missing.append("runtime_dependencies")
            else:
                for dependency in BENCHMARK_RUNTIME_DISTRIBUTIONS:
                    identity = runtime_dependencies.get(dependency)
                    if (
                        not isinstance(identity, Mapping)
                        or identity.get("status") != "observed"
                        or not identity.get("version")
                        or _native_identity_failures(
                            identity.get("record"),
                            f"runtime_dependency_{dependency}_record",
                            verify_files=True,
                        )
                    ):
                        missing.append(f"runtime_dependency_{dependency}")
            toolchains = provenance.get("toolchains")
            required_toolchains = {
                "core": ("rustc", "cc", "cxx", "linker"),
                "cuda": ("rustc", "cc", "cxx", "nvcc", "linker"),
                "rocm": ("rustc", "cc", "cxx", "hipcc", "linker"),
                "metal": ("rustc", "cc", "cxx", "linker"),
            }.get(str(result.get("backend")), ("rustc", "cc", "cxx", "linker"))
            if not isinstance(toolchains, Mapping):
                missing.append("toolchains")
            else:
                for toolchain in required_toolchains:
                    record = toolchains.get(toolchain)
                    if not isinstance(record, Mapping) or record.get("status") != "pass":
                        missing.append(f"toolchain_{toolchain}")
            clock_state = provenance.get("clock_and_power_state")
            if not isinstance(clock_state, Mapping):
                missing.append("clock_and_power_state")
            else:
                if not isinstance(clock_state.get("before"), Mapping):
                    missing.append("clock_state_before")
                if not isinstance(clock_state.get("after"), Mapping):
                    missing.append("clock_state_after")
                backend = str(result.get("backend"))
                device_key = {
                    "cuda": "nvidia_smi",
                    "rocm": "rocm_smi",
                    "metal": "system_profiler",
                }.get(backend)
                if device_key:
                    for phase in ("before", "after"):
                        phase_state = clock_state.get(phase)
                        device_state = (
                            phase_state.get(device_key)
                            if isinstance(phase_state, Mapping)
                            else None
                        )
                        if (
                            not isinstance(device_state, Mapping)
                            or device_state.get("status") != "pass"
                            or not isinstance(device_state.get("output"), str)
                            or not device_state.get("output", "").strip()
                        ):
                            missing.append(f"{device_key}_{phase}")
                        elif backend == "rocm" and not (
                            _rocm_dynamic_telemetry_fields(device_state)
                        ):
                            missing.append(
                                f"{device_key}_{phase}_dynamic_clock_or_power"
                            )
                    device_identity = provenance.get("device_identity")
                    if (
                        not isinstance(device_identity, Mapping)
                        or device_identity.get("status") != "pass"
                        or not str(device_identity.get("output", "")).strip()
                    ):
                        missing.append("device_identity")
            if provenance.get("clock_and_power_capture_point") != (
                "before and after all timed benchmark regions"
            ):
                missing.append("clock_capture_boundary")
            if not provenance.get("driver_command") or not provenance.get(
                "worker_command"
            ):
                missing.append("command")
        if not result.get("native_binaries"):
            missing.append("native_binary_sha256")
        if missing:
            failures.append(
                {"result_index": index, "variant": variant, "missing": missing}
            )
    return {
        "complete": not failures,
        "observed_variants": sorted(observed_variants),
        "incomplete_result_count": len(failures),
        "failures": failures,
        "policy": (
            "Claims require a clean source commit, one canonical benchmark-script hash, wheel "
            "SHA-256 plus embedded RECORD/runtime identity, loaded-module hashes, "
            "native binary SHA-256, command, affinity, toolchain, runtime, and clock "
            "provenance for every worker artifact."
        ),
    }


def _identity_content_fingerprint(identity: object) -> tuple[str, int] | None:
    """Compare file bytes while allowing independent environment paths."""

    if not isinstance(identity, Mapping):
        return None
    digest = identity.get("sha256")
    size = identity.get("size_bytes")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        return None
    if not isinstance(size, int) or size <= 0:
        return None
    return digest.lower(), size


def _runtime_dependencies_fingerprint(dependencies: object) -> dict[str, object] | None:
    if not isinstance(dependencies, Mapping):
        return None
    result: dict[str, object] = {}
    for name in BENCHMARK_RUNTIME_DISTRIBUTIONS:
        identity = dependencies.get(name)
        if not isinstance(identity, Mapping) or identity.get("status") != "observed":
            return None
        record = _identity_content_fingerprint(identity.get("record"))
        if not identity.get("version") or record is None:
            return None
        result[name] = {"version": str(identity["version"]), "record": record}
    return result


def _environment_comparison_view(
    environment: object, provenance: object
) -> dict[str, object] | None:
    """Compare public controls after authenticating variant-bound paths."""

    if not isinstance(environment, Mapping) or not isinstance(provenance, Mapping):
        return None
    adapted_provenance = dict(provenance)
    adapted_provenance["python_executable"] = provenance.get(
        "python_executable_identity"
    )
    return _native_environment_comparison_view(
        {
            "environment": environment,
            "source_root": provenance.get("source_root"),
            "provenance": adapted_provenance,
        }
    )


def _environment_mapping(environment: object) -> dict[str, str] | None:
    """Normalize C++ KEY=value lists and Rust/Python environment mappings."""

    if isinstance(environment, Mapping):
        if any(not isinstance(key, str) for key in environment):
            return None
        return {str(key): str(value) for key, value in environment.items()}
    if not isinstance(environment, (list, tuple)):
        return None
    result: dict[str, str] = {}
    for item in environment:
        if not isinstance(item, str) or "=" not in item:
            return None
        key, value = item.split("=", 1)
        if not key or key in result:
            return None
        result[key] = value
    return result


def _source_root_path(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        for key in ("path", "root"):
            path = value.get(key)
            if isinstance(path, str) and path:
                return path
    return None


def _native_path_tokens(
    validation: Mapping[str, object], environment: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str], tuple[str, int] | None]:
    provenance = validation.get("provenance")
    if not isinstance(provenance, Mapping):
        return {}, {}, None
    exact: dict[str, str] = {}
    roots: dict[str, str] = {}
    source_root = _source_root_path(provenance.get("source_root"))
    if source_root is None:
        source_root = _source_root_path(validation.get("source_root"))
    if source_root:
        roots[_canonical_native_path(source_root)] = "<source_root>"
    for label in (
        "benchmark_source",
        "benchmark_binary",
        "payload",
        "wheel",
        "shader",
        "metallib",
    ):
        identity = provenance.get(label)
        path = identity.get("path") if isinstance(identity, Mapping) else None
        if not isinstance(path, str) or not path:
            continue
        normalized = _canonical_native_path(path)
        exact[normalized] = f"<{label}>"
        parent = normalized.rpartition("/")[0]
        if parent:
            roots.setdefault(parent, f"<{label}_dir>")
    for collection_label in ("wheel_artifacts", "loaded_module_files"):
        collection = provenance.get(collection_label)
        if not isinstance(collection, (list, tuple)):
            continue
        identities = [
            item for item in collection if isinstance(item, Mapping)
        ]
        for index, identity in enumerate(
            sorted(identities, key=lambda item: str(item.get("path", "")))
        ):
            path = identity.get("path")
            if not isinstance(path, str) or not path:
                continue
            normalized = _canonical_native_path(path)
            exact[normalized] = f"<{collection_label}_{index}>"
            parent = normalized.rpartition("/")[0]
            if parent:
                roots.setdefault(parent, f"<{collection_label}_dir>")

    interpreter = provenance.get("python_executable")
    interpreter_fingerprint = _identity_content_fingerprint(interpreter)
    virtual_env = _canonical_native_path(environment.get("VIRTUAL_ENV", ""))
    interpreter_path = (
        interpreter.get("path") if isinstance(interpreter, Mapping) else None
    )
    normalized_interpreter = (
        _canonical_native_path(interpreter_path)
        if isinstance(interpreter_path, str)
        else ""
    )
    interpreter_parent = normalized_interpreter.rpartition("/")[0]
    interpreter_bin = interpreter_parent.rpartition("/")[2].lower()
    inferred_virtual_env = (
        interpreter_parent.rpartition("/")[0]
        if interpreter_bin in {"bin", "scripts"}
        else ""
    )
    authenticated_virtual_env = virtual_env or inferred_virtual_env
    if (
        authenticated_virtual_env
        and interpreter_fingerprint is not None
        and inferred_virtual_env == authenticated_virtual_env
    ):
        roots[authenticated_virtual_env] = "<virtual_env>"
        exact[normalized_interpreter] = "<python_executable>"
    elif virtual_env:
        interpreter_fingerprint = None
    return exact, roots, interpreter_fingerprint


def _canonical_native_path(value: str) -> str:
    """Normalize separators and Windows path case for provenance comparison."""

    normalized = value.replace("\\", "/").rstrip("/")
    if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("//"):
        return normalized.casefold()
    return normalized


def _normalize_native_path(
    value: str,
    *,
    exact: Mapping[str, str],
    roots: Mapping[str, str],
) -> str:
    normalized = _canonical_native_path(value)
    if normalized in exact:
        return str(exact[normalized])
    for root in sorted(roots, key=len, reverse=True):
        if normalized == root:
            return str(roots[root])
        prefix = root + "/"
        if normalized.startswith(prefix):
            return f"{roots[root]}/{normalized[len(prefix):]}"
    return normalized


def _native_search_path_entries(value: str) -> tuple[str, ...]:
    """Split a captured search path without treating a Windows drive as POSIX."""

    if ";" in value:
        return tuple(value.split(";"))
    if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("\\\\", "//")):
        return (value,)
    return tuple(value.split(":")) if value else ()


def _native_environment_comparison_view(
    validation: Mapping[str, object],
) -> dict[str, object] | None:
    environment = _environment_mapping(validation.get("environment"))
    if environment is None:
        return None
    exact, roots, interpreter_fingerprint = _native_path_tokens(
        validation, environment
    )
    if environment.get("VIRTUAL_ENV") and interpreter_fingerprint is None:
        return None
    semantic: dict[str, str] = {}
    paths: dict[str, str] = {}
    for key, value in sorted(environment.items()):
        if key in NATIVE_DIRECT_PATH_ENV_KEYS:
            paths[key] = _normalize_native_path(value, exact=exact, roots=roots)
            continue
        if key in NATIVE_SEARCH_PATH_ENV_KEYS:
            separator = ";" if ";" in value else ":"
            entries = _native_search_path_entries(value)
            paths[key] = separator.join(
                _normalize_native_path(entry, exact=exact, roots=roots)
                for entry in entries
            )
            continue
        semantic[key] = value
    return {
        "semantic": semantic,
        "paths": paths,
        "python_executable_identity": interpreter_fingerprint,
    }


def _clock_power_comparison_view(state: object) -> dict[str, object] | None:
    if not isinstance(state, Mapping):
        return None
    result: dict[str, object] = {}
    for phase in ("before", "after"):
        phase_state = state.get(phase)
        if not isinstance(phase_state, Mapping):
            return None
        phase_view: dict[str, object] = {
            "cpu_governor": _json_safe(phase_state.get("cpu_governor"))
        }
        for key, value in sorted(phase_state.items()):
            if key == "cpu_governor" or not isinstance(value, Mapping):
                continue
            phase_view[str(key)] = {
                "status": value.get("status"),
                "source": value.get("source"),
            }
        result[phase] = phase_view
    return result


def _gpu_clock_power_failures(backend: str, payload: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if payload.get("clock_and_power_capture_point") != (
        "before and after all timed benchmark regions"
    ):
        failures.append("clock_power_capture_boundary_required")
    state = payload.get("clock_and_power_state")
    if not isinstance(state, Mapping):
        return failures + ["clock_and_power_state_required"]
    governors: list[object] = []
    device_key = {
        "cuda": "nvidia_smi",
        "rocm": "rocm_smi",
        "metal": "system_profiler",
    }.get(backend)
    for phase in ("before", "after"):
        phase_state = state.get(phase)
        if not isinstance(phase_state, Mapping):
            failures.append(f"clock_and_power_{phase}_required")
            continue
        governor = phase_state.get("cpu_governor")
        governors.append(governor)
        if not isinstance(governor, Mapping):
            failures.append(f"cpu_governor_{phase}_required")
        else:
            status = governor.get("status")
            values = governor.get("values")
            if status == "observed":
                if not isinstance(values, list) or not values:
                    failures.append(f"cpu_governor_{phase}_values_required")
            elif status == "unavailable":
                if not isinstance(governor.get("detail"), str) or not governor.get(
                    "detail"
                ):
                    failures.append(f"cpu_governor_{phase}_unavailable_detail_required")
            else:
                failures.append(f"cpu_governor_{phase}_status_invalid")
        if device_key:
            device_state = phase_state.get(device_key)
            if (
                not isinstance(device_state, Mapping)
                or device_state.get("status") != "pass"
                or not isinstance(device_state.get("output"), str)
                or not device_state.get("output")
            ):
                failures.append(f"{device_key}_{phase}_observation_required")
            elif backend == "rocm" and not _rocm_dynamic_telemetry_fields(
                device_state
            ):
                failures.append(
                    f"{device_key}_{phase}_dynamic_clock_or_power_required"
                )
        if backend == "metal":
            cpu_power = phase_state.get("cpu_power_management")
            if not isinstance(cpu_power, Mapping):
                failures.append(f"cpu_power_management_{phase}_required")
            elif cpu_power.get("status") == "pass":
                if (
                    not isinstance(cpu_power.get("output"), str)
                    or not cpu_power.get("output")
                ):
                    failures.append(
                        f"cpu_power_management_{phase}_observation_required"
                    )
            elif cpu_power.get("status") == "unavailable":
                if (
                    not isinstance(cpu_power.get("detail"), str)
                    or not cpu_power.get("detail")
                ):
                    failures.append(
                        f"cpu_power_management_{phase}_unavailable_detail_required"
                    )
            else:
                failures.append(f"cpu_power_management_{phase}_status_invalid")
            metal_gpu = phase_state.get("metal_gpu_clock_power")
            if (
                not isinstance(metal_gpu, Mapping)
                or metal_gpu.get("status") != "unavailable"
                or not isinstance(metal_gpu.get("detail"), str)
                or not metal_gpu.get("detail")
            ):
                failures.append(
                    f"metal_gpu_clock_power_{phase}_unavailable_detail_required"
                )
    if len(governors) == 2 and governors[0] != governors[1]:
        failures.append("cpu_governor_changed_during_native_benchmark")
    return failures


def _device_identity_fingerprint(identity: object) -> str | None:
    if not isinstance(identity, Mapping) or identity.get("status") != "pass":
        return None
    output = identity.get("output")
    if not isinstance(output, str) or not output.strip():
        return None
    return re.sub(r"\s+", " ", output).strip()


def _public_stable_provenance_fingerprint(
    provenance: Mapping[str, object],
) -> str:
    benchmark_script = provenance.get("benchmark_script")
    affinity = provenance.get("process_affinity")
    return json.dumps(
        _json_safe(
            {
                "source_commit": provenance.get("source_commit"),
                "wheels": sorted(
                    str(identity.get("sha256"))
                    for identity in provenance.get("wheel_artifacts", ())
                    if isinstance(identity, Mapping) and identity.get("sha256")
                ),
                "machine": provenance.get("machine"),
                "processor": provenance.get("processor"),
                "platform": provenance.get("platform"),
                "python_version": provenance.get("python_version"),
                "python_executable_identity": _identity_content_fingerprint(
                    provenance.get("python_executable_identity")
                ),
                "runtime_dependencies": _runtime_dependencies_fingerprint(
                    provenance.get("runtime_dependencies")
                ),
                "environment": _environment_comparison_view(
                    provenance.get("environment"), provenance
                ),
                "toolchains": provenance.get("toolchains"),
                "benchmark_script_sha256": (
                    benchmark_script.get("sha256")
                    if isinstance(benchmark_script, Mapping)
                    else None
                ),
                "affinity": (
                    affinity.get("cpus") if isinstance(affinity, Mapping) else None
                ),
                "device_identity": _device_identity_fingerprint(
                    provenance.get("device_identity")
                ),
            }
        ),
        sort_keys=True,
    )


def _comparative_input_readiness(
    results: Sequence[Mapping[str, object]], variants: Sequence[Variant]
) -> dict[str, object]:
    """Ensure a two-build comparison is an equivalent-operation A/B run.

    A baseline label alone is not evidence: both variants must be independently
    bound to a clean source/wheel identity, and the two identities must differ.
    This is the path used for an 8df baseline versus the current candidate; it
    prevents comparing a stale public result or the same installation twice.
    """

    public_backends = sorted(
        {
            str(result.get("backend"))
            for result in results
            if result.get("kind") == "public" and result.get("status") == "pass"
        }
    )
    if len(public_backends) > 1:
        reports = {
            backend: _comparative_input_readiness(
                tuple(result for result in results if str(result.get("backend")) == backend),
                variants,
            )
            for backend in public_backends
        }
        failures = [
            {"backend": backend, **dict(failure)}
            for backend, report in reports.items()
            for failure in report.get("failures", ())
            if isinstance(failure, Mapping)
        ]
        commits_by_variant: dict[str, set[str]] = {
            variant.name: set() for variant in variants
        }
        wheels_by_variant: dict[str, set[str]] = {
            variant.name: set() for variant in variants
        }
        for report in reports.values():
            for variant_name, commit in report.get("variant_source_commits", {}).items():
                commits_by_variant.setdefault(str(variant_name), set()).add(str(commit))
            for variant_name, wheel_hashes in report.get("variant_wheel_hashes", {}).items():
                wheels_by_variant.setdefault(str(variant_name), set()).update(
                    str(value) for value in wheel_hashes
                )
        for variant_name, commits in commits_by_variant.items():
            if len(commits) != 1:
                failures.append(
                    {
                        "variant": variant_name,
                        "reason": "public_source_commit_provenance_inconsistent_across_backends",
                        "observed": sorted(commits),
                    }
                )
        return {
            "complete": not failures,
            "failures": failures,
            "variant_source_commits": {
                name: next(iter(values))
                for name, values in commits_by_variant.items()
                if len(values) == 1
            },
            "variant_wheel_hashes": {
                name: sorted(values) for name, values in wheels_by_variant.items()
            },
            "backends": public_backends,
            "policy": (
                "every backend is independently compared across isolated baseline and "
                "candidate environments; no first-backend provenance is reused"
            ),
        }

    failures: list[dict[str, object]] = []
    if len(variants) != 2:
        failures.append({"reason": "exactly_two_variants_required"})
    observed: dict[tuple[str, str], dict[str, object]] = {}
    stable_fingerprints: dict[tuple[str, str], set[str]] = {}
    for result in results:
        if result.get("kind") != "public" or result.get("status") != "pass":
            continue
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        variant = str(provenance.get("variant"))
        backend = str(result.get("backend"))
        key = (variant, backend)
        observed.setdefault(key, dict(provenance))
        stable_fingerprints.setdefault(key, set()).add(
            _public_stable_provenance_fingerprint(provenance)
        )
    expected_names = {variant.name for variant in variants}
    observed_backends = {backend for _, backend in observed}
    expected_keys = {
        (variant_name, backend)
        for variant_name in expected_names
        for backend in observed_backends
    }
    if set(observed) != expected_keys:
        failures.append(
            {
                "reason": "both_independent_variants_required",
                "expected": [list(key) for key in sorted(expected_keys)],
                "observed": [list(key) for key in sorted(observed)],
            }
        )
    for (variant, backend), fingerprints in stable_fingerprints.items():
        if len(fingerprints) != 1:
            failures.append(
                {
                    "variant": variant,
                    "backend": backend,
                    "reason": "public_stable_provenance_changed_between_workers",
                    "fingerprint_count": len(fingerprints),
                }
            )
    commits_by_variant = {
        variant_name: {
            str(provenance.get("source_commit"))
            for (observed_variant, _), provenance in observed.items()
            if observed_variant == variant_name
        }
        for variant_name in expected_names
    }
    for variant_name, commit_values in commits_by_variant.items():
        if len(commit_values) != 1:
            failures.append(
                {
                    "variant": variant_name,
                    "reason": "public_source_commit_provenance_inconsistent",
                    "observed": sorted(commit_values),
                }
            )
    commits = {
        variant_name: next(iter(commit_values))
        for variant_name, commit_values in commits_by_variant.items()
        if len(commit_values) == 1
    }
    wheel_hashes: dict[str, set[str]] = {}
    for variant_name in expected_names:
        wheel_hashes[variant_name] = {
            str(identity.get("sha256"))
            for (observed_variant, _), provenance in observed.items()
            if observed_variant == variant_name
            for identity in provenance.get("wheel_artifacts", ())
            if isinstance(identity, Mapping) and identity.get("sha256")
        }
        if not wheel_hashes[variant_name]:
            failures.append(
                {"variant": variant_name, "reason": "wheel_identity_required"}
            )
    if len(commits) == 2 and len(set(commits.values())) == 1 and len(
        set().union(*wheel_hashes.values())
    ) <= 1:
        failures.append(
            {
                "reason": "baseline_and_candidate_identity_are_identical",
                "source_commits": commits,
                "wheel_hashes": {name: sorted(values) for name, values in wheel_hashes.items()},
            }
        )
    if len(observed) == 2:
        baseline_name, candidate_name = (variant.name for variant in variants)
        active_backend = next(iter(observed_backends))
        baseline_provenance = observed.get((baseline_name, active_backend), {})
        candidate_provenance = observed.get((candidate_name, active_backend), {})
        for field in (
            "machine",
            "processor",
            "platform",
            "python_version",
        ):
            if baseline_provenance.get(field) != candidate_provenance.get(field):
                failures.append(
                    {
                        "reason": "baseline_and_candidate_runtime_mismatch",
                        "field": field,
                        "baseline": baseline_provenance.get(field),
                        "candidate": candidate_provenance.get(field),
                    }
                )
        baseline_interpreter = _identity_content_fingerprint(
            baseline_provenance.get("python_executable_identity")
        )
        candidate_interpreter = _identity_content_fingerprint(
            candidate_provenance.get("python_executable_identity")
        )
        if baseline_interpreter is None or candidate_interpreter is None:
            failures.append(
                {"reason": "baseline_and_candidate_interpreter_identity_required"}
            )
        elif baseline_interpreter != candidate_interpreter:
            failures.append(
                {
                    "reason": "baseline_and_candidate_runtime_mismatch",
                    "field": "python_executable_identity",
                    "baseline": baseline_interpreter,
                    "candidate": candidate_interpreter,
                }
            )
        baseline_environment = _environment_comparison_view(
            baseline_provenance.get("environment"), baseline_provenance
        )
        candidate_environment = _environment_comparison_view(
            candidate_provenance.get("environment"), candidate_provenance
        )
        if baseline_environment is None or candidate_environment is None:
            failures.append({"reason": "baseline_and_candidate_environment_required"})
        elif baseline_environment != candidate_environment:
            failures.append(
                {
                    "reason": "baseline_and_candidate_environment_mismatch",
                    "baseline": baseline_environment,
                    "candidate": candidate_environment,
                }
            )
        baseline_dependencies = _runtime_dependencies_fingerprint(
            baseline_provenance.get("runtime_dependencies")
        )
        candidate_dependencies = _runtime_dependencies_fingerprint(
            candidate_provenance.get("runtime_dependencies")
        )
        if baseline_dependencies is None or candidate_dependencies is None:
            failures.append(
                {"reason": "baseline_and_candidate_runtime_dependencies_required"}
            )
        elif baseline_dependencies != candidate_dependencies:
            failures.append(
                {
                    "reason": "baseline_and_candidate_runtime_dependencies_mismatch",
                    "baseline": baseline_dependencies,
                    "candidate": candidate_dependencies,
                }
            )
        if baseline_provenance.get("toolchains") != candidate_provenance.get(
            "toolchains"
        ):
            failures.append({"reason": "baseline_and_candidate_toolchain_mismatch"})
        baseline_script = baseline_provenance.get("benchmark_script", {})
        candidate_script = candidate_provenance.get("benchmark_script", {})
        baseline_script_hash = (
            baseline_script.get("sha256")
            if isinstance(baseline_script, Mapping)
            else None
        )
        candidate_script_hash = (
            candidate_script.get("sha256")
            if isinstance(candidate_script, Mapping)
            else None
        )
        if not baseline_script_hash or not candidate_script_hash:
            failures.append(
                {
                    "reason": "canonical_benchmark_script_identity_required",
                    "baseline": baseline_script_hash,
                    "candidate": candidate_script_hash,
                }
            )
        elif baseline_script_hash != candidate_script_hash:
            failures.append(
                {
                    "reason": "baseline_and_candidate_benchmark_script_mismatch",
                    "baseline": baseline_script_hash,
                    "candidate": candidate_script_hash,
                }
            )
        baseline_affinity = baseline_provenance.get("process_affinity")
        candidate_affinity = candidate_provenance.get("process_affinity")
        baseline_cpus = (
            baseline_affinity.get("cpus") if isinstance(baseline_affinity, Mapping) else None
        )
        candidate_cpus = (
            candidate_affinity.get("cpus") if isinstance(candidate_affinity, Mapping) else None
        )
        if baseline_cpus != candidate_cpus:
            failures.append(
                {
                    "reason": "baseline_and_candidate_affinity_mismatch",
                    "baseline": baseline_cpus,
                    "candidate": candidate_cpus,
                }
            )
        for variant_name, provenance in (
            (baseline_name, baseline_provenance),
            (candidate_name, candidate_provenance),
        ):
            clock_state = provenance.get("clock_and_power_state", {})
            before_state = (
                clock_state.get("before", {})
                if isinstance(clock_state, Mapping)
                else {}
            )
            after_state = (
                clock_state.get("after", {})
                if isinstance(clock_state, Mapping)
                else {}
            )
            before_governor = (
                before_state.get("cpu_governor")
                if isinstance(before_state, Mapping)
                else None
            )
            after_governor = (
                after_state.get("cpu_governor")
                if isinstance(after_state, Mapping)
                else None
            )
            if before_governor != after_governor:
                failures.append(
                    {
                        "variant": variant_name,
                        "reason": "cpu_governor_changed_during_worker",
                        "before": before_governor,
                        "after": after_governor,
                    }
                )
        for phase in ("before", "after"):
            governor_states = []
            for provenance in (baseline_provenance, candidate_provenance):
                clock_state = provenance.get("clock_and_power_state", {})
                phase_state = (
                    clock_state.get(phase, {})
                    if isinstance(clock_state, Mapping)
                    else {}
                )
                governor_states.append(
                    phase_state.get("cpu_governor")
                    if isinstance(phase_state, Mapping)
                    else None
                )
            if governor_states[0] != governor_states[1]:
                failures.append(
                    {
                        "reason": "baseline_and_candidate_cpu_governor_mismatch",
                        "phase": phase,
                        "baseline": governor_states[0],
                        "candidate": governor_states[1],
                    }
                )
        for backend in observed_backends:
            device_key = {
                "cuda": "nvidia_smi",
                "rocm": "rocm_smi",
                "metal": "system_profiler",
            }.get(backend)
            if not device_key:
                continue
            device_statuses = []
            for provenance in (baseline_provenance, candidate_provenance):
                clock_state = provenance.get("clock_and_power_state", {})
                phases = clock_state if isinstance(clock_state, Mapping) else {}
                phase_statuses = []
                for phase in ("before", "after"):
                    device_snapshot = (
                        phases.get(phase, {}).get(device_key)
                        if isinstance(phases.get(phase), Mapping)
                        else None
                    )
                    phase_statuses.append(
                        (
                            phase,
                            device_snapshot.get("status")
                            if isinstance(device_snapshot, Mapping)
                            else None,
                        )
                    )
                device_statuses.append(tuple(phase_statuses))
            if device_statuses[0] != device_statuses[1]:
                failures.append(
                    {
                        "reason": "baseline_and_candidate_device_provenance_mismatch",
                        "backend": backend,
                    }
                )
        baseline_device = _device_identity_fingerprint(
            baseline_provenance.get("device_identity")
        )
        candidate_device = _device_identity_fingerprint(
            candidate_provenance.get("device_identity")
        )
        if active_backend in ("cuda", "rocm", "metal") and (
            baseline_device is None or candidate_device is None
        ):
            failures.append(
                {
                    "backend": active_backend,
                    "reason": "baseline_and_candidate_device_identity_required",
                    "baseline": baseline_device,
                    "candidate": candidate_device,
                }
            )
        elif baseline_device != candidate_device:
            failures.append(
                {
                    "backend": active_backend,
                    "reason": "baseline_and_candidate_device_identity_mismatch",
                    "baseline": baseline_device,
                    "candidate": candidate_device,
                }
            )
    input_groups: dict[tuple[str, ...], dict[str, object]] = {}
    for result in results:
        if result.get("kind") != "public" or result.get("status") != "pass":
            continue
        provenance = result.get("provenance")
        workload = result.get("workload")
        if not isinstance(provenance, Mapping) or not isinstance(workload, Mapping):
            continue
        variant = str(provenance.get("variant"))
        for cell in result.get("cells", ()):
            if not isinstance(cell, Mapping) or cell.get("status") != "pass":
                continue
            identity = cell.get("input_identity")
            if not isinstance(identity, Mapping):
                failures.append(
                    {
                        "reason": "input_identity_missing",
                        "variant": variant,
                        "surface": cell.get("surface"),
                        "profile": cell.get("profile"),
                    }
                )
                continue
            key = (
                str(result.get("backend")),
                str(workload.get("name")),
                str(result.get("input_policy")),
                str(cell.get("surface")),
                str(cell.get("profile")),
            )
            input_groups.setdefault(key, {})[variant] = dict(identity)
    if len(variants) == 2:
        baseline_name, candidate_name = (variant.name for variant in variants)
        for key, identities in input_groups.items():
            if baseline_name not in identities or candidate_name not in identities:
                continue
            if identities[baseline_name] != identities[candidate_name]:
                failures.append(
                    {
                        "reason": "baseline_and_candidate_input_identity_mismatch",
                        "key": list(key),
                        "baseline": identities[baseline_name],
                        "candidate": identities[candidate_name],
                    }
                )
    return {
        "complete": not failures,
        "failures": failures,
        "variant_source_commits": commits,
        "variant_wheel_hashes": {
            name: sorted(values) for name, values in sorted(wheel_hashes.items())
        },
        "policy": (
            "baseline and candidate execute the same scheduled workload/order/surface "
            "keys, but are independently bound to distinct clean source or wheel identities"
        ),
    }


def _load_native_evidence(path: str) -> dict[str, object]:
    """Load a hash-bound native evidence manifest without manufacturing timings."""

    manifest_path = Path(path).expanduser().resolve()
    raw_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        manifest = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "status": "invalid",
            "arithmetic_claims_valid": False,
            "failures": [f"invalid_json:{exc}"],
            "manifest": None,
        }
    failures: list[str] = []
    if not isinstance(manifest, Mapping):
        failures.append("manifest_root_must_be_object")
        manifest = {}
    if manifest.get("schema") != NATIVE_EVIDENCE_SCHEMA:
        failures.append("schema_mismatch")
    status = str(manifest.get("status", "invalid"))
    if status not in ("validated", "not_collected"):
        failures.append("status_must_be_validated_or_not_collected")
    arithmetic_valid = manifest.get("arithmetic_claims_valid") is True
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        failures.append("artifacts_must_be_list")
        artifacts = []
    normalized_artifacts: list[dict[str, object]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            failures.append(f"artifact_{index}_must_be_object")
            continue
        artifact_path = item.get("path")
        artifact_hash = item.get("sha256")
        backend = item.get("backend")
        kind = item.get("kind")
        variant = item.get("variant")
        if not all(
            isinstance(value, str) and value
            for value in (artifact_path, artifact_hash, backend, kind, variant)
        ):
            failures.append(f"artifact_{index}_missing_variant_path_hash_backend_or_kind")
            continue
        if str(backend) not in BACKEND_ORDER:
            failures.append(f"artifact_{index}_unknown_backend")
        if str(kind) not in NATIVE_EVIDENCE_KINDS:
            failures.append(f"artifact_{index}_unknown_kind")
        resolved_artifact = Path(str(artifact_path))
        if not resolved_artifact.is_absolute():
            resolved_artifact = manifest_path.parent / resolved_artifact
        resolved_artifact = resolved_artifact.resolve()
        if not resolved_artifact.is_file():
            failures.append(f"artifact_{index}_missing_file")
            continue
        observed_hash = _sha256(resolved_artifact)
        if observed_hash != str(artifact_hash):
            failures.append(f"artifact_{index}_sha256_mismatch")
        artifact_validation = _validate_native_artifact(
            resolved_artifact,
            backend=str(backend),
            kind=str(kind),
            manifest_source_commit=manifest.get("source_commit"),
        )
        # Keep the detailed Metal v1 checks alongside the common validator;
        # backend-specific schemas are extensible without reverting to a hash
        # only acceptance path.
        if str(backend) == "metal" and str(kind) == "metal_events":
            metal_validation = _validate_metal_native_timing_artifact(
                resolved_artifact,
                manifest_source_commit=manifest.get("source_commit"),
            )
            if metal_validation.get("status") != "pass":
                for reason in metal_validation["failures"]:
                    failures.append(f"artifact_{index}_metal_validation:{reason}")
            artifact_validation["metal_v1"] = metal_validation
            if metal_validation.get("status") != "pass":
                artifact_validation["complete"] = False
                artifact_validation["status"] = "invalid"
                artifact_validation.setdefault("failures", []).extend(
                    f"metal_v1:{reason}" for reason in metal_validation["failures"]
                )
        for reason in artifact_validation.get("failures", ()):  # type: ignore[union-attr]
            failures.append(f"artifact_{index}_native_validation:{reason}")
        normalized_artifacts.append(
            {
                "backend": str(backend),
                "kind": str(kind),
                "variant": str(variant),
                "source_commit": artifact_validation.get("source_commit"),
                "path": str(resolved_artifact),
                "sha256": str(artifact_hash),
                "size_bytes": resolved_artifact.stat().st_size,
                "validation": artifact_validation,
            }
        )
    if status == "validated":
        if not arithmetic_valid:
            failures.append("validated_manifest_must_set_arithmetic_claims_valid")
        if not normalized_artifacts:
            failures.append("validated_manifest_requires_artifacts")
    elif status == "not_collected":
        if arithmetic_valid:
            failures.append("not_collected_manifest_cannot_validate_arithmetic_claims")
        if artifacts:
            failures.append("not_collected_manifest_must_have_no_artifacts")
    valid = not failures
    if status == "validated":
        source_commit = manifest.get("source_commit")
        if not isinstance(source_commit, str) or re.fullmatch(
            r"[0-9a-fA-F]{40}", source_commit
        ) is None:
            failures.append("validated_manifest_requires_full_source_commit")
        valid = not failures
    source_commits_by_variant: dict[str, str] = {}
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and re.fullmatch(
        r"[0-9a-fA-F]{40}", source_commit
    ) is not None:
        source_commits_by_variant = {
            str(artifact["variant"]): source_commit
            for artifact in normalized_artifacts
            if artifact.get("variant")
        }
    return {
        "path": str(manifest_path),
        "sha256": manifest_hash,
        "status": status if valid else "invalid",
        "arithmetic_claims_valid": bool(valid and status == "validated" and arithmetic_valid),
        "valid": valid,
        "failures": failures,
        "artifacts": normalized_artifacts,
        "source_commits_by_variant": source_commits_by_variant,
        "manifest": _json_safe(manifest),
    }


def _load_native_evidence_specs(
    specs: str | Sequence[tuple[str | None, str]],
) -> dict[str, object]:
    """Load one shared manifest or merge one commit-bound manifest per variant."""

    if isinstance(specs, str):
        return _load_native_evidence(specs)
    loaded: list[tuple[str | None, dict[str, object]]] = []
    failures: list[str] = []
    for name, path in specs:
        try:
            evidence = _load_native_evidence(path)
        except OSError as exc:
            evidence = {
                "path": path,
                "status": "invalid",
                "arithmetic_claims_valid": False,
                "valid": False,
                "failures": [f"manifest_read_error:{exc}"],
                "artifacts": [],
                "manifest": None,
            }
        loaded.append((name, evidence))
        failures.extend(
            f"{name or 'shared'}:{failure}"
            for failure in evidence.get("failures", ())
        )
        if name is not None:
            for artifact in evidence.get("artifacts", ()):
                if isinstance(artifact, Mapping) and artifact.get("variant") != name:
                    failures.append(
                        f"{name}:artifact_variant_mismatch:{artifact.get('variant')}"
                    )
    artifacts = [
        artifact
        for _, evidence in loaded
        for artifact in evidence.get("artifacts", ())
        if isinstance(artifact, Mapping)
    ]
    source_commits_by_variant: dict[str, str] = {}
    for name, evidence in loaded:
        manifest = evidence.get("manifest")
        source_commit = manifest.get("source_commit") if isinstance(manifest, Mapping) else None
        if not isinstance(source_commit, str):
            continue
        artifact_variants = {
            str(artifact.get("variant"))
            for artifact in evidence.get("artifacts", ())
            if isinstance(artifact, Mapping) and artifact.get("variant")
        }
        if name is not None:
            source_commits_by_variant[name] = source_commit
        else:
            for artifact_variant in artifact_variants:
                source_commits_by_variant[artifact_variant] = source_commit
    valid = bool(loaded) and not failures and all(
        evidence.get("valid") is True for _, evidence in loaded
    )
    arithmetic_claims_valid = valid and all(
        evidence.get("arithmetic_claims_valid") is True for _, evidence in loaded
    )
    return {
        "path": [str(evidence.get("path")) for _, evidence in loaded],
        "sha256": [str(evidence.get("sha256")) for _, evidence in loaded],
        "status": "validated" if valid and arithmetic_claims_valid else "invalid",
        "valid": valid,
        "arithmetic_claims_valid": arithmetic_claims_valid,
        "failures": failures,
        "artifacts": artifacts,
        "source_commits_by_variant": source_commits_by_variant,
        "manifest": {
            str(name or "shared"): evidence.get("manifest")
            for name, evidence in loaded
        },
    }


def _validate_metal_native_timing_artifact(
    path: Path, *, manifest_source_commit: object
) -> dict[str, object]:
    """Validate supplemental Metal events plus exact ABI 1.1 payload timing."""

    failures: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "failures": [f"invalid_json:{exc}"],
            "schema": None,
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "invalid",
            "failures": ["root_must_be_object"],
            "schema": None,
        }
    if payload.get("schema") != "gafime.metal.native_timing.v1":
        failures.append("schema_mismatch")
    if payload.get("status") != "pass":
        failures.append("status_not_pass")
    if payload.get("backend") != "metal" or payload.get("profile") != "fp32":
        failures.append("backend_or_profile_mismatch")
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None:
        failures.append("full_source_commit_required")
    if (
        isinstance(manifest_source_commit, str)
        and manifest_source_commit
        and source_commit != manifest_source_commit
    ):
        failures.append("manifest_source_commit_mismatch")
    if payload.get("precision_domains") != {
        "storage": "fp32",
        "pointwise": "fp32",
        "reduction": "fp32",
        "result": "fp32",
    }:
        failures.append("fp32_precision_domains_required")
    warmups = payload.get("warmups")
    repeats = payload.get("repeats")
    if not isinstance(warmups, int) or warmups < MIN_WARMUPS:
        failures.append("warmup_threshold_not_met")
    if not isinstance(repeats, int) or repeats < MIN_REPETITIONS:
        failures.append("repetition_threshold_not_met")
    if payload.get("gpu_timing_supported") is not True:
        failures.append("complete_gpu_timestamp_support_required")

    provenance = payload.get("provenance")
    required_provenance = {
        "benchmark_source",
        "shader_source",
        "benchmark_binary",
        "metallib",
        "payload",
        "wheel",
    }
    if not isinstance(provenance, Mapping) or not required_provenance <= set(provenance):
        failures.append("complete_provenance_identity_set_required")
    else:
        for name in sorted(required_provenance):
            identity = provenance[name]
            if not isinstance(identity, Mapping):
                failures.append(f"provenance_{name}_must_be_object")
                continue
            digest = identity.get("sha256")
            size = identity.get("size_bytes")
            identity_path = identity.get("path")
            if not isinstance(identity_path, str) or not identity_path:
                failures.append(f"provenance_{name}_path_required")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                failures.append(f"provenance_{name}_sha256_required")
            if not isinstance(size, int) or size <= 0:
                failures.append(f"provenance_{name}_positive_size_required")
        for name in ("shader_source", "metallib"):
            failures.extend(_native_identity_failures(provenance.get(name), name))

    records = payload.get("records")
    expected_device_records = {
        ("metric_kernel", "pearson"),
        ("metric_kernel", "r2"),
        ("metric_kernel", "mutual_info"),
        ("metric_kernel", "spearman"),
        ("ranking_kernel", "spearman_target_ranks"),
        ("ranking_topk_and_gather", "spearman"),
    }
    expected_host_operations = {
        "matrix_allocation",
        "h2d_upload_or_unified_write",
        "planning_and_descriptor_materialization",
        "d2h_result_readback",
        "report_construction",
    }
    observed_device_records: set[tuple[str, str]] = set()
    observed_device_record_count = 0
    observed_operations: set[str] = set()
    if not isinstance(records, list) or not records:
        failures.append("records_required")
        records = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            failures.append(f"record_{record_index}_must_be_object")
            continue
        operation = str(record.get("operation"))
        metric = str(record.get("metric"))
        observed_operations.add(operation)
        samples = record.get("samples_us")
        if not isinstance(samples, list) or not isinstance(repeats, int) or len(samples) != repeats:
            failures.append(f"record_{record_index}_raw_sample_count_mismatch")
        elif any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in samples
        ):
            failures.append(f"record_{record_index}_invalid_raw_sample")
        identity = (operation, metric)
        if identity in expected_device_records:
            observed_device_record_count += 1
            observed_device_records.add(identity)
            if record.get("clock") != "metal_command_buffer_gpu_timestamps":
                failures.append(f"record_{record_index}_gpu_clock_required")
            if record.get("synchronization") != (
                "commit_then_waitUntilCompleted_before_timestamp_read"
            ):
                failures.append(f"record_{record_index}_synchronization_mismatch")
            if record.get("gpu_timestamp_valid_samples") != repeats:
                failures.append(f"record_{record_index}_gpu_timestamp_count_mismatch")
            host_samples = record.get("host_synchronized_samples_us")
            if not isinstance(host_samples, list) or len(host_samples) != repeats:
                failures.append(f"record_{record_index}_host_sync_sample_count_mismatch")
    if observed_device_records != expected_device_records:
        failures.append("complete_metric_and_ranking_device_record_set_required")
    if observed_device_record_count != len(expected_device_records):
        failures.append("duplicate_or_extra_metric_device_record")
    if not expected_host_operations <= observed_operations:
        failures.append("complete_host_decomposition_record_set_required")
    boundaries = payload.get("decomposition_boundaries")
    if not isinstance(boundaries, Mapping) or boundaries.get(
        "candidate_materialization"
    ) != "fused into each metric kernel":
        failures.append("candidate_materialization_boundary_required")

    # The standalone shader lane above is useful for command-buffer GPU event
    # timestamps, but it is supplemental evidence only. The claim is accepted
    # only when the same artifact also contains a complete lifecycle through
    # the exact wheel-installed dylib's canonical ABI 1.1 symbols.
    if payload.get("execution_mode") != "supplemental_internal_kernel":
        failures.append("supplemental_execution_mode_required")
    lifecycle = payload.get("canonical_payload_lifecycle")
    required_symbols = {
        "gafime_gpu_numeric_routes_v2",
        "gafime_gpu_matrix_alloc_v2",
        "gafime_gpu_matrix_upload_v2",
        "gafime_gpu_matrix_update_target_v2",
        "gafime_gpu_execute_v2",
        "gafime_gpu_execution_memory_peak_v2",
        "gafime_gpu_permutation_memory_peak_v2",
        "gafime_gpu_permutation_pvalues_v2",
        "gafime_gpu_interaction_diagnostics_v2",
        "gafime_gpu_matrix_free_v2",
    }
    if not isinstance(lifecycle, Mapping):
        failures.append("canonical_payload_lifecycle_required")
    else:
        if lifecycle.get("status") != "validated":
            failures.append("canonical_payload_lifecycle_not_validated")
        if lifecycle.get("schema") != "gafime.native-decomposition.v1":
            failures.append("canonical_payload_lifecycle_schema_mismatch")
        if lifecycle.get("execution_layer") != "installed_payload_dylib":
            failures.append("installed_payload_dylib_execution_required")
        if lifecycle.get("abi") != "canonical_1.1":
            failures.append("canonical_abi1_1_required")
        if lifecycle.get("route_count") != 1:
            failures.append("metal_fp32_route_count_must_be_one")
        if lifecycle.get("mixed_route_rejected") is not True:
            failures.append("metal_mixed_route_rejection_required")
        if set(lifecycle.get("symbols", ())) != required_symbols:
            failures.append("complete_canonical_payload_symbol_set_required")
        if lifecycle.get("records_field") != "canonical_payload_records":
            failures.append("canonical_payload_record_binding_required")

    canonical_records = payload.get("canonical_payload_records")
    expected_canonical_records = {
        ("matrix_allocation", "none"),
        ("h2d_upload_or_unified_write", "none"),
        ("planning_and_descriptor_materialization", "none"),
        ("metric_kernel", "pearson"),
        ("metric_kernel", "r2"),
        ("metric_kernel", "mutual_info"),
        ("metric_kernel", "spearman"),
        ("ranking_kernel", "spearman_target_ranks"),
        ("ranking_topk_and_gather", "spearman"),
        ("d2h_result_readback", "results"),
        ("report_construction", "results"),
    }
    observed_canonical_records: set[tuple[str, str]] = set()
    if not isinstance(canonical_records, list) or not canonical_records:
        failures.append("canonical_payload_records_required")
        canonical_records = []
    for index, record in enumerate(canonical_records):
        if not isinstance(record, Mapping):
            failures.append(f"canonical_record_{index}_must_be_object")
            continue
        operation = str(record.get("operation"))
        metric = str(record.get("metric"))
        identity = (operation, metric)
        observed_canonical_records.add(identity)
        samples = record.get("samples_us")
        if not isinstance(samples, list) or not isinstance(repeats, int) or len(samples) != repeats:
            failures.append(f"canonical_record_{index}_raw_sample_count_mismatch")
        elif any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in samples
        ):
            failures.append(f"canonical_record_{index}_invalid_raw_sample")
        if record.get("clock") != "host_steady_clock_canonical_abi1_1":
            failures.append(f"canonical_record_{index}_clock_mismatch")
        if record.get("synchronization") != (
            "canonical_abi1_1_payload_call_returns_after_device_completion"
        ):
            failures.append(f"canonical_record_{index}_synchronization_mismatch")
    if observed_canonical_records != expected_canonical_records:
        failures.append("complete_canonical_payload_record_set_required")

    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        payload_identity = provenance.get("payload")
        metallib_identity = provenance.get("metallib")
        wheel_identity = provenance.get("wheel")
        wheel_path = (
            Path(str(wheel_identity.get("path"))).expanduser()
            if isinstance(wheel_identity, Mapping) and wheel_identity.get("path")
            else None
        )
        if wheel_path is None or not wheel_path.is_file():
            failures.append("canonical_payload_wheel_file_required")
        else:
            try:
                with zipfile.ZipFile(wheel_path) as archive:
                    wheel_members = {
                        "payload": "gafime/_metal/libgafime_metal_v1.dylib",
                        "metallib": "gafime/_metal/gafime_metal_v1.metallib",
                    }
                    for label, member in wheel_members.items():
                        identity = payload_identity if label == "payload" else metallib_identity
                        if not isinstance(identity, Mapping):
                            failures.append(f"canonical_{label}_identity_required")
                            continue
                        identity_path = Path(str(identity.get("path", ""))).expanduser()
                        if not identity_path.is_file():
                            failures.append(f"canonical_{label}_file_required")
                            continue
                        wheel_hash = hashlib.sha256(archive.read(member)).hexdigest()
                        if wheel_hash != str(identity.get("sha256", "")).lower():
                            failures.append(f"canonical_{label}_wheel_hash_mismatch")
            except (OSError, zipfile.BadZipFile, KeyError) as exc:
                failures.append(f"canonical_payload_wheel_binding_failed:{exc}")
    return {
        "status": "pass" if not failures else "invalid",
        "failures": failures,
        "schema": payload.get("schema"),
        "source_commit": source_commit,
        "warmups": warmups,
        "repeats": repeats,
        "device_record_count": len(observed_device_records),
    }


def _native_operation_names(operation: object, metric: object) -> set[str]:
    """Map backend helper spellings onto one decomposition vocabulary."""

    raw_operation = str(operation or "").strip().lower()
    raw_metric = str(metric or "").strip().lower()
    aliases = {
        "ingest_conversion": "ingest_conversion",
        "input_conversion": "ingest_conversion",
        "matrix_allocation": "allocation",
        "device_alloc": "allocation",
        "allocation": "allocation",
        "h2d_upload": "h2d_upload",
        "h2d": "h2d_upload",
        "matrix_upload": "h2d_upload",
        "h2d_upload_or_unified_write": "h2d_upload",
        "planning": "planning",
        "planning_and_descriptor_materialization": "planning",
        "candidate_materialization": "candidate_materialization",
        "metric_kernel": f"metric:{raw_metric}" if raw_metric else "metric",
        "ranking_kernel": "ranking_target_ranks",
        "target_ranks": "ranking_target_ranks",
        "ranking_target_ranks": "ranking_target_ranks",
        "ranking_topk": "ranking_topk",
        "ranking_topk_and_gather": "ranking_topk",
        "selected_row_gather": "selected_row_gather",
        "selected_rows": "selected_row_gather",
        "d2h_transfer": "d2h_transfer",
        "d2h": "d2h_transfer",
        "d2h_result_readback": "d2h_transfer",
        "report_construction": "report_construction",
        "report": "report_construction",
    }
    canonical = aliases.get(raw_operation)
    if canonical is None:
        return set()
    if raw_operation == "ranking_topk_and_gather":
        return {"ranking_topk", "selected_row_gather"}
    return {canonical}


def _native_identity_failures(
    identity: object, label: str, *, verify_files: bool = True
) -> list[str]:
    failures: list[str] = []
    if not isinstance(identity, Mapping):
        return [f"provenance_{label}_must_be_object"]
    identity_path = identity.get("path")
    digest = identity.get("sha256")
    size = identity.get("size_bytes")
    if not isinstance(identity_path, str) or not identity_path:
        failures.append(f"provenance_{label}_path_required")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        failures.append(f"provenance_{label}_sha256_required")
    if not isinstance(size, int) or size <= 0:
        failures.append(f"provenance_{label}_positive_size_required")
    if verify_files and isinstance(identity_path, str) and identity_path:
        resolved = Path(identity_path).expanduser()
        if not resolved.is_file():
            failures.append(f"provenance_{label}_file_missing")
        elif isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            try:
                if _sha256(resolved) != digest.lower():
                    failures.append(f"provenance_{label}_sha256_mismatch")
                if resolved.stat().st_size != size:
                    failures.append(f"provenance_{label}_size_mismatch")
            except OSError:
                failures.append(f"provenance_{label}_stat_failed")
    return failures


def _native_payload_wheel_failures(
    backend: str, provenance: Mapping[str, object]
) -> list[str]:
    """Bind a native payload identity to the exact wheel member it came from."""

    if backend == "core":
        return []
    wheel = provenance.get("wheel")
    payload = provenance.get("payload")
    if not isinstance(wheel, Mapping) or not isinstance(payload, Mapping):
        return ["native_payload_wheel_binding_required"]
    wheel_path = wheel.get("path")
    payload_sha = str(payload.get("sha256", "")).lower()
    if not isinstance(wheel_path, str) or not wheel_path:
        return ["native_payload_wheel_path_required"]
    if not re.fullmatch(r"[0-9a-f]{64}", payload_sha):
        return ["native_payload_sha256_required"]
    allowed_members = CANONICAL_ABI_WHEEL_MEMBERS.get(backend, frozenset())
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            present_members = [
                member for member in sorted(allowed_members) if member in archive.namelist()
            ]
            matching_members = [
                member
                for member in present_members
                if hashlib.sha256(archive.read(member)).hexdigest() == payload_sha
            ]
    except (OSError, KeyError, zipfile.BadZipFile):
        return ["native_payload_wheel_unreadable"]
    if not present_members:
        return ["native_payload_wheel_member_missing"]
    if not matching_members:
        return ["native_payload_not_from_declared_wheel"]
    return []


def _native_core_wheel_failures(provenance: Mapping[str, object]) -> list[str]:
    wheel = provenance.get("wheel")
    wheel_path = wheel.get("path") if isinstance(wheel, Mapping) else None
    if not isinstance(wheel_path, str) or not wheel_path:
        return ["core_native_wheel_path_required"]
    identity = _wheel_identity(wheel_path)
    failures: list[str] = []
    if identity.get("status") == "invalid":
        failures.append("core_native_wheel_invalid")
    if identity.get("canonical_distribution") != "gafime":
        failures.append("core_native_wheel_distribution_mismatch")
    return failures


def _native_payload_route_failures(
    backend: str, payload: Mapping[str, object], *, kind: str
) -> list[str]:
    """Reject native evidence that cannot exercise the canonical generic ABI.

    The native CUDA helper can still run its supplemental in-binary kernels
    when an old payload only exports dtype-suffixed entry points.  That is
    useful diagnostics, but it is not a comparable measurement of the generic
    ABI 1.1 lifecycle used by the cold/public lanes.  Keep such evidence
    explicitly unsupported so a candidate helper cannot be paired with an
    incompatible baseline payload.
    """

    if backend == "core" or kind == "native_decomposition":
        return []
    if backend == "cuda":
        resolution = payload.get("canonical_payload_resolution")
        if not isinstance(resolution, Mapping):
            return ["native_generic_abi_route_evidence_required"]
        if resolution.get("status") != "resolved":
            return ["native_generic_abi_route_unsupported"]
        observed = {
            str(symbol)
            for symbol in resolution.get("symbols", ())
            if isinstance(symbol, str)
        }
        # The CUDA helper's resolution probe does not query numeric route
        # enumeration; the independently validated lifecycle covers that
        # symbol.  These four calls are the minimum generic payload surface
        # required to distinguish it from the old dtype-suffixed ABI.
        required = {
            "gafime_gpu_matrix_alloc_v2",
            "gafime_gpu_matrix_upload_v2",
            "gafime_gpu_execute_v2",
            "gafime_gpu_matrix_free_v2",
        }
        if not required <= observed:
            return ["native_generic_abi_route_unsupported"]
        return []
    if backend == "rocm":
        checks = payload.get("self_checks")
        if not isinstance(checks, Mapping) or checks.get("canonical_routes") is not True:
            return ["native_generic_abi_route_unsupported"]
        return []
    if backend == "metal":
        lifecycle = payload.get("canonical_payload_lifecycle")
        observed = (
            {
                str(symbol)
                for symbol in lifecycle.get("symbols", ())
                if isinstance(symbol, str)
            }
            if isinstance(lifecycle, Mapping)
            else set()
        )
        if (
            not isinstance(lifecycle, Mapping)
            or lifecycle.get("status") != "validated"
            or not CANONICAL_ABI_GENERIC_SYMBOLS <= observed
        ):
            return ["native_generic_abi_route_unsupported"]
        return []
    return ["native_generic_abi_route_unsupported"]


def _compiler_provenance_failures(
    backend: str, compiler: object
) -> list[str]:
    """Require observed backend tool records while allowing optional fields."""

    if not isinstance(compiler, Mapping) or not compiler:
        return ["compiler_provenance_required"]

    def has_value(name: str) -> bool:
        value = compiler.get(name)
        return value is not None and str(value).strip() != ""

    def observed_tool(name: str) -> bool:
        record = compiler.get(name)
        return (
            isinstance(record, Mapping)
            and record.get("status") == "observed"
            and isinstance(record.get("version"), str)
            and bool(record.get("version", "").strip())
        )

    failures: list[str] = []
    if backend == "core":
        if not any(has_value(name) for name in ("rustc", "rustc_version")):
            failures.append("compiler_rustc_version_required")
    elif backend == "cuda":
        if not all(has_value(name) for name in ("nvcc_major", "nvcc_minor")):
            failures.append("compiler_nvcc_version_required")
        for name in ("nvcc", "host_cxx", "linker"):
            if not observed_tool(name):
                failures.append(f"compiler_{name}_observed_version_required")
    elif backend == "rocm":
        if not has_value("predefined_version"):
            failures.append("compiler_rocm_predefined_version_required")
        for name in ("hipcc", "clangxx", "linker"):
            if not observed_tool(name):
                failures.append(f"compiler_{name}_observed_version_required")
    elif backend == "metal":
        if not has_value("clang"):
            failures.append("compiler_clang_version_required")
    else:
        if not any(has_value(name) for name in compiler):
            failures.append("compiler_version_required")
    return failures


def _canonical_lifecycle_failures(
    payload: object,
    *,
    backend: str,
    source_commit: object,
    artifact_provenance: object,
) -> list[str]:
    """Authenticate an independently executed canonical ABI lifecycle."""

    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return ["canonical_payload_lifecycle_must_be_json_object"]
    if payload.get("schema") != "gafime.native-decomposition.v1":
        failures.append("canonical_payload_lifecycle_schema_mismatch")
    if payload.get("status") != "pass":
        failures.append("canonical_payload_lifecycle_status_invalid")
    if payload.get("execution_mode") != "canonical_payload":
        failures.append("canonical_payload_lifecycle_must_be_canonical")
    if payload.get("execution_layer") != "independent_abi_1_1_c_consumer":
        failures.append("canonical_payload_lifecycle_independent_consumer_required")
    if payload.get("abi") != "canonical_1.1":
        failures.append("canonical_payload_lifecycle_abi_mismatch")
    if payload.get("backend") != backend:
        failures.append("canonical_payload_lifecycle_backend_mismatch")
    if payload.get("source_commit") != source_commit:
        failures.append("canonical_payload_lifecycle_source_commit_mismatch")
    tree_state = payload.get("source_tree_state")
    if not isinstance(tree_state, Mapping) or tree_state.get("status") != "clean":
        failures.append("canonical_payload_lifecycle_clean_source_tree_required")
    expected_profiles = set(BACKEND_PROFILES.get(backend, ()))
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or set(map(str, raw_profiles)) != expected_profiles:
        failures.append("canonical_payload_lifecycle_profile_coverage_incomplete")
    raw_operations = payload.get("operations")
    if not isinstance(raw_operations, list) or set(map(str, raw_operations)) != set(
        CANONICAL_ABI_LIFECYCLE_OPERATIONS
    ):
        failures.append("canonical_payload_lifecycle_operation_coverage_incomplete")

    result = payload.get("consumer_result")
    marker: object = None
    if not isinstance(result, Mapping):
        failures.append("canonical_payload_lifecycle_consumer_result_required")
    else:
        if result.get("schema") != "gafime.abi-1.1-consumer-result.v1":
            failures.append("canonical_payload_lifecycle_consumer_schema_mismatch")
        if result.get("status") != "pass" or result.get("returncode") != 0:
            failures.append("canonical_payload_lifecycle_consumer_did_not_pass")
        marker = result.get("marker")
    if not isinstance(marker, Mapping):
        failures.append("canonical_payload_lifecycle_consumer_marker_required")
    else:
        if marker.get("schema") != "gafime.abi-1.1-consumer-result.v1":
            failures.append("canonical_payload_lifecycle_marker_schema_mismatch")
        if marker.get("status") != "pass":
            failures.append("canonical_payload_lifecycle_marker_status_invalid")
        if marker.get("backend_kind") != CANONICAL_ABI_BACKEND_KINDS.get(backend):
            failures.append("canonical_payload_lifecycle_marker_backend_mismatch")
        if marker.get("route_count") != len(expected_profiles):
            failures.append("canonical_payload_lifecycle_marker_route_count_mismatch")
        marker_operations = marker.get("operations")
        if not isinstance(marker_operations, list) or set(map(str, marker_operations)) != set(
            CANONICAL_ABI_LIFECYCLE_OPERATIONS
        ):
            failures.append("canonical_payload_lifecycle_marker_operations_incomplete")

    provenance = payload.get("provenance")
    required_identities = {"payload", "wheel", "consumer_binary", "consumer_source"}
    if not isinstance(provenance, Mapping):
        failures.append("canonical_payload_lifecycle_provenance_required")
        provenance = {}
    for name in sorted(required_identities):
        failures.extend(
            _native_identity_failures(
                provenance.get(name), f"canonical_payload_lifecycle_{name}"
            )
        )
    if isinstance(artifact_provenance, Mapping):
        for name in ("payload", "wheel"):
            lifecycle_identity = provenance.get(name)
            artifact_identity = artifact_provenance.get(name)
            if (
                isinstance(lifecycle_identity, Mapping)
                and isinstance(artifact_identity, Mapping)
                and str(lifecycle_identity.get("sha256", "")).lower()
                != str(artifact_identity.get("sha256", "")).lower()
            ):
                failures.append(
                    f"canonical_payload_lifecycle_{name}_sha256_mismatch"
                )

    member = payload.get("wheel_member")
    member_digest = payload.get("wheel_member_sha256")
    allowed_members = CANONICAL_ABI_WHEEL_MEMBERS.get(backend, frozenset())
    if member not in allowed_members:
        failures.append("canonical_payload_lifecycle_wheel_member_mismatch")
    elif isinstance(provenance, Mapping):
        wheel_identity = provenance.get("wheel")
        payload_identity = provenance.get("payload")
        wheel_path = (
            Path(str(wheel_identity.get("path"))).expanduser()
            if isinstance(wheel_identity, Mapping) and wheel_identity.get("path")
            else None
        )
        if wheel_path is not None and wheel_path.is_file():
            try:
                with zipfile.ZipFile(wheel_path) as archive:
                    embedded_digest = hashlib.sha256(archive.read(str(member))).hexdigest()
                if embedded_digest != member_digest:
                    failures.append("canonical_payload_lifecycle_wheel_member_sha256_mismatch")
                if (
                    not isinstance(payload_identity, Mapping)
                    or embedded_digest
                    != str(payload_identity.get("sha256", "")).lower()
                ):
                    failures.append("canonical_payload_lifecycle_payload_not_from_wheel")
            except (KeyError, OSError, zipfile.BadZipFile):
                failures.append("canonical_payload_lifecycle_wheel_member_unreadable")
    return failures


def _validate_native_artifact(
    path: Path,
    *,
    backend: str,
    kind: str,
    manifest_source_commit: object,
) -> dict[str, object]:
    """Validate the content and coverage of one backend-native artifact.

    The manifest hash protects transport integrity only.  This validator also
    authenticates the artifact's schema, source commit, profile records,
    operation decomposition, raw sample counts, and provenance identities.
    """

    failures: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "complete": False, "failures": [f"invalid_json:{exc}"]}
    if not isinstance(payload, Mapping):
        return {
            "status": "invalid",
            "complete": False,
            "failures": ["root_must_be_object"],
        }
    schema = payload.get("schema", payload.get("format"))
    allowed_schemas = NATIVE_ARTIFACT_SCHEMAS.get(kind, {}).get(backend, frozenset())
    if schema not in allowed_schemas:
        failures.append(
            f"backend_schema_mismatch:expected={sorted(allowed_schemas)}:observed={schema}"
        )
    if payload.get("status") not in ("pass", "validated"):
        failures.append("status_not_pass_or_validated")
    payload_backend = payload.get("backend")
    if payload_backend is not None and str(payload_backend) != backend:
        failures.append("backend_mismatch")
    source_commit = payload.get("source_commit", payload.get("git_head"))
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None:
        failures.append("full_source_commit_required")
    if (
        isinstance(manifest_source_commit, str)
        and manifest_source_commit
        and source_commit != manifest_source_commit
    ):
        failures.append("manifest_source_commit_mismatch")

    source_tree_state = payload.get("source_tree_state")
    if not isinstance(source_tree_state, Mapping):
        failures.append("source_tree_state_required")
    elif source_tree_state.get("status") != "clean":
        failures.append("clean_source_tree_required")

    raw_input_policy = payload.get(
        "input_policy", payload.get("input_policy_name")
    )
    if not isinstance(raw_input_policy, str) or raw_input_policy not in INPUT_POLICIES:
        failures.append("input_policy_required")
    raw_input_identity = payload.get(
        "input_identity", payload.get("dataset_identity")
    )
    if not isinstance(raw_input_identity, Mapping) or not raw_input_identity:
        failures.append("input_identity_required")
    environment = _environment_mapping(payload.get("environment"))
    if environment is None:
        failures.append("environment_provenance_mapping_or_key_value_list_required")
    if backend in ("cuda", "rocm", "metal"):
        failures.extend(_gpu_clock_power_failures(backend, payload))

    if kind != "native_decomposition":
        compiler = payload.get("compiler")
        failures.extend(_compiler_provenance_failures(backend, compiler))
        if backend in ("cuda", "rocm", "metal"):
            device = payload.get("device")
            if not isinstance(device, Mapping) or not device:
                failures.append("device_hardware_provenance_required")
            elif not any(
                key in device and device.get(key) not in (None, "")
                for key in ("name", "id", "registry_id", "gcn_arch", "compute_major")
            ):
                failures.append("device_identity_required")
        else:
            device = payload.get("device")
            if (
                not isinstance(device, Mapping)
                or not isinstance(device.get("identity"), str)
                or not str(device.get("identity")).strip()
            ):
                failures.append("cpu_hardware_identity_required")
            affinity = payload.get("process_affinity", payload.get("affinity"))
            if not isinstance(affinity, (Mapping, list, tuple, str)) or not affinity:
                failures.append("process_affinity_provenance_required")
            clock = payload.get("clock", payload.get("timing_clock"))
            if not isinstance(clock, (Mapping, str)) or not clock:
                failures.append("clock_provenance_required")
            clock_power = payload.get("clock_and_power_state")
            before = (
                clock_power.get("before")
                if isinstance(clock_power, Mapping)
                else None
            )
            after = (
                clock_power.get("after")
                if isinstance(clock_power, Mapping)
                else None
            )
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                failures.append("cpu_before_after_governor_provenance_required")
            elif before.get("cpu_governor") != after.get("cpu_governor"):
                failures.append("cpu_governor_changed_during_native_benchmark")

    raw_profiles = payload.get("profiles")
    if raw_profiles is None:
        raw_profile = payload.get("profile")
        raw_profiles = [raw_profile] if raw_profile is not None else []
    if not isinstance(raw_profiles, (list, tuple)):
        failures.append("profiles_must_be_list")
        raw_profiles = []
    profiles = {str(profile) for profile in raw_profiles if isinstance(profile, str)}
    supported_profiles = set(BACKEND_PROFILES.get(backend, ()))
    if not profiles:
        failures.append("profile_coverage_required")
    if not profiles <= supported_profiles:
        failures.append("unsupported_profile_in_native_artifact")

    warmups = payload.get("warmups")
    repeats = payload.get("repeats", payload.get("repetitions"))
    if not isinstance(warmups, int) or warmups < MIN_WARMUPS:
        failures.append("warmup_threshold_not_met")
    if not isinstance(repeats, int) or repeats < MIN_REPETITIONS:
        failures.append("repetition_threshold_not_met")
    if backend == "core" and kind == "core_microbenchmark":
        target_region_ns = payload.get("target_region_ns")
        if not isinstance(target_region_ns, int) or target_region_ns <= 0:
            failures.append("core_native_target_region_required")
        if payload.get("measurement_scope") != "native_arithmetic_only":
            failures.append("core_native_measurement_scope_required")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append("complete_provenance_required")
        provenance = {}
    required_provenance = NATIVE_REQUIRED_PROVENANCE_BY_BACKEND.get(
        backend, NATIVE_REQUIRED_PROVENANCE
    )
    missing_provenance = required_provenance - set(provenance)
    failures.extend(f"missing_provenance_{name}" for name in sorted(missing_provenance))
    for name in sorted(required_provenance & set(provenance)):
        failures.extend(_native_identity_failures(provenance[name], name))
    if isinstance(environment, Mapping) and environment.get("VIRTUAL_ENV"):
        interpreter = provenance.get("python_executable")
        virtual_env = str(environment["VIRTUAL_ENV"]).replace("\\", "/").rstrip("/")
        interpreter_path = (
            interpreter.get("path") if isinstance(interpreter, Mapping) else None
        )
        normalized_interpreter = (
            interpreter_path.replace("\\", "/").rstrip("/")
            if isinstance(interpreter_path, str)
            else ""
        )
        if not (
            normalized_interpreter.startswith(virtual_env + "/bin/")
            or normalized_interpreter.startswith(virtual_env + "/Scripts/")
        ):
            failures.append("python_executable_not_bound_to_virtual_env")
    if backend == "core":
        failures.extend(_native_core_wheel_failures(provenance))
    failures.extend(_native_payload_wheel_failures(backend, provenance))
    failures.extend(_native_payload_route_failures(backend, payload, kind=kind))

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        failures.append("records_required")
        records = []
    observed_by_profile: dict[str, set[str]] = {profile: set() for profile in profiles}
    native_statistics: list[dict[str, object]] = []
    observed_orders: set[tuple[str, ...]] = set()
    observed_order_indices: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            failures.append(f"record_{index}_must_be_object")
            continue
        record_profile = record.get("profile")
        record_profiles = (
            [str(record_profile)]
            if isinstance(record_profile, str)
            else list(profiles)
            if len(profiles) == 1
            else []
        )
        if not record_profiles:
            failures.append(f"record_{index}_profile_required")
        raw_order = record.get("profile_order")
        if isinstance(raw_order, (list, tuple)) and raw_order:
            observed_orders.add(tuple(str(item) for item in raw_order))
        raw_order_index = record.get("order_index")
        if isinstance(raw_order_index, int) and raw_order_index >= 0:
            observed_order_indices.add(raw_order_index)
        metric = record.get("metric")
        operations = _native_operation_names(record.get("operation"), metric)
        if not operations:
            failures.append(f"record_{index}_unknown_operation")
        if (
            backend == "core"
            and kind == "core_microbenchmark"
            and "report_construction" in operations
        ):
            failures.append("core_native_report_construction_must_not_be_claimed")
        samples = record.get("samples_us", record.get("samples_ns"))
        raw_durations = record.get(
            "raw_samples_us",
            record.get("raw_samples_ns", record.get("raw_region_ns", samples)),
        )
        if not isinstance(samples, list):
            failures.append(f"record_{index}_normalized_samples_required")
        else:
            if isinstance(repeats, int) and len(samples) != repeats:
                failures.append(f"record_{index}_raw_sample_count_mismatch")
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in samples
            ):
                failures.append(f"record_{index}_invalid_raw_sample")
            if samples and all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in samples
            ):
                if not isinstance(raw_durations, list):
                    failures.append(f"record_{index}_raw_duration_list_required")
                    raw_durations = []
                elif isinstance(repeats, int) and len(raw_durations) != repeats:
                    failures.append(f"record_{index}_raw_duration_count_mismatch")
                if any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in raw_durations
                ):
                    failures.append(f"record_{index}_invalid_raw_duration")
                if raw_durations and all(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and float(value) > 0.0
                    for value in raw_durations
                ):
                    loop_count = record.get(
                        "loop_count_per_sample", record.get("loop_count")
                    )
                    target = record.get("sample_region_target_us")
                    if target is None:
                        target = record.get("sample_region_target_ns")
                    if target is None:
                        target = payload.get("target_region_ns")
                    target_met = record.get("sample_region_target_met")
                    auto_scaling: Mapping[str, object] | None = None
                    if (
                        isinstance(loop_count, int)
                        and loop_count > 0
                        and isinstance(target, (int, float))
                        and math.isfinite(float(target))
                        and float(target) > 0.0
                    ):
                        auto_scaling = {
                            "status": "observed",
                            "loop_count_per_sample": loop_count,
                            "target_duration": float(target),
                            "target_unit": (
                                "us"
                                if record.get("sample_region_target_us") is not None
                                else "ns"
                            ),
                            "target_met": target_met,
                        }
                        if target_met is False:
                            failures.append(
                                f"record_{index}_auto_scaling_target_not_met"
                            )
                    native_statistics.append(
                        {
                            "profile": record_profiles[0] if record_profiles else None,
                            "operation": record.get("operation"),
                            "metric": record.get("metric"),
                            "statistics": _native_record_statistics(
                                samples,
                                raw_values=raw_durations,
                                seed=int.from_bytes(
                                    hashlib.sha256(
                                        f"{backend}/{index}/{record.get('operation')}/{record.get('metric')}".encode(
                                            "utf-8"
                                        )
                                    ).digest()[:8],
                                    "little",
                                ),
                                unit=(
                                    "us"
                                    if isinstance(record.get("samples_us"), list)
                                    or isinstance(record.get("raw_samples_us"), list)
                                    else "ns"
                                ),
                                auto_scaling=auto_scaling,
                            ),
                        }
                    )
        for record_profile_name in record_profiles:
            if record_profile_name not in profiles:
                failures.append(f"record_{index}_profile_not_declared")
                continue
            observed_by_profile.setdefault(record_profile_name, set()).update(operations)

    boundaries = payload.get("decomposition_boundaries", payload.get("decomposition"))
    if isinstance(boundaries, Mapping):
        boundary_operations: set[str] = set()
        for operation in ("candidate_materialization", "ingest_conversion"):
            boundary = str(boundaries.get(operation, "")).lower()
            if (
                "material" in boundary
                or "convert" in boundary
                or "fused" in boundary
                or "not present" in boundary
                or "not applicable" in boundary
            ):
                boundary_operations.add(operation)
        if boundary_operations:
            observed_by_profile = {
                profile: operations | boundary_operations
                for profile, operations in observed_by_profile.items()
            }
    else:
        failures.append("decomposition_boundaries_required")

    incomplete_profiles: dict[str, list[str]] = {}
    required_operations = NATIVE_REQUIRED_OPERATIONS_BY_BACKEND.get(
        backend, NATIVE_REQUIRED_OPERATIONS
    )
    for profile in sorted(profiles):
        missing = sorted(required_operations - observed_by_profile.get(profile, set()))
        if missing:
            incomplete_profiles[profile] = missing
    if incomplete_profiles:
        failures.append("complete_decomposition_profile_coverage_required")

    if backend in ("cuda", "rocm") and len(profiles) == 3:
        expected_orders = set(itertools.permutations(PROFILE_ORDER))
        declared_orders = payload.get("profile_orders")
        declared_order_set = {
            tuple(str(item) for item in order)
            for order in declared_orders
            if isinstance(order, (list, tuple))
        } if isinstance(declared_orders, (list, tuple)) else set()
        record_orders = set(observed_orders)
        if declared_order_set and observed_order_indices:
            declared_order_list = [
                tuple(str(item) for item in order)
                for order in declared_orders
                if isinstance(order, (list, tuple))
            ]
            if declared_order_list:
                record_orders.update(
                    declared_order_list[index % len(declared_order_list)]
                    for index in observed_order_indices
                )
        all_orders = record_orders
        if all_orders != expected_orders:
            failures.append("all_six_native_profile_orders_required")

    if kind in {"cuda_events", "rocm_events", "device_events"}:
        device_operations = NATIVE_DEVICE_TIMED_OPERATIONS_BY_BACKEND.get(backend, frozenset())
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            operations = _native_operation_names(record.get("operation"), record.get("metric"))
            clock = str(record.get("clock", "")).lower()
            synchronization = str(
                record.get("synchronization", record.get("timing_scope", ""))
            ).lower()
            if not clock or not synchronization:
                failures.append(f"record_{index}_clock_and_synchronization_required")
                continue
            if operations & device_operations:
                expected_clock = "cuda_event" if backend == "cuda" else "hip_event"
                if expected_clock not in clock:
                    failures.append(
                        f"record_{index}_device_event_clock_required:{expected_clock}"
                    )
                sync_tokens = ("sync", "synchron", "event") if backend == "cuda" else ("sync", "synchron")
                if not any(token in synchronization for token in sync_tokens):
                    failures.append(f"record_{index}_device_synchronization_required")

    execution_mode = str(payload.get("execution_mode", "canonical_payload"))
    if (
        backend in ("cuda", "rocm", "metal")
        and kind == "native_decomposition"
        and execution_mode != "supplemental_internal_kernel"
    ):
        lifecycle = payload.get("canonical_payload_lifecycle")
        if not isinstance(lifecycle, Mapping) or lifecycle.get("status") != "validated":
            failures.append("native_decomposition_requires_validated_canonical_lifecycle")
        else:
            failures.extend(
                _canonical_lifecycle_failures(
                    lifecycle,
                    backend=backend,
                    source_commit=source_commit,
                    artifact_provenance=provenance,
                )
            )

    # The CUDA helper uses the canonical payload when it can resolve it, but
    # keeps a direct internal-kernel lane for supplemental decomposition.  Such
    # evidence can never stand alone: a separately captured stable-ABI lifecycle
    # artifact must be present and hash-bound.
    execution_mode = str(payload.get("execution_mode", "canonical_payload"))
    canonical_lifecycle = payload.get("canonical_payload_lifecycle")
    if execution_mode == "supplemental_internal_kernel":
        # Metal's helper keeps GPUStartTime/GPUEndTime supplemental records and
        # embeds the exact installed-payload ABI 1.1 lifecycle in the same
        # hash-bound artifact. CUDA/HIP helpers may continue to bind a separate
        # canonical lifecycle artifact through the path/sha256 branch below.
        inline_metal_lifecycle = (
            backend == "metal"
            and isinstance(canonical_lifecycle, Mapping)
            and canonical_lifecycle.get("execution_layer") == "installed_payload_dylib"
            and canonical_lifecycle.get("records_field") == "canonical_payload_records"
        )
        if inline_metal_lifecycle:
            pass
        elif not isinstance(canonical_lifecycle, Mapping) or canonical_lifecycle.get("status") != "validated":
            failures.append("supplemental_internal_kernel_requires_canonical_payload_lifecycle")
        elif canonical_lifecycle.get("schema") not in (
            "gafime.native-decomposition.v1",
            "gafime.cuda.native_timing.v2",
        ):
            failures.append("canonical_payload_lifecycle_schema_mismatch")
        else:
            lifecycle_path = canonical_lifecycle.get("path")
            lifecycle_sha = canonical_lifecycle.get("sha256")
            if not isinstance(lifecycle_path, str) or not isinstance(lifecycle_sha, str):
                failures.append("canonical_payload_lifecycle_identity_required")
            else:
                lifecycle_file = Path(lifecycle_path).expanduser()
                if not lifecycle_file.is_file():
                    failures.append("canonical_payload_lifecycle_file_missing")
                elif _sha256(lifecycle_file) != lifecycle_sha:
                    failures.append("canonical_payload_lifecycle_sha256_mismatch")
                else:
                    try:
                        lifecycle_payload = json.loads(
                            lifecycle_file.read_text(encoding="utf-8")
                        )
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        lifecycle_payload = None
                    if not isinstance(lifecycle_payload, Mapping):
                        failures.append("canonical_payload_lifecycle_must_be_json_object")
                    elif lifecycle_payload.get("status") not in ("pass", "validated"):
                        failures.append("canonical_payload_lifecycle_status_invalid")
                    elif lifecycle_payload.get("execution_mode") == "supplemental_internal_kernel":
                        failures.append("canonical_payload_lifecycle_must_not_be_supplemental")
                    failures.extend(
                        _canonical_lifecycle_failures(
                            lifecycle_payload,
                            backend=backend,
                            source_commit=source_commit,
                            artifact_provenance=provenance,
                        )
                    )
                    lifecycle_source_commit = (
                        lifecycle_payload.get(
                            "source_commit", lifecycle_payload.get("git_head")
                        )
                        if isinstance(lifecycle_payload, Mapping)
                        else None
                    )
                    if not isinstance(lifecycle_source_commit, str) or re.fullmatch(
                        r"[0-9a-fA-F]{40}", lifecycle_source_commit
                    ) is None:
                        failures.append(
                            "canonical_payload_lifecycle_full_source_commit_required"
                        )
                    elif lifecycle_source_commit != source_commit:
                        failures.append("canonical_payload_lifecycle_source_commit_mismatch")
                    if isinstance(lifecycle_payload, Mapping) and lifecycle_payload.get(
                        "backend"
                    ) not in (None, backend):
                        failures.append("canonical_payload_lifecycle_backend_mismatch")
                    lifecycle_provenance = (
                        lifecycle_payload.get("provenance")
                        if isinstance(lifecycle_payload, Mapping)
                        else None
                    )
                    if not isinstance(lifecycle_provenance, Mapping):
                        failures.append("canonical_payload_lifecycle_provenance_required")
                    else:
                        for identity_name in ("payload", "wheel"):
                            lifecycle_identity = lifecycle_provenance.get(identity_name)
                            failures.extend(
                                _native_identity_failures(
                                    lifecycle_identity,
                                    f"canonical_payload_lifecycle_{identity_name}",
                                )
                            )
                            artifact_identity = provenance.get(identity_name)
                            if (
                                isinstance(artifact_identity, Mapping)
                                and isinstance(lifecycle_identity, Mapping)
                                and str(artifact_identity.get("sha256", "")).lower()
                                != str(lifecycle_identity.get("sha256", "")).lower()
                            ):
                                failures.append(
                                    f"canonical_payload_lifecycle_{identity_name}_sha256_mismatch"
                                )
                    lifecycle_profiles = lifecycle_payload.get("profiles") if isinstance(
                        lifecycle_payload, Mapping
                    ) else None
                    if not isinstance(lifecycle_profiles, (list, tuple)) or set(
                        str(profile) for profile in lifecycle_profiles
                    ) != set(BACKEND_PROFILES.get(backend, ())):
                        failures.append("canonical_payload_lifecycle_profile_coverage_incomplete")

    return {
        "status": "pass" if not failures else "invalid",
        "complete": not failures,
        "failures": failures,
        "schema": schema,
        "backend": backend,
        "source_commit": source_commit,
        "profiles": sorted(profiles),
        "operations_by_profile": {
            profile: sorted(operations)
            for profile, operations in sorted(observed_by_profile.items())
        },
        "incomplete_profiles": incomplete_profiles,
        "native_statistics": native_statistics,
        "observed_profile_orders": [list(order) for order in sorted(record_orders if backend in ("cuda", "rocm") and len(profiles) == 3 else observed_orders)],
        "warmups": warmups,
        "repeats": repeats,
        "execution_mode": execution_mode,
        "input_policy": raw_input_policy,
        "input_identity": _json_safe(raw_input_identity),
        "source_root": _json_safe(payload.get("source_root")),
        "source_tree_state": _json_safe(source_tree_state),
        "compiler": _json_safe(payload.get("compiler")),
        "device": _json_safe(payload.get("device")),
        "clock": _json_safe(payload.get("clock", payload.get("timing_clock"))),
        "clock_and_power_state": _json_safe(payload.get("clock_and_power_state")),
        "environment": _json_safe(environment),
        "provenance": _json_safe(provenance),
    }


def _native_evidence_backend_readiness(
    native_evidence: Mapping[str, object],
    backends: Sequence[str],
    variants: Sequence[Variant],
    public_results: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if native_evidence.get("valid") is not True:
        failures.append(
            {
                "reason": "native_evidence_invalid_or_unsupported",
                "manifest_status": native_evidence.get("status"),
                "manifest_failures": list(native_evidence.get("failures", ())),
            }
        )
    elif native_evidence.get("arithmetic_claims_valid") is not True:
        failures.append(
            {
                "reason": "native_arithmetic_claims_not_validated",
                "manifest_status": native_evidence.get("status"),
            }
        )
    artifacts = native_evidence.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    by_backend: dict[str, list[str]] = {}
    by_variant_backend: dict[tuple[str, str], list[str]] = {}
    coverage_by_variant_backend: dict[
        tuple[str, str], list[Mapping[str, object]]
    ] = {}
    validation_by_variant_backend: dict[
        tuple[str, str], list[Mapping[str, object]]
    ] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        by_backend.setdefault(str(artifact.get("backend")), []).append(
            str(artifact.get("kind"))
        )
        by_variant_backend.setdefault(
            (str(artifact.get("variant")), str(artifact.get("backend"))), []
        ).append(str(artifact.get("kind")))
        validation = artifact.get("validation")
        if isinstance(validation, Mapping):
            validation_by_variant_backend.setdefault(
                (str(artifact.get("variant")), str(artifact.get("backend"))), []
            ).append(validation)
        if isinstance(validation, Mapping) and validation.get("complete") is True:
            coverage_by_variant_backend.setdefault(
                (str(artifact.get("variant")), str(artifact.get("backend"))), []
            ).append(validation)
    public_commits: dict[str, set[str]] = {}
    for result in public_results:
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        variant = provenance.get("variant")
        commit = provenance.get("source_commit")
        if variant and commit:
            public_commits.setdefault(str(variant), set()).add(str(commit))
    native_commits = native_evidence.get("source_commits_by_variant", {})
    if not isinstance(native_commits, Mapping):
        native_commits = {}
    accepted_kinds = {
        "core": {"core_microbenchmark", "native_decomposition"},
        "cuda": {"cuda_events", "device_events", "native_decomposition"},
        "rocm": {"rocm_events", "device_events", "native_decomposition"},
        "metal": {"metal_events", "device_events", "native_decomposition"},
    }
    public_provenance_by_variant_backend: dict[
        tuple[str, str], Mapping[str, object]
    ] = {}
    for result in public_results:
        if result.get("kind") != "public" or result.get("status") != "pass":
            continue
        provenance = result.get("provenance")
        if isinstance(provenance, Mapping) and provenance.get("variant"):
            public_provenance_by_variant_backend[
                (str(provenance["variant"]), str(result.get("backend")))
            ] = provenance
    for variant in variants:
        for backend in backends:
            key = (variant.name, backend)
            expected_public_commits = public_commits.get(variant.name, set())
            native_commit = native_commits.get(variant.name)
            if public_results and len(expected_public_commits) != 1:
                failures.append(
                    {
                        "variant": variant.name,
                        "backend": backend,
                        "reason": "public_source_commit_provenance_incomplete",
                        "observed": sorted(expected_public_commits),
                    }
                )
            elif public_results and native_commit not in expected_public_commits:
                failures.append(
                    {
                        "variant": variant.name,
                        "backend": backend,
                        "reason": "native_source_commit_does_not_match_public_variant",
                        "native": native_commit,
                        "public": sorted(expected_public_commits),
                    }
                )
            public_provenance = public_provenance_by_variant_backend.get(key)
            for validation in validation_by_variant_backend.get(key, ()):
                native_provenance = validation.get("provenance")
                if not isinstance(native_provenance, Mapping):
                    continue
                if public_provenance is not None:
                    native_wheel = native_provenance.get("wheel")
                    public_wheels = public_provenance.get("wheel_artifacts", ())
                    public_wheel_hashes = {
                        str(identity.get("sha256")).lower()
                        for identity in public_wheels
                        if isinstance(identity, Mapping) and identity.get("sha256")
                    }
                    native_wheel_hash = (
                        str(native_wheel.get("sha256")).lower()
                        if isinstance(native_wheel, Mapping)
                        and native_wheel.get("sha256")
                        else None
                    )
                    if native_wheel_hash not in public_wheel_hashes:
                        failures.append(
                            {
                                "variant": variant.name,
                                "backend": backend,
                                "reason": "native_wheel_does_not_match_public_variant",
                                "native": native_wheel_hash,
                                "public": sorted(public_wheel_hashes),
                            }
                        )
                    native_payload = native_provenance.get("payload")
                    native_payload_hash = (
                        str(native_payload.get("sha256")).lower()
                        if isinstance(native_payload, Mapping)
                        and native_payload.get("sha256")
                        else None
                    )
                    public_binary_hashes = {
                        str(identity.get("sha256")).lower()
                        for collection in (
                            public_provenance.get("native_binaries", ()),
                            public_provenance.get("loaded_module_files", ()),
                        )
                        for identity in collection
                        if isinstance(identity, Mapping) and identity.get("sha256")
                    }
                    if (
                        native_payload_hash
                        and public_binary_hashes
                        and native_payload_hash not in public_binary_hashes
                    ):
                        failures.append(
                            {
                                "variant": variant.name,
                                "backend": backend,
                                "reason": "native_payload_does_not_match_public_runtime",
                                "native": native_payload_hash,
                                "public": sorted(public_binary_hashes),
                            }
                        )
            observed = set(by_variant_backend.get(key, ()))
            accepted = observed & accepted_kinds[backend]
            validations = coverage_by_variant_backend.get(key, [])
            required_profiles = set(BACKEND_PROFILES[backend])
            covered_profiles: set[str] = set()
            operations_by_profile: dict[str, set[str]] = {}
            for validation in validations:
                covered_profiles.update(
                    str(profile) for profile in validation.get("profiles", ())
                )
                raw_operations = validation.get("operations_by_profile", {})
                if isinstance(raw_operations, Mapping):
                    for profile, operations in raw_operations.items():
                        if isinstance(operations, (list, tuple, set, frozenset)):
                            operations_by_profile.setdefault(str(profile), set()).update(
                                str(operation) for operation in operations
                            )
            missing_profiles = sorted(required_profiles - covered_profiles)
            incomplete_profiles = {
                profile: sorted(
                    NATIVE_REQUIRED_OPERATIONS_BY_BACKEND.get(
                        backend, NATIVE_REQUIRED_OPERATIONS
                    )
                    - operations_by_profile.get(profile, set())
                )
                for profile in sorted(required_profiles)
                if NATIVE_REQUIRED_OPERATIONS_BY_BACKEND.get(
                    backend, NATIVE_REQUIRED_OPERATIONS
                )
                - operations_by_profile.get(profile, set())
            }
            if not accepted:
                failures.append(
                    {
                        "variant": variant.name,
                        "backend": backend,
                        "reason": "validated_backend_schema_artifact_missing",
                        "accepted_kinds": sorted(accepted_kinds[backend]),
                        "observed_kinds": sorted(observed),
                    }
                )
            if missing_profiles:
                failures.append(
                    {
                        "variant": variant.name,
                        "backend": backend,
                        "reason": "native_profile_coverage_incomplete",
                        "required_profiles": sorted(required_profiles),
                        "covered_profiles": sorted(covered_profiles),
                        "missing_profiles": missing_profiles,
                    }
                )
            if incomplete_profiles:
                failures.append(
                    {
                        "variant": variant.name,
                        "backend": backend,
                        "reason": "native_decomposition_profile_coverage_incomplete",
                        "missing_operations_by_profile": incomplete_profiles,
                    }
                )
    if len(variants) == 2:
        baseline_name, candidate_name = (variant.name for variant in variants)
        for backend in backends:
            baseline_validations = validation_by_variant_backend.get(
                (baseline_name, backend), ()
            )
            candidate_validations = validation_by_variant_backend.get(
                (candidate_name, backend), ()
            )
            if not baseline_validations or not candidate_validations:
                continue
            for field in (
                "compiler",
                "device",
                "environment",
                "clock",
                "clock_and_power_state",
                "execution_mode",
            ):
                if field == "environment":
                    baseline_views = [
                        _native_environment_comparison_view(validation)
                        for validation in baseline_validations
                    ]
                    candidate_views = [
                        _native_environment_comparison_view(validation)
                        for validation in candidate_validations
                    ]
                    if any(view is None for view in baseline_views + candidate_views):
                        failures.append(
                            {
                                "backend": backend,
                                "reason": "native_environment_provenance_invalid",
                            }
                        )
                    baseline_values = {
                        json.dumps(_json_safe(view), sort_keys=True)
                        for view in baseline_views
                        if view is not None
                    }
                    candidate_values = {
                        json.dumps(_json_safe(view), sort_keys=True)
                        for view in candidate_views
                        if view is not None
                    }
                elif field == "clock_and_power_state":
                    baseline_views = [
                        _clock_power_comparison_view(
                            validation.get("clock_and_power_state")
                        )
                        for validation in baseline_validations
                    ]
                    candidate_views = [
                        _clock_power_comparison_view(
                            validation.get("clock_and_power_state")
                        )
                        for validation in candidate_validations
                    ]
                    if any(view is None for view in baseline_views + candidate_views):
                        failures.append(
                            {
                                "backend": backend,
                                "reason": "native_clock_power_provenance_invalid",
                            }
                        )
                    baseline_values = {
                        json.dumps(_json_safe(view), sort_keys=True)
                        for view in baseline_views
                        if view is not None
                    }
                    candidate_values = {
                        json.dumps(_json_safe(view), sort_keys=True)
                        for view in candidate_views
                        if view is not None
                    }
                else:
                    baseline_values = {
                        json.dumps(_json_safe(validation.get(field)), sort_keys=True)
                        for validation in baseline_validations
                        if validation.get(field) is not None
                    }
                    candidate_values = {
                        json.dumps(_json_safe(validation.get(field)), sort_keys=True)
                        for validation in candidate_validations
                        if validation.get(field) is not None
                    }
                if baseline_values and candidate_values and baseline_values != candidate_values:
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "baseline_and_candidate_native_provenance_mismatch",
                            "field": field,
                            "baseline": sorted(baseline_values),
                            "candidate": sorted(candidate_values),
                        }
                    )
    complete = bool(
        native_evidence.get("valid") is True
        and native_evidence.get("arithmetic_claims_valid") is True
        and not failures
    )
    return {
        "complete": complete,
        "failures": failures,
        "artifacts_by_backend": by_backend,
        "artifacts_by_variant_backend": {
            f"{variant}/{backend}": kinds
            for (variant, backend), kinds in sorted(by_variant_backend.items())
        },
        "coverage_by_variant_backend": {
            f"{variant}/{backend}": {
                "profiles": sorted(
                    {
                        str(profile)
                        for validation in validations
                        for profile in validation.get("profiles", ())
                    }
                ),
                "artifact_count": len(validations),
            }
            for (variant, backend), validations in sorted(
                coverage_by_variant_backend.items()
            )
        },
        "policy": (
            "each requested backend needs a backend-schema-validated artifact whose "
            "union covers every supported profile and every decomposition operation; "
            "set intersection with an arbitrary hashed file is insufficient"
        ),
    }


def _driver_main(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    results: list[dict[str, object]] = []
    schedule: list[dict[str, object]] = []
    try:
        native_evidence = _load_native_evidence_specs(args.native_evidence)
    except OSError as exc:
        native_evidence = {
            "path": str(args.native_evidence),
            "status": "invalid",
            "arithmetic_claims_valid": False,
            "valid": False,
            "failures": [f"manifest_read_error:{exc}"],
            "manifest": None,
        }
    for backend in args.backends:
        profiles = _backend_profile_order(
            backend,
            args.profiles,
            lambda message: (_ for _ in ()).throw(ValueError(message)),
        )
        orders = _profile_orders(profiles)
        for order_repeat in range(args.order_repetitions):
            rotated_orders = list(orders)
            rng.shuffle(rotated_orders)
            for order_index, profile_order in enumerate(rotated_orders):
                variant_blocks = _variant_sequences(args.variants, args.ab_blocks, rng)
                if args.layer in ("all", "cold"):
                    cold_workload = WORKLOADS[args.workloads[0]]
                    for cold_policy in args.input_policies:
                        for block_index, variant_sequence in enumerate(variant_blocks):
                            for profile in profile_order:
                                for variant in variant_sequence:
                                    job = {
                                        "kind": "cold",
                                        "backend": backend,
                                        "precision": profile,
                                        "profile_order_context": list(profile_order),
                                        "order_repeat": order_repeat,
                                        "order_index": order_index,
                                        "ab_block": block_index,
                                        "variant_sequence": [
                                            value.name for value in variant_sequence
                                        ],
                                        "input_policy": cold_policy,
                                        "workload": _workload_payload(cold_workload),
                                        "seed": args.seed,
                                        "device_id": args.device_id,
                                    }
                                    schedule.append(
                                        {
                                            "kind": "cold",
                                            "variant": variant.name,
                                            "backend": backend,
                                            "profile_order": list(profile_order),
                                            "profile": profile,
                                            "input_policy": cold_policy,
                                            "ab_block": block_index,
                                            "variant_sequence": [
                                                value.name for value in variant_sequence
                                            ],
                                        }
                                    )
                                    results.append(
                                        _run_worker(
                                            variant, job, timeout=args.timeout_seconds
                                        )
                                    )
                if args.layer in ("all", "public"):
                    for workload_name in args.workloads:
                        for input_policy in args.input_policies:
                            workload = WORKLOADS[workload_name]
                            for block_index, variant_sequence in enumerate(
                                variant_blocks
                            ):
                                for variant in variant_sequence:
                                    surface_order = _surface_order(
                                        order_index + block_index, args.surfaces
                                    )
                                    job = {
                                        "kind": "public",
                                        "backend": backend,
                                        "profile_order": list(profile_order),
                                        "surface_order": list(surface_order),
                                        "order_repeat": order_repeat,
                                        "order_index": order_index,
                                        "ab_block": block_index,
                                        "variant_sequence": [
                                            value.name for value in variant_sequence
                                        ],
                                        "input_policy": input_policy,
                                        "workload": _workload_payload(workload),
                                        "warmups": args.warmups,
                                        "repetitions": args.repetitions,
                                        "bootstrap_resamples": args.bootstrap_resamples,
                                        "min_sample_ns": int(
                                            args.min_sample_ms * 1.0e6
                                        ),
                                        "max_loop_count": args.max_loop_count,
                                        "interleaved_control": (
                                            args.interleaved_control
                                            and order_index == 0
                                        ),
                                        "seed": args.seed,
                                        "device_id": args.device_id,
                                    }
                                    schedule.append(
                                        {
                                            "kind": "public",
                                            "variant": variant.name,
                                            "backend": backend,
                                            "profile_order": list(profile_order),
                                            "surface_order": list(surface_order),
                                            "workload": workload_name,
                                            "input_policy": input_policy,
                                            "ab_block": block_index,
                                            "variant_sequence": [
                                                value.name for value in variant_sequence
                                            ],
                                            "interleaved_control": (
                                                args.interleaved_control
                                                and order_index == 0
                                            ),
                                        }
                                    )
                                    results.append(
                                        _run_worker(
                                            variant, job, timeout=args.timeout_seconds
                                        )
                                    )
    provenance_readiness = _provenance_readiness(results)
    order_sensitivity = _order_sensitivity(results)
    interleaved_order_sensitivity = _interleaved_order_sensitivity(results)
    ab_comparisons = _ab_comparisons(
        results,
        args.variants,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    coverage_readiness = _coverage_readiness(args, schedule)
    ab_schedule_readiness = _ab_schedule_readiness(
        schedule, args.variants, args.ab_blocks
    )
    comparative_input_readiness = _comparative_input_readiness(results, args.variants)
    sample_target_readiness = _sample_target_readiness(results, args.repetitions)
    result_matrix_readiness = _result_matrix_readiness(
        results,
        expected_public_result_count=sum(
            item.get("kind") == "public" for item in schedule
        ),
    )
    cold_summaries = _cold_summaries(
        results, bootstrap_resamples=args.bootstrap_resamples, seed=args.seed
    )
    native_backend_readiness = _native_evidence_backend_readiness(
        native_evidence, args.backends, args.variants, public_results=results
    )
    cold_comparisons = _cold_comparisons(
        cold_summaries,
        args.variants,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    native_comparisons = _native_ab_comparisons(
        native_evidence,
        args.variants,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    threshold_readiness = _threshold_readiness(
        order_sensitivity,
        ab_comparisons,
        require_comparison=len(args.variants) == 2,
        interleaved_order_sensitivity=interleaved_order_sensitivity,
        cold_comparisons=cold_comparisons,
        native_comparisons=native_comparisons,
        require_cold_comparison=len(args.variants) == 2,
        require_native_comparison=len(args.variants) == 2,
    )
    e2e_claim_ready = bool(
        provenance_readiness["complete"]
        and coverage_readiness["complete"]
        and sample_target_readiness["complete"]
        and result_matrix_readiness["complete"]
        and threshold_readiness["complete"]
    )
    arithmetic_claim_ready = bool(
        e2e_claim_ready
        and native_backend_readiness["complete"]
    )
    comparative_claim_ready = bool(
        e2e_claim_ready
        and ab_schedule_readiness["complete"]
        and comparative_input_readiness["complete"]
        and bool(ab_comparisons)
    )
    full_claim_ready = bool(comparative_claim_ready and arithmetic_claim_ready)
    claim_failures: list[dict[str, object]] = []
    for name, readiness in (
        ("provenance", provenance_readiness),
        ("coverage", coverage_readiness),
        ("sample_target", sample_target_readiness),
        ("result_matrix", result_matrix_readiness),
        ("threshold", threshold_readiness),
        ("ab_schedule", ab_schedule_readiness),
        ("comparative_inputs", comparative_input_readiness),
        ("native_evidence_backend_coverage", native_backend_readiness),
    ):
        if not readiness["complete"]:
            claim_failures.append({"gate": name, "failures": readiness["failures"]})
    if not arithmetic_claim_ready:
        claim_failures.append(
            {
                "gate": "native_evidence",
                "failures": native_evidence.get("failures", ["validated_manifest_required"]),
            }
        )
    if full_claim_ready:
        output_status = "pass"
    elif not provenance_readiness["complete"]:
        output_status = "measurement_complete_but_provenance_incomplete"
    else:
        output_status = "measurement_complete_but_claim_gate_failed"
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": output_status,
        "valid_for_performance_claims": full_claim_ready,
        "valid_for_e2e_performance_claims": e2e_claim_ready,
        "valid_for_comparative_performance_claims": comparative_claim_ready,
        "valid_for_arithmetic_claims": arithmetic_claim_ready,
        "valid_for_native_arithmetic_claims": arithmetic_claim_ready,
        "valid_for_kernel_claims": arithmetic_claim_ready,
        "claim_failures": claim_failures,
        "methodology": {
            "wall_clock_primary": True,
            "cold_isolation": (
                "every cold sample runs in a fresh subprocess; clean cold interval "
                "starts at worker entry and ends after explicit artifact cleanup"
            ),
            "public_isolation": "fresh subprocess per backend, workload, input policy, profile-order, A/B block, and variant",
            "profile_order": "caller order is preserved and all permutations are exercised",
            "warmups": args.warmups,
            "measured_repetitions": args.repetitions,
            "auto_scaling_min_region_ns": int(args.min_sample_ms * 1.0e6),
            "sample_region_gate": "every recorded raw region must meet auto-scaling target",
            "interleaved_control": (
                "one already-warmed worker per workload/input/A-B block uses "
                "balanced randomized profile blocks"
                if args.interleaved_control
                else "disabled by explicit CLI request"
            ),
            "statistics": "raw durations, median, MAD, p05, p95, bootstrap median 95% CI",
            "ab_comparison": (
                "baseline and candidate fresh-worker distributions are resampled "
                "independently; no raw observation pairs are claimed"
            ),
            "cold_comparison": (
                "overall clean intervals plus every phase with an observed duration; "
                "combined/unobserved phases are excluded from phase deltas"
            ),
            "native_comparison_key": (
                "backend, workload identity, input policy/identity, profile, operation, "
                "metric, order index/order, clock, synchronization/timing boundary, and unit"
            ),
            "benchmark_harness_identity": (
                "one canonical frozen perf13 driver/worker script hash is required for "
                "both A/B variants; source roots identify builds only"
            ),
            "provenance_requirements": (
                "source commit, clean tree, wheel/module/native hashes, compiler/linker, "
                "runtime/hardware, affinity, environment, and before/after clock/power state"
            ),
            "input_identity": (
                "matrix, target, and feature-name SHA-256 identities are recorded and "
                "must match across A/B cells"
            ),
            "input_policies": list(args.input_policies),
            "continuous_arity_coverage": [3, 4, 5],
            "generated_family_performance": {
                "status": "not_claimed",
                "reason": "perf13 uses max_generated_features=0; public generated-family timing requires a separate safe API contract",
            },
            "rate_name": "candidate-sample pairs per second for the named configured metric set",
            "native_timing_boundary": "public timings are not kernel timings; CUDA/HIP/Metal event and Core native microbenchmark artifacts are separate evidence",
            "native_core_scope": (
                "Core native arithmetic uses its independently reported 5 ms target region; "
                "it does not claim public result/report construction or replace perf13's 100 ms gate"
            ),
            "native_evidence_manifest": (
                "required machine-readable input; validated hash-bound artifacts are "
                "referenced but no native timing is synthesized by this script"
            ),
            "timed_call_boundary": (
                "only the public analyze/replay call is inside perf_counter_ns; report, "
                "candidate-count, finite-value, and graph-replay validation occur after the clock stops"
            ),
            "claim_gate": (
                "valid_for_performance_claims is true only after clean source/module/wheel "
                "identity, canonical matrix/order/input/A-B coverage, sample targets, "
                "regression thresholds, and validated native evidence"
            ),
            "old_perf_12_status": "provisional_invalid_for_comparison",
        },
        "arguments": {
            "backends": list(args.backends),
            "profiles_in_requested_order": list(args.profiles),
            "profile_orders": [list(order) for order in _profile_orders(args.profiles)],
            "surfaces": list(args.surfaces),
            "workloads": list(args.workloads),
            "input_policies": list(args.input_policies),
            "layer": args.layer,
            "seed": args.seed,
            "device_id": args.device_id,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "bootstrap_resamples": args.bootstrap_resamples,
            "min_sample_ns": int(args.min_sample_ms * 1.0e6),
            "max_loop_count": args.max_loop_count,
            "order_repetitions": args.order_repetitions,
            "ab_blocks": args.ab_blocks,
            "interleaved_control": args.interleaved_control,
            "native_evidence": args.native_evidence,
            "variants": [asdict(variant) for variant in args.variants],
        },
        "workload_definitions": [
            _workload_payload(WORKLOADS[name]) for name in args.workloads
        ],
        "schedule": schedule,
        "provenance_readiness": provenance_readiness,
        "native_evidence": native_evidence,
        "native_evidence_backend_readiness": native_backend_readiness,
        "coverage_readiness": coverage_readiness,
        "ab_schedule_readiness": ab_schedule_readiness,
        "comparative_input_readiness": comparative_input_readiness,
        "sample_target_readiness": sample_target_readiness,
        "result_matrix_readiness": result_matrix_readiness,
        "threshold_readiness": threshold_readiness,
        "cold_summaries": cold_summaries,
        "cold_comparisons": cold_comparisons,
        "order_sensitivity": order_sensitivity,
        "interleaved_order_sensitivity": interleaved_order_sensitivity,
        "ab_comparisons": ab_comparisons,
        "native_comparisons": native_comparisons,
        "results": results,
    }
    rendered = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(rendered)
    else:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered)
        print(f"wrote {destination}", file=sys.stderr)
    return 0


def _self_check() -> int:
    orders = _profile_orders(("fp64", "fp32", "mixed"))
    assert len(orders) == 6
    assert orders[0] == ("fp64", "fp32", "mixed")
    assert {tuple(order) for order in orders} == set(
        itertools.permutations(PROFILE_ORDER)
    )
    assert set(COLD_PHASES) == {
        "python_import",
        "payload_discovery",
        "dynamic_library_load",
        "runtime_context_initialization",
        "code_object_or_module_registration",
        "first_capability_query",
        "first_allocation",
        "first_upload",
        "planning",
        "first_execution",
        "first_result_materialization",
        "explicit_cleanup",
        "process_exit_cleanup",
    }
    distribution = _distribution(
        list(range(100, 130)),
        1,
        bootstrap_resamples=500,
        seed=7,
        samples=10,
        candidates=3,
        metrics=("pearson",),
        sample_region_target_ns=100,
    )
    assert distribution["measured_repetitions"] == 30
    assert distribution["median_ns"] == 114.5
    assert distribution["sample_region_target_met"] is True
    unmet = _distribution(
        [10, 20],
        1,
        bootstrap_resamples=500,
        seed=7,
        samples=10,
        candidates=3,
        metrics=("pearson",),
        sample_region_target_ns=21,
    )
    assert unmet["sample_region_target_met"] is False
    timing_events: list[str] = []
    original_clock = perf_counter_ns
    clock_tick = 0

    def fake_clock() -> int:
        nonlocal clock_tick
        timing_events.append("clock")
        clock_tick += 1
        return clock_tick

    def fake_operation():
        timing_events.append("operation")
        return object()

    def fake_validate(_report) -> int:
        timing_events.append("validate")
        return 1

    globals()["perf_counter_ns"] = fake_clock
    try:
        _calibrated_samples(
            fake_operation,
            fake_validate,
            warmups=1,
            repetitions=1,
            min_sample_ns=1,
            max_loop_count=1,
        )
    finally:
        globals()["perf_counter_ns"] = original_clock
    first_clock = timing_events.index("clock")
    second_clock = timing_events.index("clock", first_clock + 1)
    assert all(event == "operation" for event in timing_events[first_clock + 1 : second_clock])
    variants = (
        Variant("a", sys.executable, None, ()),
        Variant("b", sys.executable, None, ()),
    )
    sequences = _variant_sequences(variants, 2, random.Random(7))
    assert sequences[1] == tuple(reversed(sequences[0]))
    synthetic_results = []
    for variant, median_ns, ordinal, order in (
        ("a", 100.0, 0, ("fp32", "mixed", "fp64")),
        ("a", 101.0, 1, ("mixed", "fp32", "fp64")),
        ("b", 104.0, 0, ("fp32", "mixed", "fp64")),
    ):
        synthetic_results.append(
            {
                "kind": "public",
                "status": "pass",
                "backend": "core",
                "input_policy": "common-f64",
                "workload": {"name": "small-latency"},
                "profile_order": order,
                "ab_block": 0,
                "order_repeat": 0,
                "provenance": {"variant": variant},
                "cells": [
                    {
                        "status": "pass",
                "surface": "compiled",
                "profile": "fp32",
                "profile_order_ordinal": ordinal,
                        "distribution": {
                            "median_ns": median_ns,
                            "raw_per_call_duration_ns": [median_ns] * 30,
                        },
                    }
                ],
            }
        )
    sensitivity = _order_sensitivity(synthetic_results)
    assert sensitivity[0]["max_position_median_spread_percent"] > 0.0
    comparisons = _ab_comparisons(synthetic_results, variants)
    assert comparisons[0]["candidate_latency_delta_percent"] == 4.0
    assert comparisons[0]["review_status"] == "maintainer_approval_required"
    synthetic_schedule = [
        {
            "kind": "public",
            "backend": "core",
            "profile_order": list(PROFILE_ORDER),
            "order_repeat": 0,
            "ab_block": block,
            "variant_sequence": ["a", "b"] if block == 0 else ["b", "a"],
        }
        for block in range(2)
    ]
    assert _ab_schedule_readiness(synthetic_schedule, variants, 2)["complete"]
    synthetic_target = _sample_target_readiness(
        [
            {
                "kind": "public",
                "status": "pass",
                "cells": [
                    {
                        "status": "pass",
                        "distribution": {
                            "measured_repetitions": 30,
                            "sample_region_target_met": False,
                            "sample_region_target_ns": 100,
                            "sample_region_min_observed_ns": 99,
                        },
                    }
                ],
                "interleaved_controls": [],
            }
        ],
        30,
    )
    assert synthetic_target["complete"] is False
    with tempfile.TemporaryDirectory(prefix="gafime-perf13-self-check-") as temp_dir:
        temp_root = Path(temp_dir)
        manifest_path = Path(temp_dir) / "native-evidence.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": NATIVE_EVIDENCE_SCHEMA,
                    "status": "not_collected",
                    "arithmetic_claims_valid": False,
                    "artifacts": [],
                }
            )
        )
        native_evidence = _load_native_evidence(str(manifest_path))
        assert native_evidence["valid"] is True
        assert native_evidence["arithmetic_claims_valid"] is False

        source_path = temp_root / "benchmark-source.cu"
        binary_path = temp_root / "benchmark-binary"
        payload_path = temp_root / "payload.so"
        wheel_path = temp_root / "gafime.whl"
        for path, contents in (
            (source_path, b"source"),
            (binary_path, b"binary"),
            (payload_path, b"payload"),
        ):
            path.write_bytes(contents)
        with zipfile.ZipFile(wheel_path, "w") as archive:
            archive.writestr("gafime_cuda/libgafime_cuda.so", payload_path.read_bytes())
            archive.writestr(
                "gafime-1.0.0b2.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: gafime\nVersion: 1.0.0b2\n",
            )
            archive.writestr("gafime/gafime_py.so", b"core-native")
        identity = lambda path: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        core_records = []
        for profile in PROFILE_ORDER:
            for metric in ALL_METRICS:
                core_records.append(
                    {
                        "profile": profile,
                        "operation": "metric_kernel",
                        "metric": metric,
                        "samples_us": [1.0] * 30,
                    }
                )
        artifact_path = Path(temp_dir) / "core-native.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "schema": "gafime.core-native-arithmetic.v2",
                    "status": "pass",
                    "backend": "core",
                    "profiles": list(PROFILE_ORDER),
                    "source_commit": "a" * 40,
                    "source_tree_state": {"status": "clean", "entries": []},
                    "input_policy": "common-f64",
                    "input_identity": {
                        "matrix_sha256": "1" * 64,
                        "target_sha256": "2" * 64,
                        "feature_names_sha256": "3" * 64,
                    },
                    "warmups": 10,
                    "repeats": 30,
                    "target_region_ns": 5_000_000,
                    "measurement_scope": "native_arithmetic_only",
                    "decomposition_boundaries": {
                        "candidate_materialization": "included in metric kernels",
                        "report_construction": "not measured by this native arithmetic benchmark",
                    },
                    "compiler": {"rustc": "self-check"},
                    "device": {"kind": "cpu", "identity": "self-check-cpu"},
                    "process_affinity": [0],
                    "clock": "std::time::Instant monotonic clock",
                    "clock_and_power_state": {
                        "before": {"cpu_governor": ["performance"]},
                        "after": {"cpu_governor": ["performance"]},
                    },
                    "environment": {},
                        "provenance": {
                            "benchmark_source": identity(source_path),
                            "benchmark_binary": identity(binary_path),
                            "python_executable": identity(Path(sys.executable)),
                            "wheel": identity(wheel_path),
                        },
                    "records": core_records,
                }
            )
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": NATIVE_EVIDENCE_SCHEMA,
                    "status": "validated",
                    "arithmetic_claims_valid": True,
                    "source_commit": "a" * 40,
                    "artifacts": [
                        {
                            "variant": "a",
                            "backend": "core",
                            "kind": "core_microbenchmark",
                            "path": str(artifact_path),
                            "sha256": _sha256(artifact_path),
                        }
                    ],
                }
            )
        )
        native_evidence = _load_native_evidence(str(manifest_path))
        assert native_evidence["valid"] is True
        assert _native_evidence_backend_readiness(
            native_evidence, ("core",), (variants[0],)
        )["complete"] is True
        # A syntactically valid hash entry with arbitrary JSON is not native
        # evidence: backend schema and complete records are mandatory.
        arbitrary_path = temp_root / "arbitrary.json"
        arbitrary_path.write_text("{\"native\":true}\n")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": NATIVE_EVIDENCE_SCHEMA,
                    "status": "validated",
                    "arithmetic_claims_valid": True,
                    "source_commit": "a" * 40,
                    "artifacts": [
                        {
                            "variant": "a",
                            "backend": "core",
                            "kind": "core_microbenchmark",
                            "path": str(arbitrary_path),
                            "sha256": _sha256(arbitrary_path),
                        }
                    ],
                }
            )
        )
        native_evidence = _load_native_evidence(str(manifest_path))
        assert native_evidence["valid"] is False
        assert "artifact_0_native_validation" in " ".join(native_evidence["failures"])

        # A CUDA direct-kernel artifact must carry a separate canonical payload
        # lifecycle proof; otherwise it remains supplemental and cannot pass.
        cuda_path = temp_root / "cuda-native.json"
        cuda_records = []
        for profile in PROFILE_ORDER:
            for operation in (
                "ingest_conversion",
                "planning",
                "allocation",
                "h2d_upload",
                "candidate_materialization",
                "ranking_kernel",
                "ranking_topk",
                "selected_row_gather",
                "d2h_transfer",
                "report_construction",
            ):
                cuda_records.append(
                    {
                        "profile": profile,
                        "operation": operation,
                        "metric": "none",
                        "samples_us": [1.0] * 30,
                    }
                )
            for metric in ALL_METRICS:
                cuda_records.append(
                    {
                        "profile": profile,
                        "operation": "metric_kernel",
                        "metric": metric,
                        "samples_us": [1.0] * 30,
                    }
                )
        cuda_path.write_text(
            json.dumps(
                {
                    "schema": "gafime.cuda.native_timing.v2",
                    "status": "pass",
                    "backend": "cuda",
                    "profiles": list(PROFILE_ORDER),
                    "source_commit": "a" * 40,
                    "source_tree_state": {"status": "clean", "entries": []},
                    "input_policy": "common-f64",
                    "input_identity": {
                        "matrix_sha256": "1" * 64,
                        "target_sha256": "2" * 64,
                        "feature_names_sha256": "3" * 64,
                    },
                    "warmups": 10,
                    "repeats": 30,
                    "execution_mode": "supplemental_internal_kernel",
                    "decomposition_boundaries": {
                        "candidate_materialization": "supplemental"
                    },
                    "compiler": {
                        "nvcc_major": 13,
                        "nvcc_minor": 3,
                        "nvcc": {"status": "observed", "version": "self-check"},
                        "host_cxx": {"status": "observed", "version": "self-check"},
                        "linker": {"status": "observed", "version": "self-check"},
                    },
                    "device": {"name": "self-check", "runtime_version": 1},
                    "process_affinity": [0],
                    "environment": {},
                    "clock": {"host": "steady_clock", "device": "cudaEvent"},
                    "clock_and_power_capture_point": (
                        "before and after all timed benchmark regions"
                    ),
                    "clock_and_power_state": {
                        "before": {
                            "cpu_governor": {
                                "status": "observed",
                                "values": ["performance"],
                            },
                            "nvidia_smi": {
                                "status": "pass",
                                "source": "command",
                                "output": "self-check-p8",
                            },
                        },
                        "after": {
                            "cpu_governor": {
                                "status": "observed",
                                "values": ["performance"],
                            },
                            "nvidia_smi": {
                                "status": "pass",
                                "source": "command",
                                "output": "self-check-p0",
                            },
                        },
                    },
                    "provenance": {
                        name: identity(path)
                        for name, path in (
                            ("benchmark_source", source_path),
                            ("benchmark_binary", binary_path),
                            ("payload", payload_path),
                            ("wheel", wheel_path),
                        )
                    },
                    "records": cuda_records,
                }
            )
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": NATIVE_EVIDENCE_SCHEMA,
                    "status": "validated",
                    "arithmetic_claims_valid": True,
                    "source_commit": "a" * 40,
                    "artifacts": [
                        {
                            "variant": "a",
                            "backend": "cuda",
                            "kind": "cuda_events",
                            "path": str(cuda_path),
                            "sha256": _sha256(cuda_path),
                        }
                    ],
                }
            )
        )
        native_evidence = _load_native_evidence(str(manifest_path))
        assert native_evidence["valid"] is False
        assert "supplemental_internal_kernel_requires_canonical_payload_lifecycle" in " ".join(
            native_evidence["failures"]
        )
    assert set(RELEASE_WORKLOADS) == set(WORKLOADS)
    sys.stdout.write(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "self_check_pass",
                "profile_order_count": len(orders),
                "cold_phase_count": len(COLD_PHASES),
                "workload_count": len(WORKLOADS),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def main() -> int:
    args = _parse_args()
    if args._worker:
        return _worker_main()
    if args.self_check:
        return _self_check()
    return _driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
