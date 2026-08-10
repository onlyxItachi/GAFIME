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
timing requires the backend event/microbenchmark evidence collected separately.
The perf13 cold envelope is an order-contamination diagnostic only; canonical
30-sample lifecycle evidence is produced by ``cold_lifecycle.py``.
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
    # The direct CUDA lane exposes the resident-stat preparation kernels that
    # were previously executed once outside every timed record.  Keep them as
    # supplemental device-event categories rather than folding their cost into
    # payload_execute or a metric record.
    "cuda": NATIVE_REQUIRED_OPERATIONS
    | frozenset(("target_stat_preparation", "feature_stat_preparation")),
    "rocm": NATIVE_REQUIRED_OPERATIONS,
    # Metal records native fp32 source input conversion as not present, while
    # its helper records the remaining host/device lifecycle explicitly.
    "metal": NATIVE_REQUIRED_OPERATIONS - frozenset(("ingest_conversion",)),
}
NATIVE_SUPPLEMENTAL_OPERATION_ALIASES = {
    "payload_allocation": "supplemental:payload_allocation",
    "payload_h2d_upload": "supplemental:payload_h2d_upload",
    "payload_update_target": "supplemental:payload_update_target",
    "payload_execution_memory_peak": "supplemental:payload_execution_memory_peak",
    "payload_execute": "supplemental:payload_execute",
    "target_update": "supplemental:target_update",
    "execution_memory_forecast": "supplemental:execution_memory_forecast",
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
CANONICAL_ABI_TYPED_LIFECYCLE_OPERATIONS = frozenset(
    {
        "precision_capabilities",
        "matrix_alloc",
        "matrix_upload",
        "matrix_update_target",
        "execute",
        "execution_memory_peak",
        "interaction_diagnostics",
        "matrix_free",
    }
)
CANONICAL_ABI_SURFACES = frozenset({"numeric-route-v2", "precision-typed-v1.1"})
CANONICAL_ABI_OPERATIONS_BY_SURFACE = {
    "numeric-route-v2": CANONICAL_ABI_LIFECYCLE_OPERATIONS,
    "precision-typed-v1.1": CANONICAL_ABI_TYPED_LIFECYCLE_OPERATIONS,
}
CANONICAL_ABI_MARKER_SCHEMAS = {
    "numeric-route-v2": "gafime.abi-1.1-consumer-result.v1",
    "precision-typed-v1.1": "gafime.abi-1.1-typed-consumer-result.v1",
}
CANONICAL_ABI_EXECUTION_LAYERS = {
    "numeric-route-v2": "independent_abi_1_1_c_consumer",
    "precision-typed-v1.1": "independent_abi_1_1_typed_c_consumer",
}
CANONICAL_ABI_CONTRACT_ROLES = {
    "numeric-route-v2": "candidate_canonical_numeric_route",
    "precision-typed-v1.1": "historical_pre_freeze_typed_baseline",
}
CANONICAL_ABI_SYMBOLS_BY_SURFACE = {
    "numeric-route-v2": frozenset(
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
    ),
    "precision-typed-v1.1": frozenset(
        {
            "gafime_gpu_precision_capabilities",
            "gafime_gpu_matrix_alloc_v2",
            "gafime_gpu_matrix_upload_f32_v2",
            "gafime_gpu_matrix_upload_f64_v2",
            "gafime_gpu_matrix_update_target_f32_v2",
            "gafime_gpu_matrix_update_target_f64_v2",
            "gafime_gpu_execute_f32_v2",
            "gafime_gpu_execute_f64_v2",
            "gafime_gpu_execution_memory_peak_v2",
            "gafime_gpu_interaction_diagnostics",
            "gafime_gpu_matrix_free",
        }
    ),
}
CANONICAL_ABI_BACKEND_KINDS = {"cuda": 2, "rocm": 3, "metal": 4}
CANONICAL_ABI_WHEEL_MEMBERS = {
    "cuda": frozenset(("gafime_cuda/libgafime_cuda.so", "gafime_cuda/gafime_cuda.dll")),
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
        (
            "benchmark_source",
            "benchmark_binary",
            "harness_runner",
            "python_executable",
            "wheel",
        )
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
            "target_stat_preparation",
            "feature_stat_preparation",
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
MIN_PUBLIC_ORDER_REPETITIONS = 5
GPU_NATIVE_MIN_SAMPLE_REGION_US = 5_000.0
CORE_MIN_UNTIMED_PRECONDITION_NS = 100_000_000
CORE_MIN_MEASURED_REGION_NS = 100_000_000
CORE_BALANCED_SCHEDULE_CYCLES = 5
CORE_PROFILE_ORDER_COUNT = 6
CORE_METRIC_ROTATION_COUNT = 4
CORE_ORDER_BOOTSTRAP_RESAMPLES = 200_000
CORE_ORDER_COMPARISON_CELLS = len(PROFILE_ORDER) * len(ALL_METRICS)
CORE_ORDER_POSITION_PAIR_CONTRASTS = 3
CORE_ORDER_TOTAL_COMPARISONS = (
    CORE_ORDER_COMPARISON_CELLS * CORE_ORDER_POSITION_PAIR_CONTRASTS
)
MIN_NATIVE_ORDER_REPETITIONS = 30
NATIVE_ORDER_INVESTIGATE_PERCENT = 1.0
GPU_NATIVE_CLOCK_POWER_CAPTURE_BOUNDARY = (
    "measurement-before is captured after the discarded calibration prepass and "
    "before randomized recorded cycles; measurement-after follows cycle collection "
    "and record verification"
)
NATIVE_ORDER_EVIDENCE_LANES = (
    "canonical_payload_api",
    "supplemental_internal_kernel",
    "supplemental_host_phase",
)
# CUDA and ROCm native evidence is collected in one fresh helper process per
# lane.  A lane is a measurement category, not a hint that may be inferred
# from an operation spelling.  Keep the contract here so a forged top-level
# marker cannot make records from another category claim-ready.
NATIVE_EVIDENCE_LANE_SET = frozenset(NATIVE_ORDER_EVIDENCE_LANES)
NATIVE_LANE_REQUIRED_OPERATIONS = {
    # The canonical lane is intentionally a small ABI-boundary sample.  Its
    # complete operation surface is authenticated by the canonical lifecycle
    # record; the timing artifact only needs one explicit execute boundary per
    # profile.
    "canonical_payload_api": frozenset(("supplemental:payload_execute",)),
    # Direct helper-owned kernels are the only records that can support a
    # kernel-level arithmetic claim.  The backend-specific sets preserve the
    # existing contracted device decomposition without requiring host phases
    # to be duplicated in every direct artifact.
    "supplemental_internal_kernel": frozenset(
        {
            "candidate_materialization",
            "target_stat_preparation",
            "feature_stat_preparation",
            "metric:pearson",
            "metric:spearman",
            "metric:mutual_info",
            "metric:r2",
            "ranking_target_ranks",
            "ranking_topk",
            "selected_row_gather",
        }
    ),
    # Host-control evidence answers conversion/planning/lifecycle questions;
    # it is never pooled with direct device timing.
    "supplemental_host_phase": frozenset(
        {
            "ingest_conversion",
            "planning",
            "allocation",
            "h2d_upload",
            "d2h_transfer",
            "report_construction",
        }
    ),
}
NATIVE_LANE_EXECUTION_MODES = {
    "canonical_payload_api": frozenset(("canonical_payload",)),
    "supplemental_internal_kernel": frozenset(("supplemental_internal_kernel",)),
    "supplemental_host_phase": frozenset(
        ("supplemental_host_phase", "supplemental_host_control")
    ),
}
NATIVE_PAYLOAD_NOT_LOADED_MARKERS = frozenset(
    ("not_loaded", "not_collected", "payload_not_loaded")
)
NATIVE_LANE_ISOLATION = "fresh_helper_process_per_variant_trial_and_lane"
ORDER_EFFECT_FAMILYWISE_CONFIDENCE = 0.95
ORDER_EFFECT_BOOTSTRAP_RESAMPLES = 10_000
ORDER_EFFECT_POSITION_CONTRASTS = ((0, 1), (0, 2), (1, 2))
ORDER_EFFECT_CLEAN_STATUS = (
    "no_order_effect_above_one_percent_with_95_percent_familywise_confidence"
)
ORDER_EFFECT_INCONCLUSIVE_STATUS = "inconclusive_order_effect_requires_rerun"
ORDER_EFFECT_CONTAMINATED_STATUS = "confirmed_order_contamination_above_one_percent"
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
NATIVE_AB_PROCESS_ISOLATION = "fresh_helper_process_per_variant_trial"
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
        default=MIN_PUBLIC_ORDER_REPETITIONS,
        help=(
            "repeat the six profile-order blocks this many times; release claims "
            f"require at least {MIN_PUBLIC_ORDER_REPETITIONS} complete cycles"
        ),
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
    if args.order_repetitions < MIN_PUBLIC_ORDER_REPETITIONS:
        parser.error(
            "--order-repetitions must be at least "
            f"{MIN_PUBLIC_ORDER_REPETITIONS} for release claims"
        )
    if args.ab_blocks < 1:
        parser.error("--ab-blocks must be positive")
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
    named_evidence_names = {
        name for name, _ in native_evidence_specs if name is not None
    }
    if len(args.variants) == 2:
        if (
            len(native_evidence_specs) != 2
            or any(name is None for name, _ in native_evidence_specs)
            or named_evidence_names != variant_names
        ):
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
        # Worker jobs cross a JSON subprocess boundary.  Keep the submitted
        # payload in canonical JSON types so the parent can compare the exact
        # job with the worker's decoded result without tuple/list drift.
        "metrics": list(workload.metrics),
        "expected_candidate_count": workload.expected_candidates,
    }


def _canonical_json_bytes(value: object) -> bytes:
    """Return the finite, deterministic encoding used for evidence bindings."""

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _expected_input_binding(
    job: Mapping[str, object], *, precision: str, workload: Mapping[str, object]
) -> dict[str, object]:
    """Recompute the deterministic input recipe expected for one public cell.

    The parent driver intentionally remains standard-library-only.  This recipe
    therefore authenticates the complete dataset generator inputs, shapes,
    feature-name identity, and owned dtypes independently of the worker's
    reported byte hashes; the worker's matrix/target hashes remain useful raw
    observations and are checked for structural consistency below.
    """

    input_policy = str(job["input_policy"])
    samples = int(workload["samples"])
    features = int(workload["features"])
    source_dtype = (
        "float32"
        if input_policy == "native" and precision in ("fp32", "mixed")
        else "float64"
    )
    names = [f"x{column}" for column in range(features)]
    recipe: dict[str, object] = {
        "schema": "gafime.perf13-input-binding.v1",
        "generator": "make_dataset_numpy_default_rng_v1",
        "seed": int(job["seed"]),
        "input_policy": input_policy,
        "precision": precision,
        "workload": dict(workload),
        "matrix_shape": [samples, features],
        "target_shape": [samples],
        "matrix_dtype": source_dtype,
        "target_dtype": source_dtype,
        "feature_names_sha256": hashlib.sha256(
            "\0".join(names).encode("utf-8")
        ).hexdigest(),
    }
    recipe["binding_sha256"] = hashlib.sha256(_canonical_json_bytes(recipe)).hexdigest()
    return recipe


def _expected_input_identity_shape(binding: Mapping[str, object]) -> dict[str, object]:
    """Return the independently known portion of a worker input identity."""

    return {
        key: binding[key]
        for key in (
            "matrix_shape",
            "target_shape",
            "matrix_dtype",
            "target_dtype",
            "feature_names_sha256",
        )
        if key in binding
    }


def _public_schedule_key(item: Mapping[str, object]) -> tuple[object, ...] | None:
    """Normalize one scheduled/result public job into an exact coverage key."""

    workload = item.get("workload")
    workload_name = (
        str(workload.get("name"))
        if isinstance(workload, Mapping)
        else str(workload)
        if isinstance(workload, str)
        else None
    )
    variant = item.get("variant")
    if variant is None:
        provenance = item.get("provenance")
        variant = provenance.get("variant") if isinstance(provenance, Mapping) else None
    profile_order = item.get("profile_order")
    surface_order = item.get("surface_order")
    sequence = item.get("variant_sequence")
    if not (
        isinstance(variant, str)
        and isinstance(workload_name, str)
        and isinstance(profile_order, (list, tuple))
        and isinstance(surface_order, (list, tuple))
        and isinstance(sequence, (list, tuple))
    ):
        return None
    order_repeat = item.get("order_repeat")
    order_index = item.get("order_index")
    block = item.get("ab_block")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (order_repeat, order_index, block)
    ):
        return None
    return (
        str(item.get("backend")),
        variant,
        tuple(str(value) for value in profile_order),
        tuple(str(value) for value in surface_order),
        workload_name,
        str(item.get("input_policy")),
        int(order_repeat),
        int(order_index),
        int(block),
        tuple(str(value) for value in sequence),
        bool(item.get("interleaved_control", False)),
    )


def _worker_binding_payload(job: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "kind",
        "variant",
        "backend",
        "precision",
        "profile_order_context",
        "profile_order",
        "surface_order",
        "input_policy",
        "workload",
        "warmups",
        "repetitions",
        "bootstrap_resamples",
        "min_sample_ns",
        "max_loop_count",
        "order_repeat",
        "order_index",
        "ab_block",
        "variant_sequence",
        "interleaved_control",
        "seed",
        "device_id",
    )
    return {field: _json_safe(job.get(field)) for field in fields if field in job}


def _worker_binding_digest(job: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_worker_binding_payload(job))
    ).hexdigest()


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
        "bootstrap_resamples": bootstrap_resamples,
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


def _distribution_consistency_failures(
    distribution: Mapping[str, object],
    *,
    expected_repetitions: int | None = None,
    expected_target_ns: int | None = None,
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
    require_full: bool = True,
) -> list[str]:
    """Recompute public/control distribution fields from finite raw samples."""

    failures: list[str] = []
    raw = distribution.get("raw_per_call_duration_ns")
    if not isinstance(raw, list) or not raw:
        return ["raw_per_call_duration_ns_required"]
    if expected_repetitions is not None and len(raw) != expected_repetitions:
        failures.append("raw_per_call_duration_ns_count_mismatch")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in raw
    ):
        failures.append("raw_per_call_duration_ns_must_be_finite_positive")
        return failures
    values = [float(value) for value in raw]
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    expected_stats = {
        "median_ns": median,
        "mad_ns": mad,
        "p05_ns": _percentile(values, 0.05),
        "p95_ns": _percentile(values, 0.95),
    }
    for name, expected in expected_stats.items():
        observed = distribution.get(name)
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            if require_full:
                failures.append(f"{name}_required")
            continue
        if not math.isfinite(float(observed)) or not math.isclose(
            float(observed), expected, rel_tol=1.0e-12, abs_tol=1.0e-9
        ):
            failures.append(f"{name}_does_not_match_raw_samples")

    measured = distribution.get("measured_repetitions")
    if measured is not None and measured != len(raw):
        failures.append("measured_repetitions_does_not_match_raw_samples")
    elif measured is None and require_full:
        failures.append("measured_repetitions_required")

    loop_count = distribution.get("loop_count_per_repetition")
    raw_region = distribution.get("raw_region_duration_ns")
    if require_full or raw_region is not None or loop_count is not None:
        if (
            isinstance(loop_count, bool)
            or not isinstance(loop_count, int)
            or loop_count < 1
        ):
            failures.append("loop_count_per_repetition_required")
        if not isinstance(raw_region, list) or len(raw_region) != len(raw):
            failures.append("raw_region_duration_ns_count_mismatch")
        elif any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in raw_region
        ):
            failures.append("raw_region_duration_ns_must_be_finite_positive")
        elif isinstance(loop_count, int) and loop_count > 0:
            if any(
                not math.isclose(
                    float(per_call),
                    float(region) / loop_count,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                )
                for per_call, region in zip(raw, raw_region)
            ):
                failures.append("raw_and_normalized_duration_mismatch")

        target = distribution.get("sample_region_target_ns")
        if expected_target_ns is not None and target != expected_target_ns:
            failures.append("sample_region_target_does_not_match_job")
        if not isinstance(target, (int, float)) or isinstance(target, bool):
            failures.append("sample_region_target_required")
        elif not math.isfinite(float(target)) or float(target) <= 0.0:
            failures.append("sample_region_target_must_be_finite_positive")
        if (
            isinstance(raw_region, list)
            and raw_region
            and not any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in raw_region
            )
        ):
            expected_min = min(float(value) for value in raw_region)
            observed_min = distribution.get("sample_region_min_observed_ns")
            if not isinstance(observed_min, (int, float)) or not math.isclose(
                float(observed_min), expected_min, rel_tol=0.0, abs_tol=1.0e-9
            ):
                failures.append("sample_region_min_does_not_match_raw_region")
            observed_target_met = distribution.get("sample_region_target_met")
            if isinstance(target, (int, float)) and not isinstance(target, bool):
                if observed_target_met is not all(
                    float(value) >= float(target) for value in raw_region
                ):
                    failures.append("sample_region_target_marker_mismatch")

    if bootstrap_seed is not None or bootstrap_resamples is not None:
        count = distribution.get("bootstrap_resamples")
        if count is None and require_full:
            failures.append("bootstrap_resamples_required")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            failures.append("bootstrap_resamples_invalid")
        elif bootstrap_resamples is not None and count != bootstrap_resamples:
            failures.append("bootstrap_resamples_does_not_match_job")
        if isinstance(count, int) and count > 0 and bootstrap_seed is not None:
            rng = random.Random(bootstrap_seed)
            bootstrap = [
                statistics.median(rng.choice(values) for _ in values)
                for _ in range(count)
            ]
            expected_ci = [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
            observed_ci = distribution.get("bootstrap_median_95_ci_ns")
            if (
                not isinstance(observed_ci, (list, tuple))
                or len(observed_ci) != 2
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in observed_ci
                )
                or any(
                    not math.isclose(
                        float(observed), expected, rel_tol=1.0e-12, abs_tol=1.0e-9
                    )
                    for observed, expected in zip(observed_ci, expected_ci)
                )
            ):
                failures.append("bootstrap_ci_does_not_match_raw_samples")
    return failures


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
    if numeric and all(value == numeric[0] for value in numeric):
        # A constant empirical distribution has a point-mass bootstrap median.
        # Preserve the declared resample count while avoiding thousands of
        # identical draws for every order/profile fixture record.
        bootstrap_interval = [center, center]
    else:
        rng = random.Random(seed)
        bootstrap = [
            statistics.median(rng.choice(numeric) for _ in numeric)
            for _ in range(DEFAULT_BOOTSTRAP_RESAMPLES)
        ]
        bootstrap_interval = [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
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
        "bootstrap_median_95_ci": bootstrap_interval,
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "auto_scaling": dict(
            auto_scaling or {"status": "not_observed_in_native_artifact"}
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


def _trusted_git_executable() -> Path | None:
    """Return one absolute system Git, never a caller-controlled PATH shim."""

    for candidate in (Path("/usr/bin/git"), Path("/bin/git")):
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


def _git_environment() -> tuple[dict[str, str], list[str]]:
    """Remove inherited Git redirection and disable global/system config."""

    removed = sorted(key for key in os.environ if key.startswith("GIT_"))
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    return environment, removed


def _trusted_git_output(
    source_root: str | Path, *arguments: str, timeout: int = 10
) -> dict[str, object]:
    executable = _trusted_git_executable()
    if executable is None:
        return {
            "status": "unavailable",
            "command": [],
            "output": "trusted absolute Git executable unavailable",
        }
    environment, _ = _git_environment()
    command = [str(executable), "-C", str(source_root), *arguments]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "command": command, "output": str(exc)}
    output = (result.stdout + result.stderr).strip()
    return {
        "status": "pass" if result.returncode == 0 else "error",
        "command": command,
        "returncode": result.returncode,
        "output": output[:16_384],
    }


def _git_commit(source_root: str | None) -> str | None:
    if source_root is None:
        return None
    result = _trusted_git_output(source_root, "rev-parse", "HEAD")
    if result.get("status") != "pass":
        return None
    commit = str(result["output"]).splitlines()[0].strip()
    return commit if re.fullmatch(r"[0-9a-fA-F]{40}", commit) else None


def _git_state(source_root: str | None) -> dict[str, object]:
    if source_root is None:
        return {"status": "not_supplied", "entries": []}
    result = _trusted_git_output(
        source_root, "status", "--porcelain=v1", "--untracked-files=all"
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


def _git_source_blob(
    source_root: str | Path, source_path: str | Path
) -> dict[str, object]:
    """Bind current source bytes to the exact blob at ``HEAD``."""

    root = Path(source_root).expanduser().resolve()
    path = Path(source_path).expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return {"status": "invalid", "detail": "source_path_outside_git_root"}
    if not path.is_file():
        return {"status": "invalid", "detail": "source_path_missing"}
    relative_text = relative.as_posix()
    current = _trusted_git_output(
        root, "hash-object", "--no-filters", "--", relative_text
    )
    head = _trusted_git_output(root, "rev-parse", f"HEAD:{relative_text}")
    current_blob = (
        str(current.get("output", "")).splitlines()[0].strip()
        if current.get("status") == "pass"
        else None
    )
    head_blob = (
        str(head.get("output", "")).splitlines()[0].strip()
        if head.get("status") == "pass"
        else None
    )
    valid = bool(
        isinstance(current_blob, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", current_blob)
        and current_blob == head_blob
    )
    return {
        "status": "tracked_at_head" if valid else "invalid",
        "path": str(path),
        "relative_path": relative_text,
        "source_sha256": _sha256(path),
        "current_git_blob": current_blob,
        "head_git_blob": head_blob,
    }


def _git_provenance(
    source_root: str | Path, *, source_path: str | Path
) -> dict[str, object]:
    """Capture fail-closed public Git identity and one tracked source blob."""

    root = Path(source_root).expanduser().resolve()
    executable = _trusted_git_executable()
    environment, removed = _git_environment()
    del environment
    top_level = _trusted_git_output(root, "rev-parse", "--show-toplevel")
    git_dir = _trusted_git_output(root, "rev-parse", "--absolute-git-dir")
    common_dir = _trusted_git_output(root, "rev-parse", "--git-common-dir")
    version = (
        _trusted_git_output(root, "--version")
        if root.is_dir()
        else {"status": "unavailable", "output": None}
    )
    commit = _git_commit(str(root))
    tree = _trusted_git_output(root, "rev-parse", "HEAD^{tree}")
    reported_root = (
        Path(str(top_level.get("output"))).resolve()
        if top_level.get("status") == "pass"
        else None
    )
    reported_git_dir = (
        Path(str(git_dir.get("output"))).resolve()
        if git_dir.get("status") == "pass"
        else None
    )
    raw_common_dir = str(common_dir.get("output", ""))
    reported_common_dir = None
    if common_dir.get("status") == "pass" and raw_common_dir:
        common_path = Path(raw_common_dir)
        reported_common_dir = (
            common_path.resolve()
            if common_path.is_absolute()
            else (root / common_path).resolve()
        )
    source_blob = _git_source_blob(root, source_path)
    tree_value = str(tree.get("output", "")).splitlines()[0].strip()
    trusted = bool(
        executable is not None
        and reported_root == root
        and reported_git_dir is not None
        and reported_common_dir is not None
        and commit is not None
        and re.fullmatch(r"[0-9a-fA-F]{40}", tree_value)
        and source_blob.get("status") == "tracked_at_head"
    )
    return {
        "status": "trusted" if trusted else "invalid",
        "root": str(root),
        "commit": commit,
        "tree": tree_value if tree.get("status") == "pass" else None,
        "git": {
            "path": str(executable) if executable is not None else None,
            "sha256": _sha256(executable) if executable is not None else None,
            "version": version.get("output"),
            "git_dir": str(reported_git_dir) if reported_git_dir else None,
            "git_common_dir": (
                str(reported_common_dir) if reported_common_dir else None
            ),
            "removed_environment": removed,
            "config_isolation": {
                "system": os.devnull,
                "global": os.devnull,
                "nosystem": True,
            },
        },
        "source_blob": source_blob,
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
                    archive.read(record_paths[0])
                    .decode("utf-8", errors="strict")
                    .splitlines()
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
        str(value.get("path")) for value in loaded_modules if value.get("path")
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
                relative = (
                    Path(module_path)
                    .resolve()
                    .relative_to(Path(root).resolve())
                    .as_posix()
                )
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
                relative = (
                    Path(str(binary_path))
                    .resolve()
                    .relative_to(Path(root).resolve())
                    .as_posix()
                )
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
        "wheel_identities": [
            _wheel_identity_summary(identity) for identity in identities
        ],
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
        if rocm_smi.get("status") == "pass" and _rocm_dynamic_telemetry_fields(
            rocm_smi
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
        parsed = _strict_json_loads(output)
    except (json.JSONDecodeError, _DuplicateJsonKeyError):
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
            token in leaf for token in ("average", "current", "draw", "consumption")
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
    source_git_provenance: dict[str, object] | None = None
    if source_root:
        source_git_provenance = _git_provenance(
            str(source_root), source_path=Path(str(source_root)) / "Cargo.lock"
        )
    benchmark_root = Path(__file__).resolve().parents[2]
    benchmark_git_provenance = _git_provenance(
        benchmark_root, source_path=Path(__file__).resolve()
    )
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
        "source_commit": (
            source_git_provenance.get("commit")
            if isinstance(source_git_provenance, Mapping)
            else None
        ),
        "source_tree_state": source_tree_state,
        "source_git_provenance": source_git_provenance,
        "benchmark_script": benchmark_script,
        "benchmark_script_canonical": benchmark_script_canonical,
        "benchmark_git_provenance": benchmark_git_provenance,
        "wheel_artifacts": [
            _wheel_identity_summary(_wheel_identity(wheel)) for wheel in wheels
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
                        and getattr(artifact_holder[0], "graph_replayed", None)
                        is not True
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
                        "input_binding": _expected_input_binding(
                            job,
                            precision=precision,
                            workload=_workload_payload(workload),
                        ),
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
            "order_sensitivity_status": "pending_simultaneous_parent_assessment",
            "order_sensitivity_interpretation": (
                "the point spread is diagnostic only; the parent report resamples "
                "complete six-order cycles and applies a familywise three-state gate"
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
        "measurement_category": "public_end_to_end_order_control",
        "profile_block_orders": [list(order) for order in measured_orders],
        "all_possible_orders_covered": {tuple(order) for order in measured_orders}
        == set(possible_orders),
        "timing_scope": (
            "already-warmed public backend; each recorded block interleaves profiles "
            "in a balanced randomized order to expose thermal or clock contamination; "
            + (
                "resident samples still include public coercion, digest, cache lookup, "
                "execution, and report materialization and are not pure device timings"
                if surface == "resident"
                else "the named public surface remains the measured boundary"
            )
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
        "input_binding": _expected_input_binding(
            job, precision=precision, workload=_workload_payload(workload)
        ),
        "output_row_count": candidate_count,
        "measurement_category": "public_end_to_end",
        "timing_scope": (
            "public resident-cache hit including input coercion/ownership checks, "
            "input digest hashing, cache lookup, execution, and report materialization; "
            "not a pure resident or device timer"
            if surface == "resident"
            else "public wall clock including the report materialization performed by "
            "this public surface; not a device-kernel timer"
        ),
        "unobservable_phases": (
            [
                "resident cache lookup versus digest hashing",
                "payload wrapper versus device execution",
                "device result transfer versus public report construction",
            ]
            if surface == "resident"
            else [
                "payload wrapper versus device execution",
                "device result transfer versus public report construction",
            ]
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
        "variant": job["variant"],
        "profile": precision,
        "backend": backend,
        "profile_order_context": list(job["profile_order_context"]),
        "order_repeat": job["order_repeat"],
        "order_index": job["order_index"],
        "ab_block": job["ab_block"],
        "variant_sequence": list(job.get("variant_sequence", ())),
        "input_policy": job["input_policy"],
        "seed": job["seed"],
        "device_id": job["device_id"],
        "job_binding": job["job_binding"],
        "source_dtype": source_dtype,
        "input_identity": _dataset_identity(matrix, target, names),
        "input_binding": _expected_input_binding(
            job, precision=precision, workload=_workload_payload(workload)
        ),
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
        "variant": job["variant"],
        "input_policy": job["input_policy"],
        "workload": _workload_payload(workload),
        "ab_block": job["ab_block"],
        "variant_sequence": list(job.get("variant_sequence", ())),
        "order_repeat": job["order_repeat"],
        "order_index": job["order_index"],
        "profile_order": list(job["profile_order"]),
        "surface_order": list(job["surface_order"]),
        "interleaved_control": bool(job.get("interleaved_control", False)),
        "seed": job["seed"],
        "device_id": job["device_id"],
        "warmups": job["warmups"],
        "repetitions": job["repetitions"],
        "bootstrap_resamples": job["bootstrap_resamples"],
        "min_sample_ns": job["min_sample_ns"],
        "max_loop_count": job["max_loop_count"],
        "job_binding": job["job_binding"],
        "capabilities": capability_records,
        "cells": cells,
        "interleaved_controls": interleaved_controls,
        "native_binaries": native_binaries,
        "worker_elapsed_ns": worker_elapsed_ns,
        "provenance": provenance,
    }


def _worker_main() -> int:
    process_start_ns = perf_counter_ns()
    job = _strict_json_loads(sys.stdin.read())
    if not isinstance(job, Mapping):
        raise ValueError("worker input must be a JSON object")
    if job["kind"] == "cold":
        result = _cold_worker(job, process_start_ns)
    elif job["kind"] == "public":
        result = _public_worker(job, process_start_ns)
    else:
        raise ValueError(f"unknown worker kind {job['kind']!r}")
    sys.stdout.write(
        json.dumps(_json_safe(result), sort_keys=True, allow_nan=False) + "\n"
    )
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
    job["job_binding"] = _worker_binding_digest(job)
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
    worker_environment["PATH"] = os.pathsep.join((str(python_bin), *filtered_path))
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
        input=json.dumps(job, allow_nan=False),
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
        result = _strict_json_loads(completed.stdout)
    except (json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise RuntimeError(
            f"worker returned invalid JSON: {completed.stdout[-4096:]}"
        ) from exc
    if not isinstance(result, Mapping):
        raise RuntimeError("worker returned a non-object JSON result")
    expected_binding = _worker_binding_digest(job)
    if result.get("job_binding") != expected_binding:
        raise RuntimeError(
            "worker result job_binding does not match the submitted job: "
            f"expected {expected_binding}, observed {result.get('job_binding')!r}"
        )
    expected_fields = (
        "kind",
        "variant",
        "backend",
        "input_policy",
        "workload",
        "order_repeat",
        "order_index",
        "ab_block",
        "variant_sequence",
    )
    if job.get("kind") == "public":
        expected_fields += (
            "profile_order",
            "surface_order",
            "interleaved_control",
            "seed",
            "device_id",
        )
    else:
        expected_fields += ("profile_order_context", "seed", "device_id")
    for field in expected_fields:
        if result.get(field) != job.get(field):
            raise RuntimeError(
                f"worker result field {field!r} does not match the submitted job: "
                f"expected {job.get(field)!r}, observed {result.get(field)!r}"
            )
    if job.get("kind") == "cold" and result.get("profile") != job.get("precision"):
        raise RuntimeError(
            "cold worker result field 'profile' does not match the submitted "
            "precision: "
            f"expected {job.get('precision')!r}, observed {result.get('profile')!r}"
        )
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
    groups: dict[
        tuple[object, ...],
        dict[object, dict[object, dict[int, list[float]]]],
    ] = {}
    orders_by_cluster: dict[
        tuple[object, ...], dict[tuple[object, object], set[tuple[str, ...]]]
    ] = {}
    counts_by_cluster: dict[tuple[object, ...], dict[tuple[object, object], int]] = {}
    invalid_cells: set[tuple[object, ...]] = set()
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
        if len(profile_order) < 2:
            continue
        expected_orders = set(itertools.permutations(profile_order))
        if (
            set(profile_order) != set(PROFILE_ORDER)
            or profile_order not in expected_orders
        ):
            continue
        order_repeat = result.get("order_repeat")
        ab_block = result.get("ab_block")
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
            profile = str(cell.get("profile"))
            raw = distribution.get("raw_per_call_duration_ns")
            ordinal = cell.get("profile_order_ordinal")
            if not (
                isinstance(order_repeat, int)
                and not isinstance(order_repeat, bool)
                and order_repeat >= 0
                and isinstance(ab_block, int)
                and not isinstance(ab_block, bool)
                and ab_block >= 0
                and profile in profile_order
                and isinstance(ordinal, int)
                and not isinstance(ordinal, bool)
                and ordinal == profile_order.index(profile)
                and isinstance(raw, list)
                and len(raw) >= MIN_REPETITIONS
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0.0
                    for value in raw
                )
            ):
                invalid_cells.add(key)
                continue
            cluster_key = (ab_block, order_repeat)
            stratum_key = (
                variant,
                result.get("backend"),
                workload_name,
                result.get("input_policy"),
                ab_block,
            )
            groups.setdefault(key, {}).setdefault(stratum_key, {}).setdefault(
                order_repeat, {0: [], 1: [], 2: []}
            )[ordinal].append(statistics.median(float(value) for value in raw))
            orders_by_cluster.setdefault(key, {}).setdefault(cluster_key, set()).add(
                profile_order
            )
            counts_by_cluster.setdefault(key, {}).setdefault(cluster_key, 0)
            counts_by_cluster[key][cluster_key] += 1

    complete: dict[
        tuple[object, ...],
        dict[object, dict[object, dict[int, list[float]]]],
    ] = {}
    incomplete: set[tuple[object, ...]] = set(invalid_cells)
    canonical_orders = set(itertools.permutations(PROFILE_ORDER))
    for key, strata in groups.items():
        cell_complete = True
        for stratum, cycles in strata.items():
            if len(cycles) < MIN_PUBLIC_ORDER_REPETITIONS:
                cell_complete = False
            for cycle, positions in cycles.items():
                ab_block = stratum[-1] if isinstance(stratum, tuple) else stratum
                cluster_key = (ab_block, cycle)
                if (
                    set(positions) != {0, 1, 2}
                    or any(len(positions[position]) != 2 for position in range(3))
                    or orders_by_cluster.get(key, {}).get(cluster_key)
                    != canonical_orders
                    or counts_by_cluster.get(key, {}).get(cluster_key) != 6
                ):
                    cell_complete = False
        if cell_complete and strata:
            complete[key] = strata
        else:
            incomplete.add(key)

    assessment = (
        _simultaneous_clustered_order_assessment(
            complete,
            seed=0x5055424C49434F52,
        )
        if complete
        else {"cells": {}}
    )
    assessed_cells = assessment.get("cells", {})
    summaries: list[dict[str, object]] = []
    for key in sorted(set(groups) | incomplete, key=repr):
        cell_assessment = (
            assessed_cells.get(key, {}) if isinstance(assessed_cells, Mapping) else {}
        )
        position_medians = cell_assessment.get("position_medians", {})
        spread_percent = cell_assessment.get(
            "max_observed_position_spread_percent", 0.0
        )
        cluster_count = sum(len(cycles) for cycles in groups.get(key, {}).values())
        summaries.append(
            {
                "variant": key[0],
                "backend": key[1],
                "workload": key[2],
                "input_policy": key[3],
                "surface": key[4],
                "profile": key[5],
                "observation_count": cluster_count * 6,
                "position_median_ns": position_medians,
                "max_position_median_spread_percent": spread_percent,
                "status": cell_assessment.get(
                    "status", "insufficient_complete_order_cycle_evidence"
                ),
                "position_contrasts": cell_assessment.get("position_contrasts", []),
                "cluster_count": cluster_count,
                "familywise_confidence_level": assessment.get(
                    "familywise_confidence_level"
                ),
                "multiple_comparison_correction": assessment.get(
                    "multiple_comparison_correction"
                ),
                "comparison_cells": assessment.get("comparison_cells"),
                "total_comparisons": assessment.get("total_comparisons"),
                "bootstrap_resamples": assessment.get("bootstrap_resamples"),
                "raw_sample_clustering": assessment.get("raw_sample_clustering"),
                "interpretation": (
                    "Clean requires every simultaneous upper absolute position-contrast "
                    "bound to be at most one percent; an interval overlapping the "
                    "threshold is inconclusive, not evidence of equivalence."
                ),
            }
        )
    return summaries


def _interleaved_order_sensitivity(
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    cells: dict[
        tuple[object, ...],
        dict[object, dict[object, dict[int, list[float]]]],
    ] = {}
    metadata: dict[tuple[object, ...], dict[str, object]] = {}
    incomplete: set[tuple[object, ...]] = set()
    canonical_orders = set(itertools.permutations(PROFILE_ORDER))
    for result_index, result in enumerate(results):
        if result.get("kind") != "public":
            continue
        for control_index, control in enumerate(result.get("interleaved_controls", ())):
            if not isinstance(control, Mapping) or control.get("status") != "pass":
                continue
            profiles = control.get("profiles")
            if not isinstance(profiles, Mapping):
                continue
            backend = str(result.get("backend"))
            expected_profiles = set(BACKEND_PROFILES.get(backend, ()))
            normalized_profiles = {
                str(profile): profile_result
                for profile, profile_result in profiles.items()
            }
            observed_profiles = {
                profile
                for profile, profile_result in normalized_profiles.items()
                if isinstance(profile_result, Mapping)
            }
            complete_profile_coverage = (
                bool(expected_profiles) and observed_profiles == expected_profiles
            )
            raw_orders = control.get("profile_block_orders")
            complete_order_cycles = (
                isinstance(raw_orders, list)
                and len(raw_orders) >= MIN_REPETITIONS
                and len(raw_orders) % 6 == 0
                and all(
                    {
                        tuple(str(item) for item in order)
                        for order in raw_orders[start : start + 6]
                        if isinstance(order, (list, tuple))
                    }
                    == canonical_orders
                    for start in range(0, len(raw_orders), 6)
                )
            )
            for profile in sorted(expected_profiles | set(normalized_profiles)):
                profile_result = normalized_profiles.get(profile)
                key = (result_index, control_index, profile)
                metadata[key] = {
                    "result_index": result_index,
                    "control_index": control_index,
                    "backend": backend,
                    "workload": result.get("workload", {}).get("name")
                    if isinstance(result.get("workload"), Mapping)
                    else None,
                    "input_policy": result.get("input_policy"),
                    "surface": control.get("surface"),
                    "profile": profile,
                }
                if not complete_profile_coverage or not isinstance(
                    profile_result, Mapping
                ):
                    incomplete.add(key)
                    continue
                distribution = profile_result.get("distribution")
                raw_values = (
                    distribution.get("raw_per_call_duration_ns")
                    if isinstance(distribution, Mapping)
                    else None
                )
                if not (
                    complete_order_cycles
                    and isinstance(raw_orders, list)
                    and isinstance(raw_values, list)
                    and len(raw_values) == len(raw_orders)
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) > 0.0
                        for value in raw_values
                    )
                ):
                    incomplete.add(key)
                    continue
                stratum = (result_index, control_index)
                for sample_index, (order, value) in enumerate(
                    zip(raw_orders, raw_values)
                ):
                    order_tuple = (
                        tuple(str(item) for item in order)
                        if isinstance(order, (list, tuple))
                        else ()
                    )
                    if (
                        order_tuple not in canonical_orders
                        or str(profile) not in order_tuple
                    ):
                        incomplete.add(key)
                        continue
                    cycle = sample_index // 6
                    position = order_tuple.index(str(profile))
                    cells.setdefault(key, {}).setdefault(stratum, {}).setdefault(
                        cycle, {0: [], 1: [], 2: []}
                    )[position].append(float(value))

    complete: dict[
        tuple[object, ...],
        dict[object, dict[object, dict[int, list[float]]]],
    ] = {}
    for key, strata in cells.items():
        valid = key not in incomplete
        for cycles in strata.values():
            valid = valid and len(cycles) >= MIN_PUBLIC_ORDER_REPETITIONS
            for positions in cycles.values():
                valid = (
                    valid
                    and set(positions) == {0, 1, 2}
                    and all(len(positions[position]) == 2 for position in range(3))
                )
        if valid:
            complete[key] = strata
        else:
            incomplete.add(key)

    assessment = (
        _simultaneous_clustered_order_assessment(
            complete,
            seed=0x494E5445524C4541,
        )
        if complete
        else {"cells": {}}
    )
    assessed_cells = assessment.get("cells", {})
    summaries: list[dict[str, object]] = []
    for key in sorted(metadata, key=repr):
        cell = (
            assessed_cells.get(key, {}) if isinstance(assessed_cells, Mapping) else {}
        )
        summaries.append(
            {
                **metadata[key],
                "status": cell.get(
                    "status", "insufficient_complete_order_cycle_evidence"
                ),
                "max_order_position_spread_percent": cell.get(
                    "max_observed_position_spread_percent"
                ),
                "position_contrasts": cell.get("position_contrasts", []),
                "familywise_confidence_level": assessment.get(
                    "familywise_confidence_level"
                ),
                "multiple_comparison_correction": assessment.get(
                    "multiple_comparison_correction"
                ),
                "comparison_cells": assessment.get("comparison_cells"),
                "total_comparisons": assessment.get("total_comparisons"),
                "bootstrap_resamples": assessment.get("bootstrap_resamples"),
                "raw_sample_clustering": assessment.get("raw_sample_clustering"),
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
        baseline_distribution_failures = _distribution_consistency_failures(
            baseline_distribution, require_full=False
        )
        candidate_distribution_failures = _distribution_consistency_failures(
            candidate_distribution, require_full=False
        )
        if baseline_distribution_failures or candidate_distribution_failures:
            # Invalid or forged distributions are never compared.  The public
            # readiness gate reports the detailed failure for the production
            # schedule; this lower-level helper must not consume declared stats.
            continue
        baseline_raw = baseline_distribution.get("raw_per_call_duration_ns", ())
        candidate_raw = candidate_distribution.get("raw_per_call_duration_ns", ())
        if not isinstance(baseline_raw, list) or not isinstance(candidate_raw, list):
            continue
        baseline_ns = float(statistics.median(float(value) for value in baseline_raw))
        candidate_ns = float(statistics.median(float(value) for value in candidate_raw))
        if (
            baseline_ns <= 0.0
            or not math.isfinite(baseline_ns)
            or not math.isfinite(candidate_ns)
        ):
            continue
        delta_percent = (candidate_ns - baseline_ns) * 100.0 / baseline_ns
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
                baseline_sample = [baseline_values[index] for index in baseline_indices]
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
            delta_ci_percent = [value * 100.0 / baseline_ns for value in delta_ci_ns]
        classification = _comparison_classification(delta_percent, delta_ci_percent)
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
                "sample_count_baseline": len(baseline_raw)
                if isinstance(baseline_raw, (list, tuple))
                else 0,
                "sample_count_candidate": len(candidate_raw)
                if isinstance(candidate_raw, (list, tuple))
                else 0,
                "effective_comparison_sample_count": effective_comparison_sample_count,
                "comparison_sample_count": effective_comparison_sample_count,
                "pairing": "independent_worker_distributions",
                "bootstrap_delta_median_95_ci_ns": delta_ci_ns,
                "bootstrap_candidate_latency_delta_95_ci_percent": delta_ci_percent,
                **classification,
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
                if (
                    not math.isfinite(float(phase_duration))
                    or float(phase_duration) <= 0
                ):
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
    delta_ci_percent = [value * 100.0 / baseline_median for value in delta_ci]
    return {
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "candidate_latency_delta_percent": delta_percent,
        "sample_count_baseline": len(baseline),
        "sample_count_candidate": len(candidate),
        "effective_comparison_sample_count": min(len(baseline), len(candidate)),
        "comparison_sample_count": min(len(baseline), len(candidate)),
        "bootstrap_delta_median_95_ci": delta_ci,
        "bootstrap_candidate_latency_delta_95_ci_percent": delta_ci_percent,
        "pairing": "independent_worker_distributions",
        **_comparison_classification(delta_percent, delta_ci_percent),
    }


def _comparison_classification(
    delta_percent: float, confidence_interval_percent: object
) -> dict[str, object]:
    """Classify an A/B delta only when its independent-bootstrap CI supports it.

    A point estimate is not a repeatable regression.  A regression or
    improvement is confirmed only when the 95 percent interval excludes zero;
    intervals crossing zero remain explicitly inconclusive and never trigger
    the one/three-percent escalation thresholds.
    """

    if (
        not isinstance(confidence_interval_percent, (list, tuple))
        or len(confidence_interval_percent) != 2
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in confidence_interval_percent
        )
    ):
        return {
            "review_status": "bootstrap_interval_missing",
            "confidence_interpretation": "insufficient_evidence",
            "repeatable_regression": False,
            "escalation": "measurement_invalid",
        }
    lower, upper = (float(value) for value in confidence_interval_percent)
    if lower > upper:
        return {
            "review_status": "bootstrap_interval_invalid",
            "confidence_interpretation": "insufficient_evidence",
            "repeatable_regression": False,
            "escalation": "measurement_invalid",
        }
    if lower <= 0.0 <= upper:
        return {
            "review_status": "inconclusive_ci_crosses_zero",
            "confidence_interpretation": "no_direction_confirmed",
            "repeatable_regression": False,
            "escalation": "none_inconclusive",
        }
    if upper < 0.0:
        return {
            "review_status": "confirmed_improvement",
            "confidence_interpretation": "candidate_faster_ci_excludes_zero",
            "repeatable_regression": False,
            "escalation": "none",
        }
    if delta_percent > 3.0:
        return {
            "review_status": "confirmed_regression_above_three_percent",
            "confidence_interpretation": "candidate_slower_ci_excludes_zero",
            "repeatable_regression": True,
            "escalation": "maintainer_approval_required",
        }
    if delta_percent > 1.0:
        return {
            "review_status": "confirmed_regression_above_one_percent",
            "confidence_interpretation": "candidate_slower_ci_excludes_zero",
            "repeatable_regression": True,
            "escalation": "investigate",
        }
    return {
        "review_status": "confirmed_regression_within_one_percent",
        "confidence_interpretation": "candidate_slower_ci_excludes_zero",
        "repeatable_regression": True,
        "escalation": "none",
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
            "raw_durations_ns",
            values[baseline.name].get("raw_clean_cold_interval_ns", ()),
        )
        candidate_raw = values[candidate.name].get(
            "raw_durations_ns",
            values[candidate.name].get("raw_clean_cold_interval_ns", ()),
        )
        if not isinstance(baseline_raw, (list, tuple)) or not isinstance(
            candidate_raw, (list, tuple)
        ):
            continue
        comparison = _independent_delta_summary(
            baseline_raw,
            candidate_raw,
            seed=seed
            ^ int.from_bytes(
                hashlib.sha256(repr(key).encode("utf-8")).digest()[:8], "little"
            ),
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
    # Never manufacture a native A/B from invalid/unsupported evidence.  Order
    # claim readiness is lane-scoped: a contaminated host-control family must
    # not erase an independently clean canonical-payload comparison, while
    # records from the non-clean family remain visible in the raw artifact.
    if len(variants) != 2 or native_evidence.get("valid") is not True:
        return []
    baseline, candidate = variants
    groups: dict[tuple[object, ...], dict[str, list[float]]] = {}
    for artifact in native_evidence.get("artifacts", ()):
        if not isinstance(artifact, Mapping):
            continue
        validation = artifact.get("validation")
        if (
            not isinstance(validation, Mapping)
            or validation.get("complete") is not True
        ):
            continue
        raw_claim_families = validation.get("native_order_claim_families")
        family_assessments = (
            raw_claim_families.get("families")
            if isinstance(raw_claim_families, Mapping)
            else None
        )
        variant = str(artifact.get("variant"))
        backend = str(artifact.get("backend"))
        path = artifact.get("path")
        if not isinstance(path, str):
            continue
        try:
            payload = _strict_json_loads(Path(path).read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJsonKeyError,
        ):
            continue
        records = payload.get("records") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            continue
        artifact_lane = validation.get("evidence_lane")
        strict_lane_contract = validation.get("lane_contract_active") is True
        if strict_lane_contract and artifact_lane not in NATIVE_EVIDENCE_LANE_SET:
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
        workload_identity = json.dumps(_json_safe(workload_descriptor), sort_keys=True)
        raw_input_policy = payload.get("input_policy", payload.get("input_policy_name"))
        raw_input_identity = payload.get(
            "input_identity", payload.get("dataset_identity")
        )
        manifest_schedule = artifact.get("schedule")
        manifest_schedule = (
            manifest_schedule if isinstance(manifest_schedule, Mapping) else {}
        )
        raw_variant_sequence = payload.get("variant_sequence")
        if raw_variant_sequence is None:
            raw_variant_sequence = manifest_schedule.get("variant_sequence")
        variant_sequence = (
            tuple(str(value) for value in raw_variant_sequence)
            if isinstance(raw_variant_sequence, (list, tuple))
            else None
        )
        ab_block = payload.get("ab_block")
        if ab_block is None:
            ab_block = manifest_schedule.get("ab_block")
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
        input_identity = json.dumps(_json_safe(raw_input_identity), sort_keys=True)
        loop_plan_identity = _native_loop_plan_identity(payload)
        if loop_plan_identity is None:
            loop_plan_identity = _native_loop_plan_identity(
                {"loop_plan": manifest_schedule.get("loop_plan")}
            )
        for record in records:
            if not isinstance(record, Mapping):
                continue
            record_lane = record.get("evidence_lane")
            if strict_lane_contract:
                if record_lane != artifact_lane:
                    continue
                evidence_lane = str(artifact_lane)
            else:
                evidence_lane = str(
                    record_lane
                    if record_lane is not None
                    else record.get(
                        "timing_mode",
                        payload.get("execution_mode", "unspecified"),
                    )
                )
            if isinstance(family_assessments, Mapping):
                family_assessment = family_assessments.get(evidence_lane)
                if (
                    not isinstance(family_assessment, Mapping)
                    or family_assessment.get("claim_ready") is not True
                ):
                    continue
            elif validation.get("performance_claim_ready") is not True:
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
            elif normalized_order_index is not None and declared_orders:
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
                ab_block,
                variant_sequence,
                evidence_lane,
                str(record.get("comparability", "unspecified")),
                record.get("loop_count_per_sample"),
                loop_plan_identity,
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
            seed=seed
            ^ int.from_bytes(
                hashlib.sha256(repr(key).encode("utf-8")).digest()[:8], "little"
            ),
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
                "ab_block": key[12],
                "variant_sequence": (
                    list(key[13]) if isinstance(key[13], tuple) else key[13]
                ),
                "measurement_category": key[14],
                "comparability": key[15],
                "loop_count_per_sample": key[16],
                "loop_plan_sha256": (
                    key[17][0] if isinstance(key[17], tuple) else key[17]
                ),
                "loop_plan_file_sha256": (
                    key[17][1] if isinstance(key[17], tuple) else None
                ),
                "baseline_variant": baseline.name,
                "candidate_variant": candidate.name,
                **comparison,
            }
        )
    return comparisons


def _native_ab_loop_count_failures(
    native_evidence: Mapping[str, object],
    variants: Sequence[Variant],
    backends: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Find A/B cells that cannot be compared because calibration differs.

    Native samples are normalized by each record's fixed loop count.  A
    baseline and candidate cell with different counts is not a like-for-like
    measurement, even if the normalized values happen to look similar.  Such a
    cell is reported as incomparable and is excluded from
    :func:`_native_ab_comparisons` by the loop-count-bearing key there.
    """

    if len(variants) != 2 or native_evidence.get("valid") is not True:
        return []
    baseline, candidate = variants
    requested_backends = {str(backend) for backend in backends}
    groups: dict[tuple[object, ...], dict[str, set[int | None]]] = {}
    plan_groups: dict[tuple[object, ...], dict[str, set[object]]] = {}
    for artifact in native_evidence.get("artifacts", ()):
        if not isinstance(artifact, Mapping):
            continue
        validation = artifact.get("validation")
        if (
            not isinstance(validation, Mapping)
            or validation.get("complete") is not True
        ):
            continue
        variant = str(artifact.get("variant"))
        if variant not in {baseline.name, candidate.name}:
            continue
        backend = str(artifact.get("backend"))
        if requested_backends and backend not in requested_backends:
            continue
        path = artifact.get("path")
        if not isinstance(path, str):
            continue
        try:
            payload = _strict_json_loads(Path(path).read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJsonKeyError,
        ):
            continue
        if not isinstance(payload, Mapping):
            continue
        records = payload.get("records")
        if not isinstance(records, list):
            continue
        artifact_lane = validation.get("evidence_lane")
        strict_lane_contract = validation.get("lane_contract_active") is True
        if strict_lane_contract and artifact_lane not in NATIVE_EVIDENCE_LANE_SET:
            continue
        raw_claim_families = validation.get("native_order_claim_families")
        family_assessments = (
            raw_claim_families.get("families")
            if isinstance(raw_claim_families, Mapping)
            else None
        )
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
        workload_identity = json.dumps(_json_safe(workload_descriptor), sort_keys=True)
        raw_input_policy = payload.get("input_policy", payload.get("input_policy_name"))
        raw_input_identity = payload.get(
            "input_identity", payload.get("dataset_identity")
        )
        manifest_schedule = artifact.get("schedule")
        manifest_schedule = (
            manifest_schedule if isinstance(manifest_schedule, Mapping) else {}
        )
        raw_variant_sequence = payload.get("variant_sequence")
        if raw_variant_sequence is None:
            raw_variant_sequence = manifest_schedule.get("variant_sequence")
        variant_sequence = (
            tuple(str(value) for value in raw_variant_sequence)
            if isinstance(raw_variant_sequence, (list, tuple))
            else None
        )
        ab_block = payload.get("ab_block")
        if ab_block is None:
            ab_block = manifest_schedule.get("ab_block")
        if (
            not isinstance(raw_input_policy, str)
            or raw_input_policy not in INPUT_POLICIES
            or not isinstance(raw_input_identity, Mapping)
            or not raw_input_identity
        ):
            continue
        input_identity = json.dumps(_json_safe(raw_input_identity), sort_keys=True)
        loop_plan_identity = _native_loop_plan_identity(payload)
        if loop_plan_identity is None:
            loop_plan_identity = _native_loop_plan_identity(
                {"loop_plan": manifest_schedule.get("loop_plan")}
            )
        for record in records:
            if not isinstance(record, Mapping):
                continue
            record_lane = record.get("evidence_lane")
            if strict_lane_contract:
                if record_lane != artifact_lane:
                    continue
                evidence_lane = str(artifact_lane)
            else:
                evidence_lane = str(
                    record_lane
                    if record_lane is not None
                    else record.get(
                        "timing_mode", payload.get("execution_mode", "unspecified")
                    )
                )
            if isinstance(family_assessments, Mapping):
                family_assessment = family_assessments.get(evidence_lane)
                if (
                    not isinstance(family_assessment, Mapping)
                    or family_assessment.get("claim_ready") is not True
                ):
                    continue
            elif validation.get("performance_claim_ready") is not True:
                continue
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
            elif normalized_order_index is not None and declared_orders:
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
                raw_input_policy,
                input_identity,
                str(record.get("profile")),
                str(record.get("operation")),
                str(record.get("metric")),
                normalized_order_index,
                profile_order,
                clock,
                timing_boundary,
                record.get("unit", "us" if "samples_us" in record else "ns"),
                ab_block,
                variant_sequence,
                evidence_lane,
                str(record.get("comparability", "unspecified")),
            )
            plan_groups.setdefault(key, {}).setdefault(variant, set()).add(
                loop_plan_identity
            )
            raw_loop_count = record.get("loop_count_per_sample")
            loop_count = (
                raw_loop_count
                if isinstance(raw_loop_count, int)
                and not isinstance(raw_loop_count, bool)
                and raw_loop_count > 0
                else None
            )
            groups.setdefault(key, {}).setdefault(variant, set()).add(loop_count)

    failures: list[dict[str, object]] = []
    for key, values in plan_groups.items():
        baseline_plans = values.get(baseline.name)
        candidate_plans = values.get(candidate.name)
        if (
            not baseline_plans
            or not candidate_plans
            or baseline_plans == candidate_plans
        ):
            continue
        failures.append(
            {
                "reason": "native_ab_loop_plan_mismatch_incomparable",
                "backend": key[0],
                "profile": key[4],
                "operation": key[5],
                "metric": key[6],
                "order_index": key[7],
                "evidence_lane": key[14],
                "baseline_plan_sha256": sorted(str(value) for value in baseline_plans),
                "candidate_plan_sha256": sorted(
                    str(value) for value in candidate_plans
                ),
            }
        )
    for key, values in groups.items():
        baseline_counts = values.get(baseline.name)
        candidate_counts = values.get(candidate.name)
        if (
            not baseline_counts
            or not candidate_counts
            or baseline_counts == candidate_counts
        ):
            continue
        failures.append(
            {
                "reason": "native_ab_loop_count_mismatch_incomparable",
                "backend": key[0],
                "profile": key[4],
                "operation": key[5],
                "metric": key[6],
                "order_index": key[7],
                "evidence_lane": key[14],
                "baseline_loop_counts": sorted(
                    value if value is not None else 0 for value in baseline_counts
                ),
                "candidate_loop_counts": sorted(
                    value if value is not None else 0 for value in candidate_counts
                ),
            }
        )
    return failures


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
        grouped.setdefault(key, {}).setdefault(
            int(item.get("ab_block", -1)), set()
        ).add(tuple(str(value) for value in raw_sequence))
    for key, observed_sets in sorted(grouped.items(), key=lambda pair: repr(pair[0])):
        if set(observed_sets) != set(range(blocks)):
            failures.append(
                {
                    "group": list(key),
                    "reason": "missing_ab_block",
                    "observed": sorted(observed_sets),
                }
            )
            continue
        if any(len(sequences) != 1 for sequences in observed_sets.values()):
            failures.append(
                {"group": list(key), "reason": "inconsistent_variant_sequence"}
            )
            continue
        observed = {
            block: next(iter(sequences)) for block, sequences in observed_sets.items()
        }
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


def _native_ab_schedule_readiness(
    native_evidence: Mapping[str, object],
    variants: Sequence[Variant],
    backends: Sequence[str],
) -> dict[str, object]:
    """Authenticate fresh-process native A/B and reversed B/A blocks.

    Native helper outputs are collected outside this driver.  The manifest may
    attach schedule metadata to each artifact (needed for a frozen helper that
    predates these fields), while current helpers also emit the same fields in
    their JSON.  File, source, environment, workload, and input identities are
    still derived from the hash-bound artifact rather than trusted from the
    schedule declaration.
    """

    failures: list[dict[str, object]] = []
    if len(variants) != 2:
        return {
            "complete": False,
            "schedule": [],
            "failures": [{"reason": "exactly_two_variants_required"}],
            "policy": "native comparative claims require fresh A/B and B/A helper processes",
        }
    variant_names = (variants[0].name, variants[1].name)
    expected_sequences = {variant_names, tuple(reversed(variant_names))}
    entries: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, str], dict[int, list[dict[str, object]]]] = {}

    for artifact_index, artifact in enumerate(native_evidence.get("artifacts", ())):
        if not isinstance(artifact, Mapping):
            failures.append(
                {"artifact_index": artifact_index, "reason": "artifact_not_object"}
            )
            continue
        validation = artifact.get("validation")
        if (
            not isinstance(validation, Mapping)
            or validation.get("complete") is not True
        ):
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_artifact_not_validated",
                }
            )
            continue
        artifact_path = artifact.get("path")
        try:
            payload = _strict_json_loads(
                Path(str(artifact_path)).read_text(encoding="utf-8")
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJsonKeyError,
        ) as exc:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_artifact_unreadable",
                    "detail": str(exc),
                }
            )
            continue
        if not isinstance(payload, Mapping):
            failures.append(
                {"artifact_index": artifact_index, "reason": "artifact_root_not_object"}
            )
            continue
        schedule = artifact.get("schedule")
        schedule = schedule if isinstance(schedule, Mapping) else {}

        def scheduled_field(name: str) -> object:
            if name in schedule and name in payload and schedule[name] != payload[name]:
                failures.append(
                    {
                        "artifact_index": artifact_index,
                        "reason": "artifact_schedule_payload_mismatch",
                        "field": name,
                        "scheduled": _json_safe(schedule[name]),
                        "payload": _json_safe(payload[name]),
                    }
                )
            # Payload is the authenticated source of truth.  The manifest
            # schedule remains useful only for older schedule-only fixtures.
            return payload[name] if name in payload else schedule.get(name)

        variant = str(artifact.get("variant"))
        payload_variant = scheduled_field("variant")
        if payload_variant is not None and str(payload_variant) != variant:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "artifact_schedule_variant_mismatch",
                    "manifest_variant": variant,
                    "payload_variant": payload_variant,
                }
            )
        if variant not in variant_names:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "unknown_native_ab_variant",
                    "variant": variant,
                }
            )
        ab_block = scheduled_field("ab_block")
        if not isinstance(ab_block, int) or isinstance(ab_block, bool) or ab_block < 0:
            failures.append(
                {"artifact_index": artifact_index, "reason": "ab_block_required"}
            )
            continue
        raw_sequence = scheduled_field("variant_sequence")
        variant_sequence = (
            tuple(str(value) for value in raw_sequence)
            if isinstance(raw_sequence, (list, tuple))
            else ()
        )
        if variant_sequence not in expected_sequences:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "variant_sequence_must_be_ab_or_ba",
                    "observed": list(variant_sequence),
                    "expected": [list(value) for value in sorted(expected_sequences)],
                }
            )
        process_isolation = scheduled_field("process_isolation")
        if process_isolation != NATIVE_AB_PROCESS_ISOLATION:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "fresh_native_helper_process_required",
                    "observed": process_isolation,
                }
            )

        validation_lane = validation.get("evidence_lane")
        scheduled_lane = scheduled_field("evidence_lane")
        lane = validation_lane if validation_lane is not None else scheduled_lane
        strict_lane_contract = validation.get("lane_contract_active") is True
        if strict_lane_contract and lane not in NATIVE_EVIDENCE_LANE_SET:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_evidence_lane_required_for_schedule",
                    "observed": lane,
                }
            )
        if (
            validation_lane is not None
            and scheduled_lane is not None
            and validation_lane != scheduled_lane
        ):
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "artifact_schedule_lane_mismatch",
                    "validated": validation_lane,
                    "scheduled": scheduled_lane,
                }
            )
        raw_plan = scheduled_field("loop_plan")
        plan_semantic_sha = (
            raw_plan.get("semantic_sha256") if isinstance(raw_plan, Mapping) else None
        )
        if plan_semantic_sha is None and isinstance(raw_plan, Mapping):
            plan_semantic_sha = raw_plan.get("sha256")
        plan_file_sha = (
            raw_plan.get("file_sha256") if isinstance(raw_plan, Mapping) else None
        )
        if strict_lane_contract and (
            not isinstance(plan_semantic_sha, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", plan_semantic_sha) is None
            or not isinstance(plan_file_sha, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", plan_file_sha) is None
        ):
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_loop_plan_identity_required_for_schedule",
                }
            )

        backend = str(artifact.get("backend"))
        payload_input_policy = payload.get(
            "input_policy", payload.get("input_policy_name")
        )
        scheduled_field("input_policy")
        if "input_policy" not in schedule:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_schedule_input_policy_required",
                }
            )
        if schedule.get("input_policy") != payload_input_policy:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_schedule_input_policy_payload_mismatch",
                    "scheduled": _json_safe(schedule.get("input_policy")),
                    "payload": _json_safe(payload_input_policy),
                }
            )
        input_policy = payload_input_policy
        input_identity = payload.get("input_identity", payload.get("dataset_identity"))
        workload = payload.get("workload")
        if not isinstance(workload, Mapping) or not workload:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "structured_native_workload_required",
                }
            )
            workload = {}
        if input_policy not in INPUT_POLICIES:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_input_policy_required",
                }
            )
        if not isinstance(input_identity, Mapping) or not input_identity:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_input_identity_required",
                }
            )
            input_identity = {}

        provenance = validation.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        required_identities = ["benchmark_binary", "harness_source", "wheel"]
        if backend == "core":
            required_identities.append("harness_runner")
        else:
            required_identities.append("payload")
        identities: dict[str, object] = {}
        for name in required_identities:
            identity = provenance.get(name)
            identity_failures = _native_identity_failures(identity, name)
            if identity_failures:
                failures.append(
                    {
                        "artifact_index": artifact_index,
                        "reason": "native_ab_identity_incomplete",
                        "identity": name,
                        "details": identity_failures,
                    }
                )
            identities[name] = _json_safe(identity)
        environment_view = _native_environment_comparison_view(validation)
        if environment_view is None:
            failures.append(
                {
                    "artifact_index": artifact_index,
                    "reason": "native_ab_environment_identity_unavailable",
                }
            )
        product_identity = {
            "source_commit": validation.get("source_commit"),
            "source_tree_state": validation.get("source_tree_state"),
            "source_root": validation.get("source_root"),
        }
        harness_identity = {
            "source_commit": validation.get("native_harness_source_commit"),
            "source": validation.get("native_harness_source"),
            "source_blob": validation.get("native_harness_source_blob"),
            "runner": validation.get("native_harness_runner"),
            "runner_blob": validation.get("native_harness_runner_blob"),
        }
        workload_identity = json.dumps(_json_safe(workload), sort_keys=True)
        input_identity_json = json.dumps(_json_safe(input_identity), sort_keys=True)
        entry = {
            "artifact_index": artifact_index,
            "artifact": {
                "path": artifact_path,
                "size_bytes": artifact.get("size_bytes"),
                "sha256": artifact.get("sha256"),
            },
            "backend": backend,
            "ab_block": ab_block,
            "variant": variant,
            "variant_sequence": list(variant_sequence),
            "evidence_lane": lane,
            "loop_plan_semantic_sha256": plan_semantic_sha,
            "loop_plan_file_sha256": plan_file_sha,
            "process_isolation": process_isolation,
            "binary": identities.get("benchmark_binary"),
            "wheel": identities.get("wheel"),
            "payload": identities.get("payload"),
            "harness": harness_identity,
            "product": product_identity,
            "environment": _json_safe(environment_view),
            "process_affinity": _json_safe(validation.get("process_affinity")),
            "command_line": _json_safe(validation.get("command_line")),
            "command_line_comparison": _json_safe(
                _native_command_line_comparison_view(validation.get("command_line"))
            ),
            "workload": _json_safe(workload),
            "input_policy": input_policy,
            "input_identity": _json_safe(input_identity),
            "runner_pid": validation.get("runner_pid"),
            "process_id": validation.get("process_id"),
            "runner_invocation_id": validation.get("runner_invocation_id"),
        }
        entries.append(entry)
        group_key = (
            backend,
            workload_identity,
            str(input_policy),
            input_identity_json,
            str(lane),
            str(plan_semantic_sha),
            str(plan_file_sha),
        )
        grouped.setdefault(group_key, {}).setdefault(ab_block, []).append(entry)

    for backend in backends:
        if not any(key[0] == backend for key in grouped):
            failures.append(
                {"backend": backend, "reason": "no_native_ab_schedule_group"}
            )

    # Native evidence must independently prove that A and B are different
    # products.  Do not rely on a public-result matrix being present: a native-
    # only invocation with two labels pointing at the same commit/wheel/payload
    # is not an A/B comparison.
    product_identities: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        product = entry.get("product")
        product_commit = (
            product.get("source_commit") if isinstance(product, Mapping) else None
        )
        fingerprint = json.dumps(
            _json_safe(
                {
                    "source_commit": product_commit,
                    "wheel": _identity_content_fingerprint(entry.get("wheel")),
                    "payload": _identity_content_fingerprint(entry.get("payload")),
                }
            ),
            sort_keys=True,
        )
        product_identities.setdefault(
            (str(entry.get("backend")), str(entry.get("variant"))), set()
        ).add(fingerprint)
    for backend in backends:
        baseline_products = product_identities.get((backend, variant_names[0]), set())
        candidate_products = product_identities.get((backend, variant_names[1]), set())
        if len(baseline_products) > 1 or len(candidate_products) > 1:
            failures.append(
                {
                    "backend": backend,
                    "reason": "native_variant_product_identity_changed",
                }
            )
        if (
            len(baseline_products) == 1
            and len(candidate_products) == 1
            and baseline_products == candidate_products
        ):
            failures.append(
                {
                    "backend": backend,
                    "reason": "native_baseline_candidate_product_identities_must_differ",
                }
            )

    seen_artifact_paths: set[str] = set()
    seen_artifact_hashes: set[str] = set()
    seen_process_attestations: set[tuple[object, ...]] = set()
    for entry in entries:
        artifact_identity = entry.get("artifact")
        path = (
            str(artifact_identity.get("path"))
            if isinstance(artifact_identity, Mapping)
            else ""
        )
        digest = (
            str(artifact_identity.get("sha256"))
            if isinstance(artifact_identity, Mapping)
            else ""
        )
        if path in seen_artifact_paths or digest in seen_artifact_hashes:
            failures.append(
                {
                    "artifact": _json_safe(artifact_identity),
                    "reason": "native_artifact_reused_across_process_trials",
                }
            )
        seen_artifact_paths.add(path)
        seen_artifact_hashes.add(digest)
        if entry.get("evidence_lane") in NATIVE_EVIDENCE_LANE_SET:
            process_key = (
                entry.get("backend"),
                entry.get("runner_pid"),
                entry.get("process_id"),
                entry.get("runner_invocation_id"),
            )
            if any(value in (None, "") for value in process_key[1:]):
                failures.append(
                    {
                        "artifact": _json_safe(artifact_identity),
                        "reason": "native_process_attestation_missing",
                    }
                )
            elif process_key in seen_process_attestations:
                failures.append(
                    {
                        "artifact": _json_safe(artifact_identity),
                        "reason": "native_process_attestation_reused_across_trials",
                    }
                )
            seen_process_attestations.add(process_key)

    for key, blocks in sorted(grouped.items(), key=lambda item: repr(item[0])):
        group_label = {
            "backend": key[0],
            "workload": key[1],
            "input_policy": key[2],
            "input_identity": key[3],
            "evidence_lane": key[4],
            "loop_plan_sha256": key[5],
            "loop_plan_semantic_sha256": key[5],
            "loop_plan_file_sha256": key[6],
        }
        observed_sequences: set[tuple[str, ...]] = set()
        stable_by_variant: dict[str, str] = {}
        common_harnesses: set[str] = set()
        common_environments: set[str] = set()
        common_affinities: set[str] = set()
        common_command_lines: set[str] = set()
        for block, block_entries in sorted(blocks.items()):
            block_variants = [str(entry["variant"]) for entry in block_entries]
            block_sequences = {
                tuple(str(value) for value in entry["variant_sequence"])
                for entry in block_entries
            }
            if (
                sorted(block_variants) != sorted(variant_names)
                or len(block_entries) != 2
            ):
                failures.append(
                    {
                        "group": group_label,
                        "ab_block": block,
                        "reason": "native_ab_block_requires_each_variant_once",
                        "observed": block_variants,
                    }
                )
            if len(block_sequences) != 1:
                failures.append(
                    {
                        "group": group_label,
                        "ab_block": block,
                        "reason": "native_ab_block_sequence_disagreement",
                    }
                )
                continue
            sequence = next(iter(block_sequences))
            observed_sequences.add(sequence)
            for entry in block_entries:
                variant = str(entry["variant"])
                stable_fingerprint = json.dumps(
                    _json_safe(
                        {
                            "binary": entry["binary"],
                            "wheel": entry["wheel"],
                            "payload": entry["payload"],
                            "harness": entry["harness"],
                            "product": entry["product"],
                            "environment": entry["environment"],
                            "process_affinity": entry["process_affinity"],
                            "command_line_comparison": entry["command_line_comparison"],
                            "workload": entry["workload"],
                            "input_policy": entry["input_policy"],
                            "input_identity": entry["input_identity"],
                        }
                    ),
                    sort_keys=True,
                )
                previous = stable_by_variant.setdefault(variant, stable_fingerprint)
                if previous != stable_fingerprint:
                    failures.append(
                        {
                            "group": group_label,
                            "variant": variant,
                            "reason": "native_variant_identity_changed_between_blocks",
                        }
                    )
                harness = entry["harness"]
                harness_source = (
                    harness.get("source") if isinstance(harness, Mapping) else None
                )
                harness_blob = (
                    harness.get("source_blob") if isinstance(harness, Mapping) else None
                )
                harness_runner = (
                    harness.get("runner") if isinstance(harness, Mapping) else None
                )
                harness_runner_blob = (
                    harness.get("runner_blob") if isinstance(harness, Mapping) else None
                )
                common_harnesses.add(
                    json.dumps(
                        _json_safe(
                            {
                                "source_commit": (
                                    harness.get("source_commit")
                                    if isinstance(harness, Mapping)
                                    else None
                                ),
                                "source_identity": _identity_content_fingerprint(
                                    harness_source
                                ),
                                "relative_path": (
                                    harness_blob.get("relative_path")
                                    if isinstance(harness_blob, Mapping)
                                    else None
                                ),
                                "source_sha256": (
                                    harness_blob.get("source_sha256")
                                    if isinstance(harness_blob, Mapping)
                                    else None
                                ),
                                "current_git_blob": (
                                    harness_blob.get("current_git_blob")
                                    if isinstance(harness_blob, Mapping)
                                    else None
                                ),
                                "head_git_blob": (
                                    harness_blob.get("head_git_blob")
                                    if isinstance(harness_blob, Mapping)
                                    else None
                                ),
                                "runner_identity": _identity_content_fingerprint(
                                    harness_runner
                                ),
                                "runner_relative_path": (
                                    harness_runner_blob.get("relative_path")
                                    if isinstance(harness_runner_blob, Mapping)
                                    else None
                                ),
                                "runner_source_sha256": (
                                    harness_runner_blob.get("source_sha256")
                                    if isinstance(harness_runner_blob, Mapping)
                                    else None
                                ),
                                "runner_current_git_blob": (
                                    harness_runner_blob.get("current_git_blob")
                                    if isinstance(harness_runner_blob, Mapping)
                                    else None
                                ),
                                "runner_head_git_blob": (
                                    harness_runner_blob.get("head_git_blob")
                                    if isinstance(harness_runner_blob, Mapping)
                                    else None
                                ),
                            }
                        ),
                        sort_keys=True,
                    )
                )
                common_environments.add(
                    json.dumps(_json_safe(entry["environment"]), sort_keys=True)
                )
                common_affinities.add(
                    json.dumps(_json_safe(entry["process_affinity"]), sort_keys=True)
                )
                common_command_lines.add(
                    json.dumps(
                        _json_safe(entry["command_line_comparison"]),
                        sort_keys=True,
                    )
                )
        if observed_sequences != expected_sequences:
            failures.append(
                {
                    "group": group_label,
                    "reason": "both_native_ab_and_ba_blocks_required",
                    "observed": [list(value) for value in sorted(observed_sequences)],
                    "expected": [list(value) for value in sorted(expected_sequences)],
                }
            )
        if len(common_harnesses) != 1:
            failures.append(
                {
                    "group": group_label,
                    "reason": "common_native_harness_identity_required",
                }
            )
        if len(common_environments) != 1:
            failures.append(
                {"group": group_label, "reason": "native_environment_mismatch"}
            )
        if len(common_affinities) != 1:
            failures.append(
                {
                    "group": group_label,
                    "reason": "native_process_affinity_mismatch",
                }
            )
        if len(common_command_lines) != 1:
            failures.append(
                {
                    "group": group_label,
                    "reason": "native_command_line_mismatch",
                }
            )

    input_policy_coverage = {
        backend: sorted(
            {
                str(entry["input_policy"])
                for entry in entries
                if entry["backend"] == backend
                and entry["input_policy"] in INPUT_POLICIES
            }
        )
        for backend in backends
    }
    for backend in ("cuda", "rocm", "metal"):
        if backend in backends and set(input_policy_coverage.get(backend, ())) != set(
            INPUT_POLICIES
        ):
            failures.append(
                {
                    "backend": backend,
                    "reason": "both_native_input_policies_required",
                    "observed": input_policy_coverage.get(backend, []),
                    "expected": list(INPUT_POLICIES),
                }
            )
    return {
        "complete": not failures,
        "schedule": entries,
        "input_policy_coverage": input_policy_coverage,
        "failures": failures,
        "policy": (
            "each native comparison cell requires distinct fresh helper-process artifacts "
            "in baseline/candidate and candidate/baseline blocks, with exact binary, wheel, "
            "payload, harness, product, environment, workload, and input identities"
        ),
        "claim_scope": (
            "native arithmetic comparisons cover only the explicitly observed input policies; "
            "public end-to-end evidence remains responsible for both public source policies"
        ),
    }


def _coverage_readiness(
    args: argparse.Namespace, schedule: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if args.layer != "all":
        failures.append(
            {"reason": "all_cold_and_public_layers_required", "layer": args.layer}
        )
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
    if set(args.profiles) != required_profiles or len(args.profiles) != len(
        required_profiles
    ):
        failures.append(
            {
                "reason": "all_distributed_profiles_for_requested_backends_required",
                "profiles": list(args.profiles),
                "required_profiles": sorted(required_profiles),
            }
        )
    if set(args.surfaces) != set(SURFACES) or len(args.surfaces) != len(SURFACES):
        failures.append(
            {"reason": "all_public_surfaces_required", "surfaces": list(args.surfaces)}
        )
    if set(args.input_policies) != set(INPUT_POLICIES) or len(
        args.input_policies
    ) != len(INPUT_POLICIES):
        failures.append(
            {
                "reason": "both_input_policies_required",
                "input_policies": list(args.input_policies),
            }
        )
    if set(args.workloads) != set(RELEASE_WORKLOADS) or len(args.workloads) != len(
        RELEASE_WORKLOADS
    ):
        failures.append(
            {
                "reason": "release_workload_matrix_required",
                "workloads": list(args.workloads),
            }
        )
    if args.warmups < MIN_WARMUPS:
        failures.append(
            {"reason": "warmup_threshold_not_met", "observed": args.warmups}
        )
    if args.repetitions < MIN_REPETITIONS:
        failures.append(
            {"reason": "repetition_threshold_not_met", "observed": args.repetitions}
        )
    if args.bootstrap_resamples < 500:
        failures.append(
            {
                "reason": "bootstrap_threshold_not_met",
                "observed": args.bootstrap_resamples,
            }
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
                    list(key)
                    for key in sorted(expected_control_keys - observed_control_keys)
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
            profile = str(cell.get("profile"))
            surface = str(cell.get("surface"))
            seed = result.get("seed")
            bootstrap_resamples = result.get("bootstrap_resamples")
            target_ns = result.get("min_sample_ns")
            if not isinstance(seed, int) or isinstance(seed, bool):
                seed = None
            if not isinstance(bootstrap_resamples, int) or isinstance(
                bootstrap_resamples, bool
            ):
                bootstrap_resamples = None
            if seed is None or bootstrap_resamples is None:
                failures.append(
                    {
                        "result_index": result_index,
                        "cell_index": cell_index,
                        "reason": "public_distribution_binding_fields_required",
                    }
                )
            validation_failures = _distribution_consistency_failures(
                distribution,
                expected_repetitions=expected_repetitions,
                expected_target_ns=(
                    int(target_ns)
                    if isinstance(target_ns, int) and not isinstance(target_ns, bool)
                    else None
                ),
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=(
                    seed
                    ^ int.from_bytes(
                        hashlib.sha256(
                            f"{profile}/{surface}/{result.get('workload', {}).get('name') if isinstance(result.get('workload'), Mapping) else ''}".encode(
                                "utf-8"
                            )
                        ).digest()[:8],
                        "little",
                    )
                    if seed is not None
                    else None
                ),
                require_full=True,
            )
            failures.extend(
                {
                    "result_index": result_index,
                    "cell_index": cell_index,
                    "reason": reason,
                }
                for reason in validation_failures
            )
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
                        "minimum_observed_ns": distribution.get(
                            "sample_region_min_observed_ns"
                        ),
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
                seed = result.get("seed")
                bootstrap_resamples = result.get("bootstrap_resamples")
                target_ns = result.get("min_sample_ns")
                if not isinstance(seed, int) or isinstance(seed, bool):
                    seed = None
                if not isinstance(bootstrap_resamples, int) or isinstance(
                    bootstrap_resamples, bool
                ):
                    bootstrap_resamples = None
                control_surface = str(control.get("surface"))
                if seed is None or bootstrap_resamples is None:
                    failures.append(
                        {
                            "result_index": result_index,
                            "control_profile": str(profile),
                            "reason": "control_distribution_binding_fields_required",
                        }
                    )
                validation_failures = _distribution_consistency_failures(
                    distribution,
                    expected_repetitions=expected_repetitions,
                    expected_target_ns=(
                        int(target_ns)
                        if isinstance(target_ns, int)
                        and not isinstance(target_ns, bool)
                        else None
                    ),
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=(
                        seed
                        ^ int.from_bytes(
                            hashlib.sha256(
                                f"interleaved/{profile}/{control_surface}/{result.get('workload', {}).get('name') if isinstance(result.get('workload'), Mapping) else ''}".encode(
                                    "utf-8"
                                )
                            ).digest()[:8],
                            "little",
                        )
                        if seed is not None
                        else None
                    ),
                    require_full=True,
                )
                failures.extend(
                    {
                        "result_index": result_index,
                        "control_profile": str(profile),
                        "reason": reason,
                    }
                    for reason in validation_failures
                )
                if distribution.get("sample_region_target_met") is not True:
                    failures.append(
                        {
                            "result_index": result_index,
                            "control_profile": str(profile),
                            "reason": "control_sample_region_target_not_met",
                            "target_ns": distribution.get("sample_region_target_ns"),
                            "minimum_observed_ns": distribution.get(
                                "sample_region_min_observed_ns"
                            ),
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
    results: Sequence[Mapping[str, object]],
    expected_public_result_count: int | None = None,
    expected_schedule: Sequence[Mapping[str, object]] | None = None,
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
                {
                    "result_index": result_index,
                    "reason": "public_worker_not_pass",
                    "status": result.get("status"),
                }
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
                failures.append(
                    {
                        "result_index": result_index,
                        "cell_index": cell_index,
                        "reason": "cell_not_object",
                    }
                )
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
                failures.append(
                    {"result_index": result_index, "reason": "control_not_object"}
                )
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
    if expected_schedule is not None:
        expected_keys: list[tuple[object, ...]] = []
        for index, item in enumerate(expected_schedule):
            if item.get("kind") != "public":
                continue
            key = _public_schedule_key(item)
            if key is None:
                failures.append(
                    {
                        "schedule_index": index,
                        "reason": "public_schedule_key_invalid",
                    }
                )
                continue
            expected_keys.append(key)
        observed_keys: list[tuple[object, ...]] = []
        for index, result in enumerate(results):
            if result.get("kind") != "public":
                continue
            key = _public_schedule_key(result)
            if key is None:
                failures.append(
                    {
                        "result_index": index,
                        "reason": "public_result_schedule_key_invalid",
                    }
                )
                continue
            observed_keys.append(key)
        expected_counts: dict[tuple[object, ...], int] = {}
        observed_counts: dict[tuple[object, ...], int] = {}
        for key in expected_keys:
            expected_counts[key] = expected_counts.get(key, 0) + 1
        for key in observed_keys:
            observed_counts[key] = observed_counts.get(key, 0) + 1
        if any(count != 1 for count in expected_counts.values()):
            failures.append(
                {
                    "reason": "duplicate_public_schedule_key",
                    "keys": [
                        list(key)
                        for key, count in expected_counts.items()
                        if count != 1
                    ],
                }
            )
        if observed_counts != expected_counts:
            failures.append(
                {
                    "reason": "public_schedule_result_key_coverage_mismatch",
                    "missing": [
                        list(key)
                        for key, count in expected_counts.items()
                        if observed_counts.get(key, 0) < count
                    ],
                    "extra_or_duplicate": [
                        list(key)
                        for key, count in observed_counts.items()
                        if count > expected_counts.get(key, 0)
                    ],
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
        "exact_schedule_binding": expected_schedule is not None,
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
    native_order_sensitivity: Sequence[Mapping[str, object]] = (),
    require_cold_comparison: bool = False,
    require_native_comparison: bool = False,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index, summary in enumerate(order_sensitivity):
        if summary.get("status") != ORDER_EFFECT_CLEAN_STATUS:
            failures.append(
                {
                    "kind": "order_sensitivity",
                    "index": index,
                    "status": summary.get("status"),
                    "threshold_percent": 1.0,
                }
            )
    for index, summary in enumerate(interleaved_order_sensitivity):
        if summary.get("status") != ORDER_EFFECT_CLEAN_STATUS:
            failures.append(
                {
                    "kind": "interleaved_order_sensitivity",
                    "index": index,
                    "status": summary.get("status"),
                    "threshold_percent": 1.0,
                }
            )
    for index, summary in enumerate(native_order_sensitivity):
        if summary.get("status") != ORDER_EFFECT_CLEAN_STATUS:
            failure = {
                "kind": "native_order_sensitivity",
                "index": index,
                "status": summary.get("status"),
                "threshold_percent": NATIVE_ORDER_INVESTIGATE_PERCENT,
            }
            failures.append(failure)
    # perf13's order-rotated cold envelope is diagnostic only.  Canonical cold
    # regression classification is owned by cold_lifecycle.py, so these
    # summaries are deliberately absent from this release threshold gate.
    all_comparisons = [("public", comparison) for comparison in comparisons] + [
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
        minimum_comparison_samples = 2 if comparison_kind == "cold" else MIN_REPETITIONS
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
        interval = comparison.get("bootstrap_candidate_latency_delta_95_ci_percent")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in interval
            )
        ):
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": "independent_bootstrap_delta_ci_missing",
                }
            )
        delta = comparison.get("candidate_latency_delta_percent")
        if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": "candidate_latency_delta_missing_or_invalid",
                }
            )
            derived = _comparison_classification(float("nan"), None)
        else:
            derived = _comparison_classification(float(delta), interval)
        status = derived["review_status"]
        declared_status = comparison.get("review_status")
        if declared_status != status:
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": "bootstrap_regression_classification_mismatch",
                    "declared": declared_status,
                    "derived": status,
                }
            )
        if status in {
            "confirmed_regression_above_one_percent",
            "confirmed_regression_above_three_percent",
        }:
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": status,
                    "delta_percent": delta,
                    "threshold_percent": 1.0,
                }
            )
        elif status not in {
            "confirmed_regression_within_one_percent",
            "confirmed_improvement",
            "inconclusive_ci_crosses_zero",
        }:
            failures.append(
                {
                    "kind": f"{comparison_kind}_regression",
                    "index": index,
                    "status": status,
                    "reason": "bootstrap_regression_classification_missing_or_invalid",
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
            "order evidence is clean only when every simultaneous familywise upper "
            "absolute contrast bound is at most one percent, is contaminated when a "
            "lower bound exceeds one percent, and is otherwise inconclusive; candidate latency "
            "direction is confirmed only when the independent-bootstrap interval excludes "
            "zero, and only repeatable regressions above one/three percent escalate; "
            "public/native deltas require 30 raw observations per variant, while "
            "canonical cold-lifecycle regression evidence is owned by cold_lifecycle.py"
        ),
    }


def _public_git_provenance_failures(provenance: object, *, label: str) -> list[str]:
    """Validate trusted-Git and current-equals-HEAD source binding."""

    if not isinstance(provenance, Mapping) or provenance.get("status") != "trusted":
        return [f"{label}_trusted_git_provenance"]
    failures: list[str] = []
    git = provenance.get("git")
    trusted_paths = {
        str(path.resolve())
        for path in (Path("/usr/bin/git"), Path("/bin/git"))
        if path.is_file()
    }
    if not isinstance(git, Mapping):
        failures.append(f"{label}_trusted_git_identity")
    else:
        if git.get("path") not in trusted_paths:
            failures.append(f"{label}_trusted_git_path")
        executable = Path(str(git.get("path", "")))
        if (
            not executable.is_file()
            or not isinstance(git.get("sha256"), str)
            or _sha256(executable) != git.get("sha256")
        ):
            failures.append(f"{label}_trusted_git_sha256")
        removed = git.get("removed_environment")
        if not isinstance(removed, list) or any(
            not isinstance(name, str) or not name.startswith("GIT_") for name in removed
        ):
            failures.append(f"{label}_git_environment_scrub")
        if git.get("config_isolation") != {
            "system": os.devnull,
            "global": os.devnull,
            "nosystem": True,
        }:
            failures.append(f"{label}_git_config_isolation")
        for name in ("git_dir", "git_common_dir"):
            value = git.get(name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                failures.append(f"{label}_{name}")
    blob = provenance.get("source_blob")
    current = blob.get("current_git_blob") if isinstance(blob, Mapping) else None
    head = blob.get("head_git_blob") if isinstance(blob, Mapping) else None
    if (
        not isinstance(blob, Mapping)
        or blob.get("status") != "tracked_at_head"
        or not isinstance(current, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", current) is None
        or current != head
        or not isinstance(blob.get("source_sha256"), str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", str(blob.get("source_sha256"))) is None
    ):
        failures.append(f"{label}_source_blob_current_head_binding")
    return failures


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
            missing.extend(
                _public_git_provenance_failures(
                    provenance.get("source_git_provenance"), label="source"
                )
            )
            missing.extend(
                _public_git_provenance_failures(
                    provenance.get("benchmark_git_provenance"), label="benchmark"
                )
            )
            wheel_binding = provenance.get("wheel_runtime_binding", {})
            if not isinstance(wheel_binding, Mapping) or not wheel_binding.get(
                "complete"
            ):
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
                    if (
                        not isinstance(record, Mapping)
                        or record.get("status") != "pass"
                    ):
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
            "SHA-256 plus embedded RECORD/runtime identity, absolute trusted Git with all "
            "inherited GIT_* and global/system config redirection scrubbed, current source "
            "blobs equal to HEAD, loaded-module hashes, "
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
        identities = [item for item in collection if isinstance(item, Mapping)]
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
            return f"{roots[root]}/{normalized[len(prefix) :]}"
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
    exact, roots, interpreter_fingerprint = _native_path_tokens(validation, environment)
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


def _native_process_affinity_view(value: object) -> dict[str, object] | None:
    """Normalize backend affinity encodings for exact A/B identity checks."""

    if isinstance(value, str):
        if re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", value) is None:
            return None
        return {"linux_cpu_list": value}
    raw_cpus: object = value
    current_cpu: object = None
    if isinstance(value, Mapping):
        raw_cpus = value.get("cpus", value.get("allowed_cpus"))
        current_cpu = value.get("current_cpu")
    if not isinstance(raw_cpus, (list, tuple)) or not raw_cpus:
        return None
    if any(
        not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in raw_cpus
    ):
        return None
    cpus = list(raw_cpus)
    if len(set(cpus)) != len(cpus):
        return None
    result: dict[str, object] = {"cpus": cpus}
    if current_cpu is not None:
        if (
            not isinstance(current_cpu, int)
            or isinstance(current_cpu, bool)
            or current_cpu not in cpus
        ):
            return None
        result["current_cpu"] = current_cpu
    return result


def _native_command_line_view(value: object) -> list[str] | None:
    """Accept only an exact, nonempty argv vector without embedded NULs."""

    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in value
        )
    ):
        return None
    return list(value)


def _native_command_line_comparison_view(
    value: object,
) -> list[str] | None:
    """Normalize only authenticated per-variant/schedule argv fields."""

    command_line = _native_command_line_view(value)
    if command_line is None:
        return None
    normalized = ["<benchmark_binary>"]
    variant_fields = {
        "--payload": "<payload>",
        "--wheel": "<wheel>",
        "--source-root": "<source_root>",
        "--canonical-evidence": "<canonical_evidence>",
        "--csv": "<output>",
        "--json": "<output>",
        "--output": "<output>",
        "--metallib": "<metallib>",
        "--shader-source": "<shader_source>",
        "--source-commit": "<source_commit>",
        "--variant": "<variant>",
        "--ab-block": "<ab_block>",
        "--variant-sequence": "<variant_sequence>",
    }
    index = 1
    while index < len(command_line):
        argument = command_line[index]
        matched_equals = False
        for option, placeholder in variant_fields.items():
            prefix = option + "="
            if argument.startswith(prefix):
                normalized.append(prefix + placeholder)
                matched_equals = True
                break
        if matched_equals:
            index += 1
            continue
        normalized.append(argument)
        placeholder = variant_fields.get(argument)
        if placeholder is not None:
            index += 1
            if index >= len(command_line):
                return None
            normalized.append(placeholder)
        index += 1
    return normalized


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
    expected_boundary = (
        GPU_NATIVE_CLOCK_POWER_CAPTURE_BOUNDARY
        if backend in ("cuda", "rocm")
        else "before and after all timed benchmark regions"
    )
    if payload.get("clock_and_power_capture_point") != expected_boundary:
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
            elif backend == "rocm" and not _rocm_dynamic_telemetry_fields(device_state):
                failures.append(f"{device_key}_{phase}_dynamic_clock_or_power_required")
        if backend == "metal":
            cpu_power = phase_state.get("cpu_power_management")
            if not isinstance(cpu_power, Mapping):
                failures.append(f"cpu_power_management_{phase}_required")
            elif cpu_power.get("status") == "pass":
                if not isinstance(cpu_power.get("output"), str) or not cpu_power.get(
                    "output"
                ):
                    failures.append(
                        f"cpu_power_management_{phase}_observation_required"
                    )
            elif cpu_power.get("status") == "unavailable":
                if not isinstance(cpu_power.get("detail"), str) or not cpu_power.get(
                    "detail"
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


def _gpu_calibration_prepass_failures(
    backend: str,
    payload: Mapping[str, object],
    records: Sequence[object],
    repeats: object,
) -> list[str]:
    """Authenticate the discarded GPU calibration pass before accepting data."""

    prefix = f"{backend}_native_calibration_prepass"
    summary = payload.get("calibration_prepass")
    if not isinstance(summary, Mapping):
        return [f"{prefix}_required"]
    failures: list[str] = []
    if summary.get("performed") is not True:
        failures.append(f"{prefix}_performed_required")
    expected_order = tuple(BACKEND_PROFILES.get(backend, ()))
    raw_order = summary.get("profile_order")
    observed_order = (
        tuple(str(value) for value in raw_order)
        if isinstance(raw_order, (list, tuple))
        else None
    )
    if observed_order != expected_order:
        failures.append(f"{prefix}_profile_order_required")

    count_values: dict[str, int] = {}
    for field in (
        "records_discarded",
        "samples_discarded",
        "calibrated_key_count",
    ):
        value = summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            failures.append(f"{prefix}_{field}_required")
        else:
            count_values[field] = value

    if summary.get("uses_shared_calibration_cache") is not True:
        failures.append(f"{prefix}_shared_cache_required")
    if summary.get("included_in_profile_order_cycles") is not False:
        failures.append(f"{prefix}_must_be_excluded_from_recorded_cycles")
    if not isinstance(summary.get("included_payload_api"), bool):
        failures.append(f"{prefix}_payload_api_scope_required")

    if (
        count_values.get("records_discarded") is not None
        and count_values.get("samples_discarded") is not None
        and isinstance(repeats, int)
        and not isinstance(repeats, bool)
        and repeats > 0
        and count_values["samples_discarded"]
        != count_values["records_discarded"] * repeats
    ):
        failures.append(f"{prefix}_sample_count_mismatch")

    # The helper cache key is lane/profile/operation/metric.  Optional payload
    # admission/decomposition aliases can be appended to an artifact without
    # creating a new calibrated timing cell, so only required timing records
    # form the lower-bound coverage check.
    observed_keys = {
        (
            str(record.get("profile")),
            str(record.get("operation")),
            str(record.get("metric")),
            str(
                record.get(
                    "evidence_lane",
                    record.get(
                        "timing_mode", payload.get("execution_mode", "unspecified")
                    ),
                )
            ),
        )
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("operation", ""))
        not in NATIVE_SUPPLEMENTAL_OPERATION_ALIASES
    }
    calibrated_key_count = count_values.get("calibrated_key_count")
    if calibrated_key_count is not None and calibrated_key_count < len(observed_keys):
        failures.append(f"{prefix}_key_coverage_mismatch")

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


def _public_input_binding_failures(
    result: Mapping[str, object],
    cell: Mapping[str, object],
    *,
    control: bool = False,
) -> list[dict[str, object]]:
    """Check worker input metadata against a parent-recomputed recipe."""

    workload = result.get("workload")
    profile = cell.get("profile")
    if not isinstance(workload, Mapping) or not isinstance(profile, str):
        return [{"reason": "public_input_binding_fields_missing"}]
    seed = result.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        return [{"reason": "public_input_binding_seed_missing"}]
    job = {
        "input_policy": result.get("input_policy"),
        "seed": seed,
    }
    try:
        expected = _expected_input_binding(job, precision=profile, workload=workload)
    except (KeyError, TypeError, ValueError):
        return [{"reason": "public_input_binding_workload_invalid"}]
    failures: list[dict[str, object]] = []
    observed_binding = cell.get("input_binding")
    if observed_binding != expected:
        failures.append(
            {
                "reason": "public_input_binding_mismatch",
                "control": control,
                "profile": profile,
                "expected": expected,
                "observed": _json_safe(observed_binding),
            }
        )
    observed_identity = cell.get("input_identity")
    expected_shape = _expected_input_identity_shape(expected)
    if not isinstance(observed_identity, Mapping):
        failures.append(
            {
                "reason": "public_input_identity_missing",
                "control": control,
                "profile": profile,
            }
        )
    else:
        for field, expected_value in expected_shape.items():
            if observed_identity.get(field) != expected_value:
                failures.append(
                    {
                        "reason": "public_input_identity_shape_mismatch",
                        "control": control,
                        "profile": profile,
                        "field": field,
                        "expected": expected_value,
                        "observed": observed_identity.get(field),
                    }
                )
    return failures


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
                tuple(
                    result
                    for result in results
                    if str(result.get("backend")) == backend
                ),
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
            for variant_name, commit in report.get(
                "variant_source_commits", {}
            ).items():
                commits_by_variant.setdefault(str(variant_name), set()).add(str(commit))
            for variant_name, wheel_hashes in report.get(
                "variant_wheel_hashes", {}
            ).items():
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
    if (
        len(commits) == 2
        and len(set(commits.values())) == 1
        and len(set().union(*wheel_hashes.values())) <= 1
    ):
        failures.append(
            {
                "reason": "baseline_and_candidate_identity_are_identical",
                "source_commits": commits,
                "wheel_hashes": {
                    name: sorted(values) for name, values in wheel_hashes.items()
                },
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
            baseline_affinity.get("cpus")
            if isinstance(baseline_affinity, Mapping)
            else None
        )
        candidate_cpus = (
            candidate_affinity.get("cpus")
            if isinstance(candidate_affinity, Mapping)
            else None
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
                clock_state.get("after", {}) if isinstance(clock_state, Mapping) else {}
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
            failures.extend(_public_input_binding_failures(result, cell))
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
        for control in result.get("interleaved_controls", ()):
            if not isinstance(control, Mapping) or control.get("status") != "pass":
                continue
            profiles = control.get("profiles")
            if not isinstance(profiles, Mapping):
                continue
            for profile, profile_result in profiles.items():
                if not isinstance(profile_result, Mapping):
                    continue
                control_cell = {
                    "profile": str(profile),
                    "input_binding": profile_result.get("input_binding"),
                    "input_identity": profile_result.get("input_identity"),
                }
                failures.extend(
                    _public_input_binding_failures(result, control_cell, control=True)
                )
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
        manifest = _strict_json_loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        return {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "status": "invalid",
            "evidence_integrity_status": "invalid",
            "evidence_integrity_valid": False,
            "performance_claim_ready": False,
            "arithmetic_claims_valid": False,
            "claim_failures": [],
            "failures": [f"invalid_json:{exc}"],
            "manifest": None,
        }
    failures: list[str] = []
    claim_failures: list[str] = []
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
            failures.append(
                f"artifact_{index}_missing_variant_path_hash_backend_or_kind"
            )
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
            evidence_root=manifest_path.parent,
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
                artifact_validation["evidence_integrity_status"] = "invalid"
                artifact_validation["evidence_integrity_valid"] = False
                artifact_validation["performance_claim_ready"] = False
                artifact_validation.setdefault("failures", []).extend(
                    f"metal_v1:{reason}" for reason in metal_validation["failures"]
                )
        for reason in artifact_validation.get("failures", ()):  # type: ignore[union-attr]
            failures.append(f"artifact_{index}_native_validation:{reason}")
        for reason in artifact_validation.get("claim_failures", ()):  # type: ignore[union-attr]
            claim_failures.append(f"artifact_{index}_native_claim:{reason}")
        if artifact_validation.get("performance_claim_ready") is not True:
            claim_failures.append(f"artifact_{index}_performance_claim_not_ready")
        normalized_artifacts.append(
            {
                "backend": str(backend),
                "kind": str(kind),
                "variant": str(variant),
                "source_commit": artifact_validation.get("source_commit"),
                "path": str(resolved_artifact),
                "sha256": str(artifact_hash),
                "size_bytes": resolved_artifact.stat().st_size,
                "schedule": _json_safe(
                    item.get(
                        "schedule",
                        {
                            key: item.get(key)
                            for key in (
                                "ab_block",
                                "variant_sequence",
                                "process_isolation",
                            )
                            if item.get(key) is not None
                        },
                    )
                ),
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
        if (
            not isinstance(source_commit, str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None
        ):
            failures.append("validated_manifest_requires_full_source_commit")
        valid = not failures
    source_commits_by_variant: dict[str, str] = {}
    source_commit = manifest.get("source_commit")
    if (
        isinstance(source_commit, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is not None
    ):
        source_commits_by_variant = {
            str(artifact["variant"]): source_commit
            for artifact in normalized_artifacts
            if artifact.get("variant")
        }
    performance_claim_ready = bool(
        valid
        and status == "validated"
        and arithmetic_valid
        and normalized_artifacts
        and all(
            isinstance(artifact.get("validation"), Mapping)
            and artifact["validation"].get("performance_claim_ready") is True
            for artifact in normalized_artifacts
        )
    )
    return {
        "path": str(manifest_path),
        "sha256": manifest_hash,
        "status": status if valid else "invalid",
        "evidence_integrity_status": "valid" if valid else "invalid",
        "evidence_integrity_valid": valid,
        "performance_claim_ready": performance_claim_ready,
        "arithmetic_claims_valid": performance_claim_ready,
        "valid": valid,
        "failures": failures,
        "claim_failures": claim_failures,
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
    claim_failures: list[str] = []
    for name, path in specs:
        try:
            evidence = _load_native_evidence(path)
        except OSError as exc:
            evidence = {
                "path": path,
                "status": "invalid",
                "evidence_integrity_status": "invalid",
                "evidence_integrity_valid": False,
                "performance_claim_ready": False,
                "arithmetic_claims_valid": False,
                "valid": False,
                "claim_failures": [],
                "failures": [f"manifest_read_error:{exc}"],
                "artifacts": [],
                "manifest": None,
            }
        loaded.append((name, evidence))
        failures.extend(
            f"{name or 'shared'}:{failure}" for failure in evidence.get("failures", ())
        )
        claim_failures.extend(
            f"{name or 'shared'}:{failure}"
            for failure in evidence.get("claim_failures", ())
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
        source_commit = (
            manifest.get("source_commit") if isinstance(manifest, Mapping) else None
        )
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
    valid = (
        bool(loaded)
        and not failures
        and all(evidence.get("valid") is True for _, evidence in loaded)
    )
    arithmetic_claims_valid = valid and all(
        evidence.get("arithmetic_claims_valid") is True for _, evidence in loaded
    )
    statuses = {str(evidence.get("status")) for _, evidence in loaded}
    merged_status = (
        "validated"
        if valid and statuses == {"validated"}
        else "not_collected"
        if valid and statuses == {"not_collected"}
        else "invalid"
    )
    return {
        "path": [str(evidence.get("path")) for _, evidence in loaded],
        "sha256": [str(evidence.get("sha256")) for _, evidence in loaded],
        "status": merged_status,
        "evidence_integrity_status": "valid" if valid else "invalid",
        "evidence_integrity_valid": valid,
        "performance_claim_ready": arithmetic_claims_valid,
        "valid": valid,
        "arithmetic_claims_valid": arithmetic_claims_valid,
        "failures": failures,
        "claim_failures": claim_failures,
        "artifacts": artifacts,
        "source_commits_by_variant": source_commits_by_variant,
        "manifest": {
            str(name or "shared"): evidence.get("manifest") for name, evidence in loaded
        },
    }


def _metal_inline_lifecycle_provenance_failures(
    lifecycle: object, artifact: Mapping[str, object]
) -> list[str]:
    """Authenticate the direct Metal lifecycle embedded in a timing artifact.

    Metal's supplemental helper owns this lifecycle rather than a separate C
    consumer process.  It must nevertheless carry the same distinct product
    and common-harness identities as the external CUDA/ROCm lifecycle record.
    """

    failures: list[str] = []
    if not isinstance(lifecycle, Mapping):
        return ["canonical_inline_lifecycle_must_be_object"]

    product_commit = artifact.get(
        "product_source_commit", artifact.get("source_commit")
    )
    if lifecycle.get("source_commit") != artifact.get("source_commit"):
        failures.append("canonical_inline_source_commit_mismatch")
    if lifecycle.get("product_source_commit") != product_commit:
        failures.append("canonical_inline_product_source_commit_mismatch")
    for field in ("source_tree_state", "product_source_tree_state"):
        observed = lifecycle.get(field)
        expected = artifact.get(field)
        if not isinstance(observed, Mapping) or observed.get("status") != "clean":
            failures.append(f"canonical_inline_{field}_clean_required")
        if isinstance(expected, Mapping) and observed != expected:
            failures.append(f"canonical_inline_{field}_mismatch")
    for field in ("source_root", "product_source_root"):
        if lifecycle.get(field) != artifact.get(field):
            failures.append(f"canonical_inline_{field}_mismatch")

    product_binding = lifecycle.get("product_source_binding")
    expected_product_binding = artifact.get("product_source_binding")
    if not isinstance(product_binding, Mapping):
        failures.append("canonical_inline_product_source_binding_required")
    elif (
        isinstance(expected_product_binding, Mapping)
        and product_binding != expected_product_binding
    ):
        failures.append("canonical_inline_product_source_binding_mismatch")

    harness_commit = lifecycle.get("harness_source_commit")
    expected_harness_commit = artifact.get("harness_source_commit")
    if (
        not isinstance(harness_commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", harness_commit) is None
    ):
        failures.append("canonical_inline_harness_source_commit_required")
    elif harness_commit != expected_harness_commit:
        failures.append("canonical_inline_harness_source_commit_mismatch")
    for field in ("harness_source_tree_state",):
        observed = lifecycle.get(field)
        expected = artifact.get(field)
        if not isinstance(observed, Mapping) or observed.get("status") != "clean":
            failures.append(f"canonical_inline_{field}_clean_required")
        if isinstance(expected, Mapping) and observed != expected:
            failures.append(f"canonical_inline_{field}_mismatch")
    if lifecycle.get("harness_source_root") != artifact.get("harness_source_root"):
        failures.append("canonical_inline_harness_source_root_mismatch")
    for field in ("harness_source_binding", "harness_source_blob"):
        observed = lifecycle.get(field)
        expected = artifact.get(field)
        if not isinstance(observed, Mapping):
            failures.append(f"canonical_inline_{field}_required")
        elif isinstance(expected, Mapping) and observed != expected:
            failures.append(f"canonical_inline_{field}_mismatch")

    lifecycle_provenance = lifecycle.get("provenance")
    artifact_provenance = artifact.get("provenance")
    if not isinstance(lifecycle_provenance, Mapping):
        failures.append("canonical_inline_provenance_required")
        lifecycle_provenance = {}
    if not isinstance(artifact_provenance, Mapping):
        failures.append("canonical_inline_artifact_provenance_required")
        artifact_provenance = {}
    for name in ("payload", "wheel", "harness_source"):
        failures.extend(
            _native_identity_failures(
                lifecycle_provenance.get(name), f"canonical_inline_{name}"
            )
        )
        observed = lifecycle_provenance.get(name)
        expected = artifact_provenance.get(name)
        if (
            isinstance(observed, Mapping)
            and isinstance(expected, Mapping)
            and observed != expected
        ):
            failures.append(f"canonical_inline_{name}_identity_mismatch")

    wheel_member = lifecycle.get("wheel_member")
    if wheel_member != "gafime/_metal/libgafime_metal_v1.dylib":
        failures.append("canonical_inline_wheel_member_mismatch")
    payload_identity = lifecycle_provenance.get("payload")
    wheel_identity = lifecycle_provenance.get("wheel")
    member_digest = lifecycle.get("wheel_member_sha256")
    if (
        not isinstance(member_digest, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", member_digest) is None
    ):
        failures.append("canonical_inline_wheel_member_sha256_required")
    elif (
        isinstance(payload_identity, Mapping)
        and member_digest.lower() != str(payload_identity.get("sha256", "")).lower()
    ):
        failures.append("canonical_inline_wheel_member_payload_sha256_mismatch")
    if isinstance(wheel_identity, Mapping) and wheel_identity.get("path"):
        try:
            with zipfile.ZipFile(
                Path(str(wheel_identity["path"])).expanduser()
            ) as archive:
                embedded_digest = hashlib.sha256(
                    archive.read("gafime/_metal/libgafime_metal_v1.dylib")
                ).hexdigest()
            if embedded_digest != str(member_digest).lower():
                failures.append("canonical_inline_wheel_member_sha256_mismatch")
        except (KeyError, OSError, zipfile.BadZipFile):
            failures.append("canonical_inline_wheel_member_unreadable")
    return failures


def _metal_input_policy_failures(
    input_policy: object, input_identity: object
) -> list[str]:
    """Require Metal policy provenance to distinguish source from execution dtype."""

    failures: list[str] = []
    if input_policy not in INPUT_POLICIES:
        failures.append("metal_input_policy_required")
        return failures
    if not isinstance(input_identity, Mapping):
        return ["metal_input_identity_required"]
    expected = (
        (
            "float64",
            "deterministic_integer_modulus.common_f64.v1",
        )
        if input_policy == "common-f64"
        else (
            "float32",
            "deterministic_integer_modulus.native_fp32.v1",
        )
    )
    if input_identity.get("algorithm") != "gafime.metal.native_timing.dataset.v2":
        failures.append("metal_input_identity_algorithm_mismatch")
    if input_identity.get("input_policy") != input_policy:
        failures.append("metal_input_identity_policy_mismatch")
    if input_identity.get("source_dtype") != expected[0]:
        failures.append("metal_input_source_dtype_mismatch")
    if input_identity.get("generator") != expected[1]:
        failures.append("metal_input_generator_mismatch")
    for field in ("matrix_dtype", "target_dtype"):
        if input_identity.get(field) != expected[0]:
            failures.append(f"metal_input_{field}_mismatch")
    for field in (
        "execution_dtype",
        "execution_matrix_dtype",
        "execution_target_dtype",
    ):
        if input_identity.get(field) != "float32":
            failures.append(f"metal_input_{field}_must_be_float32")
    if input_identity.get("layout") != "row_major":
        failures.append("metal_input_layout_mismatch")
    for field in (
        "matrix_sha256",
        "target_sha256",
        "execution_matrix_sha256",
        "execution_target_sha256",
    ):
        value = input_identity.get(field)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
        ):
            failures.append(f"metal_input_{field}_sha256_required")
    if input_policy == "common-f64":
        if input_identity.get("matrix_sha256") == input_identity.get(
            "execution_matrix_sha256"
        ):
            failures.append(
                "metal_common_f64_source_and_execution_identity_must_differ"
            )
    elif input_identity.get("matrix_sha256") != input_identity.get(
        "execution_matrix_sha256"
    ):
        failures.append("metal_native_source_and_execution_identity_must_match")
    return failures


class _DuplicateJsonKeyError(ValueError):
    """Raised when a machine-readable evidence object repeats a JSON key."""


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _strict_json_loads(source: str | bytes | bytearray) -> object:
    """Parse evidence JSON without duplicate keys or non-finite numbers."""

    def reject_nonfinite(value: str) -> None:
        raise json.JSONDecodeError(
            f"non-finite JSON constant {value!r} is forbidden", value, 0
        )

    return json.loads(
        source,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=reject_nonfinite,
    )


def _metal_timing_record_failures(
    record: Mapping[str, object],
    *,
    prefix: str,
    repeats: object,
    expected_target_us: object,
) -> list[str]:
    """Validate one Metal record's fixed-loop and round-trip timing contract."""

    failures: list[str] = []
    samples = record.get("samples_us")
    raw_samples = record.get("raw_samples_us")
    loop_count = record.get("loop_count_per_sample")
    loop_counts = record.get("loop_counts_per_sample")
    valid_repeats = (
        isinstance(repeats, int) and not isinstance(repeats, bool) and repeats > 0
    )
    valid_loop_count = (
        isinstance(loop_count, int)
        and not isinstance(loop_count, bool)
        and loop_count > 0
    )
    valid_samples = (
        isinstance(samples, list)
        and valid_repeats
        and len(samples) == repeats
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in samples
        )
    )
    valid_raw_samples = (
        isinstance(raw_samples, list)
        and valid_repeats
        and len(raw_samples) == repeats
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in raw_samples
        )
    )
    valid_loop_counts = (
        isinstance(loop_counts, list)
        and valid_repeats
        and len(loop_counts) == repeats
        and valid_loop_count
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == loop_count
            for value in loop_counts
        )
    )
    if not valid_samples:
        failures.append(f"{prefix}_normalized_samples_invalid")
    if not valid_raw_samples:
        failures.append(f"{prefix}_raw_samples_invalid")
    if not valid_loop_counts:
        failures.append(f"{prefix}_fixed_loop_count_required")
    if valid_samples and valid_raw_samples and valid_loop_count:
        assert isinstance(samples, list)
        assert isinstance(raw_samples, list)
        assert isinstance(loop_count, int)
        if any(
            abs(float(normalized) - (float(raw) / loop_count))
            > max(1.0e-9, 4.0 * math.ulp(float(raw) / loop_count))
            for normalized, raw in zip(samples, raw_samples)
        ):
            failures.append(f"{prefix}_duration_normalization_mismatch")

    target_us = record.get("sample_region_target_us")
    observed_minimum = record.get("sample_region_min_observed_us")
    valid_expected_target = (
        isinstance(expected_target_us, (int, float))
        and not isinstance(expected_target_us, bool)
        and math.isfinite(float(expected_target_us))
        and float(expected_target_us) >= GPU_NATIVE_MIN_SAMPLE_REGION_US
    )
    valid_record_target = (
        isinstance(target_us, (int, float))
        and not isinstance(target_us, bool)
        and math.isfinite(float(target_us))
        and valid_expected_target
        and float(target_us) == float(expected_target_us)
    )
    raw_minimum = (
        min(float(value) for value in raw_samples)
        if valid_raw_samples and isinstance(raw_samples, list)
        else None
    )
    if (
        not valid_record_target
        or record.get("sample_region_target_met") is not True
        or raw_minimum is None
        or raw_minimum < float(expected_target_us)
        or not isinstance(observed_minimum, (int, float))
        or isinstance(observed_minimum, bool)
        or not math.isfinite(float(observed_minimum))
        or not math.isclose(
            float(observed_minimum), raw_minimum, rel_tol=1.0e-12, abs_tol=1.0e-9
        )
    ):
        failures.append(f"{prefix}_sample_region_gate_invalid")

    host_samples = record.get("host_synchronized_samples_us")
    raw_host_samples = record.get("raw_host_synchronized_samples_us")
    host_lane_present = bool(host_samples) or bool(raw_host_samples)
    if host_lane_present:
        valid_host_samples = (
            isinstance(host_samples, list)
            and valid_repeats
            and len(host_samples) == repeats
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in host_samples
            )
        )
        valid_raw_host_samples = (
            isinstance(raw_host_samples, list)
            and valid_repeats
            and len(raw_host_samples) == repeats
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in raw_host_samples
            )
        )
        if not valid_host_samples or not valid_raw_host_samples:
            failures.append(f"{prefix}_host_timing_samples_invalid")
        elif valid_loop_count:
            assert isinstance(host_samples, list)
            assert isinstance(raw_host_samples, list)
            assert isinstance(loop_count, int)
            if any(
                abs(float(normalized) - (float(raw) / loop_count))
                > max(1.0e-9, 4.0 * math.ulp(float(raw) / loop_count))
                for normalized, raw in zip(host_samples, raw_host_samples)
            ):
                failures.append(f"{prefix}_host_duration_normalization_mismatch")

    gpu_samples = record.get("gpu_timestamp_samples_us")
    raw_gpu_samples = record.get("raw_gpu_timestamp_samples_us")
    gpu_lane_present = gpu_samples is not None or raw_gpu_samples is not None
    if gpu_lane_present:
        valid_gpu_samples = (
            isinstance(gpu_samples, list)
            and valid_repeats
            and len(gpu_samples) == repeats
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in gpu_samples
            )
        )
        valid_raw_gpu_samples = (
            isinstance(raw_gpu_samples, list)
            and valid_repeats
            and len(raw_gpu_samples) == repeats
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in raw_gpu_samples
            )
        )
        if not valid_gpu_samples or not valid_raw_gpu_samples:
            failures.append(f"{prefix}_gpu_timestamp_samples_invalid")
        elif valid_loop_count:
            assert isinstance(gpu_samples, list)
            assert isinstance(raw_gpu_samples, list)
            assert isinstance(loop_count, int)
            for normalized, raw in zip(gpu_samples, raw_gpu_samples):
                raw_value = float(raw)
                normalized_value = float(normalized)
                if raw_value == 0.0:
                    if normalized_value != 1.0e-6:
                        failures.append(
                            f"{prefix}_gpu_timestamp_missing_sample_sentinel_invalid"
                        )
                    continue
                expected = raw_value / loop_count
                if abs(normalized_value - expected) > max(
                    1.0e-9, 4.0 * math.ulp(expected)
                ):
                    failures.append(
                        f"{prefix}_gpu_timestamp_duration_normalization_mismatch"
                    )
                    break
            declared_valid = record.get("gpu_timestamp_valid_samples")
            observed_valid = sum(float(value) > 0.0 for value in raw_gpu_samples)
            if declared_valid != observed_valid:
                failures.append(f"{prefix}_gpu_timestamp_valid_count_mismatch")
    return failures


def _validate_metal_native_timing_artifact(
    path: Path, *, manifest_source_commit: object
) -> dict[str, object]:
    """Validate supplemental Metal events plus exact ABI 1.1 payload timing."""

    failures: list[str] = []
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
    ) as exc:
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
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None
    ):
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
    sample_region_target_us = payload.get("sample_region_target_us")
    if (
        not isinstance(sample_region_target_us, (int, float))
        or isinstance(sample_region_target_us, bool)
        or not math.isfinite(float(sample_region_target_us))
        or float(sample_region_target_us) < GPU_NATIVE_MIN_SAMPLE_REGION_US
    ):
        failures.append("metal_sample_region_target_required")
    if payload.get("gpu_timing_supported") is not True:
        failures.append("complete_gpu_timestamp_support_required")
    failures.extend(
        _metal_input_policy_failures(
            payload.get("input_policy", payload.get("input_policy_name")),
            payload.get("input_identity", payload.get("dataset_identity")),
        )
    )

    provenance = payload.get("provenance")
    required_provenance = {
        "benchmark_source",
        "shader_source",
        "benchmark_binary",
        "metallib",
        "payload",
        "wheel",
    }
    if not isinstance(provenance, Mapping) or not required_provenance <= set(
        provenance
    ):
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
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
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
        if (
            not isinstance(samples, list)
            or not isinstance(repeats, int)
            or len(samples) != repeats
        ):
            failures.append(f"record_{record_index}_raw_sample_count_mismatch")
        elif any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in samples
        ):
            failures.append(f"record_{record_index}_invalid_raw_sample")
        failures.extend(
            _metal_timing_record_failures(
                record,
                prefix=f"record_{record_index}",
                repeats=repeats,
                expected_target_us=sample_region_target_us,
            )
        )
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
            gpu_samples = record.get("gpu_timestamp_samples_us")
            raw_gpu_samples = record.get("raw_gpu_timestamp_samples_us")
            if (
                not isinstance(gpu_samples, list)
                or len(gpu_samples) != repeats
                or not isinstance(raw_gpu_samples, list)
                or len(raw_gpu_samples) != repeats
            ):
                failures.append(
                    f"record_{record_index}_gpu_timestamp_diagnostic_arrays_required"
                )
            host_samples = record.get("host_synchronized_samples_us")
            if not isinstance(host_samples, list) or len(host_samples) != repeats:
                failures.append(
                    f"record_{record_index}_host_sync_sample_count_mismatch"
                )
    if observed_device_records != expected_device_records:
        failures.append("complete_metric_and_ranking_device_record_set_required")
    if observed_device_record_count != len(expected_device_records):
        failures.append("duplicate_or_extra_metric_device_record")
    if not expected_host_operations <= observed_operations:
        failures.append("complete_host_decomposition_record_set_required")
    boundaries = payload.get("decomposition_boundaries")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("candidate_materialization")
        != "fused into each metric kernel"
    ):
        failures.append("candidate_materialization_boundary_required")
    ingest_boundary = (
        str(boundaries.get("ingest_conversion", "")).lower()
        if isinstance(boundaries, Mapping)
        else ""
    )
    input_policy = payload.get("input_policy", payload.get("input_policy_name"))
    if input_policy == "common-f64" and "convert" not in ingest_boundary:
        failures.append("metal_common_f64_ingest_conversion_boundary_required")
    if input_policy == "native" and not (
        "native" in ingest_boundary and "not present" in ingest_boundary
    ):
        failures.append("metal_native_ingest_conversion_must_be_absent")

    # The standalone shader lane above is useful for command-buffer GPU event
    # timestamps, but it is supplemental evidence only. The claim is accepted
    # only when the same artifact also contains a complete lifecycle through
    # the exact wheel-installed dylib's canonical ABI 1.1 symbols.
    if payload.get("execution_mode") != "supplemental_internal_kernel":
        failures.append("supplemental_execution_mode_required")
    lifecycle = payload.get("canonical_payload_lifecycle")
    canonical_permutation_required = False
    if not isinstance(lifecycle, Mapping):
        failures.append("canonical_payload_lifecycle_required")
    else:
        if lifecycle.get("status") != "validated":
            failures.append("canonical_payload_lifecycle_not_validated")
        if lifecycle.get("schema") != "gafime.native-decomposition.v1":
            failures.append("canonical_payload_lifecycle_schema_mismatch")
        if lifecycle.get("execution_layer") != "installed_payload_dylib":
            failures.append("installed_payload_dylib_execution_required")
        if lifecycle.get("abi") != "1.1":
            failures.append("canonical_abi1_1_required")
        if lifecycle.get("route_count") != 1:
            failures.append("metal_fp32_route_count_must_be_one")
        if lifecycle.get("mixed_route_rejected") is not True:
            failures.append("metal_mixed_route_rejection_required")
        if lifecycle.get("fp64_route_rejected") is not True:
            failures.append("metal_fp64_route_rejection_required")
        if lifecycle.get("profile_mask") != 0x1:
            failures.append("metal_fp32_profile_mask_required")
        if lifecycle.get("storage_dtype_mask") != 0x1:
            failures.append("metal_fp32_storage_dtype_mask_required")
        if lifecycle.get("result_dtype_mask") != 0x1:
            failures.append("metal_fp32_result_dtype_mask_required")
        for status_field in (
            "route_query_status",
            "route_fill_status",
            "matrix_alloc_status",
            "matrix_upload_status",
            "execute_status",
            "matrix_free_status",
        ):
            if lifecycle.get(status_field) != 0:
                failures.append(f"metal_{status_field}_must_be_ok")
        lifecycle_surface = lifecycle.get("abi_surface")
        raw_symbols = lifecycle.get("symbols")
        if not isinstance(raw_symbols, list) or len(raw_symbols) != len(
            set(map(str, raw_symbols))
        ):
            failures.append("canonical_payload_symbols_must_be_unique")
        required_symbols = CANONICAL_ABI_SYMBOLS_BY_SURFACE.get(str(lifecycle_surface))
        if (
            lifecycle_surface not in CANONICAL_ABI_SURFACES
            or required_symbols is None
            or not isinstance(raw_symbols, list)
            or set(map(str, raw_symbols)) != required_symbols
        ):
            failures.append("complete_canonical_payload_symbol_set_required")
        if lifecycle.get("records_field") != "canonical_payload_records":
            failures.append("canonical_payload_record_binding_required")
        operation_status = lifecycle.get("operation_status")
        if not isinstance(operation_status, Mapping):
            failures.append("metal_canonical_operation_status_required")
            operation_status = {}

        def require_operation_ok(
            name: str, *, positive_field: str | None = None
        ) -> None:
            observed = operation_status.get(name)
            if (
                not isinstance(observed, Mapping)
                or observed.get("status") != "pass"
                or observed.get("abi_status") != 0
            ):
                failures.append(f"metal_{name}_operation_status_must_be_ok")
                return
            if positive_field is not None:
                value = observed.get(positive_field)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    failures.append(f"metal_{name}_{positive_field}_must_be_positive")

        require_operation_ok("matrix_update_target")
        require_operation_ok("execution_memory_peak", positive_field="bytes")
        require_operation_ok("interaction_diagnostics")

        typed_optional_permutation_symbols = {
            "gafime_gpu_permutation_memory_peak_v2",
            "gafime_gpu_permutation_pvalues_f32_v2",
            "gafime_gpu_permutation_pvalues_f64_v2",
        }
        optional_symbols = lifecycle.get("optional_symbols")
        if not isinstance(optional_symbols, list) or len(optional_symbols) != len(
            set(map(str, optional_symbols))
        ):
            failures.append("metal_optional_symbols_must_be_unique")
        optional_symbol_set = (
            set(map(str, optional_symbols))
            if isinstance(optional_symbols, list)
            else set()
        )
        if lifecycle_surface == "numeric-route-v2":
            canonical_permutation_required = True
        elif lifecycle_surface == "precision-typed-v1.1":
            if optional_symbol_set not in (set(), typed_optional_permutation_symbols):
                failures.append(
                    "metal_typed_optional_permutation_family_must_be_complete"
                )
            canonical_permutation_required = (
                optional_symbol_set == typed_optional_permutation_symbols
            )
        if canonical_permutation_required:
            if lifecycle.get("permutation_supported") is not True:
                failures.append("metal_permutation_support_marker_required")
            require_operation_ok("permutation_memory_peak", positive_field="bytes")
            require_operation_ok("permutation_pvalues", positive_field="row_count")
        else:
            if lifecycle.get("permutation_supported") is not False:
                failures.append("metal_permutation_unsupported_marker_required")
            for name in ("permutation_memory_peak", "permutation_pvalues"):
                observed = operation_status.get(name)
                if (
                    not isinstance(observed, Mapping)
                    or observed.get("status") != "unsupported"
                    or observed.get("abi_status") != -2
                ):
                    failures.append(f"metal_{name}_unsupported_status_required")
        checksum = lifecycle.get("result_checksum")
        if (
            not isinstance(checksum, (int, float))
            or isinstance(checksum, bool)
            or not math.isfinite(float(checksum))
        ):
            failures.append("metal_canonical_result_checksum_required")
        failures.extend(_metal_inline_lifecycle_provenance_failures(lifecycle, payload))

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
    if canonical_permutation_required:
        expected_canonical_records.update(
            {
                ("permutation_memory_peak", "none"),
                ("permutation_pvalues", "pearson"),
            }
        )
    observed_canonical_records: set[tuple[str, str]] = set()
    observed_canonical_record_count = 0
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
        observed_canonical_record_count += 1
        if identity in observed_canonical_records:
            failures.append(f"canonical_record_{index}_duplicate_identity")
        observed_canonical_records.add(identity)
        samples = record.get("samples_us")
        if (
            not isinstance(samples, list)
            or not isinstance(repeats, int)
            or len(samples) != repeats
        ):
            failures.append(f"canonical_record_{index}_raw_sample_count_mismatch")
        elif any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in samples
        ):
            failures.append(f"canonical_record_{index}_invalid_raw_sample")
        failures.extend(
            _metal_timing_record_failures(
                record,
                prefix=f"canonical_record_{index}",
                repeats=repeats,
                expected_target_us=sample_region_target_us,
            )
        )
        if record.get("clock") != "host_steady_clock_canonical_abi1_1":
            failures.append(f"canonical_record_{index}_clock_mismatch")
        if record.get("synchronization") != (
            "canonical_abi1_1_payload_call_returns_after_device_completion"
        ):
            failures.append(f"canonical_record_{index}_synchronization_mismatch")
    if observed_canonical_records != expected_canonical_records:
        failures.append("complete_canonical_payload_record_set_required")
    if observed_canonical_record_count != len(expected_canonical_records):
        failures.append("canonical_payload_record_count_mismatch")

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
                        identity = (
                            payload_identity
                            if label == "payload"
                            else metallib_identity
                        )
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
        "target_stat_preparation": "target_stat_preparation",
        "feature_stat_preparation": "feature_stat_preparation",
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
        "d2h_managed_resource_synchronize": "d2h_transfer",
        "report_construction": "report_construction",
        "report": "report_construction",
    }
    # Payload-boundary and admission-query records are real supplemental
    # observations, but they are not interchangeable with the direct
    # decomposition operations above.  Keep their taxonomy explicit so a
    # helper cannot make an unvalidated claim by inventing a new spelling.
    supplemental = NATIVE_SUPPLEMENTAL_OPERATION_ALIASES.get(raw_operation)
    if supplemental is not None:
        return {supplemental}
    if raw_operation in {"execute", "canonical_execute", "payload_execute"}:
        return {"supplemental:payload_execute"}
    canonical = aliases.get(raw_operation)
    if canonical is None:
        return set()
    if raw_operation == "ranking_topk_and_gather":
        return {"ranking_topk", "selected_row_gather"}
    return {canonical}


def _native_lane_contract_active(payload: Mapping[str, object]) -> bool:
    """Return whether the artifact opts into the strict lane contract.

    Older ABI-1.0 transport fixtures in the repository predate lane-isolated
    helper processes and remain useful for compatibility tests.  Current
    CUDA/ROCm helpers advertise ``lane_isolation``; only those artifacts are
    admitted to the new lane-aware path.  Once a current artifact advertises
    either lane marker, missing or contradictory lane metadata is an
    integrity failure rather than a legacy fallback.
    """

    return (
        payload.get("lane_isolation") is not None
        or payload.get("evidence_lane") is not None
    )


def _native_process_attestation_failures(
    payload: Mapping[str, object], *, backend: str
) -> list[str]:
    """Validate the runner/child identity bound into a modern GPU artifact."""

    if backend not in ("cuda", "rocm") or not _native_lane_contract_active(payload):
        return []
    failures: list[str] = []
    if payload.get("lane_isolation") != NATIVE_LANE_ISOLATION:
        failures.append("native_lane_isolation_marker_required")
    runner_pid = payload.get("runner_pid")
    process_id = payload.get("process_id")
    if (
        isinstance(runner_pid, bool)
        or not isinstance(runner_pid, int)
        or runner_pid <= 0
    ):
        failures.append("native_runner_pid_attestation_required")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        failures.append("native_process_id_attestation_required")
    if (
        isinstance(runner_pid, int)
        and not isinstance(runner_pid, bool)
        and isinstance(process_id, int)
        and not isinstance(process_id, bool)
        and runner_pid == process_id
    ):
        failures.append("native_runner_and_child_pid_must_differ")
    invocation_id = payload.get("runner_invocation_id")
    if not isinstance(invocation_id, str) or not re.fullmatch(
        r"[0-9a-fA-F]{32}", invocation_id
    ):
        failures.append("native_runner_invocation_attestation_required")
    environment = _environment_mapping(payload.get("environment"))
    if environment is None:
        failures.append("native_process_attestation_environment_required")
    else:
        if isinstance(runner_pid, int) and str(runner_pid) != environment.get(
            "GAFIME_NATIVE_RUNNER_PID"
        ):
            failures.append("native_runner_pid_environment_mismatch")
        if isinstance(invocation_id, str) and invocation_id != environment.get(
            "GAFIME_NATIVE_RUNNER_INVOCATION_ID"
        ):
            failures.append("native_runner_invocation_environment_mismatch")
    return failures


def _native_lane_contract_failures(
    payload: Mapping[str, object],
    records: Sequence[object],
    *,
    backend: str,
    profiles: set[str],
) -> tuple[list[str], str | None, dict[str, set[str]]]:
    """Validate one strict CUDA/ROCm artifact's lane and operation scope.

    The result includes operation coverage keyed by profile and is deliberately
    scoped to one lane.  Backend readiness performs the union across separate
    lane/policy/A-B artifacts; this function never treats a full decomposition
    split across files as if it were present in this artifact.
    """

    if backend not in ("cuda", "rocm") or not _native_lane_contract_active(payload):
        return [], None, {}
    failures: list[str] = []
    raw_lane = payload.get("evidence_lane")
    lane = raw_lane if isinstance(raw_lane, str) else None
    if lane not in NATIVE_EVIDENCE_LANE_SET:
        failures.append("native_evidence_lane_required_and_known")
        lane = None
    failures.extend(_native_process_attestation_failures(payload, backend=backend))

    observed_by_profile: dict[str, set[str]] = {profile: set() for profile in profiles}
    if lane is not None:
        mode = payload.get("execution_mode")
        if mode not in NATIVE_LANE_EXECUTION_MODES[lane]:
            failures.append(f"native_{lane}_execution_mode_mismatch")
        if lane == "canonical_payload_api":
            resolution = payload.get("canonical_payload_resolution")
            if not isinstance(resolution, Mapping) or resolution.get("status") != (
                "resolved"
            ):
                failures.append("native_canonical_payload_resolution_required")
            elif not isinstance(resolution.get("symbols"), (list, tuple)) or not (
                resolution.get("symbols")
            ):
                failures.append("native_canonical_payload_symbols_required")
        else:
            # A direct/host record is not a payload timing record.  Requiring
            # an explicit positive marker prevents a missing field or a
            # canonical lifecycle side effect from being interpreted as
            # payload-free evidence.
            if payload.get("payload_not_loaded") is not True:
                failures.append("native_supplemental_payload_not_loaded_required")
            if payload.get("payload_loaded") is True:
                failures.append("native_supplemental_payload_loaded_forbidden")
            marker = payload.get("payload_execution_mode")
            if marker not in NATIVE_PAYLOAD_NOT_LOADED_MARKERS:
                failures.append("native_supplemental_payload_execution_marker_required")
            resolution = payload.get("canonical_payload_resolution")
            if isinstance(resolution, Mapping) and resolution.get("status") == (
                "resolved"
            ):
                failures.append("native_supplemental_payload_resolution_forbidden")
            lifecycle = payload.get("canonical_payload_lifecycle")
            if (
                isinstance(lifecycle, Mapping)
                and lifecycle.get("status") == ("validated")
                and lifecycle.get("binding") != "external_canonical_evidence"
            ):
                failures.append("native_supplemental_payload_lifecycle_forbidden")

        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                failures.append(f"record_{index}_must_be_object")
                continue
            record_lane = record.get("evidence_lane")
            if record_lane != lane:
                failures.append(
                    f"record_{index}_evidence_lane_mismatch:expected={lane}:observed={record_lane}"
                )
                continue
            record_profile = record.get("profile")
            if isinstance(record_profile, str) and record_profile in profiles:
                observed_by_profile.setdefault(record_profile, set()).update(
                    _native_operation_names(
                        record.get("operation"), record.get("metric")
                    )
                )

        required_operations = NATIVE_LANE_REQUIRED_OPERATIONS[lane]
        # CUDA direct evidence includes two extra preparation kernels; ROCm's
        # canonical/direct helper does not expose those as independent
        # records.  The lane contract remains explicit per backend.
        if lane == "supplemental_internal_kernel":
            required_operations = (
                NATIVE_LANE_REQUIRED_OPERATIONS[lane]
                if backend == "cuda"
                else NATIVE_LANE_REQUIRED_OPERATIONS[lane]
                - frozenset(("target_stat_preparation", "feature_stat_preparation"))
            )
        for profile in sorted(profiles):
            missing = sorted(
                required_operations - observed_by_profile.get(profile, set())
            )
            if missing:
                failures.append(
                    f"native_{lane}_profile_{profile}_operation_coverage_incomplete:"
                    + ",".join(missing)
                )
    return failures, lane, observed_by_profile


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
                member
                for member in sorted(allowed_members)
                if member in archive.namelist()
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


def _native_helper_provenance_failures(
    payload: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    source_commit: object,
    kind: str,
) -> list[str]:
    """Authenticate the common benchmark source independently of the product.

    Native A/B helpers are intentionally compiled once from the candidate
    harness and load/link each product variant separately.  Product commits
    must differ across the comparison, while the tracked helper commit, blob,
    and SHA-256 must be identical.  A generic native-decomposition record may
    be produced by a separate lifecycle consumer, so this rule applies only to
    the backend timing/microbenchmark helpers themselves.
    """

    if kind == "native_decomposition":
        return []
    failures: list[str] = []
    if payload.get("product_source_commit") != source_commit:
        failures.append("native_product_source_commit_mismatch")
    product_tree = payload.get("product_source_tree_state")
    if not isinstance(product_tree, Mapping) or product_tree.get("status") != "clean":
        failures.append("native_clean_product_source_tree_required")
    harness_commit = payload.get("harness_source_commit")
    if (
        not isinstance(harness_commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", harness_commit) is None
    ):
        failures.append("native_harness_source_commit_required")
    harness_tree = payload.get("harness_source_tree_state")
    if not isinstance(harness_tree, Mapping) or harness_tree.get("status") != "clean":
        failures.append("native_clean_harness_source_tree_required")

    harness_identity = provenance.get("harness_source")
    failures.extend(_native_identity_failures(harness_identity, "harness_source"))
    benchmark_identity = provenance.get("benchmark_source")
    if isinstance(harness_identity, Mapping) and isinstance(
        benchmark_identity, Mapping
    ):
        harness_sha = str(harness_identity.get("sha256", "")).lower()
        benchmark_sha = str(benchmark_identity.get("sha256", "")).lower()
        if harness_sha != benchmark_sha:
            failures.append("native_harness_and_benchmark_source_sha256_mismatch")

    source_blob = payload.get("harness_source_blob")
    if not isinstance(source_blob, Mapping):
        failures.append("native_harness_source_blob_required")
    else:
        relative_path = source_blob.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            failures.append("native_harness_relative_path_invalid")
        blob_source_sha = str(source_blob.get("source_sha256", "")).lower()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", blob_source_sha)
            or not isinstance(harness_identity, Mapping)
            or blob_source_sha != str(harness_identity.get("sha256", "")).lower()
        ):
            failures.append("native_harness_blob_source_sha256_mismatch")
        current_blob = str(source_blob.get("current_git_blob", "")).lower()
        head_blob = str(source_blob.get("head_git_blob", "")).lower()
        if (
            re.fullmatch(r"[0-9a-f]{40}", current_blob) is None
            or current_blob != head_blob
        ):
            failures.append("native_harness_git_blob_mismatch")
    if kind == "core_microbenchmark":
        # Core's Rust arithmetic helper is compiled by a tracked Python runner.
        # Its behavior affects compiler arguments and the exact linked product
        # rlib, so authenticating only the Rust source would leave a provenance
        # gap in an otherwise common-harness A/B comparison.
        runner_identity = provenance.get("harness_runner")
        runner_blob = payload.get("harness_runner_blob")
        if not isinstance(runner_blob, Mapping):
            failures.append("native_harness_runner_blob_required")
        else:
            runner_relative_path = runner_blob.get("relative_path")
            if (
                not isinstance(runner_relative_path, str)
                or not runner_relative_path
                or Path(runner_relative_path).is_absolute()
                or ".." in Path(runner_relative_path).parts
            ):
                failures.append("native_harness_runner_relative_path_invalid")
            runner_source_sha = str(runner_blob.get("source_sha256", "")).lower()
            if (
                re.fullmatch(r"[0-9a-f]{64}", runner_source_sha) is None
                or not isinstance(runner_identity, Mapping)
                or runner_source_sha != str(runner_identity.get("sha256", "")).lower()
            ):
                failures.append("native_harness_runner_blob_source_sha256_mismatch")
            runner_current_blob = str(runner_blob.get("current_git_blob", "")).lower()
            runner_head_blob = str(runner_blob.get("head_git_blob", "")).lower()
            if (
                re.fullmatch(r"[0-9a-f]{40}", runner_current_blob) is None
                or runner_current_blob != runner_head_blob
            ):
                failures.append("native_harness_runner_git_blob_mismatch")
    return failures


def _native_payload_route_failures(
    backend: str, payload: Mapping[str, object], *, kind: str
) -> list[str]:
    """Reject native evidence that cannot exercise one authenticated ABI surface.

    The native CUDA helper can still run its supplemental in-binary kernels
    when an old payload only exports dtype-suffixed entry points.  That is
    useful diagnostics, but it is not a comparable measurement of the generic
    ABI 1.1 lifecycle used by the cold/public lanes.  Keep such evidence
    explicitly unsupported so a candidate helper cannot be paired with an
    incompatible baseline payload.  The exact pre-freeze typed ABI 1.1 baseline
    is accepted as historical A/B evidence; partial or mixed symbol ownership
    is rejected, and the candidate remains responsible for the generic route.
    """

    if backend == "core" or kind == "native_decomposition":
        return []
    lane = payload.get("evidence_lane")
    strict_lane_contract = _native_lane_contract_active(payload)
    if strict_lane_contract and backend in ("cuda", "rocm"):
        if lane in ("supplemental_internal_kernel", "supplemental_host_phase"):
            resolution = payload.get("canonical_payload_resolution")
            if (
                isinstance(resolution, Mapping)
                and resolution.get("status") == "resolved"
            ):
                return ["native_supplemental_live_payload_resolution_forbidden"]
            lifecycle = payload.get("canonical_payload_lifecycle")
            if not isinstance(lifecycle, Mapping):
                return ["native_supplemental_external_canonical_lifecycle_required"]
            lifecycle_path = lifecycle.get("path")
            lifecycle_sha = lifecycle.get("sha256")
            if (
                lifecycle.get("status") != "validated"
                or lifecycle.get("binding") != "external_canonical_evidence"
                or not isinstance(lifecycle_path, str)
                or not lifecycle_path
                or not isinstance(lifecycle_sha, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", lifecycle_sha) is None
            ):
                return ["native_supplemental_external_canonical_lifecycle_required"]
            # The hash-bound external lifecycle is opened and fully
            # reauthenticated later in the common artifact validator.  A
            # supplemental lane never resolves the product payload live.
            return []
        if lane != "canonical_payload_api":
            return ["native_canonical_payload_lane_required"]
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
        surface = resolution.get("abi_surface")
        if strict_lane_contract and surface not in CANONICAL_ABI_SURFACES:
            return ["native_generic_abi_surface_required"]
        required = (
            CANONICAL_ABI_SYMBOLS_BY_SURFACE.get(str(surface), frozenset())
            if surface in CANONICAL_ABI_SURFACES
            else {
                # The CUDA helper's resolution probe historically did not
                # query the surface name.  Keep this conservative fallback for
                # generic payloads while requiring the complete symbol set
                # whenever a surface is declared.
                "gafime_gpu_matrix_alloc_v2",
                "gafime_gpu_matrix_upload_v2",
                "gafime_gpu_execute_v2",
                "gafime_gpu_matrix_free_v2",
            }
        )
        if not required <= observed or (
            strict_lane_contract and observed != set(required)
        ):
            return ["native_generic_abi_route_unsupported"]
        return []
    if backend == "rocm":
        checks = payload.get("self_checks")
        if not isinstance(checks, Mapping):
            return ["native_generic_abi_route_unsupported"]
        surface = checks.get("abi_surface", payload.get("abi_surface"))
        if surface not in CANONICAL_ABI_SURFACES:
            return ["native_generic_abi_surface_required"]
        if checks.get("canonical_symbols_authenticated") is not True:
            return ["native_generic_abi_route_unsupported"]
        if (
            surface == "numeric-route-v2" and checks.get("canonical_routes") is not True
        ) or (
            surface == "precision-typed-v1.1"
            and checks.get("typed_precision_profiles") is not True
        ):
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
        surface = (
            lifecycle.get("abi_surface") if isinstance(lifecycle, Mapping) else None
        )
        required = CANONICAL_ABI_SYMBOLS_BY_SURFACE.get(str(surface))
        if (
            not isinstance(lifecycle, Mapping)
            or lifecycle.get("status") != "validated"
            or surface not in CANONICAL_ABI_SURFACES
            or required is None
            or observed != required
        ):
            return ["native_generic_abi_route_unsupported"]
        return []
    return ["native_generic_abi_route_unsupported"]


def _rocm_direct_kernel_product_failures(
    payload: Mapping[str, object],
    *,
    evidence_lane: object,
    source_commit: object,
) -> list[str]:
    """Authenticate the exact ROCm lane binary and linked direct sources."""

    failures: list[str] = []
    if payload.get("compiled_lane") != evidence_lane:
        failures.append("rocm_compiled_lane_evidence_lane_mismatch")
    product = payload.get("direct_kernel_product")
    if not isinstance(product, Mapping):
        return failures + ["rocm_direct_kernel_product_required"]
    expected_compiled = evidence_lane == "supplemental_internal_kernel"
    if product.get("compiled") is not expected_compiled:
        failures.append("rocm_direct_kernel_product_compiled_marker_mismatch")
    if not expected_compiled:
        # Canonical and host controls are deliberately compiled without the
        # product HIP translation unit.  Their empty compile-bound identity is
        # positive evidence of that isolation, not a missing direct binding.
        if any(
            product.get(field) not in (None, "")
            for field in (
                "root",
                "commit",
                "kernels_sha256",
                "kernels_header_sha256",
                "direct_source_sha256",
            )
        ):
            failures.append("rocm_control_direct_kernel_product_identity_present")
        return failures
    product_root = payload.get("product_source_root", payload.get("source_root"))
    if (
        not isinstance(product.get("root"), str)
        or not product.get("root")
        or product.get("root") != product_root
    ):
        failures.append("rocm_direct_kernel_product_root_mismatch")
    if product.get("commit") != source_commit:
        failures.append("rocm_direct_kernel_product_commit_mismatch")
    digests: dict[str, str] = {}
    for field in ("kernels_sha256", "kernels_header_sha256", "direct_source_sha256"):
        digest = product.get(field)
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        ):
            failures.append(f"rocm_direct_kernel_product_{field}_required")
        else:
            digests[field] = digest.lower()

    source_blob = payload.get("source_blob")
    source_blob_sha = (
        str(source_blob.get("source_sha256", "")).lower()
        if isinstance(source_blob, Mapping)
        else ""
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", source_blob_sha) is None
        or digests.get("kernels_sha256") != source_blob_sha
    ):
        failures.append("rocm_direct_kernel_product_kernels_source_blob_mismatch")

    harness_root = payload.get("harness_source_root")
    bound_sources = (
        (
            "kernels",
            product_root,
            "src/rocm/kernels.hip",
            digests.get("kernels_sha256"),
        ),
        (
            "kernels_header",
            product_root,
            "src/rocm/kernels.hpp",
            digests.get("kernels_header_sha256"),
        ),
        (
            "direct_source",
            harness_root,
            "tests/gpu/rocm_native_direct_lane.hip",
            digests.get("direct_source_sha256"),
        ),
    )
    for label, root, relative_path, digest in bound_sources:
        if not isinstance(root, str) or not root:
            failures.append(f"rocm_direct_kernel_product_{label}_root_required")
            continue
        observed = _git_source_blob(root, Path(root) / relative_path)
        if observed.get("status") != "tracked_at_head":
            failures.append(
                f"rocm_direct_kernel_product_{label}_tracked_binding_required"
            )
            continue
        if digest is not None and observed.get("source_sha256") != digest:
            failures.append(f"rocm_direct_kernel_product_{label}_sha256_mismatch")
    if (
        isinstance(source_blob, Mapping)
        and isinstance(product_root, str)
        and product_root
    ):
        observed_source = _git_source_blob(
            product_root, Path(product_root) / "src/rocm/kernels.hip"
        )
        for field in (
            "status",
            "path",
            "relative_path",
            "source_sha256",
            "current_git_blob",
            "head_git_blob",
        ):
            if source_blob.get(field) != observed_source.get(field):
                failures.append(
                    "rocm_direct_kernel_product_source_blob_tracked_binding_mismatch"
                )
                break
    return failures


def _compiler_provenance_failures(backend: str, compiler: object) -> list[str]:
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
    """Authenticate one explicit ABI 1.1 lifecycle surface.

    The product source commit and the common standalone consumer harness are
    separate provenance domains.  This is important for the exact historical
    baseline: its payload may be checked out at the frozen commit while the
    same current consumer harness is compiled from the candidate tree.  A
    lifecycle record that omits either domain is rejected rather than being
    silently treated as a candidate-source build.
    """

    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return ["canonical_payload_lifecycle_must_be_json_object"]
    abi_surface = payload.get("abi_surface")
    if abi_surface not in CANONICAL_ABI_SURFACES:
        failures.append("canonical_payload_lifecycle_abi_surface_required")
        abi_surface = None
    expected_operations = CANONICAL_ABI_OPERATIONS_BY_SURFACE.get(
        str(abi_surface), frozenset()
    )
    expected_symbols = CANONICAL_ABI_SYMBOLS_BY_SURFACE.get(
        str(abi_surface), frozenset()
    )
    expected_marker_schema = CANONICAL_ABI_MARKER_SCHEMAS.get(str(abi_surface))
    expected_execution_layer = CANONICAL_ABI_EXECUTION_LAYERS.get(str(abi_surface))
    expected_contract_role = CANONICAL_ABI_CONTRACT_ROLES.get(str(abi_surface))
    if payload.get("schema") != "gafime.native-decomposition.v1":
        failures.append("canonical_payload_lifecycle_schema_mismatch")
    if payload.get("status") != "pass":
        failures.append("canonical_payload_lifecycle_status_invalid")
    if payload.get("execution_mode") != "canonical_payload":
        failures.append("canonical_payload_lifecycle_must_be_canonical")
    if payload.get("execution_layer") != expected_execution_layer:
        failures.append("canonical_payload_lifecycle_independent_consumer_required")
    if payload.get("abi") != "1.1":
        failures.append("canonical_payload_lifecycle_abi_mismatch")
    if payload.get("contract_role") != expected_contract_role:
        failures.append("canonical_payload_lifecycle_contract_role_mismatch")
    if payload.get("backend") != backend:
        failures.append("canonical_payload_lifecycle_backend_mismatch")
    product_source_commit = payload.get("product_source_commit")
    if (
        product_source_commit != source_commit
        or payload.get("source_commit") != source_commit
    ):
        failures.append("canonical_payload_lifecycle_source_commit_mismatch")
    harness_source_commit = payload.get("harness_source_commit")
    if (
        not isinstance(harness_source_commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", harness_source_commit) is None
    ):
        failures.append("canonical_payload_lifecycle_harness_source_commit_required")
    harness_tree_state = payload.get("harness_source_tree_state")
    if (
        not isinstance(harness_tree_state, Mapping)
        or harness_tree_state.get("status") != "clean"
    ):
        failures.append(
            "canonical_payload_lifecycle_clean_harness_source_tree_required"
        )
    if payload.get("source_commit") != product_source_commit:
        failures.append("canonical_payload_lifecycle_source_commit_mismatch")
    tree_state = payload.get("source_tree_state")
    if not isinstance(tree_state, Mapping) or tree_state.get("status") != "clean":
        failures.append("canonical_payload_lifecycle_clean_source_tree_required")
    expected_profiles = set(BACKEND_PROFILES.get(backend, ()))
    raw_profiles = payload.get("profiles")
    if (
        not isinstance(raw_profiles, list)
        or set(map(str, raw_profiles)) != expected_profiles
    ):
        failures.append("canonical_payload_lifecycle_profile_coverage_incomplete")
    raw_operations = payload.get("operations")
    if (
        not isinstance(raw_operations, list)
        or len(raw_operations) != len(expected_operations)
        or set(map(str, raw_operations)) != set(expected_operations)
    ):
        failures.append("canonical_payload_lifecycle_operation_coverage_incomplete")
    raw_symbols = payload.get("symbols")
    if (
        not isinstance(raw_symbols, list)
        or len(raw_symbols) != len(expected_symbols)
        or set(map(str, raw_symbols)) != set(expected_symbols)
    ):
        failures.append("canonical_payload_lifecycle_symbol_coverage_incomplete")
    if payload.get("route_count") != len(expected_profiles):
        failures.append("canonical_payload_lifecycle_route_count_mismatch")

    result = payload.get("consumer_result")
    marker: object = None
    if not isinstance(result, Mapping):
        failures.append("canonical_payload_lifecycle_consumer_result_required")
    else:
        if result.get("schema") != expected_marker_schema:
            failures.append("canonical_payload_lifecycle_consumer_schema_mismatch")
        if result.get("status") != "pass" or result.get("returncode") != 0:
            failures.append("canonical_payload_lifecycle_consumer_did_not_pass")
        marker = result.get("marker")
    if not isinstance(marker, Mapping):
        failures.append("canonical_payload_lifecycle_consumer_marker_required")
    else:
        if marker.get("schema") != expected_marker_schema:
            failures.append("canonical_payload_lifecycle_marker_schema_mismatch")
        if marker.get("status") != "pass":
            failures.append("canonical_payload_lifecycle_marker_status_invalid")
        if marker.get("abi_surface") != abi_surface:
            failures.append("canonical_payload_lifecycle_marker_surface_mismatch")
        if marker.get("backend_kind") != CANONICAL_ABI_BACKEND_KINDS.get(backend):
            failures.append("canonical_payload_lifecycle_marker_backend_mismatch")
        marker_count = marker.get("route_count", marker.get("profile_count"))
        if marker_count != len(expected_profiles):
            failures.append("canonical_payload_lifecycle_marker_route_count_mismatch")
        marker_operations = marker.get("operations")
        if (
            not isinstance(marker_operations, list)
            or len(marker_operations) != len(expected_operations)
            or set(map(str, marker_operations)) != set(expected_operations)
        ):
            failures.append("canonical_payload_lifecycle_marker_operations_incomplete")

    provenance = payload.get("provenance")
    required_identities = {
        "payload",
        "wheel",
        "consumer_binary",
        "consumer_source",
        "harness_source",
    }
    if not isinstance(provenance, Mapping):
        failures.append("canonical_payload_lifecycle_provenance_required")
        provenance = {}
    for name in sorted(required_identities):
        failures.extend(
            _native_identity_failures(
                provenance.get(name), f"canonical_payload_lifecycle_{name}"
            )
        )
    consumer_identity = provenance.get("consumer_source")
    harness_identity = provenance.get("harness_source")
    if (
        isinstance(consumer_identity, Mapping)
        and isinstance(harness_identity, Mapping)
        and str(consumer_identity.get("sha256", "")).lower()
        != str(harness_identity.get("sha256", "")).lower()
    ):
        failures.append("canonical_payload_lifecycle_harness_source_sha256_mismatch")
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
                failures.append(f"canonical_payload_lifecycle_{name}_sha256_mismatch")

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
                    embedded_digest = hashlib.sha256(
                        archive.read(str(member))
                    ).hexdigest()
                if embedded_digest != member_digest:
                    failures.append(
                        "canonical_payload_lifecycle_wheel_member_sha256_mismatch"
                    )
                if (
                    not isinstance(payload_identity, Mapping)
                    or embedded_digest
                    != str(payload_identity.get("sha256", "")).lower()
                ):
                    failures.append(
                        "canonical_payload_lifecycle_payload_not_from_wheel"
                    )
            except (KeyError, OSError, zipfile.BadZipFile):
                failures.append("canonical_payload_lifecycle_wheel_member_unreadable")
    return failures


def _position_contrast_effects(
    position_samples: Mapping[int, Sequence[float]],
) -> tuple[dict[str, float], dict[tuple[int, int], float]]:
    """Return position medians and signed percent contrasts for one cell."""

    normalized = {
        position: [float(value) for value in position_samples[position]]
        for position in range(3)
    }
    position_medians = {
        str(position): statistics.median(normalized[position]) for position in range(3)
    }
    center = statistics.median(
        value for values in normalized.values() for value in values
    )
    denominator = max(center, float.fromhex("0x1.0p-1022"))
    effects = {
        contrast: (
            position_medians[str(contrast[1])] - position_medians[str(contrast[0])]
        )
        * 100.0
        / denominator
        for contrast in ORDER_EFFECT_POSITION_CONTRASTS
    }
    return position_medians, effects


def _simultaneous_clustered_order_assessment(
    cells: Mapping[
        tuple[object, ...],
        Mapping[object, Mapping[object, Mapping[int, Sequence[float]]]],
    ],
    *,
    seed: int,
    bootstrap_resamples: int = ORDER_EFFECT_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Build simultaneous order-position intervals from complete cycle clusters.

    Each cell is stratified (for example by A/B block), and each stratum owns
    complete six-order cycles.  A bootstrap draw resamples whole cycles within
    every stratum.  Values inside an order record are already represented by a
    single record median, so the repeated raw timings are never promoted to
    independent experimental units.  A joint maximum standardized error over
    every cell and all three position contrasts supplies familywise intervals.
    """

    observed: dict[tuple[tuple[object, ...], tuple[int, int]], float] = {}
    position_medians_by_cell: dict[tuple[object, ...], dict[str, float]] = {}
    observations_per_position: dict[tuple[object, ...], dict[str, int]] = {}
    for cell_key, strata in sorted(cells.items(), key=lambda item: repr(item[0])):
        positions = {position: [] for position in range(3)}
        for clusters in strata.values():
            for cluster in clusters.values():
                for position in range(3):
                    positions[position].extend(
                        float(value) for value in cluster[position]
                    )
        position_medians, effects = _position_contrast_effects(positions)
        position_medians_by_cell[cell_key] = position_medians
        observations_per_position[cell_key] = {
            str(position): len(values) for position, values in positions.items()
        }
        for contrast, effect in effects.items():
            observed[(cell_key, contrast)] = effect

    ordered_cells = sorted(cells.items(), key=lambda item: repr(item[0]))
    cluster_ids_by_stratum: dict[object, list[object]] = {}
    for _, strata in ordered_cells:
        for stratum, clusters in strata.items():
            cluster_ids = sorted(clusters, key=repr)
            existing = cluster_ids_by_stratum.setdefault(stratum, cluster_ids)
            if existing != cluster_ids:
                return {
                    "status": "insufficient_complete_order_cycle_evidence",
                    "cells": {},
                    "comparison_cells": len(cells),
                    "position_pair_contrasts_per_cell": len(
                        ORDER_EFFECT_POSITION_CONTRASTS
                    ),
                    "total_comparisons": len(observed),
                    "stratum_cycle_id_mismatch": True,
                    "decision_rule": (
                        "cells sharing a schedule stratum must retain identical "
                        "complete cycle ids"
                    ),
                }
    exact_clusterwise_equivalence = all(
        all(
            sorted(float(value) for value in cluster[0])
            == sorted(float(value) for value in cluster[position])
            for position in (1, 2)
        )
        for _, strata in ordered_cells
        for clusters in strata.values()
        for cluster in clusters.values()
    )
    scales: dict[tuple[tuple[object, ...], tuple[int, int]], float]
    if exact_clusterwise_equivalence:
        # Every complete-cycle experimental unit has exactly the same sample
        # multiset at all three positions.  Any shared cycle resample therefore
        # has identically zero position contrasts, so the simultaneous bound is
        # analytically zero; iterating identical bootstrap draws adds no evidence.
        scales = {contrast_id: 0.0 for contrast_id in observed}
        critical_value = 0.0
    else:
        rng = random.Random(seed)
        bootstrap_effects = {contrast_id: [] for contrast_id in observed}
        for _ in range(bootstrap_resamples):
            sampled_clusters_by_stratum = {
                stratum: [
                    cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids
                ]
                for stratum, cluster_ids in sorted(
                    cluster_ids_by_stratum.items(), key=lambda item: repr(item[0])
                )
            }
            for cell_key, strata in ordered_cells:
                positions = {position: [] for position in range(3)}
                for stratum, clusters in sorted(
                    strata.items(), key=lambda item: repr(item[0])
                ):
                    for sampled_cluster in sampled_clusters_by_stratum[stratum]:
                        sampled = clusters[sampled_cluster]
                        for position in range(3):
                            positions[position].extend(
                                float(value) for value in sampled[position]
                            )
                _, effects = _position_contrast_effects(positions)
                for contrast, effect in effects.items():
                    bootstrap_effects[(cell_key, contrast)].append(effect)

        errors = {
            contrast_id: [
                value - observed[contrast_id]
                for value in bootstrap_effects[contrast_id]
            ]
            for contrast_id in observed
        }
        scales = {}
        for contrast_id, values in errors.items():
            scale = statistics.pstdev(values) if len(values) > 1 else 0.0
            if scale <= 1.0e-15:
                scale = max((abs(value) for value in values), default=0.0)
            scales[contrast_id] = scale
        max_standardized_errors: list[float] = []
        for bootstrap_index in range(bootstrap_resamples):
            maximum = 0.0
            for contrast_id, contrast_errors in errors.items():
                scale = scales[contrast_id]
                error = abs(contrast_errors[bootstrap_index])
                standardized = error / scale if scale > 0.0 else 0.0
                maximum = max(maximum, standardized)
            max_standardized_errors.append(maximum)
        critical_value = _percentile(
            max_standardized_errors, ORDER_EFFECT_FAMILYWISE_CONFIDENCE
        )

    assessments: dict[tuple[object, ...], dict[str, object]] = {}
    for cell_key in sorted(cells, key=repr):
        contrasts: list[dict[str, object]] = []
        classifications: list[str] = []
        for positions in ORDER_EFFECT_POSITION_CONTRASTS:
            contrast_id = (cell_key, positions)
            effect = observed[contrast_id]
            half_width = critical_value * scales[contrast_id]
            interval = [effect - half_width, effect + half_width]
            if interval[0] > 0.0:
                lower_absolute = interval[0]
            elif interval[1] < 0.0:
                lower_absolute = abs(interval[1])
            else:
                lower_absolute = 0.0
            upper_absolute = max(abs(interval[0]), abs(interval[1]))
            if lower_absolute > NATIVE_ORDER_INVESTIGATE_PERCENT:
                classification = ORDER_EFFECT_CONTAMINATED_STATUS
            elif upper_absolute <= NATIVE_ORDER_INVESTIGATE_PERCENT:
                classification = ORDER_EFFECT_CLEAN_STATUS
            else:
                classification = ORDER_EFFECT_INCONCLUSIVE_STATUS
            classifications.append(classification)
            contrasts.append(
                {
                    "positions": list(positions),
                    "observed_signed_percent": effect,
                    "simultaneous_95_percent_ci_percent": interval,
                    "simultaneous_lower_absolute_bound_percent": lower_absolute,
                    "simultaneous_upper_absolute_bound_percent": upper_absolute,
                    "status": classification,
                }
            )
        if ORDER_EFFECT_CONTAMINATED_STATUS in classifications:
            status = ORDER_EFFECT_CONTAMINATED_STATUS
        elif all(value == ORDER_EFFECT_CLEAN_STATUS for value in classifications):
            status = ORDER_EFFECT_CLEAN_STATUS
        else:
            status = ORDER_EFFECT_INCONCLUSIVE_STATUS
        position_medians = position_medians_by_cell[cell_key]
        median_values = list(position_medians.values())
        center = statistics.median(median_values)
        assessments[cell_key] = {
            "status": status,
            "position_medians": position_medians,
            "observations_per_position": observations_per_position[cell_key],
            "max_observed_position_spread_percent": (
                (max(median_values) - min(median_values)) * 100.0 / center
                if center > 0.0
                else 0.0
            ),
            "position_contrasts": contrasts,
        }
    return {
        "status": (
            ORDER_EFFECT_CONTAMINATED_STATUS
            if any(
                cell["status"] == ORDER_EFFECT_CONTAMINATED_STATUS
                for cell in assessments.values()
            )
            else ORDER_EFFECT_CLEAN_STATUS
            if assessments
            and all(
                cell["status"] == ORDER_EFFECT_CLEAN_STATUS
                for cell in assessments.values()
            )
            else ORDER_EFFECT_INCONCLUSIVE_STATUS
        ),
        "cells": assessments,
        "bootstrap_resamples": bootstrap_resamples,
        "familywise_confidence_level": ORDER_EFFECT_FAMILYWISE_CONFIDENCE,
        "multiple_comparison_correction": (
            "joint_max_standardized_complete_cycle_cluster_bootstrap_across_"
            "all_cells_and_position_pair_contrasts"
        ),
        "bootstrap_unit": "complete_six_order_cycle_within_stratum",
        "raw_sample_clustering": (
            "within-record raw timings collapsed to one median and never treated "
            "as independent order assignments"
        ),
        "comparison_cells": len(cells),
        "position_pair_contrasts_per_cell": len(ORDER_EFFECT_POSITION_CONTRASTS),
        "total_comparisons": len(observed),
        "simultaneous_critical_value": critical_value,
        "bootstrap_execution": (
            "analytic_zero_effect_complete_cycle_degeneracy"
            if exact_clusterwise_equivalence
            else "joint_shared_cycle_resampling"
        ),
        "decision_rule": (
            "clean only when every simultaneous upper absolute bound is at most "
            "one percent; contaminated when any simultaneous lower absolute bound "
            "exceeds one percent; otherwise inconclusive"
        ),
    }


def _recorded_native_profile_order_cycles(
    records: Sequence[Mapping[str, object]],
    *,
    order_repetitions: object,
) -> tuple[list[list[tuple[str, ...]]], list[str]]:
    """Recover the exact six-order sequence for every native schedule cycle."""

    if (
        not isinstance(order_repetitions, int)
        or isinstance(order_repetitions, bool)
        or order_repetitions < MIN_NATIVE_ORDER_REPETITIONS
    ):
        return [], ["native_order_repetitions_required"]

    expected_orders = set(itertools.permutations(PROFILE_ORDER))
    expected_indices = set(range(order_repetitions * len(expected_orders)))
    order_by_index: dict[int, tuple[str, ...]] = {}
    failures: list[str] = []
    for record_index, record in enumerate(records):
        order_index = record.get("order_index")
        raw_order = record.get("profile_order")
        order = (
            tuple(str(value) for value in raw_order)
            if isinstance(raw_order, (list, tuple))
            else ()
        )
        if (
            not isinstance(order_index, int)
            or isinstance(order_index, bool)
            or order_index not in expected_indices
            or order not in expected_orders
        ):
            failures.append(f"record_{record_index}_native_order_assignment_invalid")
            continue
        prior = order_by_index.setdefault(order_index, order)
        if prior != order:
            failures.append(f"record_{record_index}_native_order_assignment_conflict")

    if set(order_by_index) != expected_indices:
        failures.append("native_record_order_cycle_coverage_required")
        return [], failures

    cycles: list[list[tuple[str, ...]]] = []
    order_count = len(expected_orders)
    for cycle_index in range(order_repetitions):
        cycle = [
            order_by_index[cycle_index * order_count + order_index]
            for order_index in range(order_count)
        ]
        if len(set(cycle)) != order_count or set(cycle) != expected_orders:
            failures.append(f"native_order_cycle_{cycle_index}_incomplete")
        cycles.append(cycle)
    if len({tuple(cycle) for cycle in cycles}) < 2:
        failures.append("native_profile_order_cycle_variation_required")
    if any(left == right for left, right in itertools.pairwise(cycles)):
        failures.append("native_adjacent_profile_order_cycle_reuse_forbidden")
    return cycles, failures


def _native_order_sensitivity(
    records: Sequence[Mapping[str, object]],
    *,
    order_repetitions: object,
) -> dict[str, object]:
    """Assess GPU order effects with simultaneous complete-cycle intervals."""

    if not isinstance(order_repetitions, int) or isinstance(order_repetitions, bool):
        return {
            "status": "not_evaluated_order_repetitions_missing",
            "order_repetitions": order_repetitions,
            "required_order_repetitions": MIN_NATIVE_ORDER_REPETITIONS,
            "raw_per_order_data": True,
            "cells": [],
        }
    if order_repetitions < MIN_NATIVE_ORDER_REPETITIONS:
        return {
            "status": "insufficient_order_repeatability",
            "order_repetitions": order_repetitions,
            "required_order_repetitions": MIN_NATIVE_ORDER_REPETITIONS,
            "raw_per_order_data": True,
            "cells": [],
        }

    recorded_order_cycles, order_cycle_failures = _recorded_native_profile_order_cycles(
        records,
        order_repetitions=order_repetitions,
    )

    expected_orders = set(itertools.permutations(PROFILE_ORDER))
    cells: dict[
        tuple[object, ...],
        dict[object, dict[object, dict[int, list[float]]]],
    ] = {}
    orders_by_cell_cycle: dict[tuple[object, ...], dict[int, set[tuple[str, ...]]]] = {}
    record_counts_by_cell_cycle: dict[tuple[object, ...], dict[int, int]] = {}
    loop_counts_by_cell: dict[tuple[object, ...], set[int]] = {}
    raw_sample_records = 0
    raw_scaled_records = 0
    records_with_missing_raw_samples = 0
    invalid_records = 0
    for record in records:
        profile = record.get("profile")
        order_index = record.get("order_index")
        profile_order = record.get("profile_order")
        samples = record.get("samples_us", record.get("samples_ns"))
        raw_samples = record.get(
            "raw_samples_us",
            record.get("raw_samples_ns", record.get("raw_region_ns")),
        )
        if (
            not isinstance(profile, str)
            or not isinstance(order_index, int)
            or isinstance(order_index, bool)
            or order_index < 0
            or not isinstance(profile_order, (list, tuple))
            or profile not in profile_order
            or tuple(str(value) for value in profile_order) not in expected_orders
            or not isinstance(samples, list)
            or len(samples) < MIN_REPETITIONS
        ):
            invalid_records += 1
            continue
        if not isinstance(raw_samples, list) or len(raw_samples) < MIN_REPETITIONS:
            records_with_missing_raw_samples += 1
            continue
        if len(raw_samples) != len(samples):
            records_with_missing_raw_samples += 1
            continue
        numeric_samples = [
            float(value)
            for value in samples
            if isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        ]
        numeric_raw_samples = [
            float(value)
            for value in raw_samples
            if isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        ]
        if len(numeric_samples) != len(samples) or len(numeric_raw_samples) != len(
            raw_samples
        ):
            invalid_records += 1
            continue
        loop_counts = record.get("loop_counts_per_sample")
        if not (
            isinstance(loop_counts, list)
            and len(loop_counts) == len(numeric_raw_samples)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in loop_counts
            )
            and len(set(loop_counts)) == 1
        ):
            invalid_records += 1
            continue
        raw_sample_records += 1
        raw_scaled_records += 1
        comparable_samples = [
            value / loop_count
            for value, loop_count in zip(numeric_raw_samples, loop_counts)
        ]
        cycle = order_index // 6
        position = list(profile_order).index(profile)
        cell_key = (
            str(record.get("operation")),
            str(record.get("metric")),
            profile,
            str(record.get("clock", record.get("timing_scope", ""))),
            str(record.get("timing_scope", "")),
            str(record.get("evidence_lane", "")),
            str(record.get("comparability", "")),
        )
        cells.setdefault(cell_key, {}).setdefault("all_orders", {}).setdefault(
            cycle, {0: [], 1: [], 2: []}
        )[position].append(statistics.median(comparable_samples))
        orders_by_cell_cycle.setdefault(cell_key, {}).setdefault(cycle, set()).add(
            tuple(str(value) for value in profile_order)
        )
        record_counts_by_cell_cycle.setdefault(cell_key, {}).setdefault(cycle, 0)
        record_counts_by_cell_cycle[cell_key][cycle] += 1
        loop_counts_by_cell.setdefault(cell_key, set()).add(loop_counts[0])

    expected_cycles = set(range(order_repetitions))
    incomplete_cells: list[list[object]] = []
    complete_cells: dict[
        tuple[object, ...],
        dict[object, dict[object, dict[int, list[float]]]],
    ] = {}
    for cell_key, strata in cells.items():
        cycle_values = strata["all_orders"]
        complete = (
            set(cycle_values) == expected_cycles
            and loop_counts_by_cell.get(cell_key, set())
            and len(loop_counts_by_cell[cell_key]) == 1
        )
        for cycle in expected_cycles:
            values = cycle_values.get(cycle, {})
            complete = complete and (
                set(values) == {0, 1, 2}
                and all(len(values[position]) == 2 for position in range(3))
                and orders_by_cell_cycle.get(cell_key, {}).get(cycle) == expected_orders
                and record_counts_by_cell_cycle.get(cell_key, {}).get(cycle) == 6
            )
        if complete:
            complete_cells[cell_key] = strata
        else:
            incomplete_cells.append(list(cell_key))

    if (
        not complete_cells
        or incomplete_cells
        or invalid_records
        or records_with_missing_raw_samples
        or order_cycle_failures
    ):
        assessment: dict[str, object] = {
            "status": "insufficient_complete_order_cycle_evidence",
            "cells": {},
            "comparison_cells": len(complete_cells),
            "total_comparisons": 0,
        }
    else:
        assessment = _simultaneous_clustered_order_assessment(
            complete_cells,
            seed=0x474146494D454F52,
        )

    cell_summaries: list[dict[str, object]] = []
    assessed_cells = assessment.get("cells", {})
    for cell_key in sorted(complete_cells, key=repr):
        cell = (
            assessed_cells.get(cell_key, {})
            if isinstance(assessed_cells, Mapping)
            else {}
        )
        cell_summaries.append(
            {
                "operation": cell_key[0],
                "metric": cell_key[1],
                "profile": cell_key[2],
                "clock": cell_key[3],
                "timing_scope": cell_key[4],
                "evidence_lane": cell_key[5],
                "comparability": cell_key[6],
                "cycles_observed": order_repetitions,
                **cell,
            }
        )

    return {
        **assessment,
        "order_repetitions": order_repetitions,
        "required_order_repetitions": MIN_NATIVE_ORDER_REPETITIONS,
        "raw_per_order_data": True,
        "raw_sample_records": raw_sample_records,
        "raw_scaled_records": raw_scaled_records,
        "records_with_missing_raw_samples": records_with_missing_raw_samples,
        "invalid_records": invalid_records,
        "incomplete_cells": incomplete_cells,
        "profile_order_cycles": [
            [list(order) for order in cycle] for cycle in recorded_order_cycles
        ],
        "distinct_profile_order_cycle_count": len(
            {tuple(cycle) for cycle in recorded_order_cycles}
        ),
        "order_cycle_schedule_failures": order_cycle_failures,
        "investigate_threshold_percent": NATIVE_ORDER_INVESTIGATE_PERCENT,
        "cells": cell_summaries,
    }


def _native_order_claim_families(
    records: Sequence[Mapping[str, object]],
    *,
    order_repetitions: object,
    combined_assessment: Mapping[str, object] | None = None,
    required_lanes: Sequence[str] | None = None,
) -> dict[str, object]:
    """Assess each predeclared native evidence lane without pooling claims.

    The installed-payload ABI surface and the helper-owned direct-kernel lane
    answer different performance questions.  Each lane therefore receives its
    own simultaneous complete-cycle assessment.  The all-record assessment is
    retained as the multiplicity-corrected gate for a combined native claim.
    Missing or unknown lane metadata can never become absence-of-evidence PASS.
    """

    by_lane = {
        lane: [record for record in records if record.get("evidence_lane") == lane]
        for lane in NATIVE_ORDER_EVIDENCE_LANES
    }
    unclassified_records = [
        record
        for record in records
        if record.get("evidence_lane") not in NATIVE_ORDER_EVIDENCE_LANES
    ]
    families: dict[str, dict[str, object]] = {}
    for lane, lane_records in by_lane.items():
        if not lane_records:
            families[lane] = {
                "status": "not_present",
                "claim_ready": False,
                "record_count": 0,
                "cells": [],
            }
            continue
        assessment = _native_order_sensitivity(
            lane_records,
            order_repetitions=order_repetitions,
        )
        family = {
            **assessment,
            "claim_ready": assessment.get("status") == ORDER_EFFECT_CLEAN_STATUS,
            "record_count": len(lane_records),
        }
        families[lane] = family

    combined = (
        dict(combined_assessment)
        if isinstance(combined_assessment, Mapping)
        else _native_order_sensitivity(
            records,
            order_repetitions=order_repetitions,
        )
    )
    combined_clean = combined.get("status") == ORDER_EFFECT_CLEAN_STATUS
    if required_lanes is None:
        claim_lanes = NATIVE_ORDER_EVIDENCE_LANES
    else:
        claim_lanes = tuple(
            lane for lane in required_lanes if lane in NATIVE_EVIDENCE_LANE_SET
        )
    missing_families = [
        lane for lane in claim_lanes if families[lane].get("status") == "not_present"
    ]
    claim_ready = bool(
        combined_clean
        and not missing_families
        and all(families[lane].get("claim_ready") is True for lane in claim_lanes)
        and not unclassified_records
    )
    return {
        "status": combined.get("status"),
        "claim_ready": claim_ready,
        "combined_all_lanes": combined,
        "families": families,
        "present_families": [
            lane for lane in NATIVE_ORDER_EVIDENCE_LANES if by_lane[lane]
        ],
        "required_families": list(claim_lanes),
        "missing_required_families": missing_families,
        "unclassified_record_count": len(unclassified_records),
        "unclassified_evidence_lanes": sorted(
            {
                str(record.get("evidence_lane", "<missing>"))
                for record in unclassified_records
            }
        ),
        "policy": (
            "evidence integrity is independent of claim readiness; canonical "
            "payload/API, supplemental internal-kernel, and supplemental "
            "host-phase lanes are assessed separately, while a combined claim "
            "also requires the simultaneous all-lane assessment to be clean"
        ),
    }


def _core_methodology_failures(
    payload: Mapping[str, object],
    *,
    profiles: set[str],
    repeats: object,
    claim_failures: list[str] | None = None,
) -> list[str]:
    """Validate the balanced Core schedule and simultaneous order-effect gate."""

    failures: list[str] = []
    target_region_ns = payload.get("target_region_ns")
    calibration_target_region_ns = payload.get("calibration_target_region_ns")
    if (
        not isinstance(target_region_ns, int)
        or isinstance(target_region_ns, bool)
        or target_region_ns < CORE_MIN_MEASURED_REGION_NS
    ):
        failures.append("core_native_100ms_target_region_required")
        target_region_ns = None
    if (
        not isinstance(calibration_target_region_ns, int)
        or isinstance(calibration_target_region_ns, bool)
        or calibration_target_region_ns
        < 2
        * (
            target_region_ns
            if isinstance(target_region_ns, int)
            else CORE_MIN_MEASURED_REGION_NS
        )
    ):
        failures.append("core_native_200ms_calibration_headroom_required")
    if payload.get("calibration_policy") != (
        "fixed_loop_count_per_cell_no_per_sample_rescaling"
    ):
        failures.append("core_native_fixed_calibration_policy_required")

    cycles = payload.get("balanced_schedule_cycles")
    pair_repetitions = payload.get("profile_order_metric_rotation_pair_repetitions")
    expected_repeats = (
        cycles * CORE_PROFILE_ORDER_COUNT * CORE_METRIC_ROTATION_COUNT
        if isinstance(cycles, int) and not isinstance(cycles, bool)
        else None
    )
    if (
        not isinstance(cycles, int)
        or isinstance(cycles, bool)
        or cycles < CORE_BALANCED_SCHEDULE_CYCLES
    ):
        failures.append("core_native_five_balanced_schedule_cycles_required")
    if pair_repetitions != cycles:
        failures.append("core_native_pair_repetition_count_mismatch")
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or repeats < 30
        or repeats != expected_repeats
    ):
        failures.append("core_native_balanced_repeat_count_mismatch")
    if payload.get("metric_rotations") != list(range(CORE_METRIC_ROTATION_COUNT)):
        failures.append("core_native_metric_rotations_required")
    if payload.get("all_six_profile_orders_covered") is not True:
        failures.append("core_native_all_profile_orders_marker_required")
    if payload.get("all_profile_order_metric_rotation_pairs_covered") is not True:
        failures.append("core_native_full_order_rotation_cross_marker_required")

    canonical_orders = list(itertools.permutations(PROFILE_ORDER))
    measured_schedule = payload.get("measured_schedule")
    schedule_by_block: dict[int, tuple[int, int, int, tuple[str, ...]]] = {}
    pair_counts: dict[tuple[int, int, int], int] = {}
    if not isinstance(measured_schedule, list) or (
        isinstance(repeats, int) and len(measured_schedule) != repeats
    ):
        failures.append("core_native_measured_schedule_required")
    else:
        for entry in measured_schedule:
            if not isinstance(entry, Mapping):
                failures.append("core_native_measured_schedule_entry_invalid")
                continue
            block_index = entry.get("block_index")
            balanced_cycle = entry.get("balanced_cycle")
            order_index = entry.get("order_index")
            metric_rotation = entry.get("metric_rotation")
            profile_order = entry.get("profile_order")
            valid = (
                isinstance(block_index, int)
                and not isinstance(block_index, bool)
                and isinstance(balanced_cycle, int)
                and not isinstance(balanced_cycle, bool)
                and isinstance(order_index, int)
                and not isinstance(order_index, bool)
                and isinstance(metric_rotation, int)
                and not isinstance(metric_rotation, bool)
                and isinstance(profile_order, list)
                and 0 <= order_index < len(canonical_orders)
                and 0 <= metric_rotation < CORE_METRIC_ROTATION_COUNT
                and isinstance(cycles, int)
                and 0 <= balanced_cycle < cycles
                and tuple(profile_order) == canonical_orders[order_index]
            )
            if not valid:
                failures.append("core_native_measured_schedule_entry_invalid")
                continue
            assert isinstance(block_index, int)
            if block_index in schedule_by_block:
                failures.append("core_native_measured_schedule_duplicate_block")
                continue
            schedule_by_block[block_index] = (
                balanced_cycle,
                order_index,
                metric_rotation,
                tuple(str(item) for item in profile_order),
            )
            pair = (balanced_cycle, order_index, metric_rotation)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if isinstance(repeats, int) and set(schedule_by_block) != set(range(repeats)):
            failures.append("core_native_measured_schedule_block_coverage_incomplete")
        if isinstance(cycles, int) and cycles >= 0:
            expected_pairs = {
                (cycle, order_index, metric_rotation)
                for cycle in range(cycles)
                for order_index in range(CORE_PROFILE_ORDER_COUNT)
                for metric_rotation in range(CORE_METRIC_ROTATION_COUNT)
            }
            if set(pair_counts) != expected_pairs or any(
                count != 1 for count in pair_counts.values()
            ):
                failures.append("core_native_order_rotation_cross_incomplete")

    raw_order = payload.get("raw_order")
    expected_metrics = set(ALL_METRICS)
    expected_raw_keys = {
        (block_index, profile, metric)
        for block_index in schedule_by_block
        for profile in profiles
        for metric in expected_metrics
    }
    observed_raw_keys: set[tuple[int, str, str]] = set()
    if not isinstance(raw_order, list):
        failures.append("core_native_raw_schedule_evidence_required")
    else:
        for observation in raw_order:
            if not isinstance(observation, Mapping):
                failures.append("core_native_raw_schedule_observation_invalid")
                continue
            block_index = observation.get("block_index")
            profile = observation.get("profile")
            metric = observation.get("metric")
            if (
                not isinstance(block_index, int)
                or isinstance(block_index, bool)
                or block_index not in schedule_by_block
                or profile not in profiles
                or metric not in expected_metrics
            ):
                failures.append("core_native_raw_schedule_observation_invalid")
                continue
            cycle, order_index, metric_rotation, profile_order = schedule_by_block[
                block_index
            ]
            expected_position = profile_order.index(str(profile))
            if (
                observation.get("balanced_cycle") != cycle
                or observation.get("order_index") != order_index
                or observation.get("metric_rotation") != metric_rotation
                or tuple(observation.get("profile_order", ())) != profile_order
                or observation.get("position") != expected_position
            ):
                failures.append("core_native_raw_schedule_binding_mismatch")
            duration_ns = observation.get("duration_ns")
            if target_region_ns is not None and (
                not isinstance(duration_ns, int)
                or isinstance(duration_ns, bool)
                or duration_ns < target_region_ns
            ):
                failures.append("core_native_raw_region_below_100ms")
            key = (block_index, str(profile), str(metric))
            if key in observed_raw_keys:
                failures.append("core_native_raw_schedule_duplicate_observation")
            observed_raw_keys.add(key)
        if observed_raw_keys != expected_raw_keys:
            failures.append("core_native_raw_schedule_coverage_incomplete")

    sample_region_gate = payload.get("sample_region_gate")
    under_target_cells = (
        sample_region_gate.get("under_target_cells")
        if isinstance(sample_region_gate, Mapping)
        else None
    )
    if (
        not isinstance(sample_region_gate, Mapping)
        or sample_region_gate.get("minimum_required_ns") != CORE_MIN_MEASURED_REGION_NS
        or not isinstance(under_target_cells, int)
        or isinstance(under_target_cells, bool)
        or under_target_cells != 0
        or sample_region_gate.get("status") != "all_raw_regions_meet_minimum"
    ):
        failures.append("core_native_sample_region_gate_not_clean")

    sensitivity = payload.get("order_sensitivity")
    order_claim_failures = claim_failures if claim_failures is not None else []
    clean_status = (
        "no_order_effect_above_one_percent_with_95_percent_familywise_confidence"
    )
    allowed_statuses = {
        clean_status,
        ORDER_EFFECT_INCONCLUSIVE_STATUS,
        ORDER_EFFECT_CONTAMINATED_STATUS,
    }
    expected_confidence = 1.0 - 0.05 / CORE_ORDER_TOTAL_COMPARISONS
    if not isinstance(sensitivity, Mapping):
        failures.append("core_native_order_sensitivity_required")
    else:
        confirmed_cells = sensitivity.get("confirmed_contamination_cells")
        inconclusive_cells = sensitivity.get("inconclusive_cells")
        sensitivity_status = sensitivity.get("status")
        counts_valid = bool(
            isinstance(confirmed_cells, int)
            and not isinstance(confirmed_cells, bool)
            and confirmed_cells >= 0
            and isinstance(inconclusive_cells, int)
            and not isinstance(inconclusive_cells, bool)
            and inconclusive_cells >= 0
        )
        status_consistent = bool(
            sensitivity_status in allowed_statuses
            and (
                (
                    sensitivity_status == clean_status
                    and confirmed_cells == 0
                    and inconclusive_cells == 0
                )
                or (
                    sensitivity_status == ORDER_EFFECT_INCONCLUSIVE_STATUS
                    and confirmed_cells == 0
                    and isinstance(inconclusive_cells, int)
                    and inconclusive_cells > 0
                )
                or (
                    sensitivity_status == ORDER_EFFECT_CONTAMINATED_STATUS
                    and isinstance(confirmed_cells, int)
                    and confirmed_cells > 0
                )
            )
        )
        if not counts_valid or not status_consistent:
            failures.append("core_native_order_sensitivity_structure_invalid")
        elif sensitivity_status != clean_status:
            order_claim_failures.append("core_native_order_sensitivity_not_claim_ready")
            if sensitivity_status == ORDER_EFFECT_CONTAMINATED_STATUS:
                order_claim_failures.append(
                    "core_native_order_contamination_above_one_percent"
                )
        threshold_percent = sensitivity.get("threshold_percent")
        familywise_confidence = sensitivity.get("familywise_confidence_level")
        if (
            not isinstance(threshold_percent, (int, float))
            or isinstance(threshold_percent, bool)
            or not math.isclose(
                float(threshold_percent), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
            or not isinstance(familywise_confidence, (int, float))
            or isinstance(familywise_confidence, bool)
            or not math.isclose(
                float(familywise_confidence), 0.95, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            failures.append("core_native_order_threshold_contract_mismatch")
        if (
            sensitivity.get("multiple_comparison_correction")
            != "bonferroni_two_sided_across_profile_metric_cells_and_position_pair_contrasts"
            or sensitivity.get("comparison_cells") != CORE_ORDER_COMPARISON_CELLS
            or sensitivity.get("position_pair_contrasts_per_cell")
            != CORE_ORDER_POSITION_PAIR_CONTRASTS
            or sensitivity.get("total_comparisons") != CORE_ORDER_TOTAL_COMPARISONS
            or sensitivity.get("bootstrap_stratification")
            != "whole_balanced_cycle_cluster"
        ):
            failures.append("core_native_order_correction_contract_mismatch")
        bootstrap_resamples = sensitivity.get("bootstrap_resamples")
        corrected_confidence = sensitivity.get(
            "corrected_per_contrast_confidence_level"
        )
        if (
            not isinstance(bootstrap_resamples, int)
            or isinstance(bootstrap_resamples, bool)
            or bootstrap_resamples < CORE_ORDER_BOOTSTRAP_RESAMPLES
            or not isinstance(corrected_confidence, (int, float))
            or isinstance(corrected_confidence, bool)
            or not math.isclose(
                float(corrected_confidence),
                expected_confidence,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            failures.append("core_native_order_confidence_contract_mismatch")
        cells = sensitivity.get("cells")
        declared_cell_count = len(profiles) * len(expected_metrics)
        if not isinstance(cells, list) or len(cells) != declared_cell_count:
            failures.append("core_native_order_cells_incomplete")
        else:
            observed_cells: set[tuple[str, str]] = set()
            expected_pairs = {tuple(pair) for pair in ((0, 1), (0, 2), (1, 2))}
            for cell in cells:
                if not isinstance(cell, Mapping):
                    failures.append("core_native_order_cell_invalid")
                    continue
                profile = cell.get("profile")
                metric = cell.get("metric")
                contrasts = cell.get("position_contrasts")
                position_medians = cell.get("order_position_median_ns")
                observed_spread = cell.get("max_order_position_spread_percent")
                cell_confidence = cell.get("corrected_per_contrast_confidence_level")
                if profile not in profiles or metric not in expected_metrics:
                    failures.append("core_native_order_cell_invalid")
                    continue
                observed_cells.add((str(profile), str(metric)))
                cell_status = cell.get("status")
                if (
                    cell_status not in allowed_statuses
                    or not isinstance(position_medians, list)
                    or len(position_medians) != 3
                    or not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) > 0.0
                        for value in position_medians
                    )
                    or not isinstance(observed_spread, (int, float))
                    or isinstance(observed_spread, bool)
                    or not math.isfinite(float(observed_spread))
                    or float(observed_spread) < 0.0
                    or not isinstance(cell_confidence, (int, float))
                    or isinstance(cell_confidence, bool)
                    or not math.isclose(
                        float(cell_confidence),
                        expected_confidence,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or cell.get("balanced_cycle_cluster_count") != cycles
                    or not isinstance(repeats, int)
                    or isinstance(repeats, bool)
                    or cell.get("observations_per_position") != repeats // 3
                    or not isinstance(contrasts, list)
                    or len(contrasts) != CORE_ORDER_POSITION_PAIR_CONTRASTS
                ):
                    failures.append("core_native_order_cell_structure_invalid")
                    continue
                if cell_status != clean_status:
                    order_claim_failures.append(
                        f"core_native_order_cell_not_claim_ready:{profile}:{metric}"
                    )
                observed_pairs: set[tuple[int, int]] = set()
                for contrast in contrasts:
                    if not isinstance(contrast, Mapping):
                        failures.append("core_native_order_contrast_invalid")
                        continue
                    positions = contrast.get("positions")
                    ci = contrast.get("corrected_bootstrap_ci_percent")
                    observed_signed_percent = contrast.get("observed_signed_percent")
                    contrast_status = contrast.get("status")
                    if (
                        not isinstance(positions, list)
                        or len(positions) != 2
                        or not all(
                            isinstance(value, int) and not isinstance(value, bool)
                            for value in positions
                        )
                        or not isinstance(ci, list)
                        or len(ci) != 2
                        or not all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            for value in ci
                        )
                        or contrast_status not in allowed_statuses
                        or not isinstance(observed_signed_percent, (int, float))
                        or isinstance(observed_signed_percent, bool)
                        or not math.isfinite(float(observed_signed_percent))
                        or float(ci[0]) > float(ci[1])
                        or (
                            contrast_status == clean_status
                            and (float(ci[0]) < -1.0 or float(ci[1]) > 1.0)
                        )
                    ):
                        failures.append("core_native_order_contrast_structure_invalid")
                        continue
                    if contrast_status != clean_status:
                        order_claim_failures.append(
                            "core_native_order_contrast_not_claim_ready:"
                            f"{profile}:{metric}:{positions[0]}-{positions[1]}"
                        )
                    observed_pairs.add((int(positions[0]), int(positions[1])))
                if observed_pairs != expected_pairs:
                    failures.append("core_native_order_contrast_coverage_incomplete")
            if observed_cells != {
                (profile, metric) for profile in profiles for metric in expected_metrics
            }:
                failures.append("core_native_order_cell_coverage_incomplete")

    preemption = payload.get("preemption_observation")
    if (
        not isinstance(preemption, Mapping)
        or preemption.get("status") != "not_used_for_sample_filtering"
        or "portable reliable" not in str(preemption.get("reason", ""))
    ):
        failures.append("core_native_preemption_policy_required")
    return failures


def _native_loop_plan_digest(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned["plan_sha256"] = "0" * 64
    return hashlib.sha256(
        (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def _native_loop_plan_identity(payload: Mapping[str, object]) -> object:
    loop_plan = payload.get("loop_plan")
    if not isinstance(loop_plan, Mapping):
        return None
    semantic = loop_plan.get("semantic_sha256")
    if semantic is None:
        # Preserve comparability for pre-contract synthetic Core fixtures; the
        # strict CUDA/ROCm lane validator below requires the explicit pair.
        semantic = loop_plan.get("sha256")
    file_digest = loop_plan.get("file_sha256")
    if file_digest is None:
        return semantic
    return (semantic, file_digest)


def _native_calibration_binding_view(
    payload: Mapping[str, object],
) -> dict[str, object]:
    view: dict[str, object] = {}
    for key in (
        "backend",
        "variant",
        "source_commit",
        "product_source_commit",
        "harness_source_commit",
        "workload",
        "input_policy",
        "input_identity",
        "device",
        "binary",
        "payload",
        "wheel",
        "source_root",
        "product_source_root",
        "harness_source_root",
        "source_tree_state",
        "product_source_tree_state",
        "harness_source_tree_state",
        "source_blob",
        "harness_source_blob",
        "git",
        "git_identity",
    ):
        if key in payload:
            view[key] = payload[key]
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        view["provenance"] = dict(provenance)
    if "input_identity" in payload:
        view["input_identity"] = payload["input_identity"]
    if "command_line" in payload:
        view["command_line"] = payload["command_line"]
    return view


def _native_calibration_scope_view(
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "backend": payload.get("backend"),
        "workload": payload.get("workload"),
        "input_policy": payload.get("input_policy"),
        "evidence_lane": payload.get("evidence_lane"),
        "artifact_kind": payload.get("artifact_kind"),
        "device": payload.get("device"),
        "scope_id": payload.get("scope_id"),
    }


def _native_calibration_entry_map(
    payload: Mapping[str, object],
) -> tuple[dict[str, int] | None, list[str]]:
    failures: list[str] = []
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return None, ["entries_required"]
    result: dict[str, int] = {}
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            failures.append(f"entry_{index}_invalid")
            continue
        key = item.get("key")
        count = item.get("loop_count")
        if (
            not isinstance(key, str)
            or not key
            or key in result
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
        ):
            failures.append(f"entry_{index}_invalid")
            continue
        result[key] = count
    if payload.get("entry_count") != len(result):
        failures.append("entry_count_mismatch")
    return (result if not failures else None), failures


def _native_calibration_provenance_failures(
    payload: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    for name in ("source_root", "product_source_root", "harness_source_root"):
        value = payload.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            failures.append(f"{name}_required")
    for name in (
        "source_tree_state",
        "product_source_tree_state",
        "harness_source_tree_state",
    ):
        value = payload.get(name)
        if (
            not isinstance(value, Mapping)
            or value.get("status") != "clean"
            or not isinstance(value.get("entries"), list)
        ):
            failures.append(f"{name}_clean_required")
    for name in ("source_blob", "harness_source_blob"):
        value = payload.get(name)
        digest = (
            value.get("source_sha256", value.get("sha256"))
            if isinstance(value, Mapping)
            else None
        )
        if (
            not isinstance(value, Mapping)
            or not isinstance(value.get("relative_path"), str)
            or not value.get("relative_path")
            or ".." in Path(str(value.get("relative_path"))).parts
            or Path(str(value.get("relative_path"))).is_absolute()
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            or not isinstance(value.get("current_git_blob"), str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", str(value.get("current_git_blob")))
            is None
            or not isinstance(value.get("head_git_blob"), str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", str(value.get("head_git_blob"))) is None
            or value.get("current_git_blob") != value.get("head_git_blob")
        ):
            failures.append(f"{name}_tracked_identity_required")
    git = payload.get("git", payload.get("git_identity"))
    if not isinstance(git, Mapping):
        failures.append("git_identity_required")
    else:
        trusted = {
            path.resolve()
            for path in (Path("/usr/bin/git"), Path("/bin/git"))
            if path.is_file()
        }
        raw_path = git.get("path")
        git_path = (
            Path(str(raw_path)).expanduser().resolve()
            if isinstance(raw_path, str)
            else None
        )
        if git_path is None or git_path not in trusted:
            failures.append("trusted_git_executable_required")
        digest = git.get("sha256")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            or git_path is None
            or not git_path.is_file()
            or _sha256(git_path) != digest
        ):
            failures.append("git_executable_sha256_mismatch")
        if not isinstance(git.get("version"), str) or not git.get("version"):
            failures.append("git_executable_version_required")
        removed = git.get("removed_environment")
        if not isinstance(removed, list) or any(
            not isinstance(name, str) or not name.startswith("GIT_") for name in removed
        ):
            failures.append("git_environment_scrub_attestation_required")
        for name in ("git_dir", "git_common_dir"):
            if (
                not isinstance(git.get(name), str)
                or not Path(str(git.get(name))).is_absolute()
            ):
                failures.append(f"{name}_required")
    return failures


def _native_bound_path(
    raw_path: object,
    relative_path: object,
    base: Path,
    evidence_root: Path,
) -> tuple[Path | None, bool]:
    """Resolve an evidence binding without permitting bundle-root escape.

    The absolute path is the live-run identity.  A plan-relative path is only
    a portability fallback when that identity is no longer present in a moved
    evidence directory; an existing absolute path is never replaced after a
    hash failure.  The runner's ``artifacts/../calibration`` sibling layout is
    accepted because it resolves within the explicit manifest evidence root;
    symlink, traversal, and absolute-path escapes are rejected.
    """

    if not isinstance(raw_path, str) or not raw_path:
        return None, False
    root = evidence_root.expanduser().resolve()
    raw = Path(raw_path).expanduser()
    if raw.is_absolute():
        absolute = raw.resolve()
        if not _path_is_below(absolute, root):
            return None, False
        if absolute.is_file():
            return absolute, True
    else:
        relative = (base / raw).resolve()
        if not _path_is_below(relative, root):
            return None, False
        if relative.is_file():
            return relative, False
    if (
        isinstance(relative_path, str)
        and relative_path
        and not Path(relative_path).is_absolute()
    ):
        fallback = (base / relative_path).resolve()
        if not _path_is_below(fallback, root):
            return None, False
        if fallback.is_file():
            return fallback, False
    candidate = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    return (candidate if _path_is_below(candidate, root) else None), False


def _native_loop_plan_failures(
    artifact_path: Path,
    payload: Mapping[str, object],
    backend: str,
    records: Sequence[object],
    *,
    evidence_root: Path | None = None,
) -> list[str]:
    """Authenticate one immutable helper loop plan and every record lookup.

    A native artifact is not comparable merely because it reports normalized
    durations.  The exact calibration key and immutable count are part of the
    semantic cell, and every plan key must be consumed exactly once by the
    artifact's record set.
    """

    if backend not in ("cuda", "rocm"):
        return []
    failures: list[str] = []
    root = (
        evidence_root.expanduser().resolve()
        if evidence_root is not None
        else (
            artifact_path.parent.parent.resolve()
            if artifact_path.parent.name == "artifacts"
            else artifact_path.parent.resolve()
        )
    )
    if not _path_is_below(artifact_path, root):
        failures.append("native_artifact_outside_evidence_root")
    if payload.get("lane_isolation") != (
        "fresh_helper_process_per_variant_trial_and_lane"
    ):
        failures.append("native_loop_plan_lane_isolation_required")
    evidence_lane = payload.get("evidence_lane")
    artifact_kind = payload.get("artifact_kind")
    scope_id = payload.get("scope_id")
    if not isinstance(evidence_lane, str) or not evidence_lane:
        failures.append("native_loop_plan_evidence_lane_required")
    if not isinstance(artifact_kind, str) or not artifact_kind:
        failures.append("native_loop_plan_artifact_kind_required")
    if not isinstance(scope_id, str) or not scope_id:
        failures.append("native_loop_plan_scope_id_required")
    metadata = payload.get("loop_plan")
    if not isinstance(metadata, Mapping):
        return failures + ["native_loop_plan_metadata_required"]
    if metadata.get("mode") != "immutable":
        failures.append("native_loop_plan_immutable_mode_required")
    plan_path_value = metadata.get("path")
    plan_relative_path = metadata.get("relative_path")
    plan_semantic_sha = metadata.get("semantic_sha256")
    plan_file_sha = metadata.get("file_sha256")
    if not isinstance(plan_path_value, str) or not plan_path_value:
        failures.append("native_loop_plan_path_required")
    if (
        not isinstance(plan_relative_path, str)
        or not plan_relative_path
        or Path(plan_relative_path).is_absolute()
    ):
        failures.append("native_loop_plan_relative_path_required")
    if (
        not isinstance(plan_semantic_sha, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", plan_semantic_sha) is None
    ):
        failures.append("native_loop_plan_semantic_sha256_required")
    if (
        not isinstance(plan_file_sha, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", plan_file_sha) is None
    ):
        failures.append("native_loop_plan_file_sha256_required")
    plan_path: Path | None = None
    plan: Mapping[str, object] | None = None
    entry_map: dict[str, int] = {}
    if isinstance(plan_path_value, str) and plan_path_value:
        plan_path, _ = _native_bound_path(
            plan_path_value, plan_relative_path, artifact_path.parent, root
        )
        if plan_path is None:
            failures.append("native_loop_plan_path_outside_evidence_root")
        elif not plan_path.is_file():
            failures.append("native_loop_plan_file_missing")
        else:
            try:
                plan_bytes = plan_path.read_bytes()
                loaded = _strict_json_loads(plan_bytes)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                _DuplicateJsonKeyError,
            ) as exc:
                failures.append(f"native_loop_plan_invalid_json:{exc}")
            else:
                if not isinstance(loaded, Mapping):
                    failures.append("native_loop_plan_root_must_be_object")
                else:
                    plan = loaded
                    if (
                        isinstance(plan_file_sha, str)
                        and hashlib.sha256(plan_bytes).hexdigest() != plan_file_sha
                    ):
                        failures.append("native_loop_plan_file_sha256_mismatch")
                    if plan_bytes != _canonical_json_bytes(plan) + b"\n":
                        failures.append("native_loop_plan_noncanonical_json")
                    if plan.get("schema") != "gafime.native-loop-plan.v1":
                        failures.append("native_loop_plan_schema_mismatch")
                    if plan.get("version") != 1:
                        failures.append("native_loop_plan_version_mismatch")
                    if plan.get("source_count") != 2:
                        failures.append("native_loop_plan_source_count_must_be_two")
                    if plan.get("variants") != ["baseline", "candidate"]:
                        failures.append(
                            "native_loop_plan_baseline_and_candidate_variants_required"
                        )
                    raw_root_commits = plan.get("source_commits")
                    root_commits = (
                        [str(value) for value in raw_root_commits]
                        if isinstance(raw_root_commits, list)
                        else []
                    )
                    if (
                        len(root_commits) != 2
                        or any(
                            re.fullmatch(r"[0-9a-fA-F]{40}", value) is None
                            for value in root_commits
                        )
                        or len(set(root_commits)) != 2
                    ):
                        failures.append(
                            "native_loop_plan_distinct_full_source_commits_required"
                        )
                    if plan.get("policy") != (
                        "max_calibration_count_times_fixed_headroom_factor"
                    ):
                        failures.append("native_loop_plan_policy_mismatch")
                    declared_semantic = plan.get("plan_sha256")
                    if declared_semantic != _native_loop_plan_digest(plan):
                        failures.append("native_loop_plan_digest_mismatch")
                    if declared_semantic != plan_semantic_sha:
                        failures.append("native_loop_plan_semantic_sha256_mismatch")
                    scope = plan.get("scope")
                    if not isinstance(scope, Mapping):
                        failures.append("native_loop_plan_scope_required")
                    else:
                        if scope.get("backend") != backend:
                            failures.append("native_loop_plan_backend_scope_mismatch")
                        if scope.get("input_policy") != payload.get("input_policy"):
                            failures.append("native_loop_plan_input_scope_mismatch")
                        if scope.get("evidence_lane") != evidence_lane:
                            failures.append("native_loop_plan_lane_scope_mismatch")
                        if scope.get("artifact_kind") != artifact_kind:
                            failures.append("native_loop_plan_artifact_scope_mismatch")
                        if scope.get("scope_id") != scope_id:
                            failures.append("native_loop_plan_scope_id_mismatch")
                    entries = plan.get("entries")
                    if not isinstance(entries, list) or not entries:
                        failures.append("native_loop_plan_entries_required")
                    else:
                        for index, item in enumerate(entries):
                            if not isinstance(item, Mapping):
                                failures.append(
                                    f"native_loop_plan_entry_{index}_invalid"
                                )
                                continue
                            key = item.get("key")
                            count = item.get("loop_count")
                            cap = plan.get("max_loop_count")
                            if (
                                not isinstance(key, str)
                                or not key
                                or key in entry_map
                                or not isinstance(count, int)
                                or isinstance(count, bool)
                                or count < 1
                                or (
                                    isinstance(cap, int)
                                    and not isinstance(cap, bool)
                                    and count > cap
                                )
                            ):
                                failures.append(
                                    f"native_loop_plan_entry_{index}_invalid"
                                )
                                continue
                            entry_map[key] = count
                        if plan.get("entry_count") != len(entry_map):
                            failures.append("native_loop_plan_entry_count_mismatch")

                    bindings = plan.get("bindings")
                    calibration_maps: dict[str, dict[str, int]] = {}
                    calibration_payloads: dict[str, Mapping[str, object]] = {}
                    binding_variants: set[str] = set()
                    binding_commits: dict[str, str] = {}
                    if not isinstance(bindings, list) or len(bindings) != 2:
                        failures.append(
                            "native_loop_plan_calibration_bindings_required"
                        )
                    else:
                        for index, binding in enumerate(bindings):
                            if not isinstance(binding, Mapping):
                                failures.append(
                                    f"native_loop_plan_calibration_binding_{index}_invalid"
                                )
                                continue
                            variant = binding.get("variant")
                            if (
                                variant not in ("baseline", "candidate")
                                or variant in binding_variants
                            ):
                                failures.append(
                                    f"native_loop_plan_calibration_binding_{index}_variant_invalid"
                                )
                                continue
                            binding_variants.add(str(variant))
                            binding_commit = binding.get("source_commit")
                            if (
                                not isinstance(binding_commit, str)
                                or re.fullmatch(r"[0-9a-fA-F]{40}", binding_commit)
                                is None
                            ):
                                failures.append(
                                    f"native_loop_plan_calibration_binding_{variant}_source_commit_invalid"
                                )
                            else:
                                binding_commits[str(variant)] = binding_commit
                            if binding.get("product_source_commit") != binding_commit:
                                failures.append(
                                    f"native_loop_plan_calibration_binding_{variant}_product_commit_mismatch"
                                )
                            relative_binding = binding.get("relative_path")
                            if (
                                not isinstance(relative_binding, str)
                                or not relative_binding
                                or Path(relative_binding).is_absolute()
                                or ".." in Path(relative_binding).parts
                            ):
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_relative_path_invalid"
                                )
                                continue
                            binding_path, _ = _native_bound_path(
                                binding.get("path"),
                                binding.get("relative_path"),
                                plan_path.parent,
                                root,
                            )
                            if binding_path is None:
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_path_outside_evidence_root"
                                )
                                continue
                            if not binding_path.is_file():
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_file_missing"
                                )
                                continue
                            binding_digest = binding.get("sha256")
                            if (
                                not isinstance(binding_digest, str)
                                or hashlib.sha256(binding_path.read_bytes()).hexdigest()
                                != binding_digest
                            ):
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_file_sha256_mismatch"
                                )
                                continue
                            try:
                                calibration = _strict_json_loads(
                                    binding_path.read_bytes()
                                )
                            except (
                                OSError,
                                UnicodeDecodeError,
                                json.JSONDecodeError,
                                _DuplicateJsonKeyError,
                            ) as exc:
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_invalid_json:{exc}"
                                )
                                continue
                            if not isinstance(calibration, Mapping):
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_root_invalid"
                                )
                                continue
                            if (
                                calibration.get("schema")
                                != "gafime.native-loop-calibration.v1"
                            ):
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_schema_mismatch"
                                )
                            if calibration.get("status") != "calibration_only":
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_status_mismatch"
                                )
                            failures.extend(
                                f"native_loop_plan_calibration_{variant}_{reason}"
                                for reason in _native_calibration_provenance_failures(
                                    calibration
                                )
                            )
                            for field, expected in _native_calibration_binding_view(
                                calibration
                            ).items():
                                if binding.get(field) != expected:
                                    failures.append(
                                        f"native_loop_plan_calibration_{variant}_binding_{field}_mismatch"
                                    )
                            calibration_scope = _native_calibration_scope_view(
                                calibration
                            )
                            if isinstance(scope, Mapping) and calibration_scope != dict(
                                scope
                            ):
                                failures.append(
                                    f"native_loop_plan_calibration_{variant}_scope_mismatch"
                                )
                            calibration_map, map_failures = (
                                _native_calibration_entry_map(calibration)
                            )
                            failures.extend(
                                f"native_loop_plan_calibration_{variant}_{reason}"
                                for reason in map_failures
                            )
                            if calibration_map is not None:
                                calibration_maps[str(variant)] = calibration_map
                            calibration_payloads[str(variant)] = calibration
                    if binding_variants != {"baseline", "candidate"}:
                        failures.append(
                            "native_loop_plan_calibration_variants_incomplete"
                        )
                    if (
                        set(binding_commits) != {"baseline", "candidate"}
                        or len(set(binding_commits.values())) != 2
                    ):
                        failures.append(
                            "native_loop_plan_binding_source_commits_must_differ"
                        )
                    if len(root_commits) == 2 and set(binding_commits.values()) != set(
                        root_commits
                    ):
                        failures.append(
                            "native_loop_plan_binding_source_commits_root_mismatch"
                        )
                    if set(calibration_maps) == {"baseline", "candidate"}:
                        baseline = calibration_maps["baseline"]
                        candidate = calibration_maps["candidate"]
                        if set(baseline) != set(candidate):
                            failures.append(
                                "native_loop_plan_calibration_key_set_mismatch"
                            )
                        factor = plan.get("headroom_factor")
                        cap = plan.get("max_loop_count")
                        if (
                            isinstance(factor, bool)
                            or not isinstance(factor, int)
                            or factor < 1
                            or isinstance(cap, bool)
                            or not isinstance(cap, int)
                            or cap < 1
                        ):
                            failures.append("native_loop_plan_headroom_factor_invalid")
                        else:
                            if set(entry_map) != set(baseline):
                                failures.append(
                                    "native_loop_plan_entry_calibration_key_set_mismatch"
                                )
                            for key in sorted(set(baseline) & set(candidate)):
                                expected_count = (
                                    max(baseline[key], candidate[key]) * factor
                                )
                                if (
                                    expected_count > cap
                                    or entry_map.get(key) != expected_count
                                ):
                                    failures.append(
                                        "native_loop_plan_entry_not_derived_from_calibration"
                                    )
                                    break
                    elif calibration_payloads:
                        failures.append(
                            "native_loop_plan_calibration_reauthentication_incomplete"
                        )
    used: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        key = record.get("calibration_key")
        if not isinstance(key, str) or not key:
            failures.append(f"record_{index}_native_loop_plan_key_required")
            continue
        if key not in entry_map:
            failures.append(f"record_{index}_native_loop_plan_key_out_of_scope")
            continue
        used.add(key)
        if record.get("loop_count_per_sample") != entry_map[key]:
            failures.append(f"record_{index}_native_loop_plan_count_mismatch")
    unused = sorted(set(entry_map) - used)
    if unused:
        failures.append("native_loop_plan_unused_key:" + unused[0])
    return failures


def _validate_native_artifact(
    path: Path,
    *,
    backend: str,
    kind: str,
    manifest_source_commit: object,
    evidence_root: Path | None = None,
) -> dict[str, object]:
    """Validate the content and coverage of one backend-native artifact.

    The manifest hash protects transport integrity only.  This validator also
    authenticates the artifact's schema, source commit, profile records,
    operation decomposition, raw sample counts, and provenance identities.
    """

    failures: list[str] = []
    claim_failures: list[str] = []
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
    ) as exc:
        return {
            "status": "invalid",
            "complete": False,
            "evidence_integrity_status": "invalid",
            "evidence_integrity_valid": False,
            "performance_claim_ready": False,
            "claim_failures": [],
            "failures": [f"invalid_json:{exc}"],
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "invalid",
            "complete": False,
            "evidence_integrity_status": "invalid",
            "evidence_integrity_valid": False,
            "performance_claim_ready": False,
            "claim_failures": [],
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
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None
    ):
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

    raw_input_policy = payload.get("input_policy", payload.get("input_policy_name"))
    if not isinstance(raw_input_policy, str) or raw_input_policy not in INPUT_POLICIES:
        failures.append("input_policy_required")
    raw_input_identity = payload.get("input_identity", payload.get("dataset_identity"))
    if not isinstance(raw_input_identity, Mapping) or not raw_input_identity:
        failures.append("input_identity_required")
    if backend == "metal":
        failures.extend(
            _metal_input_policy_failures(raw_input_policy, raw_input_identity)
        )
    environment = _environment_mapping(payload.get("environment"))
    if environment is None:
        failures.append("environment_provenance_mapping_or_key_value_list_required")
    process_affinity = _native_process_affinity_view(
        payload.get("process_affinity", payload.get("affinity"))
    )
    if backend in ("core", "cuda", "rocm") and process_affinity is None:
        failures.append("process_affinity_provenance_required")
    command_line = _native_command_line_view(payload.get("command_line"))
    if kind != "native_decomposition" and command_line is None:
        failures.append("native_command_line_required")
    if backend in ("cuda", "rocm", "metal"):
        failures.extend(_gpu_clock_power_failures(backend, payload))
    fixed_gpu_timing_backend = (
        backend
        if (
            (backend == "cuda" and kind in ("cuda_events", "device_events"))
            or (backend == "rocm" and kind in ("rocm_events", "device_events"))
        )
        else None
    )
    gpu_sample_region_target_us: float | None = None
    if fixed_gpu_timing_backend is not None:
        declared_sample_target = payload.get("sample_region_target_us")
        if (
            not isinstance(declared_sample_target, (int, float))
            or isinstance(declared_sample_target, bool)
            or not math.isfinite(float(declared_sample_target))
            or float(declared_sample_target) < GPU_NATIVE_MIN_SAMPLE_REGION_US
        ):
            failures.append(
                f"{fixed_gpu_timing_backend}_native_sample_region_target_us_required"
            )
        else:
            gpu_sample_region_target_us = float(declared_sample_target)
        precondition_batch_limit = payload.get("max_precondition_batch_iterations")
        if (
            not isinstance(precondition_batch_limit, int)
            or isinstance(precondition_batch_limit, bool)
            or precondition_batch_limit < 1
        ):
            failures.append(
                f"{fixed_gpu_timing_backend}_native_precondition_batch_cap_required"
            )
            precondition_batch_limit = None
        precondition_count = payload.get("per_record_untimed_same_cell_preconditions")
        precondition_min_us = payload.get("per_record_untimed_precondition_min_us")
        if (
            not isinstance(precondition_count, int)
            or isinstance(precondition_count, bool)
            or precondition_count < MIN_WARMUPS
        ):
            failures.append(
                f"{fixed_gpu_timing_backend}_native_same_cell_precondition_count_required"
            )
        if (
            not isinstance(precondition_min_us, (int, float))
            or isinstance(precondition_min_us, bool)
            or not math.isfinite(float(precondition_min_us))
            or float(precondition_min_us) < 100_000.0
        ):
            failures.append(
                f"{fixed_gpu_timing_backend}_native_same_cell_precondition_time_floor_required"
            )
        if payload.get("calibration_policy") != (
            "fixed_loop_count_per_cell_no_per_sample_rescaling"
        ):
            failures.append(
                f"{fixed_gpu_timing_backend}_native_fixed_calibration_policy_required"
            )

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
            clock = payload.get("clock", payload.get("timing_clock"))
            if not isinstance(clock, (Mapping, str)) or not clock:
                failures.append("clock_provenance_required")
            clock_power = payload.get("clock_and_power_state")
            before = (
                clock_power.get("before") if isinstance(clock_power, Mapping) else None
            )
            after = (
                clock_power.get("after") if isinstance(clock_power, Mapping) else None
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
        minimum_calls = payload.get("per_sample_untimed_same_cell_preconditions")
        minimum_ns = payload.get("per_sample_untimed_precondition_min_ns")
        if not isinstance(minimum_calls, int) or minimum_calls < MIN_WARMUPS:
            failures.append("core_native_precondition_call_floor_required")
        if (
            not isinstance(minimum_ns, int)
            or minimum_ns < CORE_MIN_UNTIMED_PRECONDITION_NS
        ):
            failures.append("core_native_precondition_time_floor_required")
        raw_order = payload.get("raw_order")
        expected_raw_count = (
            repeats * len(profiles) * len(ALL_METRICS)
            if isinstance(repeats, int)
            else 0
        )
        if not isinstance(raw_order, list) or len(raw_order) != expected_raw_count:
            failures.append("core_native_raw_precondition_evidence_required")
        elif any(
            not isinstance(observation, Mapping)
            or not isinstance(observation.get("precondition_iterations"), int)
            or observation["precondition_iterations"] < MIN_WARMUPS
            or not isinstance(observation.get("precondition_duration_ns"), int)
            or observation["precondition_duration_ns"]
            < CORE_MIN_UNTIMED_PRECONDITION_NS
            for observation in raw_order
        ):
            failures.append("core_native_raw_precondition_floor_not_met")
        failures.extend(
            _core_methodology_failures(
                payload,
                profiles=profiles,
                repeats=repeats,
                claim_failures=claim_failures,
            )
        )

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
    if (
        kind != "native_decomposition"
        and isinstance(command_line, list)
        and command_line
        and isinstance(command_line[0], str)
        and command_line[0]
    ):
        binary_identity = provenance.get("benchmark_binary")
        binary_path = (
            binary_identity.get("path")
            if isinstance(binary_identity, Mapping)
            else None
        )
        try:
            command_binary = Path(command_line[0]).expanduser().resolve(strict=True)
            observed_binary = Path(str(binary_path)).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            failures.append("native_command_binary_unresolvable")
        else:
            if command_binary != observed_binary:
                failures.append("native_command_binary_identity_mismatch")
    failures.extend(
        _native_helper_provenance_failures(
            payload,
            provenance,
            source_commit=source_commit,
            kind=kind,
        )
    )
    if backend == "cuda" and kind in ("cuda_events", "device_events"):
        linked_product = payload.get("linked_direct_kernel_product")
        if not isinstance(linked_product, Mapping):
            failures.append("cuda_linked_direct_kernel_product_required")
        else:
            linked_root = linked_product.get("root")
            product_root = payload.get(
                "product_source_root", payload.get("source_root")
            )
            if (
                not isinstance(linked_root, str)
                or not linked_root
                or (isinstance(product_root, str) and linked_root != product_root)
            ):
                failures.append("cuda_linked_direct_kernel_product_root_mismatch")
            if linked_product.get("commit") != source_commit:
                failures.append("cuda_linked_direct_kernel_product_commit_mismatch")
            for digest_field in (
                "precision_source_sha256",
                "precision_header_sha256",
            ):
                digest = linked_product.get(digest_field)
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    failures.append(
                        f"cuda_linked_direct_kernel_product_{digest_field}_required"
                    )
            if not isinstance(linked_product.get("continuous_unary_available"), bool):
                failures.append(
                    "cuda_linked_direct_kernel_product_capability_marker_required"
                )
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
    lane_contract_failures, artifact_evidence_lane, lane_operations_by_profile = (
        _native_lane_contract_failures(
            payload,
            records,
            backend=backend,
            profiles=profiles,
        )
    )
    failures.extend(lane_contract_failures)
    strict_lane_contract = artifact_evidence_lane is not None
    if backend == "rocm" and strict_lane_contract:
        failures.extend(
            _rocm_direct_kernel_product_failures(
                payload,
                evidence_lane=artifact_evidence_lane,
                source_commit=source_commit,
            )
        )
    if fixed_gpu_timing_backend is not None:
        # The immutable plan contract is activated by the current helper's
        # lane identity marker.  Historical synthetic/transport fixtures that
        # predate this marker remain useful for the older methodology tests;
        # any current helper artifact advertising lane isolation is strict.
        first_record_has_plan_key = bool(
            records
            and isinstance(records[0], Mapping)
            and "calibration_key" in records[0]
        )
        if (
            "loop_plan" in payload
            or "lane_isolation" in payload
            or first_record_has_plan_key
        ):
            failures.extend(
                _native_loop_plan_failures(
                    path,
                    payload,
                    fixed_gpu_timing_backend,
                    records,
                    evidence_root=evidence_root,
                )
            )
        failures.extend(
            _gpu_calibration_prepass_failures(
                fixed_gpu_timing_backend,
                payload,
                records,
                repeats,
            )
        )
    observed_by_profile: dict[str, set[str]] = {profile: set() for profile in profiles}
    native_statistics: list[dict[str, object]] = []
    observed_orders: set[tuple[str, ...]] = set()
    observed_order_indices: set[int] = set()
    native_order_sensitivity: dict[str, object] | None = None
    native_order_claim_families: dict[str, object] | None = None
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
        elif backend == "rocm" and len(profiles) == 3:
            failures.append(f"record_{index}_rocm_profile_order_required")
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
        if backend == "core" and kind == "core_microbenchmark":
            core_samples_ns = record.get("samples_ns")
            core_raw_samples_ns = record.get("raw_samples_ns")
            core_loop_count = record.get("loop_count_per_sample")
            core_raw_samples_valid = (
                isinstance(core_raw_samples_ns, list)
                and isinstance(repeats, int)
                and not isinstance(repeats, bool)
                and len(core_raw_samples_ns) == repeats
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= CORE_MIN_MEASURED_REGION_NS
                    and value <= (1 << 128) - 1
                    for value in core_raw_samples_ns
                )
            )
            core_samples_valid = (
                isinstance(core_samples_ns, list)
                and isinstance(repeats, int)
                and not isinstance(repeats, bool)
                and len(core_samples_ns) == repeats
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0.0
                    for value in core_samples_ns
                )
            )
            core_loop_count_valid = (
                isinstance(core_loop_count, int)
                and not isinstance(core_loop_count, bool)
                and core_loop_count > 0
                and core_loop_count <= (1 << 64) - 1
            )
            if not core_raw_samples_valid:
                failures.append(f"record_{index}_core_raw_region_below_100ms")
            if not core_loop_count_valid:
                failures.append(f"record_{index}_core_fixed_loop_count_required")
            if not core_samples_valid:
                failures.append(f"record_{index}_core_normalized_samples_required")
            if (
                core_raw_samples_valid
                and core_samples_valid
                and core_loop_count_valid
                and any(
                    abs(float(normalized) - (float(raw) / core_loop_count))
                    > max(
                        1.0e-9,
                        4.0 * math.ulp(float(raw) / core_loop_count),
                    )
                    for normalized, raw in zip(
                        core_samples_ns,
                        core_raw_samples_ns,
                    )
                )
            ):
                failures.append(f"record_{index}_core_duration_normalization_mismatch")
            record_target_ns = record.get("sample_region_target_ns")
            record_minimum_ns = record.get("sample_region_min_observed_ns")
            if (
                not isinstance(record_target_ns, int)
                or isinstance(record_target_ns, bool)
                or record_target_ns < CORE_MIN_MEASURED_REGION_NS
                or record_target_ns != payload.get("target_region_ns")
                or record.get("sample_region_target_met") is not True
                or not isinstance(record_minimum_ns, int)
                or isinstance(record_minimum_ns, bool)
                or record_minimum_ns < CORE_MIN_MEASURED_REGION_NS
                or (
                    core_raw_samples_valid
                    and record_minimum_ns != min(core_raw_samples_ns)
                )
            ):
                failures.append(f"record_{index}_core_sample_region_gate_not_clean")
        if fixed_gpu_timing_backend is not None:
            required_sample_target_us = (
                gpu_sample_region_target_us
                if gpu_sample_region_target_us is not None
                else GPU_NATIVE_MIN_SAMPLE_REGION_US
            )
            record_sample_target_us = record.get("sample_region_target_us")
            record_sample_min_us = record.get("sample_region_min_observed_us")
            gpu_raw_samples_us = record.get("raw_samples_us")
            record_target_valid = (
                isinstance(record_sample_target_us, (int, float))
                and not isinstance(record_sample_target_us, bool)
                and math.isfinite(float(record_sample_target_us))
                and float(record_sample_target_us) >= GPU_NATIVE_MIN_SAMPLE_REGION_US
                and (
                    gpu_sample_region_target_us is None
                    or math.isclose(
                        float(record_sample_target_us),
                        gpu_sample_region_target_us,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                )
            )
            if not record_target_valid:
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_sample_region_target_metadata_required"
                )
            if record.get("sample_region_target_met") is not True:
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_sample_region_target_not_met"
                )
            raw_floor_valid = (
                isinstance(gpu_raw_samples_us, list)
                and isinstance(repeats, int)
                and not isinstance(repeats, bool)
                and len(gpu_raw_samples_us) == repeats
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= required_sample_target_us
                    for value in gpu_raw_samples_us
                )
            )
            if not raw_floor_valid:
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_raw_region_below_declared_target"
                )
            observed_raw_minimum = (
                min(float(value) for value in gpu_raw_samples_us)
                if raw_floor_valid and gpu_raw_samples_us
                else None
            )
            if (
                not isinstance(record_sample_min_us, (int, float))
                or isinstance(record_sample_min_us, bool)
                or not math.isfinite(float(record_sample_min_us))
                or float(record_sample_min_us) < required_sample_target_us
                or observed_raw_minimum is None
                or not math.isclose(
                    float(record_sample_min_us),
                    observed_raw_minimum,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                )
            ):
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_sample_region_minimum_invalid"
                )
            precondition_iterations = record.get("precondition_iterations")
            precondition_duration_us = record.get("precondition_duration_us")
            precondition_max_batch = record.get("precondition_max_batch_iterations")
            precondition_clock = record.get("precondition_clock")
            if (
                not isinstance(precondition_iterations, int)
                or isinstance(precondition_iterations, bool)
                or not isinstance(precondition_count, int)
                or precondition_iterations < precondition_count
            ):
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_same_cell_precondition_required"
                )
            if (
                not isinstance(precondition_duration_us, (int, float))
                or isinstance(precondition_duration_us, bool)
                or not math.isfinite(float(precondition_duration_us))
                or not isinstance(precondition_min_us, (int, float))
                or float(precondition_duration_us) < float(precondition_min_us)
            ):
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_same_cell_precondition_floor_not_met"
                )
            if (
                not isinstance(precondition_max_batch, int)
                or isinstance(precondition_max_batch, bool)
                or precondition_max_batch < 1
                or precondition_batch_limit is None
                or precondition_max_batch > precondition_batch_limit
            ):
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_precondition_batch_bound_invalid"
                )
            allowed_precondition_clocks = (
                {"host_steady_clock", "cuda_event_stream"}
                if fixed_gpu_timing_backend == "cuda"
                else {
                    "host_steady_clock",
                    "host_steady_clock_with_hip_synchronization",
                    "hip_event_default_stream",
                }
            )
            if precondition_clock not in allowed_precondition_clocks:
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_precondition_clock_required"
                )
            loop_counts = record.get("loop_counts_per_sample")
            loop_count = record.get("loop_count_per_sample")
            if (
                not isinstance(loop_counts, list)
                or not isinstance(repeats, int)
                or len(loop_counts) != repeats
                or not isinstance(loop_count, int)
                or isinstance(loop_count, bool)
                or loop_count < 1
            ):
                failures.append(
                    f"record_{index}_{fixed_gpu_timing_backend}_fixed_loop_count_required"
                )
            elif loop_counts:
                if (
                    any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 1
                        for value in loop_counts
                    )
                    or len(set(loop_counts)) != 1
                    or loop_counts[0] != loop_count
                ):
                    failures.append(
                        f"record_{index}_{fixed_gpu_timing_backend}_fixed_loop_count_required"
                    )
                elif (
                    raw_floor_valid
                    and isinstance(samples, list)
                    and len(samples) == len(gpu_raw_samples_us)
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) > 0.0
                        for value in samples
                    )
                    and any(
                        abs(float(normalized) - (float(raw) / loop_count))
                        > max(
                            1.0e-9,
                            4.0 * math.ulp(float(raw) / loop_count),
                        )
                        for normalized, raw in zip(samples, gpu_raw_samples_us)
                    )
                ):
                    failures.append(
                        f"record_{index}_{fixed_gpu_timing_backend}_duration_normalization_mismatch"
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
            observed_by_profile.setdefault(record_profile_name, set()).update(
                operations
            )

    boundaries = payload.get("decomposition_boundaries", payload.get("decomposition"))
    if isinstance(boundaries, Mapping) and not strict_lane_contract:
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
    elif not strict_lane_contract:
        failures.append("decomposition_boundaries_required")

    incomplete_profiles: dict[str, list[str]] = {}
    if strict_lane_contract and artifact_evidence_lane is not None:
        # Lane artifacts are intentionally partial.  Their union is checked at
        # manifest readiness; never borrow host/device operations from another
        # artifact while validating this file.
        observed_by_profile = {
            profile: set(lane_operations_by_profile.get(profile, set()))
            for profile in profiles
        }
        required_operations = NATIVE_LANE_REQUIRED_OPERATIONS[artifact_evidence_lane]
        if (
            artifact_evidence_lane == "supplemental_internal_kernel"
            and backend == "rocm"
        ):
            required_operations = required_operations - frozenset(
                ("target_stat_preparation", "feature_stat_preparation")
            )
    else:
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
        order_repetitions = payload.get("order_repetitions")
        order_seed = payload.get("order_seed")
        if payload.get("order_schedule") != "deterministic_per_cycle_shuffle_v1":
            failures.append("native_deterministic_per_cycle_schedule_required")
        if (
            not isinstance(order_seed, int)
            or isinstance(order_seed, bool)
            or order_seed < 0
            or order_seed > 0xFFFF_FFFF_FFFF_FFFF
        ):
            failures.append("native_order_seed_required")
        if (
            not isinstance(order_repetitions, int)
            or isinstance(order_repetitions, bool)
            or order_repetitions < MIN_NATIVE_ORDER_REPETITIONS
        ):
            failures.append("native_order_repetitions_required")
        declared_orders = payload.get("profile_orders")
        declared_order_set = (
            {
                tuple(str(item) for item in order)
                for order in declared_orders
                if isinstance(order, (list, tuple))
            }
            if isinstance(declared_orders, (list, tuple))
            else set()
        )
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
        native_order_sensitivity = _native_order_sensitivity(
            [record for record in records if isinstance(record, Mapping)],
            order_repetitions=order_repetitions,
        )
        native_order_claim_families = _native_order_claim_families(
            [record for record in records if isinstance(record, Mapping)],
            order_repetitions=order_repetitions,
            combined_assessment=native_order_sensitivity,
            required_lanes=(artifact_evidence_lane,)
            if strict_lane_contract and artifact_evidence_lane is not None
            else None,
        )
        recorded_cycles = native_order_sensitivity.get("profile_order_cycles")
        declared_cycles_raw = payload.get("profile_order_cycles")
        declared_cycles: list[list[list[str]]] = []
        declared_cycles_valid = (
            isinstance(order_repetitions, int)
            and not isinstance(order_repetitions, bool)
            and isinstance(declared_cycles_raw, list)
            and len(declared_cycles_raw) == order_repetitions
        )
        if declared_cycles_valid:
            for raw_cycle in declared_cycles_raw:
                if not isinstance(raw_cycle, list) or len(raw_cycle) != len(
                    expected_orders
                ):
                    declared_cycles_valid = False
                    break
                cycle = [
                    [str(value) for value in raw_order]
                    for raw_order in raw_cycle
                    if isinstance(raw_order, (list, tuple))
                ]
                cycle_tuples = {tuple(order) for order in cycle}
                if (
                    len(cycle) != len(expected_orders)
                    or cycle_tuples != expected_orders
                ):
                    declared_cycles_valid = False
                    break
                declared_cycles.append(cycle)
        if not declared_cycles_valid:
            failures.append("native_profile_order_cycles_required")
        else:
            distinct_declared_cycles = {
                tuple(tuple(order) for order in cycle) for cycle in declared_cycles
            }
            if len(distinct_declared_cycles) < 2:
                failures.append("native_profile_order_cycle_variation_required")
            if any(
                left == right for left, right in itertools.pairwise(declared_cycles)
            ):
                failures.append("native_adjacent_profile_order_cycle_reuse_forbidden")
            if recorded_cycles != declared_cycles:
                failures.append("native_declared_record_order_cycles_mismatch")
        failures.extend(
            str(failure)
            for failure in native_order_sensitivity.get(
                "order_cycle_schedule_failures", ()
            )
        )
        if native_order_sensitivity.get("status") in {
            "not_evaluated_order_repetitions_missing",
            "insufficient_order_repeatability",
            "insufficient_complete_order_cycle_evidence",
        }:
            failures.append("native_order_repeatability_cycles_required")
        elif native_order_sensitivity.get("status") != ORDER_EFFECT_CLEAN_STATUS:
            claim_failures.append("native_order_sensitivity_not_claim_ready")
            if (
                native_order_sensitivity.get("status")
                == ORDER_EFFECT_CONTAMINATED_STATUS
            ):
                claim_failures.append("native_order_contamination_above_one_percent")
        if isinstance(native_order_claim_families, Mapping):
            if native_order_claim_families.get("unclassified_record_count"):
                claim_failures.append("native_order_evidence_lane_unclassified")
            raw_families = native_order_claim_families.get("families")
            if isinstance(raw_families, Mapping):
                for lane, assessment in raw_families.items():
                    required_families = native_order_claim_families.get(
                        "required_families", NATIVE_ORDER_EVIDENCE_LANES
                    )
                    if lane not in required_families:
                        continue
                    if (
                        isinstance(assessment, Mapping)
                        and assessment.get("claim_ready") is not True
                    ):
                        claim_failures.append(
                            f"native_order_claim_family_{lane}_not_claim_ready:"
                            f"{assessment.get('status')}"
                        )

    if kind in {"cuda_events", "rocm_events", "device_events"}:
        device_operations = NATIVE_DEVICE_TIMED_OPERATIONS_BY_BACKEND.get(
            backend, frozenset()
        )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            operations = _native_operation_names(
                record.get("operation"), record.get("metric")
            )
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
                sync_tokens = (
                    ("sync", "synchron", "event")
                    if backend == "cuda"
                    else ("sync", "synchron")
                )
                if not any(token in synchronization for token in sync_tokens):
                    failures.append(f"record_{index}_device_synchronization_required")

    execution_mode = str(payload.get("execution_mode", "canonical_payload"))
    canonical_lifecycle_summary: Mapping[str, object] | None = None
    if (
        backend in ("cuda", "rocm", "metal")
        and kind == "native_decomposition"
        and execution_mode != "supplemental_internal_kernel"
    ):
        lifecycle = payload.get("canonical_payload_lifecycle")
        if not isinstance(lifecycle, Mapping) or lifecycle.get("status") != "validated":
            failures.append(
                "native_decomposition_requires_validated_canonical_lifecycle"
            )
        else:
            canonical_lifecycle_summary = lifecycle
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
    supplemental_external_lane = strict_lane_contract and artifact_evidence_lane in (
        "supplemental_internal_kernel",
        "supplemental_host_phase",
    )
    if execution_mode == "supplemental_internal_kernel" or supplemental_external_lane:
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
            # The Metal artifact owns both its supplemental GPU-timestamp lane
            # and the installed-payload lifecycle.  Preserve that lifecycle in
            # the common validation summary so A/B readiness can authenticate
            # the exact ABI surface and common harness on both variants.
            canonical_lifecycle_summary = canonical_lifecycle
        elif (
            not isinstance(canonical_lifecycle, Mapping)
            or canonical_lifecycle.get("status") != "validated"
        ):
            failures.append(
                "supplemental_internal_kernel_requires_canonical_payload_lifecycle"
            )
        elif canonical_lifecycle.get("schema") not in (
            "gafime.native-decomposition.v1",
            "gafime.cuda.native_timing.v2",
        ):
            failures.append("canonical_payload_lifecycle_schema_mismatch")
        elif canonical_lifecycle.get("binding") != "external_canonical_evidence":
            failures.append("canonical_payload_lifecycle_external_binding_required")
        else:
            lifecycle_path = canonical_lifecycle.get("path")
            lifecycle_sha = canonical_lifecycle.get("sha256")
            if not isinstance(lifecycle_path, str) or not isinstance(
                lifecycle_sha, str
            ):
                failures.append("canonical_payload_lifecycle_identity_required")
            else:
                lifecycle_file = Path(lifecycle_path).expanduser()
                if not lifecycle_file.is_absolute():
                    failures.append(
                        "canonical_payload_lifecycle_absolute_path_required"
                    )
                elif not lifecycle_file.is_file():
                    failures.append("canonical_payload_lifecycle_file_missing")
                elif _sha256(lifecycle_file) != lifecycle_sha:
                    failures.append("canonical_payload_lifecycle_sha256_mismatch")
                else:
                    try:
                        lifecycle_payload = _strict_json_loads(
                            lifecycle_file.read_text(encoding="utf-8")
                        )
                    except (
                        OSError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        _DuplicateJsonKeyError,
                    ):
                        lifecycle_payload = None
                    if not isinstance(lifecycle_payload, Mapping):
                        failures.append(
                            "canonical_payload_lifecycle_must_be_json_object"
                        )
                    elif lifecycle_payload.get("status") not in ("pass", "validated"):
                        failures.append("canonical_payload_lifecycle_status_invalid")
                    elif (
                        lifecycle_payload.get("execution_mode")
                        == "supplemental_internal_kernel"
                    ):
                        failures.append(
                            "canonical_payload_lifecycle_must_not_be_supplemental"
                        )
                    if isinstance(lifecycle_payload, Mapping):
                        canonical_lifecycle_summary = lifecycle_payload
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
                    if (
                        not isinstance(lifecycle_source_commit, str)
                        or re.fullmatch(r"[0-9a-fA-F]{40}", lifecycle_source_commit)
                        is None
                    ):
                        failures.append(
                            "canonical_payload_lifecycle_full_source_commit_required"
                        )
                    elif lifecycle_source_commit != source_commit:
                        failures.append(
                            "canonical_payload_lifecycle_source_commit_mismatch"
                        )
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
                        failures.append(
                            "canonical_payload_lifecycle_provenance_required"
                        )
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
                    lifecycle_profiles = (
                        lifecycle_payload.get("profiles")
                        if isinstance(lifecycle_payload, Mapping)
                        else None
                    )
                    if not isinstance(lifecycle_profiles, (list, tuple)) or set(
                        str(profile) for profile in lifecycle_profiles
                    ) != set(BACKEND_PROFILES.get(backend, ())):
                        failures.append(
                            "canonical_payload_lifecycle_profile_coverage_incomplete"
                        )

    canonical_surface = (
        canonical_lifecycle_summary.get("abi_surface")
        if isinstance(canonical_lifecycle_summary, Mapping)
        else None
    )
    canonical_harness_commit = (
        canonical_lifecycle_summary.get("harness_source_commit")
        if isinstance(canonical_lifecycle_summary, Mapping)
        else None
    )
    canonical_harness_identity = None
    if isinstance(canonical_lifecycle_summary, Mapping):
        lifecycle_provenance = canonical_lifecycle_summary.get("provenance")
        if isinstance(lifecycle_provenance, Mapping):
            canonical_harness_identity = lifecycle_provenance.get("harness_source")

    evidence_integrity_valid = not failures
    performance_claim_ready = bool(
        evidence_integrity_valid
        and not claim_failures
        and (
            native_order_claim_families is None
            or native_order_claim_families.get("claim_ready") is True
        )
    )
    return {
        "status": "pass" if not failures else "invalid",
        "complete": not failures,
        "evidence_integrity_status": (
            "valid" if evidence_integrity_valid else "invalid"
        ),
        "evidence_integrity_valid": evidence_integrity_valid,
        "performance_claim_ready": performance_claim_ready,
        "failures": failures,
        "claim_failures": claim_failures,
        "schema": schema,
        "backend": backend,
        "source_commit": source_commit,
        "evidence_lane": artifact_evidence_lane,
        "lane_contract_active": strict_lane_contract,
        "profiles": sorted(profiles),
        "operations_by_profile": {
            profile: sorted(operations)
            for profile, operations in sorted(observed_by_profile.items())
        },
        "operations_by_profile_lane": {
            profile: sorted(operations)
            for profile, operations in sorted(lane_operations_by_profile.items())
        },
        "incomplete_profiles": incomplete_profiles,
        "native_statistics": native_statistics,
        "native_order_sensitivity": _json_safe(native_order_sensitivity),
        "native_order_claim_families": _json_safe(native_order_claim_families),
        "order_repetitions": payload.get("order_repetitions"),
        "observed_profile_orders": [
            list(order)
            for order in sorted(
                record_orders
                if backend in ("cuda", "rocm") and len(profiles) == 3
                else observed_orders
            )
        ],
        "warmups": warmups,
        "repeats": repeats,
        "sample_region_target_us": _json_safe(payload.get("sample_region_target_us")),
        "calibration_prepass": _json_safe(payload.get("calibration_prepass")),
        "execution_mode": execution_mode,
        "abi_surface": canonical_surface,
        "canonical_harness_source_commit": canonical_harness_commit,
        "canonical_harness_source": _json_safe(canonical_harness_identity),
        "native_harness_source_commit": _json_safe(
            payload.get("harness_source_commit")
        ),
        "native_harness_source": _json_safe(provenance.get("harness_source")),
        "native_harness_source_blob": _json_safe(payload.get("harness_source_blob")),
        "native_harness_runner": _json_safe(provenance.get("harness_runner")),
        "native_harness_runner_blob": _json_safe(payload.get("harness_runner_blob")),
        "input_policy": raw_input_policy,
        "input_identity": _json_safe(raw_input_identity),
        "source_root": _json_safe(payload.get("source_root")),
        "source_tree_state": _json_safe(source_tree_state),
        "compiler": _json_safe(payload.get("compiler")),
        "device": _json_safe(payload.get("device")),
        "clock": _json_safe(payload.get("clock", payload.get("timing_clock"))),
        "clock_and_power_state": _json_safe(payload.get("clock_and_power_state")),
        "environment": _json_safe(environment),
        "process_affinity": _json_safe(process_affinity),
        "runner_pid": _json_safe(payload.get("runner_pid")),
        "process_id": _json_safe(payload.get("process_id")),
        "runner_invocation_id": _json_safe(payload.get("runner_invocation_id")),
        "command_line": _json_safe(command_line),
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
                "claim_failures": list(native_evidence.get("claim_failures", ())),
            }
        )
    artifacts = native_evidence.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    by_backend: dict[str, list[str]] = {}
    by_variant_backend: dict[tuple[str, str], list[str]] = {}
    coverage_by_variant_backend: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    validation_by_variant_backend: dict[
        tuple[str, str], list[Mapping[str, object]]
    ] = {}
    lane_artifacts_by_variant_backend: dict[
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
            if validation.get("lane_contract_active") is True:
                lane_artifacts_by_variant_backend.setdefault(
                    (str(artifact.get("variant")), str(artifact.get("backend"))), []
                ).append(artifact)
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
            all_validations = validation_by_variant_backend.get(key, [])
            strict_lane_artifacts = lane_artifacts_by_variant_backend.get(key, [])
            strict_lane_mode = bool(strict_lane_artifacts)
            if strict_lane_mode and len(strict_lane_artifacts) != len(all_validations):
                failures.append(
                    {
                        "variant": variant.name,
                        "backend": backend,
                        "reason": "native_lane_contract_mixed_with_legacy_artifact",
                    }
                )
            required_profiles = set(BACKEND_PROFILES[backend])
            covered_profiles: set[str] = set()
            operations_by_profile: dict[str, set[str]] = {}
            operations_by_lane_profile: dict[tuple[str, str], set[str]] = {}
            for validation in validations:
                covered_profiles.update(
                    str(profile) for profile in validation.get("profiles", ())
                )
                raw_operations = validation.get("operations_by_profile", {})
                if isinstance(raw_operations, Mapping):
                    for profile, operations in raw_operations.items():
                        if isinstance(operations, (list, tuple, set, frozenset)):
                            operations_by_profile.setdefault(
                                str(profile), set()
                            ).update(str(operation) for operation in operations)
                lane = validation.get("evidence_lane")
                lane_operations = validation.get("operations_by_profile_lane", {})
                if lane in NATIVE_EVIDENCE_LANE_SET and isinstance(
                    lane_operations, Mapping
                ):
                    # perf13 emits profile -> operations for one validated
                    # artifact.  Accept the equivalent lane -> profile ->
                    # operations shape from standalone consumers as well, but
                    # never merge two lane namespaces into one profile bucket.
                    nested = lane_operations.get(lane)
                    if isinstance(nested, Mapping):
                        lane_operations = nested
                    for profile, operations in lane_operations.items():
                        if isinstance(operations, (list, tuple, set, frozenset)):
                            operations_by_lane_profile.setdefault(
                                (str(lane), str(profile)), set()
                            ).update(str(operation) for operation in operations)
            missing_profiles = sorted(required_profiles - covered_profiles)
            incomplete_profiles = (
                {}
                if strict_lane_mode
                else {
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
            )
            lane_matrix_failures: list[dict[str, object]] = []
            if strict_lane_mode and backend in ("cuda", "rocm"):
                expected_lanes = tuple(NATIVE_ORDER_EVIDENCE_LANES)
                expected_cells = {
                    (lane, policy, block)
                    for lane in expected_lanes
                    for policy in INPUT_POLICIES
                    for block in (0, 1)
                }
                observed_cells: dict[
                    tuple[str, str, int], list[Mapping[str, object]]
                ] = {}
                seen_paths: set[str] = set()
                seen_hashes: set[str] = set()
                seen_processes: set[tuple[object, ...]] = set()
                for artifact in strict_lane_artifacts:
                    validation = artifact.get("validation")
                    if not isinstance(validation, Mapping):
                        continue
                    lane = validation.get("evidence_lane")
                    policy = validation.get("input_policy")
                    schedule = artifact.get("schedule")
                    schedule = schedule if isinstance(schedule, Mapping) else {}
                    path = str(artifact.get("path", ""))
                    digest = str(artifact.get("sha256", ""))
                    payload: object = {}
                    try:
                        payload = _strict_json_loads(
                            Path(path).read_text(encoding="utf-8")
                        )
                    except (
                        OSError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        _DuplicateJsonKeyError,
                    ):
                        payload = {}
                    scheduled_lane = schedule.get("evidence_lane")
                    if scheduled_lane is not None and scheduled_lane != validation.get(
                        "evidence_lane"
                    ):
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_schedule_lane_mismatch",
                                "path": path,
                                "validated_lane": validation.get("evidence_lane"),
                                "scheduled_lane": scheduled_lane,
                            }
                        )
                    payload_block = (
                        payload.get("ab_block")
                        if isinstance(payload, Mapping)
                        else None
                    )
                    scheduled_block = schedule.get("ab_block")
                    if (
                        "ab_block" in schedule
                        and isinstance(payload, Mapping)
                        and "ab_block" in payload
                        and scheduled_block != payload_block
                    ):
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_schedule_payload_mismatch",
                                "path": path,
                                "field": "ab_block",
                                "scheduled": scheduled_block,
                                "payload": payload_block,
                            }
                        )
                    block = (
                        payload_block
                        if isinstance(payload, Mapping) and "ab_block" in payload
                        else scheduled_block
                    )
                    cell = (str(lane), str(policy), block)
                    if (
                        lane not in expected_lanes
                        or policy not in INPUT_POLICIES
                        or block not in (0, 1)
                    ):
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_cell_invalid",
                                "lane": lane,
                                "input_policy": policy,
                                "ab_block": block,
                                "path": path,
                            }
                        )
                        continue
                    observed_cells.setdefault(cell, []).append(artifact)
                    if path in seen_paths:
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_duplicate_artifact_path",
                                "path": path,
                            }
                        )
                    if digest in seen_hashes:
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_duplicate_artifact_hash",
                                "sha256": digest,
                            }
                        )
                    seen_paths.add(path)
                    seen_hashes.add(digest)
                    process_key = (
                        validation.get("runner_pid"),
                        validation.get("process_id"),
                        validation.get("runner_invocation_id"),
                    )
                    if any(value in (None, "") for value in process_key):
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_process_attestation_missing",
                                "path": path,
                            }
                        )
                    elif process_key in seen_processes:
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_duplicate_process_attestation",
                                "process": list(process_key),
                            }
                        )
                    seen_processes.add(process_key)
                for cell in sorted(expected_cells):
                    observed = observed_cells.get(cell, [])
                    if len(observed) == 0:
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_cell_missing",
                                "lane": cell[0],
                                "input_policy": cell[1],
                                "ab_block": cell[2],
                            }
                        )
                    elif len(observed) != 1:
                        lane_matrix_failures.append(
                            {
                                "reason": "native_lane_matrix_cell_duplicate",
                                "lane": cell[0],
                                "input_policy": cell[1],
                                "ab_block": cell[2],
                                "count": len(observed),
                            }
                        )
                failures.extend(
                    {
                        "variant": variant.name,
                        "backend": backend,
                        **failure,
                    }
                    for failure in lane_matrix_failures
                )
                lane_incomplete_profiles: dict[str, dict[str, list[str]]] = {}
                for lane in expected_lanes:
                    lane_required_operations = NATIVE_LANE_REQUIRED_OPERATIONS[lane]
                    if lane == "supplemental_internal_kernel" and backend == "rocm":
                        lane_required_operations = lane_required_operations - frozenset(
                            ("target_stat_preparation", "feature_stat_preparation")
                        )
                    missing_for_lane = {
                        profile: sorted(
                            lane_required_operations
                            - operations_by_lane_profile.get((lane, profile), set())
                        )
                        for profile in sorted(required_profiles)
                        if lane_required_operations
                        - operations_by_lane_profile.get((lane, profile), set())
                    }
                    if missing_for_lane:
                        lane_incomplete_profiles[lane] = missing_for_lane
                if lane_incomplete_profiles:
                    failures.append(
                        {
                            "variant": variant.name,
                            "backend": backend,
                            "reason": "native_lane_operation_coverage_incomplete",
                            "missing_operations_by_lane_profile": lane_incomplete_profiles,
                        }
                    )
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
            # The product commits are intentionally different in an A/B run,
            # but the standalone lifecycle harness must be the same exact
            # source/hash on both sides.  Do not compare the ABI surface here:
            # the historical baseline is typed while the candidate is generic.
            for field in ("canonical_harness_source_commit",):

                def _harness_value(validation: Mapping[str, object]) -> str | None:
                    value = validation.get(field)
                    if value is None:
                        return None
                    return str(value)

                baseline_values = {
                    value
                    for validation in baseline_validations
                    if (value := _harness_value(validation)) is not None
                }
                candidate_values = {
                    value
                    for validation in candidate_validations
                    if (value := _harness_value(validation)) is not None
                }
                if not baseline_values or not candidate_values:
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "common_canonical_harness_provenance_required",
                            "field": field,
                        }
                    )
                elif baseline_values != candidate_values:
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "baseline_and_candidate_canonical_harness_mismatch",
                            "field": field,
                            "baseline": sorted(baseline_values),
                            "candidate": sorted(candidate_values),
                        }
                    )
            for field in ("native_harness_source_commit",):
                baseline_values = {
                    str(validation[field])
                    for validation in baseline_validations
                    if validation.get(field) is not None
                }
                candidate_values = {
                    str(validation[field])
                    for validation in candidate_validations
                    if validation.get(field) is not None
                }
                if not baseline_values or not candidate_values:
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "common_native_harness_provenance_required",
                            "field": field,
                        }
                    )
                elif baseline_values != candidate_values:
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "baseline_and_candidate_native_harness_mismatch",
                            "field": field,
                            "baseline": sorted(baseline_values),
                            "candidate": sorted(candidate_values),
                        }
                    )
            baseline_harness_hashes = {
                str(identity.get("sha256", "")).lower()
                for validation in baseline_validations
                if isinstance(
                    (identity := validation.get("native_harness_source")), Mapping
                )
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(identity.get("sha256", "")).lower()
                )
            }
            candidate_harness_hashes = {
                str(identity.get("sha256", "")).lower()
                for validation in candidate_validations
                if isinstance(
                    (identity := validation.get("native_harness_source")), Mapping
                )
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(identity.get("sha256", "")).lower()
                )
            }
            if not baseline_harness_hashes or not candidate_harness_hashes:
                failures.append(
                    {
                        "backend": backend,
                        "reason": "common_native_harness_source_sha256_required",
                    }
                )
            elif baseline_harness_hashes != candidate_harness_hashes:
                failures.append(
                    {
                        "backend": backend,
                        "reason": "baseline_and_candidate_native_harness_source_mismatch",
                        "baseline": sorted(baseline_harness_hashes),
                        "candidate": sorted(candidate_harness_hashes),
                    }
                )
            if backend == "core":

                def runner_fingerprint(
                    validation: Mapping[str, object],
                ) -> str | None:
                    identity = _identity_content_fingerprint(
                        validation.get("native_harness_runner")
                    )
                    blob = validation.get("native_harness_runner_blob")
                    if identity is None or not isinstance(blob, Mapping):
                        return None
                    return json.dumps(
                        _json_safe(
                            {
                                "identity": identity,
                                "relative_path": blob.get("relative_path"),
                                "source_sha256": blob.get("source_sha256"),
                                "current_git_blob": blob.get("current_git_blob"),
                                "head_git_blob": blob.get("head_git_blob"),
                            }
                        ),
                        sort_keys=True,
                    )

                baseline_runner_fingerprints = {
                    fingerprint
                    for validation in baseline_validations
                    if (fingerprint := runner_fingerprint(validation)) is not None
                }
                candidate_runner_fingerprints = {
                    fingerprint
                    for validation in candidate_validations
                    if (fingerprint := runner_fingerprint(validation)) is not None
                }
                if (
                    not baseline_runner_fingerprints
                    or not candidate_runner_fingerprints
                ):
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "common_core_harness_runner_identity_required",
                        }
                    )
                elif baseline_runner_fingerprints != candidate_runner_fingerprints:
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "baseline_and_candidate_core_harness_runner_mismatch",
                            "baseline": sorted(baseline_runner_fingerprints),
                            "candidate": sorted(candidate_runner_fingerprints),
                        }
                    )
            for field in (
                "compiler",
                "device",
                "environment",
                "process_affinity",
                "command_line",
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
                elif field == "command_line":
                    baseline_views = [
                        _native_command_line_comparison_view(
                            validation.get("command_line")
                        )
                        for validation in baseline_validations
                    ]
                    candidate_views = [
                        _native_command_line_comparison_view(
                            validation.get("command_line")
                        )
                        for validation in candidate_validations
                    ]
                    if any(view is None for view in baseline_views + candidate_views):
                        failures.append(
                            {
                                "backend": backend,
                                "reason": "native_command_line_provenance_invalid",
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
                if (
                    baseline_values
                    and candidate_values
                    and baseline_values != candidate_values
                ):
                    failures.append(
                        {
                            "backend": backend,
                            "reason": "baseline_and_candidate_native_provenance_mismatch",
                            "field": field,
                            "baseline": sorted(baseline_values),
                            "candidate": sorted(candidate_values),
                        }
                    )
    loop_count_failures = _native_ab_loop_count_failures(
        native_evidence, variants, backends
    )
    failures.extend(loop_count_failures)
    complete = bool(
        native_evidence.get("valid") is True
        and native_evidence.get("arithmetic_claims_valid") is True
        and not failures
    )
    return {
        "complete": complete,
        "failures": failures,
        "loop_count_failures": loop_count_failures,
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
                "lane_operation_profiles": {
                    f"{lane}/{profile}": sorted(operations)
                    for (lane, profile), operations in sorted(
                        {
                            (str(validation.get("evidence_lane")), str(profile)): set(
                                str(operation) for operation in operations
                            )
                            for validation in validations
                            if validation.get("lane_contract_active") is True
                            and isinstance(
                                validation.get("operations_by_profile_lane"), Mapping
                            )
                            for profile, operations in validation[
                                "operations_by_profile_lane"
                            ].items()
                            if isinstance(operations, (list, tuple, set, frozenset))
                        }.items()
                    )
                },
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
                                            "order_repeat": order_repeat,
                                            "order_index": order_index,
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
                                            "order_repeat": order_repeat,
                                            "order_index": order_index,
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
        expected_schedule=schedule,
    )
    cold_summaries = _cold_summaries(
        results, bootstrap_resamples=args.bootstrap_resamples, seed=args.seed
    )
    native_backend_readiness = _native_evidence_backend_readiness(
        native_evidence, args.backends, args.variants, public_results=results
    )
    native_order_sensitivity = tuple(
        validation.get("native_order_sensitivity")
        for artifact in native_evidence.get("artifacts", ())
        if isinstance(artifact, Mapping)
        and isinstance(validation := artifact.get("validation"), Mapping)
        and isinstance(validation.get("native_order_sensitivity"), Mapping)
    )
    native_order_claim_families = tuple(
        validation.get("native_order_claim_families")
        for artifact in native_evidence.get("artifacts", ())
        if isinstance(artifact, Mapping)
        and isinstance(validation := artifact.get("validation"), Mapping)
        and isinstance(validation.get("native_order_claim_families"), Mapping)
    )
    native_ab_schedule_readiness = _native_ab_schedule_readiness(
        native_evidence, args.variants, args.backends
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
        native_order_sensitivity=native_order_sensitivity,
        # perf13 cold envelopes rotate order and diagnose contamination.  The
        # canonical 30-sample lifecycle comparison is produced by
        # cold_lifecycle.py and is intentionally not claimed here.
        require_cold_comparison=False,
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
        e2e_claim_ready and native_backend_readiness["complete"]
    )
    comparative_claim_ready = bool(
        e2e_claim_ready
        and ab_schedule_readiness["complete"]
        and comparative_input_readiness["complete"]
        and bool(ab_comparisons)
    )
    full_claim_ready = bool(
        comparative_claim_ready
        and arithmetic_claim_ready
        and native_ab_schedule_readiness["complete"]
    )
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
        ("native_ab_schedule", native_ab_schedule_readiness),
    ):
        if not readiness["complete"]:
            claim_failures.append({"gate": name, "failures": readiness["failures"]})
    if not arithmetic_claim_ready:
        claim_failures.append(
            {
                "gate": "native_evidence",
                "failures": (
                    list(native_evidence.get("claim_failures", ()))
                    or list(native_evidence.get("failures", ()))
                    or ["validated_manifest_required"]
                ),
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
        "valid_for_canonical_cold_lifecycle_claims": False,
        "claim_failures": claim_failures,
        "methodology": {
            "wall_clock_primary": True,
            "cold_isolation": (
                "perf13 cold samples are fresh-subprocess order-rotation diagnostics; "
                "canonical 30-sample cold lifecycle evidence must come from cold_lifecycle.py"
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
                "diagnostic only: overall clean intervals plus observed phases are emitted, "
                "but perf13 never upgrades them to canonical cold-lifecycle evidence"
            ),
            "native_comparison_key": (
                "backend, workload identity, input policy/identity, profile, operation, "
                "metric, order index/order, clock, synchronization/timing boundary, unit, "
                "A/B block/variant sequence, evidence lane, comparability category, and "
                "loop_count_per_sample and immutable loop-plan SHA-256; cells with "
                "differing plans or fixed loop counts are incomparable and fail the "
                "native A/B claim gate"
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
            "native_order_claim_families": (
                "evidence integrity is reported separately from claim readiness; "
                "canonical payload/API, supplemental internal-kernel, and "
                "supplemental host-phase lanes have independent three-state "
                "order assessments, and a combined claim also requires the "
                "all-lane simultaneous assessment to be clean"
            ),
            "native_core_scope": (
                "Core native arithmetic uses a fixed calibrated region whose every raw "
                "sample must reach 100 ms, five complete 6-order x 4-metric-rotation "
                "cycles, and whole-cycle cluster intervals; it does not claim public "
                "result/report construction"
            ),
            "native_evidence_manifest": (
                "required machine-readable input; validated hash-bound artifacts are "
                "referenced but no native timing is synthesized by this script"
            ),
            "native_ab_isolation": (
                "native manifests must contain fresh helper-process baseline/candidate and "
                "candidate/baseline blocks with exact product, binary, wheel, payload, "
                "harness, environment, workload, and input identities"
            ),
            "order_effect_classification": (
                "public, interleaved, CUDA, and ROCm controls resample complete six-order "
                "cycles and apply joint 95 percent familywise intervals across all cells "
                "and position contrasts; clean requires every upper absolute bound at or "
                "below one percent, a lower bound above one percent proves contamination, "
                "and every boundary-overlapping interval is inconclusive"
            ),
            "regression_classification": (
                "a direction is confirmed only when the independent-bootstrap 95 percent "
                "interval excludes zero; CI-crossing results are inconclusive, and only "
                "confirmed repeatable regressions above one/three percent escalate"
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
        "native_ab_schedule_readiness": native_ab_schedule_readiness,
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
        "native_order_sensitivity": native_order_sensitivity,
        "native_order_claim_families": native_order_claim_families,
        "results": results,
    }
    rendered = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
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
    assert all(
        event == "operation" for event in timing_events[first_clock + 1 : second_clock]
    )
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
    assert sensitivity[0]["status"] == "insufficient_complete_order_cycle_evidence"
    comparisons = _ab_comparisons(synthetic_results, variants)
    assert comparisons[0]["candidate_latency_delta_percent"] == 4.0
    assert comparisons[0]["review_status"] == (
        "confirmed_regression_above_three_percent"
    )
    assert comparisons[0]["escalation"] == "maintainer_approval_required"
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
        runner_path = temp_root / "run-core-benchmark.py"
        binary_path = temp_root / "benchmark-binary"
        payload_path = temp_root / "payload.so"
        wheel_path = temp_root / "gafime.whl"
        for path, contents in (
            (source_path, b"source"),
            (runner_path, b"runner"),
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

        def identity(path: Path) -> dict[str, object]:
            return {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }

        core_repeats = (
            CORE_BALANCED_SCHEDULE_CYCLES
            * CORE_PROFILE_ORDER_COUNT
            * CORE_METRIC_ROTATION_COUNT
        )
        core_orders = list(itertools.permutations(PROFILE_ORDER))
        core_schedule = [
            {
                "block_index": block_index,
                "balanced_cycle": cycle,
                "order_index": order_index,
                "profile_order": list(core_orders[order_index]),
                "metric_rotation": metric_rotation,
            }
            for block_index, (cycle, order_index, metric_rotation) in enumerate(
                (
                    (cycle, order_index, metric_rotation)
                    for cycle in range(CORE_BALANCED_SCHEDULE_CYCLES)
                    for order_index in range(CORE_PROFILE_ORDER_COUNT)
                    for metric_rotation in range(CORE_METRIC_ROTATION_COUNT)
                )
            )
        ]
        core_clean_status = (
            "no_order_effect_above_one_percent_with_95_percent_familywise_confidence"
        )
        core_contrasts = [
            {
                "positions": list(positions),
                "observed_signed_percent": 0.0,
                "corrected_bootstrap_ci_percent": [-0.1, 0.1],
                "status": core_clean_status,
            }
            for positions in ((0, 1), (0, 2), (1, 2))
        ]
        core_sensitivity_cells = [
            {
                "profile": profile,
                "metric": metric,
                "order_position_median_ns": [1.0, 1.0, 1.0],
                "max_order_position_spread_percent": 0.0,
                "position_contrasts": core_contrasts,
                "corrected_per_contrast_confidence_level": 1.0
                - 0.05 / CORE_ORDER_TOTAL_COMPARISONS,
                "balanced_cycle_cluster_count": CORE_BALANCED_SCHEDULE_CYCLES,
                "observations_per_position": core_repeats // 3,
                "status": core_clean_status,
            }
            for profile in PROFILE_ORDER
            for metric in ALL_METRICS
        ]
        core_records = []
        for profile in PROFILE_ORDER:
            for metric in ALL_METRICS:
                core_records.append(
                    {
                        "profile": profile,
                        "operation": "metric_kernel",
                        "metric": metric,
                        "samples_ns": [200_000_000.0] * core_repeats,
                        "raw_samples_ns": [200_000_000] * core_repeats,
                        "loop_count_per_sample": 1,
                        "sample_region_target_ns": CORE_MIN_MEASURED_REGION_NS,
                        "sample_region_min_observed_ns": 200_000_000,
                        "sample_region_target_met": True,
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
                    "product_source_commit": "a" * 40,
                    "product_source_tree_state": {
                        "status": "clean",
                        "entries": [],
                    },
                    "harness_source_commit": "b" * 40,
                    "harness_source_tree_state": {
                        "status": "clean",
                        "entries": [],
                    },
                    "harness_source_blob": {
                        "relative_path": source_path.name,
                        "source_sha256": _sha256(source_path),
                        "current_git_blob": "c" * 40,
                        "head_git_blob": "c" * 40,
                    },
                    "harness_runner_blob": {
                        "relative_path": runner_path.name,
                        "source_sha256": _sha256(runner_path),
                        "current_git_blob": "d" * 40,
                        "head_git_blob": "d" * 40,
                    },
                    "source_tree_state": {"status": "clean", "entries": []},
                    "input_policy": "common-f64",
                    "input_identity": {
                        "matrix_sha256": "1" * 64,
                        "target_sha256": "2" * 64,
                        "feature_names_sha256": "3" * 64,
                    },
                    "warmups": 10,
                    "repeats": core_repeats,
                    "per_sample_untimed_same_cell_preconditions": 10,
                    "per_sample_untimed_precondition_min_ns": 100_000_000,
                    "target_region_ns": CORE_MIN_MEASURED_REGION_NS,
                    "calibration_target_region_ns": 200_000_000,
                    "calibration_policy": (
                        "fixed_loop_count_per_cell_no_per_sample_rescaling"
                    ),
                    "metric_rotations": list(range(CORE_METRIC_ROTATION_COUNT)),
                    "balanced_schedule_cycles": CORE_BALANCED_SCHEDULE_CYCLES,
                    "profile_order_metric_rotation_pair_repetitions": (
                        CORE_BALANCED_SCHEDULE_CYCLES
                    ),
                    "measured_schedule": core_schedule,
                    "all_six_profile_orders_covered": True,
                    "all_profile_order_metric_rotation_pairs_covered": True,
                    "order_sensitivity": {
                        "threshold_percent": 1.0,
                        "maximum_spread_percent": 0.0,
                        "confirmed_contamination_cells": 0,
                        "inconclusive_cells": 0,
                        "bootstrap_resamples": CORE_ORDER_BOOTSTRAP_RESAMPLES,
                        "familywise_confidence_level": 0.95,
                        "multiple_comparison_correction": (
                            "bonferroni_two_sided_across_profile_metric_cells_and_position_pair_contrasts"
                        ),
                        "comparison_cells": CORE_ORDER_COMPARISON_CELLS,
                        "position_pair_contrasts_per_cell": (
                            CORE_ORDER_POSITION_PAIR_CONTRASTS
                        ),
                        "total_comparisons": CORE_ORDER_TOTAL_COMPARISONS,
                        "corrected_per_contrast_confidence_level": 1.0
                        - 0.05 / CORE_ORDER_TOTAL_COMPARISONS,
                        "bootstrap_stratification": "whole_balanced_cycle_cluster",
                        "status": core_clean_status,
                        "cells": core_sensitivity_cells,
                    },
                    "sample_region_gate": {
                        "minimum_required_ns": CORE_MIN_MEASURED_REGION_NS,
                        "under_target_cells": 0,
                        "status": "all_raw_regions_meet_minimum",
                    },
                    "preemption_observation": {
                        "status": "not_used_for_sample_filtering",
                        "reason": (
                            "no portable reliable per-region involuntary-context-switch "
                            "counter; fixed long regions and whole-cycle cluster intervals "
                            "retain scheduler effects"
                        ),
                    },
                    "measurement_scope": "native_arithmetic_only",
                    "decomposition_boundaries": {
                        "candidate_materialization": "included in metric kernels",
                        "report_construction": "not measured by this native arithmetic benchmark",
                    },
                    "compiler": {"rustc": "self-check"},
                    "device": {"kind": "cpu", "identity": "self-check-cpu"},
                    "process_affinity": [0],
                    "command_line": [str(binary_path)],
                    "clock": "std::time::Instant monotonic clock",
                    "clock_and_power_state": {
                        "before": {"cpu_governor": ["performance"]},
                        "after": {"cpu_governor": ["performance"]},
                    },
                    "environment": {},
                    "provenance": {
                        "benchmark_source": identity(source_path),
                        "harness_source": identity(source_path),
                        "harness_runner": identity(runner_path),
                        "benchmark_binary": identity(binary_path),
                        "python_executable": identity(Path(sys.executable)),
                        "wheel": identity(wheel_path),
                    },
                    "records": core_records,
                    "raw_order": [
                        {
                            "profile": profile,
                            "metric": metric,
                            "block_index": block["block_index"],
                            "balanced_cycle": block["balanced_cycle"],
                            "order_index": block["order_index"],
                            "metric_rotation": block["metric_rotation"],
                            "position": block["profile_order"].index(profile),
                            "profile_order": block["profile_order"],
                            "precondition_iterations": 10,
                            "precondition_duration_ns": 100_000_000,
                            "duration_ns": 200_000_000,
                        }
                        for block in core_schedule
                        for profile in PROFILE_ORDER
                        for metric in ALL_METRICS
                    ],
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
        assert (
            _native_evidence_backend_readiness(
                native_evidence, ("core",), (variants[0],)
            )["complete"]
            is True
        )
        # A syntactically valid hash entry with arbitrary JSON is not native
        # evidence: backend schema and complete records are mandatory.
        arbitrary_path = temp_root / "arbitrary.json"
        arbitrary_path.write_text('{"native":true}\n')
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
        assert (
            "supplemental_internal_kernel_requires_canonical_payload_lifecycle"
            in " ".join(native_evidence["failures"])
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
