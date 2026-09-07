#!/usr/bin/env python3
"""Generate the authoritative GAFIME v1 API reference notebook.

This is a repository documentation tool, not part of the ``gafime`` runtime
API.  Keep the generated notebook deterministic so CI can compare it byte for
byte with the tracked artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

try:
    import nbformat
except ImportError as exc:  # pragma: no cover - exercised by the CLI failure path
    raise SystemExit(
        "Generating the v1 API reference requires the documentation development "
        "dependency nbformat>=5.10. Install the repository dev dependencies first."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs" / "notebooks" / "gafime_v1_api_reference.ipynb"

TOP_LEVEL_PUBLIC_API = (
    "__version__",
    "BackendInfo",
    "BackendCapabilities",
    "CapabilityValue",
    "CompiledGafime",
    "CompileFlags",
    "ComputeBudget",
    "Decision",
    "DecisionPathCandidate",
    "DiagnosticReport",
    "EngineConfig",
    "FamilyCapability",
    "FamilySignificanceSupport",
    "dataload",
    "GafimeEngine",
    "GafimeSelector",
    "GafimeStreamer",
    "GafimeV1Error",
    "InteractionResult",
    "NativeCompiledGafime",
    "V1UnsupportedError",
    "available_families",
    "backend_capabilities",
    "compile",
    "family_capability",
    "generate_tutorial",
    "require_family_supported",
    "subfunctions",
)

SEMANTIC_PUBLIC_API = (
    "AcceptedSet",
    "Candidate",
    "CandidateSet",
    "Constraint",
    "Evidence",
    "EvidenceReport",
    "FeatureTable",
    "Graph",
    "Labels",
    "SelectionPolicy",
    "Snapshot",
    "TabularSession",
)

SEMANTIC_METHOD_COVERAGE = {
    "TabularSession": (
        "frame",
        "configured_backend",
        "selected_backend",
        "precision",
        "retained_bytes",
        "capabilities",
        "diagnostics",
        "snapshot",
        "begin_round",
        "source",
        "describe",
        "propose",
        "absolute_difference",
        "softsign",
        "centered_product",
        "evaluate",
        "select",
        "transform",
        "clear_materializations",
        "close",
        "__enter__",
        "__exit__",
    ),
    "CandidateSet": ("__len__", "__getitem__"),
    "AcceptedSet": ("__len__", "__getitem__"),
    "Candidate": (),
    "Constraint": (),
    "SelectionPolicy": (),
    "Snapshot": (
        "rows",
        "feature_names",
        "row_keys",
        "row_domain",
        "provenance",
        "precision",
        "role",
        "labels",
        "graph",
    ),
    "Labels": ("support", "provenance"),
    "Graph": ("edges", "provenance"),
    "Evidence": (
        "reference",
        "paired",
        "labels",
        "graph",
        "rebind_reference",
        "rebind_paired",
        "rebind_labels",
        "rebind_graph",
        "name",
        "semantics",
        "bins",
    ),
    "EvidenceReport": (
        "backend",
        "precision",
        "provenance",
        "candidates",
        "context",
        "value",
        "__arrow_c_array__",
    ),
    "FeatureTable": (
        "feature_names",
        "row_keys",
        "precision",
        "rows",
        "__arrow_c_array__",
    ),
}

DOCUMENTED_DATACLASS_FIELDS = {
    "ComputeBudget": (
        "max_comb_size",
        "max_combinations_per_k",
        "top_features_for_higher_k",
        "max_generated_features",
        "keep_in_vram",
        "vram_budget_mb",
        "max_time_series_candidates",
        "top_k_features_for_time_series",
        "max_feature_candidate",
    ),
    "EngineConfig": (
        "budget",
        "metric_names",
        "num_repeats",
        "permutation_tests",
        "random_seed",
        "stability_std_threshold",
        "permutation_p_threshold",
        "mi_bins",
        "backend",
        "device_id",
        "precision",
        "enable_time_series_functions",
        "time_series_lags",
        "time_series_windows",
        "enable_decision_path_functions",
        "decision_path_max_depth",
        "decision_path_rounds",
        "decision_path_max_paths",
        "decision_path_max_bins",
        "decision_path_min_leaf",
        "decision_path_learning_rate",
        "decision_path_top_k_features",
        "significance_top_n",
        "mi_approximate",
    ),
    "CompileFlags": ("plan", "graph", "export"),
    "CapabilityValue": ("value", "source", "detail"),
    "BackendCapabilities": (
        "configured_backend",
        "selected_backend",
        "selection_status",
        "selection_detail",
        "probe_performed",
        "native_boundary",
        "native_version",
        "families",
        "graph_support",
        "device_significance",
        "host_significance_fallback",
        "permutation_significance",
        "stability_significance",
        "mi_estimator",
        "mi_bin_ceiling",
        "precision_contract",
        "payload_build_policy",
        "arrow_ingest_mode",
        "generated_family_graph_limit",
        "device",
        "probe_details",
    ),
    "BackendInfo": (
        "name",
        "device",
        "is_gpu",
        "memory_total_mb",
        "memory_free_mb",
        "selected_backend",
        "execution_placement",
        "requested_precision",
        "effective_precision",
        "storage_dtype",
        "interaction_arithmetic",
        "reduction_dtype",
        "result_dtype",
        "metric_accumulators",
        "scale_normalization",
        "compensated_summation",
        "interaction_diagnostics_available",
    ),
    "InteractionResult": (
        "combo",
        "feature_names",
        "metrics",
        "family",
        "expression",
        "params",
        "candidate_id",
        "interaction_overflow_rows",
        "interaction_overflow_ratio",
        "source_nonfinite",
        "precision_diagnostics_available",
    ),
    "StabilityResult": (
        "combo",
        "metrics_mean",
        "metrics_std",
        "family",
        "expression",
        "params",
        "candidate_id",
    ),
    "PermutationResult": (
        "combo",
        "p_values",
        "family",
        "expression",
        "params",
        "candidate_id",
    ),
    "Decision": ("signal_detected", "message"),
    "DiagnosticReport": (
        "config",
        "feature_names",
        "interactions",
        "stability",
        "permutations",
        "warnings",
        "decision",
        "backend",
    ),
    "DecisionPathCandidate": (
        "features",
        "thresholds",
        "signs",
        "gain",
        "support",
        "round_id",
        "native_candidate_id",
        "candidate_id",
    ),
    "FamilySignificanceSupport": ("permutation", "stability", "detail"),
    "FamilyCapability": (
        "name",
        "family_id",
        "continuous_input",
        "cpu_kernel",
        "cuda_kernel",
        "rocm_kernel",
        "metal_kernel",
        "python_candidate_loop",
        "generation_placement",
        "scoring_placement",
        "graph_scope",
        "native_compact_scoring",
        "significance_support",
    ),
    "ExportHandles": (
        "backend_name",
        "feature_matrix_handle",
        "result_table_handle",
        "candidate_table_handle",
    ),
    "ChunkRange": ("first_chunk_id", "chunk_count", "chunk_size"),
    "ContinuousArityDescriptor": (
        "arity",
        "feature_start",
        "feature_stop",
        "universe_count",
        "planned_count",
        "offset",
        "chunk_range",
        "saturated",
    ),
    "TimeSeriesDescriptor": (
        "feature_start",
        "feature_stop",
        "lag_count",
        "window_count",
        "template_count",
        "universe_count",
        "planned_count",
        "offset",
        "saturated",
    ),
    "ScenarioPlan": (
        "n_samples",
        "n_features",
        "feature_candidate_count",
        "precision",
        "continuous",
        "time_series",
        "warnings",
        "metric_ids",
    ),
}

PUBLIC_METHOD_COVERAGE = {
    "GafimeEngine": ("analyze", "compile"),
    "NativeCompiledGafime": (
        "flags",
        "backend",
        "scenario_plan",
        "from_engine",
        "graph_replayed",
        "continuous_metric_cache_hits",
        "continuous_metric_cache_builds",
        "candidate_table_cache_hits",
        "exports",
        "analyze",
        "update_target",
        "__arrow_c_array__",
        "export_arrow",
        "close",
    ),
    "BackendCapabilities": ("to_dict",),
    "DiagnosticReport": ("configured_backend", "to_dict"),
    "GafimeSelector": (
        "fit",
        "transform",
        "fit_transform",
        "get_params",
        "set_params",
    ),
    "GafimeStreamer": (
        "total_rows",
        "estimate_optimal_batch_size",
        "stream",
        "stream_with_target",
    ),
    "DecisionPathCandidate": ("combo", "params"),
    "FamilyCapability": ("supported", "scoring_backends", "generation_backend"),
}


def _md(source: str):
    return nbformat.v4.new_markdown_cell(source=dedent(source).strip() + "\n")


def _code(source: str, *, test: str = "syntax"):
    return nbformat.v4.new_code_cell(
        source=dedent(source).strip() + "\n",
        execution_count=None,
        outputs=[],
        metadata={"gafime_test": test},
    )


def _top_level_index() -> str:
    rows = ["| Symbol | Reference role |", "|---|---|"]
    roles = {
        "__version__": "installed runtime identity",
        "BackendInfo": "selected backend and precision result metadata",
        "BackendCapabilities": "structured capability snapshot",
        "CapabilityValue": "capability value plus evidence source",
        "CompiledGafime": "compatibility alias of `NativeCompiledGafime`",
        "CompileFlags": "compiled-plan, graph, and export requests",
        "ComputeBudget": "candidate and resident-memory bounds",
        "Decision": "report-level threshold decision",
        "DecisionPathCandidate": "portable v0.5 compatibility descriptor",
        "DiagnosticReport": "structured analysis report",
        "EngineConfig": "complete execution configuration",
        "FamilyCapability": "family placement record",
        "FamilySignificanceSupport": "family significance record",
        "dataload": "Polars-backed file ingest and analysis",
        "GafimeEngine": "one-shot and compiled execution entry point",
        "GafimeSelector": "scikit-learn-style interaction transformer",
        "GafimeStreamer": "CSV/Parquet batch reader; not an execution engine",
        "GafimeV1Error": "base explicit v1 error",
        "InteractionResult": "one surfaced candidate result",
        "NativeCompiledGafime": "thread-affine compiled artifact",
        "V1UnsupportedError": "unsupported request error",
        "available_families": "enumerate family placement contracts",
        "backend_capabilities": "static or probed backend capability query",
        "compile": "top-level compiled-artifact factory",
        "family_capability": "look up one family contract",
        "generate_tutorial": "generate the compact practice notebook",
        "require_family_supported": "fail-closed family lookup",
        "subfunctions": "advanced compatibility/native helper boundary",
    }
    rows.extend(f"| `gafime.{name}` | {roles[name]} |" for name in TOP_LEVEL_PUBLIC_API)
    return "\n".join(rows)


def _cells() -> list:
    cells = [
        _md(
            """
            # GAFIME v1 Public API Reference & Cookbook

            This is the authoritative long-form reference for the current GAFIME
            v1 public Python surface. It targets the current v1 source tree; mutable
            publication state and exact install identities live in
            [docs/releases/STATUS.md](../releases/STATUS.md), GitHub Releases, and
            PyPI.

            The compact [practice notebook](gafime_tutorial.ipynb) is the guided
            introduction. The old
            [full API notebook](gafime_full_api_reference_notebook.ipynb) is retained
            as historical pre-v1 evidence and is not current guidance. The durable
            coverage inventory is [docs/public-api-coverage.md](../public-api-coverage.md).
            """
        ),
        _md(
            """
            ## 0. How to use this reference

            Read sections 1–8 to configure a first analysis. Sections 9–18 cover
            execution paths, generated families, significance, and reports. Sections
            19–25 cover integrations, operational workflows, failures, and the full
            API index. Code cells marked `core`, `sklearn`, or `polars` in cell
            metadata are bounded documentation smokes; vendor-hardware examples are
            syntax checked but intentionally not executed by ordinary CI.
            """
        ),
        _md(
            """
            ## 1. Installation and package topology

            GAFIME supports CPython 3.10–3.14 with dedicated interpreter-specific
            wheels rather than `abi3`. Install the current published prerelease;
            payload metadata enforces exact Core alignment:

            ```bash
            python -m pip install --pre gafime
            python -m pip install --pre gafime gafime-cuda
            python -m pip install --pre gafime gafime-rocm
            ```

            An unqualified `pip install gafime` prefers the latest stable release;
            it does not select this beta automatically.
            """
        ),
        _md(
            """
            | Surface | Distribution rule | Runtime rule |
            |---|---|---|
            | Core | `gafime` | Rust CPU execution is always present |
            | CUDA | `gafime-cuda` on Linux/Windows x86_64 | exact matching Core; system CUDA 13 runtime |
            | ROCm/HIP | `gafime-rocm`; PyPI sdist plus raw Linux wheel on the GitHub Release | exact matching Core; thin system ROCm 7.2.x runtime |
            | Metal | embedded only in the macOS arm64 Core wheel | no standalone Metal package; fp32 only |

            Core never depends on a GPU payload. Payload packages require the exact
            same Core version. RT/OptiX remains experimental and local-only: it is
            not a release package or artifact.
            """
        ),
        _md(
            """
            ## 2. Core mental model

            Python is the supported declaration and result boundary. Rust validates
            requests, plans candidates, owns scheduling/backend selection and
            lifecycle policy, and runs CPU native/SIMD execution. CUDA, ROCm/HIP,
            and Metal own their native runtime interaction and hot device execution.

            Ordinary users work with `EngineConfig`, `ComputeBudget`,
            `GafimeEngine`, `DiagnosticReport`, and optional integration objects.
            They do not need to call a C ABI, vendor runtime, or Python data-plane
            loop. An explicit backend request never silently becomes another backend.
            """
        ),
        _md(
            """
            The public execution flow is:

            ```text
            Python configuration and data
                       ↓
            Rust validation, planning, scheduling, lifecycle
                       ↓
            Rust Core or selected native GPU payload
                       ↓
            compact structured Python report
            ```

            Production Core scores independent candidates through candidate-level
            Rayon parallelism, with semantics-preserving SIMD/native arithmetic
            inside each candidate. This is topology documentation, not a universal
            performance claim.
            """
        ),
        _md("## 3. Safe imports, version, and deterministic practice data"),
        _code(
            """
            import gafime
            from gafime import ComputeBudget, EngineConfig, GafimeEngine

            print(gafime.__version__)
            assert gafime.__version__
            """,
            test="core",
        ),
        _md(
            """
            These bounded rows plant an interaction while remaining quick enough for
            documentation CI. Python lists are accepted; NumPy guidance appears in
            section 19.
            """
        ),
        _code(
            """
            X = [
                [float(i), float((i * 7) % 11), float((i % 5) - 2)]
                for i in range(48)
            ]
            y = [0.4 * row[0] * row[1] - 0.2 * row[2] for row in X]
            feature_names = ["trend", "cycle", "offset"]
            assert len(X) == len(y) == 48
            """,
            test="core",
        ),
        _md("## 4. Minimal first analysis"),
        _code(
            """
            first_config = EngineConfig(
                backend="core",
                precision="mixed",
                metric_names=("pearson", "r2"),
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=64),
            )
            first_report = GafimeEngine(first_config).analyze(
                X, y, feature_names=feature_names
            )
            print(first_report.backend.selected_backend)
            print(list(first_report.interactions.top_k(5, metric_name="pearson")))
            """,
            test="core",
        ),
        _md(
            """
            `analyze()` validates and owns a native copy in the selected storage
            dtype. It returns a `DiagnosticReport`; it does not add generated columns
            to the caller's matrix. `report.interactions` is a lazy/native-backed
            sequence when the native boundary supplies one. Use `ranked()` or
            `top_k()` instead of materializing the entire table when possible.
            """
        ),
        _md("## 5. `EngineConfig`: every public parameter"),
        _md(
            """
            | Parameter | Default | Meaning and interaction |
            |---|---:|---|
            | `budget` | `ComputeBudget()` | candidate and residency bounds |
            | `metric_names` | all four metrics | non-empty tuple drawn from the supported names |
            | `num_repeats` | `3` | selected-candidate bootstrap repeats; `1` disables bootstrap work |
            | `permutation_tests` | `25` | maxT target permutations; `0` disables permutation work |
            | `random_seed` | `7` | deterministic seed; `None` requests a fresh stream per analysis |
            | `stability_std_threshold` | `0.10` | report decision threshold for conditional bootstrap variability |
            | `permutation_p_threshold` | `0.05` | report decision threshold for maxT p-values |
            | `mi_bins` | `96` | adaptive maximum, rounded down to a sample-safe supported template |
            | `backend` | `"auto"` | `auto`, `core`/`cpu`, `cuda`, `rocm`/`hip`, or `metal` |
            | `device_id` | `0` | non-negative device index for GPU selection/probing |
            | `precision` | `"mixed"` | keyword-only `fp32`, `mixed`, or `fp64` profile |
            | `enable_time_series_functions` | `False` | enable row-order-aware generated temporal columns |
            | `time_series_lags` | `(1,2,4,8,16)` | configured lag/delta/velocity/acceleration offsets |
            | `time_series_windows` | `(4,8,16,32)` | rolling mean/std/sum windows |
            | `enable_decision_path_functions` | `False` | enable native target-dependent path discovery |
            | `decision_path_max_depth` | `2` | maximum discovered path depth |
            | `decision_path_rounds` | `1` | discovery rounds; must be at least one |
            | `decision_path_max_paths` | `32` | bounded number of discovered paths |
            | `decision_path_max_bins` | `0` | split-boundary cap; zero means exhaustive current split search |
            | `decision_path_min_leaf` | `8` | minimum structural leaf support |
            | `decision_path_learning_rate` | `1.0` | residual-discovery update weight |
            | `decision_path_top_k_features` | `50` | unary screening bound used for path discovery |
            | `significance_top_n` | `50` | positive maximum of surfaced candidates evaluated/reported for significance |
            | `mi_approximate` | `False` | Core uses adaptive quantile MI unless fixed-width approximation is requested |

            The first ten historical parameters remain positional through
            `device_id`; current family and precision controls are keyword-only.
            Deprecated `storage_dtype`/`compute_policy` pairs are compatibility-only
            and should not be used in new code.
            """
        ),
        _code(
            """
            from dataclasses import fields

            engine_config_fields = tuple(field.name for field in fields(EngineConfig))
            assert "precision" in engine_config_fields
            assert "significance_top_n" in engine_config_fields
            print(engine_config_fields)
            """,
            test="core",
        ),
        _md(
            """
            Important separations:

            - `significance_top_n` bounds significance/report selection;
              `budget.top_features_for_higher_k` bounds the unary shortlist used to
              generate higher-order candidates.
            - `mi_bins` is a ceiling, not a request to force that exact histogram.
            - `backend="auto"` may select a different eligible backend;
              explicit vendor names never fall back.
            - `random_seed=None` remains stochastic even when a resident or compiled
              artifact reuses its matrix.
            """
        ),
        _code(
            """
            configured = EngineConfig(
                budget=ComputeBudget(
                    max_comb_size=3,
                    max_combinations_per_k=500,
                    top_features_for_higher_k=20,
                ),
                metric_names=("pearson", "spearman", "mutual_info", "r2"),
                backend="core",
                precision="fp64",
                random_seed=11,
                permutation_tests=0,
                num_repeats=1,
            )
            assert configured.precision == "fp64"
            """,
            test="core",
        ),
        _md("## 6. `ComputeBudget` and candidate bounds"),
        _md(
            """
            | Field | Default | Current meaning |
            |---|---:|---|
            | `max_comb_size` | `2` | requested interaction arity ceiling; must be positive and is capped by feature count |
            | `max_combinations_per_k` | `5000` | independent candidate cap for each arity |
            | `top_features_for_higher_k` | `50` | unary shortlist available to arity 2+ planning |
            | `max_generated_features` | `0` | compatibility-reserved; not a current behavioral control |
            | `keep_in_vram` | `True` | permit the bounded resident path for the selected backend |
            | `vram_budget_mb` | `6144` | GPU admission ceiling; zero disables the admission ceiling, not physical-memory limits |
            | `max_time_series_candidates` | `100000` | cap on generated temporal candidates; zero generates none |
            | `top_k_features_for_time_series` | `50` | unary screening bound for temporal generation; zero generates none |
            | `max_feature_candidate` | `None` | `None` uses all; nonnegative values cap the base feature prefix (`0` means none); values below `-1` reject |

            Planning scores unary candidates first, uses a stable unary shortlist for
            higher arities, and applies the per-arity cap. Candidate budgets control
            work and memory; they do not change metric definitions.

            `max_feature_candidate=-1` is power-user mode. With otherwise default
            candidate limits, it retains a practical 1,024-feature guard; changing
            another candidate limit from its default makes the explicitly bounded
            configuration authoritative.
            """
        ),
        _code(
            """
            budget = ComputeBudget(
                max_comb_size=2,
                max_combinations_per_k=128,
                top_features_for_higher_k=12,
                keep_in_vram=False,
                max_feature_candidate=64,
            )
            assert budget.max_generated_features == 0
            assert budget.keep_in_vram is False
            """,
            test="core",
        ),
        _md(
            """
            Arity is candidate structure, not a materialization operator. The engine
            scores its planned product interactions. If you need `add`, `subtract`,
            or guarded `divide` materialization for selected pairs, use
            `GafimeSelector` as described in section 18.
            """
        ),
        _md("## 7. Metrics"),
        _md(
            """
            | Name | Public interpretation | Important limitation |
            |---|---|---|
            | `pearson` | signed linear correlation | non-finite/overflowed reductions remain non-finite; zero variance maps to zero |
            | `spearman` | signed rank correlation | ties and finite-pair handling follow the native contract |
            | `mutual_info` | non-negative dependency score | Core default uses adaptive quantile bins; GPU scoring uses fixed equal-width adaptive templates |
            | `r2` | Pearson correlation squared, clamped to `[0, 1]` | exact zero variance maps to zero; arithmetic failure remains `NaN`; this is not a fitted-model coefficient of determination |

            Ranking treats Pearson and Spearman by magnitude when no metric is named;
            pass `metric_name` explicitly when sign or metric identity matters.
            """
        ),
        _code(
            """
            metric_config = EngineConfig(
                backend="core",
                metric_names=("pearson", "spearman", "mutual_info", "r2"),
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            )
            metric_report = GafimeEngine(metric_config).analyze(X, y, feature_names)
            print(metric_report.interactions[0].metrics)
            """,
            test="core",
        ),
        _md(
            """
            Mutual-information template levels are `2, 4, 8, 12, 16, 24, 32,
            48, 64, 96`, with the sample-density rule `8 * bins^2 <= n_samples`.
            Metal has a 48-bin ceiling. Significance and observed scoring reuse the
            relevant estimator/template contract rather than silently changing MI.
            """
        ),
        _md("## 8. Precision profiles"),
        _md(
            """
            | Profile | Ingest/storage | Pointwise/materialization | Reductions/ranking/results | Backends |
            |---|---|---|---|---|
            | `fp32` | f32 | f32 | f32 | Core, CUDA, ROCm, Metal |
            | `mixed` (default) | f32 | f32 | f64 | Core, CUDA, ROCm |
            | `fp64` | f64 | f64 | f64 | Core, CUDA, ROCm |

            Metal is genuinely fp32-only because MSL has no native shader FP64.
            Explicit Metal `mixed`/`fp64` requests fail before input coercion,
            payload discovery, or allocation. `auto` excludes an ineligible Metal
            route and may select Core.
            """
        ),
        _code(
            """
            import numpy as np

            X_fp32 = np.asarray(X, dtype=np.float32)
            X_fp64 = np.asarray(X, dtype=np.float64)
            assert X_fp32.dtype.name == "float32"
            assert X_fp64.dtype.name == "float64"
            # Choose input dtype to match intent; EngineConfig is still authoritative.
            """,
            test="core",
        ),
        _md(
            """
            Finite values outside the chosen storage range are rejected. NaN and
            infinity that are representable reach native arithmetic and are reported
            according to metric semantics. GAFIME does not expose TF32, fast-math,
            or separate precision packages as public precision choices.
            """
        ),
        _md("## 9. Backend selection and aliases"),
        _md(
            """
            | Public name | Normalized route | Selection behavior |
            |---|---|---|
            | `auto` | ranked resolver | probes eligible configured GPU payloads, then CPU ISA paths |
            | `core`, `cpu` | Rust CPU | explicit and always native; no Python compute fallback |
            | `cuda` | CUDA payload | explicit; missing/incompatible payload or device is an error |
            | `rocm`, `hip` | ROCm/HIP payload | aliases of the same explicit route |
            | `metal` | embedded macOS arm64 payload | explicit, fp32 only |

            `backend="gpu"` is rejected as ambiguous. Runtime implementation aliases
            such as `v1-rust-cpu` may normalize internally, but new user code should
            use the documented names above.
            """
        ),
        _code(
            """
            cpu_alias = EngineConfig(
                backend="cpu", permutation_tests=0, num_repeats=1
            )
            assert cpu_alias.backend == "cpu"
            """,
            test="core",
        ),
        _md(
            """
            Auto selection is capability-driven. A GPU candidate is eligible only
            when its configured library loads, the requested `device_id` validates,
            and ABI capability masks support the precision route. CPU vector ISA is
            then ranked `AVX512 > AVX2 > SSE4.2/NEON > scalar`. Hardware/product
            names alone are not capability proof.
            """
        ),
        _md("## 10. `backend_capabilities` and diagnostics"),
        _code(
            """
            from gafime import backend_capabilities

            caps = backend_capabilities("core", probe=True, precision="mixed")
            print("configured:", caps.configured_backend)
            print("selected:", caps.selected_backend)
            print("status:", caps.selection_status)
            print("precision:", caps.precision_contract.value)
            assert caps.configured_backend == "core"
            assert caps.selected_backend == "core"
            """,
            test="core",
        ),
        _md(
            """
            `configured_backend` is the normalized requested route.
            `selected_backend` is the route validated by the resolver, or `None`.
            `selection_status` and `selection_detail` explain the outcome.
            `probe=False` does not load a GPU payload and leaves runtime-only facts
            unknown. `probe=True` performs identity/device/capability calls but does
            not allocate a feature matrix or run scoring.

            Every `CapabilityValue` has `value`, `source`, and optional `detail`.
            Sources are `runtime`, `package`, `static`, or `unknown`; unknown is not
            a negative hardware claim. Other fields cover graph support,
            significance placement, MI, precision, payload build policy, Arrow
            ingest, family placement, device data, and per-candidate probe details.
            """
        ),
        _md(
            """
            Complete capability schema:

            | Record | Fields |
            |---|---|
            | `CapabilityValue` | `value`, `source`, `detail` |
            | selection | `configured_backend`, `selected_backend`, `selection_status`, `selection_detail`, `probe_performed`, `probe_details` |
            | native identity | `native_boundary`, `native_version` |
            | placement | `families`, `graph_support`, `generated_family_graph_limit`, `device` |
            | significance | `device_significance`, `host_significance_fallback`, `permutation_significance`, `stability_significance` |
            | numerical | `mi_estimator`, `mi_bin_ceiling`, `precision_contract` |
            | packaging/ingest | `payload_build_policy`, `arrow_ingest_mode` |

            `FamilyCapability` fields are `name`, `family_id`, `continuous_input`,
            compatibility scoring aliases `cpu_kernel`/`cuda_kernel`/`rocm_kernel`/
            `metal_kernel`, `python_candidate_loop`, `generation_placement`,
            `scoring_placement`, `graph_scope`, `native_compact_scoring`, and
            `significance_support`. The nested `FamilySignificanceSupport` exposes
            `permutation`, `stability`, and explanatory `detail`.
            """
        ),
        _code(
            """
            caps_dict = caps.to_dict()
            assert caps_dict["configured_backend"] == "core"
            assert caps.precision_contract.source in {"runtime", "static"}
            for family in caps.families:
                print(family.name, family.generation_placement, family.scoring_backends)
            """,
            test="core",
        ),
        _md("## 11. One-shot and resident execution"),
        _code(
            """
            one_shot = GafimeEngine(first_config).analyze(X, y, feature_names)
            assert one_shot.configured_backend == "core"
            """,
            test="core",
        ),
        _md(
            """
            Continuous eager analysis has two lifetime modes. Setting
            `GAFIME_V1_ANALYZE_CACHE_SIZE=0` or `keep_in_vram=False` selects the
            stateless one-shot boundary. A positive cache capacity with
            `keep_in_vram=True` permits a content-keyed, thread-local resident LRU.
            Mutable object identity is never trusted: cache lookup includes selected-
            dtype content identity. The cache is bounded and is not a process-global
            session API.
            """
        ),
        _code(
            """
            # Set this before analyses in a process that requires stateless eager calls.
            # import os
            # os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = "0"
            stateless_config = EngineConfig(
                backend="core",
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(max_comb_size=1, keep_in_vram=False),
            )
            """
        ),
        _md("## 12. Compiled execution and lifecycle"),
        _code(
            """
            from gafime import CompileFlags, compile

            artifact = compile(
                X,
                y,
                feature_names,
                config=first_config,
                flags=CompileFlags(plan=True, graph=False, export=False),
            )
            try:
                compiled_report = artifact.analyze()
                artifact.update_target([value + 0.01 for value in y])
                updated_report = artifact.analyze()
                print(artifact.backend.selected_backend, artifact.scenario_plan)
            finally:
                artifact.close()
            """,
            test="core",
        ),
        _md(
            """
            `gafime.compile(...)` and `GafimeEngine.compile(...)` return the same
            `NativeCompiledGafime` type (`CompiledGafime` is its compatibility alias).
            A compiled artifact owns its coerced matrix, compact plan, backend
            session, and optional graph/export state. It is thread-affine: analyze,
            target update, export, and close must occur on its creation thread.

            There is no context-manager or `run()` API. Use explicit `try/finally` and
            `close()`. `update_target()` keeps features resident and invalidates
            target-dependent plans/caches; decision paths are rediscovered. A native
            failure that closes the underlying state makes the wrapper fail closed.
            """
        ),
        _md(
            """
            `CompileFlags(plan=True)` exposes bounded compatibility plan metadata.
            `graph=True` requires the selected GPU payload to confirm real graph
            replay; it never silently degrades. `export=True` enables
            `export_arrow()` for the compact result table. `__arrow_c_array__()` is
            the corresponding supported Arrow C Data protocol; ordinary application
            code can use the explicit alias. The deprecated `.exports` compatibility
            view is not a new candidate-table API. Metrics and
            candidate identity remain unchanged across one-shot, resident, and
            compiled paths for a fixed seed and configuration.
            """
        ),
        _md("## 13. Continuous candidate family"),
        _code(
            """
            from gafime import (
                available_families,
                family_capability,
                require_family_supported,
            )

            family_names = tuple(family.name for family in available_families())
            continuous = family_capability("continuous")
            assert continuous.generation_placement == "native_continuous"
            assert continuous.supported
            assert require_family_supported("continuous") is continuous
            assert family_names == ("continuous", "decision_path", "time_series")
            print(continuous.scoring_backends)
            """,
            test="core",
        ),
        _md(
            """
            Continuous candidates are products of source/generated feature columns,
            planned by Rust and scored natively. The engine preserves deterministic
            candidate/result placement after parallel Core execution. `combo`,
            `feature_names`, `expression`, `candidate_id`, and `family` identify a
            surfaced candidate. Precision diagnostics distinguish source non-finite
            values from finite-input pointwise overflow when the backend advertises
            that capability.
            """
        ),
        _md("## 14. Decision-path family"),
        _code(
            """
            decision_config = EngineConfig(
                backend="core",
                precision="mixed",
                metric_names=("pearson", "r2"),
                enable_decision_path_functions=True,
                decision_path_max_depth=2,
                decision_path_rounds=1,
                decision_path_max_paths=8,
                decision_path_min_leaf=4,
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=32),
            )
            decision_report = GafimeEngine(decision_config).analyze(
                X, y, feature_names
            )
            print(decision_report.warnings)
            """,
            test="core",
        ),
        _md(
            """
            Rust/Core discovers target-dependent threshold paths and owns scheduling.
            Generated membership columns may then use the selected continuous scorer.
            Native CUDA RT/OptiX remains a local experiment and is not a public
            route or package.

            Decision-path permutation maxT must rediscover paths for every permuted
            target, rebuild the expanded family, and rescan the configured arities.
            Reusing observed target-derived paths would be statistically invalid and
            is forbidden. `DecisionPathCandidate` and the helpers in
            `gafime.decision_path` are compatibility/descriptor utilities; they do
            not create a second execution engine.
            """
        ),
        _md("## 15. Time-series family"),
        _code(
            """
            time_config = EngineConfig(
                backend="core",
                precision="mixed",
                metric_names=("pearson",),
                enable_time_series_functions=True,
                time_series_lags=(1, 2),
                time_series_windows=(4,),
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(
                    max_comb_size=1,
                    max_combinations_per_k=64,
                    max_time_series_candidates=32,
                    top_k_features_for_time_series=3,
                ),
            )
            time_report = GafimeEngine(time_config).analyze(X, y, feature_names)
            print(time_report.warnings)
            """,
            test="core",
        ),
        _md(
            """
            Current templates are lag, delta, velocity, acceleration, rolling mean,
            rolling standard deviation, and rolling sum. They use supplied row order.
            Sort observations first and partition entities outside GAFIME so a lag or
            window never crosses an entity boundary. Generation remains on
            `gafime_cpu`; only the later continuous scoring stage may use an eligible
            GPU backend or graph.

            A lag of zero or a lag greater than/equal to the row count emits no
            temporal feature. A rolling window below two or above the row count also
            emits nothing. Negative values fail the unsigned-integer conversion at
            the native boundary; they are not interpreted as offsets from the end.
            """
        ),
        _md("## 16. Significance, stability, and reproducibility"),
        _code(
            """
            significance_config = EngineConfig(
                backend="core",
                metric_names=("pearson",),
                random_seed=7,
                permutation_tests=3,
                significance_top_n=3,
                num_repeats=2,
                budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            )
            significance_report = GafimeEngine(significance_config).analyze(
                X, y, feature_names
            )
            print(list(significance_report.permutations))
            print(list(significance_report.stability))
            """,
            test="core",
        ),
        _md(
            """
            Permutation p-values use family-wise Westfall–Young maxT. Exceedance uses
            the exact `permuted >= observed` relation. GPU observations keep
            permutation scoring on the observed backend through a native compact
            route or bounded same-device ranking; they do not become a hidden CPU
            permutation loop.

            Bootstrap `metrics_std` resamples an already-selected candidate on the
            same rows. It measures variability conditional on selection. It is not
            out-of-sample evidence, does not correct selection bias/winner's curse,
            and should be complemented by an untouched holdout or nested CV.
            """
        ),
        _md(
            """
            Fixed seeds make one-shot, resident, and compiled runs reproducible under
            the current implementation contract. `random_seed=None` intentionally
            requests a fresh stream for each `analyze()`, including compiled replay.
            Record package/payload versions, configuration, selected backend,
            precision fields, device facts, feature ordering, and seed with results.
            """
        ),
        _md("## 17. Reports, results, ranking, and diagnostics"),
        _code(
            """
            report = first_report
            print(report.configured_backend)
            print(report.backend.selected_backend, report.backend.execution_placement)
            print(report.backend.effective_precision, report.backend.result_dtype)
            best = report.interactions.top_k(2, metric_name="pearson")
            for result in best:
                print(result.candidate_id, result.combo, result.metrics)
            print(report.decision, report.warnings)
            """,
            test="core",
        ),
        _md(
            """
            | Object | Important public fields/operations |
            |---|---|
            | `DiagnosticReport` | `config`, `feature_names`, `interactions`, `stability`, `permutations`, `warnings`, `decision`, `backend`, `configured_backend` |
            | result sequence | iteration, indexing/slicing, `ranked(metric_name=..., descending=..., limit=...)`, `top_k(k, metric_name=...)` |
            | `InteractionResult` | identity, combo/names, metric map, family/expression/params, overflow/non-finite diagnostics |
            | `StabilityResult` | candidate identity plus `metrics_mean` and conditional `metrics_std` |
            | `PermutationResult` | candidate identity plus family-wise maxT `p_values` |
            | `Decision` | `signal_detected`, `message` using configured thresholds |
            | `BackendInfo` | configured/selected placement, memory facts, precision domains, accumulator and diagnostic facts |

            `DiagnosticReport.to_dict()` is deprecated because it materializes the
            native report. Prefer structured fields and bounded ranking. A previously
            returned report remains readable after its compiled artifact closes.
            """
        ),
        _code(
            """
            descending = list(report.interactions.ranked("pearson", limit=3))
            ascending = list(
                report.interactions.ranked("pearson", descending=False, limit=3)
            )
            assert len(descending) <= 3 and len(ascending) <= 3
            """,
            test="core",
        ),
        _md("## 18. scikit-learn integration"),
        _code(
            """
            from gafime import GafimeSelector

            selector = GafimeSelector(
                k=2,
                backend="core",
                metric="pearson",
                operator="multiply",
                precision="mixed",
            )
            augmented = selector.fit_transform(X, y)
            print(selector.top_interactions_)
            assert len(augmented[0]) == len(X[0]) + len(selector.top_interactions_)
            """,
            test="sklearn",
        ),
        _md(
            """
            Supported materialization operators are `multiply`, `add`, `subtract`,
            and guarded `divide`; the divide fallback uses a positive `1e-8`
            denominator in the selected pointwise domain when `abs(denominator)` is
            too small. `n_jobs` and `verbose` are stored compatibility parameters;
            they do not schedule work or enable logging in the current selector.

            `k` must be non-negative and is validated at construction and through
            `set_params()` before discovery or materialization.

            Put `GafimeSelector` inside a scikit-learn `Pipeline` so interaction
            discovery is refit independently in each training fold. Transform adds
            selected pair columns as Python lists; it is an integration boundary,
            not the scalable production data plane.
            """
        ),
        _code(
            """
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline

            pipeline = Pipeline(
                [
                    ("gafime", GafimeSelector(k=1, backend="core", precision="mixed")),
                    ("model", Ridge()),
                ]
            )
            pipeline.fit(X, y)
            """,
            test="sklearn",
        ),
        _md("## 19. Input formats, file ingest, and streaming"),
        _code(
            """
            import numpy as np

            numpy_report = GafimeEngine(first_config).analyze(
                np.asarray(X, dtype=np.float32),
                np.asarray(y, dtype=np.float32),
                feature_names,
            )
            assert len(numpy_report.interactions) == len(first_report.interactions)
            """,
            test="core",
        ),
        _md(
            """
            `GafimeEngine.analyze` accepts rectangular numeric iterables, NumPy
            arrays, Polars DataFrames through their `to_dicts()`/`columns` protocol,
            and pandas-like numeric 2D frames that are NumPy-coercible. Rows must
            have one consistent feature count, target length must match, and feature
            names must align. GAFIME validates values and owns contiguous row-major
            storage after coercion; this is not zero-copy compute memory.

            `gafime.dataload(path, target, features=None, *, config=..., **kwargs)`
            supports Parquet, CSV/TSV/text, and Arrow IPC/Feather through Polars. It
            selects one target column, casts to the profile's resident dtype, rechunks
            to one Arrow record batch, and runs analysis. A raw Arrow table/stream is
            not a top-level `analyze()` input; use the shipped file-oriented
            `dataload()` boundary for Arrow IPC.

            GAFIME v1 deliberately supports `polars>=1.3,<2`. Polars 2 changes API
            and its migration cost is outside the v1 objective; dedicated v1.1 or
            v1.2 work will handle that migration. The upper bound keeps file ingest
            and streaming on the compatibility surface validated for v1.
            """
        ),
        _code(
            """
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import polars as pl
            from gafime import dataload

            with TemporaryDirectory(prefix="gafime-doc-") as temp_dir:
                path = Path(temp_dir) / "sample.arrow"
                frame = pl.DataFrame(
                    {
                        "trend": [row[0] for row in X],
                        "cycle": [row[1] for row in X],
                        "offset": [row[2] for row in X],
                        "target": y,
                    }
                )
                direct_frame_report = GafimeEngine(first_config).analyze(
                    frame.select(feature_names), frame["target"], feature_names
                )
                assert direct_frame_report.feature_names == feature_names
                frame.write_ipc(path)
                loaded_report = dataload(path, "target", config=first_config)
                assert loaded_report.feature_names == feature_names
            """,
            test="polars",
        ),
        _md(
            """
            `GafimeStreamer` is a Polars-backed CSV/Parquet batch reader. It estimates
            a conservative batch size, yields feature rows, and optionally yields a
            target. It does **not** execute `GafimeEngine`, and it does not support
            Arrow IPC. `target_cols` is the retained compatibility name for an
            explicit feature-column list; `y_col` identifies the target used by
            `stream_with_target()`. An explicit `batch_size` must be positive and is
            validated before the reader evaluates the source row count.
            """
        ),
        _code(
            """
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import polars as pl
            from gafime import GafimeStreamer

            with TemporaryDirectory(prefix="gafime-stream-doc-") as temp_dir:
                path = Path(temp_dir) / "sample.csv"
                pl.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "y": [0, 1, 0]}).write_csv(path)
                streamer = GafimeStreamer(path, y_col="y", precision="mixed")
                batches = list(streamer.stream_with_target(batch_size=2))
                assert sum(len(rows) for rows, _ in batches) == 3
            """,
            test="polars",
        ),
        _md("## 20. CLI and installation checks"),
        _md(
            """
            ```bash
            gafime --version
            gafime --check --backend core --precision mixed
            gafime --check --backend auto --device-id 0 --precision mixed
            gafime --check --backend cuda --device-id 0 --precision fp64
            ```

            The check command prints package/native identity, configured and selected
            backend, status, probe details, graph/significance/MI/precision facts,
            payload policy, Arrow ingest, and family placement. It exits zero only
            when the requested route is available. An explicit unavailable vendor
            route exits nonzero and is not replaced by Core.
            """
        ),
        _code(
            """
            import subprocess
            import sys

            version_check = subprocess.run(
                [sys.executable, "-m", "gafime", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            assert f"gafime {gafime.__version__}" in version_check.stdout
            """,
            test="core",
        ),
        _md("## 21. Error handling and fail-closed behavior"),
        _code(
            """
            from gafime import GafimeV1Error, V1UnsupportedError

            try:
                backend_capabilities("metal", probe=True, precision="mixed")
            except ValueError as exc:
                print("expected unsupported precision:", exc)
            else:
                raise AssertionError("Metal mixed precision must fail closed")

            assert issubclass(V1UnsupportedError, GafimeV1Error)
            """,
            test="core",
        ),
        _md(
            """
            Expect `ValueError`/`TypeError` for malformed public configuration or
            input and `V1UnsupportedError` for a well-formed request absent from the
            current runtime. Explicit payload discovery/load/ABI/device failures are
            errors. GAFIME does not silently substitute Python, another GPU vendor,
            or Core for an explicit vendor request.

            Treat warnings as evidence: candidate caps, generated-family limits, and
            numeric diagnostics can change which rows are surfaced without changing
            the metric contract.
            """
        ),
        _md("## 22. Reproducibility and environment reporting"),
        _code(
            """
            environment_record = {
                "gafime_version": gafime.__version__,
                "config": first_report.config,
                "configured_backend": first_report.configured_backend,
                "selected_backend": first_report.backend.selected_backend,
                "execution_placement": first_report.backend.execution_placement,
                "precision": first_report.backend.effective_precision,
                "storage_dtype": first_report.backend.storage_dtype,
                "result_dtype": first_report.backend.result_dtype,
                "seed": first_report.config.random_seed,
            }
            print(environment_record)
            """,
            test="core",
        ),
        _md(
            """
            Also record exact Core/payload distribution versions, capability `source`
            values, device facts, Python/platform identity, data fingerprint, feature
            order, and candidate budget. A capability probe is an environment fact,
            not proof of correctness or performance. Do not generalize shape-specific
            benchmark results into universal speed claims.
            """
        ),
        _md("## 23. Common workflows"),
        _md(
            """
            ### CPU-only

            Install Core, use `backend="core"` for a strict CPU route or `auto` when
            Core fallback is acceptable, choose a precision profile, bound candidate
            work, and inspect `report.backend`.
            """
        ),
        _code(
            """
            cpu_report = GafimeEngine(
                EngineConfig(
                    backend="core",
                    precision="mixed",
                    permutation_tests=0,
                    num_repeats=1,
                    budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=32),
                )
            ).analyze(X, y, feature_names)
            assert cpu_report.backend.selected_backend == "core"
            """,
            test="core",
        ),
        _md(
            """
            ### CUDA

            Install exact matching Core and `gafime-cuda`, provide a compatible
            system CUDA 13 runtime, probe the intended device/profile, then request
            CUDA explicitly. A failed probe is a blocker for that workflow, not
            permission to relabel a Core run as CUDA.
            """
        ),
        _code(
            """
            cuda_caps = backend_capabilities("cuda", 0, probe=True, precision="mixed")
            if cuda_caps.selection_status != "available":
                raise RuntimeError(cuda_caps.selection_detail or "CUDA unavailable")
            cuda_report = GafimeEngine(
                EngineConfig(backend="cuda", device_id=0, precision="mixed")
            ).analyze(X, y, feature_names)
            assert cuda_report.backend.selected_backend == "cuda"
            """
        ),
        _md(
            """
            ### ROCm/HIP

            Install exact matching Core plus the thin ROCm payload appropriate to
            its distribution channel, provide the system ROCm 7.2.x runtime, and
            probe `rocm` or its public `hip` alias. Mixed-runtime coexistence and
            caller-device ownership are reported/fail-closed by the native boundary.
            """
        ),
        _code(
            """
            rocm_caps = backend_capabilities("rocm", 0, probe=True, precision="fp64")
            if rocm_caps.selection_status != "available":
                raise RuntimeError(rocm_caps.selection_detail or "ROCm unavailable")
            rocm_report = GafimeEngine(
                EngineConfig(backend="rocm", device_id=0, precision="fp64")
            ).analyze(X, y, feature_names)
            assert rocm_report.backend.selected_backend == "rocm"
            """
        ),
        _md(
            """
            ### macOS / Metal

            Install plain Core on macOS arm64; its wheel embeds Metal. There is no
            `gafime-metal` distribution. Probe and execute only `precision="fp32"`.
            """
        ),
        _code(
            """
            metal_caps = backend_capabilities("metal", 0, probe=True, precision="fp32")
            if metal_caps.selection_status != "available":
                raise RuntimeError(metal_caps.selection_detail or "Metal unavailable")
            metal_report = GafimeEngine(
                EngineConfig(backend="metal", device_id=0, precision="fp32")
            ).analyze(X, y, feature_names)
            assert metal_report.backend.selected_backend == "metal"
            """
        ),
        _md(
            """
            ### scikit-learn

            Install the current published prerelease `sklearn` extra, place
            `GafimeSelector` inside the pipeline, and let each fold run its own fit.
            See section 18 for the executable bounded example.
            """
        ),
        _md("## 24. Common mistakes and unsupported assumptions"),
        _md(
            """
            - Do not install or import a standalone Metal package; none exists.
            - Do not request `backend="gpu"`; choose `auto` or a vendor route.
            - Do not expect explicit CUDA/ROCm/Metal to fall back to Core.
            - Do not treat `probe=False` unknown fields as negative hardware facts.
            - Do not use `mixed`/`fp64` with explicit Metal.
            - Do not interpret R2 as a generic fitted-model score.
            - Do not interpret bootstrap variability as generalization evidence.
            - Do not let time-series windows cross entities or unsorted time order.
            - Do not build a compiled artifact on one thread and use/close it on another.
            - Do not expect `GafimeStreamer` to execute analysis or read Arrow IPC.
            - Do not expect selector `n_jobs`/`verbose` to schedule/log current work.
            - Do not pass negative selector `k`; it fails validation.
            - Do not pass a non-positive streamer `batch_size`; it fails validation.
            - Do not construct `NativeCompiledGafime` from raw handles; use `compile()`.
            - Do not use deprecated `DiagnosticReport.to_dict()` as a hot data path.
            - Do not treat `max_generated_features` as an active v1 limit.
            - Do not treat RT/OptiX or future Candidate IR/frontends as shipped APIs.
            """
        ),
        _md("## 25. Compatibility surfaces and complete API index"),
        _md(
            """
            The top-level names below are supported or explicitly retained
            compatibility surfaces. “Public” does not mean every constructor should
            be called directly: reports/capabilities are normally returned by
            factories, `NativeCompiledGafime` comes from `compile()`, and
            `subfunctions` is an advanced compatibility/native boundary. The
            per-symbol documentation/coverage disposition is maintained in
            [docs/public-api-coverage.md](../public-api-coverage.md).
            """
        ),
        _md(_top_level_index()),
        _code(
            """
            from gafime import CompiledGafime, DecisionPathCandidate, NativeCompiledGafime

            portable_path = DecisionPathCandidate(
                features=(0, 1),
                thresholds=(3.0, 2.0),
                signs=(-1, 1),
                candidate_id="decision_path:example",
            )
            assert portable_path.combo == (0, 1)
            assert CompiledGafime is NativeCompiledGafime
            assert callable(gafime.generate_tutorial)
            assert gafime.subfunctions.__name__ == "gafime.subfunctions"
            """,
            test="core",
        ),
        _md(
            """
            ### `gafime.semantic`: bounded tabular lifecycle

            `gafime.semantic` is an additive Rust-owned namespace for an explicit
            candidate/evidence/selection lifecycle. Core supplies the complete
            current vocabulary. Explicit CUDA/ROCm sessions can instead negotiate
            a deliberately smaller runtime intersection from a complete optional
            semantic primitive table and the installed Rust lowering; they never
            silently substitute Core. The executable reference stays on Core and
            makes no physical accelerator-execution claim. Metal is explicitly
            unsupported for this product, and `auto` deliberately selects Core for
            the complete vocabulary.

            Separately, the installed-payload semantic lifecycle/parity suite passed
            29/29 hardware-conditional public cases on CUDA device 0 / RTX 4060
            Laptop (`sm89`, driver `610.57.04`) and 29/29 on ROCm device 0 / AMD
            Radeon Graphics (`gfx1150`, runtime `70253211`, system LLVM `21.1.8`).
            Frozen legacy C ABI CMake fixtures separately passed CUDA 11/11 and
            ROCm 10/10. This is configuration-specific correctness evidence, not a
            performance, timing, counter, or general hardware-availability claim;
            the Core-only executable reference itself still proves no GPU execution.

            The namespace is separate from `GafimeEngine.analyze()` and `compile()`:
            it does not inherit legacy metrics, candidate-family, significance,
            ranking, or budget defaults. `EngineConfig` contributes only `backend`,
            `device_id`, and `precision`; a non-default unrelated field is rejected
            rather than silently changing semantic behavior. Backend intent is
            normalized once: `cpu`, `rust`, and `v1-rust-cpu` select Core, while
            `hip` normalizes to ROCm. No Python feature-computation loop is admitted.

            `TabularSession.capabilities` is operation-specific. Core has a static
            complete record; a GPU session reports only its runtime payload/lowering
            intersection and configured/selected device identity. GPU diagnostics
            report backend, retained bytes, and unavailable native work counters;
            they do not invent elapsed time, cache speed, process RSS, Arrow copies,
            or a device probe. The session is synchronous and thread-affine;
            `close()` is terminal and idempotent, including getters. `describe()`
            supplies bounded program metadata and opaque operand handles only,
            never a serializable/native launch descriptor.

            The module exports `TabularSession`, `Snapshot`, `Candidate`,
            `CandidateSet`, `AcceptedSet`, `Evidence`, `Constraint`,
            `SelectionPolicy`, `EvidenceReport`, `Labels`, `Graph`, and
            `FeatureTable`. All are module-scoped. `gafime.semantic` is an explicit
            namespace (`from gafime import semantic` or `import gafime.semantic`),
            intentionally omitted from legacy `gafime.__all__` wildcard imports so
            source-only compatibility imports do not require the native extension.
            Candidate collections and accepted collections are bounded, indexable
            opaque handles rather than serializable program IDs or output-column
            identities.

            A session copies an exact-profile 2-D typed buffer or one Arrow record
            batch into a Rust-owned snapshot. It requires explicit feature names,
            ordered unique row keys, a row domain, and provenance. The caller's
            later mutation does not modify that snapshot. Label and graph
            provenance are caller assertions, not proof that a split is leakage-free
            or that a graph is statistically appropriate.

            Begin a round before declaration. `source()`, `softsign()`,
            `absolute_difference()`, and `centered_product()` make manual bounded
            declarations; `propose(("source", "softsign", "absolute_difference"),
            atoms=None, limit=...)` applies only that closed proposal vocabulary in
            declared operator order and canonical atom order. Collection arguments
            (operators, atoms, candidates, channels, operands, means, row keys,
            labels, and graph edges) are bounded indexed sequences; numeric frames
            are exact-profile typed 2-D buffers or one Arrow record batch. This
            boundary does not execute arbitrary generator callbacks. Centered-product
            means are explicit frozen constants, never fit from later inference rows.
            Passing one `AcceptedSet`, or a bounded indexed sequence of
            `AcceptedSet` values, to a later `begin_round()` is the only way to
            authorize those accepted atoms there. The union preserves each native
            acceptance record; raw `Candidate` handles do not acquire reuse
            authority merely by being placed in a Python collection.

            `Evidence.reference()`, `paired()`, `labels()`, and `graph()` ask
            distinct contextual questions. Reference/labels association is absolute;
            paired-view association retains sign; fixed NMI requires an exact
            supported `bins` value; graph evidence is the uncentered weighted
            edge-energy ratio. `Evidence.rebind_reference()`, `rebind_paired()`,
            `rebind_labels()`, and `rebind_graph()` preserve the measurement kind
            and estimator rather than converting one kind into another. A channel
            is not a universal quality score or an implicit target. Evidence
            evaluation is allowed only on discovery or holdout snapshots; inference
            snapshots are transform-only and reject evidence before native work.

            `SelectionPolicy` has one named maximize/minimize primary plus
            inclusive `Constraint` bounds in the individual channel units. Missing
            evidence is explicit (`reject`/`error`, with `ignore` only for an
            optional constraint); the policy is not a weighted score or Pareto
            optimizer. `EvidenceReport.value(candidate, channel)` returns
            `state`, `value`, `support`, and `reason`. `EvidenceReport.context`
            retains the original row/channel declarations, but it is not proof of
            split independence, label provenance, graph validity, or leakage
            safety. Report and feature-table
            `__arrow_c_array__()` directly expose Arrow C Data rather than a Python
            row loop. `FeatureTable` retains row keys and profile-native values;
            its buffers remain valid after the session closes.

            Calls are synchronous and thread-affine. `close()` is terminal and
            idempotent. Cross-session/stale handles, incompatible snapshot contexts,
            and selection from a prior round fail closed. See the full
            [tabular semantic product contract](../v1.1-tabular-semantic-product.md)
            for the bounded support boundary.
            """
        ),
        _code(
            """
            from array import array
            from gafime import semantic

            semantic_storage = array(
                "f", [0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0]
            )
            semantic_matrix = memoryview(semantic_storage).cast("B").cast(
                "f", shape=(4, 2)
            )
            semantic_keys = [101, 102, 103, 104]
            semantic_session = semantic.TabularSession(
                semantic_matrix,
                feature_names=["left", "right"],
                row_keys=semantic_keys,
                row_domain="reference-demo",
                provenance="reference-demo-input",
            )
            try:
                assert semantic_session.configured_backend == "auto"
                assert semantic_session.selected_backend == "core"
                assert semantic_session.precision == "mixed"
                semantic_capabilities = semantic_session.capabilities
                assert semantic_capabilities["selected_backend"] == "core"
                assert semantic_capabilities["configured_device_id"] == 0
                assert semantic_capabilities["selected_device_id"] is None
                assert "centered_product" in semantic_capabilities["programs"]
                assert semantic_session.begin_round() == 1

                semantic_left = semantic_session.source("left")
                assert semantic_session.describe(semantic_left)["operation"] == "source"
                semantic_proposed = semantic_session.propose(
                    ("source", "softsign", "absolute_difference"), limit=8
                )
                assert 1 <= len(semantic_proposed) <= 8

                semantic_labels = semantic_session.frame.labels(
                    row_keys=semantic_keys,
                    values=[0.0, 1.0, 2.0, 3.0],
                    provenance="reference-demo-labels",
                )
                semantic_channel = semantic.Evidence.labels(
                    "outcome", semantic_labels, statistic="pearson"
                )
                semantic_report = semantic_session.evaluate(
                    [semantic_left], [semantic_channel]
                )
                assert semantic_report.context["role"] == "discovery"
                semantic_value = semantic_report.value(semantic_left, semantic_channel)
                assert semantic_value["state"] == "measured"
                assert semantic_value["support"] == 4
                assert semantic_session.diagnostics["evidence_kernel_calls"] >= 1
                semantic_accepted = semantic_session.select(
                    semantic_report,
                    semantic.SelectionPolicy(
                        semantic_channel, direction="maximize", limit=1
                    ),
                )
                assert len(semantic_accepted) == 1

                inference_storage = array("f", [9.0, 19.0])
                inference_matrix = memoryview(inference_storage).cast("B").cast(
                    "f", shape=(1, 2)
                )
                inference = semantic_session.snapshot(
                    inference_matrix,
                    feature_names=["left", "right"],
                    row_keys=[9001],
                    row_domain="inference-demo",
                    provenance="reference-demo-inference",
                )
                semantic_features = semantic_session.transform(
                    semantic_accepted, inference
                )
                assert semantic_features.row_keys == [9001]
                assert semantic_features.rows == 1
                assert len(semantic_features.__arrow_c_array__()) == 2
            finally:
                semantic_session.close()
            """,
            test="semantic",
        ),
        _md(
            """
            Additional explicit module surfaces:

            - `gafime.reporting`: `BackendInfo`, `Decision`, `DiagnosticReport`,
              `InteractionResult`, `StabilityResult`, `PermutationResult`, plus
              advanced `NativeContinuousInteractions` and compatibility
              `NativeReportBuilder`.
            - `gafime.decision_path`: descriptor conversion, description,
              evaluation, scoring, and feature-name helpers around
              `DecisionPathCandidate`.
            - `gafime.io`: `GafimeStreamer`, compatibility `create_streamer`, and
              bounded diagnostic helper `benchmark_streaming`. Its `n_batches`
              argument must be positive and is validated before the file is opened.
            - `gafime.compile`: callable compile module, flags, and compiled aliases;
              `gafime.compile.scenario` exposes bounded compatibility plan metadata;
              `gafime.compile.exports` exposes deprecated handle compatibility.
            - `gafime.semantic`: Rust-owned `TabularSession` lifecycle plus opaque
              candidates/accepted values, contextual evidence, explicit selection,
              and Arrow-output result objects. It is additive: Core owns the
              complete vocabulary while explicit CUDA/ROCm may expose a bounded
              negotiated subset without Core substitution. It is not a generic
              candidate IR or Python data plane.
            - `gafime.subfunctions`: the advanced native compatibility proxy. Prefer
              the top-level safe engine/capability APIs for new application code.

            Public method/property index: `GafimeEngine.analyze()` and `compile()`;
            `NativeCompiledGafime.flags`, `backend`, `scenario_plan`, `from_engine()`,
            `graph_replayed`, `continuous_metric_cache_hits`,
            `continuous_metric_cache_builds`, `candidate_table_cache_hits`, deprecated
            `exports`, `analyze()`, `update_target()`, `__arrow_c_array__()`,
            `export_arrow()`, and `close()`; `BackendCapabilities.to_dict()`;
            `DiagnosticReport.configured_backend` and deprecated `to_dict()`;
            `GafimeSelector.fit()`, `transform()`, `fit_transform()`, `get_params()`,
            and `set_params()`; `GafimeStreamer.total_rows`,
            `estimate_optimal_batch_size()`, `stream()`, and `stream_with_target()`;
            `DecisionPathCandidate.combo` and `params()`; and
            `FamilyCapability.supported`, `scoring_backends`, and
            `generation_backend`. Semantic public methods/properties are
            `TabularSession.frame`, `configured_backend`, `selected_backend`,
            `precision`, `retained_bytes`, `capabilities`, `diagnostics`,
            `snapshot()`, `begin_round()`, `source()`, `describe()`, `propose()`,
            `absolute_difference()`, `softsign()`,
            `centered_product()`, `evaluate()`, `select()`, `transform()`,
            `clear_materializations()`, `close()`, `__enter__()`, and `__exit__()`;
            `Snapshot.rows`, `feature_names`, `row_keys`, `row_domain`,
            `provenance`, `precision`, `role`, `labels()`, and `graph()`;
            `Evidence.reference()`, `paired()`, `labels()`, `graph()`, all four
            `rebind_*()` forms, `name`, `semantics`, and `bins`; collection
            `__len__()`/`__getitem__()`; `EvidenceReport.backend`, `precision`,
            `provenance`, `candidates`, `context`, `value()`, and
            `__arrow_c_array__()`; and
            `FeatureTable.feature_names`, `row_keys`, `precision`, `rows`, and
            `__arrow_c_array__()`. `Labels.support`/`provenance` and
            `Graph.edges`/`provenance` are contextual result metadata.

            CLI entry points are `gafime` and `python -m gafime`. No future
            Candidate IR/decorator/JIT surface is part of the current API.
            """
        ),
    ]
    return cells


def _merge_adjacent_markdown(cells: list) -> list:
    """Keep the long reference readable without one prose cell per paragraph."""

    merged = []
    for cell in cells:
        if (
            merged
            and cell["cell_type"] == "markdown"
            and merged[-1]["cell_type"] == "markdown"
        ):
            merged[-1]["source"] = (
                str(merged[-1]["source"]).rstrip()
                + "\n\n"
                + str(cell["source"]).lstrip()
            )
        else:
            merged.append(cell)
    return merged


def build_notebook():
    cells = _merge_adjacent_markdown(_cells())
    for index, cell in enumerate(cells):
        cell["id"] = f"gafime-v1-{index:03d}"
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "gafime_reference": {
                "purpose": "Authoritative current v1 public API reference and cookbook",
                "release_scope": "GAFIME v1 public API",
                "generator": "docs/notebooks/generate_v1_api_reference.py",
                "coverage": "docs/public-api-coverage.md",
                "cells": len(cells),
                "top_level_symbols": len(TOP_LEVEL_PUBLIC_API),
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def render_notebook() -> str:
    return nbformat.writes(build_notebook(), version=4)


def generate(output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_notebook(), encoding="utf-8")
    return output


def check(output: Path = DEFAULT_OUTPUT) -> None:
    if not output.is_file():
        raise SystemExit(f"missing generated notebook: {output}")
    expected = render_notebook()
    actual = output.read_text(encoding="utf-8")
    if actual != expected:
        try:
            display_path = output.relative_to(ROOT)
        except ValueError:
            display_path = output
        raise SystemExit(
            f"{display_path} differs from its deterministic generator; "
            "run docs/notebooks/generate_v1_api_reference.py"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.check:
        check(output)
    else:
        generate(output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
