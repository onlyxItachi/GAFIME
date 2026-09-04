# GAFIME v1 Public API Documentation Coverage

This is the authoritative coverage inventory for the current GAFIME v1 Python
surface. It is checked against the installed package and the generated
[v1 API reference](notebooks/gafime_v1_api_reference.ipynb). “Public” includes
explicit compatibility exports, but it does not imply that every result or
native-handle type should be constructed directly.

Coverage classes:

- **primary** — recommended application surface;
- **result model** — returned by a primary API and normally not user-constructed;
- **integration** — supported optional/file/tooling boundary;
- **compatibility** — retained for source compatibility but not preferred for new
  application code;
- **advanced compatibility** — low-level native or plan metadata whose direct use
  requires the caller to preserve documented invariants.

## Top-Level Package

| Symbol | Classification | Source documentation | Long-form reference | Executable coverage | Notes |
|---|---|---|---|---|---|
| `gafime.__version__` | result model | module metadata | environment and API index | CLI/Core smoke | PEP 440 runtime identity |
| `gafime.BackendInfo` | result model | class and fields | report/backend sections | Core report smoke | returned as `report.backend` |
| `gafime.BackendCapabilities` | result model | class, fields, `to_dict()` | capability section | Core capability smoke | returned by `backend_capabilities()` |
| `gafime.CapabilityValue` | result model | class and evidence semantics | capability section | Core capability smoke | `source` is runtime/package/static/unknown |
| `gafime.CompiledGafime` | compatibility | alias documentation | compiled section and API index | compile lifecycle smoke | exact alias of `NativeCompiledGafime` |
| `gafime.CompileFlags` | primary | class and fields | compiled section | compile lifecycle smoke | requests plan/graph/export; does not promise support |
| `gafime.ComputeBudget` | primary | class and every field | complete parameter table | Core smoke and field snapshot | `max_generated_features` is compatibility-reserved |
| `gafime.Decision` | result model | class and fields | report table | Core report smoke | threshold-derived report decision |
| `gafime.DecisionPathCandidate` | compatibility | class, fields, methods | decision-path and compatibility sections | focused descriptor test | portable v0.5 descriptor, not a second engine |
| `gafime.DiagnosticReport` | result model | class, fields, ranking/materialization caveat | report section | Core report smoke | `to_dict()` is deprecated |
| `gafime.EngineConfig` | primary | class and every field | complete parameter table | field/signature and Core smokes | precision/family controls are keyword-only |
| `gafime.FamilyCapability` | result model | class, fields, properties | family/capability sections | family smoke | separates generation from scoring placement |
| `gafime.FamilySignificanceSupport` | result model | class and fields | family/significance sections | family smoke | family-specific significance truth |
| `gafime.dataload` | integration | arguments, formats, return/failures | input section | supported Polars 1.x Arrow-IPC smoke | reads and analyzes one file through Polars |
| `gafime.GafimeEngine` | primary | constructor, `analyze`, `compile` | execution sections | Core/compiled smokes | safe public execution boundary |
| `gafime.GafimeSelector` | integration | constructor and estimator methods | sklearn section | sklearn smoke | `n_jobs`/`verbose` are stored compatibility parameters; negative `k` fails validation |
| `gafime.GafimeStreamer` | integration | constructor and batch methods | input/streaming section | supported Polars 1.x CSV smoke | reader only; non-positive `batch_size` fails before reading |
| `gafime.GafimeV1Error` | result model | exception purpose | fail-closed section | expected-error smoke | base explicit v1 runtime error |
| `gafime.InteractionResult` | result model | class and every field | report table | Core report smoke | includes candidate/provenance/numeric diagnostics |
| `gafime.NativeCompiledGafime` | primary | factory-only construction and every public method/property | compiled lifecycle section | compile/update/close smoke | factory-returned, thread-affine lifecycle; explicit close; no context manager |
| `gafime.V1UnsupportedError` | result model | exception purpose | fail-closed section | expected-error smoke | well-formed but unavailable request |
| `gafime.available_families` | primary | return semantics | family sections | family smoke | returns the current immutable family tuple |
| `gafime.backend_capabilities` | primary | complete signature/probe/failure semantics | capability section | Core and expected-error smokes | `probe=False` never proves GPU availability |
| `gafime.compile` | primary | complete signature/lifecycle | compiled section | compile lifecycle smoke | preferred compiled-artifact factory |
| `gafime.family_capability` | primary | lookup/failure semantics | family sections | family smoke | exact-name lookup |
| `gafime.generate_tutorial` | integration | generator contract | documentation hierarchy | deterministic tutorial parity test | generates the compact practice notebook only |
| `gafime.require_family_supported` | primary | lookup/failure semantics | family sections | family smoke | fail-closed family admission |
| `gafime.subfunctions` | advanced compatibility | proxy and compatibility boundary | compatibility/API-index section | existing compatibility tests | prefer top-level safe APIs for new code |

The table covers every name in `gafime.__all__` plus the public runtime version
attribute. CI fails if either side changes without updating this record and the
long-form generator.

## Public Configuration Fields

The reference and signature tests cover all fields of:

- `ComputeBudget`: `max_comb_size`, `max_combinations_per_k`,
  `top_features_for_higher_k`, `max_generated_features`, `keep_in_vram`,
  `vram_budget_mb`, `max_time_series_candidates`,
  `top_k_features_for_time_series`, `max_feature_candidate`;
- `EngineConfig`: `budget`, `metric_names`, `num_repeats`,
  `permutation_tests`, `random_seed`, `stability_std_threshold`,
  `permutation_p_threshold`, `mi_bins`, `backend`, `device_id`, `precision`,
  all time-series and decision-path controls, `significance_top_n`, and
  `mi_approximate`;
- `CompileFlags`: `plan`, `graph`, `export`.

## Returned Objects and Public Methods

The reference describes the fields of `BackendInfo`, `InteractionResult`,
`StabilityResult`, `PermutationResult`, `Decision`, and `DiagnosticReport`.
It covers result-sequence iteration/indexing plus `ranked()` and `top_k()`, and
the complete supported lifecycle of `NativeCompiledGafime`: `flags`, `backend`,
`scenario_plan`, `from_engine`, graph/cache diagnostics, deprecated `exports`,
`analyze()`, `update_target()`, `__arrow_c_array__()`, `export_arrow()`, and
`close()`. `__arrow_c_array__()` is the supported Arrow C Data protocol;
application code normally calls `export_arrow()` explicitly.

Field coverage is exhaustive:

- `CapabilityValue`: `value`, `source`, `detail`;
- `BackendCapabilities`: `configured_backend`, `selected_backend`,
  `selection_status`, `selection_detail`, `probe_performed`, `native_boundary`,
  `native_version`, `families`, `graph_support`, `device_significance`,
  `host_significance_fallback`, `permutation_significance`,
  `stability_significance`, `mi_estimator`, `mi_bin_ceiling`,
  `precision_contract`, `payload_build_policy`, `arrow_ingest_mode`,
  `generated_family_graph_limit`, `device`, `probe_details`;
- `FamilySignificanceSupport`: `permutation`, `stability`, `detail`;
- `FamilyCapability`: `name`, `family_id`, `continuous_input`, `cpu_kernel`,
  `cuda_kernel`, `rocm_kernel`, `metal_kernel`, `python_candidate_loop`,
  `generation_placement`, `scoring_placement`, `graph_scope`,
  `native_compact_scoring`, `significance_support`;
- `DecisionPathCandidate`: `features`, `thresholds`, `signs`, `gain`,
  `support`, `round_id`, `native_candidate_id`, `candidate_id`;
- `BackendInfo`: `name`, `device`, `is_gpu`, `memory_total_mb`,
  `memory_free_mb`, `selected_backend`, `execution_placement`,
  `requested_precision`, `effective_precision`, `storage_dtype`,
  `interaction_arithmetic`, `reduction_dtype`, `result_dtype`,
  `metric_accumulators`, `scale_normalization`, `compensated_summation`,
  `interaction_diagnostics_available`;
- `InteractionResult`: `combo`, `feature_names`, `metrics`, `family`,
  `expression`, `params`, `candidate_id`, `interaction_overflow_rows`,
  `interaction_overflow_ratio`, `source_nonfinite`,
  `precision_diagnostics_available`;
- `StabilityResult`: `combo`, `metrics_mean`, `metrics_std`, `family`,
  `expression`, `params`, `candidate_id`;
- `PermutationResult`: `combo`, `p_values`, `family`, `expression`, `params`,
  `candidate_id`;
- `Decision`: `signal_detected`, `message`;
- `DiagnosticReport`: `config`, `feature_names`, `interactions`, `stability`,
  `permutations`, `warnings`, `decision`, `backend`.

## Explicit Module and Compatibility Exports

These names are indexed so their presence is deliberate rather than an
unexplained import accident.

| Module | Explicit exports | Disposition |
|---|---|---|
| `gafime.reporting` | `BackendInfo`, `Decision`, `DiagnosticReport`, `InteractionResult`, `NativeContinuousInteractions`, `NativeReportBuilder`, `PermutationResult`, `StabilityResult` | result models; native sequence and builder are advanced/compatibility |
| `gafime.decision_path` | `DecisionPathCandidate`, `decision_path_candidate_from_record`, `decision_path_candidate_from_result`, `decision_path_feature_names`, `describe_decision_path_candidate`, `evaluate_decision_path_candidate`, `score_decision_path_candidates` | portable descriptor compatibility helpers; not a second planner/executor |
| `gafime.io` | `GafimeStreamer`, `benchmark_streaming`, `create_streamer` | streamer is supported integration; functions are compatibility/diagnostic helpers |
| `gafime.compile` | `compile`, `CompileFlags`, `CompiledGafime`, `NativeCompiledGafime` | callable module and top-level aliases |
| `gafime.compile.exports` | `ExportHandles`, `unsupported_export` | deprecated handle compatibility |
| `gafime.compile.scenario` | `ChunkRange`, `ContinuousArityDescriptor`, `DEFAULT_CHUNK_SIZE`, `DEFAULT_UNKNOWN_POWER_USER_FEATURE_CAP`, `ScenarioPlan`, `TimeSeriesDescriptor`, `UINT32_MAX`, `UINT64_MAX`, `UINT128_MAX`, `build_scenario_plan`, `build_scenario_plan_from_shape` | bounded compatibility metadata; Rust plans remain authoritative |

Compatibility record fields are also explicit: `ExportHandles` has
`backend_name`, `feature_matrix_handle`, `result_table_handle`, and
`candidate_table_handle`; `ChunkRange` has `first_chunk_id`, `chunk_count`, and
`chunk_size`; `ContinuousArityDescriptor` has `arity`, `feature_start`,
`feature_stop`, `universe_count`, `planned_count`, `offset`, `chunk_range`, and
`saturated`; `TimeSeriesDescriptor` has `feature_start`, `feature_stop`,
`lag_count`, `window_count`, `template_count`, `universe_count`, `planned_count`,
`offset`, and `saturated`; `ScenarioPlan` has `n_samples`, `n_features`,
`feature_candidate_count`, `precision`, `continuous`, `time_series`, `warnings`,
and `metric_ids`.

`gafime.subfunctions` dynamically proxies the installed native boundary. The
current exact package exposes `BOUNDARY_NAME`, `BatchScheduler`,
`CacheAwareScheduler`, `CompiledContinuousArtifact`, `ContinuousRecord`,
`ContinuousReport`, `DataQualityAnalyzer`, `OTSEncoder`, `SmartScheduler`,
`analyze_continuous`, `analyze_continuous_arrow`, `analyze_continuous_buffers`,
`analyze_continuous_cpu`, `analyze_continuous_rows`, `analyze_decision_path`,
`analyze_time_series`, `compile_continuous`, `compile_continuous_buffers`,
`compile_continuous_rows`, `compile_decision_path`, `compile_time_series`,
`native_version`, and `runtime_capabilities`. These are advanced compatibility
exports. New application code should use `GafimeEngine`, `compile`,
`backend_capabilities`, `dataload`, and structured reports.

The supported CLI entry points are `gafime` and `python -m gafime`, with
`--version` and `--check --backend ... --device-id ... --precision ...`.

## Deliberate Non-API

Private names, internal adapter functions, native C ABI symbols, backend runtime
types, experimental RT/OptiX code, and future Candidate IR/decorator/JIT ideas
are not public Python API. Documentation helpers and skills consume the API
above; they do not authorize additional runtime surfaces.

## Input-validation coverage

Focused regression tests require `GafimeSelector(k=...)` to reject negative
values at construction and assignment. Streamer batch sizes and the bounded
`benchmark_streaming(..., n_batches=...)` diagnostic reject non-positive values
before reading input.
