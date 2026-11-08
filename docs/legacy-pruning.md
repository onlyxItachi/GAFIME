# Pre-Stable Legacy Pruning And Retained API Boundaries

The maintainer approved removing obsolete v0.x compatibility before v1.0.0
stable qualification. This is an intentional Python source-compatibility break
relative to published RC3, not a change to the native numerical or GPU ABI
contracts. Historical release notes and evidence remain unchanged.

## Removed Compatibility

- Configuration pair mappings, discrete switches (including false) and
  positional-layout translation. Canonical EngineConfig/ComputeBudget prefixes
  and keyword-only controls remain; removed requests fail rather than remap.
- Python scenario-plan projection, `NativeCompiledGafime.scenario_plan`,
  deprecated `exports`, `ExportHandles`, and projection-only `CompileFlags.plan`.
  Native Rust planning is always authoritative. Flags are keyword-only so an
  old positional plan value cannot silently become a graph request.
- Native `BatchScheduler`, `CacheAwareScheduler` and `SmartScheduler` classes
  and their bindings. No authoritative Rust scheduler/planner is replaced.
- The helper proxy's `gafime.gafime_cpu` and top-level `gafime_cpu` lookup paths.
  The contracted extension is `gafime.gafime_py`. Native exports are explicitly
  listed rather than automatically publishing every native implementation name.

The retired schedulers had distinct historical semantics: homogeneous
arity/template grouping, anchor/cache-local ordering and launch hints; pairwise
anchor/partner windows and n-ary grouping; and mutable commutative-key dedup,
iteration and reset. The modern planner is not claimed to be a drop-in
replacement for those helper APIs. They have no production-runtime caller and
are deliberately removed, not silently routed into a different algorithm.

## Maintained Subfunctions

`gafime.subfunctions` remains an intentional lazy namespace. Import, `dir` and
`__all__` inspection do not load the extension. Its explicit exports retain
DataQualityAnalyzer, OTSEncoder and the documented native report/artifact types
and low-level functions. Private acquisition classes/functions are not promoted
to public API. Main analysis still belongs to the top-level safe interfaces.

DataQualityAnalyzer emits `DeprecationWarning` on construction, not import or
attribute lookup. Existing NaN/Inf, missing-value, entropy and feature-quality
behavior is retained; warning-as-error handling follows normal Python behavior.
Future replacement is tracked by [issue #87](https://github.com/onlyxItachi/GAFIME/issues/87)
for the Polars migration in v1.1/v1.2. It must be explicitly requested by the
caller, with no silent cleaning, feature deletion or caller-data mutation.

OTSEncoder's source and behavior are retained unchanged. Backend-neutral OTS
modernization is [issue #119](https://github.com/onlyxItachi/GAFIME/issues/119),
not v1.0.0 work. No categorical kernels, embeddings or pipeline integration are
introduced here. The maintained Rust home is `gafime-py/src/subfunctions/`.

## Other Surfaces: Deliberately Not Pruned

| Surface | Disposition and reason |
|---|---|
| `CompiledGafime` / `from_engine` | Retained alias/factory compatibility; both use the modern native compiled lifecycle. Separate public-API approval is needed for removal. |
| Python decision-path descriptors/utilities | Retained explicit compatibility surface; not the native GBDT discovery owner. No blanket removal or change to native discovery. |
| `NativeReportBuilder` | Retained public compatibility builder. Mutable append and report-snapshot semantics are externally usable; its result-sequence type also serves active reports. |
| Custom-boundary fallbacks | Retained opt-in compatibility with explicit environment/module selection and tests. Removing them requires a separate extension-policy decision and caller audit, not a claim that a differently named route is equivalent. |
| Streamer factories/diagnostics | GafimeStreamer is supported; automatic batch sizing is used by its streaming methods. `create_streamer` and `benchmark_streaming` remain published factory/diagnostic helpers. |
| Selector `n_jobs` / `verbose` | Stored sklearn constructor/get_params/set_params compatibility, including cloning; not native scheduler controls. Removing them needs a separate estimator-contract decision. |
| `ComputeBudget.max_generated_features` | Retained reserved fourth positional field, serialized and parsed as u32 into the Rust `repr(C)` budget. It is not a generated-family limit. Removal affects source/payload/ABI contracts and requires a separate decision. |

## Unchanged Execution And Evidence

Keep GafimeEngine, EngineConfig, ComputeBudget, NativeCompiledGafime, selector,
streamer, dataload, backend capabilities and all current feature families.
Rust owns planning, scheduling, validation and lifecycle. Core SIMD/Rayon and
CUDA/HIP/Metal execution, fp32/mixed/fp64 domains, native candidate IDs and
ordering, significance and ABI 1.0/1.1 remain unchanged.

Arrow C Stream acquisition and Arrow C Data result export remain distinct.
`CompileFlags(export=True)`, `export_arrow()` and `__arrow_c_array__()` preserve
result ownership and release callbacks; removing raw compatibility handles does
not remove Arrow export. Input snapshots are still owned, not atomic against
external writers or a promise of resident zero-copy.

Focused tests replace obsolete acceptance/projection assertions without
deleting release-gate files, changing numerical goldens or relaxing tolerances.
Source checks do not substitute for installed-package checks or eventual
exact-source stable qualification. No performance claim or release authority
follows from pruning; published RC3 and its frozen bytes remain immutable.
