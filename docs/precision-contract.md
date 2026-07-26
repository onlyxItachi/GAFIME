# GAFIME Precision Contract

GAFIME treats matrix storage, interaction arithmetic, reduction arithmetic,
and result storage as separate choices. A backend name is not a precision
claim.

## Public Request

`EngineConfig` exposes two keyword-only fields:

```python
EngineConfig(storage_dtype="float32", compute_policy="stable")
```

The current executable contract supports only `float32 + stable`.
`storage_dtype="float64"` and `compute_policy="exact"` are recognized future
requests and fail closed before input coercion or native execution. The
`compute_policy="fast"` name is also reserved: stable execution already selects
the tuned fast kernel for safe ranges, and users cannot disable the
high-dynamic normalization guard.

`backend_capabilities(...).precision_contract` reports the request, the
effective pair when a backend was selected, supported values, interaction and
result widths, per-metric accumulator widths, normalization policy, and the
reason for a rejected request. `DiagnosticReport.config` preserves the request;
`DiagnosticReport.backend` records the effective execution contract.

## ABI Boundary

`GAFIME_DTYPE_F32 = 1` remains the only accepted matrix storage dtype.
`GAFIME_DTYPE_F64 = 2` is an additive reserved enum value. Every current
payload rejects an f64 matrix descriptor without allocating storage, and no
payload sets `GAFIME_GPU_DEVICE_FLAG_F64_STORAGE`.

The reservation does not make the current upload ABI an f64 ABI. Upload and
target-update buffers, resident matrix allocation, graph/cache ownership, and
memory forecasting all remain fp32. Result-table metrics also remain fp32.

CUDA and ROCm payloads separately advertise
`GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64` when their compile-time MI policy
uses double-precision contribution, reduction, correction, and normalization.
That flag does not imply f64 storage or f64 interaction arithmetic.

## Effective Pipeline

| Backend | Matrix storage | Interaction arithmetic | Pearson/R2 accumulator | Spearman accumulator | MI accumulator | Result |
|---|---|---|---|---|---|---|
| Core | fp32 | fp32 product | fp64 centered reduction | fp64 | fp64 | fp32 |
| CUDA | fp32 | fp32 product with adaptive scale guard | fp32 | fp64 | payload-reported fp32 or fp64 | fp32 |
| ROCm | fp32 | fp32 product with adaptive scale guard | fp32 | fp64 | payload-reported fp32 or fp64 | fp32 |
| Metal | fp32 | fp32 product with adaptive scale guard | fp32 | fp32 | fp32 | fp32 |

Distributed CUDA and ROCm wheels compile the fp32 MI mode. Local native builds
may opt into the fp64 MI accumulator described in `docs/capabilities.md`; this
changes only MI arithmetic after histogram construction.

## Interaction Materialization Diagnostics

Every current Core build diagnoses the surfaced result rows after scoring.
Current CUDA, ROCm, and Metal payloads expose the optional
`gafime_gpu_interaction_diagnostics` C ABI; an older same-ABI payload without
that symbol remains loadable and reports diagnostics as unavailable.

The ABI consumes a `GafimeInteractionDiagnosticBatch` whose `combo_indices`
contains `row_count` rows of `max_arity` `uint32_t` values. Each row has one to
five feature indices followed only by `UINT32_MAX` padding. The payload writes
one `uint64_t` overflow-row count and one flags word per row.
`GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE` is the only current flag.
Reserved fields must be zero.

The diagnostic definition is deliberately narrower than "the score is
non-finite":

- `source_nonfinite` is true when a referenced raw feature, its stored mean, or
  the target contains a non-finite value;
- `interaction_overflow_rows` counts sample rows whose referenced feature
  sources and means are finite but whose centered subtraction or sequential
  left-to-right fp32 product becomes non-finite;
- a non-finite target sets `source_nonfinite` but does not suppress a
  finite-feature interaction-overflow count;
- unary candidates have no interaction-product overflow and therefore report a
  zero count;
- the count covers surfaced candidates only. It does not rescan rejected
  candidates or change candidate IDs, metric values, ranking, graph state, or
  cache identity.

`InteractionResult` exposes `interaction_overflow_rows`,
`interaction_overflow_ratio`, `source_nonfinite`, and
`precision_diagnostics_available`. `BackendInfo` and
`backend_capabilities(...).precision_contract` disclose availability. A report
adds one aggregate warning when at least one surfaced candidate has a non-zero
finite-input overflow count. Source non-finite flags alone do not add another
warning because input validation and metric finiteness remain separate
contracts.

Native reports retain diagnostics in compact Rust storage. Python retrieves one
diagnostic on indexed access or an aligned batch during iteration; constructing
the report does not eagerly allocate a Python tuple for every surfaced
candidate. Aggregate warning construction reads native affected-candidate and
maximum-row counts. The legacy full-list property remains available for
same-version compatibility but is not the normal reporting path.

The ordinary path does not add a second matrix-row scan. Core gathers extrema
while transposing the input; GPU payloads gather finite/exponent metadata during
their existing upload conversion. A conservative prefix bound proves ordinary
selected products finite. Only surfaced combinations that cannot be proved safe
run an exact row scan, using the same centered subtraction and fp32
multiplication order as scoring. The scan is post-selection and synchronous, so
eager, resident-cache, compiled, and graph-replay execution all expose the same
diagnostic without adding work to captured graphs.

These diagnostics report lost values; they do not recover them. Widened or
log-domain interaction evaluation would be a separate numerical mode with its
own capability, reference, significance, and result contracts.

## True f64 Admission

A future `float64 + exact` implementation must land as a separate reviewed ABI
and execution change. It must jointly define:

1. f64 Python and Arrow ingest without an intermediate fp32 quantization;
2. typed upload and target-update buffers;
3. resident allocation and memory-peak accounting at eight bytes per element;
4. dtype-keyed graph, descriptor, target-stat, and feature-stat caches;
5. f64 interaction and metric kernels for Core, CUDA, and ROCm;
6. result dtype and significance-oracle policy;
7. capability-aware `auto` selection and explicit Metal rejection;
8. independent extended-precision or analytic validation.

Until all of those are present, GAFIME must not label a partial wider
accumulator as an end-to-end fp64 pipeline.
