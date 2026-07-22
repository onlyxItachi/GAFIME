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
