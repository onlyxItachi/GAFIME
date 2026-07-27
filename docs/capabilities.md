# GAFIME v1 Capability Reporting

This document describes the public capability contract for the `1.0.0b1`
pre-release. It is a report of implementation placement, not a promise that a
payload or device is installed.

## Public API

Use `gafime.backend_capabilities()` for a structured Python result:

```python
import gafime

caps = gafime.backend_capabilities("auto", probe=True)
print(caps.configured_backend, caps.selected_backend, caps.selection_status)
print(caps.precision_contract.value)
```

`CapabilityValue.source` is always one of:

- `runtime`: returned by the loaded, ABI-validated native payload.
- `package`: read from a uniquely matching installed distribution without
  importing or loading its native library.
- `static`: a checked-in Core policy that does not depend on a device claim.
- `unknown`: no compatible runtime observation is available. It is not a
  negative hardware claim.

`probe=False` never loads a GPU payload and leaves `selected_backend` unknown
for every unvalidated GPU request. `probe=True` loads the configured payload
and calls its identity/device/graph ABI only; it does not allocate a matrix or
run scoring. Explicit `cuda`, `rocm`, and `metal` probes never select another
backend. An `auto` probe lists every candidate and states when it chose Core
because no GPU payload passed the ABI probe.

The CLI exposes the same contract:

```text
gafime --check --backend auto --device-id 0
gafime --check --backend cuda
```

The latter exits nonzero when CUDA is unavailable. Core-only installations work
with `--backend core` or with `--backend auto` when auto resolves to Core.

## Family Placement

| Family | Generation placement | Scoring placement | Graph scope | Significance support |
|---|---|---|---|---|
| `continuous` | Native continuous planner/direct path | `gafime_cpu`, CUDA, ROCm, Metal | Runtime-dependent continuous scoring | Permutation maxT and bootstrap stability |
| `time_series` | `gafime_cpu` expansion | `gafime_cpu`, CUDA, ROCm, Metal continuous scoring | Continuous scoring only | Permutation maxT and bootstrap stability |
| `decision_path` | `gafime_cpu` path discovery | `gafime_cpu`, CUDA, ROCm, Metal continuous scoring; optional compact CUDA RT scoring for the exact unary Pearson/R2 shape | Continuous scoring only | Bootstrap stability only; permutation significance requires unavailable per-target path rediscovery |

`FamilyCapability.generation_backend` is the explicit alias for generation
placement, while `.scoring_backends` lists the backends that consume generated
features. `.native_compact_scoring` is narrower: it lists only family paths that
score compact candidate descriptors without first expanding feature columns.
`.significance_support` reports permutation and bootstrap-stability support per
family. Backend-wide significance placement never overrides a family exclusion.

`EngineConfig.permutation_tests` defaults to `25`, but decision-path permutation
significance is intentionally unavailable until every permuted target can
rediscover its own paths. Set `permutation_tests=0` when enabling
`decision_path`; `num_repeats > 1` remains supported for selected-candidate
bootstrap stability. Unsupported permutation requests fail closed with
`V1UnsupportedError` before backend execution.

The retained `FamilyCapability.cpu_kernel`, `.cuda_kernel`, `.rocm_kernel`, and
`.metal_kernel` fields are compatibility aliases for **scoring** support. They
do not represent generated-family CUDA, HIP, or Metal kernels. No graph capture
includes `time_series` or `decision_path` generation; a graph can only apply
after their CPU expansion reaches continuous scoring.

The `decision_path` compact CUDA route is admitted only when the loaded device
and payload report OptiX RT plus the score ABI, every feature/target/path value
is finite and RT-representable, the complete untruncated candidate set is unary,
the metrics are Pearson and/or R2, and neither graph nor significance execution
is requested. Rust still discovers paths and merges base and path rows in public
candidate order. Every other shape uses the established membership-expansion
and continuous-scoring path; an explicit require-RT policy fails closed.

## Backend Facts

The capability result includes the following facts:

- graph mode and graph-node flags from `GafimeGpuGraphCapability` when a GPU
  payload has been validated; Core graph support is statically `False`.
- permutation significance placement is reported separately from bootstrap
  stability. CUDA, ROCm, and Metal use Rust-orchestrated same-device target
  replay plus device `top_k=1` ranking when the payload advertises device
  ranking. CUDA static families may instead use the optional native fixed-plan
  p-value ABI. Rust owns the family-wise exceedance counts in either route.
  Bootstrap stability remains a selected-candidate CPU pass and preserves the
  observed backend's fixed-width MI estimator and template ceiling.
- mutual-information estimator and effective template-bin ceiling. Core uses
  adaptive quantile MI unless `mi_approximate=True`; GPU scoring uses fixed
  equal-width MI. The supported templates are `2,4,8,12,16,24,32,48,64,96`.
  Metal has a 48-bin maximum; other current backends have a 96-bin maximum.
  Sample count chooses a template at or below the reported ceiling.
- requested and effective precision as separate storage and compute policies.
  The only current executable pair is `float32 + stable`. The stable policy
  retains the tuned ordinary-range path and applies scale normalization only
  when the interaction exponent range requires it. `float64`, `exact`, and an
  explicit guard-disabling `fast` request fail closed before input coercion.
  The capability includes per-metric accumulator widths; CUDA and ROCm obtain
  the MI width from the loaded payload flag rather than inferring it from the
  backend name. See [precision-contract.md](precision-contract.md).
- CUDA and ROCm MI arithmetic is a compile-time payload policy. Distributed
  payloads and ordinary local builds use the `fast` fp32 reduction. Local
  native builds may select `-DGAFIME_CUDA_MI_ACCUMULATION_MODE=fp64` or
  `-DGAFIME_HIP_MI_ACCUMULATION_MODE=fp64`; that mode uses fp64 for MI
  contribution, reduction, finite-sample correction, and normalization before
  casting the result to fp32. It does not change fp32 matrix storage, histogram
  bin mapping, or the public dtype contract, and it adds no runtime branch or
  second kernel set to one payload. Metal remains fp32 because MSL has no fp64.
- installed CUDA and ROCm payload build policy plus the static Metal packaging
  contract. This can be reported with `probe=False`. The standard ROCm payload
  reports `system`, `userspace_bundled=false`, its ROCm 7.2.3 build inputs,
  13 GFX targets, and host-managed single-runtime requirement. Metal reports
  its `gafime` macOS arm64 core-wheel identity and paired dylib/metallib
  contract. An
  explicit external library is never attributed to an unrelated installed
  wheel. See [rocm-wheel-policy.md](rocm-wheel-policy.md).
- correlation arithmetic failures remain non-finite. Exact zero variance maps
  to Pearson/R2 zero, while an overflowed or otherwise non-finite reduction maps
  to NaN and is excluded from primary-score ranking. Device clamps never turn
  that failure into Pearson `-1` or R2 `1`. CUDA, ROCm, and Metal route
  exponent-risky descriptor chunks through scale-normalized covariance while
  retaining the established fast path for ordinary magnitudes. An arity product
  must still be representable as fp32 before normalization; see the numerical
  policy and backend validation evidence.
- interaction-materialization diagnostics. Core and current CUDA, ROCm, and
  Metal payloads report finite-input fp32 overflow counts for surfaced
  candidates separately from source non-finite flags. The precision capability
  reports `interaction_overflow_diagnostics`; an older optional GPU payload may
  truthfully report `false` while remaining loadable. Safe selected candidates
  use upload-time bounds and do not launch a row scan. See
  [precision-contract.md](precision-contract.md).
- Arrow C stream ingest. One record batch is required, and validated columns
  become a GAFIME-owned row-major `f32` compute buffer. The interface avoids
  Python object materialization but is not zero-copy into compute memory.
- CUDA RT availability only from the loaded device flags, plus the actual
  optional decision-path ABI symbols. Without a validated CUDA payload it is
  `unknown`, not inferred from a product name or environment hint.

## Payload Discovery Seam

The public native endpoint is `gafime.gafime_py.runtime_capabilities(backend,
device_id, probe)`. It uses the same `GpuBackend::*_from_env` loader seam as
normal execution. Payload packaging/discovery may evolve behind those loader
methods without changing the Python capability schema, the CLI policy, or the
explicit-backend no-fallback guarantee.
