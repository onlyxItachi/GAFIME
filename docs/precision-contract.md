# GAFIME Precision Contract

GAFIME exposes one top-level precision profile. A profile binds the complete
numeric execution contract; it is not a storage hint and a backend name is not
a precision claim.

## Public Request

`precision` is keyword-only and defaults to `mixed`:

```python
EngineConfig(precision="mixed")
EngineConfig(precision="fp32")
EngineConfig(precision="fp64")
```

The public configuration, capability report, diagnostic report, compiled
artifact, and CLI expose `precision`. They do not expose storage and compute as
independently configurable choices. A deprecated compatibility parser may
accept both old fields together only for these unambiguous mappings:

| Deprecated pair | Profile |
|---|---|
| `float32 + fast` | `fp32` |
| `float32 + stable` | `mixed` |
| `float64 + exact` | `fp64` |

Missing, contradictory, or unsupported legacy combinations fail closed.

## Four Numeric Domains

The profile controls exactly four floating-point domains:

1. input ingest and resident storage;
2. pointwise transforms and materialized interaction arithmetic;
3. reductions and statistical arithmetic;
4. ranking scores and public metric or significance results.

Planner and protocol structure remains integer/control data. Candidate IDs,
feature indices, arity, counts, shape hints, schedules, seeds, permutation
schedules, rank positions, protocol IDs, and histogram counts must not be
converted to floating point.

| Profile | Ingest/storage | Pointwise/interactions | Reductions/statistics | Ranking/public results |
|---|---|---|---|---|
| `fp32` | fp32 | fp32 | fp32 | fp32 |
| `mixed` | fp32 | fp32 | fp64 | fp64 |
| `fp64` | fp64 | fp64 | fp64 | fp64 |

`fp32` is a genuine lane-wide fp32 route, including covariance, Spearman, MI,
bootstrap, permutation, decision-path statistics, ranking, and host result
processing. Its expected dynamic-range and rounding limits must be reported
honestly; an implementation must not silently widen a reduction or public
score.

`mixed` preserves fp32 storage and pointwise throughput while performing
statistical reductions, ranking, and public result arithmetic in fp64. Public
scores remain fp64; an implementation must not rank with a hidden fp64 value
and then downcast the visible score. MI histogram indices and counts remain
integer while probability, correction, logarithm, normalization, and final
score arithmetic are fp64. Spearman rank positions remain integer while
rank-derived covariance, normalization, ranking, and output are fp64.

`fp64` preserves fp64 from ingest through output with no intermediate fp32 quantization.
NumPy, Arrow, Polars, target replacement, resident storage,
caches, compiled execution, graph replay, significance, ranking, and output
tables all remain fp64.

The Arrow result table is the canonical dtype-bearing public table boundary.
Native vector helpers for metrics and significance additionally return standard
library `array('f')` values for `fp32` and `array('d')` values for `mixed` and
`fp64`. Decision-path threshold vectors follow the pointwise domain instead:
they are `array('f')` for `fp32` and `mixed`, and `array('d')` for `fp64`;
decision-path gains follow the result domain. This preserves a dtype-bearing
boundary without adding NumPy as a mandatory runtime dependency, which would
break the CPython 3.10 Windows ARM64 distribution matrix. A Python scalar has no
binary32 type, so indexed scalar presentation carries the exact already-finalized
fp32 bit pattern in a Python `float`. Host-side fp32 significance comparisons
quantize their configured thresholds to the same fp32 lane before comparing;
presentation widening therefore adds no arithmetic precision, and
`result_dtype` remains `float32`.

## Backend Capability Matrix

| Backend | `fp32` | `mixed` | `fp64` |
|---|---:|---:|---:|
| Core | yes | yes | yes |
| CUDA | yes | yes | yes |
| ROCm | yes | yes | yes |
| Metal | yes | no | no |

Core, CUDA, and ROCm compile all three profiles into their existing
distributed artifacts. Plan construction selects a profile-specialized
function table or kernel set once; a hot loop must not branch on the profile
for each element.

Metal contains only a genuine fp32 lane. Metal Shading Language does not offer
native shader FP64 execution, so explicit `backend="metal"` with `mixed` or
`fp64` fails before coercion, allocation, payload-discovery side effects, or
execution with an actionable capability error. `backend="auto"` may exclude
Metal and select Core for those profiles. CPU double work must never execute
while the reported explicit backend is Metal, and software-emulated double must
not be labelled Metal fp64.

## Ingest Ordering

The requested profile and backend/profile compatibility are validated before
conversion. `fp32` and `mixed` intentionally own fp32 input storage; `fp64`
preserves fp64 input without an fp32 intermediate. The Polars adapter chooses
`Float32` or `Float64` only after this validation. Arrow input must match the
selected storage domain and fail closed on a mismatched dtype rather than
silently converting through fp32.

## ABI And Resident Identity

ABI 1.0 float entry points and layouts remain unchanged. The additive ABI 1.1
precision surface provides:

`GAFIME_DTYPE_F32 = 1` and `GAFIME_DTYPE_F64 = 2`; profile identifiers are
`GAFIME_PRECISION_FP32 = 1`, `GAFIME_PRECISION_MIXED = 2`, and
`GAFIME_PRECISION_FP64 = 3`. These values are additive ABI identities, not
permission to reinterpret an ABI 1.0 pointer.

- a profile capability mask and accepted storage/result dtype masks;
- profile-bearing matrix descriptors and launch protocols;
- typed f32 and f64 feature/target upload and target replacement;
- typed f32 and f64 result tables, plus typed permutation/significance tables
  when a backend exposes the optional native permutation ABI;
- dtype-correct execution memory-peak queries and a paired permutation peak
  query whenever the optional native permutation path is present.

An ABI 1.0 `float*` is never reinterpreted as `double*`. A current payload must
export the required additive capability, typed matrix, execution, and
execution-peak surface; advertise exactly the profiles it can physically
execute; and reject a profile/dtype mismatch before allocation. CUDA exports
the optional typed permutation ABI. ROCm and Metal intentionally omit those
optional symbols and use the documented Rust-orchestrated same-device ranking
path, so symbol presence never falsely advertises native significance support.
Rust matrix handles record whether their native owner is ABI 1.0 or ABI 1.1;
legacy and precision operations reject the opposite generation before any raw
pointer conversion or FFI call.

The precision profile is part of compiled-artifact, resident-matrix,
descriptor, graph, target-stat, feature-stat, and public analyze-cache
identities. Target replacement invalidates only state for the matching profile;
state from one profile can never be reused by another. Structural planner
metadata retains its existing integer/control types.

## Metrics, Families, And Significance

The selected profile applies to Pearson, Spearman, mutual information, and R2
across continuous, time-series, and decision-path families; unary screening;
arities 1 through 5; eager, resident, compiled, and graph execution; target
replacement; cache-enabled and cache-disabled execution; bootstrap mean/std;
permutation p-values and maxT; ranking, ordering, ties, and public warnings.

Generated-family structural planning remains unchanged. Every numeric value
must belong to one of the four domains above; there is no undocumented fifth
precision policy. Different profiles are not required to return identical
numbers, but each must satisfy its declared arithmetic and stable candidate
identity/order contract.

## Distribution And Frozen-Bundle Contract

Profiles do not create distributions or wheel families. The only standard
distributions remain `gafime`, `gafime-cuda`, and `gafime-rocm`, producing 40
wheels and 3 sdists. Every Core wheel carries Core `fp32`, `mixed`, and `fp64`.
The five macOS arm64 Core wheels additionally embed Metal `fp32` only. Every
CUDA and ROCm wheel carries all three specializations in one payload binary.

The system CUDA runtime, host-managed ROCm runtime, exact payload-to-Core
dependency, ROCm GitHub-Release/PyPI-sdist policy, Core-first publication order,
immutable bundle, provenance, checksum, and RT/OptiX exclusion policies do not
change. Frozen-bundle provenance records the profile contract for every package
file. Installed-wheel validation verifies Core execution and payload ABI
identity; physical backend validation verifies the exact capability mask and
supported routes. RT/OptiX sources, symbols, packages, caches, and artifacts
remain forbidden.

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
  left-to-right product in the selected pointwise dtype becomes non-finite;
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

Repeated compiled execution reuses a diagnostic only when the surfaced
combination table has the same row count, arity, exact feature-index contents,
and precision profile. Replacing the target or planning seed rebuilds the
execution plan and invalidates that cache before another report is produced.

The ordinary path does not add a second matrix-row scan. Core gathers extrema
while transposing the input; GPU payloads gather finite/exponent metadata during
their existing upload conversion. A conservative prefix bound proves ordinary
selected products finite. Only surfaced combinations that cannot be proved safe
run an exact row scan, using the same centered subtraction, selected pointwise
dtype, and multiplication order as scoring. The scan is post-selection and
synchronous, so eager, resident-cache, compiled, and graph-replay execution all
expose the same diagnostic without adding work to captured graphs.

The local safe-path A/B, binary hashes, toolchains, and hardware boundaries are
recorded in `docs/evidence/interaction-overflow-diagnostics.md`.

These diagnostics report lost values; they do not recover them. Widened or
log-domain interaction evaluation would be a separate numerical mode with its
own capability, reference, significance, and result contracts.

## Admission Evidence

Release admission requires independent lane-specific oracles and adversarial
inputs, including fp64 values that collapse when converted to fp32. Evidence
must cover all four domains, all entry points and families, profile-key cache
separation, eager/resident/compiled and graph/plain parity, target replacement,
ranking visibility, non-finite and zero-variance handling, histogram boundaries,
vector tails, and supported physical backends.

Build evidence records build time, compressed wheel size, and uncompressed
native-binary size for each backend. Binary growth alone is not a reason to
omit a supported profile. TF32, tensor-core reformulation, software FP64,
non-materialized scoring, RT/OptiX, and other new algorithms are outside this
contract.
