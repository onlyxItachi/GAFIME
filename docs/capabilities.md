# GAFIME v1 Capability Reporting

This document describes the public capability contract for the
`1.0.0-beta.2` repository release (`1.0.0b2` on Python/PyPI). It is a report
of implementation placement, not a promise that a payload or device is
installed.

## Public API

Use `gafime.backend_capabilities()` for a structured Python result:

```python
import gafime

caps = gafime.backend_capabilities("auto", probe=True, precision="mixed")
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
gafime --check --backend auto --device-id 0 --precision mixed
gafime --check --backend cuda --precision fp64
```

The latter exits nonzero when CUDA is unavailable. Core-only installations work
with `--backend core` or with `--backend auto` when auto resolves to Core.

## Family Placement

| Family | Generation placement | Scoring placement | Graph scope | Significance support |
|---|---|---|---|---|
| `continuous` | Native continuous planner/direct path | `gafime_cpu`, CUDA, ROCm, Metal | Runtime-dependent continuous scoring | Permutation maxT and bootstrap stability |
| `time_series` | `gafime_cpu` expansion | `gafime_cpu`, CUDA, ROCm, Metal continuous scoring | Continuous scoring only | Permutation maxT and bootstrap stability |
| `decision_path` | `gafime_cpu` path discovery | `gafime_cpu`, CUDA, ROCm, Metal continuous scoring | Continuous scoring only | Permutation maxT with per-target path rediscovery and bootstrap stability |

`FamilyCapability.generation_backend` is the explicit alias for generation
placement, while `.scoring_backends` lists the backends that consume generated
features. `.native_compact_scoring` is narrower: it lists only family paths that
score compact candidate descriptors without first expanding feature columns.
`.significance_support` reports permutation and bootstrap-stability support per
family. Backend-wide significance placement never overrides a family exclusion.

### Bootstrap Stability Scope

Bootstrap stability resamples an already-selected candidate on the same rows
that were used to select it. `stability_std` therefore measures metric
variability **conditional on selection**. It is not an out-of-sample or
out-of-fold estimate, does not correct selection bias or winner's curse, and
does not establish that the candidate will generalize. Use an untouched
holdout or nested cross-validation for generalization evidence.

`EngineConfig.permutation_tests` defaults to `25`. For `decision_path`, each
permuted target rediscovers its own paths, rebuilds the expanded feature family,
and rescans the full configured arity range before maxT comparison.
`num_repeats > 1` remains supported for selected-candidate bootstrap stability.

The retained `FamilyCapability.cpu_kernel`, `.cuda_kernel`, `.rocm_kernel`, and
`.metal_kernel` fields are compatibility aliases for **scoring** support. They
do not represent generated-family CUDA, HIP, or Metal kernels. No graph capture
includes `time_series` or `decision_path` generation; a graph can only apply
after their CPU expansion reaches continuous scoring.

`decision_path` discovery remains on `gafime_cpu`. Its generated membership
columns use the same continuous scorer as the other generated families, so
backend selection, graph limits, candidate ordering, and fallback ownership
remain unchanged.

## Backend Facts

The capability result includes the following facts:

- graph mode and graph-node flags from `GafimeGpuGraphCapability` when a GPU
  payload has been validated; Core graph support is statically `False`.
- permutation significance placement is reported separately from bootstrap
  stability. CUDA, ROCm, and Metal use Rust-orchestrated same-device target
  replay plus device `top_k=1` ranking when the payload advertises device
  ranking. CUDA static families may instead use the optional native fixed-plan
  p-value ABI. Rust owns the family-wise exceedance counts in either route.
  Bootstrap stability remains a selected-candidate CPU pass, preserves the
  observed backend's fixed-width MI estimator and template ceiling, and carries
  the same conditional-on-selection limitation described above.
- mutual-information estimator and effective template-bin ceiling. Core uses
  adaptive quantile MI unless `mi_approximate=True`; GPU scoring uses fixed
  equal-width MI. The supported templates are `2,4,8,12,16,24,32,48,64,96`.
  Metal has a 48-bin maximum; other current backends have a 96-bin maximum.
  Sample count chooses a template at or below the reported ceiling.
- requested and effective `precision` profiles. The report lists the exact
  supported profiles and the storage, interaction, reduction, accumulator, and
  public-result dtype for the request. Core, CUDA, and ROCm advertise `fp32`,
  `mixed`, and `fp64`; Metal advertises `fp32` only. A loaded GPU payload supplies
  the capability mask through additive ABI 1.1 rather than having support
  inferred from its backend name. See
  [precision-contract.md](precision-contract.md).
- CUDA and ROCm compile all three profile-specialized kernel sets into the same
  distributed payload. MI histogram indices/counts remain integer. MI
  probability, correction, logarithm, normalization, ranking, and output use
  the selected profile's reduction/result domain. There is no build-time MI
  precision selector and no precision-specific payload package. Metal remains
  fp32 because MSL has no native shader FP64; `mixed` and `fp64` fail closed.
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
  must still be representable in the selected profile's pointwise dtype before
  normalization; see the numerical policy and backend validation evidence.
- interaction-materialization diagnostics. Core and current CUDA, ROCm, and
  Metal payloads report finite-input selected-pointwise-domain overflow counts
  for surfaced candidates separately from source non-finite flags. The precision
  capability reports `interaction_overflow_diagnostics`; an older optional GPU
  payload may truthfully report `false` while remaining loadable. Safe selected
  candidates use upload-time bounds and do not launch a row scan. See
  [precision-contract.md](precision-contract.md).
- Arrow C stream ingest. One record batch is required. Validated columns become
  a GAFIME-owned row-major `f32` compute buffer for `fp32`/`mixed` or an `f64`
  buffer for `fp64`, with no fp32 intermediate. The interface avoids Python
  object materialization but is not zero-copy into compute memory.
## Payload Discovery Seam

The public native endpoint is `gafime.gafime_py.runtime_capabilities(backend="auto", device_id=0, probe=False, *, precision="mixed")`.
The precision argument is keyword-only. It uses the same
`GpuBackend::*_from_env` loader seam as normal execution. Payload
packaging/discovery may evolve behind those loader methods without changing the
Python capability schema, the CLI policy, or the explicit-backend no-fallback
guarantee.
