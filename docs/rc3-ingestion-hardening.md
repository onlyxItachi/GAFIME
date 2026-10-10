# RC3 public ingestion hardening

## Status and decision

Stable qualification is postponed for bounded RC3 ingestion hardening. This
document records the defect, implementation boundary and acceptance plan; it
does not declare RC3 ready or authorize tagging/publication. The initial audit
used main `c85178fe705ed71ac65d6c59203ce5dd76524239`. The earlier stable candidate
`70a08f2ef712591dd28951ff7c4d3db6bfcb6566` remains an unpublished checkpoint,
not the RC3 source. Published RC2 tags and artifacts remain immutable.

> Every kernel must earn its performance claim. Every boundary must earn its cost. Every public workflow must pass end-to-end validation.

This principle applies across development: acquisition, validation, ownership,
planning, caches, bindings, transfer, kernels and reporting all need appropriate
evidence when their critical paths change. It does not require an unrelated
hardware campaign for every documentation change. The normative rule is in
[the contract](contract.md#performance-evidence-and-boundary-costs).

## Release-blocking boundary

This section describes the initial audited source, before the planned RC3 fix;
it is not a statement that the proposed native acquisition has already landed.

The existing Polars loader projects feature/target frames while preserving
source dtype. The adapter's raw Arrow convenience route supports only a narrow
Core/no-significance configuration with matching float dtypes. Default analysis
and other configurations can reach `feature_frame.iter_rows()`, iterable-to-list
conversion, scalar coercion, numeric buffer reconstruction and Python byte
serialization before Rust decodes the buffers.

This is an acquisition/execution-policy coupling. A file's ability to supply
Arrow buffers should not depend on whether a convenience analysis function can
express the user's complete configuration. The resident engine can be fast
while users still pay excessive latency and Python-object memory amplification
before reaching it.

The [installed RC2 baseline investigation](evidence/rc3-ingestion-baseline.md)
reproduces the row-materialization problem on 100k x 20 and 100k x 100 inputs.
Its timing/RSS observations remain investigative, not release-certified evidence.
The same record now retains matched installed-development-wheel samples and a
coarse negative-control cost gate. Exact-head hosted/frozen-artifact and
applicable backend acceptance gates are still required. Do not infer a
regression magnitude from source inspection alone.

## Minimal safe implementation boundary

Keep the existing public `dataload(path, target, features, *, config, ...)`
surface and configuration semantics. Separate private native Arrow acquisition
from execution; feed validated owned numeric input into the existing full-config
Rust planner/executor rather than constructing a second Arrow-specific engine.

```text
Polars read/project, preserving source numeric values
    -> Arrow C Stream import with explicit foreign-owner lifetime
    -> Rust schema/shape/null/range validation and checked dtype conversion
    -> Rust-owned profile-typed input and content identity
    -> existing full-config resident acquisition/cache or one-shot execution
    -> existing planner, backend route, significance and report construction
```

Rust retains validation, planning, backend eligibility, memory ownership and
lifecycle policy. Python remains a declarative binding/reporting layer. CUDA,
HIP and Metal retain their existing native ownership. Preserve precision,
metric/MI semantics, arbitrary-size planning seeds, fresh `None` entropy,
candidate identities/order, ranking, significance, warnings and fail-closed
errors. Existing explicit unsupported backend/profile requests must still fail
before inappropriate acquisition or execution.

The private acquisition boundary must:

- consume all Arrow batches/chunks without first materializing Python rows;
- validate lengths, counts, schema, checked allocation arithmetic and feature/
  target row alignment, including streams with different batch boundaries;
- preserve finite-versus-source-NaN/Inf distinctions: fp32/mixed reject finite
  values beyond `f32::MAX` before narrowing, while fp64 never stages through f32;
- retain the foreign owner until import/conversion completes, transfer each
  capsule exactly once, and release imported owners on success and error;
- finish a stable Rust-owned input snapshot before content fingerprinting or
  execution, rather than retaining mutable caller aliases as resident storage;
- reuse full-config continuous, time-series and decision-path state construction,
  including original typed inputs required for adaptive reselection; and
- reuse the established cache identity, target update and invalidation rules,
  not create a separate loader cache or change thread affinity/GIL behavior.

Keep a correct generic path for supported foreign inputs outside the optimized
numeric Arrow surface. Avoid an unexpectedly unbounded intermediate Python
object graph; do not silently reject previously supported inputs, reinterpret
nulls, narrow values in Polars, or introduce a new fallback backend to improve a
benchmark. Any required compatibility change is a separate maintainer decision.
The existing strict typed convenience boundary need not become a new policy
owner merely because the loader gains full-config acquisition.

## Ownership and copy ledger

Arrow import can borrow column buffers, but that is not resident zero-copy.
Validation, dtype conversion and layout conversion must produce stable owned
compute storage under the current contract. Aim to remove Python row objects,
scalar reconstruction and Python serialization, not promise an impossible
zero-copy layout.

An owned snapshot protects execution after acquisition; it does not promise an
atomic snapshot while an external owner concurrently mutates source buffers.
The caller must keep those buffers stable during acquisition. Do not expand
that existing ownership limitation into a new borrowed resident API in RC3.

The initial design feeds owned row-major input into existing executors. Core's
`CpuPrecisionMatrix::from_row_major_f32/f64` then transposes to owned
column-major storage. That is an additional existing full-feature copy;
acquisition and transpose may coexist transiently, and adaptive families may
retain original inputs. Do not hide this copy or label the whole pipeline
one-copy. GPU upload, host significance storage, generated-family expansion,
resident cache retention and compact report allocation must also be accounted
for where they apply. A direct column-major acquisition redesign is not
automatically part of RC3.

For each measured workflow, record input bytes and dtype/layout, owned buffers,
known transient/retained copies, cache state and relevant device transfers.
Separate source-derived byte estimates from measured counters. Python-only
allocation tracing cannot establish native allocation volume. Peak RSS is a
process high-water measure, not an exact sum of logical buffers or proof of
zero-copy.

## Existing evidence and the missing gate

| Existing gate | Useful evidence | Missing ingestion-cost evidence |
|---|---|---|
| `contract_02_feature_generation_reference.py` | Tiny CSV/direct numerical parity | Default/full-config coverage, realistic shapes, latency/RSS |
| `precision_01_end_to_end_profiles.py` | Tiny IPC/backend/profile parity | File-format/shape/cache performance matrix |
| `test_v1_ingest_contracts.py` | Seeds, caps, source dtype, finite overflow and report diagnostics | No-Python-row structural guard and resource regression |
| Core production benchmark | Candidate-parallel resident executor throughput | Reading, parsing, Arrow acquisition and pre-resident conversion |
| `perf_13_precision_profiles.py` | Prepared-input public analyze/resident/compiled lifecycle | Actual public CSV/Parquet/IPC workflow |
| `cold_lifecycle.py` | Fresh-process canonical GPU ABI phases | Polars/file ingestion and its memory amplification |

Retain those gates. Add a compact focused ingestion harness instead of expanding
the precision benchmark into another giant campaign. Reuse installed-package
identity checks, matched provenance, raw-sample reporting and honest combined/
unobservable phase labels. Existing resident evidence cannot satisfy the new
public-file gate.

The permanent collector is `tests/release_measure/perf_14_public_ingest.py`;
its [baseline record](evidence/rc3-ingestion-baseline.md#permanent-reproduction-path)
shows the matched installed-wheel command. Successful collection is not release
acceptance: it preserves raw samples and checks numerical route parity, while
cost budgets, artifact provenance and applicable hardware gates remain separate.

## Staged execution and acceptance

### 1. Reproduce and attribute

Use isolated installed RC2 and candidate environments with authenticated wheel
and native-member hashes. Generate deterministic fixture files outside measured
children. Run fresh subprocesses for memory samples and predeclare the matched
workload/configuration/order. Record file/input hash, version/source identity,
dependencies, toolchain, CPU/affinity/power state and raw results.

Observe file reading/parsing, projection, numeric validation, Python
materialization, Arrow import, owned conversion, resident/cache acquisition,
planning, numerical execution and report construction where separable. Leave
inseparable stages combined and inaccessible stages not observable. Do not
manufacture phase timings by subtraction. Record wall time, user/system CPU,
fresh-process peak RSS and a copy ledger; distinguish process-cold from cold
filesystem-cache state. Do not mutate OS cache/power settings or repeatedly
rerun measurements to obtain an attractive result.

### 2. Implement and prove correctness

Build the private full-config acquisition seam, then route supported numeric
files through it. Add Rust batch/dtype/shape/lifetime/error tests and installed
public API parity tests. Force the optimized path in tests so it cannot pass by
silently using rows. Preserve independent reference checks, not merely agreement
between two consumers of the same new implementation.

Cover all three profiles and formats, all requested metrics, default and
non-default significance/budgets, generated families, seed/cap identity, cache
miss/hit and target/feature invalidation. Validate expected backend selection
and unsupported-profile errors without hidden Core substitution. Keep existing
strict raw Arrow convenience tests where their compatibility contract remains
unchanged.

### 3. Establish permanent cost gates

The representative matrix includes CSV/Parquet/IPC, fp32/mixed/fp64, source
Float32/Float64, small/medium/wide data, 100k x 20 and 100k x 100 reference
shapes, bounded larger/adversarial cases, disabled/cold/repeated cache states,
and realistic configured Core workflows. Run actual defaults on an affordable
fixture; use explicitly disclosed bounded candidate/significance settings for
large ingestion-isolation cases rather than mislabelling them default.

Test nulls, NaN/Inf, exact f32 limits and just-above-limit finite values, signed
zero, underflow/subnormals, fp64 distinctions, empty/mismatched/chunked streams
and target alignment. For CSV, compare with the actual parsed/projected numeric
input or use exactly round-trippable fixtures; serialization differences are
not engine regressions.

Ordinary V1 CI enforces installed correctness, structural no-whole-dataset-
Python-materialization guards, and the bounded 18-cell reference latency/RSS
tripwire in `ingest_cost_budget.json`. Wider default/configured/family/cache and
larger-shape investigations remain dedicated bounded qualification, not another
giant Cartesian campaign. Derive explicit performance/memory budgets from
matched evidence rather than copying unrelated kernel thresholds. Keep all raw
samples and failures, and disclose any unobserved copy or phase.

### 4. Review, integrate and qualify exact RC3 wheels

Land durable fixes through focused mainline PRs with exact-head AI review,
strict configured checks and resolved conversations. Only after reviewed fixes
are on a green main may the RC3 release branch be cut under the existing
[release-branch policy](releases/release-branches.md). Prepare canonical RC3
identity through normal governance; do not repurpose the old stable checkpoint
or reuse a frozen bundle from a different source.

Before declaring RC3 ready, validate an installed exact candidate wheel outside
the checkout: default `dataload` uses native acquisition without Python row
materialization, full-config numerical/error parity passes, matched RC2/RC3
latency/RSS and copy costs are retained, and applicable accelerator routing/
physical correctness contracts remain green. Final artifact qualification uses
the new exact-source frozen bundle, normal provenance/checksum/composition and
release gates. No source-tree-only test can substitute for installed-wheel
evidence. Tagging/publication remains a separately authorized release action.

## Scope exclusions

No GIL detachment, thread-affinity redesign, cooperative cancellation, new public
API, ABI or numerical policy, backend/family redesign, Candidate IR, RT/OptiX
distribution, separate precision package, or performance-marketing claim is
authorized by this plan. [Issue #100](https://github.com/onlyxItachi/GAFIME/issues/100)
informs stable owned acquisition, not wholesale adoption of its future ownership
roadmap. Polars 2 and broader combined-streaming work remain in
[issue #87](https://github.com/onlyxItachi/GAFIME/issues/87) for the post-v1 line.
Permanent benchmarking work in
[issue #71](https://github.com/onlyxItachi/GAFIME/issues/71) remains open. Existing
security/release contracts are preserved; this plan claims no new scan result,
certified speedup or completed RC3 qualification.
