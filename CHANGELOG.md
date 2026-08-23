# Changelog

## Unreleased

- Reject negative `GafimeSelector.k`, non-positive streaming batch sizes, and
  non-positive streaming benchmark counts before selection or input reads.
- Reorganized the public README, documentation index, and release index around
  reader intent, and made publication guidance route through mutable release
  status instead of self-invalidating prose.
- Refined project metadata and public repository discovery information for the
  v1 release candidate train.

## v1.0.0-beta.2 (unreleased checkpoint)

Prepared GAFIME v1 beta.2 release record. The qualified source and frozen build
were retained as an unreleased pre-RC checkpoint rather than rebuilt solely to
replace publication-state wording.

- Added public interaction-materialization overflow diagnostics across Core,
  CUDA, ROCm, and Metal without eagerly paying the diagnostics cost on every
  report.
- Reused exact-shape diagnostics on resident execution and kept diagnostic
  device kernels in backend kernel sources rather than launch orchestration.
- Enforced production unsafe invariants and a zero-warning workspace Clippy
  gate.
- Fused CPU covariance finite checks while preserving early non-finite
  rejection, and added a measured branchless AVX2 fixed-bin MI conversion path.
- Replaced ad hoc release-artifact knowledge with one checked manifest that
  derives the dedicated CPython/platform artifact matrix.
- Restored the permanent package architecture after the aborted b1 checkpoint:
  Metal is embedded in the Apple Silicon Core wheel; CUDA and ROCm remain
  separate payload distributions.
- Removed Python Stable ABI packaging and Core payload extras. Core has no
  CUDA/ROCm dependency; payloads require the exact matching Core version.
- Split build/freeze from publication. The publisher consumes immutable bytes
  in Core-first order, verifies public installations, and creates the GitHub
  Release last.
- Restricted RT/OptiX to local CMake builds and removed bundled ROCm userspace
  and bundled CUDA runtime libraries from every distribution path.
- Established a permanent strict mapping between SemVer repository identity
  `1.0.0-beta.2` and PEP 440 Python identity `1.0.0b2`.
- Added an abandoned-partial-publication runbook and live PyPI status verifier
  so stranded payload releases are yanked rather than deleted or silently
  treated as complete.
- Replaced independent public storage/compute knobs with the keyword-only
  `precision` profile: lane-wide `fp32`, default `mixed` (fp32
  storage/pointwise with fp64 statistics/results), and end-to-end `fp64`.
  Core, CUDA, and ROCm carry all three profiles in their existing artifacts;
  embedded Apple Metal carries fp32 only and fails closed for mixed/fp64.
- Preserved the frozen ABI 1.0 surface through shared modern internals and
  established one canonical generic numeric-route ABI 1.1 surface, with an
  additive synthetic ABI 1.2 compatibility fixture.
- Made production Core scoring candidate-parallel through Rayon with
  worker-local precision scratch, per-candidate SIMD/native arithmetic,
  guarded scalar fallbacks, and deterministic result and ranking order.
- Established the pre-RC security policy, private-reporting path, threat model,
  and standard-scan baseline. Release-blocking findings were closed with
  explicit unsafe raw-descriptor contracts, canonical tar/ZIP member
  validation, CUDA/ROCm caller-device restoration, and strict workflow-input
  handling.

## v1.0.0b1 (2026-07-26, aborted)

Tagged packaging checkpoint that did not complete as a Core or GitHub release.

- Published CUDA and ROCm payload files before the attempted Metal lane failed.
- Correctly withheld the exact-version Core package and GitHub Release after
  the dependency failure.
- Requires release-level yanks for the stranded CUDA and ROCm payloads because
  their exact-version Core dependency was never published.
- Changed the standard `gafime-rocm` policy from a bundled userspace wheel to a
  thin system-ROCm payload requiring `libamdhip64.so.7`.
- Kept the truthful raw Linux ROCm wheel in the GitHub Release and restricted
  its PyPI lane to the matching sdist instead of applying a false manylinux tag.
- Tested the checkpoint's historical `cp310-abi3` platform/payload wheels on
  CPython 3.10 through 3.14. Beta.2 supersedes that Stable ABI model with
  dedicated interpreter wheels.
- Rejected the attempted 13-artifact separate-Metal model. Beta.2 preserves
  Metal in the macOS arm64 Core wheel and replaces fixed artifact totals with
  the manifest-derived per-CPython matrix.
- Preserved all numerical and kernel behavior from `v1.0.0b0`.

## v1.0.0b0 (2026-07-22)

Second public prerelease of the GAFIME v1 native runtime and split backend
distribution.

- Preserved non-finite correlation failures as NaN and excluded them from GPU
  ranking instead of clamping them to plausible Pearson/R2 endpoints.
- Added conservative magnitude admission and three-pass scale-normalized
  covariance for exponent-risky CUDA, ROCm, and Metal descriptor chunks while
  preserving the established ordinary-range path.
- Added compile-time `fast` and `fp64` mutual-information arithmetic policies
  for CUDA and ROCm. Distributed payloads retain the fast fp32 policy; local
  builds can select fp64 contribution, logarithm, reduction, correction, and
  normalization arithmetic without carrying two kernel sets.
- Added a public precision contract that separates storage dtype from compute
  policy and reports requested/effective precision, per-metric accumulator
  widths, normalization, and explicit rejection reasons.
- Made the ROCm bundled-wheel policy explicit and immutable, with pinned build
  inputs, component/license metadata, SBOM and ELF-closure validation, size
  ceilings, deterministic policy reports, and clean installed-wheel checks.
- Serialized the integrated GPU-enabled Rust workspace release gate to prevent
  intermittent ROCm context contention without reducing package-local test
  parallelism.

## v1.0.0a0 (2026-07-22)

First public alpha release of the GAFIME v1 native runtime and split backend
distribution.

- Aligned public Python and native boundary version reporting to `1.0.0a0`
  (`1.0.0-alpha.0` in Cargo metadata).
- Added truthful public capability reporting for backend selection, graph mode,
  device significance, MI ceilings, Arrow ingest, RT availability, and family
  execution placement.
- Separated generated-family `gafime_cpu` placement from subsequent continuous
  scoring; no CUDA, ROCm, or Metal generation kernel is claimed for
  `time_series` or `decision_path`.
- Added distinct execution-path behavior for cache-disabled one-shot, resident
  eager, and explicit compiled runs, with separate cache and content-digest
  paths for each.
- Added state-aware memory-admission hardening with compatibility-aware
  forecast fallbacks for CUDA/ROCm/Metal admission and retained-significance
  pathways.
- Added the public `gafime --check` capability report and installed-package
  contract coverage.
- Added explicit truthful placement disclosure for generation, scoring, and
  significance execution, with family-level `FamilyCapability` significance
  disclosure (including decision-path permutation limits) instead of implying
  backend-specific support.
- Hardened CPU/GPU/RT algorithmic paths (safer CUDA launch-policy and
  Spearman cache behavior, RT boxed-grouping safety, and first-hit duplicate
  mask correction in RT scoring).
- Kept production CPU continuous scoring on one reusable interaction vector plus
  SIMD slice kernels after fused higher-arity CPU candidate fusion was benchmarked
  and rejected.
- Added deterministic same-version discovery for installed CUDA/ROCm payloads
  and bundled the Metal dylib/metallib pair in the macOS arm64 base wheel.
- Modularized Rust boundary ownership across FFI and Python execution layers for
  clearer ABI surfaces and release-oriented test partitioning.
- Formalized standard RT policy as RT-off for default CUDA payloads and kept
  RT-on artifacts as optional, separate distribution lane (`gafime-cuda-rt`,
  non-PyPI) with explicit runtime selection.
- Hardened payload artifacts with complete contracted CUDA source staging,
  installed ABI/separation checks, cross-platform CUDA C++20, `-O3` release
  builds, and proven `auditwheel` repair before ROCm receives a manylinux tag.
- Moved ROCm wheel compilation into the EL8-based `manylinux_2_28` baseline
  and reduced each CUDA/ROCm platform lane to one Python 3.10 stable-ABI wheel.
- Split Core, CUDA, and ROCm validation/publication dependencies while keeping
  the GitHub Release job gated on every supported artifact.
- Added PR #21 release-hardening behavior: payload-first -> Core -> GitHub
  Release ordering, fail-closed collision handling with hash-verified recovery,
  serialized publication jobs, GitHub alpha prerelease marking, and a recovery
  release path that requires the version tag plus all three PyPI lanes.
- Refreshed the v1 practice notebook and tracked support skills, added a release
  operations runbook, and made their current API/recovery contracts
  machine-checked in the release measurement suite.
- Made `GafimeSelector` cloneable through the scikit-learn estimator parameter
  contract so the documented cross-validation pipeline executes per fold.

## v0.5.0-legacy (GitHub-only checkpoint)

GAFIME v0.5.0-legacy preserves the v0.5 compile/orchestration development line
as a GitHub-only checkpoint. It is not a PyPI release.

- Added `gafime.compile` / `CompiledGafime` API groundwork.
- Added Rust-backed scenario planning through the existing PyO3 helper crate.
- Added resident-session, graph, export, decision-path, telemetry, and compact
  native-report work across the v0.5 integration branches.
- Deprecated the v0.5 architecture as the long-term direction after profiling
  showed Python/session-loop and result-materialization overheads require a
  Rust-owned orchestration rewrite.

See `docs/releases/v0.5.0-legacy.md` for the full checkpoint notes.

## v0.4.7

Development work for v0.4.7 adds an explicit ROCm/HIP native backend path.

- Added `libgafime_rocm.so` HIP kernel build support for Linux x86_64 local
  development builds.
- Added `NativeRocmBackend` and explicit `backend="rocm"` / `backend="hip"`
  resolution.
- Added payload-aware `backend="auto"` routing for ROCm installs as
  `rocm -> core`.
- Ported the CUDA-like native paths to HIP for continuous global matrix
  batches, local bucket time-series batches, soft discrete scoring, and
  adaptive discrete selector scoring.
- Added ROCm platform capability reporting based on HIP runtime properties.
  GAFIME does not infer AMD product families from ROCm target names; those
  strings remain build/diagnostic metadata only.
- Added explicit UMA host-mapped input mode for shared-system-memory AMD
  integrated GPUs. Broad matrix scans and local bucket/time-series scans use
  page-aware HIP host registration when available and fall back per buffer.
- Added ROCm tests covering arity `1..5`, discrete soft/selector paths,
  hard-mode rejection, time-series bucket scoring, and an end-to-end engine
  smoke.
- Documented local ROCm build controls and validation evidence in
  `docs/v0.4.7-rocm-native-backend.md`.
- Documented the vendor GPU payload package policy: `gafime` remains the
  stable Python/Core package, while CUDA and ROCm binaries are distributed
  through explicit payload packages such as `gafime-cuda` and `gafime-rocm`.
- Updated backend-selection documentation for separated base, CUDA payload, and
  ROCm payload install modes.
- Updated the long API reference notebook, compact tutorial notebook, Docker
  development images, and maintainer skills for the v0.4.7 release candidate.

## v0.4.1

GAFIME v0.4.1 corrects mutual-information math and split-aware discrete
candidate ranking while preserving the v0.4.0 public API.

- Changed `EngineConfig.mi_bins` default from fixed `16` behavior to adaptive
  maximum `96`.
- Added adaptive dense MI with quantile/rank bins and finite-sample bias
  correction for report metrics.
- Replaced discrete selector's sparse mask-target MI with soft-binary
  inside/outside MI.
- Added effective-support guards for discrete MI and variance-reduction
  selector scores, with adaptive small-sample support floors.
- Added CUDA native adaptive discrete selection API
  `gafime_discrete_selection_adaptive_cuda`.
- Added Rust homogeneous execution-template batching for adaptive CUDA MI
  selector launches.
- Added edge-case validation covering sklearn MI agreement, exact binary MI,
  noise floors, monotonic signal strength, and CUDA template parity.
- Added industry-standard validation against sklearn tree-stump variance
  reduction, sklearn linear-model R2, and Ridge/GBM CV gain ordering.
- Aligned Rust helper crate/module version metadata to `0.4.1` and hardened
  sdists so local native binaries cannot leak into source distributions.
- Updated tests, docs, skills, and release notes for the corrected math path.

## v0.4.0

GAFIME v0.4.0 adds the discrete function representation family inside the
existing engine API.

- Added `EngineConfig.enable_discrete_functions` and related discrete controls.
- Added discrete candidate budget controls to `ComputeBudget`.
- Added soft threshold, soft interval, value-gated threshold, soft rectangle,
  and value-in-soft-rectangle candidate families.
- Added split-aware discrete candidate ranking.
- Added CUDA/Metal soft-mode support and CPU/NumPy hard-mode support.
- Added GPU hard-mode rejection for branch-heavy hard discrete feature
  engineering.
- Added `from gafime import subfunctions` for Rust helper APIs.
- Added cache-local Rust-side batching and CUDA profiling notes.
- Added discrete application benchmarks and v0.4.0 API notebook updates.
- Updated wheel workflow for CUDA 13.2 and CPython 3.10-3.14.
