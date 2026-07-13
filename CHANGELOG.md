# Changelog

## v1.0.0a0 (pre-release readiness; not issued)

This is a pre-release readiness checkpoint, not a tagged or published release.

- Aligned public Python and native boundary version reporting to `1.0.0a0`
  (`1.0.0-alpha.0` in Cargo metadata).
- Added truthful public capability reporting for backend selection, graph mode,
  device significance, MI ceilings, Arrow ingest, RT availability, and family
  execution placement.
- Separated generated-family `gafime_cpu` placement from subsequent continuous
  scoring; no CUDA, ROCm, or Metal generation kernel is claimed for
  `time_series` or `decision_path`.
- Added the public `gafime --check` capability report and installed-package
  contract coverage.
- Added deterministic same-version discovery for installed CUDA/ROCm payloads
  and bundled the Metal dylib/metallib pair in the macOS arm64 base wheel.
- Hardened payload artifacts with complete contracted CUDA source staging,
  installed ABI/separation checks, `-O3` release builds, and proven
  `auditwheel` repair before ROCm receives a manylinux tag.
- Moved ROCm wheel compilation into the EL8-based `manylinux_2_28` baseline
  and reduced each CUDA/ROCm platform lane to one Python 3.10 stable-ABI wheel.
- Split Core, CUDA, and ROCm validation/publication dependencies while keeping
  the GitHub Release job gated on every supported artifact.

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
