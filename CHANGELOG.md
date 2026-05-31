# Changelog

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
