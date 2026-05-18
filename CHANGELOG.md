# Changelog

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

