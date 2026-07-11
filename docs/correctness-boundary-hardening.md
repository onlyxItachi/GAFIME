# Correctness Boundary Hardening

This branch is the correctness follow-up to the merged CUDA/HIP kernel
hardening work. It does not claim a new performance result. The goal is to
make invalid ownership, malformed native inputs, stale ABI payloads, and
inconsistent significance estimators fail explicitly before release.

## Runtime Invariants

- `MatrixHandle` is borrowed across execution. Native resident matrices remain
  owned by `OwnedGpuMatrix`, and CPU handles cannot outlive their matrix.
- Compiled plans validate backend, family, arity, shape hints, descriptor
  coverage, feature indices, row counts, metric counts, and result capacity
  before a backend launch.
- Permutation maxT scans the complete screened candidate family while retaining
  compact descriptors and bounded selected-row output.
- A GPU-observed mutual-information statistic that falls back to a CPU null uses
  the same fixed-width estimator and the observed backend's supported bin
  ceiling. CPU adaptive MI keeps ties together and is row-order invariant.
- CUDA, ROCm, and Metal reject execution before a complete matrix upload.
  Target updates temporarily invalidate content and publish it only after the
  replacement is complete.
- CUDA and ROCm graph/cache state is committed transactionally and invalidated
  after failed replay or content changes.
- Native null pointers return `INVALID_ARGUMENT`; stale C-ABI versions return
  `ABI_MISMATCH`; a protocol for the wrong backend returns `INVALID_ARGUMENT`.
- CUDA decision-path batches reserve the terminal offset slot, cap path count at
  `UINT32_MAX / 4` for four-vertices-per-path indexing, widen launch-shape
  arithmetic before addition, and validate every derived host/device allocation
  size before dispatch.
- Windows CUDA/ROCm payload builds define `GAFIME_GPU_BUILDING_DLL`, so exported
  C-ABI functions are compiled as `dllexport` rather than `dllimport`.
- Python Arrow fallback preserves the complete engine configuration. Target
  inputs are exactly one column and null-free at both the Python and PyO3
  boundaries.
- Graph, compile-plan, backend, candidate-family, and significance identities
  are reported from actual runtime state rather than inferred intent.

## Release Gates

The installed-wheel smoke is copied outside the checkout and removes checkout
paths before import. It verifies the ABI3 native module, `PyInit_gafime_py`, the
required PyO3 surface, Arrow target rejection, known Pearson/R2 oracle values,
exact compile/eager values, and generated-family significance identities.

Backend release scripts distinguish three outcomes:

- a selected backend executes successfully;
- an unconfigured optional payload is reported as an explicit skip;
- a configured payload failure, or a selection with no executed backend, exits
  nonzero.

Wheel validation selects the actual `cp310-abi3` artifacts across Linux,
macOS, and Windows. Public truthfulness runs against the installed native
package. Eager/compiled interaction, permutation, stability, and final-decision
parity are part of the contract workflow.

## Verified On 2026-07-11

- `cargo test --workspace` with the RT-off SM89 CUDA payload on the RTX 4060
  Laptop GPU: 146 tests passed.
- Python suite: 74 passed, 2 hardware-dependent skips.
- Installed ABI3 wheel in an external Python 3.14 virtual environment: passed.
- Installed-wheel public truthfulness: 23 passed.
- Contracts `00` through `03`, compile plan/value parity, backend availability,
  backend end-to-end core smoke, and the v1 architecture gate: passed.
- CUDA RT-off SM89 release build and the real-device malformed ABI/overflow
  test: passed.
- ROCm gfx1150 release compilation: passed; no ROCm runtime execution was used
  as evidence in this turn.
- Metal non-Apple Objective-C++ fallback syntax: passed; Metal runtime behavior
  remains a macOS hardware gate.
- Windows export selection was preprocessed with `_WIN32` and the legacy build
  spelling; it resolves `GAFIME_GPU_API` to `__declspec(dllexport)`.

No performance benchmark or profiler capture was run for this branch. There
was therefore no profiler report for PerfDigest to compact. OptiX-enabled CUDA,
ROCm runtime, and Metal runtime validation remain external hardware/toolchain
gates.
