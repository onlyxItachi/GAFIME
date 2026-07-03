# GAFIME v1 Contract

This contract defines the boundaries that GAFIME v1 implementation work must preserve. It is a maintainer policy document, not a performance tuning note. Passing tests does not make a boundary violation acceptable.

## Repository Layout

Tracked project source, runtime, test, and documentation content must converge into:

- `crates/`
- `python/gafime/`
- `tests/`
- `docs/`

Do not create new source, runtime, test, or documentation homes outside those roots. Required root metadata and bootstrap files may exist only when needed to build, package, discover, validate, or govern the repo. They must not hide backend implementation logic, fallback behavior, source orchestration, or runtime ownership.

`docs/` is the historical and design record. It may be read and extended, but historical docs must not be rewritten, deleted, or collapsed without maintainer approval.

`tests/` must preserve release-relevant tests used by previous releases and the current release. Release-gate tests must not be removed or relocated without maintainer approval.

Ignored local agent memory, release scratch, editor state, and Claude/agent skill artifacts are outside this tracked-layout rule. They must stay ignored and must not become runtime inputs.

## Kernel And Orchestration Layout

Kernel and orchestration work must keep device code, host launch code, and Rust interconnect boundaries separated by backend, file role, and compiler.

Target layout inside the GPU system crate:

```text
crates/gafime-gpu-sys/src/
  cuda/
    cuda_api.hpp
    kernels.cuh
    kernels.cu
    launcher.cu

  rocm/
    rocm_api.hpp
    kernels.hpp
    kernels.hip
    launcher.hip

  metal/
    metal_api.hpp
    shader.metal
    launcher.mm
```

CUDA `kernels.cu` owns CUDA `__global__` and `__device__` implementations. CUDA `launcher.cu` owns host launch, graph capture, and `<<< >>>` dispatch. `cuda_api.hpp` owns Rust-facing C ABI declarations.

ROCm `kernels.hip` owns HIP `__global__` and `__device__` implementations. ROCm `launcher.hip` owns host launch, graph capture, and `hipLaunchKernelGGL` dispatch. `rocm_api.hpp` owns Rust-facing C ABI declarations.

Metal `shader.metal` owns Metal device kernels. Metal `launcher.mm` owns Objective-C++ command encoder, pipeline state, and dispatch. `metal_api.hpp` owns Rust-facing C ABI declarations.

## Permitted Source Extensions

For kernel and orchestration source work, the permitted extensions are:

- `.rs`
- `.hpp`
- `.cuh`
- `.cu`
- `.hip`
- `.metal`
- `.mm`

Do not introduce any other source extension for kernel or orchestration logic without explicit maintainer approval. Build scripts, packaging metadata, documentation, tests, and distribution artifacts are outside this extension rule.

## Compiler Ownership

Compiler ownership is part of the backend contract.

- Rust `.rs` sources are owned by the Rust toolchain.
- CUDA `.cu` and `.cuh` sources are owned by NVCC.
- ROCm `.hip` and `.hpp` kernel/orchestration sources are owned by HIP/amdclang++.
- Metal `.metal` sources are owned by the Metal shading language compiler.
- Metal `.mm` sources are owned by the Objective-C++ compiler path.

Build rules and compiler flags for these sources may express only the required compiler chain, language mode, ABI/export shape, and source ownership. Do not introduce performance, lowering, tuning, backend-substitution, or undocumented compile flags without maintainer approval.

No agent may introduce a new compiler, source extension, build artifact, fallback path, backend boundary, or ownership transfer without explicit maintainer approval.

## Backend Ownership

Rust owns:

- Python boundary
- config and input validation
- orchestration
- scheduling
- `gafime.compile`
- feature planning
- CPU SIMD kernels
- backend selection
- safe public API
- memory ownership policy

CUDA owns:

- CUDA device kernels
- CUDA launcher functions
- CUDA graph capture/replay
- CUDA memory/runtime calls needed by CUDA execution

ROCm owns:

- HIP device kernels
- HIP launcher functions
- HIP graph capture/replay
- HIP graph/runtime calls needed by ROCm execution

Metal owns:

- Metal shader kernels
- Metal command encoder
- Metal pipeline state
- Metal dispatch logic

No backend may reimplement orchestration policy. No backend may silently fall back to another backend. No backend may own Python-facing behavior.

## Forbidden Cross-Boundary Calls

Rust must not:

- call CUDA, HIP, or Metal runtime APIs directly except through approved C ABI launchers
- contain GPU kernel code
- contain backend-specific GPU launch syntax
- perform unsafe ownership transfer across backend boundaries

Rust starts validated GPU launch work through backend C ABI surfaces. GPU host runtime code belongs in backend launcher files. CPU vector ISA work is not GPU launch syntax and is governed by the Rust safety policy.

CUDA, HIP, and Metal launchers must not:

- implement feature planning
- implement scheduler policy
- mutate Rust-owned config semantics
- introduce fallback behavior
- expose backend-specific types through Rust-facing API headers

Backend graph work may optimize backend-side execution, data movement, replay, and command submission. It must not replace Rust-owned orchestration, scheduling, feature planning, Python API semantics, or top-level user protocol handling.

Device kernels must not:

- call host functions
- own memory allocation policy
- encode orchestration decisions
- depend on Python or Rust data structures directly

No file may contain functionality outside its ownership section.

## Rust Safety

Rust code is safe by default.

No `unsafe`, `unsafe fn`, `unsafe impl`, raw-pointer ownership, unchecked indexing, transmute-style behavior, or broad FFI manipulation is allowed in planning, scheduling, reporting, Python bindings, or backend orchestration.

Allowed `unsafe` exceptions:

- CPU SIMD kernels where target-feature lowering requires it
- LLVM/compiler intrinsics
- unavoidable ABI/FFI boundary calls
- tightly scoped backend interconnect shims

Every `unsafe` block must:

- be isolated behind a safe Rust API
- document the safety invariant
- have focused tests
- avoid fallback behavior
- avoid implicit ownership transfer

The CPU performance-kernel exception does not permit unsafe fallback paths, ownership transfers, runtime shortcuts, broad backend rewrites, or backend orchestration policy.

## ABI Contract

Rust communicates with native backends only through approved C ABI surfaces. Backend launchers expose stable ABI. Backend types never leak into Python. Backend internal structs are private.

ABI changes must be intentional, documented, reviewed, and validated for Rust/C boundary compatibility and Python API compatibility.

## Numerical Policy

GAFIME targets bit parity with the approved reference implementation for every backend.

Integer, categorical, indexing, histogram, and all deterministic outputs require exact bit parity. Floating-point outputs are also expected to achieve bit parity whenever mathematically and architecturally possible.

If strict bit parity cannot be achieved because of unavoidable hardware or compiler differences, such as fused operations, ISA-specific instruction selection, or backend-defined floating-point behavior, the implementation must:

- explicitly document the reason
- justify why bit parity is impossible
- define the approved numerical tolerance
- prove equivalence through validation tests

Performance improvements are never accepted as a justification for undocumented numerical differences.

## Feature Generation Verification

Every PR that changes feature generation, feature expansion, candidate planning, or backend scoring must validate all public feature-generation families through the top-level Python API before backend-local claims are accepted.

The required public API verification set is:

- continuous base features and interaction candidates against a NumPy reference
- `gafime.compile(...).analyze()` against eager `GafimeEngine.analyze(...)`
- time-series lag, rolling-window, and velocity generated columns against a NumPy reference
- decision-path generated membership features against an independent scikit-learn tree reference
- `gafime.dataload(...)` Arrow/native ingest against direct top-level API analysis

These checks must run from an installed package or wheel, outside the checkout import path, so local source directories cannot shadow the user-space package. Unit-test counts such as `pytest 37/37` or `cargo test` are not sufficient unless the release-measure contract gates above also pass.

## PR, Main, And Release Gates

Implementation testing, review, and pushes normally happen on a feature branch and PR. `main` may receive implementation changes only after the work proves:

- numerical bit parity or explicitly approved numeric tolerance for the affected backend/metric
- compatibility with this contract
- verified tests and release gates for affected runtime surfaces
- a concrete beneficial update relative to the current implementation

A release must never:

- introduce silent backend behavior changes
- change numerical output without documentation
- weaken ownership rules
- weaken safety rules
- bypass PR gates
- introduce undocumented compile flags
- change ABI unexpectedly

Every PR and every commit inside a PR must pass GitHub workflows. Validation starts from the top-level Python API to guarantee user-space stability.

Each PR must validate:

- numerical correctness
- backend-local and end-to-end Python API performance
- ABI stability
- ownership, safety, compiler-chain, and extension-policy compatibility
- implementation documentation and design documentation

Every new backend feature must update `docs/`, `tests/`, and this contract.

## Regression Policy

No accepted optimization may:

- reduce correctness
- reduce numerical guarantees
- reduce backend compatibility
- reduce test coverage
- weaken architectural contracts

Every optimization must demonstrate a measurable benefit relative to the current implementation.

## Migration Rules

Do not treat placeholder GPU files as real runtime sources. Do not delete legacy backend/device code until the v1 structure carries required capability and equivalence tests pass.

Move or split real device-side code into the contracted backend layout before old backend connections are cut. Preserve roadmap, release notes, design docs, and agent contract files unless the maintainer explicitly asks for removal.

The v1 direction is Python -> PyO3/Rust -> Rust CPU / GPU C ABI. Python must not own continuous backend planning loops or GPU permutation loops. Rust owns candidate specs, compact result state, scheduling, and native backend dispatch. GPU backends expose explicit C ABI launcher surfaces to Rust and keep backend-specific kernel orchestration inside their contracted source trees.
