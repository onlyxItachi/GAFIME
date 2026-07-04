# GAFIME Agent Contract

This file mirrors `AGENT.md`. Keep both files synchronized except for agent-specific notes that are explicitly needed.
The human-readable maintainer contract is `docs/contract.md`; this file is the agent-facing operational mirror.

## Kernel And Orchestration Source Contract

GAFIME v1 backend work must keep device code, host launch orchestration, and Rust interconnect boundaries separated by file role and compiler.

Target source layout for kernel/orchestration work inside the root native source tree:

```text
src/
  cuda/
    cuda_api.hpp      # Rust interconnect / extern C ABI declarations
    kernels.cuh       # CUDA-internal declarations for NVCC
    kernels.cu        # CUDA __global__ / __device__ implementations
    launcher.cu       # CUDA host launch, graph capture, <<<>>> dispatch

  rocm/
    rocm_api.hpp      # Rust interconnect / extern C ABI declarations
    kernels.hpp       # ROCm-internal declarations for amdclang++/HIP
    kernels.hip       # HIP __global__ / __device__ implementations
    launcher.hip      # HIP host launch, graph capture, hipLaunchKernelGGL

  metal/
    metal_api.hpp     # Rust interconnect / extern C ABI declarations
    shader.metal      # Metal device functions and kernels
    launcher.mm       # Objective-C++ Metal pipeline, encoder, dispatch
```

Host launch files may contain launch syntax and graph orchestration. Device kernel files own device functions and kernels. Rust-facing API headers own ABI declarations only.

GPU payload staging and release packaging must source backend files from this root `src/` layout. CUDA payloads must compile both `kernels.cu` and `launcher.cu`. ROCm payloads must compile both `kernels.hip` and `launcher.hip`. Packaging must not reintroduce `gpu/`, crate-local native source homes, kernel-only payload builds, placeholder device files, or hidden source copies under old runtime paths.

## Repository Layout

Tracked project source, runtime, test, and documentation content must converge into these roots:

- `crates/` and its subfolders
- `src/`
- `python/gafime/`
- `tests/`
- `docs/`

Do not add new source/runtime/test/documentation homes outside those roots. All needed artifacts must be moved into the relevant allowed root before old locations are disconnected.

`docs/` is the repo's historical and design record. Agents may read it and may add new documentation when the task requires it, but must not rewrite, delete, or collapse historical docs without explicit maintainer approval.

`tests/` must preserve release-relevant tests used by previous releases and the current release. Do not remove or relocate release-gate tests without explicit maintainer approval.

Required root metadata and bootstrap files may exist only when they are needed to build, package, discover, or govern the repo. They must not contain backend implementation logic, fallback behavior, source orchestration, or hidden runtime ownership.

Ignored local agent memory, release scratch, editor state, and Claude/agent skill artifacts are outside this tracked-layout rule, but they must stay ignored and must not become required runtime inputs.

## PR And Main Gate

Testing, review, and pushes for implementation work must happen on a feature branch and PR. Do not contribute implementation changes directly to `main`.

`main` may receive a change only after the PR proves:

- numerical bit-pair equality or explicitly approved numeric tolerance for the affected backend/metric
- policy compatibility with this contract
- verified tests and release gates for the affected runtime surfaces
- a concrete beneficial update relative to the current implementation

Do not merge, fast-forward, force-push, or otherwise land work on `main` while any of those checks are unsatisfied. Passing a narrow test subset is not enough when boundary policy, layout policy, or numerical parity remains unverified.

## Release Behavior

A release must never:

- introduce silent backend behavior changes
- change numerical output without documentation
- weaken ownership rules
- weaken safety rules
- bypass PR gates
- introduce undocumented compile flags
- change ABI unexpectedly

Every new backend feature must update `docs/`, `tests/`, and this contract.

## ABI Contract

Rust communicates with native backends only through approved C ABI surfaces. Backend launchers expose stable ABI. Backend types never leak into Python. Backend internal structs are private.

ABI changes must be intentional, documented, reviewed through PR, and validated for Rust/C boundary compatibility and Python API compatibility.

CUDA may expose the optional `gafime_gpu_permutation_pvalues` ABI to compute permutation-test p-values for already-surfaced compact result rows. The symbol is optional so older payloads and non-CUDA backends remain loadable, but a payload that omits it must be treated as unsupported for native GPU p-values. `gafime_gpu_execute` returns observed scores only; permutation/null statistics must be returned through an explicit significance ABI surface, not inferred from discarded backend work.

`GafimeGpuDeviceInfo.flags` is the stable device-capability bitset for platform-aware backend behavior. It may report unified memory, integrated/discrete placement, managed-memory support, high-bandwidth memory, AMD RDNA/CDNA family, and Apple-family Metal devices. `reserved[0]` stores the portable architecture class, and `reserved[1..7]` store backend-local read-only capacity hints such as SM/gfx detail, shared/threadgroup memory, register budget, bus/cache details, and max threads. Backend launchers may use these runtime facts to choose cache, graph, memory, or storage-mode behavior inside their backend boundary. Rust may inspect them through the ABI but must not call vendor runtime APIs directly or infer undocumented backend types.

`backend="auto"` is a Rust-owned ranked resolver. It must rank usable GPU device payloads above CPU, then rank CPU vector ISA above scalar CPU (`AVX512 > AVX2 > SSE4.2/NEON > scalar`). A GPU candidate is usable only when its configured C ABI payload loads and `gafime_gpu_device_info` succeeds for the requested `device_id`. Explicit `cuda`, `rocm`/`hip`, and `metal` requests must not fall back to another backend.

Metal uses the same `gafime_gpu_*` C ABI as CUDA and ROCm. The Metal shader implements continuous Pearson/R2, fixed-bin mutual information, and Spearman scoring; numerical parity against the reference is pending Apple-hardware validation. Because Metal Shading Language has no fp64, Metal reductions accumulate in fp32 (a documented tolerance vs the f64 CUDA/CPU oracle, to be measured and approved on Apple hardware), and Metal mutual information clamps bins to <= 48 so the joint histogram fits threadgroup memory. Graph capture/replay and permutation replay remain unsupported on Metal. Unsupported Metal metrics, graph/permutation replay, missing Metal payloads, and unavailable Apple runtime support must return explicit errors through the boundary and must never silently route to CPU, Python, CUDA, or ROCm.

## Numerical Policy

GAFIME targets bit parity with the approved reference implementation for every backend.

Integer, categorical, indexing, histogram, and all deterministic outputs require exact bit parity. Floating-point outputs are also expected to achieve bit parity whenever mathematically and architecturally possible.

If strict bit parity cannot be achieved because of unavoidable hardware or compiler differences, such as fused operations, ISA-specific instruction selection, or backend-defined floating-point behavior, the implementation must:

- explicitly document the reason
- justify why bit parity is impossible
- define the approved numerical tolerance
- prove equivalence through validation tests

Performance improvements are never accepted as a justification for undocumented numerical differences.

CPU fixed-bin mutual information is the CPU parity path for the GPU-compatible MI approximation. Its SIMD implementation must preserve exact fixed-bin histogram counts against the scalar/index reference, keep the same finite-sample correction and normalization, and stay gated by release-measure architecture checks plus focused Rust tests.

## Feature Generation Verification

Every PR that changes feature generation, feature expansion, candidate planning, or backend scoring must validate all public feature-generation families through the top-level Python API before backend-local claims are accepted.

Required public API verification includes continuous base/interaction candidates against NumPy, `gafime.compile(...).analyze()` against eager analysis, all time-series generated columns (lag, delta, velocity, acceleration, rolling mean, rolling std, rolling sum) against NumPy, decision-path membership features against an independent scikit-learn tree reference, and `gafime.dataload(...)` Arrow/native ingest against direct top-level API analysis.

Arrow C Data / Arrow C Stream is the v1 framework-integration protocol. Polars is the external tabular compatibility layer for ingest/manipulation; GAFIME owns compute memory after validation and exports compact result tables over Arrow. Legacy DLPack/native-buffer export must not be reintroduced as a fallback or compatibility shortcut without explicit maintainer approval.

These checks must run from an installed package or wheel outside the checkout import path. Unit-test counts such as `pytest 37/37` or `cargo test` are not sufficient without the release-measure contract gates.

## PR Validation

Every PR and every commit inside a PR must successfully pass GitHub workflows.

Validation always starts from the top-level Python API to guarantee user-space stability.

Each PR must validate:

- numerical correctness: bit parity against the approved reference implementation whenever possible; otherwise, documented and approved numerical tolerance
- performance: backend-local benchmarks, end-to-end Python API benchmarks, and performance reports for every changed critical path
- ABI stability: public ABI compatibility, Rust/C boundary compatibility, and Python API compatibility
- contract compatibility: ownership rules, safety rules, compiler-chain rules, and extension policy
- documentation: implementation documentation, a reference to the design location, and at least one Markdown design document

After merge, relevant design and implementation information must be integrated into `docs/`. Temporary PR-specific documentation may then be removed only after the durable `docs/` record exists.

## Regression Policy

No accepted optimization may:

- reduce correctness
- reduce numerical guarantees
- reduce backend compatibility
- reduce test coverage
- weaken architectural contracts

Every optimization must demonstrate a measurable benefit relative to the current implementation.

## Backend Ownership

Rust owns:

- Python boundary
- config/input validation
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

## Compiler Ownership

Compiler ownership is part of the backend contract, not an optimization preference.

- Rust `.rs` sources are owned by the Rust toolchain.
- CUDA `.cu` and `.cuh` kernel/orchestration sources are owned by NVCC.
- ROCm `.hip` and `.hpp` kernel/orchestration sources are owned by the HIP/amdclang++ toolchain.
- Metal `.metal` sources are owned by the Metal shading language compiler.
- Metal `.mm` launch/orchestration sources are owned by the Objective-C++ compiler path.

Build rules and compiler flags for these sources may express the required compiler chain, language mode, ABI/export shape, and source ownership.

Compiler flags fall into two classes, governed differently. The distinction is numerical, not performance-vs-not: a flag is judged by whether it can change the reference numerical result, never by whether it makes the backend faster.

- **Permitted without separate approval — performance/optimization flags that do not change numerical results.** Standard optimization-level and code-generation flags are allowed because they optimize the compiled backend source without altering IEEE floating-point semantics or the reference result. Examples: `-O1`/`-O2`/`-O3` and `-Xptxas -O3` (NVCC), `-O1`/`-O2`/`-O3` (clang++/amdclang++/hipcc), `/O1`/`/O2` (MSVC), function inlining, loop unrolling, and `--generate-line-info`/`-lineinfo` for profiling.
- **Forbidden without explicit maintainer approval — math-breaking flags that change numerical results.** Any flag that relaxes IEEE semantics is forbidden because it breaks the f64/Kahan-accumulator parity oracle. Examples: `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`, `-fassociative-math`, `-freciprocal-math`, `-ffinite-math-only`, `-fno-signed-zeros`, `-ffp-contract=fast` (global FMA reassociation), flush-to-zero / denormals-are-zero (`-ftz=true`), approximate/fast-math transcendental intrinsics, `--use_fast_math` (NVCC), and `/fp:fast` (MSVC).

Backend-substitution flags, introduction of a new compiler, and undocumented ABI-changing flags remain forbidden without explicit maintainer approval.

## Forbidden Cross-Boundary Calls

Rust must not:

- call CUDA, HIP, or Metal runtime APIs directly except through approved C ABI launchers
- contain GPU kernel code
- contain backend-specific GPU launch syntax
- perform unsafe ownership transfer across backend boundaries

Rust's GPU role is to start validated GPU launch work through the backend C ABI. GPU host runtime code belongs to the backend launcher files. CPU vector ISA work is not GPU launch syntax and is governed by the Rust safety rules.

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
- not introduce fallback behavior
- not transfer ownership implicitly

The CPU performance-kernel exception does not permit unsafe fallback paths, ownership transfers, runtime shortcuts, broad backend rewrites, or backend orchestration policy.

## Functionality Placement

Feature planning lives only in Rust. Backend selection lives only in Rust. Input validation lives only in Rust. Memory ownership policy lives only in Rust.

CUDA, HIP, and Metal launchers only translate validated Rust requests into backend execution. CUDA, HIP, and Metal kernels only compute.

No file may contain functionality outside its ownership section.

Passing tests does not make a boundary violation acceptable. Tests and release gates must be hardened around these rules. If a change requires crossing ownership, compiler, extension, or safety boundaries, stop and ask for maintainer approval.

## Permitted Extensions

For kernel and orchestration source work, the permitted extensions are:

```text
.rs
.hpp
.cuh
.cu
.hip
.metal
.mm
```

Do not introduce other source extensions for kernel/orchestration logic unless there is a concrete technical requirement that cannot be satisfied inside the allowed set. Build scripts, packaging metadata, documentation, tests, and distribution artifacts are outside this extension rule.

If an exception becomes unavoidable, document why before adding it. Example: a required DLPack export source format may need a separate justification.

## Migration Rules

- Do not treat placeholder GPU files as real runtime sources.
- Do not delete legacy backend/device code until the v1 structure carries the required capability and equivalence tests pass.
- Move or split real device-side code into the contracted backend layout before cutting old backend connections.
- Preserve project memory and planning artifacts such as roadmap/docs/agent files unless the user explicitly asks for removal.
- Keep the v1 runtime separate from legacy fallback paths, but avoid broad cleanup commits that erase source history needed for parity.
- Prefer small, reviewable checkpoints with a clean rollback path.

## Backend Intent

The v1 direction is Python -> PyO3/Rust -> Rust CPU / GPU C ABI. Python should not own continuous backend planning loops or GPU permutation loops. Rust should own candidate specs, compact result state, scheduling, and native backend dispatch. GPU backends should expose explicit C ABI surfaces to Rust and keep backend-specific kernel orchestration inside their contracted source trees.
