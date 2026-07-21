# GAFIME v1 Contract

This contract defines the boundaries that GAFIME v1 implementation work must preserve. It is a maintainer policy document, not a performance tuning note. Passing tests does not make a boundary violation acceptable.

## Repository Layout

Tracked project source, runtime, test, and documentation content must converge into:

- `crates/`
- `src/`
- `python/gafime/`
- `tests/`
- `docs/`

Do not create new source, runtime, test, or documentation homes outside those roots. Required root metadata and bootstrap files may exist only when needed to build, package, discover, validate, or govern the repo. They must not hide backend implementation logic, fallback behavior, source orchestration, or runtime ownership.

`docs/` is the historical and design record. It may be read and extended, but historical docs must not be rewritten, deleted, or collapsed without maintainer approval.

`tests/` must preserve release-relevant tests used by previous releases and the current release. Release-gate tests must not be removed or relocated without maintainer approval.

Ignored local agent memory, release scratch, editor state, and Claude/agent skill artifacts are outside this tracked-layout rule. They must stay ignored and must not become runtime inputs.

## Kernel And Orchestration Layout

Kernel and orchestration work must keep device code, host launch code, and Rust interconnect boundaries separated by backend, file role, and compiler.

Target layout inside the root native source tree:

```text
src/
  cuda/
    cuda_api.hpp
    kernels.cuh
    kernels.cu
    launcher.cu
    rt_kernels.cuh
    rt_kernels.cu
    rt_launcher.cuh
    rt_launcher.cu

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

CUDA RT-core / decision-path acceleration code must stay in the explicit RT files. `rt_kernels.cu` owns RT-specific CUDA device kernels, OptiX device programs, point-packing kernels, and exact-filter kernels. `rt_launcher.cu` owns RT-specific host allocation, finite box planning, conservative ordered-float-bucket custom-AABB preparation, cached OptiX IAS/GAS/workspace, exact SM fallback, and RT membership dispatch. The generic `kernels.cu` and `launcher.cu` must not absorb RT-specific device or host execution logic beyond the public C ABI bridge from the opaque matrix handle.

ROCm `kernels.hip` owns HIP `__global__` and `__device__` implementations. ROCm `launcher.hip` owns host launch, graph capture, and `hipLaunchKernelGGL` dispatch. `rocm_api.hpp` owns Rust-facing C ABI declarations.

Metal `shader.metal` owns Metal device kernels. Metal `launcher.mm` owns Objective-C++ command encoder, pipeline state, and dispatch. `metal_api.hpp` owns Rust-facing C ABI declarations.

GPU payload staging and release packaging must source backend files from this root `src/` layout. CUDA payloads must compile both `kernels.cu` and `launcher.cu`. CUDA payloads must also compile `rt_kernels.cu` and `rt_launcher.cu` when the RT path is enabled. OptiX RT builds may generate embedded PTX from `rt_kernels.cu`, but the source of truth remains the explicit RT CUDA source. ROCm payloads must compile both `kernels.hip` and `launcher.hip`. Packaging must not reintroduce `gpu/`, crate-local native source homes, kernel-only payload builds, placeholder device files, or hidden source copies under old runtime paths.

The standard PyPI CUDA payload is the immutable RT-off distribution
`gafime-cuda`, package `gafime_cuda`. The optional non-PyPI OptiX payload is
the distinct distribution `gafime-cuda-rt`, package `gafime_cuda_rt`; it must
also use a distinct native library filename. Automatic discovery may select
either variant, but must reject a dual installation unless
`GAFIME_CUDA_V1_LIB` explicitly selects one. RT artifacts are excluded from the
standard 11-artifact release bundle and every PyPI publishing job.

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

Build rules and compiler flags for these sources may express the required compiler chain, language mode, ABI/export shape, and source ownership.

Compiler flags fall into two classes, and a flag is judged only by whether it can change the reference numerical result — never by whether it makes the backend faster:

- **Permitted without separate approval — performance/optimization flags that do not change numerical results.** Standard optimization-level and code-generation flags are allowed because they optimize the compiled backend source without altering IEEE floating-point semantics or the reference result. Examples: `-O1`/`-O2`/`-O3` and `-Xptxas -O3` (NVCC), `-O1`/`-O2`/`-O3` (clang++/amdclang++/hipcc), `/O1`/`/O2` (MSVC), function inlining, loop unrolling, and `--generate-line-info`/`-lineinfo` for profiling.
- **Forbidden without explicit maintainer approval — math-breaking flags that change numerical results.** Any flag that relaxes IEEE semantics is forbidden because it breaks the f64/Kahan-accumulator parity oracle. Examples: `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`, `-fassociative-math`, `-freciprocal-math`, `-ffinite-math-only`, `-fno-signed-zeros`, `-ffp-contract=fast` (global FMA reassociation), flush-to-zero / denormals-are-zero (`-ftz=true`), approximate/fast-math transcendental intrinsics, `--use_fast_math` (NVCC), and `/fp:fast` (MSVC).

Backend-substitution flags, introduction of a new compiler, and undocumented ABI-changing flags remain forbidden without explicit maintainer approval.

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

CUDA may expose the optional `gafime_gpu_permutation_pvalues` ABI to compute permutation-test p-values for already-surfaced compact result rows in a target-independent family. A current payload that uses this path under an active `vram_budget_mb` must also expose the non-mutating `gafime_gpu_permutation_memory_peak` query. That query accounts for the complete-family score buffers and the retained observed, family-maximum, and exceedance buffers, including old-plus-new growth transitions. Older same-ABI payloads without the query remain loadable but must use the budgeted host-orchestrated maxT path instead of bypassing admission. Target-dependent adaptive families must repeat their exact device unary screening and shortlist construction for every permutation. Rust may orchestrate that bounded sequence through target replacement plus `gafime_gpu_execute`, provided every family maximum is obtained with device `top_k=1` ranking in both directions for signed metrics, only bounded rows cross the ABI, and the original target is restored or the artifact fails closed. `gafime_gpu_execute` still returns scores only; Rust owns exceedance counts and p-value calculation and must never infer a null maximum from a report-compacted subset.

CUDA may expose the optional `gafime_gpu_decision_path_membership` ABI for RT-core/GBDT acceleration. Rust remains the owner of decision-path discovery, feature planning, scheduling, and backend selection. The CUDA payload receives compact validated `GafimeDecisionPathTerm` descriptors and materializes hard-AND membership over the resident feature-major matrix with exact `<=`, `>`, and NaN-undetermined semantics. OptiX RT traversal is allowed only for finite <=3D box batches where exact semantics are preserved. Every supported shape uses a conservative ordered-float-bucket custom AABB for traversal culling and rechecks the original fp32 values and open/closed predicates in the intersection program; 3D retains the third coordinate in that exact guard even though the acceleration lattice uses two coordinates. The payload must query `OPTIX_DEVICE_PROPERTY_RTCORE_VERSION` and fail closed when it reports no RT-core support; architecture names are not capability proofs. Duplicate intersection callbacks must not duplicate membership or direct statistics. Otherwise CUDA must use its exact SM comparator or return unsupported when RT is explicitly required. The symbol is CUDA-only during the spike; ROCm, Metal, and older CUDA payloads must report unsupported by omitting the symbol, not by falling back to another backend.

CUDA may expose the optional `gafime_gpu_decision_path_score` ABI for compact RT-core/GBDT scoring. It accepts the same Rust-owned path descriptors plus metric ids and returns compact `GafimeResultTable` rows. During the spike this score ABI supports only Pearson and R2 for finite-feature decision paths; unsupported metrics must return unsupported, not fabricated zeros. CUDA may split a mixed-axis score batch into internal <=3D RT groups and direct modes must preserve exact feature-pair groups before widening compatible lower-dimensional work, but it must preserve original path order and must not move discovery, scheduling, or fallback policy out of Rust. CUDA may use an internal duplicate-safe device bitset or direct duplicate-safe traversal statistics, but it must not copy full path-major membership to host on the scoring path. Direct traversal statistics are opt-in through `GAFIME_CUDA_DECISION_PATH_RT_SCORE=direct`; target-wide statistics and centered per-path sums use double precision, but floating atomic order remains tolerance-checked at the approved `1e-4` spike threshold and must not become the default without maintainer approval. First-hit direct traversal statistics are opt-in through `GAFIME_CUDA_DECISION_PATH_RT_SCORE=firsthit` and are allowed only when CUDA proves every RT group is finite, bounded, 2D, and non-overlapping; otherwise CUDA must return unsupported instead of falling back or changing semantics.

The public compact score route is limited to the complete, untruncated unary base-plus-path family with Pearson and/or R2, finite RT-representable inputs, and no graph or significance request. Rust executes base unary rows through the normal CUDA plan, appends compact path rows in global candidate order, and retains fallback ownership. Every other public shape uses the established membership-expansion plus continuous-scoring path. An explicit require-RT policy must fail closed rather than select that fallback.

Rust exposes decision-path execution policy as `DecisionPathRtPolicy`. `AllowSmFallback` sends no RT-required flag and permits the CUDA payload to use its exact SM implementation when OptiX cannot execute the validated batch. `RequireRt` sends `GAFIME_DECISION_PATH_FLAG_REQUIRE_RT`; Rust must reject a missing decision-path symbol or a device that does not advertise `GAFIME_GPU_DEVICE_FLAG_OPTIX_RT` as unsupported before treating any fallback as successful. CUDA must also return unsupported rather than execute the SM path when the required flag reaches the payload.

OptiX program, custom-AABB GAS, and workspace state is owned per CUDA device. A device execution lock covers the complete RT membership or score operation, so same-device calls cannot race mutable OptiX state; different device ids never share a program, context, stream, GAS, or workspace. Every execution and teardown must establish the requested device with `cudaSetDevice` and restore the calling thread's previous device. SM decision-path row grids must use the queried `maxGridDimY` and tile with a 64-bit row offset instead of assuming the legacy 65,535-block `grid.y` limit.

CUDA payloads that own this RT cache expose the optional lifecycle symbol `gafime_gpu_decision_path_release_device_state(device_id)`. `gafime-gpu-sys` shares one cleanup owner per loaded payload and device and invokes the symbol only after the final owning matrix has been freed. This teardown synchronizes against in-flight RT execution and releases both custom-AABB geometry programs and their high-water device allocations. Direct C ABI owners must call the lifecycle symbol after freeing their final matrix for a device. Older payloads may omit the symbol and remain loadable; they simply do not provide explicit RT-cache teardown. The empty host registry container may remain process-lifetime to avoid calling OptiX after vendor-library shutdown; successful last-owner cleanup must erase its device state, and its remaining host bucket footprint is bounded by valid CUDA device ids.

An older CUDA payload that omits the lifecycle symbol may also predate native
same-device RT serialization. The current Rust host must serialize its
decision-path calls per payload and device so separate backend objects cannot
race that legacy mutable state. Current payloads with the lifecycle symbol keep
their native locking path and must not pay this compatibility mutex.

Arrow C Data / Arrow C Stream is the v1 framework-integration protocol. Polars is the external tabular compatibility layer for ingest and manipulation; GAFIME owns compute memory after validation and exports compact result tables over Arrow. Legacy DLPack/native-buffer export must not be reintroduced as a fallback or compatibility shortcut without explicit maintainer approval.

Prepared continuous plans may set `GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL` only
while Rust owns the descriptor buffers and guarantees that their contents stay
immutable until the resident matrix is uploaded or its target is updated. A
backend may reuse its uploaded descriptor copies only inside that content epoch.
CUDA, ROCm, and Metal must invalidate the descriptor cache on both matrix upload
and target update; calls without the flag must upload descriptors for every
execution. `GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL` is a legacy ABI 1.0
capability and is not sufficient to negotiate content identity. Current
payloads must also advertise
`GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION`, key retained descriptors by
the nonzero generation in launch-protocol `reserved[0]`, and treat generation
zero as upload-every-call. Rust must strip the launch hint and zero the
generation for a same-ABI payload that lacks the generation capability, even if
that payload advertises the legacy immutable bit. These hints must not change
the ABI layout or any mathematical result.

`GafimeGpuDeviceInfo.flags` is the stable device-capability bitset for platform-aware backend behavior. It may report unified memory, integrated/discrete placement, managed-memory support, high-bandwidth memory, AMD RDNA/CDNA family, Apple-family Metal devices, and whether the loaded CUDA payload contains the OptiX RT implementation. `reserved[0]` stores the portable architecture class, and `reserved[1..7]` store backend-local read-only capacity hints such as SM/gfx detail, shared/threadgroup memory, register budget, bus/cache details, and max threads. Backend launchers may use these runtime facts to choose cache, graph, memory, or storage-mode behavior inside their backend boundary. Rust may inspect them through the ABI but must not call vendor runtime APIs directly or infer undocumented backend types.

ROCm managed storage requires both integrated placement and an advertised
managed/concurrent-managed capability. A failed managed allocation must return
an error; it must not fall back to a device-only pointer while retaining a
host-accessible copy mode.

`backend="auto"` is a Rust-owned ranked resolver. It must rank usable GPU device payloads above CPU, then rank CPU vector ISA above scalar CPU (`AVX512 > AVX2 > SSE4.2/NEON > scalar`). A GPU candidate is usable only when its configured C ABI payload loads and `gafime_gpu_device_info` succeeds for the requested `device_id`. Explicit `cuda`, `rocm`/`hip`, and `metal` requests must not fall back to another backend.

`GafimeEngine.analyze()` may keep a bounded v1 resident-analyze cache for continuous workloads when `ComputeBudget.keep_in_vram` is true. The cache key must be content-derived from the validated fp32 feature matrix, feature names, backend/config payload, and native boundary identity; it must not depend only on Python object identity. A target-only change may reuse the resident feature matrix through the native `update_target` boundary. A feature-content change must compile/upload a new resident matrix. `GAFIME_V1_ANALYZE_CACHE_SIZE=0` and `keep_in_vram=False` must disable this public analyze cache. Cache eviction must close native artifacts and must not introduce backend fallback or numerical changes.

Metal uses the same `gafime_gpu_*` C ABI as CUDA and ROCm. The Metal shader implements continuous Pearson/R2, fixed-bin mutual information, and Spearman scoring; numerical parity against the reference is gated by Apple-hardware validation. Because Metal Shading Language has no fp64, Metal reductions accumulate in fp32; parity tolerances against CPU and CUDA/HIP must account for backend-specific precision and reduction order, then be measured and approved on Apple hardware. Metal mutual information clamps bins to <= 48 so the joint histogram fits threadgroup memory. Graph capture/replay and backend-native permutation replay remain unsupported on Metal; Rust-orchestrated target replacement plus exact Metal screening/ranking is the approved bounded maxT path. Unsupported Metal metrics, graph replay, missing Metal payloads, and unavailable Apple runtime support must return explicit errors through the boundary and must never silently route to CPU, Python, CUDA, or ROCm.

Metal host-side interaction centering must use the same f64 column-mean
accumulation and non-finite propagation semantics as CPU, CUDA, and ROCm. The
macOS gate must execute CPU-oracle parity for all four continuous metrics on
high-dynamic and NaN/Inf inputs plus multi-block ascending/descending top-k.
`GAFIME_METAL_PARITY_TOLERANCE=0.002` is a provisional fp32 guard, not an
approved release tolerance, until Apple-hardware evidence is reviewed.

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

`EngineConfig.mi_bins` is an adaptive maximum, not a fixed histogram request.
Continuous MI planning selects the largest template in
`2,4,8,12,16,24,32,48,64,96` for which `8 * bins^2 <= n_samples`; Metal
applies the same rule with a 48-bin ceiling, and 2 is the minimum fallback when
no template satisfies the density rule. CUDA, ROCm, and the CPU fixed-bin
parity path must consume the same selected shape. Unsupported maxima resolve
downward to the nearest template and must never silently expand to 96. The
`12/24/48` intermediate shapes retain the v0.4.1 per-joint-cell sample guard
while reducing quantization jumps; their ranking-stability benefit is enforced
by the public-API release-measure contract. Permutation and bootstrap
significance passes must use the same selected shape and estimator as the
observed MI score. Target-independent CUDA families may use the native compact
permutation ABI. Adaptive CUDA families and CUDA payloads without that optional
ABI, plus ROCm and Metal families, use Rust-orchestrated target replacement and
exact same-backend device ranking. Every permutation must repeat target-dependent
screening, reduce the complete family with bounded device `top_k=1` queries, and
restore the observed target or fail the compiled artifact closed. Each ranking
query binds only its selected metric; a transient metric descriptor must not
alias the prepared immutable descriptor generation. The host must probe
`supports_device_ranking` before selecting this route, and a successful
zero-row ranking result contributes negative infinity rather than becoming a
device error. CPU bootstrap
stability for a GPU observation uses fixed equal-width MI and preserves the
observed backend's template ceiling.

## Feature Generation Verification

Every PR that changes feature generation, feature expansion, candidate planning, or backend scoring must validate all public feature-generation families through the top-level Python API before backend-local claims are accepted.

The required public API verification set is:

- continuous base features and interaction candidates against a NumPy reference
- `gafime.compile(...).analyze()` against eager `GafimeEngine.analyze(...)`
- all time-series generated columns (lag, delta, velocity, acceleration, rolling mean, rolling std, rolling sum) against a NumPy reference
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

The v1 direction is Python -> PyO3/Rust -> Rust CPU / GPU C ABI. Python must not own continuous backend execution planning loops or GPU permutation loops. Rust owns candidate specs, compact result state, scheduling, and native backend dispatch. The packaged `gafime.compile.scenario` module is a bounded v0.5 compatibility projection only: it emits at most one metadata descriptor per configured arity, never materializes candidates, and is never passed to native execution. GPU backends expose explicit C ABI launcher surfaces to Rust and keep backend-specific kernel orchestration inside their contracted source trees.
