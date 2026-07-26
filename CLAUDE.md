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
    rt_kernels.cuh    # CUDA RT/decision-path declarations for NVCC
    rt_kernels.cu     # CUDA RT/decision-path __global__ / __device__ implementations
    rt_launcher.cuh   # CUDA RT/decision-path host-launch declarations
    rt_launcher.cu    # CUDA RT/decision-path host launch, OptiX, geometry dispatch

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

CUDA RT-core / decision-path acceleration code must stay in the explicit RT files. `rt_kernels.cu` owns RT-specific CUDA device kernels, OptiX device programs, point-packing kernels, grouped point-packing kernels, and exact-filter kernels. `rt_launcher.cu` owns RT-specific host allocation, finite box planning, conservative ordered-float-bucket custom-AABB preparation, instanced IAS/GAS grouped dispatch, resident IAS/GAS geometry caching, cached OptiX workspace, exact SM fallback, and RT membership dispatch. The generic CUDA metric files must not absorb RT-specific device or host execution logic beyond the public C ABI bridge from the opaque matrix handle.

GPU payload staging and release packaging must source backend files from this root `src/` layout. CUDA payloads must compile `kernels.cu`, `rt_kernels.cu`, `launcher.cu`, and `rt_launcher.cu`. OptiX RT builds may generate embedded PTX from `rt_kernels.cu`, but the source of truth remains the explicit RT CUDA source. ROCm payloads must compile both `kernels.hip` and `launcher.hip`. Packaging must not reintroduce `gpu/`, crate-local native source homes, kernel-only payload builds, placeholder device files, or hidden source copies under old runtime paths.

The standard immutable RT-off CUDA distribution is `gafime-cuda`, package
`gafime_cuda`. The optional non-PyPI OptiX distribution is
`gafime-cuda-rt`, package `gafime_cuda_rt`, and must use a distinct native
library filename. Automatic discovery must reject a dual installation unless
`GAFIME_CUDA_V1_LIB` explicitly selects one. RT artifacts must remain outside
the standard 13-artifact release bundle and every PyPI publishing job.

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

CUDA may expose the optional `gafime_gpu_permutation_pvalues` ABI to compute permutation-test p-values for already-surfaced compact result rows in a target-independent family. The symbol is optional so older payloads and non-CUDA backends remain loadable. Target-dependent adaptive families must repeat their exact device unary screening and shortlist construction for every permutation. Rust may orchestrate that bounded sequence through target replacement plus `gafime_gpu_execute`, provided every family maximum is obtained with device `top_k=1` ranking (both directions for signed metrics), only bounded rows cross the ABI, and the original target is restored or the artifact fails closed. Each ranking pass must bind only its selected metric without aliasing the prepared immutable descriptor generation, probe device-ranking capability before selecting the route, and treat a successful zero-row result as negative infinity. `gafime_gpu_execute` still returns scores only; Rust owns the exceedance counts and p-value calculation and must never infer a null maximum from a report-compacted subset.

CUDA may expose the optional `gafime_gpu_decision_path_membership` ABI for RT-core/GBDT acceleration. Rust remains the owner of decision-path discovery, feature planning, scheduling, and backend selection. The CUDA payload receives compact validated `GafimeDecisionPathTerm` descriptors and materializes hard-AND membership over the resident feature-major matrix with exact `<=`, `>`, and NaN-undetermined semantics. OptiX RT traversal is allowed only for finite <=3D box batches where exact semantics are preserved. Every 1D, 3D, unbounded, empty, narrow, or otherwise ineligible shape uses a conservative ordered-float-bucket custom AABB for traversal culling and rechecks the original fp32 values and open/closed predicates in the intersection program. A grouped 2D score plan may use fixed-function instanced triangles only when every box is finite and bounded and each axis span is at least `2^-12 * max(1, abs(lo), abs(hi))`; triangle bounds expand by eight binary32 ULPs and any-hit rechecks the same original predicates before accepting a hit. Geometry selection is internal, signature-keyed, and must not be controlled by an environment selector. Three-dimensional paths keep their third coordinate in the exact guard even though the custom acceleration lattice uses two coordinates. The payload must query `OPTIX_DEVICE_PROPERTY_RTCORE_VERSION` and fail closed when it reports no RT-core support; architecture names are not capability proofs. Duplicate intersection callbacks must not duplicate membership or direct statistics. Otherwise CUDA must use its exact SM comparator or return unsupported when RT is explicitly required. The symbol is CUDA-only during the spike; ROCm, Metal, and older CUDA payloads must report unsupported by omitting the symbol, not by falling back to another backend.

CUDA may expose the optional `gafime_gpu_decision_path_score` ABI for compact RT-core/GBDT scoring. It accepts the same Rust-owned path descriptors plus metric ids and returns compact `GafimeResultTable` rows. During the spike this score ABI supports only Pearson and R2 for finite-feature decision paths; unsupported metrics must return unsupported, not fabricated zeros. CUDA may split a mixed-axis score batch into internal <=3D RT groups and direct modes must preserve exact feature-pair groups before widening compatible lower-dimensional work, but it must preserve original path order and must not move discovery, scheduling, or fallback policy out of Rust. CUDA may use an internal duplicate-safe device bitset or direct duplicate-safe traversal statistics, but it must not copy full path-major membership to host on the scoring path. Direct traversal statistics are opt-in through `GAFIME_CUDA_DECISION_PATH_RT_SCORE=direct`; target-wide statistics and centered per-path sums use double precision, but floating atomic order remains tolerance-checked at the approved `1e-4` spike threshold and must not become the default without maintainer approval. First-hit direct traversal statistics are opt-in through `GAFIME_CUDA_DECISION_PATH_RT_SCORE=firsthit` and are allowed only when CUDA proves every RT group is finite, bounded, 2D, and non-overlapping. The non-overlap proof plus `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT` must make each ray contribute at most once, so neither the single-group planned path nor the multi-group instanced path may allocate or clear the path-row duplicate bitset; ordinary direct mode must retain its duplicate guard. Both paths must consume one shared compile-time-testable duplicate-guard policy. Otherwise CUDA must return unsupported instead of falling back or changing semantics.

The public compact score route is limited to the complete, untruncated unary base-plus-path family with Pearson and/or R2, finite RT-representable inputs, and no graph or significance request. Rust executes base unary rows through the normal CUDA plan, appends compact path rows in global candidate order, and retains fallback ownership. Every other public shape uses the established membership-expansion plus continuous-scoring path. An explicit require-RT policy must fail closed rather than select that fallback.

CUDA RT program, custom-AABB geometry, and workspace state is owned per CUDA
device. Same-device execution is serialized around mutable OptiX state,
while different devices never share a context, stream, GAS, or workspace.
Every RT execution and teardown must restore the calling thread's previous CUDA
device. CUDA payloads that own this state expose the optional
`gafime_gpu_decision_path_release_device_state(device_id)` lifecycle symbol;
Rust shares one cleanup owner per loaded payload and device and calls it after
the final owning matrix is freed. Direct C ABI owners must do the same. Older
payloads may omit the symbol and remain loadable; the Rust host must serialize
decision-path calls per legacy payload and device because those payloads may
also predate native same-device locking. Current payloads keep their native
locking path without that compatibility mutex.

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

Metal uses the same `gafime_gpu_*` C ABI as CUDA and ROCm. The Metal shader implements continuous Pearson/R2, fixed-bin mutual information, and Spearman scoring; numerical parity against the reference is gated by Apple-hardware validation. Because Metal Shading Language has no fp64, Metal reductions accumulate in fp32; parity tolerances against CPU and CUDA/HIP must account for backend-specific precision and reduction order, then be measured and approved on Apple hardware. Metal mutual information clamps bins to <= 48 so the joint histogram fits threadgroup memory. Graph capture/replay and backend-native permutation replay remain unsupported on Metal; Rust-orchestrated target replacement plus exact Metal screening/ranking is the approved bounded maxT path. Unsupported Metal metrics, graph replay, missing Metal payloads, and unavailable Apple runtime support must return explicit errors through the boundary and must never silently route to CPU, Python, CUDA, or ROCm.

Metal host-side interaction centering must use the same f64 column-mean
accumulation and non-finite propagation semantics as CPU, CUDA, and ROCm. The
macOS gate must execute CPU-oracle parity for all four continuous metrics on
high-dynamic and NaN/Inf inputs plus multi-block ascending/descending top-k.
`GAFIME_METAL_PARITY_TOLERANCE=0.00005` is the approved absolute fp32 release
tolerance for that gate. Apple-hardware run `30207767348` observed a worst-case
absolute delta of `4.045665264e-6`; the approved bound is about `12.36x` that
measurement. Increasing the bound requires new Apple-hardware evidence and
explicit maintainer approval.

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

Precision requests separate matrix storage from compute policy. The only
current executable pair is `float32 + stable`: inputs, interaction products,
and result-table metrics remain fp32, while accumulator widths are reported per
metric and backend. `GAFIME_DTYPE_F64` and
`GAFIME_GPU_DEVICE_FLAG_F64_STORAGE` reserve an additive ABI contract, but no
current payload may advertise or accept f64 storage. A `float64`, `exact`, or
guard-disabling `fast` request must fail before fp32 coercion or backend
execution. CUDA and ROCm must advertise their separately compiled fp64 MI mode
through `GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64`; that bit must never be
interpreted as f64 storage, interaction arithmetic, or result output. The full
admission requirements are documented in `docs/precision-contract.md`.

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
observed MI score. GPU permutation passes must remain on their observed backend;
CPU bootstrap stability for a GPU observation uses fixed equal-width MI and
preserves the observed backend's template ceiling.

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

## Evidence And Claims Discipline

Verification method and claim wording are part of the contract, not reviewer preference.
`docs/evidence-discipline.md` carries the binding detail; this section is the summary.

Verification:

- A benchmark harness must verify bit-exact parity against the reference before reporting any timing, and must not report timings for a variant that failed parity.
- Parity inputs must be derived from the branch structure of the reference implementation, not from a distribution. Every conditional in the reference is a mandatory input class. For floating-point kernels that includes `NaN`, `+Inf`, `-Inf`, `-0.0`, subnormals, values at and adjacent to each clamp boundary, and chunk-tail lengths.
- Baselines must be copied verbatim with their source path and commit recorded, never paraphrased or reimplemented from documentation.
- A change that bundles several mechanisms must be measured with one variant per mechanism. A bundled result may not be attributed to a single mechanism.
- A measurement may not be described as an improvement until its production caller and call frequency are established and recorded.
- Reported figures must state host, thread count, ISA rung or backend exercised, input distribution, the statistic used, and a size sweep that crosses cache levels.

Claims:

- Agreement between two implementations that share a computation stage is not evidence of correctness; a defect in the shared stage appears as perfect agreement. Correctness requires an independent oracle of higher precision or different construction.
- Every claim must state what was not measured: which architectures, backends, ISA rungs, thread counts, and API levels fall outside the evidence.
- A change that degrades any path must state that degradation in the same artifact as the improvement, with equal specificity.
- A claim later found to be wrong must be corrected where it was made, recording what the earlier measurement actually showed and why it did not support the claim.
- Release notes must carry a Deliberate Non-Claims section and an Evidence Boundaries section. A provisional or unapproved numerical tolerance must be named as such in the release note of any release that ships the affected backend to users.

Prefer an interface where a precondition cannot be violated over an interface where the precondition is only documented.

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

## Clear-Recovery Handoff Snapshot

This section is an operational handoff for agents resuming after a `/clear` or
large context compaction. It is a snapshot, not permanent project law. The next
agent must verify it against `git`, `gh`, the local worktree, and release gates
before acting on it.

Snapshot date: 2026-07-06.

Expected workspace:

```bash
cd /home/hamza-usta/GAFIME
git status -sb
git log --oneline --decorate -12
gh pr view 16 --json url,isDraft,mergeStateStatus,statusCheckRollup,headRefName,baseRefName
```

Last verified branch state:

```text
branch: codex/cuda-hip-kernel-hardening
remote: origin/codex/cuda-hip-kernel-hardening
base: main
worktree: clean
PR: https://github.com/onlyxItachi/GAFIME/pull/16
PR state: draft, mergeStateStatus=CLEAN
GitHub checks: V1 Contract Validation jobs succeeded
```

Last verified PR checks:

```text
Contract and top-level NumPy parity: SUCCESS
Metal shader, payload, and v1 API validation: SUCCESS
```

Latest branch commits at the time of this snapshot:

```text
b7c6791 docs(gpu): record cuda unary stats benchmark
86f214b docs: commit RT GBDT paper PDF
0ff97dd perf(cuda): use unary stats outside graph replay
d488512 perf(gpu): skip unused covariance launches
854a4a6 perf(gpu): cache unary feature stats
4ab97f4 perf(gpu): cache continuous target stats
81d0851 main docs: rewrite RT GBDT paper as arxiv draft
```

Files touched by the CUDA/HIP hardening PR:

```text
src/cuda/launcher.cu
src/cuda/kernels.cu
src/cuda/kernels.cuh
src/rocm/launcher.hip
src/rocm/kernels.hip
src/rocm/kernels.hpp
crates/gafime-gpu-sys/src/lib.rs
tests/release_measure/v1_architecture_gate.py
docs/gpu-continuous-target-stats-cache.md
docs/rt-gbdt-hardware-ray-tracing-paper.tex
docs/rt-gbdt-hardware-ray-tracing-paper.pdf
```

Completed work in the PR:

- CUDA and ROCm cache compact continuous target statistics in backend-owned
  device memory.
- CUDA and ROCm refresh target statistics after target upload/update.
- CUDA and ROCm cache unary feature statistics after matrix upload.
- All-finite arity-1 Pearson/R2 chunks may use a one-pass covariance scoring
  kernel.
- Generic continuous kernels still own arity greater than 1, non-finite
  filtering, pairwise finite semantics, MI companion behavior, and Spearman
  companion behavior.
- CUDA and ROCm skip the continuous Pearson/R2 covariance sweep when a chunk
  requests only metric-specific kernels such as MI-only or Spearman-only.
- CUDA unary stats are enabled for non-permutation covariance launches, not
  only graph replay. Permutation launches remain on the generic path because
  the target changes inside the backend permutation loop.
- The RT GBDT technical disclosure PDF is committed under `docs/`.
- The RT GBDT paper source uses `\text{...}` rather than `\hbox{...}` in math
  fragments so Pandoc/HTML/PDF generation renders cleanly.

Committed PDF artifact:

```text
docs/rt-gbdt-hardware-ray-tracing-paper.pdf
```

PDF generation evidence from the final pass:

```text
generator path: pandoc LaTeX -> temporary HTML -> headless Chrome PDF
pdf property: extractable text
pdf property: no browser header/footer
pdf property: numeric citation markers and ordered reference list
local TeX engines available at the time: none found in PATH
```

Local validation commands that passed with staged payloads:

```bash
export PYTHONPATH=/home/hamza-usta/GAFIME/python
export GAFIME_CUDA_V1_LIB=/tmp/libgafime_cuda_v1.so
export GAFIME_ROCM_V1_LIB=/tmp/libgafime_rocm_v1.so

cargo test -p gafime-gpu-sys cuda_device_topk_returns_only_selected_rows_when_library_is_available -- --nocapture
cargo test -p gafime-gpu-sys cuda_permutation_protocol_preserves_observed_metrics_when_library_is_available -- --nocapture
python3 tests/release_measure/backend_02_cross_backend_parity.py
python3 tests/release_measure/graph_01_replay_parity.py
python3 tests/release_measure/v1_architecture_gate.py --include-gpu
python3 tests/release_measure/contract_00_policy_files.py
```

Representative local correctness output from the final pass:

```text
backend_02_cross_backend_parity.py:
  CUDA vs core max abs delta <= 1.19e-07, PASS at tol 0.001
  ROCm vs core max abs delta <= 8.94e-08, PASS at tol 0.001

graph_01_replay_parity.py:
  CUDA graph-vs-plain max metric delta = 0.00e+00, PASS at tol 0.0001

v1_architecture_gate.py --include-gpu:
  CPU Rust tests passed
  GPU sys CUDA/ROCm tests passed
  orchestrator tests passed
  Python boundary tests passed
  type/ABI layout tests passed
  v1 architecture gate passed
```

Benchmark evidence recorded in `docs/gpu-continuous-target-stats-cache.md`:

```text
CUDA unary stats plain median: 31.7 -> 42.0 GEval/s
CUDA unary stats graph median: 39.7 -> 41.4 GEval/s
ROCm unary stats plain: 6.66 -> 12.68 GEval/s
ROCm unary stats graph: 5.33 -> 10.97 GEval/s

CUDA MI-only: 0.171 -> 0.230 GEval/s
ROCm MI-only: 0.088 -> 0.089 GEval/s
CUDA Spearman-only: 0.049 -> 0.049 GEval/s
ROCm Spearman-only: 0.095 -> 0.102 GEval/s
```

Plateau decision for the CUDA/HIP hardening pass:

- Stop squeezing this PR with risky kernel rewrites.
- The remaining performance wins are plausible, but they are not safe
  same-turn hardening patches.
- Further work needs dedicated PRs with new parity, numerical-policy,
  performance, and release-measure validation.

Known next PR candidates:

```text
1. Spearman GPU rank acceleration
   Current rank-style Spearman is correctness-oriented and can be expensive.
   Any speedup needs a real rank/reduction design and parity gates.

2. CUDA top-k/reduction scaling
   Current top-k behavior works, but serious scale needs a dedicated device
   selection/reduction design.

3. MI histogram backend tuning
   Needs careful bin/shared-memory/occupancy work and documented parity.

4. RT GBDT real-ensemble extraction/validation
   The RT spike is documented and impressive, but the next step is real
   trained-ensemble path extraction, deterministic reductions, and broader
   device validation.

5. dtype/operator CandidateSpec IR for v1.1
   Add typed candidate-expression support as an IR, not a tensor API. Keep
   generated candidates compact and backend-streamed.
```

Do not treat the following as safe quick changes without a dedicated design:

- rewriting Spearman rank kernels for speed
- changing MI bin policy or histogram accumulation
- caching arity greater than 1 interaction statistics
- changing CUDA top-k selection semantics
- promoting RT first-hit mode to default behavior
- widening RT scoring beyond Pearson/R2 without parity work
- introducing fast-math, approximate math, or undocumented tolerance changes

If the user asks whether PR #16 can merge, verify with:

```bash
git status -sb
gh pr view 16 --json mergeStateStatus,statusCheckRollup,isDraft
```

Only merge when the user explicitly asks and the PR is clean, required checks
are successful, and no local changes exist.

If the user asks what GAFIME v1 is currently capable of, answer in this frame:

- GAFIME v1 is high-dimensional candidate discovery and ranking over feature
  interactions, not matrix multiplication.
- Matrix-shaped input is storage; execution is candidate-row work over compact
  candidate descriptors.
- Rust owns Python boundary, validation, planning, scheduling, backend
  selection, CPU SIMD, compact result ownership, and `gafime.compile`.
- CUDA/ROCm/Metal payloads own backend execution only through the stable C ABI.
- Normal runtime must stream candidate specs/results and keep bounded top-k or
  compact report state; it must not materialize expanded candidate vectors as
  the default path.
- Performance claims should be framed as candidate throughput, memory
  footprint, DRAM/cache behavior, bounded reporting memory, graph replay, and
  device occupancy, not arbitrary toy matrix probes.

Post-clear rule:

Do not trust old chat context. Re-read the repo. The source of truth is current
git state, PR state, this contract, `docs/contract.md`, release-measure gates,
and the actual files under `src/`, `crates/`, `python/gafime`, `docs/`, and
`tests/`.

## Performance Hardening Continuation (2026-07-10)

This section supersedes the earlier plateau/next-PR guidance for the current
working tree. The user explicitly reopened CUDA/HIP/Metal kernel hardening on
`codex/cuda-hip-kernel-hardening`; do not move this work to another branch.

Current checkpoint:

```text
branch: codex/cuda-hip-kernel-hardening
tracking: origin/codex/cuda-hip-kernel-hardening
base commit before this continuation: e09490c
PR: https://github.com/onlyxItachi/GAFIME/pull/16
state: implementation and validation are tracked on this branch; inspect the
       current git and PR state for the published checkpoint
validation: CUDA/ROCm correctness and macOS Metal gates complete; ROCm wave
            decision and compute-idle CUDA timing complete; CDNA runtime pending
```

The continuation implements:

- CUDA and HIP compile-time continuous/Spearman arity specializations for
  `1..5`.
- CUDA and HIP MI specializations for arity `1..5` crossed with adaptive
  capacities `2,4,8,12,16,24,32,48,64,96`.
- shared adaptive-`mi_bins` planning, including significance-path consistency
  and a public quantization-quality contract for the intermediate capacities.
- CUDA fatbinary coverage remains controlled by `GAFIME_CUDA_ARCHITECTURES`;
  launch geometry is selected at runtime from the actual device class and block
  limits, with no package-wide tuning SM.
- HIP wave-reduction A/B control through exact
  `GAFIME_HIP_WAVE_MI_MODE=off|64|96|64-96` specializations with embedded
  artifact provenance; mode `64` is the production default after repeated
  `gfx1150` measurements accepted 64 bins and rejected the 96-bin path.
- two-stage device top-k plus selected-row gather for CUDA, HIP, and Metal,
  bounded partial storage, both rank directions, deterministic tie order, and
  previous-cutoff progression that avoids a quadratic factor in `top_k`.
- Metal 64-lane continuous/MI reductions, inline arity cases, reference-aligned
  host column means, ARC-managed host resource ownership, and macOS
  CPU-oracle/top-k behavioral gates.
- capability-based HIP integrated-device detection.
- CUDA RT build separation through
  `GAFIME_CUDA_RT_BUILD_MODE=off|on|both`.
- repeatable static CUDA SASS/HIP code-object inspection and candidate-scale MI
  performance/A-B harnesses.

The full local CUDA/ROCm architecture gate passes with all 38 GPU-system tests
executing against real devices. CPU/public contracts, CUDA no-RT and RT ABI
smokes, static template/resource checks, cross-backend parity, and graph replay
also pass. The idle `gfx1150` mode-64 A/B produced exact MI parity, raw 64-bin
speedups of `1.1186x`, `1.1237x`, and `1.1040x`, and control-normalized speedups
of `1.1637x`, `1.1372x`, and `1.1835x`. Mode `96` and combined mode `64-96`
regressed their final 96-bin controls, so production enables wave reduction only
for the 64-bin specialization.

No timing captured while another GPU workload was active is release evidence.
The final CUDA run followed five `0%` SM samples and measured `42.471`, `47.028`,
and `34.565` candidate-sample GEval/s at bins `32/64/96`; persistent display
memory traffic means those are local shape rates rather than a display-free
benchmark. GitHub Actions run `29112217686` compiled the Metal shader/payload
and executed both direct Metal gates on Apple hardware under the former
provisional bound. Follow-up run `30207767348` emitted per-metric parity
evidence: the worst absolute delta was Pearson at `4.045665264e-6`, followed by
R2 at `2.622604370e-6`. The approved absolute tolerance is `0.00005`; this is
correctness evidence, not a Metal performance claim. Metal performance
evidence and real wave64/CDNA execution remain pending.

The distribution-level A/B against the exact `gafime`, `gafime-cuda`, and
`gafime-rocm` v0.4.7 packages is recorded in
`docs/v0.4.7-current-distribution-benchmark.md` and reproduced by
`tests/release_measure/perf_08_v047_distribution_ab.py`. Across the nine
ratio-bearing Pearson/R2 workloads, including the ROCm house_16H arity-2
fallback, current default cached analysis is `1.319x..2.100x` faster and
compiled replay is `2.335x..7.355x` faster with exact candidate identity and
`20/20` top-20 overlap. Explicit cache-disabled analysis remains the main
end-to-end regression, especially on CUDA; default adaptive CPU MI is
approximately flat, and Spearman remains an algorithmic scaling target. The
run covers up to 76.2 million candidate-row pairs but does not replace a
100-million-candidate compact-output scale gate.

For follow-up commits or pushes, rerun the relevant validation defined in
`docs/cuda-template-kernel-hardening.md`, retain exact artifact provenance, and
keep PR #16's hardening title/body synchronized with the branch.

## Correctness Boundary Hardening Follow-up (2026-07-11)

This section supersedes the performance-hardening continuation for the current
working branch. PR #16 is merged; this follow-up is correctness and release
hardening over that merge.

Current checkpoint:

```text
branch: codex/correctness-boundary-hardening
tracking: origin/codex/correctness-boundary-hardening
base: 47f9c6e (merged PR #16)
PR: https://github.com/onlyxItachi/GAFIME/pull/17
state: draft; local validation and GitHub Actions run 29168118683 passed
commits:
  375472f fix(core): harden plans and significance contracts
  a2d02d3 fix(gpu): harden payload lifecycle and ABI boundaries
  d0fd4ed test(release): enforce installed boundary contracts
  3b1682a docs(agent): record correctness hardening handoff
```

The follow-up establishes these invariants:

- Rust matrix handles are borrowed, compiled plans validate their complete
  descriptor/result contract, and compact output ownership remains bounded.
- permutation maxT evaluates the complete screened family; GPU-observed MI CPU
  fallback uses the same estimator and backend bin ceiling.
- CUDA/ROCm/Metal reject pre-upload execution and publish replacement content,
  graph state, and caches transactionally.
- null native inputs, stale ABI versions, wrong-backend protocols, malformed
  byte counts, and RT count/allocation overflows have explicit status behavior.
- the CUDA RT path count is bounded by the shared four-vertices-per-path u32
  geometry limit, `GAFIME_MAX_DECISION_PATH_COUNT`.
- Windows CUDA/ROCm staged and CMake payloads export the shared ABI correctly.
- Arrow/config fallback, graph replay, compile-plan visibility, generated-family
  identity, and significance shapes report actual runtime state.
- wheel and contract CI execute installed-package native checks, known numerical
  oracles, and eager/compiled interaction, permutation, stability, and decision
  parity. Optional backend skips require the exact missing-payload error.

Verified locally on 2026-07-11:

- 146 Rust workspace tests passed with the real RT-off SM89 CUDA payload on the
  RTX 4060 Laptop GPU.
- 74 Python tests passed with 2 hardware-dependent skips.
- the final `cp310-abi3` wheel passed from an external Python 3.14 environment,
  including 23 installed-package truthfulness tests.
- contracts 00-03, compile plan/value/significance/decision parity, backend
  availability/E2E, and the v1 architecture gate passed.
- CUDA SM89 and ROCm gfx1150 release payloads compiled; the focused CUDA C-ABI
  malformed-input test executed on hardware; Metal fallback syntax passed.
- scoped Ruff, YAML, shell, diff, and strict `gafime-gpu-sys` Clippy checks
  passed.
- GitHub Actions run `29168118683` passed the Linux installed-contract lane and
  the macOS Metal shader, payload, top-level, top-k, and numerical parity lane.

No performance benchmark or profiler capture was run in this turn at the
maintainer's request. PerfDigest had no report to compact. OptiX-enabled CUDA,
and ROCm runtime remain external hardware/toolchain gates. See
`docs/correctness-boundary-hardening.md` for the full invariant and gate list.

For continuation, first run:

```bash
git status -sb
git log --oneline --decorate -6
gh pr view 17 --json url,isDraft,mergeStateStatus,statusCheckRollup,headRefName,baseRefName
```

PR #17 was merged at `a3a0d65`; the following section supersedes that
checkpoint.

## Eager Path Pre-Release Hardening (2026-07-19)

This section supersedes the correctness-boundary continuation for the current
working branch. PR #17 is merged. This branch is pre-release hardening only; it
must not be merged or released without explicit maintainer approval.

Current checkpoint:

```text
branch: codex/eager-path-release-hardening
base: a3a0d65 (merged PR #17)
PR: #18 (draft) https://github.com/onlyxItachi/GAFIME/pull/18
state: implementation, bounded verification, and cross-platform CI complete
commits:
  5f58184 test: repair standalone GPU ABI protocols
  7255ae3 perf: separate one-shot and resident Python paths
  84758cd fix: restore legacy screened candidate planning
  674cdf2 perf: cache immutable compiled launch descriptors
  4366ccb fix: negotiate immutable GPU protocol capability
  9777b71 fix: restore Python adapter compatibility
  3ff6dec fix: restore legacy runtime execution contracts
  9688358 test: gate legacy distribution compatibility
  a815041 chore: clean orchestrator Clippy diagnostics
  1cab626 docs: define distinct continuous execution paths
  54d0d2f fix: preserve the legacy Python execution surface
  4398302 test: enforce full legacy report identity
  25e36d1 build: declare and gate the proven Rust MSRV
  e9b53b4 docs: clarify disabled-cache residency
  d678314 docs: record the pre-release hardening handoff
  cdeb12e test: separate pair math from tuple orientation
  982a894 fix: preserve concurrent resident reuse
  17b1409 test: scope legacy stochastic comparisons
  1535d74 ci: validate CUDA 13.3 and legacy Metal
  a5714b1 ci: install CUDA 13.3 from NVIDIA
  b8030e0 ci: install CUDA compiler runtime headers
  2043888 ci: install CUDA NVVM compiler component
```

The branch establishes these additional invariants:

- cache-disabled one-shot, resident eager LRU, and explicit compiled execution
  are distinct paths; only resident lookup computes content digests;
- setting `GAFIME_V1_ANALYZE_CACHE_SIZE=0` closes existing LRU artifacts before
  the next analysis;
- current buffer-capable boundaries ingest contiguous little-endian fp32 bytes,
  while older/custom boundaries retain nested/list compatibility;
- representable NaN and infinity inputs are preserved, finite fp32 overflow is
  rejected, and the eighth positional `EngineConfig` argument remains
  `mi_bins`; new significance/MI controls are keyword-only;
- full Python integer seed words participate in planning, `random_seed=None`
  reseeds every analysis without defeating resident cache identity, and exact
  legacy warning text is preserved;
- the global resident LRU lock protects only cache bookkeeping; independent
  resident artifacts execute concurrently under per-entry locks, and eviction
  closes an artifact only after any in-flight analysis finishes;
- maxT uses exact exceedance without a hidden epsilon, bootstrap work is skipped
  for one repeat, and sampled feature columns are reused within a bounded
  bitwise-equivalent cache;
- immutable descriptor reuse requires the distinct
  `GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION` capability; current hosts
  strip the launch hint and zero the generation for older same-ABI payloads,
  including those that advertise only the legacy immutable bit;
- the legacy A/B harness includes report order, tuple/family identity,
  candidate-id stability, warnings, decision signal, and optional
  stability/permutation snapshots. Deterministic identity and metrics remain
  strict, while stochastic values are not compared across legacy candidate-wise
  tests and current family-wise maxT; current one-shot/resident/compiled
  stochastic parity remains strict;
- Rust 1.89 is the declared and CI-gated minimum. Rust 1.76 fails the locked
  dependency set, while 1.89 compiles the workspace and supports the AVX-512
  intrinsics. CUDA remains C++20 because CUDA 13.3 officially supports CUDA C++
  through C++20, not C++23. The distribution workflow now builds CUDA 13.3 on
  `windows-2025-vs2026` with Visual Studio 18 / MSVC 14.51.36231 and installs
  the required NVIDIA `nvcc`, `crt`, `cudart`, and `nvvm` components.

Verified locally on 2026-07-19:

- 159 Rust workspace unit tests and the compile-fail doctest passed.
- 149 Python tests passed with 7 hardware-dependent skips.
- the architecture gate passed with current SM89 CUDA RT/non-RT and gfx1150
  ROCm payloads; all 47 GPU-system tests executed in that configured run.
- one-shot, resident first/repeat/update, and compiled first/repeat/update were
  exact on Core, CUDA, and ROCm before the final Python-only review corrections;
  the final Core rerun remained exact for all six comparisons.
- current host execution against pre-immutable same-ABI CUDA and ROCm payloads
  matched CPU with zero observed delta.
- `cargo +1.89.0 check --workspace`, changed-file Ruff, YAML parsing, policy
  checks, diff checks, and a clean Python 3.14 `cp310-abi3` wheel smoke passed.
  Normal workspace Clippy passes; `-D warnings` still exposes pre-existing
  warnings in untouched CPU/test modules and is not claimed as a branch gate.
- bounded ordered comparisons against both v0.4.7 and `v0.5.0-legacy` passed
  exact candidate identity, at most `5e-6` metric drift, and at least `1.0x`
  one-shot/compiled speed gates on Core, CUDA, and ROCm. The recorded numbers
  predate final compatibility-only corrections and are not publication
  throughput claims.
- an independent gpt-5.6-sol max review found four compatibility/gating defects;
  all four were fixed. Its focused review of `982a894^..2043888` found no
  remaining actionable defect.
- V1 Contract Validation run 29697333663 passed on commit `2043888`, including
  Rust 1.89, Linux contract/NumPy parity, and macOS Metal validation.
- Native Platform Validation run 29696953822 passed on commit `1535d74` for ARM
  Linux, ARM Windows, current Metal, and current-host execution of the
  pre-capability same-ABI Metal payload (`10` candidates, max delta `1.19e-7`).
- non-publishing Build and Publish Wheels run 29697338525 passed on commit
  `2043888`: core Linux x86/ARM, Windows x86/ARM, and macOS ARM wheels; CUDA
  13.3 Linux/Windows payloads and clean-install ABI probes; ROCm Linux payload
  and clean-install ABI probe; all source distributions; and clean-installed
  core wheel probes on every built platform. Publishing jobs were disabled.

No new profiler capture was produced, so PerfDigest had no report to compact.
Its capability handshake is healthy for CPU, CUDA, ROCm, and Metal report
digestion; use it when a real profiler report is available. No known
correctness or compatibility gate remains open. PR #18 must remain draft until
the maintainer explicitly requests merge. Do not merge or release from this
handoff.

## PR #18 Final Pre-Merge Review (2026-07-19)

This section supersedes the earlier PR #18 handoff. The maintainer has now
requested a complete review, RT-paper evidence refresh, and merge only after
local validation, hosted CI, and independent review are all clean. It does not
authorize a release or tag.

Current checkpoint:

```text
branch: codex/eager-path-release-hardening
tracking: origin/codex/eager-path-release-hardening
base: a3a0d65 (merged PR #17)
PR: #18 (draft) https://github.com/onlyxItachi/GAFIME/pull/18
local state: final implementation and pre-push validation complete
remaining: commit, push, hosted cross-platform CI, final independent votes
```

The final review adds or hardens these invariants:

- adaptive-family maxT rebuilds its screened shortlist for every permutation;
  its replay uses the same effective feature-candidate cap as observed planning;
  GPU-observed MI host fallback uses the same fixed-width estimator and backend
  template ceiling as the observed score;
- Python resident entries are thread-affine, BaseException-safe, and enforce a
  reduced positive cache capacity even on hits; fail-closed cleanup cannot mask
  the primary native exception;
- legacy public exports and the unambiguous historical positional config
  prefixes are restored; ambiguous trailing `ComputeBudget` positionals fail
  with a migration error, and the removed discrete family remains unsupported;
- generated-family artifacts preserve expanded names for report decoding while
  scenario metadata retains the original input feature count;
- immutable GPU descriptor reuse is keyed by a Rust-owned nonzero content
  generation, remains upload-every-call for generation zero, and publishes
  CUDA/ROCm/Metal replacements only after successful uploads;
- CUDA RT state is isolated per device, serialized per device, released by the
  final matrix owner, and rebuilt after cleanup. Required-RT policy is explicit
  in Rust and fails closed when the payload is RT-off. Every public RT call
  restores the caller's CUDA device; legacy payloads without native cleanup
  share a host compatibility lock. SM row grids are tiled by the runtime
  grid-y limit, and instanced grouped launches fall back before exceeding it;
- standard CUDA artifacts are immutably RT-off as distribution `gafime-cuda`,
  package `gafime_cuda`. Optional RT-on artifacts use the distinct non-PyPI
  identity `gafime-cuda-rt` / `gafime_cuda_rt` and native library name, and are
  excluded from the standard 13-artifact release bundle. A dual installation
  is rejected unless `GAFIME_CUDA_V1_LIB` explicitly selects a variant. The
  frozen preflight bundle is the only input to publish jobs. Optional RT
  provenance separately binds the digest-pinned wheel-builder and lifecycle
  images plus all 11 hash-pinned CUDA RPM inputs. Core wheel build tags are
  validated before mutation and again by release composition.
- automatic tag publication is restricted to `push` events; a dispatch on a
  tag cannot bypass its per-distribution opt-in. Hosted release policy must
  keep `main` review-protected, `v*` creation owner-only and immutable for
  non-owners, and the `pypi` environment restricted to `v*` tags with an owner
  deployment review. Re-verify those live settings before every release.

Final local evidence:

- `cargo +1.89.0 test --workspace --quiet`: 189 unit tests plus one compile-fail
  doctest passed.
- Python source: 215 passed and 6 explicit Metal-hardware or ROCm-E2E-deferred
  skips in an isolated dependency-complete Python 3.14 environment, with
  unraisable warnings promoted to errors.
- the GPU-inclusive architecture gate executed all 56 GPU-system tests against
  fresh SM89 CUDA RT-off/RT-on and gfx1150 ROCm payloads and passed.
- all 56 GPU-system tests also passed against the exact older CUDA RT and ROCm
  payloads used for compatibility validation, including parallel test execution;
- fresh CUDA off/both and ROCm gfx1150 CMake builds passed. Standalone CUDA and
  ROCm ABI smokes, RT cleanup/rebuild, the runtime grid boundary probe, and
  same-device OptiX concurrency at 8 threads by 40 iterations passed.
- fresh core, standard `gafime-cuda`, optional `gafime-cuda-rt`, and gfx1150
  ROCm Linux wheels built, and all four source distributions passed
  archive-level composition. Clean Python 3.14 installs outside the checkout
  passed payload discovery, license/build-policy, ABI-export, and
  `gafime --check` probes;
  CUDA reported RT unavailable for the standard wheel and available for the
  optional RT wheel, while an unselected dual install failed with an explicit
  ambiguity error. The local core wheel carries the host's `manylinux_2_34`
  tag; exact `manylinux_2_28`, macOS, and Windows wheel-platform composition
  remains a hosted-CI gate.
- the RT provenance staging contract and all four source-distribution
  compositions passed locally; the pinned manylinux image was accepted by
  cibuildwheel, all 11 CUDA RPM hashes were verified, and their exact local-RPM
  installation produced NVCC 13.3.73 inside the recorded builder image.
- the final first-hit case at 262,144 by 8,192 measured 56.109 ms first call,
  0.886 ms resident warm p50, 2.424 T membership-equivalent evaluations/s,
  2.367 G rays/s, and maximum absolute error 1.19209e-7. The 1,048,576-path
  case measured 467.746 ms first call, 20.180 ms warm p50, 13.621 T
  membership-equivalent evaluations/s, 0.104 G rays/s, and 5.58794e-9 error
  against the structure-aware partition oracle.
- a fresh Nsight Compute report was digested through PerfDigest. The OptiX
  launch was 196.992 us with 24.932% compute-pipe peak, 10.878% DRAM peak,
  54.223% achieved occupancy, 53.408% L1 hit, and 96.864% L2 hit. It exposes no
  direct RT-core saturation counter, so no saturation percentage is claimed.
- the checked-in 12-page paper PDF was regenerated with Tectonic/xdvipdfmx,
  passed qpdf, metadata, exact text-path extraction, and visual page checks,
  and has SHA-256
  `0c352ea2b9ec246a3be798c999729de39e5923ef626ad7ee17f571adde884c91`.
- changed-file Ruff, Actionlint, workflow YAML parsing, architecture/source
  policy, formatting, and `git diff --check` passed.
- GitHub main protection, the active `Protect release tags` ruleset, and the
  owner-reviewed `pypi` environment `v*` deployment policy were queried from
  the live repository after configuration.

Do not merge until the final commit is pushed, publication-disabled hosted
workflows pass for Linux ARM/x86, Windows ARM/x86, macOS Metal, CUDA, ROCm, and
all artifacts, and fresh read-only reviewers vote merge-ready. Do not create a
release or tag in this turn.

## PR #18 Memory-Preflight Closure (2026-07-20)

This section supersedes the stale checkpoint and validation counts above. Keep
PR #18 release-free: no tag, PyPI publication, or GitHub Release is authorized.

Current checkpoint:

```text
branch: codex/eager-path-release-hardening
tracking: origin/codex/eager-path-release-hardening
base: a3a0d65 (merged PR #17)
validated implementation head: f8a21c4
PR: #18 (draft) https://github.com/onlyxItachi/GAFIME/pull/18
local/remote state: aligned and clean
remaining: final independent review, restarted CI, merge
```

The final memory-preflight work adds these invariants:

- the optional `gafime_gpu_execution_memory_peak` and
  `gafime_gpu_permutation_memory_peak` ABIs remain load-compatible with older
  payloads and return stable, non-mutating, saturating forecasts;
- CUDA and ROCm forecasts include fixed matrix/stat allocations, every retained
  capacity, exact old-plus-new grow transitions, simultaneous descriptor-pair
  replacement, and a conservative top-k selected-count ceiling;
- the CUDA permutation forecast includes complete-protocol score and descriptor
  allocations plus old-and-new retained observed-value, metric-maximum, and
  exceedance-count buffers in native allocation order;
- Metal forecasts include fixed `MTLBuffer` owners, descriptor-cache growth,
  old-plus-new replacement, distinct cacheable and non-cacheable lifetimes, and
  ephemeral metric, rank, and top-k storage;
- driver graph bookkeeping, pipeline objects, and other opaque driver-owned
  allocations are explicitly outside the claimed bound;
- the orchestrator invokes supported native forecasts before execution. Under
  an active VRAM budget, an older same-ABI CUDA payload without the permutation
  query uses the existing budgeted host maxT fallback instead of bypassing
  admission.

Fresh local evidence:

- `cargo test --workspace` passed 233 unit tests plus the compile-fail doctest;
- the Python source suite passed 243 tests with 15 explicit hardware/deferred
  skips; the public CUDA adaptive-maxT path also passed against the current
  payload under an active VRAM budget;
- fresh CUDA RT-off/RT-on and gfx1150 ROCm builds passed ABI smoke; the CUDA RT
  lifecycle, policy, cleanup/rebuild, and same-device concurrency CTests passed
  5/5;
- the GPU-inclusive architecture gate passed with the fresh CUDA standard/RT
  and ROCm payloads. Core/CUDA/ROCm metric parity passed with fixed/adaptive MI
  worst delta `9.09e-8`; CUDA and ROCm graph replay each returned 78 exact rows;
- a saved schema-v2 hash (`ada58908...`) initially produced a false candidate-
  order alarm because v2 lexicographically sorted rows. Fresh v0.4.7 and
  `v0.5.0-legacy` runs under the schema-v3 report-order contract both produce
  `acaf0196ddbd4f8a00d3d5f6941bdfadefe8da999fdc577550ee8e4b9627e586`,
  exactly matching current. The matched Python 3.12 Core comparison observed
  maximum drift `2.18e-6`, eager speedup `1.78x`, and compiled speedup `4.59x`;
- the current checked-in paper is 14 pages, passes `qpdf`, identifies
  `LaTeX with hyperref` and `xdvipdfmx`, and has SHA-256
  `1da967a8775dd5fb032d2e011c934ddc7779028b429d259d6dc73b17bff5efc4`.

Hosted runs `29741393678`, `29741393695`, and `29741393782` passed at head
`37c3572`: ARM Linux/Windows, x86 Linux/Windows with Visual Studio 2026, macOS
Metal, CUDA 13.3, ROCm 7.2.3, source distributions, clean wheel installs, and
release preflight. Publishing and the separately requested OptiX artifact lane
were skipped as intended. A final reviewer then found that the CUDA native
permutation-pvalue shortcut could allocate retained significance buffers after
the normal execution preflight. Commit `f8a21c4` adds the missing state-aware
admission query and focused boundary tests. Its restarted hosted workflows are
the remaining platform gate, and the same non-overlapping reviewer is checking
the correction; do not launch duplicate reviewers while it runs.

## PR #21 Pre-Release Publication-Hardening Handoff (2026-07-21)

This section supersedes stale "current branch" and in-progress PR #18
instructions above. Those sections remain historical evidence and must not be
read as the current branch or authorization state.

Current checkpoint:

```text
branch: codex/prerelease-release-hardening
base: 1ba7efd (PR #20 merged)
PR: #21 (pre-release publication-hardening pass)
repository state: branch started clean from PR #20 merge commit 1ba7efd; work is in progress
authorization: no release, tag, or publication
```

Publication ordering, collision handling, and the corresponding tests are
owned by the main agent. This ownership includes integration, final review,
and any release-gate decision for PR #21. No release or tag may be created
from this pass.

The release workflow publishes CUDA and ROCm payloads before Core, then creates
the GitHub Release only after all three PyPI lanes succeed. Normal publication
rejects existing filenames. Manual recovery may skip only SHA-256-identical
files and may create the GitHub Release only from the version tag with all three
PyPI lanes explicitly selected.

Live repository-setting result for the current PR #21 branch-rule layer:

- Active ruleset name: `Require main release validation`
- Ruleset ID: `19444819`
- Target: `refs/heads/main`
- `current_user_can_bypass`: `never`
- Status policy: `strict`
- Required checks:
  - full-artifact preflight
  - 3 native-platform
  - 5 v1 contract

Existing review protection remains separate and unchanged.

## ROCm Wheel Policy Handoff (2026-07-22)

ROCm payload staging must select `bundled` explicitly. The checked-in
`.github/scripts/rocm_7_2_3_bundled_policy.json` is the exact source, sdist,
wheel, artifact-gate, installed-smoke, and public-diagnostics contract.
`system`, `amd-wheels`, implicit selection, unowned private libraries, missing
SBOM coverage, absolute runtime paths, and mixed-runtime coexistence claims
must fail closed. Published artifacts are immutable; this policy applies only
to wheels built from commits that contain it.

## v1.0.0b1 Release-Artifact Repair Handoff (2026-07-26)

This section supersedes the older current-branch and ROCm-policy handoffs above.
Those sections remain historical evidence for `v1.0.0b0`.

```text
branch: codex/v1.0.0b1-release-artifacts
base: 23067ae (v1.0.0b0 merge and tag)
PR: #50
scope: packaging-only b1 hotfix; no kernel or metric changes
authorization: merge, tag, and publish only after the complete reviewed gates pass
```

The standard GitHub Release has 13 artifacts: six Core, three CUDA, two ROCm,
and two Metal. Core is vendor-payload-free. `gafime-metal` owns the Apple
Silicon dylib/metallib pair. One `cp310-abi3` wheel per platform must be tested
on CPython 3.10 through 3.14.

The standard `gafime-rocm` identity uses the immutable `system` policy from
`.github/scripts/rocm_7_2_3_system_policy.json`. Its wheel contains no ROCm
userspace or runtime search path and requires system `libamdhip64.so.7`.
Because PyPI rejects the truthful `linux_x86_64` wheel and the external
dependency prevents a valid manylinux claim, attach that wheel to the GitHub
Release and publish only the matching ROCm sdist to PyPI. Bundled mode uses
the separate `gafime-rocm-bundled` identity and is not a standard b1 artifact.

CUDA, the ROCm sdist, and Metal publish before Core. GitHub Release creation
waits for all four PyPI lanes. Do not work unrelated newly opened issues during
this repair.
