# GAFIME Agent Contract

This file mirrors `CLAUDE.md`. Keep both files synchronized except for agent-specific notes that are explicitly needed.
The human-readable maintainer contract is `docs/contract.md`; this file is the agent-facing operational mirror.

## Delegated Agent Coordination

This section is Codex-specific and is intentionally not mirrored into
`CLAUDE.md`. Delegation is optional: do not create an agent or swarm when the
main agent can complete the work directly without losing an independent domain
owner or a meaningful parallel critical-path benefit.

- Every delegated agent must own one unique, non-overlapping domain with a
  concrete deliverable, evidence requirement, write scope, and stopping
  condition. Do not assign multiple agents to repeat the same review, tests, or
  repository-wide inspection.
- Reuse the same domain owner as that domain expands. For example, a CUDA owner
  may begin with compiled CUDA workflows and graph capture, then continue into
  eager CUDA execution and CUDA profiling. Do not replace it with additional
  CUDA reviewers that duplicate its accumulated responsibility.
- Agents must stop after completing their assigned work. They must not invent
  follow-up work, rerun already-proven gates without a changed dependency, or
  spawn descendant agents unless the parent has identified a genuinely distinct
  uncovered task that cannot be handled efficiently in its own scope.
- Use `gpt-5.3-codex-spark` for bounded, mechanically checkable work whose lower
  reasoning capability is acceptable, such as targeted searches, log triage,
  formatting, manifest comparisons, and narrow test-output inspection.
- When a multi-agent engineering pass is justified, start domain owners with
  `gpt-5.6-terra` at `ultra`. Keep using Terra when it completes the domain
  reliably. Escalate a specific domain to `gpt-5.6-sol` at `ultra` only after
  its complexity or failed evidence shows Terra is insufficient; do not occupy
  Sol agents speculatively.
- The main agent owns integration, resolves cross-domain decisions, and prevents
  scope overlap. A delegated agent's completion is evidence, not automatic
  authorization to merge or release.

If an equivalent policy is later written into `CLAUDE.md`, translate the model
roles in order as follows: `gpt-5.6-sol` `ultra` -> Fable 5,
`gpt-5.6-terra` `ultra` -> Opus 4.8, and `gpt-5.3-codex-spark` -> Sonnet 5.
Do not copy the Codex model names into the Claude-specific file.

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

## Context And Handoff Routing

This file stores stable repository policy only. Do not append dated PR status,
benchmark transcripts, release snapshots, or branch-specific checklists here.
Those records remain available in Git history, GitHub issues and pull requests,
release notes, and evidence documents.

For a resumed or compacted session:

1. Confirm the repository is `/home/hamza-usta/GAFIME` and read this file.
2. Read the ignored repo-root `plan.md` when it exists, but treat it only as a
   concise handoff hint.
3. Verify `git status`, `git worktree list`, open PR bases, open issues, and
   hosted checks before acting. Live Git and GitHub state override every handoff
   snapshot.
4. Read only the task-relevant detailed document and release-measure gate. Do
   not scan historical evidence or similarly named workspaces without a
   concrete need.
5. Keep transient status in `plan.md`; keep stable policy here; keep lasting
   design and evidence in `docs/`.

Remove or replace stale local handoffs instead of accumulating them. Preserve
active PR worktrees and branches until their changes are merged or explicitly
abandoned.
