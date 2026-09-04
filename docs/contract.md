# GAFIME v1 Contract

This contract defines the boundaries that GAFIME v1 implementation work must preserve. It is a maintainer policy document, not a performance tuning note. Passing tests does not make a boundary violation acceptable.

## Repository Layout

Tracked project source, runtime, test, and documentation content must converge into:

- `crates/`
- `src/`
- `python/gafime/`
- `tests/`
- `docs/`
- `.claude/skills/` for tracked maintainer/agent guidance and bounded helper
  scripts only

Do not create new source, runtime, test, or documentation homes outside those roots. Required root metadata and bootstrap files may exist only when needed to build, package, discover, validate, or govern the repo. They must not hide backend implementation logic, fallback behavior, source orchestration, or runtime ownership.

Tracked skills must remain operational guidance and validation helpers. They
must not become a runtime input, a backend implementation home, or a way to
bypass this contract.
Every tracked skill declares one mechanically validated audience: `end-user`,
`contributor`, or `both`. Contributor guidance resolves repository truth from
the active checkout. Guidance used with an installed release must match that
release's canonical tag/source; it must fail closed and disclose a mismatch
instead of silently applying newer `main` guidance. The bootstrap and audience
contract is documented in [`docs/agent-skills.md`](agent-skills.md).

`docs/` is the historical and design record. It may be read and extended, but historical docs must not be rewritten, deleted, or collapsed without maintainer approval.

`tests/` must preserve release-relevant tests used by previous releases and the current release. Release-gate tests must not be removed or relocated without maintainer approval.

Ignored local agent memory, release scratch, editor state, and generated
agent/skill caches are outside this tracked-layout rule. They must stay ignored
and must not become runtime inputs.

## Kernel And Orchestration Layout

Kernel and orchestration work must keep device code, host launch code, and Rust interconnect boundaries separated by backend, file role, and compiler.

Target layout inside the root native source tree:

```text
src/
  common/
    gafime_gpu_abi.hpp
    gafime_gpu_internal_abi.hpp

  cuda/
    cuda_api.hpp
    cuda_internal.hpp
    kernels.cuh
    precision_kernels.cuh
    precision_kernels.cu
    precision_launcher.cu
    rt_abi.hpp
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

CUDA `precision_kernels.cu` owns the shared, profile-specialized device
implementations declared by `precision_kernels.cuh`.
`precision_launcher.cu` owns canonical ABI 1.1 numeric-route dispatch plus thin
frozen ABI 1.0 adapters into those shared internals. ABI 1.0 does not retain a
separate complete device-kernel or launcher tree. `kernels.cuh` contains only
shared CUDA-internal launch policy and declarations; it is not an independent
engine owner. `src/common/gafime_gpu_abi.hpp` owns the standard Rust-facing C
ABI declarations; `src/common/gafime_gpu_internal_abi.hpp` owns private adapter
layouts; `cuda_api.hpp`, `rocm_api.hpp`, and `metal_api.hpp` are backend
export/compatibility wrappers.

CUDA RT-core / decision-path acceleration code must stay in the explicit RT files. `rt_abi.hpp` owns the local experiment's optional C ABI declarations. `rt_kernels.cu` owns RT-specific CUDA device kernels, OptiX device programs, point-packing kernels, and exact-filter kernels. `rt_launcher.cu` owns RT-specific host allocation, finite box planning, conservative ordered-float-bucket custom-AABB preparation, cached OptiX IAS/GAS/workspace, exact SM fallback, RT membership dispatch, and the local ABI bridge. `cuda_internal.hpp` may expose only the RT-free opaque matrix view needed by that bridge. The standard `precision_kernels.cu` and `precision_launcher.cu` files must not absorb RT-specific device or host execution logic.

ROCm `kernels.hip` owns HIP `__global__` and `__device__` implementations. ROCm `launcher.hip` owns host launch, graph capture, and `hipLaunchKernelGGL` dispatch.

Metal `shader.metal` owns Metal device kernels. Metal `launcher.mm` owns Objective-C++ command encoder, pipeline state, and dispatch.

GPU payload staging and release packaging must source backend files from this root `src/` layout. Standard CUDA payloads compile only `precision_kernels.cu` and `precision_launcher.cu`; `precision_kernels.cuh` is the CUDA-internal specialization surface and `kernels.cuh` contains shared launch policy. Standard ROCm payloads compile both `kernels.hip` and `launcher.hip`. Local OptiX builds may compile `rt_kernels.cu` and `rt_launcher.cu` and generate embedded PTX from `rt_kernels.cu`, but the source of truth remains the explicit RT CUDA source. Packaging must not reintroduce `gpu/`, crate-local native source homes, kernel-only payload builds, placeholder device files, hidden source copies under old runtime paths, or a second legacy engine.

The standard PyPI CUDA payload is the RT-disabled distribution `gafime-cuda`,
package `gafime_cuda`. It carries only GAFIME binaries, dynamically requires
the system CUDA runtime, and must not vendor `libcudart`, `cudart64`, or
`nvcudart` runtime libraries. OptiX RT is a local CMake experiment only and may
use a distinct local native-library filename selected explicitly through
`GAFIME_CUDA_V1_LIB`.
There is no RT distribution identity. RT source, generated PTX, libraries, and
reports must remain outside every wheel, sdist, workflow or cache artifact,
frozen release bundle, and GitHub Release.

The Linux `gafime-rocm` distribution has one immutable `system` policy,
defined by `.github/scripts/rocm_7_2_3_system_policy.json`. Its Linux wheel must
bundle no ROCm userspace, carry no RPATH or RUNPATH, declare the external
`libamdhip64.so.7` prerequisite, and retain the truthful `linux_x86_64` tag.
Because PyPI rejects raw Linux wheels and this external dependency cannot
truthfully satisfy manylinux, the wheel is attached to the GitHub Release while
PyPI receives the matching source distribution. There is no bundled-runtime
ROCm distribution policy. Coexistence with multiple ROCm userspaces in one
process is not claimed.

Apple Silicon Metal is embedded only in the `gafime` macOS arm64 core wheel.
That wheel owns exactly one paired `libgafime_metal_v1.dylib` and
`gafime_metal_v1.metallib`; every other core wheel must contain neither file.
There is no separate Metal distribution, extra, sdist, wheel, or publisher.
The exact frozen macOS core wheel must execute the installed public Metal path
on Apple hardware before publication.

Precision profiles do not create distribution identities. Every Core wheel
contains Core `fp32`, `mixed`, and `fp64`; every CUDA and ROCm wheel contains
all three backend specializations in one payload binary. The macOS arm64 Core
wheel additionally contains Metal `fp32` only. Package counts and platform
topology remain unchanged.

Core and payload wheels use dedicated CPython ABIs. Python's Stable ABI and
`abi3` are forbidden. Each declared platform must build and test a matching
wheel for CPython 3.10, 3.11, 3.12, 3.13, and 3.14. Windows ARM64 uses
cibuildwheel's NuGet `pythonarm64` provisioner for its target interpreters.

Core must not depend on CUDA or ROCm payload distributions through required
dependencies, extras, or equivalent metadata. Each payload must depend on the
exact matching Core version. Build and publication workflows remain separate:
the build workflow freezes one manifest-complete immutable bundle, and the
publisher may only verify and select byte-identical files from that bundle.
Publication order is Core, CUDA/ROCm, public exact-version install verification,
then GitHub Release. Artifact counts are derived from the per-CPython/platform
manifest and must never be hard-coded in workflow or validation logic.

Experimental CUDA RT/OptiX sources are locally buildable only through
`GAFIME_CUDA_RT_BUILD_MODE` in CMake. They must not enter a wheel, sdist,
workflow artifact, cache artifact, or GitHub Release.

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
- **Forbidden without explicit maintainer approval — math-breaking flags that change numerical results.** Any flag that relaxes IEEE semantics is forbidden because it breaks the selected profile's reviewed lane-specific numerical oracle (`fp32`, `mixed`, or `fp64`). Examples: `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`, `-fassociative-math`, `-freciprocal-math`, `-ffinite-math-only`, `-fno-signed-zeros`, `-ffp-contract=fast` (global FMA reassociation), flush-to-zero / denormals-are-zero (`-ftz=true`), approximate/fast-math transcendental intrinsics, `--use_fast_math` (NVCC), and `/fp:fast` (MSVC).

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

ABI 1.0 is frozen and remains byte-compatible through thin adapters into shared
modern internals. ABI 1.1 is the canonical generic numeric-route boundary: a
caller first enumerates complete routes into caller-owned storage, then passes
one exact route plus validated typed buffer views to generic allocation,
upload, target-update, execute, forecast, significance, diagnostics, and free
operations. Dtype masks are summaries only and never prove support for a dtype
combination. Dtype-suffixed ABI 1.1 upload, execute, result, or significance
symbol families are forbidden.

The ten generic ABI 1.1 symbols listed in `docs/abi-evolution.md` form one
normative operation table. A payload advertising
`gafime_gpu_numeric_routes_v2` must export every member, and the loader rejects
a partial table before allocation. Dynamic symbol lookup may use optional
storage while probing, but missing v2 members are not a generic fallback. The
unsuffixed ABI 1.0 capability symbols retain their separate legacy optional
semantics.

Every extensible ABI 1.1 structure starts with `abi_version` and `struct_size`.
Major mismatch, a missing stable prefix, an unknown required flag, a nonzero
known reserved field, a duplicate or contradictory known route, and an
unsupported dtype fail closed. Newer minor records and explicitly ignorable
tails may be accepted; an ABI 1.1 consumer must skip unknown future routes and
continue to use recognized float routes safely. This is the additive path for
a future ABI 1.2 integer engine: new dtype IDs, numeric routes, overflow
policies, and internal specializations must reuse the generic operations rather
than add one exported symbol family per integer width. Integer execution is not
implemented or advertised in v1. The normative layouts, enum-allocation rules,
and compatibility fixtures are documented in `docs/abi-evolution.md`.

Legacy ABI 1.0 CUDA payloads may expose the optional
`gafime_gpu_permutation_pvalues` ABI to compute permutation-test p-values for
already-surfaced compact result rows in a target-independent family. A current
legacy payload that uses this path under an active `vram_budget_mb` must also
expose the non-mutating `gafime_gpu_permutation_memory_peak` query. That query
accounts for the complete-family score buffers and the retained observed,
family-maximum, and exceedance buffers, including old-plus-new growth
transitions. Older ABI 1.0 payloads without the query remain loadable but must
use the budgeted host-orchestrated maxT path instead of bypassing admission.
Target-dependent adaptive families must repeat their exact device unary
screening and shortlist construction for every permutation. Rust may
orchestrate that bounded sequence through target replacement plus
`gafime_gpu_execute`, provided every family maximum is obtained with device
`top_k=1` ranking in both directions for signed metrics, only bounded rows cross
the ABI, and the original target is restored or the artifact fails closed.
`gafime_gpu_execute` still returns scores only; Rust owns exceedance counts and
p-value calculation and must never infer a null maximum from a
report-compacted subset.

CUDA may expose the optional `gafime_gpu_decision_path_membership` ABI for RT-core/GBDT acceleration. Rust remains the owner of decision-path discovery, feature planning, scheduling, and backend selection. The CUDA payload receives compact validated `GafimeDecisionPathTerm` descriptors and materializes hard-AND membership over the resident feature-major matrix with exact `<=`, `>`, and NaN-undetermined semantics. OptiX RT traversal is allowed only for finite <=3D box batches where exact semantics are preserved. Every 1D, 3D, unbounded, empty, narrow, or otherwise ineligible shape uses a conservative ordered-float-bucket custom AABB for traversal culling and rechecks the original fp32 values and open/closed predicates in the intersection program. A grouped 2D score plan may use fixed-function instanced triangles only when every box is finite and bounded and each axis span is at least `2^-12 * max(1, abs(lo), abs(hi))`; triangle bounds expand by eight binary32 ULPs and any-hit rechecks the same original predicates before accepting a hit. Geometry selection is internal, signature-keyed, and must not be controlled by an environment selector. Three-dimensional paths retain their third coordinate in the exact guard even though the custom acceleration lattice uses two coordinates. The payload must query `OPTIX_DEVICE_PROPERTY_RTCORE_VERSION` and fail closed when it reports no RT-core support; architecture names are not capability proofs. Duplicate intersection callbacks must not duplicate membership or direct statistics. Otherwise CUDA must use its exact SM comparator or return unsupported when RT is explicitly required. The symbol is CUDA-only during the spike; ROCm, Metal, and older CUDA payloads must report unsupported by omitting the symbol, not by falling back to another backend.

CUDA may expose the optional `gafime_gpu_decision_path_score` ABI for compact RT-core/GBDT scoring. It accepts the same Rust-owned path descriptors plus metric ids and returns compact `GafimeResultTable` rows. During the spike this score ABI supports only Pearson and R2 for finite-feature decision paths; unsupported metrics must return unsupported, not fabricated zeros. CUDA may split a mixed-axis score batch into internal <=3D RT groups and direct modes must preserve exact feature-pair groups before widening compatible lower-dimensional work, but it must preserve original path order and must not move discovery, scheduling, or fallback policy out of Rust. CUDA may use an internal duplicate-safe device bitset or direct duplicate-safe traversal statistics, but it must not copy full path-major membership to host on the scoring path. Direct traversal statistics are opt-in through `GAFIME_CUDA_DECISION_PATH_RT_SCORE=direct`; target-wide statistics and centered per-path sums use double precision, but floating atomic order remains tolerance-checked at the approved `1e-4` spike threshold and must not become the default without maintainer approval. First-hit direct traversal statistics are opt-in through `GAFIME_CUDA_DECISION_PATH_RT_SCORE=firsthit` and are allowed only when CUDA proves every RT group is finite, bounded, 2D, and non-overlapping. The non-overlap proof plus `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT` must make each ray contribute at most once, so neither the single-group planned path nor the multi-group instanced path may allocate or clear the path-row duplicate bitset; ordinary direct mode must retain its duplicate guard. Both paths must consume one shared compile-time-testable duplicate-guard policy. Otherwise CUDA must return unsupported instead of falling back or changing semantics.

The public compact score route is limited to the complete, untruncated unary base-plus-path family with Pearson and/or R2, finite RT-representable inputs, and no graph or significance request. Rust executes base unary rows through the normal CUDA plan, appends compact path rows in global candidate order, and retains fallback ownership. Every other public shape uses the established membership-expansion plus continuous-scoring path. An explicit require-RT policy must fail closed rather than select that fallback.

Rust exposes decision-path execution policy as `DecisionPathRtPolicy`. `AllowSmFallback` sends no RT-required flag and permits the CUDA payload to use its exact SM implementation when OptiX cannot execute the validated batch. `RequireRt` sends `GAFIME_DECISION_PATH_FLAG_REQUIRE_RT`; Rust must reject a missing decision-path symbol or a device that does not advertise `GAFIME_GPU_DEVICE_FLAG_OPTIX_RT` as unsupported before treating any fallback as successful. CUDA must also return unsupported rather than execute the SM path when the required flag reaches the payload.

OptiX program, custom-AABB GAS, and workspace state is owned per CUDA device. A device execution lock covers the complete RT membership or score operation, so same-device calls cannot race mutable OptiX state; different device ids never share a program, context, stream, GAS, or workspace. Every execution and teardown must establish the requested device with `cudaSetDevice` and restore the calling thread's previous device. SM decision-path row grids must use the queried `maxGridDimY` and tile with a 64-bit row offset instead of assuming the legacy 65,535-block `grid.y` limit.

CUDA payloads that own this RT cache expose the optional lifecycle symbol `gafime_gpu_decision_path_release_device_state(device_id)`. `gafime-gpu-sys` shares one cleanup owner per loaded payload and device and invokes the symbol only after the final owning matrix has been freed. This teardown synchronizes against in-flight RT execution and releases both custom-AABB geometry programs and their high-water device allocations. Direct C ABI owners must call the lifecycle symbol after freeing their final matrix for a device. Older payloads may omit the symbol and remain loadable; they simply do not provide explicit RT-cache teardown. The empty host registry container may remain process-lifetime to avoid calling OptiX after vendor-library shutdown; successful last-owner cleanup must erase its device state, and its remaining host bucket footprint is bounded by valid CUDA device ids.

An older CUDA payload that omits the lifecycle symbol may also predate native
same-device RT serialization. The current Rust host must serialize its
decision-path calls per payload and device so separate backend objects cannot
race that legacy mutable state. Current payloads with the lifecycle symbol keep
their native locking path and must not pay this compatibility mutex.

Arrow C Data / Arrow C Stream is the v1 framework-integration protocol. Polars is the external tabular compatibility layer for ingest and manipulation; GAFIME owns compute memory after validation and exports compact result tables over Arrow. GAFIME v1 deliberately supports `polars>=1.3,<2`. Polars 2 changes API and migration has a real compatibility cost that is not an objective of GAFIME v1; that migration belongs to the dedicated v1.1 or v1.2 work tracked by [issue #87](https://github.com/onlyxItachi/GAFIME/issues/87). Legacy DLPack/native-buffer export must not be reintroduced as a fallback or compatibility shortcut without explicit maintainer approval.

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

`backend="auto"` is a Rust-owned ranked resolver. It must rank usable GPU device payloads above CPU, then rank CPU vector ISA above scalar CPU (`AVX512 > AVX2 > SSE4.2/NEON > scalar`). A GPU candidate is usable only when its configured C ABI payload loads, `gafime_gpu_device_info` succeeds for the requested `device_id`, and the ABI 1.1 capability masks advertise the requested precision profile together with its required storage and result dtypes. In particular, automatic `mixed` and `fp64` selection must exclude Metal. Explicit `cuda`, `rocm`/`hip`, and `metal` requests must fail closed when the requested profile is unsupported and must not fall back to another backend.

`GafimeEngine.analyze()` may keep a bounded v1 resident-analyze cache for continuous workloads when `ComputeBudget.keep_in_vram` is true. The cache key must be content-derived from the validated feature matrix in its selected storage dtype, the precision profile, feature names, backend/config payload, and native boundary identity; it must not depend only on Python object identity. A target-only change may reuse the resident feature matrix through the matching typed native `update_target` boundary. A feature-content or precision-profile change must compile/upload a new resident matrix. `GAFIME_V1_ANALYZE_CACHE_SIZE=0` and `keep_in_vram=False` must disable this public analyze cache. Cache eviction must close native artifacts and must not introduce backend fallback or numerical changes.

Metal uses the same additive `gafime_gpu_*` C ABI as CUDA and ROCm. The Metal shader implements a genuine lane-wide `fp32` route for continuous Pearson/R2, fixed-bin mutual information, and Spearman scoring; numerical parity against the fp32 reference is gated by Apple-hardware validation. Metal Shading Language has no native shader fp64, so explicit Metal `mixed` and `fp64` requests fail before coercion, payload discovery, allocation, or execution. Automatic selection may exclude Metal and choose Core, but an explicit Metal request must never silently run CPU double arithmetic or software-emulated fp64. Metal mutual information clamps bins to <= 48 so the joint histogram fits threadgroup memory. Graph capture/replay and backend-native permutation replay remain unsupported on Metal; Rust-orchestrated target replacement plus exact Metal screening/ranking is the approved bounded maxT path. Unsupported profiles, metrics, graph replay, missing Metal payloads, and unavailable Apple runtime support must return explicit errors through the boundary and must never silently route to CPU, Python, CUDA, or ROCm.

Metal host-side interaction centering must preserve the selected `fp32` lane,
including fp32 column-mean accumulation and non-finite propagation. The
offline Metal shader compiler must use `-fno-fast-math`; Apple enables unsafe
fast math by default, which can move exact fp32 histogram-boundary values into
the wrong integer bin and can relax non-finite semantics. The integer histogram
is built in parallel, while MI probability, correction, logarithm,
normalization, and final-score accumulation use deterministic row-major bin
order in fp32. The macOS gate must execute CPU-oracle parity for all four
continuous metrics on
high-dynamic and NaN/Inf inputs plus multi-block ascending/descending top-k.
`GAFIME_METAL_PARITY_TOLERANCE=0.00005` is the approved absolute fp32 release
tolerance for the direct ABI 1.0 compatibility and short ABI 1.1 genuine-fp32
gates. Apple-hardware run `30207767348` observed a worst-case
absolute delta of `4.045665264e-6` on the legacy surface. For ABI 1.1 fp32
inputs of up to 256 rows, Metal must compute the paired finite correlation
means in Core row order before retaining its parallel fp32 covariance pass;
this must not affect ABI 1.0 or serialize larger workloads. The broader
end-to-end fp32 cross-backend gate retains its separate `2e-4` absolute and
`2e-5` relative bounds for backend reduction-order differences. Increasing
either gate requires new Apple-hardware evidence and explicit maintainer
approval.

## Numerical Policy

GAFIME targets bit parity with the approved reference implementation for every backend.

Integer, categorical, indexing, histogram, and all deterministic outputs require exact bit parity. Floating-point outputs are also expected to achieve bit parity whenever mathematically and architecturally possible.

If strict bit parity cannot be achieved because of unavoidable hardware or compiler differences, such as fused operations, ISA-specific instruction selection, or backend-defined floating-point behavior, the implementation must:

- explicitly document the reason
- justify why bit parity is impossible
- define the approved numerical tolerance
- prove equivalence through validation tests

Performance improvements are never accepted as a justification for undocumented numerical differences.

Correlation finalization must distinguish a mathematically constant input from
failed arithmetic. An exact zero variance produces correlation and R2 values of
zero. A non-finite variance, covariance, denominator, or normalized correlation
produces NaN, and R2 must preserve that NaN. CUDA, ROCm, and Metal ranking must
exclude a non-finite primary score; no clamp or min/max operation may convert an
arithmetic failure into a plausible endpoint such as Pearson `-1` or R2 `1`.

Continuous GPU covariance has two explicit numerical paths. During matrix
upload, CUDA, ROCm, and Metal record conservative base-two magnitude bounds for
each feature and the target. Immutable protocol preparation combines those
bounds with arity and sample count once per descriptor chunk. Chunks whose
selected-profile reduction sums, variances, or correlation denominator can
exceed the conservative guarded exponent range use a three-pass scale-normalized
covariance kernel; ordinary chunks keep
the established two-pass or cached-statistics kernel. Pearson and R2 are
invariant to the positive normalization, so path selection changes numerical
conditioning rather than the metric definition. Target replacement invalidates
the cached selection. The robust path cannot recover an interaction product
that itself overflows before normalization; finite interaction materialization
remains part of the input-domain contract.

Core and current GPU payloads expose post-selection interaction-materialization
diagnostics without changing that numerical contract. For each surfaced
candidate, the runtime must distinguish a non-finite source or stored mean from
a finite-input centered subtraction or sequential product in the selected
pointwise dtype that becomes non-finite. The count and ratio are observational
only: candidate identity,
scores, ranking, significance inputs, graph state, and cache identity must not
change. The ordinary finite path must use metadata gathered during existing
matrix conversion plus a conservative prefix proof; only unproven surfaced
combinations may receive an exact row scan. The canonical ABI 1.1
`gafime_gpu_interaction_diagnostics_v2` operation is mandatory for a payload
advertising `gafime_gpu_numeric_routes_v2`. The unsuffixed
`gafime_gpu_interaction_diagnostics` symbol is an optional ABI 1.0 capability,
so older ABI 1.0 payloads remain loadable and report diagnostics unavailable.
One aggregate report warning is
emitted only when a surfaced overflow count is non-zero. Widened or log-domain
interaction evaluation is a separate future numerical mode, never an implicit
diagnostic side effect. The full schema and semantics are in
`docs/precision-contract.md`.

CPU fixed-bin mutual information is the CPU parity path for the GPU-compatible MI approximation. Its SIMD implementation must preserve exact fixed-bin histogram counts against the scalar/index reference, keep the same finite-sample correction and normalization, and stay gated by release-measure architecture checks plus focused Rust tests.

`EngineConfig.precision` is the single keyword-only public precision surface
and defaults to `mixed`. `fp32` uses fp32 ingest/storage, pointwise arithmetic,
reductions, ranking, and public results. `mixed` uses fp32 ingest/storage and
pointwise arithmetic with fp64 reductions, ranking, and public results. `fp64`
preserves fp64 across all four domains with no intermediate fp32 quantization.
The profile applies to all metrics, continuous/time-series/decision-path
families, significance, ranking, eager/resident/compiled execution, graph
replay, target replacement, and cache identity. Structural planner metadata
retains its integer/control types. The narrow deprecated pair parser accepts
only `float32+fast -> fp32`, `float32+stable -> mixed`, and
`float64+exact -> fp64`; the old fields are not an independent public surface.

ABI 1.0 float layouts and entry points remain byte-for-byte compatible through
thin adapters and never reinterpret a float pointer as double. The additive ABI
1.1 surface provides authoritative numeric-route enumeration, generic typed
buffer operations, one typed numeric result/significance representation, and
dtype-correct memory forecasts. Core, CUDA, and ROCm enumerate and execute all
three routes. CUDA and ROCm compile all three specializations into each existing
payload and select a specialized function table once during resident-plan
construction. Metal enumerates and executes only fp32. The full arithmetic,
ingest-ordering, identity, admission, and negotiation requirements are
documented in `docs/precision-contract.md` and `docs/abi-evolution.md`.

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

## Release Version Policy

The Cargo workspace version is the canonical repository release input and uses
strict Semantic Versioning. `.github/scripts/release_version.py` is the
authoritative parser and mapping implementation. It derives the corresponding
PEP 440 identity and validates source metadata, Cargo lock entries, release
notes, tags, and frozen artifact metadata.

Cargo, changelog headings, release-note filenames, Git tags, and GitHub
Releases use `MAJOR.MINOR.PATCH` or the prerelease forms
`MAJOR.MINOR.PATCH-alpha.N`, `MAJOR.MINOR.PATCH-beta.N`, and
`MAJOR.MINOR.PATCH-rc.N`. Tags and release-note filenames add the `v` prefix.
Python runtime metadata, `pyproject.toml`, exact-version payload dependencies,
wheel/sdist metadata, and PyPI use the mapped PEP 440 forms
`MAJOR.MINOR.PATCH`, `MAJOR.MINOR.PATCHaN`, `MAJOR.MINOR.PATCHbN`, and
`MAJOR.MINOR.PATCHrcN`.

Public release tooling must fail closed on unsupported labels, ambiguous
spellings, SemVer build metadata, and PEP 440 development, post, epoch, or local
versions. Prerelease classification is parser-derived, not inferred from
substrings. Historical published tags and release records remain immutable;
recognizing a legacy spelling for inspection or recovery never authorizes that
spelling for a new release.

## PR, Main, And Release Gates

Implementation testing, review, and pushes normally happen on a feature branch and PR. `main` may receive implementation changes only after the work proves:

- numerical bit parity or explicitly approved numeric tolerance for the affected backend/metric
- compatibility with this contract
- verified tests and release gates for affected runtime surfaces
- a concrete beneficial update relative to the current implementation

Changes must remain reviewable and semantically focused. Independent
architecture, feature, and fix work belongs in separate focused or explicitly
stacked PRs; mutually dependent documentation, validation, and implementation
needed to make one change coherent may remain together.

Autonomous and AI-assisted contributions are subject to the same contracts,
tests, evidence, review, provenance, safety, numerical, and release gates as
every other contribution. Agent authorship neither weakens those gates nor
creates an additional human-authorship or approving-review requirement.

`main` remains protected and accepts tracked changes only through a pull request. The required GitHub approving-review count is zero; independent human approval is not required. `@onlyxItachi` is the sole final merge authority.

Before merge, every PR must have:

- a current-head AI Review Record submitted as a GitHub review
- all configured required status checks reported for the final head after executing against GitHub's current PR merge commit for that head/base pair
- all review conversations resolved

A `COMMENTED` review is valid review evidence; an `APPROVED` review state is not required. The AI Review Record must state the model, role, exact reviewed commit SHA, verdict, and findings. The reviewed SHA must equal the current PR head. A later head commit invalidates the record and requires a new review. A base change invalidates the merge-commit CI evidence and requires the configured checks to run against the new merge commit. A merge-blocking verdict or unresolved blocking finding prevents merge.

Candidate stabilization branches use `release/v<canonical-semver>` and are cut
only from an exact green `main` commit. Branch creation changes no release
identity and authorizes no tag or publication. Release-branch changes remain
PR-only, merge-commit-only, protected by the same review/check/thread gates,
and limited to bounded stabilization for the named candidate. Durable fixes
normally land on `main` first and retain backport provenance; an urgent
release-first fix must be carried into `main` through final admission. A
divergent `main` must not be merged into the candidate branch. The exact settled
release tip is the build, freeze, tag, and publication source. Before tagging,
admit that unchanged tip to `main` through a temporary branch based on current
`main` and an ordinary strict PR, making the candidate an ancestor without
moving it. Publication must mechanically bind the build head branch to
`release/<tag>` and prove build SHA, current branch tip, canonical tag, and the
admitted candidate commit are identical. After accepting the frozen candidate
and before admission, tagging, or publication, install exact-ref
update/deletion protection and retain that read-only lock after publication. A
later source fix requires a deliberate unlock, reviewed PR, new exact-tip
build, and restored lock. The detailed lifecycle is in
[`docs/releases/release-branches.md`](releases/release-branches.md).

Canonical `v*` release tags are governed by two layered active rulesets: an
authorized creation-only rule and a separate no-bypass update/deletion rule.
The creation bypass never permits an existing tag to move or be deleted. After
admission, create the canonical tag on the exact frozen tip, verify both rules
apply, and only then dispatch publication from the tag ref. Publisher workflow
identity, downstream checkouts, live tag/branch refs, build source, and
job-local pre-upload rechecks must resolve to the same commit.

A release must never:

- introduce silent backend behavior changes
- change numerical output without documentation
- weaken ownership rules
- weaken safety rules
- bypass PR gates
- introduce undocumented compile flags
- change ABI unexpectedly

Intermediate PR commits do not need to be green. Merge eligibility is based on the final reviewed head: it must have a current-head AI Review Record, and all configured required checks reported for that head must pass after validating GitHub's current PR merge commit for the exact head/base pair. Workflows configured for `main` must then validate the resulting commit on `main`; a failure blocks release use and follow-on integration until it is corrected or reverted through another PR.

Validation starts from the top-level Python API to guarantee user-space stability.

Each PR must validate:

- numerical correctness
- performance validation proportionate to affected critical paths, including
  backend-local and end-to-end Python API evidence when execution behavior
  changes; documentation/governance-only changes do not require unrelated
  benchmark campaigns
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

Production Core execution is throughput-first. Independent candidate work must
use multi-core parallel execution, while arithmetic within each candidate uses
the strongest semantics-preserving SIMD/native path available. A new dtype,
precision profile, ABI generation, operator family, or executor must not
silently serialize an established parallel production workload.

Single-core microbenchmarks are supplemental leaf-kernel diagnostics. They are
not Core product-throughput evidence and must not be reported or gated as such.

## Migration Rules

Do not treat placeholder GPU files as real runtime sources. Do not delete legacy backend/device code until the v1 structure carries required capability and equivalence tests pass.

Move or split real device-side code into the contracted backend layout before old backend connections are cut. Preserve roadmap, release notes, design docs, and agent contract files unless the maintainer explicitly asks for removal.

The v1 direction is Python -> PyO3/Rust -> Rust CPU / GPU C ABI. Python must not own continuous backend execution planning loops or GPU permutation loops. Rust owns candidate specs, compact result state, scheduling, and native backend dispatch. The packaged `gafime.compile.scenario` module is a bounded v0.5 compatibility projection only: it emits at most one metadata descriptor per configured arity, never materializes candidates, and is never passed to native execution. GPU backends expose explicit C ABI launcher surfaces to Rust and keep backend-specific kernel orchestration inside their contracted source trees.
