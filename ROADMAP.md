# GAFIME v1.x — Master Roadmap

**Supersedes `plan.md` (architecture restructure — DONE) and the v1 capability-completion notes.**
v1 is architecturally superior and faster than v0.5 legacy, but **capability-narrower**. This
roadmap closes every legacy gap, lands two frontier bets as first-class work, and rides the
current upstream toolchains — in priority order, each phase shipped as a parity-gated increment
(no big-bang; golden oracle holds after every phase).

Three goals:
1. **Full v0.5.0 legacy parity** (the inventory below is the contract).
2. **Two frontier works, first-class:** RT-core GBDT borders + hardware-autotune hardening.
3. **Upstream-aware lowering:** every phase exploits what recently landed in LLVM 22.1.x /
   CUDA 13.3 / ROCm 7.2 / Metal 4 / Rust 1.94 where it helps GAFIME.

## Implementation status (updated 2026-07-02)

| phase | status |
|---|---|
| **P-A** permutation + stability (CPU **and** GPU) | ✅ **DONE (both backends)** — native Westfall-Young maxT + bootstrap stability (`gafime-cpu/significance.rs`), surfaced through the boundary + adapter, gates `Decision`; detects signal, rejects noise. GPU run mines all candidates on-device then does the bounded top-K significance pass on the host (identical decisions to CPU on the 4060). On-device WHILE-node null distribution remains a future perf optimization. |
| **P-C** decision_path end-to-end (CPU **and** GPU) | ✅ **DONE** — depth-k GBDT + boosting + membership + expansion, wired boundary/adapter/api, family descriptors flipped. Validated on CPU and on the 4060 (routes via continuous). |
| **P-D** legacy surface | 🟡 mostly done — `GafimeSelector` sklearn transformer; **export via `CompileFlags(export=True)`** (zero-copy Arrow); saturating wide-count planning (`combos.rs`); telemetry (`tools/telemetry.py`); **VRAM budget enforcement** (fail-fast vs OOM in `prepare_continuous_execution`). Resident-session reuse + whole-sweep graph still open. |
| **P-F** CUDA build + validation | ✅ **CUDA lib builds + validates on RTX 4060** (continuous GPU==CPU arity 1-5, top-k, MI, graph replay, host permutation loop; time_series + decision_path + significance on GPU). On-device WHILE-node null distribution still open (perf). |
| **cross-cut** compiler policy | 🟡 partial — `[profile.release]` lto=fat/codegen-units=1 + CUDA nvcc flags (sm_89+PTX/-O3/line-info) landed. |
| **P-E** ROCm parity | 🟢 **continuous pearson/r2 + MI + spearman + top-k + significance on gfx1150** — ROCm lib builds (`src/rocm`), ABI-generic backend wired into gpu-sys (`rocm_from_env`) + Python (`backend="rocm"`/`"hip"` → `v1-rocm-cabi`); MI + spearman kernels ported from CUDA (match CUDA/CPU on hardware); host-side top-k. ROCm graph (device-copy) still open. |
| **P-B** metric×backend | ✅ **DONE for v1 parity pass** — spearman on CPU+CUDA+ROCm (pearson-on-ranks kernel, matches CPU on both live GPUs); MI on ROCm; pearson/r2/MI everywhere. CPU fixed-bin MI approximation now uses a fused AVX2-fed histogram path (`fixed_bin_histogram2d`) with exact parity against the previous bin-index path. Spearman rank sorting remains correctness-first scalar and is a future perf-only kernel, not a metric/backend gap. |
| **P-G** Metal | ⏳ honest stub — no Apple hardware; `backend="metal"` reports a clear capability error; Metal-4 design documented. Deferred until Apple HW. |
| **cross-cut** ILP | 🟢 centered-sum reduction widened 2→4 accumulator chains (parity-safe). |
| **P-H / P-I** | ⏳ open (RT-core OptiX borders; hardware autotune) — excluded from the current pass. |

Verification this session: cargo (incl. CUDA+ROCm gpu-sys on the live 4060 + gfx1150, run with `--test-threads=1`) + python suites green; GPU parity + significance validated on both live GPUs.
Note: the CUDA-gated `gafime-gpu-sys` tests share one GPU — run them with `cargo test -p gafime-gpu-sys -- --test-threads=1` (parallel launches contend on the single 4060).

> Current state (verified Q2 2026 on the dev box): rustc 1.94.0, clang/LLVM 22.1.6, nvcc 13.3 /
> CUDA 13.3, HIP 7.1 / ROCm 7.1 (7.2.x upstream), Python 3.14.3. Workspace deps: pyo3 0.22
> (abi3-py310), arrow 54.3.1 (ffi), rayon 1 — SIMD is hand-written `core::arch` (no `pulp`/`bindgen`).
> `crates/gafime-cpu/src/dispatch.rs` already carries the full ISA ladder (avx512/avx2/sse42/neon/
> scalar). The architecture restructure (one-directional Python→Rust→native, single PyO3 boundary,
> frozen `src/common/gafime_gpu_abi.hpp` with `GafimeLaunchProtocol` + `GafimePermutationSchedule` +
> `gpu_execute`, compact `ResultTable`) is **done** — this roadmap is about capability + performance,
> not structure.

---

## 0. Upstream levers (current releases → concrete GAFIME benefit)

| Upstream (release) | What landed | GAFIME benefit | Phase |
|---|---|---|---|
| **CUDA conditional graph nodes** — `WHILE`/`IF` (CUDA 12.4+; box=13.3) | `cudaGraphConditionalHandleCreate` + device-side `cudaGraphSetConditional(handle,v)`; WHILE body re-loops on-device until condition=0; body via `cudaStreamBeginCaptureToGraph` | **The permutation/stability loop becomes ONE on-device graph** — body = {refresh `y` → 7-node score+rank+compare → accumulate exceedances → decrement counter} with **zero host sync per permutation**. Kills the 0.715× regression at its root (loop was in Python). | P-A, P-F |
| **CUDA 13.3 green contexts** | Partition SMs into disjoint contexts/streams; "shield latency-sensitive kernels from throughput kernels" | Run RT-core border eval on one SM partition **concurrently** with MI/continuous kernels on another (RT cores are otherwise idle); dual-workload autotune; overlap rank with score | P-H, P-I, cross-cut |
| **CUDA 13.3 CompileIQ** (compiler autotuning) | Evolutionary/genetic per-kernel compiler-config search; ~15% on already-tuned GEMM/attention | Fold into the autotune harness instead of only hand-set `__launch_bounds__`; per-arch cached configs for continuous/MI/decision_path kernels | P-I |
| **CUDA 13.3 CUDA Tile (C++)** + CCCL 3.3 `DeviceFind::FindIf` (≈7×) | High-level tile kernels (auto smem/async); faster device search | Candidate path for smem-staged bf16/int8/fp8 kernels (v0.2 naive 1-warp kernels needed staging); faster top-k/rank scan | P-B, P-F |
| **LLVM 22.1 — APX + AVX10.2** | EGPR (extra GPRs), NDD 3-operand non-destructive, CCMP/conditional, `-march=novalake/wildcatlake`; more AVX/AVX-512 intrinsics constexpr + wrapping generic builtins; **atomic vector loads legal in IR** | Add an APX/AVX10.2 multiversion rung when exposed (fewer reduction-loop spills via EGPR; NDD cuts `mov` churn; CCMP = branchless border indicator); atomic vector loads help lock-free partial-histogram merge | P-B, lowering policy |
| **ROCm 7.2** | Optimized HIP graph dispatch (multi-list, doorbell-ring, memset-node opt, async-handler lock removed); `hipStreamCopyAttributes`; broader Radeon support | Lower per-launch overhead on gfx1150; mirror the CUDA host design; track HIP conditional-node availability for the on-device loop | P-E |
| **Rust 1.94** | AVX-512 target features/intrinsics stable (since 1.89); 1.94 adds AVX-512 FP16 + AArch64 NEON FP16 intrinsics + `array_windows::<N>()`; portable-simd still **nightly** | `array_windows` for fixed time_series rolling windows + SIMD chunking (bounds-check elided); optional fp16 ingest/storage rung (halve bandwidth, keep fp32 accumulate); confirms staying on `core::arch` | P-B, P-C, lowering |
| **Metal 4 / 4.1** (WWDC25) | Residency sets, `MTL4CommandBuffer` + argument tables, `MTLTensor` | Encode-once/replay-many over a residency set = the Metal analog of resident stable pointers | P-G |
| **PyO3 0.22 / arrow-rs 54 / Polars** | abi3-py310; zero-copy Arrow C Data Interface; Polars GPU (cuDF) engine | Keep `allow_threads` GIL-free Rust; free-threaded is a **non-lever** (abi3 ⊥ free-threaded, parallelism already in Rust/rayon); Polars-GPU for on-device ingest per platform | P-D, cross-cut |

Sources are listed in the Appendix.

---

## 1. v0.5.0 legacy feature inventory  ← parity contract (Goal 1)

| Legacy v0.5.0 capability | v1 status today | Closed by |
|---|---|---|
| Continuous interaction mining (combo gen, feature spine, arity 1..K, budgets) | ✅ CPU+CUDA | done |
| Metrics: **pearson / r2** all backends | ✅ CPU(SIMD)+CUDA+ROCm | done |
| Metric: **spearman** | 🟡 CPU scalar only (no GPU) | P-B |
| Metric: **mutual_info** (adaptive bins 12/24/48/96, quantile/rank, finite-sample bias correction, support floors) | 🟡 CPU scalar + CUDA; **no ROCm**; CPU not SIMD | P-B |
| **Permutation tests** (null-dist p-value, `permutation_tests`, `permutation_p_threshold`) | 🔴 **missing all backends** | **P-A** |
| **Stability** (`num_repeats`, std across resamples, `stability_std_threshold`) | 🔴 **missing all backends** | **P-A** |
| Decision gating on p-value + stability thresholds | 🔴 today `decision=bool(interactions)` | **P-A** |
| Ranking / top-k | ✅ CPU+CUDA; 🔴 ROCm | P-E |
| **time_series** family (lags, rolling windows, velocity; metric-cache/residency; budgets) | 🟡 CPU end-to-end; GPU wired-untested | P-C/P-F |
| **decision_path** family (native whole-data split finder, depth-k recursion, residual boosting, split caps, conjunction paths; NO sklearn) | 🟡 split-finder **core only**, not wired | **P-C** |
| Discrete-function family (soft threshold/interval/gated/rectangle) | ⛔ **intentionally retired** in v0.5 → replaced by decision_path | n/a (guardrail: do not resurrect GPU discrete dispatch) |
| Backends: CPU SIMD (scalar/SSE4.2/AVX2/AVX-512/NEON) | ✅ | done |
| Backends: CUDA (sm_89), ROCm/HIP (gfx1150), Metal | 🟡 CUDA wired (unbuilt this line), ROCm subset, Metal stub | P-E/P-F/P-G |
| Backend selection: `backend="auto"` payload routing, explicit cuda/rocm/hip/metal, `device_id` | 🟡 partial | P-D |
| **Compile API**: `CompileFlags(plan,graph,export)`, `CompiledGafime`, `compile()` | 🟡 plan ✅; **export → `V1UnsupportedError`**; graph plumbed | P-D |
| **Resident sessions** (matrix residency, reusable sessions, TS metric-cache reuse, in-place `y` update) | 🟡 primitives exist; reuse-through-artifact partial | P-D |
| **Graph capture** (CUDA replay/destroy/available, `matrix_update_target`; HIP mirror; Metal deferred) | 🟡 per-shape; **whole-sweep is the win** | P-A/P-F |
| **Export handles** (native buffer ownership, lifetimes, NumPy/Torch DLPack, zero-copy downstream) | 🟡 DLPack landed on native buffers; **compile-flag export errors**; device DLPack pending | P-D |
| **Power-user planning** (`max_feature_candidate=-1`, saturating counts, safety caps/warnings, no Python-list materialization) | 🟡 verify in `plan/combos.rs` | P-D |
| **Telemetry** (schema `gafime.telemetry.v0.5.0-rc1`, machine-readable artifacts, release-notes-from-artifacts) | 🔴 **missing in v1** | P-D |
| **sklearn transformer** (`gafime/sklearn.py`) | 🔴 **missing in v1** | P-D |
| CLI | ✅ `python/gafime/cli.py` | done |
| VRAM budget (`keep_in_vram`, `vram_budget_mb`) | 🟡 config carried; enforce in schedule | P-D |
| Data ingest: Polars/Arrow, **no numpy in kernel** | ✅ zero-copy Arrow ingest+export | done |
| Reporting: `DiagnosticReport`, `InteractionResult(family=…)`, stability/permutation, decision+reason | 🟡 shell ✅; stability/permutation empty | P-A |

---

## 2. Compiler & lowering policy  ← applies to EVERY phase (Goal 3)

**Math/parity invariants (never violated by any optimization):**
- **fp32 by design** (storage + lanes), **f64/Kahan reduction accumulators**; **centered/two-pass
  covariance** (`Σ(dx·dy)`, never `n·Σxy−Σx·Σy`) — the landed fp32-drift fix. Golden oracle
  (Δ≤1e-4 GPU, exact CPU) holds after every phase.
- **No global fast-math** (`-ffast-math` / nvcc `--use_fast_math`): FTZ + reassociation + approx
  transcendentals would break the f64-accumulator parity oracle. Use **selective** intrinsics
  (`__logf`/`__fdividef` in MI only where measured safe) instead.

**Rust (the CPU engine + PyO3 boundary):**
- Add `[profile.release]`: `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`. **Keep
  `panic = "unwind"`** — the pyo3 `cdylib` catches panics at the FFI boundary; `panic="abort"`
  across the Python unwinding edge is unsafe.
- **`target-cpu` strategy:** distributed wheel ships **portable** (baseline ≈ `x86-64-v3` = AVX2)
  and selects AVX-512 / future APX-AVX10.2 at **runtime** via the existing `dispatch.rs`
  multiversioning — do **not** bake `target-cpu=native`. Provide a local-bench profile
  `RUSTFLAGS="-C target-cpu=native"`.
- **FMA contraction is explicit**: stable Rust/LLVM will not fuse `mul`+`add` without fast-math →
  reduction inner loops must call `f32::mul_add` / `_mm*_fmadd_ps` (already done in
  `pearson_sums_avx512`/`centered_sums_avx512`). Audit every kernel for it.
- **ILP / instruction ordering**: FMA is latency-bound (~4-cycle latency, ~2/cycle throughput on
  Zen/Skylake-class) → need **≥4 independent accumulator chains** to saturate the FMA ports.
  Today's reductions use **2**; widen to **4** where registers allow (APX EGPR raises the register
  ceiling on future Intel). This is a real throughput lever, parity-neutral.
- **Bounds-check elision**: use Rust 1.94 `array_windows::<N>()` for fixed windows / SIMD chunking;
  prefer iterator/`chunks_exact` forms LLVM proves in-bounds.

**CUDA / nvcc 13.3 (`src/cuda/CMakeLists.txt`):**
- `-O3 -Xptxas -O3`; **`-gencode arch=compute_89,code=sm_89`** (4060) **+ `code=compute_89`** PTX
  for forward-compat; `--generate-line-info` (profiling; never `-G` in release);
  **`-Xptxas -v`** in CI to watch register/spill counts.
- **`__launch_bounds__(BLK, MINBLOCKS)` per kernel, autotuned** (P-I), not hand-set — the
  discrete-selector `(256,5)` was hand-picked; replace with measured per-arch values.
- Conditional-graph WHILE-node device kernel must compile under the graph body-graph constraints
  (kernel/memset/memcpy/child/nested-conditional nodes only; one context).

**C++ host / HIP:**
- Host `.so`: `-O3 -march=x86-64-v3` (portable) / `-mtune=native` local; `-fno-math-errno`,
  `-fno-trapping-math`, `-funroll-loops`; `-flto=thin`. Not `-ffast-math` (parity).
- HIP: `--offload-arch=gfx1150 -O3`; ride ROCm 7.2 optimized graph dispatch (lower launch latency).

---

## 3. Phases — WHAT · WHY-not-skippable · WHERE · LOWERING/UPSTREAM · DONE-WHEN

### P-A — Permutation tests + stability  🔴 **the parity blocker — start here**
- **WHAT:** compute permutation null-distribution p-values (shuffle `y`, re-score, count
  exceedances) and stability (re-score across resamples, metric std); populate
  `report.permutations` + `report.stability`; gate `decision` on `permutation_p_threshold` +
  `stability_std_threshold` instead of `bool(interactions)`.
- **WHY:** GAFIME's core differentiator — *is the interaction real or noise?* v1 has none
  (`v1_adapter.py` hardcodes `stability=[]`, `permutations=[]`). Bigger than any vendor kernel;
  without it v1 cannot replace legacy.
- **WHERE:** `python/gafime/v1_adapter.py` (stop hardcoding), `crates/gafime-orchestrator/
  src/{reduce,schedule}` (aggregate p/stability into the compact table), `crates/gafime-cpu`
  (rayon-parallel permutation re-scoring — composes with candidate parallelism),
  `src/cuda/launcher.cu` (drive `GafimePermutationSchedule` already in the ABI).
- **LOWERING/UPSTREAM:** GPU permutation loop = the **CUDA conditional WHILE-node** design — one
  graph, condition handle initialized to `permutation_count`, body refreshes `y` (memcpy node /
  on-device index gather) → scores → updates exceedance counters → `cudaGraphSetConditional(--n?1:0)`;
  **one launch, zero host sync per permutation**. CPU path = rayon over permutations with f64
  exceedance accumulators.
- **DONE-WHEN:** p-values + stability populated and gating `decision` on CPU and (built) CUDA;
  large-N parity vs a f64 oracle.

### P-B — Metric × backend completeness (+ MI-histogram SIMD) ✅ DONE for v1 parity pass
- **WHAT:** **spearman on CUDA/ROCm** (device rank-transform → pearson; `metric_supported` at
  `src/cuda/launcher.cu` historically excluded it); **mutual_info on ROCm** (port CUDA fixed-bin MI);
  **CPU spearman/MI → SIMD** (the MI-histogram approximation backend: vectorize binning with
  FMA+convert → SoA partial-histograms per lane-group (scatter→vectorized merge) → vectorized
  log-sum; fixed bins 12/24/48/96 are the enabler).
- **WHY:** same `metric_names` gives different results/errors per backend today — a correctness
  and UX hazard; CPU MI/spearman are scalar (the queued SIMD frontier).
- **WHERE:** `src/cuda/{kernels.cu,launcher.cu}`, `src/rocm/{kernels.hip,launcher.hip}`, `crates/gafime-cpu/src/
  {dispatch.rs,kernels}`.
- **LOWERING/UPSTREAM:** histogram merge benefits from **LLVM 22 atomic vector loads** (lock-free
  partial-histogram combine) and **4-accumulator ILP**; `array_windows` for the bin sweep; on
  future Intel, **APX CCMP** makes bin-edge comparisons branchless.
- **DONE-WHEN:** every configured metric runs (or errors clearly) consistently across CPU/CUDA/ROCm;
  CPU fixed-bin MI uses the SIMD-fed histogram path with parity. Further Spearman/rank SIMD is
  a performance-only follow-up, not a v1 parity blocker.

### P-C — decision_path family end-to-end
- **WHAT:** depth-k recursion + residual boosting on `best_variance_split` (core landed in
  `crates/gafime-cpu/src/decision_path.rs`) → conjunction-path membership features → wire like
  time_series (expand→continuous) → CPU+GPU; flip the family descriptor in
  `crates/gafime-orchestrator/src/family.rs` (today `cpu_kernel:false`).
- **WHY:** named GAFIME product family, contract-only today; the GBDT-method features (no sklearn).
- **WHERE:** `crates/gafime-cpu/src/decision_path.rs`, `family.rs`, `crates/gafime-py/src/lib.rs`
  (an `analyze_decision_path` entry mirroring `analyze_time_series`), `python/gafime/v1_adapter.py`.
- **LOWERING/UPSTREAM:** the hard-AND indicator is the **branch-divergence** op that motivates
  P-H; on CPU make it **branchless** (CCMP-style select / `_mm*_cmp_ps` mask) to avoid
  misprediction; `array_windows` for the bin-threshold sweep.
- **DONE-WHEN:** decision_path runnable + parity-tested on CPU (and GPU once built); descriptor flipped.

### P-D — Legacy-surface parity (compile/export, telemetry, sklearn, power-user, residency, whole-sweep graph)
- **WHAT:** (a) **export through the compile flag** end-to-end (remove the `V1UnsupportedError`;
  expose the landed zero-copy DLPack / `__arrow_c_array__` handles via `CompiledGafime`; device
  DLPack when the GPU buffer ptr is available); (b) **telemetry** — port the
  `gafime.telemetry.v0.5.0-rc1` schema + `tools/telemetry.py` (`new_record`/`span`/`write_run`) so
  benches emit machine-readable artifacts; (c) **sklearn transformer** (`python/gafime/sklearn.py`);
  (d) **power-user wide planning** (`max_feature_candidate=-1`, saturating counts, caps/warnings)
  verified in `plan/combos.rs`; (e) **resident-session reuse** through the compile artifact +
  **VRAM budget** enforcement in `schedule`; (f) **whole-sweep graph** (one graph over the
  multi-arity sweep, the latent v0.5 win) — converges with P-A's WHILE-node.
- **WHY:** these are legacy surfaces real users depend on; parity is incomplete without them.
- **WHERE:** `python/gafime/{compile,reporting,sklearn.py}`, `tools/telemetry.py`,
  `crates/gafime-orchestrator/src/{plan,schedule}`.
- **LOWERING/UPSTREAM:** export rides arrow-rs 54 FFI + DLPack (no numpy); ingest can ride
  **Polars-GPU (cuDF)** for on-device feature matrices per platform; `allow_threads` keeps it
  GIL-free (free-threaded remains a non-lever).
- **DONE-WHEN:** `CompileFlags(export=True)` returns a usable zero-copy handle; telemetry artifacts
  written; sklearn `fit/transform` parity; power-user plan describes 10M-candidate spaces in
  bounded RAM.

### P-E — ROCm parity
- **WHAT:** bring HIP to CUDA level: **MI, top-k, graph** on gfx1150 (today pearson/r2 continuous
  only; rejects `top_k`, `GRAPH_UNSUPPORTED`).
- **WHY:** ROCm is a thin subset; the dev box has a live gfx1150.
- **WHERE:** `src/rocm/{kernels.hip,launcher.hip}`.
- **LOWERING/UPSTREAM:** **ROCm 7.2 optimized HIP graph dispatch** (doorbell-ring, memset-node,
  async-handler lock removal) lowers per-launch cost; mirror the CUDA host; gfx1150 keeps
  **device-copy** mode (UMA host-mapped inputs unsafe — the landed `7c169ac` fix); track HIP
  conditional-node availability to port P-A's on-device loop (verify in ROCm 7.2; else host-relaunch).
- **DONE-WHEN:** MI/top-k/graph parity on gfx1150 within fp tolerance.

### P-F — GPU validation (close the honest gap)
- **WHAT:** build the CUDA v1 lib (`cmake src/cuda` → `GAFIME_CUDA_V1_LIB`) and validate on the
  **live RTX 4060 (sm_89)**: continuous + time_series + decision_path + the **P-A WHILE-node
  permutation graph**; parity vs CPU within fp tolerance; confirm **one launch, zero host sync per
  permutation** and a real speedup vs the 0.715× baseline.
- **WHY:** until this runs, all GPU claims are "wired", not "proven."
- **WHERE:** `src/cuda/*`, `crates/gafime-gpu-sys`.
- **LOWERING/UPSTREAM:** apply the §2 nvcc flags; profile with `-Xptxas -v` + line-info; verify
  conditional-node availability on sm_89; measure green-context overlap of rank vs score.
- **DONE-WHEN:** built lib loads, golden parity holds, permutation graph beats the host-loop baseline.

### P-G — Metal real path (deferred — no Apple HW)
- **WHAT:** Metal 4 command buffers + **residency sets** + argument tables (encode-once/replay-many)
  in `src/metal/launcher.mm`; `MTLTensor` option for the kernels.
- **WHY:** the 4th backend; honest stub until hardware exists.
- **DONE-WHEN:** capability-reporting stub remains until Apple HW; design documented against Metal 4.

### P-H — FRONTIER 1: RT-core GBDT borders  ⭐ first-class (Goal 2)
- **WHAT:** evaluate `decision_path` conjunction borders as **point-in-AABB ray queries on NVIDIA
  RT cores** (OptiX), to kill the warp **branch divergence** that collapses GPU split-evaluation. A
  conjunction `f_a>t1 ∧ f_a≤t2 ∧ f_b>t3` *is* an axis-aligned box; leaves partition space into
  disjoint AABBs; "which rows fall in this box" = point-in-AABB = fixed-function RT hardware,
  **branchless**, on silicon that sits **idle** while MI kernels saturate the SMs.
- **WHY it fits:** `decision_path_max_depth` is 2–3 → boxes are ≤3D (matches RT cores' 3D world);
  the divergent op is concrete (`evaluate_decision_path_candidate` hard-AND indicator); high query
  volume (rows × boxes) amortizes BVH build. Accelerates **evaluation/materialization**, not
  split-finding (the prefix-sum sweep stays on CUDA cores).
- **CAVEATS (measure before believing — the unmeasured-perf sin is forbidden):** each path is a
  *different* 2–3D subspace → not one clean BVH (per-feature-triple structures / projection = the
  research problem); f32 border watertightness (`≤` vs `>` for a point on the border) must match the
  CPU/CUDA indicator **exactly**; OptiX + custom AABB-intersection program + per-row ray = real
  engineering.
- **WHERE:** new `src/cuda/` OptiX path; gated behind a parity check vs `decision_path.rs`.
- **LOWERING/UPSTREAM:** **CUDA 13.3 green contexts** run RT-core border eval on one SM partition
  **concurrently** with the MI/continuous kernels on another (the whole point — reclaim idle RT
  silicon); CUDA 13.3 OptiX toolchain. **Spike → measure on the 4060 → border-parity gate.**
- **DONE-WHEN:** depth-2 candidate set → leaf AABBs → point-in-box on the 4060 via OptiX,
  **RT-core vs SM indicator benchmarked with a border-parity check**. Ship only if borders match
  and it's faster; else keep the SM path and record the measurement.

### P-I — FRONTIER 2: Hardware autotune — hardening + aggression  ⭐ first-class (Goal 2)
- **WHAT:** make autotuning (a) **more robust** — reliable across archs (NVIDIA/AMD/Intel), the
  dual-GPU host, and varying/low VRAM; never regress; degrade gracefully — and (b) **more
  aggressive** — wider search, **per-architecture** defaults (not fixed/hand-set), profile-guided,
  **cached per device**. Surface: launch params (block size / `__launch_bounds__` / minBlocks),
  occupancy vs register-spill, batch/chunk sizing, metric-cache/residency knobs, backend selection.
- **WHY:** today's tuning is conservative and partly hand-set (e.g. `__launch_bounds__(256,5)`);
  per-arch autotune is the last correctness-safe performance lever once robustness is locked.
- **WHERE:** `gpu/<vendor>/tune.cpp`, `crates/gafime-orchestrator/src/cache.rs` (reads the device
  autotune cache to shape the plan), `crates/gafime-orchestrator/src/plan/shapes.rs`.
- **LOWERING/UPSTREAM:** adopt **CUDA 13.3 CompileIQ** (evolutionary per-kernel compiler-config
  search) into the harness instead of only hand-set bounds; **green contexts** for partition-aware
  tuning + dual-GPU; persistent per-device cache keyed on `gpu_device_info`. CPU side: autotune the
  ILP accumulator count + ISA rung selection.
- **GUARDRAILS:** measure every tuning claim on the actually-installed backend; the 14GB/24-core
  host OOMs easily (never `n_jobs=-1`); live HW = RTX 4060 (sm_89) + AMD gfx1150.
- **DONE-WHEN:** per-arch cached configs beat the hand-set defaults on the 4060 + gfx1150 with no
  regression on low-VRAM; tuning survives both GPUs.

### Cross-cutting — pipeline overlap + memory/serialization (interleave; don't block P-A–P-C)
- **(a) pipeline-stage overlap** — overlap ingest/score/write/materialize to attack the Amdahl cap
  on the landed ~9.5× rayon speedup (`orchestrator/schedule`'s job); green contexts on GPU.
- **Memory/serialization** — Arrow ingest double-transpose → column-native ingest; result-table
  full pre-alloc when `top_k==0`; rayon `collect`→`Vec<Vec<f32>>` → flat/disjoint writes;
  per-record string materialization in `report.py`.

---

## 4. Sequencing

1. **P-A** (permutation/stability) — the legacy-parity blocker; highest value. **Start here.**
2. **P-B** (metric × backend completeness incl. MI-histogram SIMD).
3. **P-C** (decision_path end-to-end).
4. **P-D** (legacy-surface parity: export/telemetry/sklearn/power-user/residency/whole-sweep).
5. **P-E / P-F** (ROCm parity + GPU validation on the 4060).
6. **P-H / P-I** (RT-core borders + autotune) — first-class frontier; **after** correctness parity
   so they optimize a correct engine, each spike-measured-gated, never merged unmeasured.
7. **P-G** (Metal) — deferred until Apple HW.
- Cross-cut perf interleaves; every phase ships as a tested, golden-parity-gated increment.

## 5. Done-when (cutover vs legacy)

- Permutation p-values + stability populated and gating `decision` on all live backends.
- Every configured metric runs (or errors clearly) consistently across CPU/CUDA/ROCm.
- decision_path + time_series both runnable + validated on CPU and GPU.
- Compile/export, telemetry, sklearn, power-user planning at legacy parity.
- GPU path built + parity-validated on the 4060, permutation graph beating the host-loop baseline.
- RT-core borders + per-arch autotune measured on real HW and gated (ship-if-faster, else recorded).

## 6. Appendix — upstream sources (verified Q2 2026)

- CUDA conditional graph nodes (WHILE/IF): <https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/> · guide <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- CUDA Toolkit 13.3 (green contexts, recapture-to-graph, CompileIQ, CUDA Tile, CCCL 3.3): release notes <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html> · blog <https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates/>
- LLVM 22.1 (APX/AVX10.2, constexpr intrinsics, atomic vector loads): <https://releases.llvm.org/22.1.0/docs/ReleaseNotes.html> · Clang <https://releases.llvm.org/22.1.0/tools/clang/docs/ReleaseNotes.html>
- ROCm 7.2 (HIP graph dispatch optimizations): <https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html>
- Rust 1.94 (AVX-512/NEON fp16 intrinsics, `array_windows`): <https://www.phoronix.com/news/Rust-1.94-Released> · <https://doc.rust-lang.org/beta/releases.html>
- Metal 4 (residency sets, command buffers, MTLTensor): <https://developer.apple.com/videos/play/wwdc2025/205/>
- Rust SIMD intrinsics (stable `core::arch`): x86 <https://doc.rust-lang.org/core/arch/x86_64/> · NEON <https://doc.rust-lang.org/core/arch/aarch64/>
- Polars Arrow C Data Interface / Polars-GPU (cuDF): <https://docs.pola.rs/user-guide/misc/arrow/> · <https://pola.rs/posts/gpu-engine-release/>
- PyO3 / maturin: <https://github.com/PyO3/pyo3> · <https://github.com/PyO3/maturin>
