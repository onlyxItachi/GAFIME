# CUDA RT-Core Decision-Path Spike

## Objective

The spike tests whether shallow `decision_path` candidates from GBDT-style split borders can be evaluated faster on NVIDIA RT cores than on normal CUDA SMs.

A decision-path candidate is a hard conjunction such as:

```text
f0 > 0.5 AND f1 <= 1.25
```

For depth 2-3 paths, that conjunction maps naturally to a small axis-aligned box. The research question is whether point-in-box membership over many rows and many candidate boxes can move from divergent SM branches to RTX fixed-function traversal.

## Current Checkpoint

This branch connects the CUDA RT/decision-path membership path end to end behind
the existing GPU C ABI:

- `GafimeDecisionPathTerm` and `GafimeDecisionPathBatch` in the GPU C ABI.
- Optional `gafime_gpu_decision_path_membership` symbol, implemented by CUDA only.
- Optional `gafime_gpu_decision_path_score` symbol, implemented by CUDA only for
  compact Pearson/R2 path scoring.
- CUDA `rt_kernels.cu` owns `decision_path_membership_kernel` over the resident feature-major matrix.
- CUDA `rt_kernels.cu` also owns the OptiX device programs and the point-packing kernel used by RT traversal.
- CUDA `rt_launcher.cu` owns RT membership validation, finite box planning, custom-AABB and bounded-2D triangle geometry preparation, cached OptiX GAS/workspace, exact SM fallback, and copy-back.
- Rust optional loader/wrapper in `gafime-gpu-sys`.
- C++ ABI smoke coverage and Rust CPU-parity coverage.

The implementation preserves Rust ownership:

- Rust discovers paths, validates config, plans features, selects backends, and schedules work.
- CUDA receives compact validated path terms and computes either path-major
  membership or compact score rows.
- Missing support is explicit through optional symbols. Cross-backend fallback
  remains a Rust decision; the per-call require-RT policy can also forbid CUDA
  SM fallback.
- Generic CUDA metric files remain separate: `kernels.cu` / `launcher.cu` must not absorb RT-specific execution logic beyond the exported C ABI bridge in `launcher.cu`.

## Runtime RT Path

The default CUDA payload builds the exact SM membership comparator and fully
disables the OptiX RT-core path. CUDA RT-core support is selected at build time:

- `-DGAFIME_CUDA_RT_BUILD_MODE=off` builds only `libgafime_cuda_v1` without
  OptiX PTX or CUDA driver linkage. This is the default distribution/local
  build mode.
- `-DGAFIME_CUDA_RT_BUILD_MODE=on` builds `libgafime_cuda_v1` with OptiX PTX
  embedded. The legacy `-DGAFIME_CUDA_ENABLE_OPTIX_RT=ON` maps to this mode.
- `-DGAFIME_CUDA_RT_BUILD_MODE=both` builds a non-RT `libgafime_cuda_v1` plus
  an RT-capable `libgafime_cuda_v1_rt` sibling in the same build tree.

The RT-capable variant compiles OptiX PTX from `src/cuda/rt_kernels.cu`, embeds
that PTX in the CUDA payload, and lets `gafime_gpu_decision_path_membership`
choose the RT path when the batch is representable as finite 1D/2D/3D boxes on
RTX-class hardware.

Release packaging gives these variants distinct identities. The standard
RT-off publishing lane builds distribution `gafime-cuda`, package
`gafime_cuda`, with its own native library filename. The optional RT lane builds
distribution `gafime-cuda-rt`, package `gafime_cuda_rt`, with a distinct RT
library filename. The RT distribution is produced only by a separately selected
GitHub Actions artifact job; this document does not claim that it is available
from PyPI. Automatic discovery accepts either variant in isolation but rejects
a dual installation unless `GAFIME_CUDA_V1_LIB` explicitly selects the library.
The standard 11-artifact release bundle and every PyPI publishing job exclude
the RT payload. Exact artifact download and clean-environment installation
commands are in `docs/rt-gbdt-paper-repro.md`.

The RT path has two geometry modes:

- bounded 2D boxes use two OptiX triangles per path, so traversal can use the
  fixed-function triangle path; any-hit still rechecks the exact GAFIME
  `>`/`<=` box predicate before writing membership. Points on the shared
  rectangle diagonal are assigned to one triangle only in both single-GAS and
  instanced grouped modes, so direct score counts cannot double-count boundary
  hits.
- custom AABBs remain the exact fallback for 1D/3D or open-bound batches.

`GAFIME_CUDA_DECISION_PATH_RT_GEOMETRY=aabb` forces the custom-AABB path for
profiling and parity checks. This and the other `GAFIME_CUDA_DECISION_PATH_RT*`
selectors are process-global experimental environment controls, not per-call
public API settings.

For performance work, `gafime_gpu_decision_path_score` is the preferred path
over `gafime_gpu_decision_path_membership`. The default score path writes an
idempotent device bitset from OptiX traversal, then reduces that bitset with the
resident target into compact Pearson/R2 result rows. This keeps the public
result at `path_count * metric_count` floats instead of copying
`path_count * rows` float membership values to the host, and it reduces the
temporary device mask by 32x relative to the old `f32` membership matrix.

`GAFIME_CUDA_DECISION_PATH_RT_SCORE=direct` enables an experimental direct score
mode. In that mode OptiX any-hit accumulates duplicate-safe per-path inside
counts and target sums directly, then a compact CUDA reduction combines those
stats with target-wide stats. It removes the temporary score bitset and the
bitset reduction pass, but it uses `float` `atomicAdd` during traversal. That
means direct mode is numerically equivalent within the documented spike
tolerance (`1e-4`) but is not bit-stable. The bitset score path remains the
default because it preserves tighter deterministic parity.

`GAFIME_CUDA_DECISION_PATH_RT_SCORE=firsthit` is a stricter direct-score mode for
tree-leaf-like batches where CUDA can prove that boxes inside every RT group are
non-overlapping. In that mode the any-hit program accepts the first exact
in-box hit and terminates the ray, avoiding the triangle-diagonal ownership
filter used by general overlapping direct mode. If a requested first-hit batch
is not non-overlapping, CUDA returns unsupported instead of falling back or
changing semantics.

The RT path is used only when correctness can stay exact:

- every value in the entire uploaded feature matrix is finite, including
  unreferenced columns, because upload records one matrix-wide finiteness bit,
- all thresholds are finite,
- a membership batch uses at most three unique feature axes; compact score
  batches may use several internal groups with at most three axes each,
- the CUDA device is Turing or newer,
- OptiX runtime initialization and pipeline creation succeed.

For compact scoring, CUDA can split a mixed-axis score batch into several RT
groups when the whole batch has more than three unique axes but each group is
still representable as a finite <=3D box set. The grouping uses first-fit
compatible packing, so non-contiguous paths that share a feature pair can run in
one larger RT batch while result rows are restored to original path order. This
avoids whole-batch SM fallback for common GBDT workloads where different paths
use different feature pairs. The grouping is CUDA-internal: Rust still owns path
discovery and scheduling. In direct score mode, grouped execution computes
target-wide stats once and reuses that device buffer across the RT groups, so
grouping does not rescan the target for every feature-axis group. Grouped score
execution scatters each internal RT group's compact metric vector into one
final device buffer keyed by original path id. The grouped path uploads one
flattened original-path map for the whole grouped call, then copies the compact
metric buffer once and writes the public result table after original path order
is restored; it does not build temporary per-group result rows or copy metric
vectors to host per group.

For direct-score grouped batches where every internal group is a finite bounded
2D triangle set, CUDA now batches those groups through one instanced OptiX
launch. Each feature-pair group builds its own triangle GAS, the launcher wraps
the group GASes in one IAS, and raygen launches `(rows x group_count)` rays with
group-local prepacked points. Those instanced 2D points are packed as `x,y`
pairs rather than `x,y,z` triples, so warmed traversal does not move an unused
coordinate through the point buffer. Direct-mode grouping keeps 2D paths keyed
by exact feature pair, so overlapping pairs such as `(f0,f1)` and `(f1,f2)` do
not widen into a 3D custom group that would bypass the instanced triangle fast
path. Any-hit uses the OptiX instance id plus a compact group-path offset table
to restore the flattened path id, then a fused direct stats score kernel writes
public-order compact metrics without launching a separate scatter kernel. This
is the preferred many-group RT path because it
gives OptiX a larger launch while preserving Rust-owned scheduling and without
moving feature planning into CUDA. When the grouped region geometry signature is
unchanged, the CUDA launcher reuses the resident host grouped plan, group GASes,
and IAS. It also tracks the resident feature upload generation and can reuse the
grouped prepacked points across repeated score calls when the feature matrix and
grouped geometry are unchanged.
Target-wide statistics are cached separately by target generation and invalidated
by `gafime_gpu_matrix_upload` or `gafime_gpu_matrix_update_target`. Target-only
updates do not invalidate feature-derived packed points, but they do force fresh
target statistics before traversal. Warm direct-score calls with unchanged
features and target therefore clear compact direct statistics, launch traversal,
and write compact metrics through persistent grouped scratch buffers without
rebuilding host grouping, rebuilding geometry, repacking points, launching a
separate scatter pass, reallocating scratch, recopying the flattened scatter map,
materializing a temporary host metric vector on exact metric-stride results, or
rescanning the target. The flattened original-path scatter map and host grouped
plan are cached by their own signatures so changed public row order or path
contents cannot reuse stale mapping.

Otherwise CUDA uses the exact SM comparator inside the same backend. Rust's
per-call `DecisionPathRtPolicy::RequireRt` maps to
`GAFIME_DECISION_PATH_FLAG_REQUIRE_RT` in either decision-path batch and turns
an unrepresentable or unavailable RT path into an explicit unsupported status
instead of allowing the SM path. This flag controls fallback; it does not select
bitset, direct, first-hit, or AABB geometry. Those modes are selected by
process-global experimental environment variables read inside the CUDA payload,
so they must not be described as thread-local or per-call policy. For test runs,
`GAFIME_CUDA_DECISION_PATH_RT=off` forces SM execution, and
`GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP=1` in the C++ smoke sets the per-call
RT-required ABI flag.

## Standalone OptiX Smoke

`tests/gpu/cuda_rt_decision_path_optix_smoke.cu` remains a standalone GPU smoke
for the RT-core hypothesis and for quick custom-primitive debugging outside the
shared payload.

Build shape:

```bash
: "${CUDA_HOME:=/usr/local/cuda}"
: "${OPTIX_INCLUDE_DIR:?set this to the OptiX SDK include directory}"

"$CUDA_HOME/bin/nvcc" --std=c++20 \
  -I"$OPTIX_INCLUDE_DIR" \
  -DGAFIME_OPTIX_DEVICE --ptx tests/gpu/cuda_rt_decision_path_optix_smoke.cu \
  -o /tmp/gafime_rt_decision_path_optix.ptx

"$CUDA_HOME/bin/nvcc" --std=c++20 -O3 \
  -I"$OPTIX_INCLUDE_DIR" \
  tests/gpu/cuda_rt_decision_path_optix_smoke.cu -lcuda \
  -o /tmp/gafime_rt_decision_path_optix_smoke

/tmp/gafime_rt_decision_path_optix_smoke /tmp/gafime_rt_decision_path_optix.ptx
```

The smoke builds custom OptiX AABBs for two depth-1/2 decision-path boxes, launches row-points through OptiX, and compares path-major membership against a plain CUDA SM kernel. It uses an exact custom intersection check so lower-open `>` and upper-closed `<=` semantics remain under GAFIME control instead of relying on default closed AABB behavior.

## Numerical Contract

Membership output is path-major `f32` with one column per path:

```text
path0: row0 row1 row2 ...
path1: row0 row1 row2 ...
```

CUDA must match `gafime_cpu::decision_path::path_membership`:

- `LE` means `x <= threshold`.
- `GT` means `x > threshold`.
- If a concrete predicate fails, output is `0.0`.
- If all concrete predicates hold but a needed feature is `NaN`, output is `NaN`.
- Otherwise output is `1.0`.

The RT path itself requires the whole uploaded feature matrix to be finite, so
the NaN membership rule above is exercised by the exact SM comparator, not by
OptiX. Compact Pearson/R2 scoring separately excludes every row whose target is
non-finite from `n`, `sum(y)`, `sum(y^2)`, the path-inside count, and
`sum(inside * y)`. Direct traversal accumulates the inside count as `uint32_t`,
but final score math converts it to `float`, and the valid-target count is also a
`float`. Integer counts are therefore guaranteed exactly representable only
through `2^24`; larger counts remain under the `UINT32_MAX` RT row bound but do
not have an exact-f32-count guarantee. The reported 262,144-row cases are below
that threshold.

## Remaining RT Work

The runtime path proves low-level connectivity through the public CUDA C ABI.
The public Python decision-path adapter still materializes membership and does
not invoke compact RT scoring, so end-to-end product integration remains open:

- extend compact device-side scoring beyond Pearson/R2 only after MI/Spearman
  parity is proven,
- promote duplicate-safe direct traversal statistics only after the documented
  atomic-FP tolerance is accepted for default score behavior,
- extend the instanced grouped RT launch beyond finite bounded 2D triangle
  direct-score groups only after parity and profiling prove the new shape,
- extend membership materialization with the same mixed-axis grouping if a
  future caller truly needs path-major membership output.

## Scale Checkpoint

`tests/gpu/cuda_rt_membership_scale_bench.cpp` compares a finite 2D decision-path
box workload across:

- CPU AVX512 membership materialization,
- CUDA RT membership through `gafime_gpu_decision_path_membership` with `GAFIME_DECISION_PATH_FLAG_REQUIRE_RT`,
- CUDA SM membership through the same ABI with `GAFIME_CUDA_DECISION_PATH_RT=off`.
- compact score-only mode with `--score-only`, which skips path-major
  membership allocation and validates `gafime_gpu_decision_path_score` directly
  against a streaming CPU reference. Score-only benchmark runs default
  `GAFIME_CUDA_DECISION_PATH_RT_SCORE=direct` inside the process so they measure
  the documented direct RT-core scoring path; pass `--bitset-score` to profile
  the tighter-parity bitset scorer instead, or `--direct-score` to make the
  direct-score selection explicit when an environment override is present. Pass
  `--repeats=N` to record the first call separately and report an observed warm
  p50 from the remaining resident calls. The benchmark checks and reports the
  worst parity error observed across the cold call and every warm repetition.
- RT throughput-only score mode with `--score-only --throughput-only --rt-only`,
  which skips the full CPU score reference and SM bitset path for very large
  candidate counts. Partitioned `--firsthit-score` shapes use an exact
  `O(rows * groups + paths)` partition oracle and remain correctness-checked;
  other throughput-only shapes print `score parity skipped`.
- mixed-axis compact score mode with `--score-only --mixed-axes`, which
  alternates `(f0, f1)` and `(f2, f3)` path regions so the first-fit RT grouping
  path is measured at scale without materializing membership output.
- mixed-axis stress mode with `--score-only --mixed-axis-pairs=N`, which creates
  `N` disjoint feature pairs to measure many internal RT groups and the grouped
  scatter/feed path.
- overlapping-axis stress mode with `--score-only --overlap-axis-pairs=N`,
  which creates sliding pairs `(f0,f1)`, `(f1,f2)`, ... to prove direct-mode
  grouping preserves exact 2D triangle groups instead of widening overlapping
  pairs into a 3D custom group.
- partitioned-grid score mode with `--score-only --partitioned-grid`, which
  uses non-overlapping boxes within each feature-pair group to model tree-leaf
  partitions separately from random overlapping-box hit pressure. Use
  `--bitset-score` for the default bitset parity path, `--firsthit-score` for
  validated first-hit direct parity, or `--throughput-only` for large direct RT
  throughput runs.

On the local Ryzen AI 9 HX 370 / RTX 4060 Laptop run, all tested cases matched
exactly. After switching bounded 2D boxes to OptiX triangle geometry and caching
the RT workspace/GAS, RT membership is in the same performance band as the CUDA
SM comparator on large cases, while compact RT scoring is the clear performance
path because it avoids host membership materialization:

```text
rows=1,048,576 paths=512 evals=536.871M output=2.00 GiB
resident upload   5.857 ms
cpu_avx512      111.218 ms  4.827 G eval/s  17.983 GiB/s output
gpu_rt_abi      194.236 ms  2.764 G eval/s  10.297 GiB/s output
gpu_sm_abi      189.348 ms  2.835 G eval/s  10.563 GiB/s output
parity          rt_mismatches=0 sm_mismatches=0
```

Forcing the old custom-AABB RT path on the same 1,048,576 x 512 workload measured
208.535 ms / 2.574 G eval/s, so the triangle path is the current performance
route for bounded 2D decision regions.

The compact score ABI removes the host membership copy and reduces on device:

```text
rows=1,048,576 paths=512 evals=536.871M output=2.00 GiB membership-equivalent
gpu_rt_score    9.118 ms  58.881 G eval/s
gpu_sm_score   26.768 ms  20.056 G eval/s
score parity   rt_max_abs=7.45058e-08 sm_max_abs=7.45058e-08
```

On the 262,144 x 512 case, compact RT score measured 4.182 ms /
32.091 G eval/s versus compact SM score at 6.582 ms / 20.392 G eval/s.
The temporary score mask for 1,048,576 x 512 is 64 MiB as a bitset, versus
512 MiB as a byte mask and 2.00 GiB as an `f32` membership matrix.

The experimental direct traversal-stat score mode is faster because traversal
writes only per-path counts and sums:

```text
rows=1,048,576 paths=512 evals=536.871M
gpu_rt_score         4.512 ms  118.978 G eval/s
gpu_rt_score_timing first_ms=64.487500 warm_p50_ms=4.512349
  warm_best_ms=4.510299 warm_samples=5
gpu_sm_score        28.562 ms   18.797 G eval/s
score parity        rt_max_abs=4.45545e-06 sm_max_abs=2.08616e-07
```

Those numbers show resident direct-score behavior, not RT-core saturation or a
default-release decision. The direct mode's `float` atomics explain the observed
few-e-6 drift, so it remains opt-in under the numerical policy.

`--score-only` lets the benchmark drive larger candidate-region counts without
allocating the membership-equivalent output. Fresh compact direct-score runs on
the same RTX 4060 Laptop payload:

```text
rows=1,048,576 paths=2,048 evals=2.147B membership-equivalent output=8.00 GiB
cpu_score_ref      8881.864 ms    0.242 G eval/s
gpu_rt_score         13.913 ms  154.355 G eval/s
gpu_sm_score        101.690 ms   21.118 G eval/s
score parity        rt_max_abs=1.15633e-05 sm_max_abs=7.45058e-08

rows=1,048,576 paths=4,096 evals=4.295B membership-equivalent output=16.00 GiB
cpu_score_ref     18260.906 ms    0.235 G eval/s
gpu_rt_score         28.503 ms  150.686 G eval/s
gpu_sm_score        174.462 ms   24.618 G eval/s
score parity        rt_max_abs=1.15931e-05 sm_max_abs=7.45058e-08

rows=262,144 paths=8,192 evals=2.147B membership-equivalent output=8.00 GiB
cpu_score_ref      8913.310 ms    0.241 G eval/s
gpu_rt_score         12.934 ms  166.030 G eval/s
gpu_sm_score        119.525 ms   17.967 G eval/s
score parity        rt_max_abs=3.20524e-05 sm_max_abs=1.3411e-07

rows=262,144 paths=8,192 mixed-axis grouped evals=2.147B output=8.00 GiB
cpu_score_ref      8923.204 ms    0.241 G eval/s
gpu_rt_score         15.288 ms  140.469 G eval/s
gpu_sm_score         90.822 ms   23.645 G eval/s
score parity        rt_max_abs=6.79903e-05 sm_max_abs=5.96046e-07

rows=262,144 paths=8,192 mixed-axis stress axis_pairs=8 evals=2.147B output=8.00 GiB
cpu_score_ref      8954.569 ms    0.240 G eval/s
gpu_rt_score         13.399 ms  160.273 G eval/s
gpu_sm_score        103.068 ms   20.836 G eval/s
score parity        rt_max_abs=6.91973e-05 sm_max_abs=5.06639e-07
```

After switching the direct-score many-group path to one instanced OptiX launch
for bounded 2D groups and caching unchanged group GAS/IAS geometry, the same
RTX 4060 Laptop payload measured:

```text
rows=65,536 paths=1,024 mixed-axis stress axis_pairs=8 evals=67.109M output=256.00 MiB
cpu_score_ref       277.060 ms    0.242 G eval/s
gpu_rt_score          0.485 ms  138.288 G eval/s
gpu_sm_score          2.332 ms   28.773 G eval/s
score parity        rt_max_abs=1.44262e-06 sm_max_abs=4.47035e-07

rows=262,144 paths=8,192 mixed-axis stress axis_pairs=8 evals=2.147B output=8.00 GiB
cpu_score_ref      8766.365 ms    0.245 G eval/s
gpu_rt_score         12.445 ms  172.555 G eval/s
gpu_sm_score        107.224 ms   20.028 G eval/s
score parity        rt_max_abs=4.09828e-06 sm_max_abs=5.06639e-07
```

After adding the host grouped-plan cache, packed-point cache, target-stat
generation cache, scatter-map cache, fused score/scatter kernel, direct
result-buffer metric copy, one stream-ordered final result copy, and persistent
grouped scratch buffers, the large mixed-axis scale case measured 11.747-12.956
ms / 165.757-182.813 G eval/s across repeated RTX 4060 Laptop runs. Small cases
remain launch-overhead and clock-noise dominated, so the scale run is the
relevant throughput signal. The safety invariants are covered by CUDA
regression tests that update only the target and reorder grouped public path
rows while reusing unchanged feature-derived points.

Throughput-only RT runs can now push candidate counts beyond the inline CPU
reference limit. These runs are not correctness evidence; the benchmark labels
them with `score parity skipped` and they must be paired with the parity-covered
scale cases above:

```text
rows=65,536 paths=1,048,576 overlap-axis stress axis_pairs=8 evals=68.719B
gpu_rt_score        697.361 ms   98.542 G eval/s
score parity        skipped (--throughput-only)

rows=262,144 paths=1,048,576 overlap-axis stress axis_pairs=8 evals=274.878B
gpu_rt_score       2888.234 ms   95.172 G eval/s
score parity        skipped (--throughput-only)
```

The million-path cases show that adding more rays does not recover the
`8K`-path throughput band; very large region counts shift the next bottleneck to
the triangle acceleration structure and direct per-path statistic pressure.
That is useful for the next tuning target, but it does not replace the smaller
parity-covered validation runs.

For tree-leaf-like partitions, the same benchmark can generate non-overlapping
2D grid boxes per feature-pair group. In this shape each sample hits at most one
region per group, matching the important GBDT leaf invariant and avoiding the
random-overlap atomic explosion. The bitset scorer remains the parity reference:

```text
rows=262,144 paths=8,192 partitioned-grid overlap-axis axis_pairs=8
gpu_rt_score bitset 114.599 ms   18.739 G eval/s
gpu_sm_score         75.912 ms   28.289 G eval/s
score parity         rt_max_abs=3.72529e-08 sm_max_abs=3.72529e-08
```

Validated first-hit direct RT score removes duplicate triangle-hit ambiguity for
non-overlapping groups by terminating after the first exact in-box hit:

```text
rows=262,144 paths=8,192 partitioned-grid overlap-axis axis_pairs=8
gpu_rt_score          0.886 ms  2423.742 G eval/s
gpu_rt_score_timing first_ms=56.109044 warm_p50_ms=0.886020
  warm_best_ms=0.882500 warm_samples=5
firsthit work      groups=8 paths_per_group=1024 rays=2097152
  ray_rate=2.367 G ray/s hits=2097152 hit_rate=1.000000
score oracle      rt_max_abs=1.19209e-07
```

The literal `ray_rate` field is `rows * groups / full timed score call`. The
paper therefore names it the **end-to-end effective ray rate**, not an actual or
isolated RT-core launch rate.

First-hit mode is fail-closed. If CUDA cannot prove every requested 2D RT group
is finite, bounded, and non-overlapping, the score ABI returns unsupported
instead of silently routing the opt-in request to the SM comparator.

With that invariant, the partitioned shape shows what the RT path can do when
the region set gives the traversal hardware tree-like work instead of dense
overlapping hit lists:

```text
rows=262,144 paths=1,048,576 partitioned-grid overlap-axis axis_pairs=8
gpu_rt_score         20.180 ms  13621.251 G eval/s
gpu_rt_score_timing first_ms=467.746184 warm_p50_ms=20.180078
  warm_best_ms=19.951857 warm_samples=5
firsthit work      groups=8 paths_per_group=131072 rays=2097152
  ray_rate=0.104 G ray/s hits=2085890 hit_rate=0.994630
score oracle      rt_max_abs=5.58794e-09
```

`rows * paths / time` is an all-pairs membership-equivalent rate, not an executed
comparison count. First-hit launches `rows * groups` rays and lets the BVH prune
the path set. The result therefore demonstrates a combined algorithmic and
hardware benefit, not isolated RT-core speedup. A structure-aware CUDA baseline
is still required for attribution.

Release measurement now includes
`tests/release_measure/perf_05_cuda_rt_firsthit_scale.py`. It is skipped unless
`GAFIME_CUDA_RT_SCALE_BENCH` and `GAFIME_CUDA_V1_LIB` are provided, but when run
it executes the partitioned first-hit case, parses cold/warm timing and reported
ray work, enforces a minimum `GAFIME_CUDA_RT_FIRSTHIT_MIN_GEVALS` warm-p50
throughput, and checks `rt_max_abs` against the approved tolerance.

Nsight Compute 2026.2.1 full replay on a `65,536 x 8,192` first-hit case was
digested with PerfDigest. Replay timing is not used as benchmark latency. The
hot resident units were:

```text
grouped point packing       25.088 us
optixLaunch                196.992 us
score scatter/finalization   4.480 us
```

The `optixLaunch` digest reports 24.932% compute-pipe peak, 10.878% DRAM peak,
54.223% achieved occupancy, 53.408% L1 hit, 96.864% L2 hit, and 72
registers/thread. These counters support a cache-resident traversal bottleneck
relative to packing and scatter. They do not expose a direct RT-core saturation
percentage, so branch or SM counters must not be used as a substitute claim.
The exact report is checked in at
`docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep`, SHA-256
`5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331`.
The captured timing transcript and implementation-source manifest are also in
`docs/evidence/`; full commands and the cold/warm methodology are in
`docs/rt-gbdt-paper-repro.md`.
