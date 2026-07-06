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
- CUDA receives compact validated path terms and materializes membership only.
- Missing support is explicit through the optional symbol; no backend fallback is allowed.
- Generic CUDA metric files remain separate: `kernels.cu` / `launcher.cu` must not absorb RT-specific execution logic beyond the exported C ABI bridge in `launcher.cu`.

## Runtime RT Path

The default CUDA payload builds the exact SM membership comparator. Building with
`-DGAFIME_CUDA_ENABLE_OPTIX_RT=ON` additionally compiles OptiX PTX from
`src/cuda/rt_kernels.cu`, embeds that PTX in the CUDA payload, and lets
`gafime_gpu_decision_path_membership` choose the RT path when the batch is
representable as finite 1D/2D/3D boxes on RTX-class hardware.

The RT path has two geometry modes:

- bounded 2D boxes use two OptiX triangles per path, so traversal can use the
  fixed-function triangle path; any-hit still rechecks the exact GAFIME
  `>`/`<=` box predicate before writing membership. Points on the shared
  rectangle diagonal are assigned to one triangle only in both single-GAS and
  instanced grouped modes, so direct score counts cannot double-count boundary
  hits.
- custom AABBs remain the exact fallback for 1D/3D or open-bound batches.

`GAFIME_CUDA_DECISION_PATH_RT_GEOMETRY=aabb` forces the custom-AABB path for
profiling and parity checks.

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

The RT path is used only when correctness can stay exact:

- uploaded feature values are finite, so NaN-undetermined semantics are not lost,
- all thresholds are finite,
- the batch uses at most three unique feature axes,
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

Otherwise CUDA uses the exact SM comparator inside the same backend. Callers can
set `GAFIME_DECISION_PATH_FLAG_REQUIRE_RT` in `GafimeDecisionPathBatch.flags` to
turn an unrepresentable or unavailable RT path into an explicit unsupported
status instead of allowing the SM path. For test runs, `GAFIME_CUDA_DECISION_PATH_RT=off`
forces SM execution, and `GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP=1` in the C++ smoke
sets the RT-required ABI flag.

## Standalone OptiX Smoke

`tests/gpu/cuda_rt_decision_path_optix_smoke.cu` remains a standalone GPU smoke
for the RT-core hypothesis and for quick custom-primitive debugging outside the
shared payload.

Build shape:

```bash
/usr/local/cuda/bin/nvcc --std=c++23 \
  -I/home/hamza-usta/SDKs/optix-sdk/include \
  -DGAFIME_OPTIX_DEVICE --ptx tests/gpu/cuda_rt_decision_path_optix_smoke.cu \
  -o /tmp/gafime_rt_decision_path_optix.ptx

/usr/local/cuda/bin/nvcc --std=c++23 -O3 \
  -I/home/hamza-usta/SDKs/optix-sdk/include \
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

## Remaining RT Work

The runtime path now proves end-to-end connectivity through the public CUDA ABI
and is faster than the SM membership comparator on the measured large bounded-2D
workload. Remaining work is performance maturity:

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
  `--repeats=N` to collect best-of-N GPU timings inside one process without
  rebuilding/uploading between shell-loop runs.
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
gpu_rt_score direct  3.195 ms  168.048 G eval/s
gpu_sm_score        21.077 ms   25.471 G eval/s
score parity        rt_max_abs=5.03659e-06 sm_max_abs=7.45058e-08

rows=262,144 paths=512 evals=134.218M
gpu_rt_score direct  1.572 ms   85.397 G eval/s
gpu_sm_score         6.398 ms   20.977 G eval/s
score parity        rt_max_abs=7.58097e-06 sm_max_abs=1.19209e-07
```

Those numbers are performance evidence for the saturation direction, not a
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
relevant RT saturation signal. The safety invariants are covered by CUDA
regression tests that update only the target and reorder grouped public path
rows while reusing unchanged feature-derived points.

This is the current proof that RT scoring benefits from higher region batching:
the compact score path can evaluate multi-billion membership-equivalent
workloads while keeping public output at `paths * metrics` rows. The mixed-axis
run proves the first-fit grouping path at scale rather than only the single
feature-pair case, and `--mixed-axis-pairs=N` stress runs keep that coverage as
the number of internal RT groups increases.

NCU on the triangle OptiX launch still does not expose a direct RT-core
saturation percentage, but the visible counters changed in the desired
direction versus the old custom-AABB path:

```text
triangle cached optixLaunch:
  Memory [%]         32.30
  Compute (SM) [%]   31.37
  DRAM Cycles Active 32.30
  branch resolving    6.22%
  divergent instr     0

old custom AABB optixLaunch:
  Memory [%]         27.01
  Compute (SM) [%]   54.34
  DRAM Cycles Active  3.76
  branch resolving    8.93%
  divergent instr     0
```

That means the current limiter is not normal branch divergence. The next
checkpoint is to decide whether the direct traversal-stat tolerance is acceptable
for default scoring, and then increase region batching so OptiX has enough
parallel traversal work to stay saturated on larger RTX devices.
