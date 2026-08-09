# GPU Spearman Target-Rank Cache

> **Historical diagnostic evidence only.** The timing tables in this note use
> focused cache controls from intermediate source snapshots. They do not satisfy
> the PR #70 fresh-process, randomized A/B release methodology and must not be
> copied into the PR performance table or used for a cross-profile claim. Final
> candidate performance is reported only from the commit- and binary-bound
> `perf_13_precision_profiles.py` artifacts.

This note records the exact bounded optimization used by CUDA, ROCm, and Metal
for unary Spearman batches. It changes neither the estimator nor the public GPU
ABI: average-tie target ranks are computed once per resident target and reused
across eligible unary candidates.

## Dispatch Contract

The cache is enabled only when all of these conditions hold:

- the matrix features and target are finite,
- the launch contains Spearman and at least two unary candidates,
- the row count is in the inclusive range `128..4096`, and
- the launch is not a backend permutation pass.

Target upload or update invalidates the cached ranks. Interactions, non-finite
inputs, one-candidate launches, permutations, and shapes outside the bounded row
range retain the previous pairwise-rank kernel. The cached path is exact: it
removes repeated target comparisons but leaves feature-rank construction and
therefore overall large-row scaling at `O(candidates * rows^2)`.

## Reproduction

Build the exact baseline and branch CUDA payloads with Release flags for the same
architecture. Compile the standalone C ABI fixture once against each payload:

```bash
c++ -std=c++20 -O3 -march=native \
  tests/gpu/cuda_spearman_target_cache_bench.cpp \
  -L/path/to/payload -Wl,-rpath,/path/to/payload \
  -lgafime_cuda_v1 -o /tmp/gafime_spearman_bench

/tmp/gafime_spearman_bench ROWS CANDIDATES 2 7
```

The reviewed A/B used baseline `4d35345`, branch `4cab6ca`, CUDA 13.3.73,
driver 610.43.02, and an RTX 4060 Laptop GPU (`sm_89`). Each table entry is the
median of three fresh-process warm medians; greater speedup is better.

| Rows | Unary candidates | Baseline ms | Branch ms | Speedup | Dispatch |
| ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 32 | 0.020370 | 0.020221 | 1.007x | row-bound fallback |
| 512 | 1 | 0.257374 | 0.256985 | 1.002x | candidate-count fallback |
| 512 | 32 | 0.488948 | 0.255985 | 1.910x | cached target ranks |
| 2048 | 16 | 3.796981 | 1.915461 | 1.982x | cached target ranks |
| 4096 | 8 | 15.126992 | 7.564283 | 2.000x | cached target ranks |

Every paired run produced the same weighted metric checksum. The two deliberate
fallback controls are neutral, so the improvement does not come from unrelated
launcher changes.

## PerfDigest Attribution

Nsight Compute `--set full` reports for the `512 x 32` shape were digested with
PerfDigest. Comparing the warm baseline unary kernel with the warm cached-target
kernel gives:

| Metric | Baseline | Cached target | Delta |
| --- | ---: | ---: | ---: |
| duration | 668.256 us | 339.520 us | -49.19% |
| registers/thread | 40 | 40 | 0 |
| shared memory/block | 12,288 B | 12,288 B | 0 |
| achieved occupancy | 24.89% | 24.16% | -0.73 pp |

The one-time target-rank kernel was 86.144 us. This is consistent with removing
one of the two pairwise rank scans from every candidate and paying one shared
target scan per target generation.

CUDA correctness is covered by the native tied-rank, target-update, interaction,
one-candidate, and non-finite gates. Matching ROCm gates passed locally. Metal
has the same bounded contract and is compiled/tested on Apple CI, but this Linux
host provides no Metal performance claim. The next algorithmic step is a proven
sort/rank design that lowers feature ranking below quadratic work; this cache is
not presented as that final solution.

## ROCm ABI 1.0 Compatibility Evidence

The same narrow cache is active on the frozen ABI 1.0 ROCm adapter.  The ABI
1.0 C consumer, the ABI 1.0 dynamic-arity-6 consumer, and the ABI 1.1 C/Rust
consumers all passed physically on `gfx1150` against the candidate payload.
The dynamic arity-6 result remained the frozen tuple
`(0.188239485, 0, 0.0205882359)`.

To isolate the cache from the rest of the optimized payload, a temporary
gfx1150-only control was built from the same source snapshot with only host
target-rank preparation disabled.  No release artifact was produced from that
control.  Each row below is the median of three fresh-process trials, each with
ten untimed warmups and thirty timed host-boundary executions; the focused
candidate and control used the same `gfx1150` device, compiler, runtime,
workload, and ABI 1.0 C consumer.

| Rows | Unary candidates | Cache-disabled control (ms) | Candidate (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 32 | 0.168052 | 0.142625 | 1.178x |
| 2048 | 16 | 2.075581 | 1.154013 | 1.799x |
| 4096 | 8 | 8.074431 | 3.632449 | 2.223x |

All rows returned the same deterministic metric checksum as the control.  The
focused candidate used for this A/B was the gfx1150-only
`/home/hamza-usta/.cache/gafime-pr70-work/rocm-abi-test/libgafime_rocm_v1.so`,
2,473,936 bytes, SHA-256
`7b52289ef1a191d7d22580f88bd3fa73514cac9cfd201e4b62b01c22d998c6ab`; the
temporary gfx1150-only cache-disabled control was 2,617,488 bytes, SHA-256
`901f5a932596ff49d3e84a8821509039481bd6e2c9e39fb9cd6e383f667e3f8b`.
The control is a performance oracle only and is not part of the distributed
payload.  The separately inspected all-13-target release candidate was
27,681,520 bytes, SHA-256
`fd430119c83b4b01aa613c4dad89853f90c75d8ed278273f4d6437a15ec1d306`, built
for `gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102,gfx1150,gfx1151,gfx1200,
gfx1201,gfx90a,gfx942,gfx950`; it was not used for the focused A/B table.
ABI 1.0 arities above five continue through the single dynamic compatibility
kernel and are covered independently by the arity-6 fixture.
