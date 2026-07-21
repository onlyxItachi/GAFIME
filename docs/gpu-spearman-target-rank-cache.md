# GPU Spearman Target-Rank Cache

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
