# GPU Continuous Unary Stats Cache

This note documents the v1 continuous unary fast path added for CUDA and ROCm
payloads. It is backend-internal and does not change the Rust/Python API or the
stable GPU C ABI.

## Ownership

Rust still owns planning, backend selection, scheduling, and result reduction.
CUDA/HIP launchers only choose the legal device kernel for a validated chunk.
CUDA/HIP kernels only compute metrics over the resident feature-major matrix.

## Fast-Path Contract

The specialized kernel is legal only for continuous arity-1 chunks when both the
resident features and target are finite. The backend computes:

- target mean and centered variance after matrix upload or target update,
- unary feature mean and centered variance after matrix upload.

Those compact stats live in backend-owned device memory and are reused for unary
Pearson/R2 scoring. The specialized scoring kernel then performs only the
feature-target covariance pass for each unary candidate.

The generic continuous kernel remains responsible for:

- arity greater than 1,
- non-finite feature or target input,
- pairwise finite filtering,
- Spearman and mutual-information companion kernels.

The launcher skips the continuous Pearson/R2 covariance kernel entirely when a
chunk requests only metric-specific kernels such as mutual information or
Spearman. This avoids a full wasted candidate sweep whose outputs would be
overwritten by the metric-specific kernels.

CUDA enables the unary stats fast path for non-permutation covariance launches.
Target-only caching was too noisy for plain launches, but after feature stats
were added the one-pass unary kernel showed a positive median result for both
plain and graph launches. Permutation launches stay on the generic path because
the target changes inside the backend permutation loop. ROCm enables the fast
path for both plain and graph launches because the HIP payload showed a clear
improvement in both modes.

If target finiteness changes on `gafime_gpu_matrix_update_target`, the backend
invalidates the graph cache before the next launch. If target values change but
finiteness does not, the cached graph remains valid because kernels read the
updated device target-stats object through a stable pointer. Full matrix upload
invalidates graph caches because feature data and feature stats change together.

## Local Evidence

Temporary C ABI benchmark shape:

- CUDA: 8192 rows, 4096 unary candidates, 30 resident executions.
- ROCm: 4096 rows, 2048 unary candidates, 20 resident executions.
- Metrics: Pearson and R2.
- Payload baseline: clean `origin/main` built with the same compiler flags.

Observed representative results on this machine:

| Backend | Mode | Main | Branch | Result |
| --- | --- | ---: | ---: | --- |
| CUDA | plain | 46.38 GEval/s | 49.23 GEval/s | no regression target |
| CUDA | graph | 34.02 GEval/s | 46.20 GEval/s | graph replay lift |
| ROCm | plain | 6.66 GEval/s | 12.68 GEval/s | improvement |
| ROCm | graph | 5.33 GEval/s | 10.97 GEval/s | improvement |

CUDA policy-broadening follow-up after enabling unary stats for all
non-permutation covariance launches:

| Backend | Mode | Main median | Branch median | Result |
| --- | --- | ---: | ---: | --- |
| CUDA | plain | 31.7 GEval/s | 42.0 GEval/s | positive median |
| CUDA | graph | 39.7 GEval/s | 41.4 GEval/s | positive median |

Metric-only dispatch benchmark:

| Backend | Metric Set | Shape | Main | Branch | Result |
| --- | --- | --- | ---: | ---: | --- |
| CUDA | MI-only | 2048 x 1024 x 5 | 0.171 GEval/s | 0.230 GEval/s | skips wasted covariance sweep |
| ROCm | MI-only | 2048 x 1024 x 5 | 0.088 GEval/s | 0.089 GEval/s | neutral |
| CUDA | Spearman-only | 512 x 256 x 3 | 0.049 GEval/s | 0.049 GEval/s | neutral |
| ROCm | Spearman-only | 512 x 256 x 3 | 0.095 GEval/s | 0.102 GEval/s | small improvement |

Correctness gates run with staged payloads:

- `backend_02_cross_backend_parity.py`
- `graph_01_replay_parity.py`
- `contract_03_family_metric_backend_surface.py`
- `v1_architecture_gate.py --include-gpu`
