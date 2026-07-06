# GPU Continuous Target-Stats Cache

This note documents the v1 continuous unary fast path added for CUDA and ROCm
payloads. It is backend-internal and does not change the Rust/Python API or the
stable GPU C ABI.

## Ownership

Rust still owns planning, backend selection, scheduling, and result reduction.
CUDA/HIP launchers only choose the legal device kernel for a validated chunk.
CUDA/HIP kernels only compute metrics over the resident feature-major matrix.

## Fast-Path Contract

The specialized kernel is legal only for continuous arity-1 chunks when both the
resident features and target are finite. The backend computes target mean and
target centered variance once after matrix upload or target update, stores that
state in backend-owned device memory, and reuses it for unary Pearson/R2 scoring.

The generic continuous kernel remains responsible for:

- arity greater than 1,
- non-finite feature or target input,
- pairwise finite filtering,
- Spearman and mutual-information companion kernels.

CUDA enables the unary target-stats fast path only for graph replay launches.
Local RTX 4060 measurements showed plain launches were too noisy and sometimes
slower, while graph replay consistently benefited after the specialized kernel
was split from the generic path. ROCm enables the fast path for both plain and
graph launches because the HIP payload showed a clear improvement in both modes.

If target finiteness changes on `gafime_gpu_matrix_update_target`, the backend
invalidates the graph cache before the next launch. If target values change but
finiteness does not, the cached graph remains valid because kernels read the
updated device target-stats object through a stable pointer.

## Local Evidence

Temporary C ABI benchmark shape:

- CUDA: 8192 rows, 4096 unary candidates, 30 resident executions.
- ROCm: 4096 rows, 2048 unary candidates, 20 resident executions.
- Metrics: Pearson and R2.
- Payload baseline: clean `origin/main` built with the same compiler flags.

Observed representative results on this machine:

| Backend | Mode | Main | Branch | Result |
| --- | --- | ---: | ---: | --- |
| CUDA | plain | median approximately 38 GEval/s | median approximately 38 GEval/s | no regression target |
| CUDA | graph | median approximately 38 GEval/s | median approximately 46 GEval/s | graph replay lift |
| ROCm | plain | 6.71 GEval/s | 7.36 GEval/s | improvement |
| ROCm | graph | 5.61 GEval/s | 7.53 GEval/s | improvement |

Correctness gates run with staged payloads:

- `backend_02_cross_backend_parity.py`
- `graph_01_replay_parity.py`
- `contract_03_family_metric_backend_surface.py`
- `v1_architecture_gate.py --include-gpu`
