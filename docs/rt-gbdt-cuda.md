# CUDA RT-Core Decision-Path Spike

## Objective

The spike tests whether shallow `decision_path` candidates from GBDT-style split borders can be evaluated faster on NVIDIA RT cores than on normal CUDA SMs.

A decision-path candidate is a hard conjunction such as:

```text
f0 > 0.5 AND f1 <= 1.25
```

For depth 2-3 paths, that conjunction maps naturally to a small axis-aligned box. The research question is whether point-in-box membership over many rows and many candidate boxes can move from divergent SM branches to RTX fixed-function traversal.

## Current Checkpoint

This branch adds the CUDA SM comparator and ABI seam that RT work must beat:

- `GafimeDecisionPathTerm` and `GafimeDecisionPathBatch` in the GPU C ABI.
- Optional `gafime_gpu_decision_path_membership` symbol, implemented by CUDA only.
- CUDA `decision_path_membership_kernel` over the resident feature-major matrix.
- Rust optional loader/wrapper in `gafime-gpu-sys`.
- C++ ABI smoke coverage and Rust CPU-parity coverage.

The implementation preserves Rust ownership:

- Rust discovers paths, validates config, plans features, selects backends, and schedules work.
- CUDA receives compact validated path terms and materializes membership only.
- Missing support is explicit through the optional symbol; no backend fallback is allowed.

## OptiX Spike Smoke

`tests/gpu/cuda_rt_decision_path_optix_smoke.cu` is a standalone GPU smoke for the RT-core hypothesis. It is intentionally outside the public runtime path until the parity and performance case is strong enough to wire behind `gafime_gpu_decision_path_membership`.

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

This checkpoint does not claim OptiX or RT-core acceleration. The next checkpoint must:

- build one depth-2/3 path set as AABBs,
- run point-in-box membership through OptiX on NVIDIA RTX hardware,
- prove border and NaN parity against the SM comparator,
- benchmark RT-core membership against the SM comparator at row x candidate scale,
- abort the RT path if it is not clearly faster on large workloads.
