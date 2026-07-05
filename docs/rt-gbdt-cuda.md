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
- CUDA `rt_kernels.cu` owns `decision_path_membership_kernel` over the resident feature-major matrix.
- CUDA `rt_kernels.cu` also owns the OptiX device programs and the point-packing kernel used by RT traversal.
- CUDA `rt_launcher.cu` owns RT membership validation, finite AABB planning, temporary device buffers, OptiX GAS/pipeline launch, exact SM fallback, and copy-back.
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

The RT path is used only when correctness can stay exact:

- uploaded feature values are finite, so NaN-undetermined semantics are not lost,
- all thresholds are finite,
- the batch uses at most three unique feature axes,
- the CUDA device is Turing or newer,
- OptiX runtime initialization and pipeline creation succeed.

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

The runtime path now proves end-to-end connectivity through the public CUDA ABI.
Remaining work is performance maturity:

- cache per-feature-axis GAS and point buffers across resident sessions,
- benchmark RT-core membership against the SM comparator at row x candidate scale,
- keep the RT path disabled by policy if it is not clearly faster on large workloads,
- extend planning to split large mixed-feature batches into several <=3D RT groups instead of using a whole-batch SM fallback.

## Scale Checkpoint

`tests/gpu/cuda_rt_membership_scale_bench.cpp` compares a finite 2D decision-path
box workload across:

- CPU AVX512 membership materialization,
- CUDA RT membership through `gafime_gpu_decision_path_membership` with `GAFIME_DECISION_PATH_FLAG_REQUIRE_RT`,
- CUDA SM membership through the same ABI with `GAFIME_CUDA_DECISION_PATH_RT=off`.

On the local Ryzen AI 9 HX 370 / RTX 4060 Laptop run, all tested cases matched
exactly, but RT was slower than both CPU AVX512 and CUDA SM:

```text
rows=1,048,576 paths=512 evals=536.871M output=2.00 GiB
cpu_avx512  107.947 ms  4.973 G eval/s  18.528 GiB/s output
gpu_rt_abi  217.694 ms  2.466 G eval/s   9.187 GiB/s output
gpu_sm_abi  192.678 ms  2.786 G eval/s  10.380 GiB/s output
```

This means the current RT path is a correctness-connected prototype, not a
performance default. The measured bottleneck is the public membership
materialization path: per-call allocations/GAS work plus full device-to-host
membership copy. The next performance checkpoint must avoid materializing every
row-path membership to host and instead score/reduce on device or cache the RT
resident structures across repeated calls.
