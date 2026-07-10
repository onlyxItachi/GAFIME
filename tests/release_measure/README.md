# GAFIME v1 Release Measurement Suite

These scripts validate the v1 runtime contract from the top-level Python API
down through Rust orchestration, Rust CPU kernels, and optional GPU C ABI
payloads. They are not a compatibility suite for the removed legacy Python/C++
runtime.

## How To Run

Use a built editable install or put the thin Python package first on
`PYTHONPATH`:

```bash
export PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure
PY=/home/hamza-usta/GAFIME/.venv-release/bin/python

$PY tests/release_measure/contract_00_policy_files.py
$PY tests/release_measure/contract_01_top_level_numpy_parity.py
$PY tests/release_measure/contract_02_feature_generation_reference.py
$PY tests/release_measure/v1_architecture_gate.py
```

When `PYTHONPATH` points at this checkout, rebuild and copy the current native
extension first (`cargo build --release -p gafime-py`, then install
`target/release/libgafime_py.so` as `python/gafime/gafime_py.abi3.so`). The full
ordered sequence is in `docs/cuda-template-kernel-hardening.md`.

When native GPU payloads are available:

```bash
export GAFIME_CUDA_V1_LIB="$PWD/build/cuda-template-hardening-both/libgafime_cuda_v1.so"
export GAFIME_CUDA_RT_V1_LIB="$PWD/build/cuda-template-hardening-both/libgafime_cuda_v1_rt.so"
export GAFIME_ROCM_V1_LIB="$PWD/build/rocm-template-hardening-default/libgafime_rocm_v1.so"
export GAFIME_CUDA_ABI_SMOKE=/tmp/cuda_v1_abi_smoke
export GAFIME_CUDA_RT_ABI_SMOKE=/tmp/cuda_v1_abi_smoke_rt
export GAFIME_ROCM_ABI_SMOKE=/tmp/rocm_v1_abi_smoke
python3 tests/release_measure/v1_architecture_gate.py --include-gpu
```

Build both smoke binaries from the current tree before that command; the exact
payload and smoke build sequence is in
`docs/cuda-template-kernel-hardening.md`.

Generated CUDA/HIP artifacts can also be inspected without reserving a GPU:

```bash
python3 tests/release_measure/gpu_static_kernel_report.py \
  --cuda-lib build/cuda-template-hardening-both/libgafime_cuda_v1.so \
  --hip-lib build/rocm-template-hardening-default/libgafime_rocm_v1.so \
  --hip-target gfx1150 \
  --require-template-matrix \
  --require-topk-split \
  --require-no-spills
```

The macOS workflow runs the Metal behavioral gates with the built payload:

```bash
export GAFIME_METAL_V1_LIB=build/metal-cmake/libgafime_metal_v1.dylib
export GAFIME_METAL_V1_METALLIB=build/metal-cmake/gafime_metal_v1.metallib
export GAFIME_METAL_PARITY_TOLERANCE=0.002
cargo test -p gafime-gpu-sys \
  metal_device_topk_covers_split_directions_ties_and_large_k_when_available \
  -- --nocapture
cargo test -p gafime-gpu-sys \
  metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available \
  -- --nocapture
```

Telemetry helpers write schema `gafime.telemetry.v0.5.0-rc1` records until the
next telemetry schema bump. That schema name is historical; the runtime being
measured here is v1.

## Active Scripts

### contract

| script | validates | needs |
|---|---|---|
| `contract_00_policy_files.py` | contract docs, agent docs, compiler/safety policy text | CPU |
| `contract_01_top_level_numpy_parity.py` | top-level API bit parity against NumPy reference for base metrics | CPU |
| `contract_02_feature_generation_reference.py` | continuous, compile, time-series, decision-path, and dataload reference checks | CPU |
| `contract_03_family_metric_backend_surface.py` | all configured backends across continuous, time-series, decision-path, and all metric ids | CPU/GPU |
| `contract_04_adaptive_mi_quantization.py` | adaptive MI template resolution and ranking stability against a large-sample reference | CPU |
| `v1_architecture_gate.py` | package layout, forbidden legacy imports, native report view, CPU/GPU payload structure | CPU/GPU |

### decision_path

| script | measures | needs |
|---|---|---|
| `dp_02_openml_tour_logged.py` | baseline vs assisted lift across datasets | CPU |
| `dp_03_method_effect_gated_soft.py` | hard/gated path strategy lift comparisons | CPU |
| `dp_05_dataset_structure_map.py` | where decision-path lift appears by dataset structure | CPU |
| `dp_06_depth_rounds_sweep.py` | depth/rounds lift vs cost | CPU |
| `dp_07_boosting_residual_reduction.py` | boosting residual reduction and path growth | CPU |
| `dp_08_leakage_safety.py` | train-mined vs leaked feature generation gap | CPU |

### compile

| script | validates | needs |
|---|---|---|
| `compile_01_plan_correctness.py` | native compile artifact and plan shape | CPU |
| `compile_02_compiled_vs_eager.py` | compiled vs eager output parity and timing | CPU |

### graph

| script | validates | needs |
|---|---|---|
| `graph_01_replay_parity.py` | graph replay equals plain launch within approved tolerance | GPU |
| `graph_02_launch_shaping_timing.py` | graph vs plain launch latency | GPU |

### backends

| script | validates | needs |
|---|---|---|
| `backend_01_availability_smoke.py` | public API backend resolution and explicit errors | CPU/GPU |
| `backend_02_cross_backend_parity.py` | core vs CUDA vs ROCm numerical parity | GPU |
| `backend_03_e2e_smoke_per_backend.py` | per-backend end-to-end smoke through top-level API | CPU/GPU |

### performance

| script | measures | needs |
|---|---|---|
| `perf_01_residency_session_benefit.py` | resident compile/session reuse vs fresh analyze | CPU/GPU |
| `perf_02_metric_cache_benefit.py` | metric-cache hit rate and counters | GPU |
| `perf_04_cpu_native_kernels.py` | CPU SIMD dispatch, column layout, and scratch-reuse guardrails | CPU |
| `gpu_static_kernel_report.py` | CUDA SASS and HIP code-object size, register, shared/LDS, spill, specialization, and top-k topology checks | CUDA/HIP toolchains, no GPU |
| `perf_06_gpu_mi_specializations.py` | resident MI throughput by candidate count, candidate-sample pairs, and bins | CUDA/HIP GPU |
| `perf_07_rocm_mi_wave_ab.py` | provenance-checked, numerically guarded interleaved HIP high-bin A/B with control normalization and JSON output | HIP GPU and two payload builds |
| `perf_08_v047_distribution_ab.py` | isolated v0.4.7 Core/CUDA/ROCm distributions vs current eager-cache and compiled-replay performance, provenance, and full-result parity | CPU/GPU, scikit-learn/OpenML preparation |

`_measure_common.py` contains shared loaders, telemetry helpers, candidate
materialization helpers, and model baselines. `run_cpu_suite.sh` and
`run_gpu_suite.sh` run focused subsets, continue long enough to report every
failed script, and return nonzero if any script failed. Metal excludes the
unsupported graph and CUDA RT measurements.

## Guardrails

- Feature generation is validated from the public API.
- CPU and GPU backends must not silently fall back to another backend.
- Numerical output must be bit-equal where policy says bit parity is possible,
  otherwise the approved tolerance must be documented and tested.
- Mutual-information parity is split by estimator: default adaptive CPU MI is
  not compared to GPU fixed-bin MI; GPU MI parity is tested only through the
  explicit `mi_approximate=True` fixed-bin path. In both cases `mi_bins` is an
  adaptive maximum; fixed-bin parity uses the shared sample-size-selected
  template rather than forcing the configured ceiling.
- Performance artifacts are useful only when generated by these scripts or the
  architecture gate, not from hand-written numbers.
