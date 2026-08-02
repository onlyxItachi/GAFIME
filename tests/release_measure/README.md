# GAFIME v1 Release Measurement Suite

These scripts validate the v1 runtime contract from the top-level Python API
down through Rust orchestration, Rust CPU kernels, and optional GPU C ABI
payloads. Targeted gates also preserve public-result compatibility with the
published legacy distributions and host compatibility with older same-ABI GPU
payloads; the removed legacy Python/C++ runtime itself is not built in-tree.

## How To Run

Use a built editable install or put the thin Python package first on
`PYTHONPATH`:

```bash
export PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure
PY=/home/hamza-usta/GAFIME/.venv-release/bin/python

$PY tests/release_measure/contract_00_policy_files.py
$PY tests/release_measure/contract_01_top_level_numpy_parity.py
$PY tests/release_measure/contract_02_feature_generation_reference.py
$PY tests/release_measure/contract_06_release_facing_artifacts.py
$PY tests/release_measure/artifact_01_release_composition.py --scope source-tree
$PY tests/release_measure/v1_architecture_gate.py
```

`backend_01_availability_smoke.py` and
`backend_03_e2e_smoke_per_backend.py` accept a comma-separated
`GAFIME_BACKENDS` selection. They report missing optional payloads as skips only
when the matching `GAFIME_*_V1_LIB` variable is unset and the request reaches
that missing-payload boundary. A configured payload error or a selection that
completes no backend exits nonzero.

When `PYTHONPATH` points at this checkout, rebuild and copy the current native
extension first (`cargo build --release -p gafime-py`, then install
`target/release/libgafime_py.so` under the matching CPython extension filename in
`python/gafime/`). The full
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
python3 tests/release_measure/installed_payload_smoke.py \
  --backend cuda --source-root "$PWD" --execute-profiles
python3 tests/release_measure/installed_payload_smoke.py \
  --backend rocm --source-root "$PWD" --execute-profiles
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
export GAFIME_METAL_PARITY_TOLERANCE=0.00005
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
| `contract_06_release_facing_artifacts.py` | current README/release runbook links, documented CLI parsing, support-skill API guidance, deterministic v1 practice notebook, and generated pipeline syntax/default dependencies | CPU |
| `artifact_01_release_composition.py --scope source-tree` | manifest-owned release identities, precision profiles, ABI, platforms, artifact names, workflow globs, optional extras, and generated matrix | CPU |
| `abi_02_legacy_gpu_payload_compatibility.py` | current-host execution against an older same-ABI CUDA/ROCm/Metal payload, including exact CPU parity and immutable-protocol capability negotiation | CPU/GPU plus older payload |
| `v1_architecture_gate.py` | package layout, forbidden legacy imports, native report view, CPU/GPU payload structure | CPU/GPU |
| `installed_wheel_smoke.py` | clean installed-package import, PyO3 symbols, typed Arrow ingest, all three Core profiles, adversarial fp64 preservation, significance identity, and eager/compiled value parity | installed Core wheel |
| `installed_payload_smoke.py` | payload separation, RT exclusion, additive precision ABI exports, exact capability masks, Metal fp32-only behavior, and optional physical execution of every supported profile | installed Core/payload pair; device for `--execute-profiles` |

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
| `compile_02_compiled_vs_eager.py` | one-shot, resident-cache, and explicit-compiled parity across first/repeat/target-update runs, including non-finite inputs, large seeds, warnings, significance, and final decisions; timing is context only | CPU/GPU |

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
| `perf_08_v047_distribution_ab.py` | isolated v0.4.7 or `v0.5.0-legacy` Core/CUDA/ROCm/Metal distributions vs current one-shot, eager-cache, and compiled paths; report order, tuple/family identity, candidate-id stability, warnings, deterministic decisions, optional stochastic snapshots, numeric/performance thresholds, and provenance. Cross-distribution stochastic values are recorded but not value-gated because legacy candidate-wise permutation streams and current family-wise maxT are different statistical methods; current one-shot/resident/compiled stochastic parity remains strict. | CPU/GPU, scikit-learn/OpenML preparation |
| `perf_09_interaction_diagnostics_overhead.py` | public safe-path one-shot and resident timing distributions for base/candidate diagnostic A/B; also validates candidate count, availability, and zero false-positive diagnostics | CPU/GPU and separate base/candidate installs |
| `perf_10_cpu_covariance_finite_pass.py` | public resident Core Pearson-only and Pearson+R2 timing distributions for the finite-input SIMD covariance A/B | CPU with NumPy input |
| `perf_11_cpu_mi_histogram.py` | public resident Core fixed-bin MI timing distributions, kept separate from the ignored internal histogram/helper microbenchmark | CPU with NumPy input |

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
