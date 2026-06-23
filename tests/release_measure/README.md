# GAFIME v0.5 — Release Measurement Suite

Scripts that produce the **exact measurements** for the v0.5 release: decision_path
correctness + lift, framework export zero-copy, `gafime.compile` (plan), CUDA/HIP
graph launch-shaping, cross-backend parity, and the perf/telemetry spans for the
release notes.

> **Run LATER, against the merged integration branch.** Logged scripts emit
> canonical telemetry (schema `gafime.telemetry.v0.5.0-rc1`) into `~/gafime_telemetry/`
> (per-run JSON + `index.csv`). **Release notes cite only logged artifacts.**

## How to run

```bash
export PYTHONPATH=/home/hamza-usta/GAFIME-integration:/home/hamza-usta/gafime_release_measure
PY=/home/hamza-usta/.venvs/gafime-dl-py314/bin/python   # numpy 2.4 / sklearn / openml
$PY dp_02_openml_tour_logged.py
```
GPU scripts use the CUDA venv and a backend env var:
```bash
PY=/home/hamza-usta/.venvs/mc-torch-cu/bin/python
GAFIME_GRAPH_BACKEND=cuda  $PY graph_02_launch_shaping_timing.py   # rocm on the AMD box
GAFIME_BACKEND=cuda        $PY perf_01_residency_session_benefit.py
```

## Scripts

### decision_path (the v0.5 headline)
| script | measures | needs |
|---|---|---|
| `dp_01_parity_native_vs_reference.py` | native split-math == greedy-CART reference (exact) | CPU |
| `dp_02_openml_tour_logged.py` | baseline vs assisted lift across datasets (**release-note tour**) | CPU |
| `dp_03_method_effect_gated_soft.py` | all_hard vs gated_hard vs gated_soft × LogReg/MLP | CPU |
| `dp_04_max_bins_sweep.py` | max_bins {0,8,16,32,64}: time + candidate stability vs exact | CPU |
| `dp_05_dataset_structure_map.py` | where lift lives (rich vs poor structure) | CPU |
| `dp_06_depth_rounds_sweep.py` | depth {1,2,3} × rounds {1,5,20}: lift vs cost | CPU |
| `dp_07_boosting_residual_reduction.py` | boosting adds signal (paths ↑, train R² ↑) | CPU |
| `dp_08_leakage_safety.py` | honest (train-mined) vs leaked numbers gap | CPU |

### framework export (needs lab `46482f7` merged + gafime_core rebuilt)
| script | measures | needs |
|---|---|---|
| `export_01_zero_copy_parity.py` | torch/numpy from_dlpack share GAFIME's pointer | CPU |
| `export_02_lifetime_safety.py` | borrow outlives owner; capsule cleanup; 5k stress | CPU |
| `export_03_overhead_vs_copy.py` | zero-copy vs Python-copy at scale | CPU |

### compile (plan)
| `compile_01_plan_correctness.py` | plan exists, chunk/scenario structure, analyze works | CPU |
| `compile_02_compiled_vs_eager.py` | compiled vs eager parity + timing | CPU |

### CUDA/HIP graph (launch-shaping target only)
| `graph_01_replay_parity.py` | graph replay == plain launch (within fp tol) | **GPU** |
| `graph_02_launch_shaping_timing.py` | graph vs plain launch latency (honest, may be ~1.0×) | **GPU** |

### backends
| `backend_01_availability_smoke.py` | which backends resolve native on this host | CPU |
| `backend_02_cross_backend_parity.py` | core vs CUDA vs ROCm numerical parity | **GPU** |
| `backend_03_e2e_smoke_per_backend.py` | per-backend end-to-end + telemetry | CPU+GPU |

### performance hardening
| `perf_01_residency_session_benefit.py` | resident reuse vs fresh compile | CPU/**GPU** |
| `perf_02_metric_cache_benefit.py` | metric-cache hit rate + counters | **GPU** |
| `perf_03_telemetry_e2e_spans.py` | **the release-notes span breakdown** | CPU |

`_measure_common.py` — shared loaders / telemetry / materialization / models.
`run_cpu_suite.sh`, `run_gpu_suite.sh` — batch runners.

## Honest guardrails baked in
- Leakage-safe: mine on TRAIN, materialize train+test (`dp_08` proves the gap).
- Scaled baselines (StandardScaler) so MLP doesn't fake lift (the MSG-57 confound).
- No hand-written perf numbers — only what the artifacts log.
- GPU scripts skip cleanly when that GPU is absent (no fake passes).
