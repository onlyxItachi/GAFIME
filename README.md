# GAFIME: GPU-Accelerated Feature Interaction Mining Engine

![PyPI version](https://img.shields.io/pypi/v/gafime)
![Python Versions](https://img.shields.io/pypi/pyversions/gafime)
![License](https://img.shields.io/github/license/onlyxItachi/GAFIME)

GAFIME is a high-performance feature interaction mining engine for tabular and
structured machine-learning workflows. It treats feature generation as a native
systems problem: Python owns the user API, while C++ Core, Rust scheduling, and
CUDA kernels own the hot execution paths.

The engine is designed for workloads where candidate feature interactions,
threshold-style regions, and temporal transforms become too expensive to search
with ordinary Python loops or model-by-model trial code.

## Installation

```bash
pip install gafime
```

Optional integrations:

```bash
pip install gafime[sklearn]
pip install gafime[bench]
```

Generate the reference notebook:

```bash
gafime --init
```

## v0.4.5 Native Spine

v0.4.5 removes the old NumPy execution fallback and makes native execution the
only Engine path:

- CPU execution uses the C++ Core backend with runtime SIMD dispatch for x86
  and ARM64 hosts.
- CUDA execution uses arity-aware native batches for continuous interaction
  scans.
- Python-to-native data movement uses fp32-owned C++ buffers by default.
- Rust `subfunctions.BatchScheduler` groups candidate descriptors by arity and
  cache-local feature order before launch.
- Native C++ memory/scoring code is separated from ISA-specific accumulation
  kernels, so wheel builds do not apply AVX/NEON flags globally.
- ARM Linux and ARM Windows wheels are CPU-native distributions with Rust
  orchestration and C++ Core NEON/scalar dispatch; NVIDIA CUDA payloads are
  intentionally packaged only in x86_64 Linux and Windows x64 wheels.
- Time-series feature engineering is now an explicit Engine candidate family.
- GPU discrete feature engineering remains soft/vectorized only; hard discrete
  mode raises a clear error on GPU.

CUDA computes stats-backed report metrics (`pearson`, `r2`) on the GPU path.
`spearman` and `mutual_info` remain valid report metrics and are completed by
the native metric scorer rather than being rejected by CUDA selection.

## Candidate Families

GAFIME v0.4.x supports:

- continuous interaction candidates up to the configured arity budget,
- discrete soft thresholds, intervals, value-gated thresholds, rectangles, and
  value-in-rectangle candidates,
- explicit time-series candidates such as lag, delta, velocity, acceleration,
  rolling mean, rolling standard deviation, and rolling sum,
- scikit-learn transformer integration through `gafime.sklearn.GafimeSelector`,
- large-file streaming helpers through `GafimeStreamer`.

For the full API surface, use the notebook:

- [docs/gafime_full_api_reference_notebook.ipynb](/docs/gafime_full_api_reference_notebook.ipynb)

## Performance Notes

v0.4.5 changes the CUDA execution shape from legacy pair-oriented/materialized
paths to resident matrix plus homogeneous arity batches. On the local RTX 4060
Laptop GPU benchmark:

- 32,768 rows, 24 features, 55,454 interactions:
  v0.4.1 CUDA `11.8308 s` -> v0.4.5 CUDA `0.6363 s`.
- 131,072 rows, 24 features, 55,454 interactions:
  v0.4.1 CUDA failed with a `27.1 GiB` allocation; v0.4.5 CUDA completed in
  `0.8484 s`.
- The same large CPU/Core workload improved from `7.4427 s` to `3.0983 s`.

Reproduce the benchmark with:

```bash
python tests/benchmark_v045_native_spine.py --n-samples 131072 --n-features 24 --max-comb-size 5 --backends cpu,cuda
```

Full benchmark details:

- [docs/v0.4.5-native-spine-benchmark-results.md](/docs/v0.4.5-native-spine-benchmark-results.md)
- [docs/releases/v0.4.5.md](/docs/releases/v0.4.5.md)

## Backend Policy

`backend="auto"` resolves only native backends and uses platform-aware priority:

- macOS: `metal` → `core`
- Linux/Windows x86_64: `cuda` → `core`
- Linux/Windows ARM64: `core`

Explicit impossible requests fail clearly: CUDA is not a macOS backend, Metal is
not a non-macOS backend, and current ARM Linux/Windows wheels do not expose a
CUDA backend. `backend="gpu"` is deprecated because it is ambiguous across
platforms; use `auto`, `cuda`, `metal`, or `core`.

Reports are native structured objects. Use properties such as
`report.interactions`, `report.decision`, and `report.backend` for framework
integration. `DiagnosticReport.to_dict()` is retained only as a deprecated
export convenience and should not be used as a runtime data-flow path.

## v0.4.6 Metal and Native Report Release

v0.4.6 validates the native Metal path on GitHub's Apple Silicon macOS runner
and moves framework integration away from JSON-style report materialization.
The release keeps `to_dict()` as an explicit deprecated export boundary while
normal code should read the live `DiagnosticReport` properties directly.

Full details:

- [docs/releases/v0.4.6.md](/docs/releases/v0.4.6.md)
- [docs/v0.4.6-metal-native-backend.md](/docs/v0.4.6-metal-native-backend.md)

User-facing Rust helper imports should use:

```python
from gafime import subfunctions
```

Direct `import gafime_cpu` is an implementation detail.

## 🌌 Why GAFIME? The Performance Ceiling

In the current data science landscape, mining interaction data (like checking `Feature X * Feature Y` against the target) is painfully slow on CPUs or inefficiently memory-managed on GPUs. GAFIME achieves:

1. **Hardware-Bound Execution**: GAFIME targets physical memory bandwidth limits, minimizing the overhead of standard GPU python workflows. You hit the system's ceiling.
2. **Zero-Overhead Scaling**: Utilizing Rust's FFI capabilities on top of optimized CUDA C++, GAFIME bypasses the Python Global Interpreter Lock (GIL) ensuring every clock cycle executes pure feature logic.
3. **Cross-Platform Scalability**: Whether you're on a CPU-native workflow or an RTX workstation targeting `CUDA` registers, GAFIME auto-discovers and optimizes for your hardware at runtime.

### Caching and Branch-less Operations

GAFIME's specialized memory layout and cache-local launch ordering keep tabular
feature access predictable. GPU discrete feature engineering uses branchless
soft gates instead of hard threshold branches.

## 🛠️ Technology Stack

- **Core Engine**: C++ / CUDA for production performance paths, with a native Metal path validated through the Apple Silicon lab workflow.
- **Safety Pipeline & Schedulers**: Rust (Memory safe FFI interface scheduling)
- **Data Science Interfacing**: Python (Polars / Numpy bindings seamlessly communicating across boundaries)

## ✅ For being honest

-> Current release: **v0.4.6**.

-> The project is developed with the help of current frontier SOTA models such as Gemini 3.1 Pro (high) , Claude Opus 4.6 (high) and GPT 5.5 (xhigh).

## Project References

- [docs/releases/v0.4.6.md](/docs/releases/v0.4.6.md) - v0.4.6 release notes.
- [docs/releases/v0.4.5.md](/docs/releases/v0.4.5.md) - v0.4.5 release notes.
- [docs/v0.4.6-metal-native-backend.md](/docs/v0.4.6-metal-native-backend.md) - Metal native backend implementation notes.
- [docs/v0.4.5-native-spine-benchmark-results.md](/docs/v0.4.5-native-spine-benchmark-results.md) - source-built benchmark comparison.
- [docs/v0.4.1-math-corrections.md](/docs/v0.4.1-math-corrections.md) - adaptive MI and discrete selector math corrections.
- [docs/v0.4.0-discrete-functions.md](/docs/v0.4.0-discrete-functions.md) - discrete function family details.
- [CONTRIBUTING.md](/CONTRIBUTING.md) - local development and native compilation notes.

## Contact

Maintainer: Hamza Usta  
Email: <hamzausta2222@gmail.com>
