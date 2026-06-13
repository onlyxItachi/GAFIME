# GAFIME

![PyPI version](https://img.shields.io/pypi/v/gafime)
![Python Versions](https://img.shields.io/pypi/pyversions/gafime)
![License](https://img.shields.io/github/license/onlyxItachi/GAFIME)

GPU-Accelerated Feature Interaction Mining Engine.

GAFIME is a native feature interaction mining engine for tabular and structured
machine-learning workflows. Python owns the public API; C++ Core, Rust
scheduling, CUDA, Metal, and ROCm/HIP own the hot execution paths.

The engine is built for workloads where interaction candidates, threshold-style
regions, and temporal transforms become too expensive to search with ordinary
Python loops or model-by-model trial code.

## Install

Core package:

```bash
pip install gafime
```

Optional Python integrations:

```bash
pip install "gafime[sklearn]"
pip install "gafime[bench]"
```

Vendor GPU payloads are explicit in the v0.4.7 distribution design. Pip can
select wheels by Python, ABI, OS, and CPU architecture, but not by local GPU
vendor. CUDA and ROCm payloads therefore need explicit package selection once
the split payload packages are published:

```bash
pip install "gafime[cuda]"
pip install "gafime[rocm]"
```

Apple Silicon Metal follows the macOS arm64 wheel/platform path.

Detailed install and backend policy:

- [docs/backend-selection.md](docs/backend-selection.md)
- [BUILD.md](BUILD.md)

## Basic Usage

```python
from gafime import ComputeBudget, EngineConfig, GafimeEngine

config = EngineConfig(
    backend="auto",
    metric_names=("pearson", "r2"),
    budget=ComputeBudget(max_comb_size=2),
)

report = GafimeEngine(config).analyze(X, y, feature_names=feature_names)
print(report.backend)
print(report.interactions[:5])
```

Generate the reference notebook:

```bash
gafime --init
```

## Candidate Families

GAFIME supports:

- continuous interaction candidates,
- soft discrete thresholds, intervals, value-gated thresholds, rectangles, and
  value-in-rectangle candidates,
- explicit time-series candidates such as lag, delta, velocity, acceleration,
  rolling mean, rolling standard deviation, and rolling sum,
- scikit-learn transformer integration through `gafime.sklearn.GafimeSelector`,
- large-file streaming helpers through `GafimeStreamer`.

GPU discrete feature engineering is soft/vectorized only. Hard discrete mode is
a CPU/Core behavior and raises a clear error on GPU backends.

## Backend Policy

`backend="auto"` resolves native backends with platform-aware priority:

- macOS arm64: `metal -> core`
- Linux/Windows x86_64 with CUDA payloads: `cuda -> core`
- Linux/Windows x86_64 with ROCm payloads: `rocm -> core`
- Linux/Windows ARM64: `core`

GAFIME does not initialize every GPU runtime during `auto` resolution. On mixed
AMD iGPU + NVIDIA dGPU systems, probing CUDA and ROCm in the same process can
be unsafe. The resolver selects a vendor payload first, then initializes only
that backend.

`backend="gpu"` is deprecated because it is ambiguous across CUDA, ROCm, and
Metal. Use `auto`, `cuda`, `rocm`, `metal`, or `core`.

## Native Reports

Reports are structured Python objects. Read properties such as:

- `report.interactions`
- `report.decision`
- `report.backend`
- `report.warnings`

`DiagnosticReport.to_dict()` remains only as a deprecated export convenience.
It should not be used as a runtime data-flow path.

## Developer Docker Images

Docker files in this repository are development environments, not distribution
images. Normal users should install GAFIME from PyPI wheels.

Available source-build containers:

```bash
docker compose run --build gafime-cuda-dev
docker compose run --build gafime-core-smoke
```

The CUDA development image includes the CUDA toolkit, compiler toolchain, Rust,
CMake, and GAFIME development/benchmark/scikit-learn dependencies. Extra
workstation packages can be added with the `EXTRA_PIP_PACKAGES` Docker build
argument. The Core smoke image is a smaller CPU-native source-build check.

Docker details:

- [CONTRIBUTING.md](CONTRIBUTING.md)

## Project References

- [docs/releases/v0.4.7.md](docs/releases/v0.4.7.md)
- [docs/v0.4.7-rocm-native-backend.md](docs/v0.4.7-rocm-native-backend.md)
- [docs/backend-selection.md](docs/backend-selection.md)
- [docs/gafime_full_api_reference_notebook.ipynb](docs/gafime_full_api_reference_notebook.ipynb)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Contact

Maintainer: Hamza Usta

Email: <hamzausta2222@gmail.com>
