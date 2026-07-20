# GAFIME

![PyPI version](https://img.shields.io/pypi/v/gafime)
![Python Versions](https://img.shields.io/pypi/pyversions/gafime)
![License](https://img.shields.io/github/license/onlyxItachi/GAFIME)

GPU-Accelerated Feature Interaction Mining Engine.

GAFIME is a native feature interaction mining engine for tabular and structured
machine-learning workflows. Python exposes a thin public API; Rust owns
validation, planning, scheduling, CPU kernels, and report handles; CUDA, Metal,
and ROCm/HIP own their backend-local C ABI payloads.

The engine is built for workloads where interaction candidates, decision-path
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

Vendor GPU payloads are explicit in the v1 distribution design. Pip can select
wheels by Python, ABI, OS, and CPU architecture, but not by local GPU vendor.
CUDA and ROCm payloads therefore need explicit package selection once the split
payload packages are published:

```bash
pip install "gafime[cuda]"
pip install "gafime[rocm]"
```

The extras install the separate PyPI payload projects `gafime-cuda` and
`gafime-rocm` for the same GAFIME release.

Apple Silicon Metal follows the macOS arm64 wheel/platform path.

Detailed install and backend policy:

- [docs/backend-selection.md](docs/backend-selection.md)
- [docs/eager-resident-compiled-execution.md](docs/eager-resident-compiled-execution.md)
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

The repository also keeps historical API reference notebooks from earlier
release work. Regenerate the starter notebook with `gafime --init` for the
current public API.

- [docs/notebooks/gafime_full_api_reference_notebook.ipynb](docs/notebooks/gafime_full_api_reference_notebook.ipynb) (historical reference)

## Candidate Families

GAFIME supports:

- continuous interaction candidates,
- native decision-path candidates for threshold/region-like structure,
- explicit time-series candidates: lag, delta, velocity, acceleration, rolling
  mean, rolling std, and rolling sum,
- scikit-learn transformer integration through `gafime.sklearn.GafimeSelector`,
- native Arrow ingest through `gafime.dataload`.

The v0.4 discrete candidate family has been deprecated and removed from the
current engine path. Use decision-path candidates for tree-like threshold and
region structure.

## Backend Policy

`backend="auto"` ranks the available native execution paths:

1. configured GPU payloads whose C ABI library loads and whose `device_id`
   reports valid device info,
2. the Rust CPU vector path ranked by detected ISA
   (`AVX512 > AVX2 > SSE4.2/NEON`),
3. the scalar Rust CPU path.

Explicit `backend="cuda"`, `backend="rocm"`, and `backend="metal"` never fall
back to another backend. `auto` is the only mode that probes candidates and
selects the best available execution path.

`backend="gpu"` is rejected because it is ambiguous across CUDA, ROCm, and
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
CMake, GAFIME development/benchmark/scikit-learn dependencies, and the locally
staged `gafime-cuda` payload by default. Extra workstation packages can be
added with the `EXTRA_PIP_PACKAGES` Docker build argument. The Core smoke image
is a smaller CPU-native source-build check.

Docker details:

- [CONTRIBUTING.md](CONTRIBUTING.md)

## Project References

- [docs/releases/v0.4.7.md](docs/releases/v0.4.7.md)
- [docs/v0.4.7-rocm-native-backend.md](docs/v0.4.7-rocm-native-backend.md)
- [docs/backend-selection.md](docs/backend-selection.md)
- [docs/notebooks/gafime_full_api_reference_notebook.ipynb](docs/notebooks/gafime_full_api_reference_notebook.ipynb) (historical reference)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Contact

Maintainer: Hamza Usta

Email: <hamzausta2222@gmail.com>
