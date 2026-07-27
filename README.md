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
GPU payloads therefore use explicit same-version package selection:

```bash
pip install "gafime[cuda]"
pip install "gafime[rocm]"
pip install "gafime[metal]"
```

The extras select the separate `gafime-cuda`, `gafime-rocm`, and
`gafime-metal` projects. Core wheels contain no vendor runtime payloads. CUDA
ships wheels on Linux/Windows x86_64; Metal ships an Apple Silicon macOS wheel.
The standard ROCm wheel is thin and requires system ROCm 7.2.x. Because its
truthful `linux_x86_64` tag is not accepted by PyPI, that wheel is attached to
the matching GitHub Release while PyPI carries the buildable ROCm source
distribution.

Each platform wheel uses the CPython 3.10 Stable ABI and is tested on CPython
3.10 through 3.14; one `cp310-abi3` filename represents that compatibility
range rather than Python-3.10-only support.

Detailed install and backend policy:

- [docs/backend-selection.md](docs/backend-selection.md)
- [docs/capabilities.md](docs/capabilities.md)
- [docs/rocm-wheel-policy.md](docs/rocm-wheel-policy.md)
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

```python
import gafime

gafime.generate_tutorial()
```

The generated notebook and the tracked practice notebook share the same v1
source. The repository also keeps a clearly labeled historical API notebook
from earlier release work.

- [docs/notebooks/gafime_tutorial.ipynb](docs/notebooks/gafime_tutorial.ipynb) (current v1 practice notebook)
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

Decision-path bootstrap stability is supported, but permutation significance
requires per-target path rediscovery and is not yet available. Set
`permutation_tests=0` when enabling decision-path generation; unsupported
permutation requests fail closed rather than reporting invalid p-values.
Bootstrap `stability_std` is variability conditional on an already-selected
candidate using the same rows; it is not out-of-sample evidence and does not
correct selection bias. Validate selected candidates on untouched data.

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

- [docs/releases/v1.0.0b1.md](docs/releases/v1.0.0b1.md)
- [docs/releases/v1.0.0b0.md](docs/releases/v1.0.0b0.md) (previous beta)
- [docs/releases/v1.0.0a0.md](docs/releases/v1.0.0a0.md) (previous alpha)
- [docs/capabilities.md](docs/capabilities.md)
- [docs/eager-resident-compiled-execution.md](docs/eager-resident-compiled-execution.md)
- [docs/backend-selection.md](docs/backend-selection.md)
- [docs/notebooks/gafime_tutorial.ipynb](docs/notebooks/gafime_tutorial.ipynb)
- [USAGE.md](USAGE.md)
- [docs/notebooks/gafime_full_api_reference_notebook.ipynb](docs/notebooks/gafime_full_api_reference_notebook.ipynb) (historical reference)
- [CONTRIBUTING.md](CONTRIBUTING.md)

Maintainer release operations are documented separately in
[docs/releases/release-operations.md](docs/releases/release-operations.md).
That runbook does not authorize publication; release tags and uploads require an
explicit maintainer decision from a fully validated commit.

Historical release records remain available under `docs/releases/`, including
[v0.4.7](docs/releases/v0.4.7.md) and
[v0.5.0-legacy](docs/releases/v0.5.0-legacy.md).

## Contact

Maintainer: Hamza Usta

Email: <hamzausta2222@gmail.com>
