# GAFIME — GPU-Accelerated Feature Interaction Mining Engine

[![Latest release](https://img.shields.io/github/v/release/onlyxItachi/GAFIME?include_prereleases&sort=semver)](https://github.com/onlyxItachi/GAFIME/releases)
[![PyPI version](https://img.shields.io/pypi/v/gafime)](https://pypi.org/project/gafime/)
[![Python versions](https://img.shields.io/pypi/pyversions/gafime)](https://pypi.org/project/gafime/)
[![V1 Contract Validation](https://github.com/onlyxItachi/GAFIME/actions/workflows/v1_contract_validation.yml/badge.svg?branch=main)](https://github.com/onlyxItachi/GAFIME/actions/workflows/v1_contract_validation.yml)
[![License](https://img.shields.io/github/license/onlyxItachi/GAFIME)](LICENSE)

Native feature-interaction discovery across Rust CPU/SIMD, CUDA, ROCm/HIP,
and Metal.

GAFIME searches continuous interactions, decision-path regions, and temporal
transforms for tabular and structured machine-learning workflows. Python is
the concise public declaration and reporting surface; Rust owns validation,
planning, scheduling, lifecycle, and Core execution; each native GPU payload
owns its device-local execution.

## Quick Start

Install the current published prerelease of Core, optionally with a matching
vendor payload:

```bash
python -m pip install --pre gafime
python -m pip install --pre gafime gafime-cuda  # Linux/Windows x86_64 + CUDA
python -m pip install --pre gafime gafime-rocm  # Linux x86_64 + system ROCm
```

Core never depends on a GPU payload. Payload packages require the exact matching
Core version. Apple Silicon Core wheels contain the Metal payload; there is no
standalone Metal distribution. See the live [release status](docs/releases/STATUS.md)
and [backend installation guide](docs/backend-selection.md) before choosing a
hardware package.

```python
from gafime import ComputeBudget, EngineConfig, GafimeEngine

X = [[float(i), float((i * 7) % 11), float((i % 5) - 2)] for i in range(64)]
y = [0.4 * row[0] * row[1] - 0.2 * row[2] for row in X]

config = EngineConfig(
    backend="auto",
    precision="mixed",
    metric_names=("pearson", "r2"),
    budget=ComputeBudget(max_comb_size=2),
    permutation_tests=0,
    num_repeats=1,
)
report = GafimeEngine(config).analyze(
    X, y, feature_names=["trend", "cycle", "offset"]
)
print(report.backend)
print(report.interactions.top_k(5, metric_name="pearson"))
```

## What GAFIME Supports

- Continuous unary and higher-order interaction candidates.
- Native decision-path candidates with target-rediscovered permutation maxT.
- Lag, delta, velocity, acceleration, and rolling time-series candidates.
- Eager, resident, and explicit compiled lifecycles.
- NumPy, Polars/Arrow, file-streaming, scikit-learn, and CLI integration.

| Backend | Distribution | Precision profiles |
|---|---|---|
| Rust Core/SIMD | `gafime` | `fp32`, `mixed`, `fp64` |
| CUDA | `gafime-cuda` | `fp32`, `mixed`, `fp64` |
| ROCm/HIP | `gafime-rocm` | `fp32`, `mixed`, `fp64` |
| Metal | embedded in macOS arm64 `gafime` | `fp32` only |

Explicit unsupported backend/profile requests fail closed. `backend="auto"`
selects only from available, compatible native paths; explicit vendor requests
never silently substitute Core. RT/OptiX remains experimental and local-only.

## Documentation

### Getting Started

- [Guided tutorial](docs/notebooks/gafime_tutorial.ipynb)
- [Practical usage guide](USAGE.md)

### API

- [Authoritative v1 API reference and cookbook](docs/notebooks/gafime_v1_api_reference.ipynb)
- [Machine-checked public API coverage](docs/public-api-coverage.md)

### Execution and Backends

- [Backend selection and installation](docs/backend-selection.md)
- [Capability reporting](docs/capabilities.md)
- [Precision contract](docs/precision-contract.md)
- [Eager, resident, and compiled execution](docs/eager-resident-compiled-execution.md)

### Architecture and Development

- [Documentation index](docs/README.md)
- [Normative v1 architecture contract](docs/contract.md)
- [Build guide](BUILD.md)
- [Contribution guide](CONTRIBUTING.md)

### Security

- [Security policy and private reporting](SECURITY.md)

## Releases

- [Current release-train status](docs/releases/STATUS.md)
- [Release index](docs/releases/README.md)
- [Changelog](CHANGELOG.md)

## License / Contact

GAFIME is licensed under [Apache-2.0](LICENSE).

Maintainer: Hamza Usta — <hamzausta2222@gmail.com>

Report suspected vulnerabilities privately through the process in
[SECURITY.md](SECURITY.md), not through a public issue.
