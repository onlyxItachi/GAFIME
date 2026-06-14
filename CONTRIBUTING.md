# Contributing to GAFIME

GAFIME is a multi-language native project. Python provides the public API,
C++ Core owns CPU-native execution, Rust owns scheduling/orchestration helpers,
and CUDA, Metal, and ROCm/HIP own GPU execution paths.

This guide is for source development. Release tagging and publication are
maintainer-controlled and should not be started without explicit approval.

## Local Development

Use a project-local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel pybind11 cmake
python -m pip install -e ".[dev,sklearn]"
```

With `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,sklearn]" --no-build-isolation
```

Useful build controls:

- `GAFIME_SKIP_CUDA=1`: skip CUDA build.
- `GAFIME_SKIP_ROCM=1`: skip ROCm/HIP build.
- `STRICT_CPU=1`: fail if C++ Core or Rust subfunctions cannot build.
- `STRICT_CUDA=1`: fail if CUDA cannot build.
- `STRICT_ROCM=1`: fail if ROCm/HIP cannot build.
- `GAFIME_ROCM_ARCHS=<rocm-offload-target>[,<rocm-offload-target>...]`:
  explicit HIP targets.

Examples:

```bash
GAFIME_SKIP_CUDA=1 GAFIME_SKIP_ROCM=1 STRICT_CPU=1 \
  uv pip install --python .venv/bin/python -e . --no-build-isolation

GAFIME_SKIP_CUDA=1 GAFIME_ROCM_ARCHS=<rocm-offload-target> STRICT_CPU=1 \
  uv pip install --python .venv/bin/python -e . --no-build-isolation
```

## Verification

Run focused checks before committing native changes:

```bash
python -m gafime --check
python -m pytest tests/test_v045_native_spine.py -q
python -m pytest tests/test_rocm_native_backend.py -q
```

ROCm and CUDA tests that require device access may skip in restricted
environments. Do not count sandbox-only device skips as hardware validation.

## Developer Docker Images

Docker is for maintainers and contributors who need reproducible source-build
environments. Docker images in this repository are not distribution images.
Normal users should install GAFIME from PyPI wheels.

CUDA development environment:

```bash
docker compose run --build gafime-cuda-dev
```

CPU/Core smoke environment:

```bash
docker compose run --build gafime-core-smoke
```

The CUDA developer image includes:

- CUDA Toolkit 13.2 base image,
- compiler toolchain,
- Rust,
- CMake/Ninja,
- Python development headers,
- GAFIME `dev`, `sklearn`, and `bench` Python dependencies.
- locally staged `gafime-cuda` payload installation by default.

Extra workstation packages can be layered without editing the Dockerfile:

```bash
docker build \
  --build-arg EXTRA_PIP_PACKAGES="torch" \
  -t gafime:cuda-dev .
```

The Core smoke image skips CUDA and ROCm and verifies the C++ Core/Rust path.

## Source Layout

- `gafime/`: Python public API and backend wrappers.
- `gafime_core/`: C++ Core CPU backend and SIMD dispatch.
- `src/cuda/`: CUDA kernels.
- `src/metal/`: Apple Metal backend.
- `src/rocm/`: ROCm/HIP kernels.
- `src/cpu/gafime_cpu/`: Rust subfunctions and scheduling helpers.
- `.claude/skills/`: maintainer/agent helper skills.
- `docs/`: release notes, backend notes, validation logs, and reference docs.

## Release Safety

Do not create tags, push release tags, enable release workflows, or publish to
PyPI without maintainer approval.

The wheel workflow may be intentionally disabled during backend development.
Check `.github/workflows/build_wheels.yml` before assuming tag pushes will
publish artifacts.
