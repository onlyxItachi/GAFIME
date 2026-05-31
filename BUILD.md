# GAFIME Build and Distribution Guide

## Wheel Architecture and Payloads

GAFIME is distributed via Python wheels containing pre-compiled native binaries for CPU, CUDA, and macOS Metal backends. Since Python dynamically loads the most optimal backend at runtime, **every wheel contains all relevant native payloads for its target OS**.

### Payloads Included

- **Windows / Linux (`x86_64`)**:
  - `gafime_cpu`: Rust+C++ fallback implementation (requires OpenMP bundle)
  - `gafime_cuda`: Main hardware-accelerated backend using NVIDIA CUDA
  - `gafime_core`: C++ pybind11 internal definitions
- **macOS (`arm64`)**:
  - `gafime_cpu`: Rust+C++ fallback implementation
  - `gafime_metal`: Apple Metal GPU implementation
  - `gafime_core`: C++ pybind11 internal definitions

## Building the Wheel Locally

To emulate the CI pipeline locally, ensure you have:

1. Python 3.10+
2. Optional but recommended: `cibuildwheel`
3. CUDA Toolkit 13.2 when building GPU wheels locally

```bash
pip install build wheel
python -m build --wheel
```

Alternatively, to build just the extensions for local development testing:

```bash
python setup.py build_ext --inplace
```

## CUDA Architecture Strategy (SASS vs PTX)

To provide maximum performance on Windows and Linux without requiring users to have the heavy NVIDIA CUDA Toolkit installed, the `gafime_cuda` backend is compiled statically using `-cudart static`.

We use a "Fat Bin" approach containing pre-compiled binaries (SASS) for all modern architectures, plus a dynamic forward-fallback (PTX):

- **`sm_75`** (Turing: RTX 20-series, T4)
- **`sm_80`** (Ampere: A100, A30)
- **`sm_86`** (Ampere: RTX 30-series, A40)
- **`sm_89`** (Ada Lovelace: RTX 40-series, L40)
- **`sm_90`** (Hopper: H100, H200)
- **`sm_100`** (Blackwell datacenter)
- **`sm_120`** (Blackwell consumer)
- **`compute_120`** (PTX fallback for forward-compatible Blackwell-class drivers)

This enables `pip install gafime` to work instantly on almost any modern workstation GPU or data-center accelerator without compilation delays at runtime.

## v0.4.x Local Development Notes

Discrete function kernels and Rust cache-local scheduling should be tested
locally before release builds:

```bash
python setup.py build_ext --inplace
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --manifest-path src/cpu/gafime_cpu/Cargo.toml
python -m pytest -q
```

GPU hard discrete mode is intentionally unsupported. CUDA and Metal discrete
paths use soft/vectorized gates only.

For v0.4.1, confirm the CUDA library exports
`gafime_discrete_selection_adaptive_cuda`; otherwise the Python backend will
fall back to corrected host-side selector scoring.
The adaptive CUDA selector should be launched from homogeneous Rust template
batches via `BatchScheduler.create_template_equation_batches`.

Do not start final wheel builds, version bumps, tags, or publication without
maintainer approval.

### v0.4.x CI Wheel Build Notes

The GitHub wheel workflow targets CUDA Toolkit 13.2.0 for Windows and Linux
wheel builds. Linux manylinux builds install `cuda-nvcc-13-2` and
`cuda-cudart-devel-13-2` from NVIDIA's RHEL 8 repository and symlink
`/usr/local/cuda` to `/usr/local/cuda-13.2`. Windows builds install CUDA 13.2.0
through the pinned `Jimver/cuda-toolkit` action and export the `v13.2` toolkit
path.

The workflow also runs on `feature/**` branches so pre-release wheel builds can
be tested before merging or tagging. Tag builds still publish releases and PyPI
packages through the existing guarded release jobs.

### Strict Validation in CI

When building wheels in CI, a strict verification script (`tests/test_distribution.py`) enforces that all dependencies and OS-specific libraries are correctly bundled:

- On Linux, `auditwheel` automatically bundles necessary shared objects into `gafime.libs`.
- On Windows, `delvewheel` embeds native runtime dependencies (like `vcomp140.dll` OpenMP runtimes).
- On macOS, `delocate` packages `.dylib` frameworks.

Setting `STRICT_CUDA=1` forces CI tests to instantly fail if the wheel is improperly built and missing its GPU acceleration runtime. `STRICT_CPU=1` verifies the fallback Rust and C++ components.
