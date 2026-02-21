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

- **`sm_80`** (Ampere: A100, RTX 30-series)
- **`sm_86`** (Ampere: RTX 30-series, A40)
- **`sm_89`** (Ada Lovelace: RTX 40-series, L40)
- **`sm_90`** (Hopper: H100)
- **`compute_90`** (PTX Fallback for future unlisted architectures like Blackwell `sm_100` or `sm_120`, assuming the driver JIT can compile PTX 90 to them).

This enables `pip install gafime` to work instantly on almost any modern workstation GPU or data-center accelerator without compilation delays at runtime.

### Strict Validation in CI

When building wheels in CI, a strict verification script (`tests/test_distribution.py`) enforces that all dependencies and OS-specific libraries are correctly bundled:

- On Linux, `auditwheel` automatically bundles necessary shared objects into `gafime.libs`.
- On Windows, `delvewheel` embeds native runtime dependencies (like `vcomp140.dll` OpenMP runtimes).
- On macOS, `delocate` packages `.dylib` frameworks.

Setting `STRICT_CUDA=1` forces CI tests to instantly fail if the wheel is improperly built and missing its GPU acceleration runtime. `STRICT_CPU=1` verifies the fallback Rust and C++ components.
