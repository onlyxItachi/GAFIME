# GAFIME Build and Distribution Guide

## Wheel Architecture and Payloads

GAFIME separates the stable Python/Core API from vendor GPU runtime payloads.
This is required because Python wheel tags distinguish Python ABI, OS, and CPU
architecture, but not local GPU vendor. CUDA and ROCm Linux wheels are both
x86_64 platform artifacts from pip's point of view, so GAFIME must make vendor
GPU payloads explicit instead of relying on hardware-dependent wheel selection.

Distribution target for v0.4.7:

- `gafime`: Python API, C++ Core backend, Rust subfunctions, backend resolver.
- `gafime-cuda`: NVIDIA CUDA native payload.
- `gafime-rocm`: AMD ROCm/HIP native payload.

Convenience extras can point to the payload package for the same version:

```bash
pip install "gafime[cuda]"
pip install "gafime[rocm]"
```

Apple Silicon Metal remains selected by macOS arm64 platform wheels.

### Payloads Included

- **Windows / Linux (`x86_64`)**:
  - `gafime_cpu`: Rust helper/orchestration implementation
  - `gafime_core`: C++ pybind11 CPU backend with isolated SSE4.2/AVX2/AVX512 accumulation kernels
  - NVIDIA CUDA payloads are distributed through `gafime-cuda`
  - AMD ROCm/HIP payloads are distributed through `gafime-rocm`
- **Windows / Linux (`arm64` / `aarch64`)**:
  - `gafime_cpu`: Rust helper/orchestration implementation
  - `gafime_core`: C++ pybind11 CPU backend with isolated ARM64 NEON accumulation kernels
  - NVIDIA CUDA payloads are intentionally excluded from ARM wheels.
- **macOS (`arm64`)**:
  - `gafime_cpu`: Rust helper/orchestration implementation
  - `gafime_metal`: Apple Metal GPU implementation
  - `gafime_core`: C++ pybind11 CPU backend with isolated ARM64 NEON accumulation kernels

See [docs/backend-selection.md](docs/backend-selection.md) for resolver and
mixed CUDA/ROCm safety policy.

## Building the Wheel Locally

To emulate the CI pipeline locally, ensure you have:

1. Python 3.10+
2. Optional but recommended: `cibuildwheel`
3. CUDA Toolkit 13.2 when building the CUDA payload locally
4. ROCm/HIP toolchain when building the ROCm payload locally

```bash
pip install build wheel
python -m build --wheel
```

Alternatively, to build just the extensions for local development testing:

```bash
python setup.py build_ext --inplace
```

For local CUDA payload development builds:

```bash
uv pip install -e .
uv pip install -e packaging/gafime-cuda --no-build-isolation
```

For local ROCm/HIP payload development builds:

```bash
uv pip install -e .
GAFIME_ROCM_ARCHS=<gfx-target> uv pip install -e packaging/gafime-rocm --no-build-isolation
```

ROCm/HIP payload build controls:

- `GAFIME_ROCM_ARCHS=<gfx-target>[,<gfx-target>...]`: explicit HIP offload targets.
- Missing `hipcc` fails the `gafime-rocm` payload build.

The v0.4.7 ROCm path is explicit during development:
`backend="rocm"` or `backend="hip"`. The distribution policy is to keep ROCm in
a separate `gafime-rocm` payload. If CUDA and ROCm payloads are both installed,
`backend="auto"` defaults to CUDA and does not initialize ROCm unless requested.

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

This enables the CUDA payload package to work instantly on supported NVIDIA
workstations and data-center accelerators without compilation delays at runtime.

## CPU SIMD Safety Strategy

The C++ Core backend keeps memory ownership, pybind11 bindings, metric
orchestration, and interaction-vector construction in baseline common C++ code.
Vector accumulation kernels are separate translation units:

- `simd_scalar.cpp`
- `simd_x86_sse42.cpp`
- `simd_x86_avx2.cpp`
- `simd_x86_avx512.cpp`
- `simd_arm_neon.cpp`

Wheel builds must not use global `-march=native`, global AVX flags, or global
SVE/NEON flags. CMake applies x86 ISA flags only to the matching x86 source
file; ARM64 NEON is compiled only on ARM64 targets. Runtime dispatch selects the
best supported kernel and otherwise uses scalar fp32.

## v0.4.x Local Development Notes

Discrete function kernels and Rust cache-local scheduling should be tested
locally before release builds:

```bash
python setup.py build_ext --inplace
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --manifest-path src/cpu/gafime_cpu/Cargo.toml
python -m pytest -q
```

GPU hard discrete mode is intentionally unsupported. CUDA, Metal, and ROCm/HIP
discrete paths use soft/vectorized gates only.

For v0.4.1, confirm the CUDA library exports
`gafime_discrete_selection_adaptive_cuda`; the native-only spine must not rely
on a NumPy backend fallback.
The adaptive CUDA selector should be launched from homogeneous Rust template
batches via `BatchScheduler.create_template_batches`.

Do not start final wheel builds, version bumps, tags, or publication without
maintainer approval.

### v0.4.x CI Wheel Build Notes

The GitHub wheel workflow targets CUDA Toolkit 13.2.0 for x86_64 Windows and
x86_64 Linux wheel builds. Linux manylinux x86_64 builds install
`cuda-nvcc-13-2` and `cuda-cudart-devel-13-2` from NVIDIA's RHEL 8 repository
and symlink `/usr/local/cuda` to `/usr/local/cuda-13.2`. Windows x64 builds
install CUDA 13.2.0 through the pinned `Jimver/cuda-toolkit` action and export
the `v13.2` toolkit path.

ARM distribution wheels are built by separate jobs:

- `ubuntu-24.04-arm` -> `manylinux_2_28_aarch64`
- `windows-11-arm` -> `win_arm64`

Those jobs set `GAFIME_SKIP_CUDA=1` and `STRICT_CPU=1`, build Rust
orchestration plus the C++ Core NEON/scalar CPU backend, and verify that no
`gafime_cuda` / `libgafime_cuda` payload is present in the ARM wheel.

The workflow also runs on `feature/**` branches so pre-release wheel builds can
be tested before merging or tagging. Tag builds still publish releases and PyPI
packages through the existing guarded release jobs.

### Strict Validation in CI

When building wheels in CI, a strict verification script (`tests/test_distribution.py`) enforces that all dependencies and OS-specific libraries are correctly bundled:

- On Linux, `auditwheel` automatically bundles necessary shared objects into `gafime.libs`.
- On Windows, `delvewheel` embeds native runtime dependencies (like `vcomp140.dll` OpenMP runtimes).
- On macOS, `delocate` packages `.dylib` frameworks.

Setting `STRICT_CUDA=1` forces CI tests to instantly fail if an x86_64 GPU
wheel is improperly built and missing its GPU acceleration runtime.
`GAFIME_SKIP_CUDA=1` intentionally disables NVIDIA CUDA packaging for ARM
distribution wheels. `STRICT_CPU=1` verifies the Rust and C++ Core native
components.
