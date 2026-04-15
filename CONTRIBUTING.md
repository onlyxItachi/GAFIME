# Contributing to GAFIME

Thank you for your interest in contributing to GAFIME!
GAFIME's incredible performance stems from its multi-language architecture: C++, CUDA, Metal, Rust, and Python.

This guide explains how to compile the project locally from source for development.

## 1. Prerequisites

Before you can build GAFIME locally, you need a few core compilers:

1. **Python 3.10+**: Ensure you have Python installed.
2. **Rust Toolchain**: Install via `rustup` (<https://rustup.rs>).
3. **CMake**: Needed to orchestrate the native C++ build process.
4. *(Windows only)* **MSVC Build Tools**: GAFIME relies on the Microsoft Visual C++ compiler for native Windows builds.
5. *(Optional/CUDA)* **NVIDIA CUDA Toolkit (v12+)**: If you are modifying the `src/cuda/` kernels or want GPU acceleration locally on an NVIDIA system.

## 2. Local Installation (Editable Mode)

Because GAFIME bridges so many languages, we use a custom `setup.py` that handles C++ CMake orchestration, while Rust `setuptools-rust` bindings compile the memory layer.

To install the project in "editable" mode so your Python changes reflect instantly, clone the repo and run:

```bash
# It is highly recommended to use a virtual environment!
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install the build dependencies
pip install -r requirements.txt
pip install setuptools wheel pybind11 cmake setuptools-rust

# Install the engine natively
pip install -e .[dev,sklearn]
```

### Checking your build

Once it successfully compiles (this may take a minute or two initially as `cargo` downloads crates and `nvcc` compiles the PTX multi-arch kernels), verify it works by invoking the CLI:

```bash
gafime --init
pytest tests/
```

## 3. Directory Architecture

If you're modifying code, here is where things live:

* **`src/cuda/`**: Core C++ / CUDA `.cu` logic. This is where the extremely fast raw math operations occur.
* **`src/metal/`**: Apple Silicon GPU implementations.
* **`src/cpu/gafime_cpu/`**: The Rust crates handling CPU-bound OpenMP fallbacks, threaded orchestration, and the PyO3 bindings that connect Python to the native libraries via `src/cpu/gafime_cpu/src/ffi.rs`.
* **`gafime/`**: The Python front-end. The user-facing API classes (`GafimeEngine`, `GafimeSelector`).

## 4. Continuous Integration (Wheels)

GAFIME uses GitHub Actions and `cibuildwheel` to automatically compile native wheels for Linux (`manylinux_2_28`), Windows (`AMD64`), and macOS (`arm64`).

Check `.github/workflows/build_wheels.yml` if you need to trace how we inject the CUDA Toolkit on Ubuntu or use `delvewheel` to pack the Windows DLLs!
