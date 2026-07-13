# GAFIME Build and Distribution Guide

## Wheel Architecture and Payloads

GAFIME v1 separates the stable Python/Rust API from vendor GPU runtime payloads.
This is required because Python wheel tags distinguish Python ABI, OS, and CPU
architecture, but not local GPU vendor. CUDA and ROCm Linux wheels are both
x86_64 platform artifacts from pip's point of view, so GAFIME makes vendor GPU
payloads explicit instead of relying on hardware-dependent wheel selection.

Distribution target for v1:

- `gafime`: thin `python/gafime` package plus the PyO3/Rust boundary.
- `gafime-cuda`: NVIDIA CUDA native payload built from `src/cuda`.
- `gafime-rocm`: AMD ROCm/HIP native payload built from `src/rocm`.
- macOS arm64 `gafime`: the base wheel also carries
  `gafime/_metal/libgafime_metal_v1.dylib` and its paired
  `gafime/_metal/gafime_metal_v1.metallib`.

Convenience extras can point to the payload package for the same version:

```bash
pip install "gafime[cuda]"
pip install "gafime[rocm]"
```

`gafime[cuda]` resolves the payload only on Linux x86_64 and Windows AMD64.
`gafime[rocm]` resolves it only on Linux x86_64; Windows ROCm is not a release
install target until a repeatable payload lane exists. Apple Silicon Metal is a
native C ABI payload built from `src/metal` and bundled in the macOS arm64 base
wheel, not a fourth vendor package.

CUDA and ROCm payload distributions use a minimal CPython 3.10 stable-ABI
module solely to obtain a platform wheel. Each supported platform therefore
builds one `cp310-abi3` payload wheel that installs on Python 3.10 and newer;
the vendor library itself is not rebuilt once per Python minor version.

### Payloads Included

- **Windows / Linux (`x86_64`)**:
  - Rust/PyO3 `gafime.gafime_py`: Python boundary, orchestration, CPU kernels, and C ABI loaders
  - NVIDIA CUDA payloads are distributed through `gafime-cuda`
  - AMD ROCm/HIP payloads are distributed through `gafime-rocm`
- **Windows / Linux (`arm64` / `aarch64`)**:
  - Rust/PyO3 `gafime.gafime_py`
  - NVIDIA CUDA payloads are intentionally excluded from ARM wheels.
- **macOS (`arm64`)**:
  - Rust/PyO3 `gafime.gafime_py`
  - Apple Metal payload from `src/metal`

See [docs/backend-selection.md](docs/backend-selection.md) for resolver and
payload package policy.

## Building the Wheel Locally

To emulate the CI pipeline locally, ensure you have:

1. Python 3.10+
2. `maturin`
3. CUDA Toolkit 13.3 when building the CUDA payload locally
4. ROCm/HIP toolchain when building the ROCm payload locally

```bash
python -m pip install maturin
maturin build --release
```

For editable local development:

```bash
maturin develop
```

For local CUDA payload development builds:

```bash
uv pip install -e .
python .github/scripts/stage_gpu_payload.py cuda payload-src/gafime-cuda
uv pip install -e payload-src/gafime-cuda --no-build-isolation
```

For local ROCm/HIP payload development builds:

```bash
uv pip install -e .
python .github/scripts/stage_gpu_payload.py rocm payload-src/gafime-rocm
GAFIME_ROCM_ARCHS=<rocm-offload-target> uv pip install -e payload-src/gafime-rocm --no-build-isolation
```

ROCm/HIP payload build controls:

- `GAFIME_ROCM_ARCHS=<rocm-offload-target>[,<rocm-offload-target>...]`:
  explicit HIP offload targets.
- Missing `hipcc` fails the `gafime-rocm` payload build.
- Runtime selection remains explicit: `backend="rocm"` or `backend="hip"` loads
  only the approved ROCm/HIP C ABI payload and must not fall back silently.

Release ROCm wheels are compiled inside the `manylinux_2_28` EL8 baseline with
AMD's matching ROCm 7.2 repository. The source wheel remains an ordinary Linux
wheel until `auditwheel repair --plat manylinux_2_28_x86_64` inspects it,
bundles the required ROCm runtime libraries, and emits the final tag. Do not
set a `bdist_wheel` platform name to claim compatibility.

For local macOS arm64 base-wheel staging:

```bash
python .github/scripts/stage_metal_payload.py
maturin build --release
```

The helper builds `src/metal` through CMake and stages the dylib/metallib under
`python/gafime/_metal` for that wheel build. They are generated release inputs,
not tracked binaries.

## Developer Docker Images

Docker files in this repository are source-build development environments, not
distribution images.

```bash
docker compose run --build gafime-cuda-dev
docker compose run --build gafime-core-smoke
```

`gafime-cuda-dev` uses the CUDA 13.2 development image, installs the base
package from source, stages the local CUDA payload with
`.github/scripts/stage_gpu_payload.py`, and installs that payload without
fetching a published wheel. Set `INSTALL_CUDA_PAYLOAD=0` at build time if you
only want the base package inside the CUDA toolchain image.

CUDA payload translation units use C++20. The template-specialized kernels do
not require C++23, and C++20 keeps the CUDA 13.2 payload build compatible with
both supported Linux host compilers and Visual Studio 2022 on Windows.

`gafime-core-smoke` skips CUDA and ROCm, builds the base package, and runs a
small Rust/PyO3 CPU smoke test.

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

The staged CUDA package always compiles `kernels.cu`, `rt_kernels.cu`,
`launcher.cu`, and `rt_launcher.cu`. Distributed payload builds use
`GAFIME_CUDA_RT_BUILD_MODE=off` by default, so compiling the explicit RT source
does not enable OptiX. An OptiX-enabled single payload requires
`GAFIME_CUDA_RT_BUILD_MODE=on` plus a configured OptiX include directory. The
separate `both` RT artifact policy remains in `src/cuda/CMakeLists.txt` and is
not silently folded into the default payload. The staged NVCC build uses
relocatable device code because template specializations are defined in the
kernel translation unit and launched from the separate launcher translation
unit, matching CMake's separable-compilation contract.

## CPU SIMD Safety Strategy

Rust owns CPU execution in `crates/gafime-cpu`. Baseline planning,
orchestration, report construction, and backend selection stay in safe Rust.
ISA-specific kernels live behind the safe dispatch API in
`crates/gafime-cpu/src/dispatch.rs`.

Wheel builds must not use global `-march=native`, global AVX flags, or global
SVE/NEON flags. Runtime dispatch selects the best supported kernel and otherwise
uses scalar fp32. Unsafe Rust is allowed only for tightly scoped SIMD lowering,
compiler intrinsics, or unavoidable ABI shims, and must remain behind a safe API.

## v1 Local Development Notes

Native Rust planning, CPU kernels, GPU C ABI launchers, and public top-level
Python API behavior should be tested locally before release builds:

```bash
cargo test --workspace
PYTHONPATH="$PWD/python" python -m pytest tests/python -q
PYTHONPATH="$PWD/python" python tests/release_measure/contract_00_policy_files.py
PYTHONPATH="$PWD/python:$PWD/tests/release_measure" python tests/release_measure/contract_01_top_level_numpy_parity.py
PYTHONPATH="$PWD/python:$PWD/tests/release_measure" python tests/release_measure/contract_02_feature_generation_reference.py
python tests/release_measure/v1_architecture_gate.py
```

When CUDA and ROCm payloads are available, rebuild them and run:

```bash
python tests/release_measure/v1_architecture_gate.py --include-gpu
PYTHONPATH="$PWD/python:$PWD/tests/release_measure" python tests/release_measure/backend_02_cross_backend_parity.py
PYTHONPATH="$PWD/python:$PWD/tests/release_measure" python tests/release_measure/backend_03_e2e_smoke_per_backend.py
```

Release validation focuses on continuous interactions, native decision paths,
time-series candidates, compile artifacts, backend graph launch paths, and
native compact report/export behavior through the top-level API.

Do not start final wheel builds, version bumps, tags, or publication without
maintainer approval.

### v1 CI Wheel Build Notes

The GitHub wheel workflow targets CUDA Toolkit 13.x for x86_64 Windows and
x86_64 Linux GPU payload builds. Linux manylinux x86_64 builds install the CUDA
compiler/runtime needed by the payload package. ROCm payloads compile in the
EL8-based `manylinux_2_28` image against the pinned ROCm 7.2.3 repository and
are repaired in that same baseline. Windows x64 CUDA builds install the pinned
CUDA Toolkit action and export the matching toolkit path.

ARM distribution wheels are built by separate jobs:

- `ubuntu-24.04-arm` -> `manylinux_2_28_aarch64`
- `windows-11-arm` -> `win_arm64`

Those jobs set `GAFIME_SKIP_CUDA=1` and `STRICT_CPU=1`, build Rust
orchestration plus the Rust CPU scalar/NEON path, and verify that no CUDA
payload is present in the ARM wheel.

The workflow runs on release tags and manual dispatch only. Release and PyPI
publication jobs remain guarded: all `workflow_dispatch` publish inputs default
to `false`, and a manual run does not publish unless its matching explicit input
is set. Core, CUDA, and ROCm PyPI jobs depend only on their own build,
validation, and source-distribution lanes; the GitHub Release job remains an
all-platform, all-payload gate. Publish inputs must not be enabled without
maintainer approval.

### Strict Validation in CI

When building wheels in CI, a strict verification script (`tests/test_distribution.py`) enforces that all dependencies and OS-specific libraries are correctly bundled:

- On Linux, `auditwheel` automatically bundles necessary shared objects into `gafime.libs`.
- On Windows, `delvewheel` embeds native runtime dependencies (like `vcomp140.dll` OpenMP runtimes).
- On macOS, `delocate` packages `.dylib` frameworks.

Setting `STRICT_CUDA=1` forces CI tests to instantly fail if an x86_64 GPU
wheel is improperly built and missing its GPU acceleration runtime.
`GAFIME_SKIP_CUDA=1` intentionally disables NVIDIA CUDA packaging for ARM
distribution wheels. `STRICT_CPU=1` verifies the Rust/PyO3 CPU runtime path.

Payload artifact gates install the exact base and payload wheels outside the
checkout, compare distribution versions, exercise automatic discovery, inspect
the required C ABI exports without invoking CUDA/ROCm hardware execution, and
reject mixed vendor contents. The macOS arm64 gate additionally executes the
public `backend="metal"` API against the bundled payload on Apple hardware.
