# GAFIME Backend Selection and GPU Payload Packages

This document defines the v1 distribution policy for the Rust CPU/Core runtime,
CUDA, ROCm/HIP, and Metal backends.

## Why GPU Payload Packages Are Explicit

Python wheel installers select binary wheels using Python tag, ABI tag, and
platform tag. Those tags can distinguish macOS arm64, Linux x86_64, Windows
arm64, and similar platform targets. They do not encode the local GPU vendor.

That means pip can distinguish:

```text
macosx_..._arm64
manylinux_..._x86_64
win_arm64
```

but cannot safely choose between vendor GPU runtime payloads:

```text
Linux/Windows x86_64 + NVIDIA CUDA
Linux x86_64 + AMD ROCm
```

Extras also do not change wheel selection. They add optional dependencies for a
distribution. GAFIME therefore uses explicit vendor payload packages for GPU
runtime binaries rather than relying on hardware-dependent wheel selection.

## Install Commands

Beta.2 is not yet published. The commands below are the intended public install
surface once it is published. They use its exact PEP 440 version because an
unqualified requirement continues to prefer the latest stable release.

Core/native CPU install:

```bash
python -m pip install "gafime==1.0.0b2"
```

NVIDIA CUDA install target on Linux x86_64 or Windows AMD64 once the split
payload package is published and a compatible system CUDA 13 runtime is
available:

```bash
python -m pip install "gafime==1.0.0b2" "gafime-cuda==1.0.0b2"
```

The CUDA wheel contains only GAFIME binaries. It dynamically resolves
`libcudart.so.13` on Linux. On Windows, CUDA 13.3 links NVIDIA's shared-runtime
hybrid loader, which resolves the driver-provided `nvcudart_hybrid64.dll`. The
wheel does not vendor either runtime library.

AMD ROCm/HIP source install on Linux x86_64 with a compatible ROCm 7.2.x
development toolchain:

```bash
python -m pip install "gafime==1.0.0b2" "gafime-rocm==1.0.0b2"
```

The prebuilt thin ROCm wheel is attached to the matching GitHub Release because
PyPI rejects its truthful raw Linux platform tag. It requires a system-visible
`libamdhip64.so.7`.

Apple Silicon Metal:

```bash
python -m pip install "gafime==1.0.0b2"
```

The distribution target is:

```text
gafime       -> thin Python API, PyO3 boundary, Rust orchestration, Rust CPU kernels
gafime-cuda  -> CUDA native payload
gafime-rocm  -> system-runtime ROCm/HIP native payload
gafime (macOS arm64 wheel) -> Core plus Apple Silicon Metal dylib and metallib
```

Core intentionally exposes no CUDA or ROCm extras and has no dependency on
either payload project. Users select a vendor payload explicitly. Each payload
depends on the exact matching `gafime` version, so the dependency graph is
one-way and cannot become circular. Metal needs no second package because pip
already selects the macOS arm64 Core wheel.

PyPI treats these as three separate projects. The build workflow freezes the
complete release bundle; the separate publisher uses three ordered lanes:

```text
gafime       -> base/core distribution lane
gafime-cuda  -> CUDA payload distribution lane
gafime-rocm  -> ROCm source-distribution lane
```

Each PyPI project must have its own Trusted Publisher entry pointing at
`onlyxItachi/GAFIME`, workflow `.github/workflows/publish_release.yml`, and the
GitHub environment `pypi`. Core publishes first. CUDA and ROCm may publish only
after Core succeeds; public exact-version installs must then pass before the
GitHub Release is created.

Release-candidate artifact checks must confirm:

- base `gafime` wheels do not contain CUDA or ROCm payload files,
- the macOS arm64 `gafime` wheel contains exactly one paired Metal dylib and
  metallib while every other core wheel contains neither,
- `gafime-cuda` carries CUDA payload binaries only,
- `gafime-rocm` uses the explicit immutable `system` policy and its thin wheel
  carries no ROCm userspace;
- every Core, CUDA, and ROCm wheel carries `fp32`, `mixed`, and `fp64` in its
  existing binary; embedded Metal carries `fp32` only;
- no precision-specific distribution, extra, or wheel family exists;
- every platform/payload pair contains dedicated matching-ABI wheels for each
  manifest-declared CPython version;
- CUDA/ROCm payload metadata requires the exact Core version while Core
  metadata has no payload dependency or extra;
- no RT/OptiX source or output enters any distribution or release artifact.

ROCm artifacts compile inside the EL8-based `manylinux_2_28` image using AMD's
pinned ROCm 7.2.3 repository. The standard wheel is not repaired by
`auditwheel`: it retains a raw `linux_x86_64` tag, has no RPATH/RUNPATH, and
requires one coherent host ROCm 7.2.x runtime. The GitHub Release carries that
wheel; the ROCm PyPI lane carries only its source distribution. See
[rocm-wheel-policy.md](rocm-wheel-policy.md) for the exact ELF, size, runtime,
and diagnostics contract.

Windows ROCm/HIP packaging remains gated by repeatable HIP SDK CI support and
must be documented before release.

## Runtime Priority

`backend="auto"` is a ranked resolver, not a fixed platform alias. It probes
only configured v1 payloads and accepts a GPU candidate only when the C ABI
library loads, the requested `device_id` returns valid `GafimeGpuDeviceInfo`,
and the additive precision capability mask includes the requested profile.
Metal is excluded from `mixed` and `fp64` selection; Core remains eligible.

Before that native resolver runs, Python applies this deterministic discovery
policy:

1. A present `GAFIME_CUDA_V1_LIB`, `GAFIME_ROCM_V1_LIB`, or
   `GAFIME_METAL_V1_LIB` is never changed.
2. Otherwise, Linux x86_64 discovers exactly one matching-version
   `gafime-cuda` or `gafime-rocm` package library for the requested backend;
   Windows x86_64 discovers `gafime-cuda` only.
3. macOS arm64 discovers the paired dylib and metallib from the selected
   `gafime` core wheel under `gafime/_metal`.
4. Missing payload packages leave the existing native missing-payload error in
   place. Duplicate distributions, version mismatches, missing libraries, or
   multiple library candidates fail with a clear discovery error.

Default ranking:

| Rank | Candidate |
|---:|---|
| 1 | Usable GPU device payloads (`cuda`, `rocm`/`hip`, `metal`), scored by architecture class, discrete/integrated placement, high-bandwidth and unified-memory flags, memory capacity, multiprocessor count, and compute version |
| 2 | Rust CPU vector ISA (`AVX512 > AVX2 > SSE4.2/NEON`) |
| 3 | Rust scalar CPU |

Use explicit `backend="cuda"`, `backend="rocm"`, or `backend="metal"` when you
want to force a vendor-specific GPU backend. Explicit backend requests never
fall back to another backend.

## Strict Backend Errors

The resolver should fail clearly for impossible requests:

- CUDA requested on macOS: use `backend="metal"` or `backend="core"`.
- Metal requested outside macOS: use `backend="cuda"`, `backend="rocm"`, or
  `backend="core"` depending on installed payloads.
- ROCm requested without the ROCm payload on Linux x86_64: install
  `gafime` and the exact matching `gafime-rocm`.
- ROCm requested on an unsupported platform: use `backend="core"` or install a
  supported ROCm/HIP payload for that platform.
- CUDA requested without the CUDA payload: install `gafime` and the exact
  matching `gafime-cuda`.
- Metal requested on macOS arm64 without the paired base-wheel artifacts:
  install or reinstall the matching `gafime` macOS arm64 wheel.
- Metal requested with `precision="mixed"` or `precision="fp64"`: use Core,
  CUDA, or ROCm; Metal supports `fp32` only and never runs hidden CPU fp64 work.
- GPU payload installed but no compatible hardware/runtime is visible: fix the
  driver/runtime installation or use `backend="core"`.

`backend="gpu"` is rejected because it is ambiguous across CUDA, ROCm, and
Metal. Use `backend="auto"` or a vendor-specific backend name.

## Diagnostics

Use:

```bash
gafime --check
python .claude/skills/platform-detect/scripts/platform_detect.py
python .claude/skills/check-install/scripts/health_check.py
```

The skill scripts report installed payload distributions, visible vendor
hardware, and the recommended install/backend command.

Do not infer AMD runtime behavior from ROCm target names. ROCm target strings
are build/diagnostic metadata; runtime backend policy uses HIP capability flags
reported by the driver.
