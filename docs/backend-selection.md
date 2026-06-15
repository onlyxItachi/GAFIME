# GAFIME Backend Selection and GPU Payload Packages

This document defines the v0.4.7 distribution policy for CPU/Core, CUDA, ROCm,
and Metal backends.

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
Linux/Windows x86_64 + AMD ROCm
```

Extras also do not change wheel selection. They add optional dependencies for a
distribution. GAFIME therefore uses explicit vendor payload packages for GPU
runtime binaries rather than relying on hardware-dependent wheel selection.

## Install Commands

Core/native CPU install:

```bash
pip install gafime
```

NVIDIA CUDA install target once the split payload packages are published:

```bash
pip install "gafime[cuda]"
```

AMD ROCm/HIP install target once the split payload packages are published:

```bash
pip install "gafime[rocm]"  # Linux x86_64 only in v0.4.7
```

The distribution target is:

```text
gafime       -> Python API, C++ Core backend, Rust subfunctions, resolver
gafime-cuda  -> CUDA native payload
gafime-rocm  -> ROCm/HIP native payload
```

The extras are convenience aliases that depend on the matching payload package
for the same release version.

PyPI treats these as three separate projects. Release publishing therefore uses
three independent lanes from the same GitHub workflow:

```text
gafime       -> base/core distribution lane
gafime-cuda  -> CUDA payload distribution lane
gafime-rocm  -> ROCm payload distribution lane
```

Each PyPI project must have its own Trusted Publisher entry pointing at
`onlyxItachi/GAFIME`, workflow `.github/workflows/build_wheels.yml`, and the
GitHub environment `pypi`. A failure in one payload lane must not block the
other projects from being published.

Release-candidate artifact checks must confirm:

- base `gafime` wheels do not contain CUDA or ROCm shared libraries,
- `gafime-cuda` carries CUDA payload binaries only,
- `gafime-rocm` carries Linux x86_64 ROCm/HIP payload binaries only in v0.4.7.
- CUDA Linux, CUDA Windows, and ROCm Linux payload artifacts each contain the
  Python 3.10 through 3.14 wheels for that payload/platform.

v0.4.7 intentionally ships ROCm payload wheels for Linux x86_64 only. Windows
ROCm/HIP packaging is deferred until the Windows HIP SDK distribution path is
stable enough for repeatable CI builds.

## Runtime Priority

Backend priority is determined before initializing a GPU runtime. GAFIME does
not probe every vendor runtime during `auto` resolution.

Default policy:

| Platform / installed payloads | `backend="auto"` priority |
|---|---|
| macOS arm64 | `metal -> core` |
| Linux/Windows x86_64 + CUDA payloads | `cuda -> core` |
| Linux x86_64 + ROCm payloads | `rocm -> core` |
| Linux/Windows ARM64 | `core` |

Use explicit `backend="cuda"` or `backend="rocm"` when you want to force a
vendor-specific GPU backend.

## Strict Backend Errors

The resolver should fail clearly for impossible requests:

- CUDA requested on macOS: use `backend="metal"` or `backend="core"`.
- Metal requested outside macOS: use `backend="cuda"`, `backend="rocm"`, or
  `backend="core"` depending on installed payloads.
- ROCm requested without the ROCm payload on Linux x86_64: install
  `gafime[rocm]`.
- ROCm requested on Windows: use `backend="core"` in v0.4.7; ROCm payload
  wheels are Linux-only in this release.
- CUDA requested without the CUDA payload: install `gafime[cuda]`.
- GPU payload installed but no compatible hardware/runtime is visible: fix the
  driver/runtime installation or use `backend="core"`.

`backend="gpu"` is deprecated because it is ambiguous across CUDA, ROCm, and
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
