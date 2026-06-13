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

but cannot safely choose between:

```text
Linux x86_64 + NVIDIA CUDA
Linux x86_64 + AMD ROCm
Linux x86_64 + both AMD and NVIDIA GPUs
```

Extras also do not change wheel selection. They add optional dependencies for a
distribution. GAFIME therefore uses explicit vendor payload packages for GPU
runtime binaries rather than relying on hardware-dependent wheel selection.

## Install Commands

Core/native CPU install:

```bash
pip install gafime
```

NVIDIA CUDA install:

```bash
pip install "gafime[cuda]"
```

AMD ROCm/HIP install:

```bash
pip install "gafime[rocm]"
```

The distribution target is:

```text
gafime       -> Python API, C++ Core backend, Rust subfunctions, resolver
gafime-cuda  -> CUDA native payload
gafime-rocm  -> ROCm/HIP native payload
```

The extras are convenience aliases that depend on the matching payload package
for the same release version.

## Runtime Priority

Backend priority must be determined before initializing a GPU runtime. GAFIME
should not probe CUDA and ROCm in the same `auto` path on mixed GPU systems.

Default policy:

| Platform / installed payloads | `backend="auto"` priority |
|---|---|
| macOS arm64 | `metal -> core` |
| Linux/Windows x86_64 + CUDA payload | `cuda -> core` |
| Linux x86_64 + ROCm payload | `rocm -> core` |
| Linux x86_64 + CUDA and ROCm payloads | `cuda -> core` |
| Linux/Windows ARM64 | `core` |

ROCm on a mixed AMD iGPU + NVIDIA dGPU system is explicit:

```python
from gafime import EngineConfig

config = EngineConfig(backend="rocm")
```

## Strict Backend Errors

The resolver should fail clearly for impossible requests:

- CUDA requested on macOS: use `backend="metal"` or `backend="core"`.
- Metal requested outside macOS: use `backend="cuda"`, `backend="rocm"`, or
  `backend="core"` depending on installed payloads.
- ROCm requested without the ROCm payload: install `gafime[rocm]`.
- CUDA requested without the CUDA payload: install `gafime[cuda]`.
- GPU payload installed but no compatible hardware/runtime is visible: fix the
  driver/runtime installation or use `backend="core"`.

`backend="gpu"` is deprecated because it is ambiguous across CUDA, ROCm, and
Metal. Use `backend="auto"` or a vendor-specific backend name.

## Mixed AMD + NVIDIA Safety Rule

On systems with an AMD iGPU and NVIDIA dGPU, the safe default is CUDA. ROCm is
not initialized unless the user explicitly requests it.

Recommended resolver behavior:

```text
if both CUDA and ROCm payloads are installed:
    backend="auto" -> cuda -> core
    backend="rocm" -> initialize ROCm only
    backend="cuda" -> initialize CUDA only
```

This avoids mixed runtime initialization and keeps the user's control explicit.

## Diagnostics

Use:

```bash
gafime --check
python .claude/skills/platform-detect/scripts/platform_detect.py
python .claude/skills/check-install/scripts/health_check.py
```

The skill scripts report installed payload distributions, visible vendor
hardware, and the recommended install/backend command.
