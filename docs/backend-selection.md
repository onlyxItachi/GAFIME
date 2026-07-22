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

NVIDIA CUDA install target on Linux x86_64 or Windows AMD64 once the split
payload package is published:

```bash
pip install "gafime[cuda]"
```

AMD ROCm/HIP install target on Linux x86_64 once the split payload package is
published:

```bash
pip install "gafime[rocm]"
```

Apple Silicon Metal is bundled with the macOS arm64 base wheel:

```bash
pip install gafime
```

The distribution target is:

```text
gafime       -> thin Python API, PyO3 boundary, Rust orchestration, Rust CPU kernels
gafime-cuda  -> CUDA native payload
gafime-rocm  -> ROCm/HIP native payload
gafime macOS arm64 wheel -> bundled Metal dylib and metallib
```

The extras are convenience aliases that depend on the matching payload package
at the exact base-package version. Their environment markers deliberately avoid
requesting a CUDA payload on ARM/macOS or a ROCm payload outside Linux x86_64.
The payload packages depend back on the same exact `gafime` version; pip can
resolve that same-version dependency cycle from a matching artifact set.

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
- `gafime-rocm` carries ROCm/HIP payload binaries for approved ROCm platforms;
  its only implemented wheel policy is the explicit, immutable `bundled` mode.
- the macOS arm64 base wheel carries exactly one paired Metal dylib and
  metallib under `gafime/_metal`.
- CUDA Linux, CUDA Windows, and ROCm Linux payload artifacts each contain one
  `cp310-abi3` wheel for Python 3.10 and newer on that payload/platform.

ROCm artifacts compile inside the EL8-based `manylinux_2_28` image using AMD's
pinned ROCm 7.2.3 repository. The release workflow runs
`auditwheel repair --plat manylinux_2_28_x86_64`; a wheel is not tagged or
uploaded as manylinux unless that repair succeeds.

The repaired wheel contains the pinned ROCm 7.2.3 userspace closure, but never
the kernel driver or hardware support. Mixed use with another ROCm userspace in
the same process is unsupported. See
[rocm-wheel-policy.md](rocm-wheel-policy.md) for the exact component, SBOM,
ELF-closure, size, driver, and diagnostics contract.

Windows ROCm/HIP packaging remains gated by repeatable HIP SDK CI support and
must be documented before release.

## Runtime Priority

`backend="auto"` is a ranked resolver, not a fixed platform alias. It probes
only configured v1 payloads and accepts a GPU candidate only when the C ABI
library loads and the requested `device_id` returns valid `GafimeGpuDeviceInfo`.

Before that native resolver runs, Python applies this deterministic discovery
policy:

1. A present `GAFIME_CUDA_V1_LIB`, `GAFIME_ROCM_V1_LIB`, or
   `GAFIME_METAL_V1_LIB` is never changed.
2. Otherwise, Linux/Windows x86_64 discovers exactly one matching-version
   `gafime-cuda` or `gafime-rocm` package library for the requested backend.
3. macOS arm64 discovers the paired base-wheel Metal dylib and metallib.
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
  `gafime[rocm]`.
- ROCm requested on an unsupported platform: use `backend="core"` or install a
  supported ROCm/HIP payload for that platform.
- CUDA requested without the CUDA payload: install `gafime[cuda]`.
- Metal requested on macOS arm64 without the paired base-wheel artifacts:
  reinstall the matching `gafime` wheel.
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
