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

NVIDIA CUDA install target once the split payload packages are published:

```bash
pip install "gafime[cuda]"
```

AMD ROCm/HIP install target once the split payload packages are published:

```bash
pip install "gafime[rocm]"
```

The distribution target is:

```text
gafime       -> thin Python API, PyO3 boundary, Rust orchestration, Rust CPU kernels
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
- `gafime-rocm` carries ROCm/HIP payload binaries for approved ROCm platforms.
- CUDA Linux, CUDA Windows, and ROCm Linux payload artifacts each contain the
  Python 3.10 through 3.14 wheels for that payload/platform.

Windows ROCm/HIP packaging remains gated by repeatable HIP SDK CI support and
must be documented before release.

## Runtime Priority

`backend="auto"` is a ranked resolver, not a fixed platform alias. It probes
only configured v1 payloads and accepts a GPU candidate only when the C ABI
library loads and the requested `device_id` returns valid `GafimeGpuDeviceInfo`.

Default ranking:

| Rank | Candidate |
|---:|---|
| 1 | Usable GPU device payloads (`cuda`, `rocm`/`hip`, `metal`), scored by architecture class, discrete/integrated placement, high-bandwidth and unified-memory flags, memory capacity, multiprocessor count, and compute version |
| 2 | Rust CPU vector ISA (`AVX512 > AVX2 > SSE4.2/NEON`) |
| 3 | Rust scalar CPU |

Use explicit `backend="cuda"`, `backend="rocm"`, or `backend="metal"` when you
want to force a vendor-specific GPU backend. Explicit backend requests never
fall back to another backend.

## Family Routing Semantics (v1)

For candidate family dispatch, `gafime v1` routes public feature families to native
expansion/scoring on the selected backend, subject to installed payload
availability:

- `continuous` (base interactions): native continuous expansion on CPU, CUDA,
  ROCm/`hip`, and Metal when that backend candidate is usable.
- `time_series`: native time-series generation and scoring on CPU, CUDA, ROCm/`hip`,
  and Metal when that backend candidate is usable.
- `decision_path`: native decision-tree split/region generation and scoring on CPU,
  CUDA, ROCm/`hip`, and Metal when that backend candidate is usable.

`backend="auto"` probes usable GPU payloads first and only falls back to CPU paths
if no usable GPU candidate is found.

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
