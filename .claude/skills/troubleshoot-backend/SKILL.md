---
name: troubleshoot-backend
description: Diagnose GAFIME v1 Core, CUDA, ROCm, or Metal selection and payload-loading failures through the public capability API.
metadata:
  audience: both
---

# Backend Troubleshooting

Run:

```bash
python .claude/skills/troubleshoot-backend/scripts/diagnose_backends.py \
  --precision mixed
```

The JSON report probes `core`, `cuda`, `rocm`, `metal`, and `auto` through
`backend_capabilities(..., probe=True)`. It records package versions, platform,
whether explicit library overrides are present, selected backend, probe status,
runtime device facts, graph support, significance placement, MI policy,
precision, version-alignment warnings, and payload-discovery errors. The Metal
payload-health probe always uses its supported `fp32` profile; the requested
profile remains visible separately. The script does not execute a scoring
workload. It reports `release_status="see_docs_releases_status"`, routes
mutable publication state to `docs/releases/STATUS.md`, and offers current
prerelease commands under `prerelease_install`.

Interpret capability evidence literally:

- `runtime` is reported by an ABI-validated payload;
- `static` is checked-in Core policy;
- `unknown` means no compatible runtime observation, not that hardware is absent.

Common fixes (pin the current exact PyPI version when reproducibility is
required):

- CUDA payload missing or damaged: reinstall Core and CUDA at one explicit
  version:
  `pip install --pre --force-reinstall gafime gafime-cuda "polars>=1.3,<2"`.
- CUDA runtime load failure: verify system `libcudart.so.13` on Linux or
  driver-provided `nvcudart_hybrid64.dll` on Windows; the payload wheel does not
  vendor it.
- ROCm payload missing on Linux x86_64:
  install the exact matching Core and ROCm versions. PyPI provides the
  buildable ROCm sdist; the matching GitHub Release is the prebuilt thin
  raw-Linux wheel channel. Both require the compatible system ROCm runtime.
  The source command is
  `pip install --pre --force-reinstall gafime gafime-rocm "polars>=1.3,<2"`.
- Metal payload missing on macOS arm64: reinstall `gafime`; the dylib
  and metallib are bundled in the Core wheel.
- Core/native boundary missing: reinstall `gafime` for the active
  Python and architecture.
- Version mismatch: install Core and vendor payload packages at the exact same
  release version.
- Linux ROCm permission failure: verify `/dev/kfd` and render-node access.
- Explicit environment path failure: inspect the named `GAFIME_*_V1_LIB`
  override and remove or correct it.

Explicit `cuda`, `rocm`, and `metal` requests never fall back. `backend="auto"`
is the only ranked resolver. `backend="gpu"` is rejected as ambiguous. Metal
supports only `precision="fp32"`; mixed/fp64 requests fail closed.

Generated-family candidate creation remains in `gafime_cpu`; CUDA, ROCm, and
Metal may score the expanded continuous matrix within their supported precision
profiles. Standard release CUDA artifacts are always RT-free. Optional compact
CUDA RT decision-path scoring is a local CMake experiment selected through an
explicit validated library override only; it is never a distribution or release
artifact. Do not infer RT support from a GPU model name.
