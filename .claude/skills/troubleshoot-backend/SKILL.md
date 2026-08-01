---
name: troubleshoot-backend
description: Diagnose GAFIME v1 Core, CUDA, ROCm, or Metal selection and payload-loading failures through the public capability API.
---

# Backend Troubleshooting

Run:

```bash
python .claude/skills/troubleshoot-backend/scripts/diagnose_backends.py
```

The JSON report probes `core`, `cuda`, `rocm`, `metal`, and `auto` through
`backend_capabilities(..., probe=True)`. It records package versions, platform,
whether explicit library overrides are present, selected backend, probe status,
runtime device facts, graph support, significance placement, MI policy, and
payload-discovery errors. It does not execute a scoring workload.

Interpret capability evidence literally:

- `runtime` is reported by an ABI-validated payload;
- `static` is checked-in Core policy;
- `unknown` means no compatible runtime observation, not that hardware is absent.

Common fixes:

- CUDA payload missing or damaged: `pip install --force-reinstall gafime gafime-cuda`.
- CUDA runtime load failure: verify system `libcudart.so.13` on Linux or
  `cudart64_13.dll` on Windows; the payload wheel does not vendor it.
- ROCm payload missing on Linux x86_64:
  `pip install --force-reinstall gafime gafime-rocm`.
- Metal payload missing on macOS arm64: reinstall `gafime`; the dylib and
  metallib are bundled in the Core wheel.
- Core/native boundary missing: reinstall `gafime` for the active Python and
  architecture.
- Version mismatch: install Core and vendor payload packages at the exact same
  release version.
- Linux ROCm permission failure: verify `/dev/kfd` and render-node access.
- Explicit environment path failure: inspect the named `GAFIME_*_V1_LIB`
  override and remove or correct it.

Explicit `cuda`, `rocm`, and `metal` requests never fall back. `backend="auto"`
is the only ranked resolver. `backend="gpu"` is rejected as ambiguous.

Generated-family candidate creation remains in `gafime_cpu`; CUDA, ROCm, and
Metal may score the expanded continuous matrix. Optional compact CUDA RT
decision-path scoring is available only when the validated payload and device
report it. Do not infer RT support from a GPU model name.
