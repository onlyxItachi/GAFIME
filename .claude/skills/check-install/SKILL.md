---
name: check-install
description: Verify a GAFIME v1 installation, native boundary, backend capability reporting, optional payload distributions, starter notebook, and a small Core analysis.
metadata:
  audience: both
---

# Installation Health Check

Run the repository health check from the GAFIME checkout:

```bash
python .claude/skills/check-install/scripts/health_check.py
```

The helper routes mutable publication state to `docs/releases/STATUS.md`,
GitHub Releases, and PyPI. It validates either a published installation or the
repository development environment without embedding a self-invalidating
release-status claim.

The script checks:

- a release-supported CPython 3.10 through 3.14 interpreter and the installed
  GAFIME version;
- exact-version alignment of every installed `gafime-cuda` or `gafime-rocm`
  payload with the installed `gafime` Core distribution, plus imported-runtime
  version agreement with Core metadata;
- the public `backend_capabilities("auto", probe=True)` boundary;
- the continuous, decision-path, and time-series family registry, including
  decision-path permutation maxT with per-permuted-target path rediscovery;
- small end-to-end Rust Core analyses for `fp32`, `mixed`, and `fp64`;
- generation of the current v1 starter notebook;
- required Polars and optional scikit-learn imports.

Treat `SKIP` for scikit-learn as expected unless the user needs
`GafimeSelector`. Install a published prerelease with
`pip install --pre "gafime[sklearn]" "polars>=1.3,<2"`, or pin an exact
published version. Any `FAIL` is an installation or packaging problem and makes
the health check return nonzero.

Install Core and CUDA at the same exact version for
visible NVIDIA hardware and provide a compatible system CUDA 13 runtime:
`pip install --pre gafime gafime-cuda "polars>=1.3,<2"`.
For supported AMD hardware on Linux x86_64, install Core and ROCm at the same
exact version. PyPI provides the buildable ROCm
sdist, which needs the matching ROCm/HIP development toolchain; the matching
GitHub Release is the prebuilt thin raw-Linux wheel channel. Both require the
compatible system ROCm runtime. Use
`pip install --pre gafime gafime-rocm "polars>=1.3,<2"` for the PyPI source
path, or pin both projects to one exact published version.
Metal is bundled in the macOS arm64 Core wheel and supports
`precision="fp32"` only. Backend presence must still be confirmed by the runtime
capability probe; hardware names and package presence alone are not proof.

Do not recommend removed v0.4 discrete settings. The v1 generated families are
`time_series` and `decision_path`. Decision-path permutation significance is
supported only through full per-permuted-target path rediscovery.
