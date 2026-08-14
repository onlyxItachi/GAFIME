---
name: check-install
description: Verify a GAFIME v1 installation, native boundary, backend capability reporting, optional payload distributions, starter notebook, and a small Core analysis.
---

# Installation Health Check

Run the repository health check from the GAFIME checkout:

```bash
python .claude/skills/check-install/scripts/health_check.py
```

Beta.2 is not yet published. The current helper reports
`release_status="not_yet_published"`. Run it from the repository development
environment until publication completes; exact beta.2 install commands below
are prospective.

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
`GafimeSelector`. Once beta.2 is published, install it with the exact command
`pip install "gafime[sklearn]==1.0.0b2"`. Any `FAIL` is an installation or
packaging problem and makes the health check return nonzero.

Once beta.2 is published, install Core and CUDA at the same exact version for
visible NVIDIA hardware and provide a compatible system CUDA 13 runtime:
`pip install "gafime==1.0.0b2" "gafime-cuda==1.0.0b2"`.
For supported AMD hardware on Linux x86_64, the published target will install
Core and ROCm at the same exact version. PyPI will provide the buildable ROCm
sdist, which needs the matching ROCm/HIP development toolchain; the matching
GitHub Release will carry the prebuilt thin raw-Linux wheel. Both require the
compatible system ROCm runtime. Once published, use
`pip install "gafime==1.0.0b2" "gafime-rocm==1.0.0b2"` for the PyPI source path.
Metal is bundled in the macOS arm64 Core wheel and supports
`precision="fp32"` only. Backend presence must still be confirmed by the runtime
capability probe; hardware names and package presence alone are not proof.

Do not recommend removed v0.4 discrete settings. The v1 generated families are
`time_series` and `decision_path`. Decision-path permutation significance is
supported only through full per-permuted-target path rediscovery.
