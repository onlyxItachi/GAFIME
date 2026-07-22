---
name: check-install
description: Verify a GAFIME v1 installation, native boundary, backend capability reporting, optional payload distributions, starter notebook, and a small Core analysis.
---

# Installation Health Check

Run the repository health check from the GAFIME checkout:

```bash
python .claude/skills/check-install/scripts/health_check.py
```

The script checks:

- Python 3.10 or newer and the installed GAFIME version;
- exact versions of `gafime`, `gafime-cuda`, `gafime-rocm`, and optional
  `gafime-cuda-rt` distributions;
- the public `backend_capabilities("auto", probe=True)` boundary;
- the continuous, decision-path, and time-series family registry, including
  decision-path's permutation-significance exclusion;
- a small end-to-end Rust Core analysis;
- generation of the current v1 starter notebook;
- required Polars and optional scikit-learn imports.

Treat `SKIP` for scikit-learn as expected unless the user needs
`GafimeSelector`; install it with `pip install "gafime[sklearn]"`. Any `FAIL`
is an installation or packaging problem and makes the health check return
nonzero.

For visible NVIDIA hardware without `gafime-cuda`, install
`pip install "gafime[cuda]"`. For supported AMD hardware on Linux x86_64,
install `pip install "gafime[rocm]"`. Metal is bundled in the macOS arm64 Core
wheel. Backend presence must still be confirmed by the runtime capability probe;
hardware names and package presence alone are not proof.

Do not recommend removed v0.4 discrete settings. The v1 generated families are
`time_series` and `decision_path`; decision-path configurations must set
`permutation_tests=0`.
