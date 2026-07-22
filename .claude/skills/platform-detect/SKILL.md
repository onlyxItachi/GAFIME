---
name: platform-detect
description: Inspect platform and visible accelerator hints, then recommend a truthful GAFIME v1 installation and backend configuration.
---

# Platform Detection

Run:

```bash
python .claude/skills/platform-detect/scripts/platform_detect.py
```

The script reports OS, CPU architecture, visible NVIDIA/ROCm hints, installed
GAFIME distribution versions, and the public `auto` capability probe when
GAFIME is installed. Hardware hints guide installation only; the validated
capability result is the authority for runtime selection.

Distribution policy:

- Core: Linux x86_64/aarch64, Windows x86_64/arm64, macOS arm64.
- CUDA payload: Linux and Windows x86_64 via `gafime[cuda]`.
- ROCm payload: Linux x86_64 via `gafime[rocm]`.
- Metal: bundled in the macOS arm64 Core wheel.
- Optional OptiX RT: separate `gafime-cuda-rt` artifact, not a standard PyPI
  dependency and not part of the normal release bundle.

Prefer a configuration that lets the native resolver validate the runtime:

```python
from gafime import ComputeBudget, EngineConfig, GafimeEngine

config = EngineConfig(
    backend="auto",
    metric_names=("pearson", "spearman", "mutual_info", "r2"),
    budget=ComputeBudget(max_comb_size=2, vram_budget_mb=6144),
)
engine = GafimeEngine(config)
```

Adjust `vram_budget_mb` from validated device capacity and the whole workload,
not only the raw matrix size. Explicit backends are appropriate when failure is
preferred to fallback. Do not recommend removed discrete-family options.
