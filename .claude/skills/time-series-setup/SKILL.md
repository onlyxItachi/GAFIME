---
name: time-series-setup
description: Configure the GAFIME v1 row-ordered time-series generated family with truthful ordering, grouping, candidate-cap, and scoring-placement constraints.
metadata:
  audience: end-user
---

# Time-Series Setup

Inspect the file when available:

```bash
python .claude/skills/time-series-setup/scripts/detect_time_structure.py \
  data.parquet --target target
```

The detector identifies possible time/group columns and recommends row-unit lags
and windows. Pass the known target explicitly so descriptor estimates do not
count it as an input feature; target-name inference is only a hint. GAFIME does
not accept a time column or group column in
`EngineConfig`: it consumes the supplied row order. Sort rows before analysis,
and run separate entity groups or otherwise partition them so lag and rolling
windows never cross group boundaries.

The v1 family generates lag, delta, velocity, acceleration, rolling mean,
rolling standard deviation, and rolling sum columns in `gafime_cpu`. The
selected Core/CUDA/ROCm backend, or Metal for `fp32` only, then scores the
expanded continuous matrix. Graph capture never includes the expansion step.

```python
from gafime import ComputeBudget, EngineConfig, GafimeEngine

config = EngineConfig(
    backend="auto",
    precision="mixed",
    metric_names=("pearson", "r2"),
    enable_time_series_functions=True,
    time_series_lags=(1, 2, 4, 8, 16),
    time_series_windows=(4, 8, 16, 32),
    budget=ComputeBudget(
        max_comb_size=2,
        max_time_series_candidates=100_000,
        top_k_features_for_time_series=50,
    ),
)
report = GafimeEngine(config).analyze(X_train, y_train, feature_names)
```

Lags and windows are row counts. Reject zero lags, windows below two, and values
that leave no valid support. Keep generation inside each training fold to avoid
future leakage. Candidate volume is approximately
`source_features * (4 * lags + 3 * windows)` before row-validity and configured
caps.
