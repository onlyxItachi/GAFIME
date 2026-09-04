---
name: dataset-profiler
description: Profile a CSV or Parquet dataset for GAFIME v1 data quality, candidate scale, candidate-row work, and conservative resident-input memory.
metadata:
  audience: end-user
---

# Dataset Profiler

Run:

```bash
python .claude/skills/dataset-profiler/scripts/profile_dataset.py data.parquet \
  --target target \
  --vram 8 \
  --precision mixed \
  --max-arity 3 \
  --max-combinations-per-arity 5000
```

The JSON output includes numeric columns, nulls, constant columns, a
precision-aware resident-input estimate, the combinatorial universe for each
arity, the planned cap, and candidate-row evaluations. `fp32` and `mixed` use
four-byte resident input; `fp64` uses eight-byte resident input. Arity is limited
to the v1 range `1..5`.

The resident-input estimate is not peak memory. It excludes backend workspaces,
MI histograms, descriptor/result storage, significance replay, generated-family
columns, and graph state. Never claim a workload fits merely because the raw
matrix fits. Use the selected backend's runtime memory admission and preserve
headroom.

For large feature counts, reason first about candidate count and
`rows * planned_candidates`, then tune `max_comb_size`,
`max_combinations_per_k`, and `top_features_for_higher_k`. Keep results bounded
instead of materializing one Python object per candidate. `GafimeStreamer` can
partition input files, but per-batch reports are not automatically equivalent to
one global ranking; define aggregation semantics explicitly.

Encode non-numeric columns before mining, remove constants, and decide how
non-finite values should be handled. Time-series and decision-path generation
have separate candidate and memory costs.
