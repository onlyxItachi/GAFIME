---
name: benchmark-vs-manual
description: Compare v1 continuous pair interactions against baseline and manually authored features with leakage-safe cross-validation.
---

# Benchmark Against Manual Features

Run the bounded predictive comparison:

```bash
python .claude/skills/benchmark-vs-manual/scripts/compare_approaches.py \
  --data data.parquet \
  --target target \
  --manual-features '0,1;2,3' \
  --task classification \
  --k 10 \
  --metric pearson
```

It compares baseline, manual, GAFIME, and combined feature sets under the same
cross-validation splitter and model. `GafimeSelector` remains inside each
pipeline, so candidate discovery uses only that fold's training rows.
Feature indices refer to the reported numeric, non-target `feature_names` order;
timestamp, identifier, and categorical columns are not silently cast to floats.

Keep the claim narrow: this measures downstream predictive score for the given
dataset, split, model, metric, operator, and seed. It is not a kernel benchmark,
hardware throughput proof, or evidence that one feature-engineering method is
universally better. Report fold values, mean, standard deviation, sample count,
feature count, selected backend policy, and all preprocessing.

The helper evaluates continuous pair interactions only. Generated time-series
and decision-path families need separate fold-local discovery and faithful
materialization. Decision-path permutation significance is unavailable; do not
invent p-values or reuse target-dependent paths across folds.
