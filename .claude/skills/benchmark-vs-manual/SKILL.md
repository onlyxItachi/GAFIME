---
name: benchmark-vs-manual
description: Compare v1 continuous pair interactions against baseline and manually authored features with leakage-safe cross-validation.
metadata:
  audience: end-user
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
  --metric pearson \
  --precision mixed
```

It compares baseline, manual, GAFIME, and combined feature sets under the same
cross-validation splitter and model. `GafimeSelector` remains inside each
pipeline, so candidate discovery uses only that fold's training rows.
The output routes mutable publication state to `docs/releases/STATUS.md`. If
the integration is missing, it keeps a `--pre` command under
`prerelease_install`; use an exact version from PyPI when reproducibility is
required or use the repository development environment.
Feature indices refer to the reported numeric, non-target `feature_names` order;
timestamp, identifier, and categorical columns are not silently cast to floats.
This is a bounded sklearn integration helper: native GAFIME performs candidate
discovery, while the compatibility transformer materializes only the selected
columns at the Python/sklearn boundary. Do not present it as the scalable
production data plane.

Keep the claim narrow: this measures downstream predictive score for the given
dataset, split, model, metric, operator, and seed. It is not a kernel benchmark,
hardware throughput proof, or evidence that one feature-engineering method is
universally better. Report fold values, mean, standard deviation, sample count,
feature count, selected backend capability probe, precision profile, seed, and
all preprocessing. A capability probe is evidence about backend availability;
it is not a kernel timing result.

The helper evaluates continuous pair interactions only. Generated time-series
and decision-path families need separate fold-local discovery and faithful
materialization. Decision-path permutation maxT is supported only through
per-permuted-target path rediscovery; do not reuse target-dependent paths across
folds or permutations.
