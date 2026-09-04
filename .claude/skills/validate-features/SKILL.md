---
name: validate-features
description: Validate selected GAFIME v1 continuous pair interactions on untouched data without confusing discovery, stability, permutation maxT, and downstream model evidence.
metadata:
  audience: end-user
---

# Validate Discovered Features

For already selected continuous pair interactions, run:

```bash
python .claude/skills/validate-features/scripts/validate_features.py \
  --data data.parquet \
  --target target \
  --interactions '0,1;2,3' \
  --operator multiply \
  --precision mixed
```

The helper measures train/holdout Pearson correlation, a bootstrap interval, and
a random-pair baseline. Its `HEURISTIC_PASS` / `HEURISTIC_INCONCLUSIVE` labels
are descriptive checks, not proof that a feature is genuine or noise. It
validates only the supplied continuous pairs; it does not rerun GAFIME discovery
and must not be described as nested model validation.
Its pointwise and reduction dtypes follow the selected public precision profile;
this still remains a NumPy holdout diagnostic, not backend-parity evidence.
Pair indices refer to the numeric, non-target `feature_names` list emitted in the
result, not the original mixed-schema file column positions.

For unbiased model evidence, place discovery and materialization inside every
training fold and reserve an untouched final test set. Report GAFIME's
family-wise permutation maxT separately from bootstrap stability and downstream
model score. A missing p-value is not evidence of significance.

Generated-family rules:

- time-series transforms must be generated in temporal order inside each
  training fold, with no entity-boundary crossing;
- decision paths depend on the target and must be rediscovered in each training
  fold;
- decision-path bootstrap stability and permutation maxT are supported;
  permutation validation must rediscover paths for every permuted target.

Do not reuse removed v0.4 discrete-candidate helpers or thresholds. Prefer
candidate IDs and family names when joining interactions, stability, and
permutation rows because the same feature tuple can occur in different families.
