---
name: validate-features
description: Validate selected GAFIME v1 continuous pair interactions on untouched data without confusing discovery, stability, permutation maxT, and downstream model evidence.
---

# Validate Discovered Features

For already selected continuous pair interactions, run:

```bash
python .claude/skills/validate-features/scripts/validate_features.py \
  --data data.parquet \
  --target target \
  --interactions '0,1;2,3' \
  --operator multiply
```

The helper measures train/holdout Pearson correlation, a bootstrap interval, and
a random-pair baseline. It validates only the supplied continuous pairs; it does
not rerun GAFIME discovery and must not be described as nested model validation.

For unbiased model evidence, place discovery and materialization inside every
training fold and reserve an untouched final test set. Report GAFIME's
family-wise permutation maxT separately from bootstrap stability and downstream
model score. A missing p-value is not evidence of significance.

Generated-family rules:

- time-series transforms must be generated in temporal order inside each
  training fold, with no entity-boundary crossing;
- decision paths depend on the target and must be rediscovered in each training
  fold;
- decision-path bootstrap stability is supported, but permutation significance
  is unavailable and positive `permutation_tests` fails closed.

Do not reuse removed v0.4 discrete-candidate helpers or thresholds. Prefer
candidate IDs and family names when joining interactions, stability, and
permutation rows because the same feature tuple can occur in different families.
