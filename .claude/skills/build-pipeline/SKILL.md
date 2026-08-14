---
name: build-pipeline
description: Generate a leakage-safe scikit-learn pipeline using the v1 GafimeSelector continuous pair-interaction transformer.
---

# Build ML Pipeline

Collect task type, data path, target, model, top interaction count, operator,
and ranking metric. Then run:

```bash
python .claude/skills/build-pipeline/scripts/generate_pipeline.py \
  --task classification \
  --data data.parquet \
  --target churn \
  --model auto \
  --k 10 \
  --metric pearson \
  --precision mixed \
  --output gafime_pipeline.py
```

`auto` uses a scikit-learn-only default: logistic regression for classification
and Ridge for regression. XGBoost and CatBoost remain explicit optional choices.
The generated `GafimeSelector` is inside the sklearn `Pipeline`, so discovery is
refit on each cross-validation training fold rather than leaking held-out rows.
Native GAFIME owns candidate discovery. The sklearn compatibility transformer
materializes only the selected columns at the Python/sklearn boundary; use the
top-level native engine for scalable production analysis rather than expanding
candidate loops in Python.

This helper covers continuous pair interactions materialized with `multiply`,
`add`, `subtract`, or `divide`. It does not materialize the generated
`time_series` or `decision_path` families into sklearn columns. For those
families, run `GafimeEngine` in an explicit training-fold discovery stage and
design a separate, reviewed materialization boundary. Decision-path discovery
must remain fold-local. Decision-path permutation maxT is supported only by
rediscovering paths for every permuted target; never reuse target-dependent
paths across folds or permutations.

Beta.2 is not yet published. Run this helper from the repository development
environment and do not present its target package as currently available from
PyPI. Once beta.2 is published, require the exact command
`pip install "gafime[sklearn]==1.0.0b2"`. Before presenting a generated script,
run `python -m py_compile` on it and confirm the selected third-party model is
installed. Never describe `backend="auto"` as guaranteed GPU execution; report
the selected backend from the fitted GAFIME report or capability probe. The
helper defaults to `precision="mixed"`; Metal is fp32-only and an explicit Metal
mixed/fp64 request must fail closed.
