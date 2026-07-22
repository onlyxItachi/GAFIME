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
  --output gafime_pipeline.py
```

`auto` uses a scikit-learn-only default: logistic regression for classification
and Ridge for regression. XGBoost and CatBoost remain explicit optional choices.
The generated `GafimeSelector` is inside the sklearn `Pipeline`, so discovery is
refit on each cross-validation training fold rather than leaking held-out rows.

This helper covers continuous pair interactions materialized with `multiply`,
`add`, `subtract`, or `divide`. It does not materialize the generated
`time_series` or `decision_path` families into sklearn columns. For those
families, run `GafimeEngine` in an explicit training-fold discovery stage and
design a separate, reviewed materialization boundary. Decision-path discovery
must use `permutation_tests=0`; bootstrap stability remains available.

Require `pip install "gafime[sklearn]"`. Before presenting a generated script,
run `python -m py_compile` on it and confirm the selected third-party model is
installed. Never describe `backend="auto"` as guaranteed GPU execution; report
the selected backend from the fitted GAFIME report or capability probe.
