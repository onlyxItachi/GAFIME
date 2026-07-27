---
name: interpret-results
description: Interpret a live GAFIME v1 DiagnosticReport or an explicit to_dict export while preserving backend, family, stability, and significance evidence boundaries.
---

# Interpret GAFIME Results

Prefer live report properties:

```python
top = report.interactions.top_k(10, metric_name="pearson")
print(report.configured_backend)
print(report.backend.selected_backend, report.backend.execution_placement)
print(report.decision)
for item in top:
    print(item.candidate_id, item.family, item.expression, item.metrics)
```

`DiagnosticReport.to_dict()` materializes the native report and is intended only
as an explicit export boundary. For a user-supplied export, run:

```bash
python .claude/skills/interpret-results/scripts/explain_report.py report.json
```

Join interaction, stability, and permutation rows by `candidate_id`, not only by
the feature tuple. Explain `interaction`, `time_series`, and `decision_path`
expressions according to their family. Report both the configured and selected
backend; `auto` is a policy request, not the execution placement.

Keep interpretation conservative:

- Pearson/Spearman direction and magnitude are descriptive, not model utility.
- R2 and mutual information are nonnegative and have different scales.
- Stability rows exist only when repeats were requested. They measure bootstrap
  metric variability conditional on an already-selected candidate using the
  same rows; they are not out-of-sample evidence and do not correct selection
  bias.
- Permutation rows exist only when supported and requested; absence never means
  significant.
- Decision-path permutation significance is unavailable in v1, while bootstrap
  stability is supported.
- `signal_detected` applies configured strength, stability, and significance
  policy; it does not replace held-out model validation.

Always include warnings and recommend untouched holdout or nested
cross-validation before production use.
