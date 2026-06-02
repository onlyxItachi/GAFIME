---
name: benchmark-vs-manual
description: Compare GAFIME-discovered features against manually crafted features to show the value of automated feature interaction mining. Use when the user wants to benchmark GAFIME, compare automated vs manual feature engineering, justify using GAFIME, or says things like "is GAFIME better than manual features", "compare approaches", "benchmark against my features", "prove GAFIME works", or "GAFIME vs hand-crafted".
---

# Benchmark vs Manual Features

Compare GAFIME's automatically discovered features against a user's manually crafted features.

## Instructions

1. Ask the user for:
   - Their dataset (CSV/Parquet path)
   - Target column
   - Their manually crafted feature columns (or a script that generates them)
   - Evaluation metric preference (AUC-ROC, R-squared, RMSE)
   - Model to use (default: LogisticRegression for classification, LinearRegression for regression)

2. Run the comparison script:

   ```bash
   python .claude/skills/benchmark-vs-manual/scripts/compare_approaches.py \
       --data "<file_path>" \
       --target "<target_col>" \
       --manual-features "feat_a,feat_b,feat_c" \
       --task classification \
       --k 10
   ```

3. The script runs three experiments with 5-fold cross-validation:
   - **Baseline**: Original features only (no interactions)
   - **Manual**: Original + user's manually crafted features
   - **GAFIME**: Original + GAFIME's top-k discovered interactions

   For v0.4.5 native-spine release checks, also run the public Engine benchmark
   runner when appropriate:

   ```bash
   python tests/benchmark_v045_native_spine.py --n-samples 32768 --n-features 24 --max-comb-size 5 --backends cpu,cuda
   ```

   This compares end-to-end CPU/Core and CUDA Engine execution using the same
   public API workload. Keep LightGBM, CatBoost, and XGBoost as benchmark
   extras or local dev dependencies, not mandatory runtime requirements.

4. Present comparison results in a clear table:

   ```
   Approach          | AUC-ROC (mean +/- std)
   ------------------|----------------------
   Baseline          | 0.712 +/- 0.023
   Manual Features   | 0.745 +/- 0.019
   GAFIME (top 10)   | 0.761 +/- 0.017
   ```

5. Highlight insights:
   - Which GAFIME features the human missed
   - Whether the missed features are continuous interactions or discrete
     threshold/rectangle candidates
   - Which manual features GAFIME also found (overlap)
   - Whether combining both approaches gives the best result
   - Statistical significance of the difference

## v0.4.x Discrete Notes

- Use `EngineConfig(enable_discrete_functions=True)` to include discrete
  threshold and rectangle candidates.
- Discrete candidates follow the user's normal `metric_names`; there is no
  separate discrete metric.
- Default discrete ordering uses `discrete_ranking="split_aware"` so split,
  interval, and rectangle candidates are ranked by adaptive soft-binary mask MI,
  soft impurity reduction, and residual gain rather than Pearson alone. Use
  `discrete_ranking="metric"` only when the user explicitly wants report-metric
  ordering.
- For v0.4.5 benchmarks, report `mi_bins` as an adaptive maximum. The default
  is `96`, but small train folds use fewer bins automatically.
- Do not leak test-set information. Fit GAFIME and all discrete thresholds on
  training folds only.
- GPU backends require `discrete_mode="soft"`. Hard discrete mode is C++ Core
  only.

## Example

**User says:** "I've spent 2 days crafting features for churn prediction. Can GAFIME beat that?"

**Result:** "Your manual features improve AUC from 0.712 to 0.745. GAFIME's automated features reach 0.761. Combining both gives 0.768. GAFIME found 3 interactions you missed: log(recency) x frequency, sqrt(monetary) x tenure, and abs(velocity_30d) x churn_risk_score."
