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
   - Which manual features GAFIME also found (overlap)
   - Whether combining both approaches gives the best result
   - Statistical significance of the difference

## Example

**User says:** "I've spent 2 days crafting features for churn prediction. Can GAFIME beat that?"

**Result:** "Your manual features improve AUC from 0.712 to 0.745. GAFIME's automated features reach 0.761. Combining both gives 0.768. GAFIME found 3 interactions you missed: log(recency) x frequency, sqrt(monetary) x tenure, and abs(velocity_30d) x churn_risk_score."
