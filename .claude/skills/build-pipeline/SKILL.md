---
name: build-pipeline
description: Generate a complete scikit-learn pipeline with GAFIME feature interaction mining. Use when the user wants to create an ML pipeline, integrate GAFIME with sklearn, build a classification or regression workflow, or says things like "build a pipeline", "create a model with GAFIME", "sklearn integration", "churn prediction pipeline", "set up classification", or "generate training script".
---

# Build ML Pipeline

Generate a complete, ready-to-run Python script that integrates GAFIME into a scikit-learn pipeline.

## Instructions

1. Ask the user for:
   - **Task type**: classification or regression
   - **Data source**: CSV/Parquet path, or they'll use in-memory data
   - **Target column**: name of the target variable
   - **Model preference**: LogisticRegression, RandomForest, XGBoost, CatBoost, or auto
   - **Number of top interactions** (`k`): how many GAFIME features to add (default: 10)

2. Run the pipeline generator:

   ```bash
   python .claude/skills/build-pipeline/scripts/generate_pipeline.py \
       --task classification \
       --data "data.parquet" \
       --target "churn" \
       --model auto \
       --k 10 \
       --output "gafime_pipeline.py"
   ```

3. The script generates a complete Python file with:
   - Data loading (Polars or Pandas)
   - Train/test split
   - `GafimeSelector` in an sklearn `Pipeline`
   - Cross-validation evaluation
   - Results reporting
   - Feature importance analysis

4. Review the generated script with the user and customize if needed:
   - Adjust `k` (number of interactions)
   - Change `operator` (multiply, add, subtract, divide)
   - Modify evaluation metric
   - Add preprocessing steps

5. Explain each section of the generated pipeline so the user understands what's happening.

## Example

**User says:** "Create a churn prediction pipeline using GAFIME and CatBoost"

**Actions:** Run generator with `--task classification --model catboost --target churn`

**Result:** A complete `gafime_pipeline.py` that loads data, discovers top 10 feature interactions with GAFIME, trains CatBoost, and reports AUC-ROC with cross-validation.
