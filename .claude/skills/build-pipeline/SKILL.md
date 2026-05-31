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
   - Whether to include v0.4.x discrete threshold/rectangle candidates

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

   Note: `GafimeSelector` is the sklearn transformer for pair interaction
   augmentation. If the user wants v0.4.x discrete functions, generate an
   explicit `GafimeEngine` training-fold step with
   `enable_discrete_functions=True`, then materialize selected discrete
   candidates with `evaluate_discrete_candidate`. Do not fit thresholds on the
   test fold.

4. Review the generated script with the user and customize if needed:
   - Adjust `k` (number of interactions)
   - Change `operator` (multiply, add, subtract, divide)
   - Add `enable_discrete_functions=True` in an engine feature-generation stage
     if threshold/rectangle signals are needed
   - Modify evaluation metric
   - Add preprocessing steps

## v0.4.x Discrete Engine Snippet

```python
from gafime import ComputeBudget, EngineConfig, GafimeEngine

config = EngineConfig(
    backend="auto",
    metric_names=("pearson",),
    mi_bins=96,
    enable_discrete_functions=True,
    discrete_mode="soft",
    discrete_ranking="split_aware",
    budget=ComputeBudget(
        max_discrete_candidates=20_000,
        top_k_features_for_discrete=24,
    ),
)
report = GafimeEngine(config).analyze(X_train, y_train)
```

On CUDA/Metal, keep `discrete_mode="soft"`. Hard discrete mode is CPU/NumPy
only. Keep the default `discrete_ranking="split_aware"` for threshold,
interval, and rectangle candidates unless the user explicitly asks to rank by
the report metrics.

In v0.4.1, `mi_bins=96` is an adaptive maximum and split-aware ranking uses
soft-binary mask MI, so it is a better default than Pearson-only ranking for
discrete candidates.

5. Explain each section of the generated pipeline so the user understands what's happening.

## Example

**User says:** "Create a churn prediction pipeline using GAFIME and CatBoost"

**Actions:** Run generator with `--task classification --model catboost --target churn`

**Result:** A complete `gafime_pipeline.py` that loads data, discovers top 10 feature interactions with GAFIME, trains CatBoost, and reports AUC-ROC with cross-validation.
