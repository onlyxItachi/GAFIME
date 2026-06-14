---
name: validate-features
description: Validate whether GAFIME-discovered feature interactions are genuinely predictive or just noise. Use when the user wants to verify feature quality, test for overfitting, check out-of-sample performance of discovered interactions, or says things like "are these features real", "validate features", "check for overfitting", "out-of-sample test", "are these interactions genuine", or "test feature significance".
---

# Validate Discovered Features

Test whether GAFIME's discovered interactions are genuinely predictive on held-out data.

## Instructions

1. Ask the user for:
   - The GAFIME report or top interaction indices
   - Their full dataset (X, y)
   - Or a separate held-out validation set

2. Run the validation script:

   ```bash
   python .claude/skills/validate-features/scripts/validate_features.py \
       --data "<file_path>" \
       --target "<target_col>" \
       --interactions "0,1;2,3;0,4"
   ```

   The `--interactions` flag takes semicolon-separated pairs of feature indices.

3. The script performs:
   - **Holdout validation**: 80/20 split, compute Pearson r on the 20% holdout
   - **Baseline comparison**: Compare interaction features vs random feature pairs
   - **Bootstrap confidence intervals**: 95% CI for each interaction's Pearson r
   - **Verdict**: GENUINE (r_holdout near r_train, p significant) or NOISE (r_holdout near zero)

   For v0.4.x discrete candidates, avoid leakage:
   - Fit GAFIME and generate thresholds on the training fold only.
   - Reconstruct discrete candidates from the train report with
     `discrete_candidate_from_result`.
   - Apply `evaluate_discrete_candidate` to the holdout fold without
     recomputing thresholds from holdout data.

4. Present the results:
   - For each interaction: train r, holdout r, baseline r, CI, verdict
   - Summary: how many are genuine vs noise
   - Recommendation: which to keep for the final model

5. If many features appear to be noise, suggest:
   - Increasing `permutation_tests` in EngineConfig
   - Using more conservative `permutation_p_threshold` (e.g., 0.01 instead of 0.05)
   - Lowering `max_discrete_candidates` or `top_k_features_for_discrete` if
     discrete search is producing too many weak candidates
   - Collecting more training data

## v0.4.x Discrete Feature Handling

Discrete features appear in reports with `family == "discrete_function"` and an
`expression` string. They are not plain pairwise products. Preserve their stored
thresholds, intervals, direction, mode, and value feature when validating.
`params["selection_score"]` and `params["selection_scores"]` are ranking
diagnostics only; validate the materialized candidate on held-out data rather
than treating those train-fold selector scores as test evidence.

In current v0.4.x releases, split-aware selector MI is adaptive soft-binary MI. Do not compare
new selector scores directly against v0.4.0 fixed-histogram score magnitudes;
compare candidate order, held-out performance, and stability instead.

## Example

**User says:** "GAFIME found 10 interactions, but I want to make sure they're real"

**Result:** "I validated all 10 on a held-out 20% split. 7 of 10 maintained their Pearson r within the 95% CI — these are genuine. 3 dropped to near-zero on holdout — likely overfitting. I recommend keeping features 1-7 and discarding 8-10."
