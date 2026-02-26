---
name: interpret-results
description: Interpret and explain GAFIME DiagnosticReport results in plain English. Use when the user has run GAFIME and wants to understand the output, asks about Pearson correlation values, p-values, stability metrics, signal detection, feature interaction results, or says things like "what does this report mean", "explain the results", "is this feature interaction significant", "what does signal detected mean", or "interpret my GAFIME output".
---

# Interpret GAFIME Results

Translate GAFIME's `DiagnosticReport` into actionable, human-readable insights.

## Instructions

1. Ask the user to either:
   - Share the report output (printed `DiagnosticReport`)
   - Or provide their analysis script so you can add the interpretation code

2. If they have a report object in Python, run the explanation script:

   ```bash
   python .claude/skills/interpret-results/scripts/explain_report.py "<report_json_path>"
   ```

   Or help them generate a JSON dump with:

   ```python
   import json
   # After engine.analyze()
   report_dict = {
       "signal_detected": report.decision.signal_detected,
       "message": report.decision.message,
       "backend": report.backend.name if report.backend else "numpy",
       "n_interactions": len(report.interactions),
       "top_interactions": [
           {
               "features": list(ix.feature_names),
               "combo": list(ix.combo),
               "metrics": ix.metrics,
           }
           for ix in sorted(report.interactions, key=lambda x: abs(x.metrics.get("pearson", 0)), reverse=True)[:10]
       ],
       "stability": [
           {"combo": list(s.combo), "metrics_std": s.metrics_std}
           for s in report.stability[:10]
       ],
       "permutations": [
           {"combo": list(p.combo), "p_values": p.p_values}
           for p in report.permutations[:10]
       ],
       "warnings": report.warnings,
   }
   with open("gafime_report.json", "w") as f:
       json.dump(report_dict, f, indent=2)
   ```

3. Explain each section clearly:

   **Signal Detection:**
   - `signal_detected = True` means GAFIME found at least one feature interaction that is statistically significant, stable, and has non-zero predictive power.
   - `signal_detected = False` means no interaction passed all three tests (strength, stability, significance).

   **Top Interactions:**
   - Pearson r close to 1.0 or -1.0 = very strong linear relationship
   - Pearson r around 0.3-0.7 = moderate, potentially useful
   - Pearson r below 0.1 = weak, likely noise
   - Explain what the feature combination MEANS (e.g., "f3 x f7 = the interaction of feature 3 and feature 7")

   **Stability Analysis:**
   - metrics_std below 0.01 = extremely stable (trustworthy)
   - metrics_std 0.01-0.05 = stable (good)
   - metrics_std 0.05-0.10 = borderline (use with caution)
   - metrics_std above 0.10 = unstable (don't trust this feature)

   **Permutation Test:**
   - p-value below 0.01 = very significant (strong evidence this isn't random)
   - p-value 0.01-0.05 = significant (standard threshold)
   - p-value above 0.05 = not significant (could be random chance)

4. Provide actionable recommendations:
   - Which interactions to use in their model
   - Which to discard (high p-value or unstable)
   - Whether to increase `permutation_tests` for more confidence
   - Next steps (feed into CatBoost, XGBoost, etc.)

## Example

**User says:** "GAFIME found 47 interactions but I don't know which ones matter"

**Result:** "Of your 47 interactions, 5 are statistically significant (p < 0.05) and stable (std < 0.05). The top one is `log(feature_3) x feature_7` with Pearson r=0.82 — this is a strong signal. I'd recommend using these 5 as additional features in your model..."
