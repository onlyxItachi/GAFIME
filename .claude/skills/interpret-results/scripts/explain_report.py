#!/usr/bin/env python3
"""
GAFIME Report Interpreter

Reads a GAFIME DiagnosticReport JSON dump and produces a human-readable explanation.
"""

import argparse
import json
import sys
from pathlib import Path


def interpret_pearson(r: float) -> str:
    """Interpret Pearson correlation strength."""
    ar = abs(r)
    if ar >= 0.9:
        return "very strong"
    elif ar >= 0.7:
        return "strong"
    elif ar >= 0.5:
        return "moderate"
    elif ar >= 0.3:
        return "weak-moderate"
    elif ar >= 0.1:
        return "weak"
    else:
        return "negligible"


def interpret_pvalue(p: float) -> str:
    """Interpret p-value significance."""
    if p < 0.001:
        return "highly significant"
    elif p < 0.01:
        return "very significant"
    elif p < 0.05:
        return "significant"
    elif p < 0.10:
        return "marginally significant"
    else:
        return "not significant"


def interpret_stability(std: float) -> str:
    """Interpret stability standard deviation."""
    if std < 0.01:
        return "extremely stable"
    elif std < 0.05:
        return "stable"
    elif std < 0.10:
        return "borderline"
    else:
        return "unstable"


def explain_report(report: dict) -> dict:
    """Generate a comprehensive explanation of the report."""
    explanation = {
        "overview": {},
        "top_interactions": [],
        "actionable_features": [],
        "discarded_features": [],
        "recommendations": [],
    }

    # Overview
    explanation["overview"] = {
        "signal_detected": report.get("signal_detected", False),
        "signal_meaning": (
            "GAFIME found at least one statistically significant, stable feature interaction with predictive power."
            if report.get("signal_detected")
            else "No feature interaction passed all three validation tests (strength, stability, significance). This could mean the features are independently predictive, or the data needs more samples."
        ),
        "backend_used": report.get("backend", "unknown"),
        "total_interactions_evaluated": report.get("n_interactions", 0),
        "warnings": report.get("warnings", []),
    }

    # Build lookup maps
    stability_map = {}
    for s in report.get("stability", []):
        key = tuple(s.get("combo", []))
        stability_map[key] = s.get("metrics_std", {})

    perm_map = {}
    for p in report.get("permutations", []):
        key = tuple(p.get("combo", []))
        perm_map[key] = p.get("p_values", {})

    # Analyze top interactions
    for ix in report.get("top_interactions", []):
        combo = tuple(ix.get("combo", []))
        features = ix.get("features", [])
        metrics = ix.get("metrics", {})

        pearson = metrics.get("pearson", 0)
        spearman = metrics.get("spearman", 0)

        stds = stability_map.get(combo, {})
        p_vals = perm_map.get(combo, {})

        pearson_std = stds.get("pearson", None)
        pearson_p = p_vals.get("pearson", None)

        analysis = {
            "features": features,
            "combo_indices": list(combo),
            "pearson_r": round(pearson, 4),
            "pearson_strength": interpret_pearson(pearson),
            "spearman_r": round(spearman, 4) if spearman else None,
            "stability_std": round(pearson_std, 4) if pearson_std is not None else None,
            "stability_assessment": interpret_stability(pearson_std) if pearson_std is not None else "untested",
            "p_value": round(pearson_p, 4) if pearson_p is not None else None,
            "significance": interpret_pvalue(pearson_p) if pearson_p is not None else "untested",
        }

        # Decision: actionable or discard?
        is_strong = abs(pearson) > 0.1
        is_stable = pearson_std is None or pearson_std < 0.10
        is_significant = pearson_p is None or pearson_p < 0.05

        analysis["verdict"] = "USE" if (is_strong and is_stable and is_significant) else "DISCARD"
        analysis["reason"] = []
        if not is_strong:
            analysis["reason"].append("too weak (|r| < 0.1)")
        if not is_stable:
            analysis["reason"].append(f"unstable (std={pearson_std:.3f})")
        if not is_significant:
            analysis["reason"].append(f"not significant (p={pearson_p:.3f})")

        explanation["top_interactions"].append(analysis)

        if analysis["verdict"] == "USE":
            explanation["actionable_features"].append(analysis)
        else:
            explanation["discarded_features"].append(analysis)

    # Recommendations
    n_actionable = len(explanation["actionable_features"])
    if n_actionable > 0:
        explanation["recommendations"].append(
            f"Use the {n_actionable} actionable interaction(s) as additional features in your model."
        )
        explanation["recommendations"].append(
            "Feed them into a gradient boosting model (CatBoost, XGBoost, LightGBM) alongside your original features."
        )
    else:
        explanation["recommendations"].append(
            "No interactions passed all validation tests. Consider: increasing max_combinations_per_k, trying different operators, or adding more training samples."
        )

    if report.get("warnings"):
        explanation["recommendations"].append(
            f"Address the {len(report['warnings'])} warning(s) listed in the report."
        )

    return explanation


def main():
    parser = argparse.ArgumentParser(description="GAFIME Report Interpreter")
    parser.add_argument("report_json", help="Path to GAFIME report JSON file")
    args = parser.parse_args()

    path = Path(args.report_json)
    if not path.exists():
        print(json.dumps({"error": f"File not found: {args.report_json}"}))
        return 1

    with open(path) as f:
        report = json.load(f)

    explanation = explain_report(report)
    print(json.dumps(explanation, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
