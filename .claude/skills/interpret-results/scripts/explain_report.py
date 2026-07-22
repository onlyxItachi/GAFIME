#!/usr/bin/env python3
"""Interpret an explicit ``DiagnosticReport.to_dict()`` export.

Live integrations should use native report properties. This helper exists for
bounded, user-supplied JSON exports and understands the current v1 schema.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _strength(metric: str, value: float) -> str:
    magnitude = abs(value) if metric in {"pearson", "spearman"} else value
    if magnitude >= 0.9:
        return "very strong"
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.5:
        return "moderate"
    if magnitude >= 0.3:
        return "weak-moderate"
    if magnitude >= 0.1:
        return "weak"
    return "negligible"


def _pvalue(value: float) -> str:
    if value < 0.001:
        return "highly significant"
    if value < 0.01:
        return "very significant"
    if value < 0.05:
        return "significant"
    if value < 0.10:
        return "marginal"
    return "not significant"


def _stability(value: float) -> str:
    if value < 0.01:
        return "extremely stable"
    if value < 0.05:
        return "stable"
    if value < 0.10:
        return "borderline"
    return "unstable"


def _identity(item: dict) -> str:
    candidate_id = item.get("candidate_id")
    if candidate_id:
        return str(candidate_id)
    return json.dumps(
        [item.get("family", "interaction"), item.get("combo", [])],
        separators=(",", ":"),
    )


def explain_report(report: dict) -> dict:
    """Generate a bounded explanation from the current v1 export schema."""

    decision = report.get("decision") or {}
    backend = report.get("backend") or {}
    interactions = list(report.get("interactions", report.get("top_interactions", [])))
    stability_by_id = {
        _identity(item): item.get("metrics_std", {})
        for item in report.get("stability", [])
    }
    permutation_by_id = {
        _identity(item): item.get("p_values", {})
        for item in report.get("permutations", [])
    }

    explained = []
    for item in interactions:
        metrics = {str(name): float(value) for name, value in item.get("metrics", {}).items()}
        candidate_id = _identity(item)
        stds = stability_by_id.get(candidate_id, {})
        p_values = permutation_by_id.get(candidate_id, {})
        metric_details = {}
        for metric, value in metrics.items():
            metric_details[metric] = {
                "value": value,
                "strength": _strength(metric, value),
                "stability_std": stds.get(metric),
                "stability": (
                    _stability(float(stds[metric])) if metric in stds else "not requested"
                ),
                "p_value": p_values.get(metric),
                "significance": (
                    _pvalue(float(p_values[metric]))
                    if metric in p_values
                    else "not requested or unsupported for this family"
                ),
            }
        explained.append(
            {
                "candidate_id": item.get("candidate_id", candidate_id),
                "family": item.get("family", "interaction"),
                "combo": item.get("combo", []),
                "feature_names": item.get("feature_names", item.get("features", [])),
                "expression": item.get("expression", ""),
                "metrics": metric_details,
            }
        )

    return {
        "deprecation": (
            "DiagnosticReport.to_dict() is an explicit export boundary. Prefer live "
            "report properties for normal Python integration."
        ),
        "overview": {
            "signal_detected": bool(decision.get("signal_detected", False)),
            "decision_message": decision.get("message"),
            "configured_backend": report.get("configured_backend"),
            "selected_backend": backend.get("selected_backend", backend.get("name")),
            "execution_placement": backend.get("execution_placement"),
            "interaction_count": len(interactions),
            "stability_count": len(report.get("stability", [])),
            "permutation_count": len(report.get("permutations", [])),
            "warnings": report.get("warnings", []),
        },
        "interactions": explained,
        "interpretation_limits": [
            "Metric magnitude is domain-dependent and is not a model-quality guarantee.",
            "Missing p-values do not mean significance; the mode may be disabled or unsupported.",
            "Decision-path permutation significance is unavailable in v1; bootstrap stability is supported.",
            "Validate selected candidates in an untouched evaluation split or nested cross-validation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interpret a GAFIME DiagnosticReport.to_dict() JSON export"
    )
    parser.add_argument("report_json", help="Path to the explicit report export")
    args = parser.parse_args()

    path = Path(args.report_json)
    if not path.exists():
        print(json.dumps({"error": f"File not found: {args.report_json}"}))
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(explain_report(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
