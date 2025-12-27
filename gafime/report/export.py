"""Report export implementations."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable, Mapping

from gafime.report.schema import FeatureScore, InteractionScore, Report, Summary


def build_report(
    *,
    diagnosis: str,
    max_abs_pearson: float,
    max_mutual_information: float,
    feature_scores: Iterable[FeatureScore],
    interaction_scores: Iterable[InteractionScore] | None,
    config: Mapping[str, object],
    permutation_pvalue: float | None = None,
) -> Report:
    summary = Summary(
        diagnosis=diagnosis,
        max_abs_pearson=max_abs_pearson,
        max_mutual_information=max_mutual_information,
        permutation_pvalue=permutation_pvalue,
    )
    return Report(
        summary=summary,
        feature_scores=list(feature_scores),
        interaction_scores=list(interaction_scores or []),
        config=dict(config),
    )


def report_to_dict(report: Report) -> dict:
    return asdict(report)


def report_to_json(report: Report, *, indent: int = 2) -> str:
    return json.dumps(report_to_dict(report), indent=indent, sort_keys=True)
