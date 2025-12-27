"""Report schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_report(
    *,
    summary: dict,
    config: dict,
    unary_results: list[dict],
    pairwise_results: list[dict] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "version": "0.1",
        },
        "summary": summary,
        "config": config,
        "unary_results": unary_results,
        "pairwise_results": pairwise_results or [],
        "diagnostics": diagnostics or {},
    }
