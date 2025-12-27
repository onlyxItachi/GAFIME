"""JSON report export."""
from __future__ import annotations

import json
from typing import Any, Dict

from .schema import Report


def export_report(report: Report) -> str:
    """Serialize report to JSON string."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def export_report_dict(report: Report) -> Dict[str, Any]:
    """Return report as a dict for in-memory usage."""
    return report.to_dict()
