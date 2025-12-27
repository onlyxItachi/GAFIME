"""JSON export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def export_report(report: Dict[str, Any], path: str | Path | None = None) -> str:
    """Serialize report to JSON and optionally write to disk."""
    payload = json.dumps(report, indent=2, sort_keys=True)
    if path is not None:
        Path(path).write_text(payload, encoding="utf-8")
    return payload
