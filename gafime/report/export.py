"""JSON report export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_json(report: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
