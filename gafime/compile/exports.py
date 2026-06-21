from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExportHandles:
    backend_name: str
    feature_matrix_handle: Any = None
    result_table_handle: Any = None
    candidate_table_handle: Any = None


def unsupported_export(backend_name: str) -> RuntimeError:
    return RuntimeError(f"Compiled export handles are not available for backend '{backend_name}'.")
