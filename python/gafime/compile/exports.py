from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExportHandles:
    """Deprecated compatibility view of compiled native handles.

    Current v1 artifacts expose the owning feature-matrix handle and, after an
    analysis, the compact result-table handle.  They do not expose an
    independent candidate-table handle, so that field is normally ``None``.
    New code should request ``CompileFlags(export=True)`` and call
    ``NativeCompiledGafime.export_arrow()`` instead.
    """

    backend_name: str
    feature_matrix_handle: Any = None
    result_table_handle: Any = None
    candidate_table_handle: Any = None


def unsupported_export(backend_name: str) -> RuntimeError:
    """Build the compatibility error used for an unavailable export surface."""

    return RuntimeError(
        f"Compiled export handles are not available for backend '{backend_name}'."
    )


__all__ = ["ExportHandles", "unsupported_export"]
