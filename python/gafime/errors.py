from __future__ import annotations


class GafimeV1Error(RuntimeError):
    """Base exception for explicit GAFIME v1 runtime contract failures."""


class V1UnsupportedError(GafimeV1Error):
    """Raised when v1 cannot honor a request without changing its semantics.

    Examples include unsupported backend/precision pairs, unavailable graph or
    export surfaces, and absent native family entry points.  The exception
    represents fail-closed behavior; callers should change the request or use
    ``backend="auto"`` explicitly instead of assuming an implicit fallback.
    """
