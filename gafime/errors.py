from __future__ import annotations


class GafimeV1Error(RuntimeError):
    """Base class for explicit v1 runtime errors."""


class V1UnsupportedError(GafimeV1Error):
    """Raised when a requested feature is not available in the v1 runtime."""
