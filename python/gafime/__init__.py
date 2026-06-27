from __future__ import annotations

from .api import GafimeEngine, compile
from .config import ComputeBudget, EngineConfig
from .errors import GafimeV1Error, V1UnsupportedError
from .reporting import BackendInfo, Decision, DiagnosticReport, InteractionResult
from .v1_adapter import NativeCompiledGafime

CompiledGafime = NativeCompiledGafime

__all__ = [
    "BackendInfo",
    "CompiledGafime",
    "ComputeBudget",
    "Decision",
    "DiagnosticReport",
    "EngineConfig",
    "GafimeEngine",
    "GafimeV1Error",
    "InteractionResult",
    "NativeCompiledGafime",
    "V1UnsupportedError",
    "compile",
]

__version__ = "1.0.0a0"
