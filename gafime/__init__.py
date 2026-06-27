from __future__ import annotations

from .api import GafimeEngine, compile
from .config import ComputeBudget, EngineConfig
from .errors import GafimeV1Error, V1UnsupportedError
from .families import FamilyCapability, available_families, family_capability, require_family_supported
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
    "FamilyCapability",
    "GafimeEngine",
    "GafimeV1Error",
    "InteractionResult",
    "NativeCompiledGafime",
    "V1UnsupportedError",
    "available_families",
    "compile",
    "family_capability",
    "require_family_supported",
]

__version__ = "1.0.0a0"
