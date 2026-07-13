from __future__ import annotations

from ._version import __version__
from .api import GafimeEngine, compile as _compile
from .capabilities import BackendCapabilities, CapabilityValue, backend_capabilities
from .compile.flags import CompileFlags
from .config import ComputeBudget, EngineConfig
from .dataloader import dataload
from .errors import GafimeV1Error, V1UnsupportedError
from .families import FamilyCapability, available_families, family_capability, require_family_supported
from .reporting import BackendInfo, Decision, DiagnosticReport, InteractionResult
from .v1_adapter import NativeCompiledGafime

CompiledGafime = NativeCompiledGafime
compile = _compile

__all__ = [
    "BackendInfo",
    "BackendCapabilities",
    "CapabilityValue",
    "CompiledGafime",
    "CompileFlags",
    "ComputeBudget",
    "Decision",
    "DiagnosticReport",
    "EngineConfig",
    "FamilyCapability",
    "dataload",
    "GafimeEngine",
    "GafimeV1Error",
    "InteractionResult",
    "NativeCompiledGafime",
    "V1UnsupportedError",
    "available_families",
    "backend_capabilities",
    "compile",
    "family_capability",
    "require_family_supported",
]
