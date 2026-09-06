from __future__ import annotations

from ._version import __version__ as __version__
from .api import GafimeEngine, compile as _compile
from .capabilities import BackendCapabilities, CapabilityValue, backend_capabilities
from .compile.flags import CompileFlags
from .config import ComputeBudget, EngineConfig
from .dataloader import dataload
from .decision_path import DecisionPathCandidate
from .errors import GafimeV1Error, V1UnsupportedError
from .families import (
    FamilyCapability,
    FamilySignificanceSupport,
    available_families,
    family_capability,
    require_family_supported,
)
from .io import GafimeStreamer
from .reporting import BackendInfo, Decision, DiagnosticReport, InteractionResult
from .sklearn import GafimeSelector
from . import subfunctions as subfunctions
from . import semantic as semantic
from .tutorial import generate_tutorial
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
    "DecisionPathCandidate",
    "DiagnosticReport",
    "EngineConfig",
    "FamilyCapability",
    "FamilySignificanceSupport",
    "dataload",
    "GafimeEngine",
    "GafimeSelector",
    "GafimeStreamer",
    "GafimeV1Error",
    "InteractionResult",
    "NativeCompiledGafime",
    "V1UnsupportedError",
    "available_families",
    "backend_capabilities",
    "compile",
    "family_capability",
    "generate_tutorial",
    "require_family_supported",
    "subfunctions",
    "semantic",
]
