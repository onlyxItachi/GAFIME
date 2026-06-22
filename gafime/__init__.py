from .config import ComputeBudget, EngineConfig
from .compile import CompileFlags, CompiledGafime, compile
from .decision_path import DecisionPathCandidate
from .engine import GafimeEngine
from .io import GafimeStreamer
from .tutorial import generate_tutorial

__all__ = [
    "GafimeEngine",
    "EngineConfig",
    "ComputeBudget",
    "GafimeStreamer",
    "DecisionPathCandidate",
    "generate_tutorial",
    "compile",
    "CompileFlags",
    "CompiledGafime",
]

try:
    from .sklearn import GafimeSelector
    __all__.append("GafimeSelector")
except ImportError:
    pass

try:
    from . import subfunctions
    __all__.append("subfunctions")
except ImportError:
    pass

__version__ = "0.4.7"
