from .config import ComputeBudget, EngineConfig
from .engine import GafimeEngine
from .io import GafimeStreamer
from .tutorial import generate_tutorial

__all__ = ["GafimeEngine", "EngineConfig", "ComputeBudget", "GafimeStreamer", "generate_tutorial"]

try:
    from .sklearn import GafimeSelector
    __all__.append("GafimeSelector")
except ImportError:
    pass

__version__ = "0.3.2"
