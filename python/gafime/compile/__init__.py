from types import ModuleType
import sys

from .api import compile
from .flags import CompileFlags
from ..v1_adapter import NativeCompiledGafime

CompiledGafime = NativeCompiledGafime

__all__ = ["compile", "CompileFlags", "CompiledGafime", "NativeCompiledGafime"]


class _CallableCompileModule(ModuleType):
    def __call__(self, *args, **kwargs):
        return compile(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableCompileModule
