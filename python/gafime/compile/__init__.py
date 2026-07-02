from .api import compile
from .flags import CompileFlags
from ..v1_adapter import NativeCompiledGafime

CompiledGafime = NativeCompiledGafime

__all__ = ["compile", "CompileFlags", "CompiledGafime", "NativeCompiledGafime"]
