from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompileFlags:
    plan: bool = True
    graph: bool = False
    export: bool = False

    def __post_init__(self) -> None:
        for name in ("plan", "graph", "export"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"CompileFlags.{name} must be a bool.")
