from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompileFlags:
    """Options for constructing an explicit compiled artifact.

    ``plan`` exposes the bounded v0.5 scenario-plan compatibility projection.
    ``graph`` requires a selected CUDA or ROCm/HIP payload that proves graph
    support and replay; Core and Metal reject it.  ``export`` enables Arrow C
    Data Interface export of the compact result table.  All values must be
    booleans.
    """

    plan: bool = True
    graph: bool = False
    export: bool = False

    def __post_init__(self) -> None:
        for name in ("plan", "graph", "export"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"CompileFlags.{name} must be a bool.")
