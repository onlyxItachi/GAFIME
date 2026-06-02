from __future__ import annotations


class Orchestrator:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "Optimizer orchestration was removed from the v0.4.5 native-only spine. "
            "Use GafimeEngine; Rust scheduling and native backends are integrated there."
        )
