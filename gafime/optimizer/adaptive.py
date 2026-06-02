from __future__ import annotations


class AdaptiveOptimizer:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "AdaptiveOptimizer was removed from the v0.4.5 native-only spine. "
            "Use GafimeEngine with native CUDA/Core backends."
        )
