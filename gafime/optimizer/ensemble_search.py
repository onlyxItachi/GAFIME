from __future__ import annotations


class FeatureRecipe:
    pass


class CandidateGenerator:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "CandidateGenerator was removed from the v0.4.5 native-only spine. "
            "Engine planning now owns candidate generation."
        )


class EnsembleSearch:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "EnsembleSearch was removed from the v0.4.5 native-only spine. "
            "Use GafimeEngine with native CUDA/Core backends."
        )
