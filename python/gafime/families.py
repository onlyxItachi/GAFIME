from __future__ import annotations

from dataclasses import dataclass

from .errors import V1UnsupportedError


@dataclass(frozen=True)
class FamilySignificanceSupport:
    """Family-specific significance modes and their execution constraint."""

    permutation: bool
    stability: bool
    detail: str


_FULL_SIGNIFICANCE_SUPPORT = FamilySignificanceSupport(
    permutation=True,
    stability=True,
    detail="Permutation maxT and selected-candidate bootstrap stability are supported.",
)
_DECISION_PATH_SIGNIFICANCE_SUPPORT = FamilySignificanceSupport(
    permutation=False,
    stability=True,
    detail=(
        "Permutation significance is unavailable because every permuted target "
        "requires decision-path rediscovery; selected-candidate bootstrap stability "
        "is supported."
    ),
)


@dataclass(frozen=True)
class FamilyCapability:
    """Public family placement contract.

    ``*_kernel`` fields are retained as compatibility aliases for scoring
    support. They do not claim a feature-generation kernel on that backend.
    Use ``generation_placement`` and ``scoring_placement`` for new code.
    ``significance_support`` is family-specific and takes precedence over
    backend-wide significance placement.
    """

    name: str
    family_id: int
    continuous_input: bool
    cpu_kernel: bool
    cuda_kernel: bool
    rocm_kernel: bool
    metal_kernel: bool
    python_candidate_loop: bool = False
    generation_placement: str = "native_continuous"
    scoring_placement: tuple[str, ...] = ()
    graph_scope: str = "backend_runtime"
    native_compact_scoring: tuple[str, ...] = ()
    significance_support: FamilySignificanceSupport = _FULL_SIGNIFICANCE_SUPPORT

    @property
    def supported(self) -> bool:
        return bool(self.scoring_placement) or any(
            (self.cpu_kernel, self.cuda_kernel, self.rocm_kernel, self.metal_kernel)
        )

    @property
    def scoring_backends(self) -> tuple[str, ...]:
        """Alias for the explicit scoring placement list."""

        if self.scoring_placement:
            return self.scoring_placement
        return tuple(
            backend
            for backend, enabled in (
                ("gafime_cpu", self.cpu_kernel),
                ("cuda", self.cuda_kernel),
                ("rocm", self.rocm_kernel),
                ("metal", self.metal_kernel),
            )
            if enabled
        )

    @property
    def generation_backend(self) -> str:
        """Explicit alias for the backend that creates family candidates."""

        return self.generation_placement


_FAMILIES: tuple[FamilyCapability, ...] = (
    FamilyCapability(
        "continuous",
        1,
        True,
        True,
        True,
        True,
        True,
        generation_placement="native_continuous",
        scoring_placement=("gafime_cpu", "cuda", "rocm", "metal"),
        graph_scope="backend_runtime",
        native_compact_scoring=("gafime_cpu", "cuda", "rocm", "metal"),
        significance_support=_FULL_SIGNIFICANCE_SUPPORT,
    ),
    FamilyCapability(
        "decision_path",
        2,
        True,
        True,
        True,
        True,
        True,
        generation_placement="gafime_cpu",
        scoring_placement=("gafime_cpu", "cuda", "rocm", "metal"),
        graph_scope="continuous_scoring_only",
        native_compact_scoring=("cuda_rt_optional",),
        significance_support=_DECISION_PATH_SIGNIFICANCE_SUPPORT,
    ),
    FamilyCapability(
        "time_series",
        3,
        True,
        True,
        True,
        True,
        True,
        generation_placement="gafime_cpu",
        scoring_placement=("gafime_cpu", "cuda", "rocm", "metal"),
        graph_scope="continuous_scoring_only",
        significance_support=_FULL_SIGNIFICANCE_SUPPORT,
    ),
)


def available_families() -> tuple[FamilyCapability, ...]:
    return _FAMILIES


def family_capability(name: str) -> FamilyCapability:
    for capability in _FAMILIES:
        if capability.name == name:
            return capability
    raise V1UnsupportedError(f"unknown v1 family: {name!r}")


def require_family_supported(name: str) -> FamilyCapability:
    capability = family_capability(name)
    if not capability.supported:
        raise V1UnsupportedError(
            f"v1 family {name!r} has no native device kernel wired yet; "
            "Python candidate loops are not available as a fallback."
        )
    return capability
