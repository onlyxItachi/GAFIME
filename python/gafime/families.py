from __future__ import annotations

from dataclasses import dataclass

from .errors import V1UnsupportedError


BOOTSTRAP_STABILITY_SCOPE = (
    "Bootstrap stability resamples an already-selected candidate on the same "
    "rows. It measures metric variability conditional on selection; it is not "
    "out-of-sample or out-of-fold evidence and does not correct selection bias."
)


@dataclass(frozen=True)
class FamilySignificanceSupport:
    """Family-specific significance modes and their execution constraint."""

    permutation: bool
    stability: bool
    detail: str


_FULL_SIGNIFICANCE_SUPPORT = FamilySignificanceSupport(
    permutation=True,
    stability=True,
    detail=f"Permutation maxT is supported. {BOOTSTRAP_STABILITY_SCOPE}",
)
_DECISION_PATH_SIGNIFICANCE_SUPPORT = FamilySignificanceSupport(
    permutation=True,
    stability=True,
    detail=(
        "Permutation maxT performs decision-path rediscovery for every permuted target "
        f"before rescoring the full expanded family. {BOOTSTRAP_STABILITY_SCOPE}"
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
        """Whether at least one declared native scoring placement exists."""

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
    """Return immutable capability records for all current v1 families.

    These records describe placement policy; they do not probe installed
    payloads or guarantee that a requested device is currently available.
    """

    return _FAMILIES


def family_capability(name: str) -> FamilyCapability:
    """Return the exact named family contract or raise ``V1UnsupportedError``."""

    for capability in _FAMILIES:
        if capability.name == name:
            return capability
    raise V1UnsupportedError(f"unknown v1 family: {name!r}")


def require_family_supported(name: str) -> FamilyCapability:
    """Return a family only when its native scoring placement is supported.

    Unknown families and families without a native scoring route raise
    :class:`V1UnsupportedError`; no Python candidate-loop fallback is created.
    """

    capability = family_capability(name)
    if not capability.supported:
        raise V1UnsupportedError(
            f"v1 family {name!r} has no native device kernel wired yet; "
            "Python candidate loops are not available as a fallback."
        )
    return capability
