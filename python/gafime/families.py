from __future__ import annotations

from dataclasses import dataclass

from .errors import V1UnsupportedError


@dataclass(frozen=True)
class FamilyCapability:
    name: str
    family_id: int
    continuous_input: bool
    cpu_kernel: bool
    cuda_kernel: bool
    rocm_kernel: bool
    python_candidate_loop: bool = False

    @property
    def supported(self) -> bool:
        return self.cpu_kernel or self.cuda_kernel or self.rocm_kernel


_FAMILIES: tuple[FamilyCapability, ...] = (
    FamilyCapability("continuous", 1, True, True, True, False),
    # decision_path + time_series are wired via native feature-expansion +
    # continuous mining, so both run on CPU and CUDA.
    FamilyCapability("decision_path", 2, True, True, True, False),
    FamilyCapability("time_series", 3, True, True, True, False),
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
