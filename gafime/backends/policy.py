from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import List


X86_MACHINES = {"x86_64", "amd64", "x64"}
ARM_MACHINES = {"arm64", "aarch64"}


@dataclass(frozen=True)
class PlatformProfile:
    system: str
    machine: str

    @property
    def is_macos(self) -> bool:
        return self.system == "darwin"

    @property
    def is_linux(self) -> bool:
        return self.system == "linux"

    @property
    def is_windows(self) -> bool:
        return self.system == "windows"

    @property
    def is_x86(self) -> bool:
        return self.machine in X86_MACHINES

    @property
    def is_arm(self) -> bool:
        return self.machine in ARM_MACHINES

    @property
    def label(self) -> str:
        return f"{self.system}/{self.machine}"


def current_platform_profile() -> PlatformProfile:
    return PlatformProfile(
        system=platform.system().lower(),
        machine=platform.machine().lower(),
    )


def backend_priority(requested: str, profile: PlatformProfile | None = None) -> List[str]:
    """Return native backend names in platform-specific priority order."""
    profile = profile or current_platform_profile()
    requested = (requested or "auto").lower()

    if requested in {"cpu", "core", "cpp"}:
        return ["core"]

    if requested in {"rocm", "hip"}:
        if profile.is_macos:
            raise RuntimeError("ROCm/HIP backend is not supported on macOS. Use backend='metal' or backend='core'.")
        if not profile.is_linux:
            raise RuntimeError("ROCm/HIP backend currently requires Linux. Use backend='core'.")
        if profile.is_arm:
            raise RuntimeError(
                f"ROCm/HIP backend is not supported by current GAFIME ARM wheels on {profile.label}. "
                "Use backend='core'."
            )
        return ["rocm"]

    if requested == "auto":
        if profile.is_macos:
            return ["metal", "core"]
        if (profile.is_linux or profile.is_windows) and profile.is_x86:
            return ["cuda", "core"]
        return ["core"]

    if requested == "gpu":
        if profile.is_macos:
            return ["metal"]
        if (profile.is_linux or profile.is_windows) and profile.is_x86:
            return ["cuda"]
        raise RuntimeError(
            f"backend='gpu' has no supported GPU backend on {profile.label}. "
            "Use backend='core'."
        )

    if requested == "cuda":
        if profile.is_macos:
            raise RuntimeError("CUDA backend is not supported on macOS. Use backend='metal' or backend='core'.")
        if (profile.is_linux or profile.is_windows) and profile.is_arm:
            raise RuntimeError(
                f"CUDA backend is not supported by current GAFIME ARM wheels on {profile.label}. "
                "Use backend='core'."
            )
        return ["cuda"]

    if requested == "metal":
        if not profile.is_macos:
            raise RuntimeError("Metal backend is only supported on macOS. Use backend='cuda' or backend='core'.")
        return ["metal"]

    raise ValueError(f"Unknown backend '{requested}'.")
