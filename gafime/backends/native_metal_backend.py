from __future__ import annotations

from .base import Backend


class NativeMetalBackend(Backend):
    name = "metal-native"
    device_label = "metal"
    is_gpu = True

    def __init__(self) -> None:
        raise ImportError(
            "Native Metal backend is disabled in GAFIME v0.4.5. "
            "Use the C++ Core backend for CPU execution in this release."
        )
