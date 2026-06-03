from __future__ import annotations

from .base import Backend


class NativeMetalBackend(Backend):
    name = "metal-native"
    device_label = "metal"
    is_gpu = True

    def __init__(self) -> None:
        raise ImportError(
            "Metal kernels have known issues in GAFIME v0.4.5 and will be fixed in v0.4.6. "
            "Use the C++ Core backend for CPU execution in this release."
        )
