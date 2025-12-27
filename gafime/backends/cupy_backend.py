from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import ComputeBudget
from ..metrics import cpu_metrics
from .base import Backend, BackendInfo


_DLL_DIR_HANDLES: list[object] = []


def _augment_cuda_paths() -> None:
    import importlib.util
    import os
    from pathlib import Path

    candidates = ["nvidia.cuda_nvrtc", "nvidia.cuda_runtime"]
    current_path = os.environ.get("PATH", "")
    path_parts = {part.lower() for part in current_path.split(os.pathsep) if part}

    for module_name in candidates:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.submodule_search_locations:
            base = Path(spec.submodule_search_locations[0])
            bin_path = base / "bin"
            if bin_path.exists() and str(bin_path).lower() not in path_parts:
                if hasattr(os, "add_dll_directory"):
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(str(bin_path)))
                current_path = f"{bin_path}{os.pathsep}{current_path}"
                path_parts.add(str(bin_path).lower())

    os.environ["PATH"] = current_path


class CupyBackend(Backend):
    name = "cuda"
    device_label = "cuda"
    is_gpu = True

    def __init__(self, device_id: int = 0) -> None:
        _augment_cuda_paths()
        import cupy as cp

        super().__init__(device_id=device_id)
        self.cp = cp
        self.xp = cp
        self.metrics_ops = cpu_metrics
        self.device = cp.cuda.Device(device_id)
        self.device.use()
        self.device_label = f"cuda:{device_id}"

    def info(self) -> BackendInfo:
        free_bytes, total_bytes = self.device.mem_info
        return BackendInfo(
            name=self.name,
            device=self.device_label,
            is_gpu=self.is_gpu,
            memory_total_mb=int(total_bytes / (1024**2)),
            memory_free_mb=int(free_bytes / (1024**2)),
        )

    def check_budget(
        self,
        X: np.ndarray,
        y: np.ndarray,
        budget: ComputeBudget,
    ) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        if not budget.keep_in_vram:
            warnings.append("keep_in_vram is False; GPU backend disabled.")
            return False, warnings

        required_bytes = self.estimate_bytes(X, y)
        free_bytes, _total_bytes = self.device.mem_info
        if budget.vram_budget_mb > 0:
            budget_bytes = budget.vram_budget_mb * 1024 * 1024
            effective_limit = min(budget_bytes, free_bytes)
        else:
            effective_limit = free_bytes

        if required_bytes > effective_limit:
            warnings.append("VRAM budget exceeded; falling back to CPU backend.")
            return False, warnings

        return True, warnings

    def to_device(self, array: np.ndarray):
        return self.cp.asarray(array)

    def to_host(self, array):
        return self.cp.asnumpy(array)

    def sample_indices(self, n_samples: int, rng: np.random.Generator):
        indices = rng.integers(0, n_samples, size=n_samples)
        return self.cp.asarray(indices)

    def permute(self, y, rng: np.random.Generator):
        indices = rng.permutation(y.shape[0])
        return y[self.cp.asarray(indices)]
