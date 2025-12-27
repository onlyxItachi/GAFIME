from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import ComputeBudget, EngineConfig
from ..metrics import MetricSuite
from .base import Backend, BackendInfo


class TorchMetrics:
    def __init__(self, torch_module):
        self.torch = torch_module

    def _safe_pearson(self, x, y) -> float:
        torch = self.torch
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        denom = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
        if float(denom) == 0.0:
            return 0.0
        return float(torch.sum(x_centered * y_centered) / denom)

    def _rankdata(self, values):
        torch = self.torch
        sorter = torch.argsort(values)
        sorted_values = values[sorter]
        if sorted_values.numel() == 0:
            return sorted_values

        diff = sorted_values[1:] != sorted_values[:-1]
        change_idx = torch.nonzero(diff, as_tuple=False).flatten() + 1
        starts = torch.cat([values.new_tensor([0], dtype=torch.long), change_idx])
        ends = torch.cat([starts[1:], values.new_tensor([values.numel()], dtype=torch.long)])

        ranks = torch.empty_like(sorted_values, dtype=torch.float32)
        for start, end in zip(starts.tolist(), ends.tolist()):
            avg_rank = 0.5 * (start + end - 1)
            ranks[start:end] = avg_rank

        inv = torch.empty_like(sorter)
        inv[sorter] = torch.arange(values.numel(), device=values.device)
        return ranks[inv]

    def pearson_corr(self, x, y, xp=None) -> float:
        return self._safe_pearson(x, y)

    def spearman_corr(self, x, y, xp=None) -> float:
        x_rank = self._rankdata(x)
        y_rank = self._rankdata(y)
        return self._safe_pearson(x_rank, y_rank)

    def mutual_info(self, x, y, bins: int = 16, xp=None) -> float:
        torch = self.torch
        if bins < 2:
            return 0.0
        x_min, x_max = torch.min(x), torch.max(x)
        y_min, y_max = torch.min(y), torch.max(y)
        if float(x_min) == float(x_max) or float(y_min) == float(y_max):
            return 0.0

        x_edges = torch.linspace(x_min, x_max, bins + 1, device=x.device)
        y_edges = torch.linspace(y_min, y_max, bins + 1, device=y.device)
        x_bin = torch.bucketize(x, x_edges) - 1
        y_bin = torch.bucketize(y, y_edges) - 1

        valid = (x_bin >= 0) & (x_bin < bins) & (y_bin >= 0) & (y_bin < bins)
        x_bin = x_bin[valid]
        y_bin = y_bin[valid]
        if x_bin.numel() == 0:
            return 0.0

        idx = x_bin * bins + y_bin
        hist = torch.bincount(idx, minlength=bins * bins).reshape(bins, bins).float()
        total = torch.sum(hist)
        if float(total) == 0.0:
            return 0.0

        p_xy = hist / total
        p_x = torch.sum(p_xy, dim=1, keepdim=True)
        p_y = torch.sum(p_xy, dim=0, keepdim=True)
        expected = p_x * p_y
        nonzero = p_xy > 0
        mi = torch.sum(p_xy[nonzero] * torch.log(p_xy[nonzero] / expected[nonzero]))
        return float(mi)

    def linear_r2(self, x, y, xp=None) -> float:
        corr = self._safe_pearson(x, y)
        return float(corr * corr)


class TorchBackend(Backend):
    name = "rocm"
    device_label = "cuda"
    is_gpu = True

    def __init__(self, mode: str, device_id: int = 0) -> None:
        import torch

        super().__init__(device_id=device_id)
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("Torch GPU backend requested but CUDA/HIP is not available.")
        if mode == "rocm" and torch.version.hip is None:
            raise RuntimeError("ROCm backend requested but torch is not built with HIP support.")
        if mode not in ("cuda", "rocm"):
            raise ValueError("Torch backend mode must be 'cuda' or 'rocm'.")

        self.mode = mode
        self.device = torch.device("cuda", device_id)
        self.name = "rocm" if mode == "rocm" else "cuda-torch"
        self.device_label = f"{self.name}:{device_id}"
        self.metrics_ops = TorchMetrics(torch)
        self.xp = None

    def metric_suite(self, config: EngineConfig) -> MetricSuite:
        return MetricSuite(
            config.metric_names,
            mi_bins=config.mi_bins,
            xp=None,
            ops=self.metrics_ops,
        )

    def info(self) -> BackendInfo:
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.device_id)
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
        free_bytes, _total_bytes = self.torch.cuda.mem_get_info(self.device_id)
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
        return self.torch.as_tensor(array, dtype=self.torch.float32, device=self.device)

    def to_host(self, array):
        return array.detach().cpu().numpy()

    def build_interaction_vector(self, X, combo: Tuple[int, ...]):
        idx = list(combo)
        if len(idx) == 1:
            return X[:, idx[0]]
        slice_data = X[:, idx]
        centered = slice_data - self.torch.mean(slice_data, dim=0)
        return self.torch.prod(centered, dim=1)

    def sample_indices(self, n_samples: int, rng: np.random.Generator):
        indices = rng.integers(0, n_samples, size=n_samples)
        return self.torch.as_tensor(indices, dtype=self.torch.long, device=self.device)

    def permute(self, y, rng: np.random.Generator):
        indices = rng.permutation(y.shape[0])
        idx_tensor = self.torch.as_tensor(indices, dtype=self.torch.long, device=self.device)
        return y[idx_tensor]
