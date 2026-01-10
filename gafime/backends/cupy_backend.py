from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..config import ComputeBudget
from ..metrics import MetricSuite, cpu_metrics
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

    def score_combos(
        self,
        X,
        y,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        combos_list = list(combos)
        if not combos_list:
            return {}

        # Ensure data is on device
        X = self.to_device(X)
        y = self.to_device(y)

        n_samples, n_features = X.shape
        n_combos = len(combos_list)

        # Determine needed metrics
        need_pearson = "pearson" in metric_suite.metric_names or "r2" in metric_suite.metric_names

        # Prepare result structure
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {c: {} for c in combos_list}

        # Vectorized Pearson/R2
        if need_pearson:
            # 1. Precompute means of features and y
            # We assume X is already on device (cp.ndarray)
            # Center y
            y_mean = self.cp.mean(y)
            y_centered = y - y_mean
            y_norm = self.cp.sqrt(self.cp.sum(y_centered ** 2))

            # 2. Build interaction matrix: (n_samples, n_combos)
            # This can be memory intensive, but Engine chunks combos now.
            # We construct it column by column or vectorized if k is uniform?
            # Since combos can have mixed k, we might need to handle them carefully.
            # But usually combos in a chunk are processed together.

            # Construct interaction matrix Z
            Z = self.cp.empty((n_samples, n_combos), dtype=X.dtype)

            # To vectorize construction, we can group by k?
            # For now, let's use a loop to build Z, but then use matrix multiplication for scoring.
            # Building Z is still O(N*C), but scoring is then fast.
            # Ideally Z construction is also vectorized, but irregular combos make it hard without mask.
            # The "loop bottleneck" in plan referred to "Python-side iteration ... one sync per metric per combo".
            # By building Z and then scoring, we reduce syncs.

            # Wait, if we loop in Python to build Z, we still iterate.
            # But the plan says: "use matrix operations: (Batch, Samples) @ (Samples, 1)".
            # This implies Z is built.

            X_means = self.cp.mean(X, axis=0)

            for i, combo in enumerate(combos_list):
                if len(combo) == 1:
                    Z[:, i] = X[:, combo[0]]
                else:
                    # Centered product for interactions > 1
                    # This is element-wise product of centered features
                    term = X[:, combo[0]] - X_means[combo[0]]
                    for idx in combo[1:]:
                        term *= (X[:, idx] - X_means[idx])
                    Z[:, i] = term

            # 3. Compute correlations
            # Z: (n_samples, n_combos)
            # Z_centered: (n_samples, n_combos)
            Z_means = self.cp.mean(Z, axis=0)
            Z_centered = Z - Z_means

            # Dots: (n_combos,)
            dots = self.cp.dot(Z_centered.T, y_centered)

            # Norms: (n_combos,)
            Z_norms = self.cp.sqrt(self.cp.sum(Z_centered ** 2, axis=0))

            corrs = dots / (Z_norms * y_norm)
            # Handle div by zero
            corrs = self.cp.nan_to_num(corrs)

            corrs_host = self.cp.asnumpy(corrs)

            for i, combo in enumerate(combos_list):
                val = float(corrs_host[i])
                if "pearson" in metric_suite.metric_names:
                    scores[combo]["pearson"] = val
                if "r2" in metric_suite.metric_names:
                    scores[combo]["r2"] = val * val

        # Handle other metrics via fallback (loop) or further vectorization
        other_metrics = [m for m in metric_suite.metric_names if m not in ("pearson", "r2")]
        if other_metrics:
            # We can reuse Z?
            # Yes, Z contains the interaction values.
            # But Z might be consumed/modified? No.
            # We need to compute other metrics on Z columns vs y.

            # If we already have Z, we can iterate over columns of Z on GPU?
            # Or just use the standard loop but pass Z column?
            # The standard `metric_suite.score` takes `vector, y`.
            # We can pass `Z[:, i]` as vector.

            # However, `score` might trigger syncs.
            # For Spearman/MI, they are expensive anyway.
            # Plan only asked for Pearson/R2 vectorization.

            for i, combo in enumerate(combos_list):
                # If we didn't compute Z (e.g. only Spearman requested), we need to build it.
                # But typically Pearson is requested.
                # If Z exists:
                if need_pearson:
                    vec = Z[:, i]
                else:
                     # Fallback to build
                    vec = self.build_interaction_vector(X, combo)

                # We only need to compute other metrics
                # Create a mini suite or call specific ops?
                # metric_suite.score computes all. We already have pearson/r2.
                # Let's just compute the rest.

                # We need to manually call ops for other metrics
                for name in other_metrics:
                    # This is not ideal as it duplicates logic from MetricSuite
                    # But MetricSuite doesn't support "compute only X".
                    # Actually, we can just let it recompute or refactor MetricSuite.
                    # Given constraints, let's just loop and call ops.
                    if name == "spearman":
                        val = self.metrics_ops.spearman_corr(vec, y, xp=self.xp)
                        scores[combo]["spearman"] = val
                    elif name == "mutual_info":
                        # Mutual info might require bins
                        val = self.metrics_ops.mutual_info(vec, y, bins=metric_suite.mi_bins, xp=self.xp)
                        scores[combo]["mutual_info"] = val

        return scores
