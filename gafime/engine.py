"""Main orchestrator for GAFIME."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List

import numpy as np

from gafime.backend import gpu_cupy
from gafime.backend.memory_guard import enforce_keep_in_vram
from gafime.backend.metrics import mutual_information_binned, pearson_correlation
from gafime.validation.permutation import permutation_test


@dataclass
class EngineConfig:
    max_comb_size: int = 2
    max_combinations_per_k: int = 1000
    top_features_for_higher_k: int = 20
    max_generated_features: int = 1000
    keep_in_vram: bool = False
    use_gpu: bool = True
    mi_bins: int = 16
    min_abs_corr: float = 0.1
    min_mi: float = 0.01
    permutation_runs: int = 100
    permutation_seed: int | None = 0


class GAFIMEEngine:
    """Core analysis engine (CPU-first with optional GPU acceleration)."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.config.keep_in_vram = enforce_keep_in_vram(self.config.keep_in_vram)

    def analyze(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching row counts")

        warnings: List[str] = []
        use_gpu = self.config.use_gpu and gpu_cupy.is_available()
        if self.config.use_gpu and not gpu_cupy.is_available():
            warnings.append("GPU requested but CuPy is unavailable; falling back to CPU.")

        backend = "gpu" if use_gpu else "cpu"

        if use_gpu:
            try:
                unary_scores = self._score_unary_gpu(X, y)
            except Exception as exc:
                warnings.append(
                    f"GPU execution failed ({exc}); falling back to CPU."
                )
                backend = "cpu"
                unary_scores = self._score_unary_cpu(X, y)
        else:
            unary_scores = self._score_unary_cpu(X, y)

        interaction_scores: List[Dict[str, Any]] = []
        if self.config.max_comb_size >= 2:
            try:
                interaction_scores = self._score_pairwise(X, y, unary_scores, backend)
            except Exception as exc:
                warnings.append(f"Pairwise scoring failed ({exc}); skipping interactions.")
        if self.config.max_comb_size > 2:
            warnings.append("k > 2 interactions are not implemented in this version.")

        permutation_result = self._run_permutation_test(X, y, unary_scores)

        signal_detected = any(
            (abs(item["pearson"]) >= self.config.min_abs_corr)
            or (item["mutual_info"] >= self.config.min_mi)
            for item in unary_scores
        )

        summary = (
            "Learnable feature-based signal detected."
            if signal_detected
            else "No learnable feature-based signal detected for this target"
        )

        return {
            "summary": summary,
            "config": self._config_dict(),
            "warnings": warnings,
            "backend": backend,
            "unary_scores": unary_scores,
            "pairwise_scores": interaction_scores,
            "permutation_test": permutation_result,
        }

    def _score_unary_cpu(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        scores: List[Dict[str, Any]] = []
        for idx in range(X.shape[1]):
            feature = X[:, idx]
            pearson = pearson_correlation(feature, y)
            mi = mutual_information_binned(feature, y, bins=self.config.mi_bins)
            scores.append(
                {
                    "feature_index": idx,
                    "pearson": pearson,
                    "mutual_info": mi,
                    "combined_score": self._combined_score(pearson, mi),
                }
            )
        return scores

    def _score_unary_gpu(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        scores: List[Dict[str, Any]] = []
        for idx in range(X.shape[1]):
            feature = X[:, idx]
            pearson = gpu_cupy.pearson_correlation(feature, y)
            mi = gpu_cupy.mutual_information_binned(feature, y, bins=self.config.mi_bins)
            scores.append(
                {
                    "feature_index": idx,
                    "pearson": pearson,
                    "mutual_info": mi,
                    "combined_score": self._combined_score(pearson, mi),
                }
            )
        return scores

    def _score_pairwise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unary_scores: List[Dict[str, Any]],
        backend: str,
    ) -> List[Dict[str, Any]]:
        sorted_features = sorted(
            unary_scores, key=lambda item: item["combined_score"], reverse=True
        )
        top_count = min(self.config.top_features_for_higher_k, len(sorted_features))
        top_indices = [item["feature_index"] for item in sorted_features[:top_count]]

        results: List[Dict[str, Any]] = []
        for idx_a, idx_b in combinations(top_indices, 2):
            if len(results) >= self.config.max_combinations_per_k:
                break
            interaction_feature = X[:, idx_a] * X[:, idx_b]
            if backend == "gpu":
                pearson = gpu_cupy.pearson_correlation(interaction_feature, y)
                mi = gpu_cupy.mutual_information_binned(
                    interaction_feature, y, bins=self.config.mi_bins
                )
            else:
                pearson = pearson_correlation(interaction_feature, y)
                mi = mutual_information_binned(
                    interaction_feature, y, bins=self.config.mi_bins
                )
            results.append(
                {
                    "feature_indices": (idx_a, idx_b),
                    "pearson": pearson,
                    "mutual_info": mi,
                    "combined_score": self._combined_score(pearson, mi),
                }
            )
        return results

    def _run_permutation_test(
        self, X: np.ndarray, y: np.ndarray, unary_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not unary_scores:
            return {}
        best = max(unary_scores, key=lambda item: item["combined_score"])
        feature_idx = best["feature_index"]
        feature = X[:, feature_idx]
        result = permutation_test(
            feature,
            y,
            pearson_correlation,
            n_permutations=self.config.permutation_runs,
            seed=self.config.permutation_seed,
        )
        return {"feature_index": feature_idx, **result}

    @staticmethod
    def _combined_score(pearson: float, mi: float) -> float:
        return max(abs(pearson), mi)

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "max_comb_size": self.config.max_comb_size,
            "max_combinations_per_k": self.config.max_combinations_per_k,
            "top_features_for_higher_k": self.config.top_features_for_higher_k,
            "max_generated_features": self.config.max_generated_features,
            "keep_in_vram": self.config.keep_in_vram,
            "use_gpu": self.config.use_gpu,
            "mi_bins": self.config.mi_bins,
            "min_abs_corr": self.config.min_abs_corr,
            "min_mi": self.config.min_mi,
            "permutation_runs": self.config.permutation_runs,
            "permutation_seed": self.config.permutation_seed,
        }
