"""Core orchestration entrypoint for GAFIME."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .backend.cpu_numpy import CpuBackend
from .backend.memory_guard import estimate_array_bytes, get_vram_info, should_keep_in_vram
from .fe.simple import SimpleFeatureGenerator
from .report.schema import FeatureScore, Report
from .validation.permutation import permutation_test

try:
    from .backend.gpu_cupy import GpuBackend
except Exception:  # pragma: no cover - optional backend
    GpuBackend = None


@dataclass
class EngineConfig:
    max_comb_size: int = 1
    max_combinations_per_k: int = 1000
    top_features_for_higher_k: int = 50
    max_generated_features: int = 5000
    keep_in_vram: bool = False
    mi_bins: int = 16
    use_gpu: bool = True
    permutation_runs: int = 20


class GAFIMEEngine:
    """Main orchestration class."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.backend = self._select_backend()
        self.feature_generator = SimpleFeatureGenerator()

    def _select_backend(self):
        if not self.config.use_gpu or GpuBackend is None:
            return CpuBackend(mi_bins=self.config.mi_bins)

        try:
            return GpuBackend(mi_bins=self.config.mi_bins)
        except Exception:
            return CpuBackend(mi_bins=self.config.mi_bins)

    def run(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run feature analysis and return report dict."""
        effective_keep_in_vram = self._effective_keep_in_vram(x)
        unary_results = self.backend.unary_metrics(x, y)
        unary_sorted = self.backend.sort_by_signal(unary_results)
        unary_scores = [
            FeatureScore(
                feature_indices=[item.feature_index],
                pearson=item.pearson,
                mutual_information=item.mutual_information,
            )
            for item in unary_sorted
        ]

        interaction_scores = []
        if self.config.max_comb_size >= 2:
            interaction_scores = self._pairwise_scores(x, y, unary_sorted)

        signal_detected = self._has_signal(unary_scores, interaction_scores)
        message = (
            "Signal detected"
            if signal_detected
            else "No learnable feature-based signal detected for this target"
        )

        validation = self._permutation_validation(x, y)

        report = Report(
            status="ok",
            signal_detected=signal_detected,
            message=message,
            unary_metrics=unary_scores,
            interaction_metrics=interaction_scores,
            parameters=self._report_parameters(effective_keep_in_vram),
            validation=validation,
        )
        return report.to_dict()

    def _pairwise_scores(self, x: np.ndarray, y: np.ndarray, unary_sorted) -> list[FeatureScore]:
        top_indices = [item.feature_index for item in unary_sorted[: self.config.top_features_for_higher_k]]
        max_pairs = min(self.config.max_combinations_per_k, self.config.max_generated_features)
        generated = list(self.feature_generator.generate_pairwise(x, top_indices, max_pairs))
        if not generated:
            return []
        interaction_matrix = np.column_stack([values for _, values in generated])
        results = self.backend.unary_metrics(interaction_matrix, y)
        scores = []
        for result, (indices, _) in zip(results, generated):
            scores.append(
                FeatureScore(
                    feature_indices=list(indices),
                    pearson=result.pearson,
                    mutual_information=result.mutual_information,
                )
            )
        return self.backend.sort_by_signal(scores)

    def _has_signal(self, unary_scores, interaction_scores) -> bool:
        def is_signal(item):
            return abs(item.pearson) > 0.05 or item.mutual_information > 0.01

        return any(is_signal(item) for item in unary_scores + interaction_scores)

    def _permutation_validation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        def score_fn(x_val: np.ndarray, y_val: np.ndarray) -> float:
            results = self.backend.unary_metrics(x_val, y_val)
            if not results:
                return 0.0
            sorted_results = self.backend.sort_by_signal(results)
            top = sorted_results[0]
            return abs(top.pearson) + top.mutual_information

        result = permutation_test(
            x,
            y,
            score_fn,
            permutations=self.config.permutation_runs,
        )
        return {
            "permutation_runs": self.config.permutation_runs,
            "observed_score": result.observed_score,
            "p_value": result.p_value,
        }

    def _effective_keep_in_vram(self, x: np.ndarray) -> bool:
        vram_info = get_vram_info()
        required_bytes = estimate_array_bytes(x.shape[0], x.shape[1])
        return should_keep_in_vram(self.config.keep_in_vram, required_bytes, vram_info)

    def _report_parameters(self, effective_keep_in_vram: bool) -> Dict[str, Any]:
        return {
            "max_comb_size": self.config.max_comb_size,
            "max_combinations_per_k": self.config.max_combinations_per_k,
            "top_features_for_higher_k": self.config.top_features_for_higher_k,
            "max_generated_features": self.config.max_generated_features,
            "keep_in_vram": self.config.keep_in_vram,
            "effective_keep_in_vram": effective_keep_in_vram,
            "mi_bins": self.config.mi_bins,
            "use_gpu": self.config.use_gpu,
            "permutation_runs": self.config.permutation_runs,
        }
