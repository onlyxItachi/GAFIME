"""Core orchestration for GAFIME."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence

import numpy as np

from gafime.backend import gpu_cupy
from gafime.backend.metrics import mutual_information, pearson_corr, summarize_metrics
from gafime.backend.memory_guard import enforce_keep_in_vram, estimate_required_bytes
from gafime.report.export import build_report, report_to_dict
from gafime.report.schema import FeatureScore, InteractionScore
from gafime.validation.permutation import permutation_test


DEFAULT_DIAGNOSIS = "No learnable feature-based signal detected for this target"


@dataclass
class EngineConfig:
    max_comb_size: int = 2
    max_combinations_per_k: int = 500
    top_features_for_higher_k: int = 20
    max_generated_features: int = 100
    keep_in_vram: bool = True
    use_gpu: bool = True
    mi_bins: int = 10
    min_abs_pearson: float = 0.1
    min_mutual_information: float = 0.01
    permutation_rounds: int = 25
    permutation_seed: int | None = None


class GAFIMEEngine:
    """CPU-first engine implementation with optional GPU acceleration."""

    def __init__(self, **kwargs: object) -> None:
        self.config = EngineConfig(**kwargs)

    def analyze(
        self,
        features: np.ndarray,
        target: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> dict:
        features_arr = np.asarray(features, dtype=float)
        target_arr = np.asarray(target, dtype=float)
        if features_arr.ndim != 2:
            raise ValueError("features must be a 2D array (n_samples, n_features)")
        if target_arr.ndim != 1:
            raise ValueError("target must be a 1D array")
        if features_arr.shape[0] != target_arr.shape[0]:
            raise ValueError("features and target must have the same number of rows")

        names = self._resolve_feature_names(features_arr.shape[1], feature_names)

        scores, backend_info = self._score_unary(features_arr, target_arr, names)
        interaction_scores = self._score_interactions(features_arr, target_arr, names, backend_info)
        max_abs_pearson = max((abs(score.pearson) for score in scores), default=0.0)
        max_mi = max((score.mutual_information for score in scores), default=0.0)

        perm_result = self._permutation_summary(features_arr, target_arr)
        diagnosis = self._diagnose(max_abs_pearson, max_mi, perm_result.pvalue)
        report = build_report(
            diagnosis=diagnosis,
            max_abs_pearson=max_abs_pearson,
            max_mutual_information=max_mi,
            feature_scores=scores,
            interaction_scores=interaction_scores,
            config={
                **self._config_dict(),
                **backend_info,
            },
            permutation_pvalue=perm_result.pvalue,
        )
        return report_to_dict(report)

    def _score_unary(
        self,
        features: np.ndarray,
        target: np.ndarray,
        names: Sequence[str],
    ) -> tuple[List[FeatureScore], dict]:
        backend_info: dict = {"gpu_used": False, "gpu_fallback_reason": None}
        keep_in_vram = self.config.keep_in_vram
        if self.config.use_gpu:
            required_bytes = estimate_required_bytes(features.shape, target.shape, features.itemsize)
            keep_in_vram, reason = enforce_keep_in_vram(
                keep_in_vram=keep_in_vram,
                required_bytes=required_bytes,
            )
            if not keep_in_vram:
                backend_info["gpu_fallback_reason"] = reason
            elif gpu_cupy.is_available():
                backend_info["gpu_used"] = True
                backend_info["gpu_fallback_reason"] = None
                scores = gpu_cupy.score_unary(features, target, names, bins=self.config.mi_bins)
                backend_info["keep_in_vram_effective"] = True
                return scores, backend_info
            else:
                backend_info["gpu_fallback_reason"] = "gpu_unavailable"

        backend_info["keep_in_vram_effective"] = False
        scores: List[FeatureScore] = []
        for idx, name in enumerate(names):
            feature = features[:, idx]
            pearson = pearson_corr(feature, target)
            mi = mutual_information(feature, target, bins=self.config.mi_bins)
            pearson_value, mi_value = summarize_metrics(pearson, mi)
            scores.append(
                FeatureScore(
                    name=name,
                    pearson=pearson_value,
                    mutual_information=mi_value,
                )
            )
        return scores, backend_info

    def _score_interactions(
        self,
        features: np.ndarray,
        target: np.ndarray,
        names: Sequence[str],
        backend_info: dict,
    ) -> List[InteractionScore]:
        if self.config.max_comb_size < 2:
            return []

        ranked = self._rank_features_for_interactions(names, features, target)
        top_names = [name for name, _ in ranked[: self.config.top_features_for_higher_k]]
        if len(top_names) < 2:
            return []

        name_to_idx = {name: idx for idx, name in enumerate(names)}
        pairs = list(combinations(top_names, 2))[: self.config.max_combinations_per_k]
        interaction_scores: List[InteractionScore] = []
        for name_a, name_b in pairs:
            idx_a = name_to_idx[name_a]
            idx_b = name_to_idx[name_b]
            interaction = features[:, idx_a] * features[:, idx_b]
            if backend_info.get("gpu_used"):
                pearson = gpu_cupy.pearson_corr(interaction, target)
                mi = gpu_cupy.mutual_information(interaction, target, bins=self.config.mi_bins)
            else:
                pearson = pearson_corr(interaction, target)
                mi = mutual_information(interaction, target, bins=self.config.mi_bins)
            pearson_value, mi_value = summarize_metrics(pearson, mi)
            interaction_scores.append(
                InteractionScore(
                    features=[name_a, name_b],
                    pearson=pearson_value,
                    mutual_information=mi_value,
                )
            )
        return interaction_scores

    def _rank_features_for_interactions(
        self, names: Sequence[str], features: np.ndarray, target: np.ndarray
    ) -> List[tuple[str, float]]:
        ranked: List[tuple[str, float]] = []
        for idx, name in enumerate(names):
            feature = features[:, idx]
            pearson = pearson_corr(feature, target)
            mi = mutual_information(feature, target, bins=self.config.mi_bins)
            pearson_value, mi_value = summarize_metrics(pearson, mi)
            score = abs(pearson_value) + mi_value
            ranked.append((name, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _permutation_summary(self, features: np.ndarray, target: np.ndarray):
        rng = np.random.default_rng(self.config.permutation_seed)

        def score_fn(shuffled_target: np.ndarray) -> float:
            scores = []
            for idx in range(features.shape[1]):
                pearson = pearson_corr(features[:, idx], shuffled_target)
                mi = mutual_information(features[:, idx], shuffled_target, bins=self.config.mi_bins)
                pearson_value, mi_value = summarize_metrics(pearson, mi)
                scores.append(abs(pearson_value) + mi_value)
            return float(max(scores) if scores else 0.0)

        return permutation_test(
            score_fn,
            target,
            n_permutations=self.config.permutation_rounds,
            rng=rng,
        )

    def _diagnose(self, max_abs_pearson: float, max_mi: float, pvalue: float | None) -> str:
        if max_abs_pearson < self.config.min_abs_pearson and max_mi < self.config.min_mutual_information:
            return DEFAULT_DIAGNOSIS
        if pvalue is not None and pvalue > 0.2:
            return DEFAULT_DIAGNOSIS
        return "Signal detected in unary features"

    def _resolve_feature_names(
        self, feature_count: int, feature_names: Sequence[str] | None
    ) -> List[str]:
        if feature_names is None:
            return [f"feature_{idx}" for idx in range(feature_count)]
        if len(feature_names) != feature_count:
            raise ValueError("feature_names must match number of features")
        return list(feature_names)

    def _config_dict(self) -> dict:
        return {
            "max_comb_size": self.config.max_comb_size,
            "max_combinations_per_k": self.config.max_combinations_per_k,
            "top_features_for_higher_k": self.config.top_features_for_higher_k,
            "max_generated_features": self.config.max_generated_features,
            "keep_in_vram": self.config.keep_in_vram,
            "use_gpu": self.config.use_gpu,
            "mi_bins": self.config.mi_bins,
            "min_abs_pearson": self.config.min_abs_pearson,
            "min_mutual_information": self.config.min_mutual_information,
            "permutation_rounds": self.config.permutation_rounds,
            "permutation_seed": self.config.permutation_seed,
        }
