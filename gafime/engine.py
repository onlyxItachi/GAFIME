"""Core engine orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from gafime.backend import gpu_cupy
from gafime.backend.memory_guard import plan_gpu_batches
from gafime.backend.metrics import MetricResult, score_features
from gafime.report.schema import build_report
from gafime.validation.permutation import permutation_test_max_pearson


@dataclass(frozen=True)
class EngineConfig:
    max_comb_size: int = 1
    max_combinations_per_k: int = 1000
    top_features_for_higher_k: int = 25
    max_generated_features: int = 10_000
    keep_in_vram: bool = False
    use_gpu: bool = True
    mi_bins: int = 16
    min_abs_pearson: float = 0.1
    min_mutual_information: float = 0.01
    permutation_tests: int = 50


class GAFIMEEngine:
    """Analyze feature-target relationships and report diagnostics."""

    def __init__(self, **kwargs) -> None:
        config = EngineConfig(**kwargs)
        if config.max_comb_size < 1:
            raise ValueError("max_comb_size must be >= 1")
        if config.max_combinations_per_k < 1:
            raise ValueError("max_combinations_per_k must be >= 1")
        if config.top_features_for_higher_k < 1:
            raise ValueError("top_features_for_higher_k must be >= 1")
        if config.max_generated_features < 1:
            raise ValueError("max_generated_features must be >= 1")
        if config.mi_bins < 2:
            raise ValueError("mi_bins must be >= 2")
        if config.permutation_tests < 0:
            raise ValueError("permutation_tests must be >= 0")
        self.config = config

    def analyze(
        self,
        features: np.ndarray,
        target: np.ndarray,
        feature_names: Iterable[str] | None = None,
    ) -> dict:
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)

        if x.ndim != 2:
            raise ValueError("features must be a 2D array")
        if y.ndim != 1:
            raise ValueError("target must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and target must share the same number of rows")

        num_features = x.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(num_features)]
        else:
            feature_names = list(feature_names)
            if len(feature_names) != num_features:
                raise ValueError("feature_names length must match number of columns")

        diagnostics: dict[str, object] = {}
        metric_results = self._score_unary(x, y, diagnostics)
        unary_results = self._format_unary_results(feature_names, metric_results)
        pairwise_results = self._score_pairwise(x, y, unary_results, feature_names)

        if self.config.permutation_tests > 0:
            perm_result = permutation_test_max_pearson(
                x,
                y,
                n_permutations=self.config.permutation_tests,
            )
            diagnostics["permutation_test"] = {
                "observed_max_abs_pearson": perm_result.observed,
                "p_value": perm_result.p_value,
                "n_permutations": perm_result.n_permutations,
            }

        status_message = self._signal_status(unary_results, pairwise_results)
        report = build_report(
            summary={
                "status": status_message,
                "num_samples": x.shape[0],
                "num_features": num_features,
            },
            config=asdict(self.config),
            unary_results=unary_results,
            pairwise_results=pairwise_results,
            diagnostics=diagnostics,
        )
        return report

    def _score_unary(self, x: np.ndarray, y: np.ndarray, diagnostics: dict[str, str]) -> list[MetricResult]:
        if not self.config.use_gpu:
            return score_features([x[:, i] for i in range(x.shape[1])], y, bins=self.config.mi_bins)

        status = gpu_cupy.gpu_available()
        if not status.available:
            diagnostics["gpu"] = f"GPU unavailable: {status.reason}"
            return score_features([x[:, i] for i in range(x.shape[1])], y, bins=self.config.mi_bins)

        bytes_per_row = (x.shape[1] + 1) * x.dtype.itemsize
        decision = plan_gpu_batches(
            total_rows=x.shape[0],
            bytes_per_row=bytes_per_row,
            keep_in_vram=self.config.keep_in_vram,
        )
        if not decision.use_gpu:
            diagnostics["gpu"] = f"GPU disabled: {decision.reason}"
            return score_features([x[:, i] for i in range(x.shape[1])], y, bins=self.config.mi_bins)

        if decision.reason:
            diagnostics["gpu"] = decision.reason

        if decision.batch_rows >= x.shape[0]:
            return self._score_unary_gpu_full(x, y)

        return self._score_unary_gpu_batched(x, y, decision.batch_rows)

    def _score_unary_gpu_full(self, x: np.ndarray, y: np.ndarray) -> list[MetricResult]:
        if gpu_cupy.cp is None:
            return score_features([x[:, i] for i in range(x.shape[1])], y, bins=self.config.mi_bins)
        x_gpu = gpu_cupy.cp.asarray(x)
        y_gpu = gpu_cupy.cp.asarray(y)
        return gpu_cupy.score_features_gpu(x_gpu, y_gpu, bins=self.config.mi_bins)

    def _score_unary_gpu_batched(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_rows: int,
    ) -> list[MetricResult]:
        if gpu_cupy.cp is None:
            return score_features([x[:, i] for i in range(x.shape[1])], y, bins=self.config.mi_bins)

        num_features = x.shape[1]
        total_rows = x.shape[0]
        sum_x = np.zeros(num_features)
        sum_y = 0.0
        sum_x2 = np.zeros(num_features)
        sum_y2 = 0.0
        sum_xy = np.zeros(num_features)

        x_edges = []
        y_edges = np.linspace(np.min(y), np.max(y), self.config.mi_bins + 1)
        for i in range(num_features):
            x_edges.append(np.linspace(np.min(x[:, i]), np.max(x[:, i]), self.config.mi_bins + 1))
        hist_counts = [np.zeros((self.config.mi_bins, self.config.mi_bins)) for _ in range(num_features)]

        for start in range(0, total_rows, batch_rows):
            end = min(start + batch_rows, total_rows)
            batch_x = x[start:end]
            batch_y = y[start:end]
            x_gpu = gpu_cupy.cp.asarray(batch_x)
            y_gpu = gpu_cupy.cp.asarray(batch_y)

            sum_x += gpu_cupy.cp.asnumpy(gpu_cupy.cp.sum(x_gpu, axis=0))
            sum_y += float(gpu_cupy.cp.sum(y_gpu))
            sum_x2 += gpu_cupy.cp.asnumpy(gpu_cupy.cp.sum(x_gpu ** 2, axis=0))
            sum_y2 += float(gpu_cupy.cp.sum(y_gpu ** 2))
            sum_xy += gpu_cupy.cp.asnumpy(gpu_cupy.cp.sum(x_gpu * y_gpu[:, None], axis=0))

            for i in range(num_features):
                hist_2d, _, _ = gpu_cupy.cp.histogram2d(
                    x_gpu[:, i],
                    y_gpu,
                    bins=(x_edges[i], y_edges),
                )
                hist_counts[i] += gpu_cupy.cp.asnumpy(hist_2d)

        results: list[MetricResult] = []
        n = float(total_rows)
        mean_x = sum_x / n
        mean_y = sum_y / n
        var_x = sum_x2 / n - mean_x ** 2
        var_y = sum_y2 / n - mean_y ** 2
        for i in range(num_features):
            if var_x[i] <= 0 or var_y <= 0:
                pearson = 0.0
            else:
                cov = sum_xy[i] / n - mean_x[i] * mean_y
                pearson = float(cov / (np.sqrt(var_x[i]) * np.sqrt(var_y)))
            mi = self._mi_from_hist(hist_counts[i])
            results.append(MetricResult(pearson=pearson, mutual_information=mi))
        return results

    def _score_pairwise(
        self,
        x: np.ndarray,
        y: np.ndarray,
        unary_results: list[dict],
        feature_names: list[str],
    ) -> list[dict]:
        if self.config.max_comb_size < 2:
            return []

        scores = []
        for idx, result in enumerate(unary_results):
            score = max(abs(result["pearson"]), result["mutual_information"])
            scores.append((score, idx))
        scores.sort(reverse=True)

        top_n = min(len(scores), self.config.top_features_for_higher_k)
        top_indices = [idx for _, idx in scores[:top_n]]

        pairs = list(combinations(top_indices, 2))
        limit = min(self.config.max_combinations_per_k, self.config.max_generated_features)
        pairs = pairs[:limit]

        if not pairs:
            return []

        interactions = []
        names = []
        for i, j in pairs:
            interactions.append(x[:, i] * x[:, j])
            names.append(f"{feature_names[i]}*{feature_names[j]}")

        results = score_features(interactions, y, bins=self.config.mi_bins)
        formatted = []
        for name, metrics in zip(names, results):
            formatted.append(
                {
                    "interaction": name,
                    "pearson": metrics.pearson,
                    "mutual_information": metrics.mutual_information,
                }
            )
        return formatted

    def _mi_from_hist(self, hist_2d: np.ndarray) -> float:
        total = hist_2d.sum()
        if total == 0:
            return 0.0
        pxy = hist_2d / total
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        px_py = px[:, None] * py[None, :]
        nonzero = pxy > 0
        return float(np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero])))

    def _format_unary_results(
        self,
        names: Iterable[str],
        results: list[MetricResult],
    ) -> list[dict]:
        formatted = []
        for name, metrics in zip(names, results):
            formatted.append(
                {
                    "feature": name,
                    "pearson": metrics.pearson,
                    "mutual_information": metrics.mutual_information,
                }
            )
        return formatted

    def _signal_status(self, unary_results: list[dict], pairwise_results: list[dict]) -> str:
        if not unary_results and not pairwise_results:
            return "No learnable feature-based signal detected for this target"

        for entry in unary_results:
            if abs(entry["pearson"]) >= self.config.min_abs_pearson:
                return "Signal detected"
            if entry["mutual_information"] >= self.config.min_mutual_information:
                return "Signal detected"
        for entry in pairwise_results:
            if abs(entry["pearson"]) >= self.config.min_abs_pearson:
                return "Signal detected"
            if entry["mutual_information"] >= self.config.min_mutual_information:
                return "Signal detected"
        return "No learnable feature-based signal detected for this target"
