from __future__ import annotations

from typing import Dict, Iterable, Tuple

from . import cpu_metrics


SUPPORTED_METRICS = ("pearson", "spearman", "mutual_info", "r2")


class MetricSuite:
    def __init__(
        self,
        metric_names: Iterable[str],
        mi_bins: int = 96,
        ops=cpu_metrics,
    ) -> None:
        self.metric_names: Tuple[str, ...] = tuple(metric_names)
        self.mi_bins = mi_bins
        self.ops = ops
        self._validate()

    def _validate(self) -> None:
        unsupported = [name for name in self.metric_names if name not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(f"Unsupported metrics: {unsupported}.")

    def score(self, x, y) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name in self.metric_names:
            if name == "pearson":
                results[name] = self.ops.pearson_corr(x, y)
            elif name == "spearman":
                results[name] = self.ops.spearman_corr(x, y)
            elif name == "mutual_info":
                results[name] = self.ops.mutual_info(x, y, bins=self.mi_bins)
            elif name == "r2":
                results[name] = self.ops.linear_r2(x, y)
        return results
