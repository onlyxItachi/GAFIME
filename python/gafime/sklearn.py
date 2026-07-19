from __future__ import annotations

from typing import Iterable, List, Sequence

from .api import GafimeEngine
from .config import ComputeBudget, EngineConfig


class GafimeSelector:
    """Native-list interaction transformer.

    This wrapper keeps the familiar fit/transform shape while letting the v1
    native engine choose and rank interaction candidates through the public API.
    It returns Python lists so callers can adapt the result to sklearn or another
    ML stack without exposing backend-owned buffers.
    """

    def __init__(
        self,
        k: int = 10,
        backend: str = "auto",
        metric: str = "pearson",
        operator: str = "multiply",
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> None:
        self.k = int(k)
        self.backend = backend
        self.metric = metric
        self.operator = operator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[float]):
        rows = [[float(value) for value in row] for row in X]
        target = [float(value) for value in y]
        if not rows or not rows[0]:
            raise ValueError("X must be a non-empty 2D numeric iterable.")
        if len(target) != len(rows):
            raise ValueError("X and y must have the same number of samples.")
        n_features = len(rows[0])
        if any(len(row) != n_features for row in rows):
            raise ValueError("X rows must all have the same feature count.")
        cfg = EngineConfig(
            metric_names=(self.metric,),
            backend=self.backend,
            budget=ComputeBudget(max_comb_size=2),
            permutation_tests=0,
            num_repeats=1,
        )
        report = GafimeEngine(cfg).analyze(rows, target)
        pairs = [
            result.combo
            for result in sorted(
                report.interactions,
                key=lambda item: abs(item.metrics.get(self.metric, 0.0)),
                reverse=True,
            )
            if len(result.combo) == 2
        ]
        self.top_interactions_ = pairs[: self.k]
        self.n_features_in_ = n_features
        return self

    def transform(self, X: Iterable[Iterable[float]]) -> List[List[float]]:
        rows = [[float(value) for value in row] for row in X]
        if not hasattr(self, "top_interactions_"):
            raise RuntimeError("GafimeSelector must be fitted before transform.")
        for row in rows:
            if len(row) != self.n_features_in_:
                raise ValueError("X feature count does not match fitted data.")
        return [row + self._interaction_values(row) for row in rows]

    def fit_transform(self, X: Iterable[Iterable[float]], y: Iterable[float]) -> List[List[float]]:
        return self.fit(X, y).transform(X)

    def _interaction_values(self, row: Sequence[float]) -> List[float]:
        values: List[float] = []
        for i, j in self.top_interactions_:
            a = float(row[i])
            b = float(row[j])
            if self.operator == "multiply":
                values.append(a * b)
            elif self.operator == "add":
                values.append(a + b)
            elif self.operator == "subtract":
                values.append(a - b)
            elif self.operator == "divide":
                values.append(a / (b if abs(b) > 1e-8 else 1e-8))
            else:
                raise ValueError("operator must be one of multiply, add, subtract, divide.")
        return values
