from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .config import ComputeBudget, EngineConfig
from .engine import GafimeEngine
from .native_data import coerce_inputs


class GafimeSelector:
    """Native-list interaction transformer.

    The v0.4.5 native-only spine uses Python-native containers. This class keeps the
    familiar fit/transform shape but returns Python lists so callers can decide
    how to adapt the result to sklearn or another ML stack.
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
        X_matrix, y_vector, _ = coerce_inputs(X, y)
        cfg = EngineConfig(
            metric_names=(self.metric,),
            backend=self.backend,
            budget=ComputeBudget(max_comb_size=2),
            permutation_tests=0,
            num_repeats=1,
        )
        report = GafimeEngine(cfg).analyze(X_matrix.rows(), y_vector.to_list())
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
        self.n_features_in_ = X_matrix.n_features
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
