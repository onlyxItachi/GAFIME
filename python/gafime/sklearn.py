from __future__ import annotations

import struct
from typing import Iterable, List, Sequence

from .api import GafimeEngine
from .config import ComputeBudget, EngineConfig


class GafimeSelector:
    """Discover native pair candidates and append chosen interaction columns.

    This wrapper keeps the familiar fit/transform shape while letting the v1
    native engine choose and rank interaction candidates through the public API.
    It returns Python lists so callers can adapt the result to sklearn or another
    ML stack without exposing backend-owned buffers.

    ``k`` is the maximum number of pair interactions retained and must be
    non-negative. Invalid values fail before discovery or materialization.
    ``backend``, ``metric``, and keyword-only ``precision`` are passed to the native
    discovery run.  ``operator`` controls only post-discovery materialization
    and accepts ``multiply``, ``add``, ``subtract``, or ``divide``; division
    uses a positive ``1e-8`` guard in the selected pointwise dtype.
    ``n_jobs`` and ``verbose`` are retained constructor parameters for
    historical/scikit-learn cloning compatibility but do not currently alter
    native scheduling or logging.

    Put the selector inside each training fold: fitting on the full dataset
    before cross-validation leaks target-guided interaction discovery.
    """

    def __init__(
        self,
        k: int = 10,
        backend: str = "auto",
        metric: str = "pearson",
        operator: str = "multiply",
        n_jobs: int = -1,
        verbose: bool = False,
        *,
        precision: str = "mixed",
    ) -> None:
        self.k = self._validate_k(k)
        self.backend = backend
        self.metric = metric
        self.operator = operator
        self.n_jobs = n_jobs
        self.verbose = verbose
        from ._precision import normalize_precision

        # Scikit-learn requires constructor parameters to be stored without
        # replacing their object identity so clone() can verify the estimator
        # contract. Validate here, then let EngineConfig retain the canonical
        # normalized execution value used by fit/transform.
        normalize_precision(precision)
        self.precision = precision

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[float]):
        """Discover and retain up to ``k`` ranked pair interactions.

        ``X`` must be a non-empty rectangular numeric matrix and ``y`` must
        contain one value per row.  Backend/profile compatibility is validated
        before input coercion; actual payload/device resolution occurs later in
        native analysis and may still fail closed.  Fitting sets
        ``top_interactions_`` and ``n_features_in_`` and returns ``self``.
        """

        cfg = EngineConfig(
            metric_names=(self.metric,),
            backend=self.backend,
            budget=ComputeBudget(max_comb_size=2),
            permutation_tests=0,
            num_repeats=1,
            precision=self.precision,
        )
        # Reject impossible backend/profile pairs before touching caller-owned
        # values. In particular, explicit Metal mixed/fp64 must not coerce an
        # input or trigger payload discovery before its capability error.
        from .v1_adapter import _validate_precision_config

        _validate_precision_config(cfg)
        self._effective_precision_ = cfg.precision
        rows = [[float(value) for value in row] for row in X]
        target = [float(value) for value in y]
        if not rows or not rows[0]:
            raise ValueError("X must be a non-empty 2D numeric iterable.")
        if len(target) != len(rows):
            raise ValueError("X and y must have the same number of samples.")
        n_features = len(rows[0])
        if any(len(row) != n_features for row in rows):
            raise ValueError("X rows must all have the same feature count.")
        report = GafimeEngine(cfg).analyze(rows, target)
        pairs = [
            result.combo
            for result in report.interactions.ranked(metric_name=self.metric)
            if len(result.combo) == 2
        ]
        self.top_interactions_ = pairs[: self.k]
        self.n_features_in_ = n_features
        return self

    def transform(self, X: Iterable[Iterable[float]]) -> List[List[float]]:
        """Append fitted interaction columns and return Python rows.

        :meth:`fit` must have run and every row must match ``n_features_in_``.
        The original columns remain first; generated pair columns follow in
        ``top_interactions_`` order.
        """

        rows = [[float(value) for value in row] for row in X]
        if not hasattr(self, "top_interactions_"):
            raise RuntimeError("GafimeSelector must be fitted before transform.")
        for row in rows:
            if len(row) != self.n_features_in_:
                raise ValueError("X feature count does not match fitted data.")
        return [row + self._interaction_values(row) for row in rows]

    def fit_transform(
        self, X: Iterable[Iterable[float]], y: Iterable[float]
    ) -> List[List[float]]:
        """Fit on ``X, y`` and transform the same rows."""

        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return constructor parameters for scikit-learn cloning.

        ``deep`` is accepted for estimator compatibility; the selector has no
        nested estimator parameters of its own.
        """

        del deep
        return {
            "k": self.k,
            "backend": self.backend,
            "metric": self.metric,
            "operator": self.operator,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "precision": self.precision,
        }

    def set_params(self, **params: object):
        """Set known constructor parameters and return ``self``.

        Unknown names raise :class:`ValueError`. Precision and ``k`` are
        validated at assignment; backend/metric/operator validity is enforced
        by fit or transform at the boundary where it becomes relevant.
        """

        valid = self.get_params(deep=False)
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for GafimeSelector. "
                    f"Valid parameters are: {', '.join(sorted(valid))}."
                )
            if name == "precision":
                from ._precision import normalize_precision

                normalize_precision(value)
            if name == "k":
                value = self._validate_k(value)
            setattr(self, name, value)
        return self

    @staticmethod
    def _validate_k(value: object) -> int:
        k = int(value)
        if k < 0:
            raise ValueError("k must be non-negative.")
        return k

    def _interaction_values(self, row: Sequence[float]) -> List[float]:
        values: List[float] = []
        for i, j in self.top_interactions_:
            a = self._pointwise_value(row[i])
            b = self._pointwise_value(row[j])
            if self.operator == "multiply":
                value = a * b
            elif self.operator == "add":
                value = a + b
            elif self.operator == "subtract":
                value = a - b
            elif self.operator == "divide":
                epsilon = self._pointwise_value(1e-8)
                value = a / (b if abs(b) > epsilon else epsilon)
            else:
                raise ValueError(
                    "operator must be one of multiply, add, subtract, divide."
                )
            values.append(self._pointwise_value(value))
        return values

    def _pointwise_value(self, value: float) -> float:
        value = float(value)
        precision = getattr(self, "_effective_precision_", self.precision)
        if precision == "fp64":
            return value
        return struct.unpack("<f", struct.pack("<f", value))[0]
