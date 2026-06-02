from __future__ import annotations

from array import array
import importlib
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, List, Tuple


NumberRows = Iterable[Iterable[float]]


def _core():
    try:
        return importlib.import_module("gafime.gafime_core")
    except ImportError:
        return importlib.import_module("gafime_core")


def _sequence_or_list(values: Any, label: str):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be numeric, not a string.")
    if hasattr(values, "__len__") and hasattr(values, "__getitem__"):
        return values
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable numeric sequence.") from exc


def _matrix_input(X: Any):
    if hasattr(X, "to_dicts") and hasattr(X, "columns"):
        columns = [str(col) for col in X.columns]
        return [[row[col] for col in columns] for row in X.to_dicts()]
    return _sequence_or_list(X, "X")


@dataclass(frozen=True)
class NativeVector:
    _buffer: Any

    @classmethod
    def from_values(cls, values: Any) -> "NativeVector":
        values = _sequence_or_list(values, "y")
        return cls(_core().NativeVectorBuffer(values))

    @property
    def data(self) -> Tuple[float, ...]:
        return tuple(float(value) for value in self._buffer.to_list())

    @property
    def shape(self) -> Tuple[int]:
        return (len(self),)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self.to_list())

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.data[item]
        return float(self._buffer[int(item)])

    def to_list(self) -> List[float]:
        return [float(value) for value in self._buffer.to_list()]

    def as_array(self, typecode: str = "f") -> array:
        return array(typecode, memoryview(self._buffer))

    @property
    def buffer(self):
        return self._buffer

    def select(self, indices: Sequence[int]) -> "NativeVector":
        return NativeVector(self._buffer.select(list(indices)))

    def shuffled(self, rng: random.Random) -> "NativeVector":
        values = self.to_list()
        rng.shuffle(values)
        return NativeVector.from_values(values)

    @property
    def nbytes(self) -> int:
        return int(self._buffer.nbytes)


@dataclass(frozen=True)
class NativeMatrix:
    _buffer: Any

    @classmethod
    def from_rows(cls, rows: Any) -> "NativeMatrix":
        rows = _matrix_input(rows)
        return cls(_core().NativeMatrixBuffer(rows))

    @property
    def data(self) -> Tuple[float, ...]:
        return tuple(float(value) for value in self._buffer.to_list())

    @property
    def n_samples(self) -> int:
        return int(self._buffer.n_samples)

    @property
    def n_features(self) -> int:
        return int(self._buffer.n_features)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_samples, self.n_features)

    @property
    def nbytes(self) -> int:
        return int(self._buffer.nbytes)

    @property
    def buffer(self):
        return self._buffer

    def value(self, row: int, col: int) -> float:
        return float(self._buffer.value(int(row), int(col)))

    def row(self, row: int) -> Tuple[float, ...]:
        return tuple(float(value) for value in self._buffer.row(int(row)))

    def rows(self) -> List[List[float]]:
        return [[float(value) for value in row] for row in self._buffer.rows()]

    def column(self, col: int) -> List[float]:
        return [float(value) for value in self._buffer.column(int(col))]

    def column_buffer(self, col: int):
        return self._buffer.column_buffer(int(col))

    def columns(self, cols: Sequence[int]) -> List[List[float]]:
        return [self.column(col) for col in cols]

    def select_rows(self, indices: Sequence[int]) -> "NativeMatrix":
        return NativeMatrix(self._buffer.select_rows(list(indices)))

    def as_row_major_array(self, typecode: str = "f") -> array:
        return array(typecode, memoryview(self._buffer))

    def as_feature_major_array(self, typecode: str = "f") -> array:
        if typecode == "f":
            return array(typecode, memoryview(self._buffer.feature_major()))
        return array(typecode, self._buffer.feature_major().to_list())

    def centered_column_array(self, feature: int, mean_value: float, typecode: str = "f") -> array:
        vector = self._buffer.centered_column(int(feature), float(mean_value))
        return array(typecode, memoryview(vector))

    def centered_column_buffer(self, feature: int, mean_value: float):
        return self._buffer.centered_column(int(feature), float(mean_value))

    def feature_major_buffer(self):
        return self._buffer.feature_major()


def coerce_inputs(
    X: Any,
    y: Any,
    feature_names: Sequence[str] | None = None,
) -> Tuple[NativeMatrix, NativeVector, List[str]]:
    try:
        X_native = NativeMatrix.from_rows(X)
        y_native = NativeVector.from_values(y)
    except TypeError as exc:
        raise ValueError("X must be a 2D iterable and y must be a 1D iterable of numeric values.") from exc

    if X_native.n_samples == 0:
        raise ValueError("X must contain at least one sample.")
    if X_native.n_features == 0:
        raise ValueError("X must contain at least one feature.")
    if len(y_native) != X_native.n_samples:
        raise ValueError("X and y must have the same number of samples.")

    if feature_names is None:
        names = [f"f{i}" for i in range(X_native.n_features)]
    else:
        names = [str(name) for name in feature_names]
        if len(names) != X_native.n_features:
            raise ValueError("feature_names length must match X's feature count.")

    return X_native, y_native, names


def build_interaction_vector(X: NativeMatrix, combo: Iterable[int]) -> List[float]:
    combo_tuple = tuple(int(idx) for idx in combo)
    if not combo_tuple:
        raise ValueError("combo must contain at least one feature index.")
    for idx in combo_tuple:
        if idx < 0 or idx >= X.n_features:
            raise ValueError("combo index out of feature bounds.")
    if len(combo_tuple) == 1:
        return X.column(combo_tuple[0])

    means = column_means(X)
    values: List[float] = []
    for row in range(X.n_samples):
        product = 1.0
        for feature in combo_tuple:
            product *= X.value(row, feature) - means[feature]
        values.append(product)
    return values


def column_means(X: NativeMatrix) -> List[float]:
    return [float(value) for value in X.buffer.column_means()]


def column_std(X: NativeMatrix, feature: int) -> float:
    return float(X.buffer.column_std(int(feature)))


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / float(len(values)) if values else 0.0


def variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.fsum((value - mu) * (value - mu) for value in values)


def std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(variance(values) / float(len(values)))


def quantiles(values: Sequence[float], qs: Sequence[float]) -> List[float]:
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    if n == 0:
        return []
    out: List[float] = []
    for q in qs:
        q = min(max(float(q), 0.0), 1.0)
        pos = q * float(n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(ordered[lo])
        else:
            weight = pos - float(lo)
            out.append(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)
    return out


def is_finite_sequence(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def bootstrap_indices(n_samples: int, rng: random.Random) -> List[int]:
    return [rng.randrange(n_samples) for _ in range(n_samples)]


def permutation(values: NativeVector, rng: random.Random) -> NativeVector:
    return values.shuffled(rng)
