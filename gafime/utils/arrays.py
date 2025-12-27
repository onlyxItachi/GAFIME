from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def coerce_inputs(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like of shape (n_samples, n_features).")
    if y_array.ndim != 1:
        raise ValueError("y must be a 1D array-like of shape (n_samples,).")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not np.issubdtype(X_array.dtype, np.number):
        raise ValueError("X must be numeric.")
    if not np.issubdtype(y_array.dtype, np.number):
        raise ValueError("y must be numeric.")
    if not np.isfinite(X_array).all() or not np.isfinite(y_array).all():
        raise ValueError("X and y must be finite (no NaN or inf).")

    X_array = X_array.astype(float, copy=False)
    y_array = y_array.astype(float, copy=False)

    if feature_names is None:
        feature_names_list = [f"f{i}" for i in range(X_array.shape[1])]
    else:
        feature_names_list = [str(name) for name in feature_names]
        if len(feature_names_list) != X_array.shape[1]:
            raise ValueError("feature_names length must match X's feature count.")

    return X_array, y_array, feature_names_list


def build_interaction_vector(X: np.ndarray, combo: Iterable[int], xp=np) -> np.ndarray:
    combo_tuple = tuple(int(idx) for idx in combo)
    if len(combo_tuple) == 1:
        return X[:, combo_tuple[0]]
    slice_data = X[:, combo_tuple]
    centered = slice_data - xp.mean(slice_data, axis=0)
    return xp.prod(centered, axis=1)
