"""Simple feature generator."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence, Tuple

import numpy as np

from .base import BaseFeatureGenerator


class SimpleFeatureGenerator(BaseFeatureGenerator):
    """Generates multiplicative pairwise interactions."""

    def generate_pairwise(
        self,
        x: np.ndarray,
        top_indices: Sequence[int],
        max_pairs: int,
    ) -> Iterable[Tuple[Tuple[int, int], np.ndarray]]:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        pairs_yielded = 0
        for idx_a, idx_b in combinations(top_indices, 2):
            if pairs_yielded >= max_pairs:
                break
            values = x[:, idx_a] * x[:, idx_b]
            yield (idx_a, idx_b), values
            pairs_yielded += 1
