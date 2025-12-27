"""Base feature generator interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Tuple

import numpy as np


class BaseFeatureGenerator(ABC):
    @abstractmethod
    def generate_pairwise(
        self,
        x: np.ndarray,
        top_indices: Sequence[int],
        max_pairs: int,
    ) -> Iterable[Tuple[Tuple[int, int], np.ndarray]]:
        """Yield pairwise interactions as (feature_indices, values)."""
