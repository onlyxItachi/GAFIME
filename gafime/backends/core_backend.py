from __future__ import annotations

from typing import Dict, Iterable, Tuple

from ..metrics import MetricSuite
from ..native_data import NativeMatrix, NativeVector, build_interaction_vector
from .base import Backend


class CoreBackend(Backend):
    name = "core"
    device_label = "cpu"
    is_gpu = False

    def __init__(self) -> None:
        try:
            # When distributed via Wheel, the pybind .so is packaged inside the `gafime` folder
            from gafime import gafime_core
        except ImportError:
            try:
                # Fallback for local edit mode (pip install -e .) where it lives at the root
                import gafime_core
            except ImportError:
                raise ModuleNotFoundError(
                    "gafime_core package found but compiled extension is missing. "
                    "Ensure 'gafime_core.so' or '.pyd' is built and present."
                )

        # Validate the module actually has the compiled C++ extension.
        if not hasattr(gafime_core, "score_combos_buffer"):
            raise ModuleNotFoundError(
                "gafime_core found but does not contain expected C++ bindings. "
                "Ensure C++ core is properly compiled."
            )

        super().__init__()
        self.core = gafime_core

    def score_combos(
        self,
        X: NativeMatrix,
        y: NativeVector,
        combos: Iterable[Tuple[int, ...]],
        metric_suite: MetricSuite,
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        combos_list = list(combos)
        if not combos_list:
            return {}

        if not hasattr(self.core, "score_combos_buffer"):
            raise ModuleNotFoundError(
                "gafime_core was loaded without the required native buffer scorer. "
                "Rebuild the local native extension."
            )
        metrics = self.core.score_combos_buffer(
            X.buffer,
            y.buffer,
            combos_list,
            metric_suite.metric_names,
            metric_suite.mi_bins,
        )
        metric_names = metric_suite.metric_names
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for combo, row in zip(combos_list, metrics):
            scores[combo] = {name: float(row[i]) for i, name in enumerate(metric_names)}
        return scores

    def build_interaction_vector(self, X: NativeMatrix, combo: Tuple[int, ...]):
        return build_interaction_vector(X, combo)
