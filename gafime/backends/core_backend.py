from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

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
        combos_list = [tuple(int(idx) for idx in combo) for combo in combos]
        if not combos_list:
            return {}

        if not hasattr(self.core, "score_combos_buffer"):
            raise ModuleNotFoundError(
                "gafime_core was loaded without the required native buffer scorer. "
                "Rebuild the local native extension."
            )
        ordered_combos = _cache_local_combo_order(combos_list)
        metrics = self.core.score_combos_buffer(
            X.buffer,
            y.buffer,
            ordered_combos,
            metric_suite.metric_names,
            metric_suite.mi_bins,
        )
        metric_names = metric_suite.metric_names
        scores: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for combo, row in zip(ordered_combos, metrics):
            scores[combo] = {name: float(row[i]) for i, name in enumerate(metric_names)}
        return scores

    def build_interaction_vector(self, X: NativeMatrix, combo: Tuple[int, ...]):
        return build_interaction_vector(X, combo)

    def find_decision_path_candidates(
        self,
        X: NativeMatrix,
        y: NativeVector,
        *,
        feature_ids: Iterable[int] | None,
        max_depth: int,
        max_paths: int,
        max_bins_per_feature: int,
        min_leaf: int,
        rounds: int,
        learning_rate: float,
    ) -> List[object]:
        if not hasattr(self.core, "find_decision_path_candidates"):
            raise ModuleNotFoundError(
                "gafime_core was loaded without the native decision_path finder. "
                "Rebuild the local native extension."
            )
        from ..decision_path import decision_path_candidate_from_record

        feature_ids_arg = None if feature_ids is None else [int(idx) for idx in feature_ids]
        records = self.core.find_decision_path_candidates(
            X.buffer,
            y.buffer,
            feature_ids_arg,
            int(max_depth),
            int(max_paths),
            int(max_bins_per_feature),
            int(min_leaf),
            int(rounds),
            float(learning_rate),
        )
        return [decision_path_candidate_from_record(record) for record in records]


def _cache_local_combo_order(combos: Sequence[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Order CPU equations through the Rust cache-local scheduler.

    CUDA already routes descriptors through ``subfunctions.BatchScheduler``.
    CPU/Core uses the same orchestration so adjacent native-core equations tend
    to reuse hot feature columns instead of arriving in raw planning order.
    """
    try:
        from .. import subfunctions
    except ImportError:
        return list(combos)

    scheduler = subfunctions.BatchScheduler(max_blocks=1024)
    feature_sets = [[int(feature) for feature in combo] for combo in combos]
    template_ids = [len(combo) for combo in combos]
    batches = scheduler.create_equation_batches(feature_sets, template_ids)
    ordered: List[Tuple[int, ...]] = []
    for batch in batches:
        for index in batch:
            ordered.append(combos[int(index)])
    return ordered
