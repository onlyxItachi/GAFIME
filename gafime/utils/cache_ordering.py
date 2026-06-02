from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


def order_indices_cache_aware(
    feature_sets: Sequence[Sequence[int]],
    template_ids: Optional[Sequence[int]] = None,
    max_blocks: int = 1024,
) -> List[int]:
    """Return a cache-locality-aware order for equation descriptors.

    Rust owns the primary implementation through ``subfunctions.BatchScheduler``.
    The Python fallback mirrors the same policy closely enough for source
    installs where the Rust helper is unavailable. This is launch ordering only:
    it does not request CUDA L2 persistence or reserve any cache partition.
    """
    if template_ids is not None and len(template_ids) != len(feature_sets):
        raise ValueError("template_ids must have the same length as feature_sets")
    if not feature_sets:
        return []

    normalized = _normalize_feature_sets(feature_sets)
    template_list = None if template_ids is None else [int(value) for value in template_ids]

    try:
        from .. import subfunctions

        scheduler = subfunctions.BatchScheduler(max_blocks=max_blocks)
        order = [
            int(index)
            for index in scheduler.order_equations(normalized, template_list)
        ]
        if _is_valid_order(order, len(normalized)):
            return order
    except Exception:
        pass

    return _fallback_order_indices(normalized, template_list)


def order_items_cache_aware(
    items: Sequence[T],
    feature_sets: Sequence[Sequence[int]],
    template_ids: Optional[Sequence[int]] = None,
    max_blocks: int = 1024,
) -> List[T]:
    if len(items) != len(feature_sets):
        raise ValueError("items and feature_sets must have the same length")
    order = order_indices_cache_aware(
        feature_sets,
        template_ids=template_ids,
        max_blocks=max_blocks,
    )
    return [items[index] for index in order]


def batch_items_by_template_cache_aware(
    items: Sequence[T],
    feature_sets: Sequence[Sequence[int]],
    template_ids: Sequence[int],
    max_blocks: int = 1024,
) -> List[Tuple[int, List[T]]]:
    """Return homogeneous-template, cache-local batches.

    Template IDs represent execution shapes, not just sort hints. Rust owns the
    primary batching path so native launches can switch on a single template per
    batch. The Python fallback preserves the same invariant.
    """
    if len(items) != len(feature_sets) or len(items) != len(template_ids):
        raise ValueError("items, feature_sets, and template_ids must have the same length")
    if not items:
        return []

    normalized = _normalize_feature_sets(feature_sets)
    template_list = [int(value) for value in template_ids]

    try:
        from .. import subfunctions

        scheduler = subfunctions.BatchScheduler(max_blocks=max_blocks)
        raw_batches = scheduler.create_template_batches(
            normalized,
            template_list,
        )
        batches: List[Tuple[int, List[T]]] = []
        seen: List[int] = []
        for template_id, indices in raw_batches:
            indices_list = [int(index) for index in indices]
            if not indices_list:
                continue
            if any(index < 0 or index >= len(items) for index in indices_list):
                raise ValueError("Rust scheduler returned an out-of-range index")
            if any(template_list[index] != int(template_id) for index in indices_list):
                raise ValueError("Rust scheduler returned a mixed-template batch")
            seen.extend(indices_list)
            batches.append((int(template_id), [items[index] for index in indices_list]))
        if sorted(seen) == list(range(len(items))):
            return batches
    except Exception:
        pass

    return _fallback_template_batches(items, normalized, template_list, max_blocks=max_blocks)


def _normalize_feature_sets(feature_sets: Sequence[Sequence[int]]) -> List[List[int]]:
    normalized: List[List[int]] = []
    for features in feature_sets:
        normalized.append(sorted({int(feature) for feature in features}))
    return normalized


def _is_valid_order(order: Sequence[int], n_items: int) -> bool:
    return len(order) == n_items and sorted(order) == list(range(n_items))


def _fallback_order_indices(
    feature_sets: Sequence[Sequence[int]],
    template_ids: Optional[Sequence[int]],
) -> List[int]:
    frequencies: Counter[int] = Counter()
    for features in feature_sets:
        frequencies.update(features)

    def key(index: int):
        features = tuple(feature_sets[index])
        template_id = 0 if template_ids is None else int(template_ids[index])
        if not features:
            return (0, 0, 0, (), (), template_id, index)

        anchor = max(features, key=lambda feature: (frequencies[feature], -feature))
        rest = tuple(
            sorted(
                (feature for feature in features if feature != anchor),
                key=lambda feature: (-frequencies[feature], feature),
            )
        )
        return (
            -frequencies[anchor],
            anchor,
            len(features),
            rest,
            features,
            template_id,
            index,
        )

    return sorted(range(len(feature_sets)), key=key)


def _fallback_template_batches(
    items: Sequence[T],
    feature_sets: Sequence[Sequence[int]],
    template_ids: Sequence[int],
    *,
    max_blocks: int,
) -> List[Tuple[int, List[T]]]:
    batch_size = max(1, min(int(max_blocks), 1024))
    batches: List[Tuple[int, List[T]]] = []
    for template_id in sorted(set(template_ids)):
        indices = [
            index
            for index, current_template in enumerate(template_ids)
            if current_template == template_id
        ]
        local_feature_sets = [feature_sets[index] for index in indices]
        local_order = _fallback_order_indices(local_feature_sets, None)
        ordered = [indices[index] for index in local_order]
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            batches.append((int(template_id), [items[index] for index in chunk]))
    return batches
