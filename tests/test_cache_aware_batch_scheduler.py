import pytest

from gafime.discrete import (
    DiscreteFunctionCandidate,
    order_discrete_candidates_cache_aware,
)
from gafime.utils.cache_ordering import order_indices_cache_aware


def test_cache_aware_python_adapter_groups_hot_feature():
    feature_sets = [
        (8, 9),
        (1, 2),
        (5, 6),
        (1, 3),
        (1, 4),
    ]

    order = order_indices_cache_aware(feature_sets)

    assert sorted(order) == list(range(len(feature_sets)))
    assert all(1 in feature_sets[index] for index in order[:3])


def test_rust_batch_scheduler_exposes_equation_order():
    try:
        from gafime import subfunctions
    except Exception as exc:
        pytest.skip(f"Rust helper unavailable: {exc}")

    scheduler = subfunctions.BatchScheduler(max_blocks=1024)
    if not hasattr(scheduler, "order_equations"):
        pytest.skip("BatchScheduler.order_equations is not built.")

    feature_sets = [
        [8, 9],
        [1, 2],
        [5, 6],
        [1, 3],
        [1, 4],
    ]
    order = scheduler.order_equations(feature_sets, [0] * len(feature_sets))

    assert sorted(order) == list(range(len(feature_sets)))
    assert all(1 in feature_sets[index] for index in order[:3])


def test_rust_create_batches_reorders_pairs_for_cache_locality():
    try:
        from gafime import subfunctions
    except Exception as exc:
        pytest.skip(f"Rust helper unavailable: {exc}")

    scheduler = subfunctions.BatchScheduler(max_blocks=1024)
    if not hasattr(scheduler, "order_equations"):
        pytest.skip("BatchScheduler.order_equations is not built.")

    feature_pairs = [(8, 9), (1, 2), (5, 6), (1, 3), (1, 4)]
    batches = scheduler.create_batches(
        feature_pairs,
        [(0, 0)] * len(feature_pairs),
        [0] * len(feature_pairs),
    )
    first_batch_indices = batches[0][0]
    first_three_pairs = [
        tuple(first_batch_indices[index:index + 2])
        for index in range(0, 6, 2)
    ]

    assert all(1 in pair for pair in first_three_pairs)


def test_discrete_candidates_use_cache_aware_ordering():
    candidates = [
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_rectangle",
            feature_indices=(8, 9),
            intervals=((0.0, 1.0), (0.0, 1.0)),
            candidate_id="cold-rect",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(1,),
            thresholds=(0.0,),
            candidate_id="hot-threshold",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_rectangle",
            feature_indices=(5, 6),
            intervals=((0.0, 1.0), (0.0, 1.0)),
            candidate_id="cold-rect-2",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_value_gated_threshold",
            feature_indices=(3,),
            thresholds=(0.0,),
            value_feature=1,
            candidate_id="hot-value-threshold",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_rectangle",
            feature_indices=(1, 4),
            intervals=((0.0, 1.0), (0.0, 1.0)),
            candidate_id="hot-rect",
        ),
    ]

    ordered = order_discrete_candidates_cache_aware(candidates)

    assert sorted(candidate.candidate_id for candidate in ordered) == sorted(
        candidate.candidate_id for candidate in candidates
    )
    assert all(1 in candidate.combo for candidate in ordered[:3])
