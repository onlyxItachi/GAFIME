from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


_RELEASE_MEASURE = Path(__file__).resolve().parents[1] / "release_measure"
if str(_RELEASE_MEASURE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_MEASURE))

from perf_08_v047_distribution_ab import (  # noqa: E402
    _candidate_count,
    _normalized_work,
    _snapshot,
)


def test_legacy_candidate_count_models_screening_and_per_arity_caps():
    assert _candidate_count(21, 3, 100_000, 12) == 21 + 66 + 220
    assert _candidate_count(6, 3, 3, 3) == 3 + 3 + 1
    assert _candidate_count(6, 3, 3, 0) == 3


def test_old_full_universe_work_normalizes_for_new_harness_field():
    old_result = {
        "dataset": {"features": 21},
        "work": {
            "max_arity": 3,
            "max_combinations_per_arity": 100_000,
        },
    }

    assert _normalized_work(old_result)["top_features_for_higher_k"] == 21


def test_candidate_identity_preserves_feature_tuple_order():
    forward = SimpleNamespace(
        interactions=[SimpleNamespace(combo=(0, 1), metrics={"pearson": 0.5})]
    )
    reverse = SimpleNamespace(
        interactions=[SimpleNamespace(combo=(1, 0), metrics={"pearson": 0.5})]
    )

    forward_snapshot = _snapshot(forward, ("pearson",))
    reverse_snapshot = _snapshot(reverse, ("pearson",))

    assert forward_snapshot["candidate_identity_contract"] == "ordered-feature-tuples-v2"
    assert (
        forward_snapshot["candidate_identity_sha256"]
        != reverse_snapshot["candidate_identity_sha256"]
    )
