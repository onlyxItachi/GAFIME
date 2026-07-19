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
    _snapshot_max_abs_deltas,
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

    assert (
        forward_snapshot["candidate_identity_contract"]
        == "report-order-feature-tuples-families-v3"
    )
    assert (
        forward_snapshot["candidate_identity_sha256"]
        != reverse_snapshot["candidate_identity_sha256"]
    )


def test_snapshot_identity_includes_report_order_and_family():
    first = SimpleNamespace(
        interactions=[
            SimpleNamespace(
                combo=(0,), family="interaction", candidate_id="id-0", metrics={"pearson": 0.5}
            ),
            SimpleNamespace(
                combo=(1,), family="interaction", candidate_id="id-1", metrics={"pearson": 0.5}
            ),
        ]
    )
    reordered = SimpleNamespace(interactions=list(reversed(first.interactions)))
    changed_family = SimpleNamespace(
        interactions=[
            first.interactions[0],
            SimpleNamespace(
                combo=(1,),
                family="time_series",
                candidate_id="id-1",
                metrics={"pearson": 0.5},
            ),
        ]
    )

    reference = _snapshot(first, ("pearson",))
    assert reference["candidate_identity_sha256"] != _snapshot(
        reordered, ("pearson",)
    )["candidate_identity_sha256"]
    assert reference["candidate_identity_sha256"] != _snapshot(
        changed_family, ("pearson",)
    )["candidate_identity_sha256"]


def test_snapshot_contract_rejects_candidate_id_warning_and_decision_changes():
    def report(candidate_id, warning, signal):
        return SimpleNamespace(
            interactions=[
                SimpleNamespace(
                    combo=(0,),
                    family="interaction",
                    candidate_id=candidate_id,
                    metrics={"pearson": 0.5},
                )
            ],
            stability=[],
            permutations=[],
            warnings=[warning],
            decision=SimpleNamespace(signal_detected=signal, message="decision"),
        )

    reference = _snapshot(report("id-0", "warning", True), ("pearson",))
    changed_id = _snapshot(report("id-1", "warning", True), ("pearson",))
    changed_warning = _snapshot(report("id-0", "different", True), ("pearson",))
    changed_decision = _snapshot(report("id-0", "warning", False), ("pearson",))

    for changed in (changed_id, changed_warning, changed_decision):
        try:
            _snapshot_max_abs_deltas(reference, changed, ("pearson",))
        except AssertionError:
            pass
        else:
            raise AssertionError("public report contract mutation was not detected")


def test_cross_distribution_allows_legacy_empty_ids_but_checks_current_ids():
    legacy = SimpleNamespace(
        interactions=[
            SimpleNamespace(combo=(0,), family="interaction", candidate_id="", metrics={"pearson": 0.5})
        ],
        warnings=[],
        decision=SimpleNamespace(signal_detected=True, message="legacy"),
    )
    current = SimpleNamespace(
        interactions=[
            SimpleNamespace(
                combo=(0,),
                family="interaction",
                candidate_id="interaction:0",
                metrics={"pearson": 0.5},
            )
        ],
        warnings=[],
        decision=SimpleNamespace(signal_detected=True, message="current"),
    )

    assert _snapshot_max_abs_deltas(
        _snapshot(legacy, ("pearson",)),
        _snapshot(current, ("pearson",)),
        ("pearson",),
        cross_distribution=True,
    ) == {"pearson": 0.0}


def test_snapshot_contract_captures_stability_and_permutation_values():
    def report(p_value):
        identity = {
            "combo": (0,),
            "family": "interaction",
            "candidate_id": "interaction:0",
        }
        return SimpleNamespace(
            interactions=[
                SimpleNamespace(**identity, metrics={"pearson": 0.5})
            ],
            stability=[
                SimpleNamespace(
                    **identity,
                    metrics_mean={"pearson": 0.5},
                    metrics_std={"pearson": 0.1},
                )
            ],
            permutations=[
                SimpleNamespace(**identity, p_values={"pearson": p_value})
            ],
            warnings=[],
            decision=SimpleNamespace(signal_detected=True, message="decision"),
        )

    deltas = _snapshot_max_abs_deltas(
        _snapshot(report(0.25), ("pearson",)),
        _snapshot(report(0.50), ("pearson",)),
        ("pearson",),
    )

    assert deltas["permutations.p_values.pearson"] == 0.25
