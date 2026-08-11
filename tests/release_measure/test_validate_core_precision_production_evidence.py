"""Policy tests for Core production A/B evidence escalation.

These exercise the comparison policy only; they do not compile or run GAFIME.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SCRIPT = Path(__file__).with_name("validate_core_precision_production_evidence.py")
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("gafime_core_production_validator", _SCRIPT)
assert _SPEC and _SPEC.loader
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)


def _comparison(delta: float, interval: list[float]) -> dict[str, object]:
    return {
        "profile": "fp32",
        "metric": "pearson",
        "workload": "medium",
        "worker_mode": "default",
        "candidate_latency_delta_percent": delta,
        "bootstrap_candidate_latency_delta_95_ci_percent": interval,
    }


def test_confirmed_over_three_percent_regression_blocks_stable_mode() -> None:
    assessment = validator._comparison_escalation([_comparison(4.0, [3.1, 7.8])])

    assert assessment["stable_policy_ready"] is False
    assert len(assessment["hard_blockers_confirmed_over_three_percent"]) == 1
    assert assessment["investigate_confirmed_over_one_percent"] == []
    assert assessment["maintainer_regression_approval"] is None


def test_confirmed_over_one_percent_regression_remains_investigate_not_ready() -> None:
    assessment = validator._comparison_escalation([_comparison(2.0, [1.1, 3.8])])

    assert assessment["stable_policy_ready"] is False
    assert assessment["hard_blockers_confirmed_over_three_percent"] == []
    assert len(assessment["investigate_confirmed_over_one_percent"]) == 1
    assert assessment["maintainer_regression_approval"] is None


def test_inconclusive_and_invalid_cells_remain_visible_without_forging_a_speed_claim() -> None:
    inconclusive = validator._comparison_escalation([_comparison(4.0, [-1.0, 8.0])])
    invalid = validator._comparison_escalation([_comparison(4.0, [])])

    assert inconclusive["stable_policy_ready"] is False
    assert len(inconclusive["inconclusive_comparisons"]) == 1
    assert invalid["stable_policy_ready"] is False
    assert len(invalid["invalid_comparisons"]) == 1


def test_ci_crossing_zero_is_clean_when_upper_bound_is_within_margin() -> None:
    assessment = validator._comparison_escalation([_comparison(0.1, [-0.8, 0.9])])

    assert assessment["stable_policy_ready"] is True
    assert assessment["inconclusive_comparisons"] == []
    assert len(assessment["confirmed_non_regression_comparisons"]) == 1


def test_ci_threshold_boundaries_follow_strict_lower_and_inclusive_upper_policy() -> None:
    clean_at_one = validator._comparison_escalation(
        [_comparison(0.2, [-0.5, 1.0])]
    )
    inconclusive_at_lower_one = validator._comparison_escalation(
        [_comparison(2.0, [1.0, 2.5])]
    )
    investigate_at_lower_three = validator._comparison_escalation(
        [_comparison(4.0, [3.0, 5.0])]
    )

    assert clean_at_one["stable_policy_ready"] is True
    assert len(clean_at_one["confirmed_non_regression_comparisons"]) == 1
    assert inconclusive_at_lower_one["stable_policy_ready"] is False
    assert len(inconclusive_at_lower_one["inconclusive_comparisons"]) == 1
    assert investigate_at_lower_three["stable_policy_ready"] is False
    assert investigate_at_lower_three["hard_blockers_confirmed_over_three_percent"] == []
    assert len(investigate_at_lower_three["investigate_confirmed_over_one_percent"]) == 1


def test_scaling_inconclusive_is_diagnostic_not_primary_release_policy() -> None:
    comparison = _comparison(4.0, [-1.0, 8.0])
    comparison["worker_mode"] = "4"

    assessment = validator._comparison_escalation([comparison])

    assert assessment["stable_policy_ready"] is True
    assert assessment["inconclusive_comparisons"] == []
    assert len(assessment["thread_scaling_diagnostics"]) == 1


def test_informational_diagnostics_can_be_complete_without_publishing_claims() -> None:
    informational = validator._published_claim_readiness(
        mode="informational", diagnostic_ready=True, stable_ready=True
    )
    stable = validator._published_claim_readiness(
        mode="stable", diagnostic_ready=True, stable_ready=True
    )

    assert set(informational.values()) == {False}
    assert set(stable.values()) == {True}


def _snapshot(profile: str, *, candidate: int = 7, value: float = 0.5) -> dict[str, object]:
    if profile == "fp32":
        import struct

        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        dtype = "f32"
        text = f"{value:.9e}"
    else:
        import struct

        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        dtype = "f64"
        text = f"{value:.17e}"
    return {
        "result_dtype": dtype,
        "row_count": 1,
        "max_arity": 1,
        "metric_count": 1,
        "result_flags": 0,
        "metric_ids": [1],
        "combo_indices": [3],
        "ranks": [0],
        "families": [1],
        "candidate_ids": [candidate],
        "row_flags": [0],
        "metric_value_bits": [bits],
        "metric_value_text": [text],
        "metric_value_classes": ["finite"],
    }


def test_snapshot_contract_requires_exact_identity_and_fp32_bits() -> None:
    assert validator._snapshot_pair_failures(
        _snapshot("fp32"), _snapshot("fp32"), profile="fp32", metric="pearson"
    ) == []
    changed_id = _snapshot("fp32", candidate=8)
    changed_value = _snapshot("fp32", value=0.50000006)

    assert "snapshot_candidate_ids_mismatch" in validator._snapshot_pair_failures(
        _snapshot("fp32"), changed_id, profile="fp32", metric="pearson"
    )
    assert "snapshot_fp32_bits_mismatch" in validator._snapshot_pair_failures(
        _snapshot("fp32"), changed_value, profile="fp32", metric="pearson"
    )
    changed_metadata = _snapshot("fp32")
    changed_metadata["row_flags"] = [1]
    assert "snapshot_row_flags_mismatch" in validator._snapshot_pair_failures(
        _snapshot("fp32"), changed_metadata, profile="fp32", metric="pearson"
    )


def test_snapshot_contract_applies_existing_f64_tolerance_and_classification() -> None:
    assert validator._snapshot_pair_failures(
        _snapshot("mixed", value=1.0),
        _snapshot("mixed", value=1.0 + 5.0e-13),
        profile="mixed",
        metric="pearson",
    ) == []
    assert validator._snapshot_pair_failures(
        _snapshot("fp64", value=1.0),
        _snapshot("fp64", value=1.0 + 1.5e-12),
        profile="fp64",
        metric="spearman",
    ) == []
    assert any(
        reason.endswith("_mismatch")
        for reason in validator._snapshot_pair_failures(
            _snapshot("mixed", value=1.0),
            _snapshot("mixed", value=1.0 + 1.5e-12),
            profile="mixed",
            metric="r2",
        )
    )
    assert any(
        reason.endswith("_mismatch")
        for reason in validator._snapshot_pair_failures(
            _snapshot("fp64", value=1.0),
            _snapshot("fp64", value=1.0 + 2.5e-12),
            profile="fp64",
            metric="pearson",
        )
    )
    classified = _snapshot("mixed")
    classified["metric_value_classes"] = ["nan"]
    assert "snapshot_value_classification_mismatch" in validator._snapshot_pair_failures(
        _snapshot("mixed"), classified, profile="mixed", metric="spearman"
    )


def test_snapshot_contract_uses_absolute_only_tolerance_and_bit_exact_mi() -> None:
    assert any(
        reason.endswith("_mismatch")
        for reason in validator._snapshot_pair_failures(
            _snapshot("fp64", value=1.0e12),
            _snapshot("fp64", value=1.0e12 + 1.0),
            profile="fp64",
            metric="r2",
        )
    )
    assert "snapshot_mutual_info_bits_mismatch" in validator._snapshot_pair_failures(
        _snapshot("mixed", value=1.0),
        _snapshot("mixed", value=1.0 + 5.0e-13),
        profile="mixed",
        metric="mutual_info",
    )
    assert validator._snapshot_pair_failures(
        _snapshot("fp64", value=1.0),
        _snapshot("fp64", value=1.0),
        profile="fp64",
        metric="mutual_info",
    ) == []
