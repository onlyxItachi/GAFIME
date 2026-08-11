#!/usr/bin/env python3
"""Validate and summarize tracked Core production-executor A/B evidence.

This intentionally consumes raw artifacts produced by
``run_core_precision_production_benchmark.py``.  It does not run GAFIME or
manufacture measurements. Informational mode may validate exactly its declared
reduced matrix as auditable diagnostic evidence but never publishes a release
performance claim; stable mode requires the complete release matrix and both
A/B and B/A blocks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from perf_13_precision_profiles import (
    CORE_SNAPSHOT_PARITY_POLICY,
    Variant,
    _comparison_classification,
    _load_native_evidence_specs,
    _native_ab_comparisons,
    _native_ab_schedule_readiness,
)


def _full_sha(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} must be a full lowercase 40-character SHA")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha", required=True)
    parser.add_argument("--expected-candidate-sha", required=True)
    parser.add_argument("--mode", choices=("informational", "stable"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _artifact_summary(evidence: dict[str, object]) -> list[dict[str, object]]:
    summaries = []
    for artifact in evidence.get("artifacts", ()):  # type: ignore[union-attr]
        if not isinstance(artifact, dict):
            continue
        validation = artifact.get("validation")
        summaries.append(
            {
                "variant": artifact.get("variant"),
                "backend": artifact.get("backend"),
                "kind": artifact.get("kind"),
                "path": artifact.get("path"),
                "sha256": artifact.get("sha256"),
                "complete": validation.get("complete")
                if isinstance(validation, dict)
                else False,
                "performance_claim_ready": validation.get("performance_claim_ready")
                if isinstance(validation, dict)
                else False,
                "failures": validation.get("failures", [])
                if isinstance(validation, dict)
                else ["artifact_validation_missing"],
                "claim_failures": validation.get("claim_failures", [])
                if isinstance(validation, dict)
                else [],
                "raw_measurement_claim_ready": validation.get(
                    "raw_measurement_claim_ready"
                )
                if isinstance(validation, dict)
                else False,
                "thread_scaling_tables": validation.get("thread_scaling_tables", [])
                if isinstance(validation, dict)
                else [],
                "measurement_mode": validation.get("measurement_mode")
                if isinstance(validation, dict)
                else None,
            }
        )
    return summaries


def _comparison_escalation(
    comparisons: Sequence[object],
) -> dict[str, object]:
    """Classify independently derived comparisons for stable-mode policy.

    Raw integrity and an A/B+B/A schedule make a comparison *auditable*; they
    do not authorize a stable claim in the face of a confirmed regression.
    This task intentionally has no maintainer-regression override input.
    """

    blockers: list[dict[str, object]] = []
    investigations: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    inconclusive: list[dict[str, object]] = []
    confirmed_non_regression: list[dict[str, object]] = []
    scaling_diagnostics: list[dict[str, object]] = []
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            invalid.append({"index": index, "reason": "comparison_not_mapping"})
            continue
        delta = comparison.get("candidate_latency_delta_percent")
        interval = comparison.get("bootstrap_candidate_latency_delta_95_ci_percent")
        if not isinstance(delta, (int, float)):
            invalid.append(
                {
                    "index": index,
                    "reason": "candidate_latency_delta_missing_or_invalid",
                }
            )
            continue
        derived = _comparison_classification(float(delta), interval)
        status = str(derived["review_status"])
        summary = {
            "index": index,
            "profile": comparison.get("profile"),
            "metric": comparison.get("metric"),
            "workload": comparison.get("workload"),
            "worker_mode": comparison.get("worker_mode"),
            "candidate_latency_delta_percent": float(delta),
            "bootstrap_candidate_latency_delta_95_ci_percent": interval,
            "review_status": status,
            "escalation": derived["escalation"],
        }
        if comparison.get("worker_mode") != "default":
            scaling_diagnostics.append(summary)
            continue
        if status == "confirmed_regression_above_three_percent":
            blockers.append(summary)
        elif status == "confirmed_regression_above_one_percent":
            investigations.append(summary)
        elif status in {"bootstrap_interval_missing", "bootstrap_interval_invalid"}:
            invalid.append(summary)
        elif status == "inconclusive_regression_margin_above_one_percent":
            inconclusive.append(summary)
        else:
            confirmed_non_regression.append(summary)

    # Primary default-worker cells are judged against the allowed regression
    # margin, not against zero.  A CI that crosses zero is clean when its upper
    # bound remains at or below one percent. Scaling cells remain separately
    # visible diagnostics.
    stable_policy_ready = (
        not blockers and not investigations and not invalid and not inconclusive
    )
    return {
        "maintainer_regression_approval": None,
        "stable_policy_ready": stable_policy_ready,
        "hard_blockers_confirmed_over_three_percent": blockers,
        "investigate_confirmed_over_one_percent": investigations,
        "invalid_comparisons": invalid,
        "inconclusive_comparisons": inconclusive,
        "confirmed_non_regression_comparisons": confirmed_non_regression,
        "thread_scaling_diagnostics": scaling_diagnostics,
        "interpretation": (
            "a lower 95 percent CI bound above three percent is a hard blocker; "
            "a lower bound above one percent remains investigate/not-ready; an "
            "upper bound at or below one percent is clean even when the interval "
            "crosses zero; intervals overlapping the one-percent margin are "
            "inconclusive/not-ready; scaling cells are diagnostic only"
        ),
    }


def _snapshot_pair_failures(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    profile: str,
    metric: str,
) -> list[str]:
    failures: list[str] = []
    for field in (
        "result_dtype",
        "row_count",
        "max_arity",
        "metric_count",
        "result_flags",
        "metric_ids",
        "combo_indices",
        "ranks",
        "families",
        "candidate_ids",
        "row_flags",
    ):
        if left.get(field) != right.get(field):
            failures.append(f"snapshot_{field}_mismatch")
    left_classes = left.get("metric_value_classes")
    right_classes = right.get("metric_value_classes")
    if left_classes != right_classes:
        failures.append("snapshot_value_classification_mismatch")
        return failures
    left_bits = left.get("metric_value_bits")
    right_bits = right.get("metric_value_bits")
    if profile == "fp32":
        if left_bits != right_bits:
            failures.append("snapshot_fp32_bits_mismatch")
        return failures
    if metric == "mutual_info":
        if left_bits != right_bits:
            failures.append("snapshot_mutual_info_bits_mismatch")
        return failures
    left_text = left.get("metric_value_text")
    right_text = right.get("metric_value_text")
    if (
        not isinstance(left_text, list)
        or not isinstance(right_text, list)
        or len(left_text) != len(right_text)
        or not isinstance(left_classes, list)
        or len(left_classes) != len(left_text)
    ):
        return failures + ["snapshot_f64_values_malformed"]
    for index, (left_value, right_value, value_class) in enumerate(
        zip(left_text, right_text, left_classes, strict=True)
    ):
        if value_class != "finite":
            continue
        try:
            left_float = float(left_value)
            right_float = float(right_value)
        except (TypeError, ValueError):
            failures.append(f"snapshot_f64_value_{index}_invalid")
            continue
        tolerance = 1.0e-12 if profile == "mixed" else 2.0e-12
        if not math.isclose(
            left_float, right_float, rel_tol=0.0, abs_tol=tolerance
        ):
            failures.append(f"snapshot_f64_value_{index}_mismatch")
    return failures


def _production_artifacts(
    evidence: Mapping[str, object],
) -> list[tuple[str, Mapping[str, object], Mapping[str, object]]]:
    observed = []
    for artifact in evidence.get("artifacts", ()):  # type: ignore[union-attr]
        if not isinstance(artifact, Mapping) or artifact.get("kind") != (
            "core_production_executor"
        ):
            continue
        validation = artifact.get("validation")
        if isinstance(validation, Mapping):
            observed.append((str(artifact.get("variant")), artifact, validation))
    return observed


def _result_snapshot_readiness(evidence: Mapping[str, object]) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    cells: dict[tuple[object, ...], Mapping[str, object]] = {}
    for variant, artifact, validation in _production_artifacts(evidence):
        records = validation.get("production_records")
        if not isinstance(records, list):
            failures.append({"reason": "production_records_missing", "variant": variant})
            continue
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            workload = record.get("workload")
            topology = record.get("execution_topology")
            result = record.get("result")
            snapshot = result.get("untimed_snapshot") if isinstance(result, Mapping) else None
            key = (
                record.get("ab_block"),
                record.get("input_policy"),
                workload.get("name") if isinstance(workload, Mapping) else None,
                record.get("metric"),
                record.get("profile"),
                topology.get("worker_mode") if isinstance(topology, Mapping) else None,
                variant,
            )
            if key in cells or not isinstance(snapshot, Mapping):
                failures.append(
                    {"reason": "snapshot_cell_duplicate_or_missing", "key": list(key)}
                )
            else:
                cells[key] = snapshot

    paired = 0
    semantic_keys = {key[:-1] for key in cells}
    for key in sorted(semantic_keys, key=repr):
        baseline = cells.get((*key, "baseline"))
        candidate = cells.get((*key, "candidate"))
        if baseline is None or candidate is None:
            failures.append({"reason": "snapshot_ab_pair_missing", "key": list(key)})
            continue
        paired += 1
        for reason in _snapshot_pair_failures(
            baseline,
            candidate,
            profile=str(key[4]),
            metric=str(key[3]),
        ):
            failures.append({"reason": reason, "key": list(key)})

    # Each variant must also be semantically stable when the A/B process order
    # reverses. This catches block-dependent contamination independently of the
    # baseline/candidate equality check.
    cross_block_groups: dict[tuple[object, ...], dict[object, Mapping[str, object]]] = {}
    for key, snapshot in cells.items():
        block, *rest = key
        cross_block_groups.setdefault(tuple(rest), {})[block] = snapshot
    for key, blocks in cross_block_groups.items():
        if set(blocks) != {0, 1}:
            failures.append({"reason": "snapshot_ab_ba_block_missing", "key": list(key)})
            continue
        for reason in _snapshot_pair_failures(
            blocks[0],
            blocks[1],
            profile=str(key[3]),
            metric=str(key[2]),
        ):
            failures.append({"reason": f"cross_block_{reason}", "key": list(key)})
    return {
        "complete": bool(cells) and not failures,
        "paired_cell_count": paired,
        "failures": failures,
        "declared_policy": CORE_SNAPSHOT_PARITY_POLICY,
        "policy": (
            "ordered combo indices, ranks, families, candidate IDs, and row flags "
            "are exact; fp32 and mutual-info visible values are bit-exact; "
            "mixed/fp64 classifications are exact and other finite values use "
            "absolute-only profile tolerances of 1e-12/2e-12"
        ),
    }


def _production_schedule_readiness(evidence: Mapping[str, object]) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    by_block: dict[object, dict[str, Mapping[str, object]]] = {}
    sequences: dict[object, tuple[str, ...]] = {}
    runner_pids: set[int] = set()
    for variant, artifact, validation in _production_artifacts(evidence):
        schedule = validation.get("cell_schedule")
        try:
            payload = json.loads(Path(str(artifact.get("path"))).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        block = payload.get("ab_block") if isinstance(payload, Mapping) else None
        sequence = payload.get("variant_sequence") if isinstance(payload, Mapping) else None
        runner_pid = payload.get("runner_pid") if isinstance(payload, Mapping) else None
        if (
            not isinstance(runner_pid, int)
            or isinstance(runner_pid, bool)
            or runner_pid < 1
            or runner_pid in runner_pids
        ):
            failures.append({"reason": "fresh_unique_runner_pid_required", "variant": variant, "block": block})
        else:
            runner_pids.add(runner_pid)
        if not isinstance(schedule, Mapping) or block not in (0, 1):
            failures.append({"reason": "schedule_or_block_missing", "variant": variant})
            continue
        if variant in by_block.setdefault(block, {}):
            failures.append({"reason": "duplicate_schedule_variant", "block": block})
        by_block[block][variant] = schedule
        observed_sequence = tuple(str(value) for value in sequence) if isinstance(sequence, list) else ()
        previous = sequences.setdefault(block, observed_sequence)
        if previous != observed_sequence:
            failures.append({"reason": "within_block_variant_sequence_mismatch", "block": block})
        counts = schedule.get("profile_order_counts")
        count_values = (
            list(counts.values()) if isinstance(counts, Mapping) else []
        )
        if (
            not isinstance(counts, Mapping)
            or not counts
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in count_values
            )
            or max(count_values, default=0) - min(count_values, default=0) > 1
        ):
            failures.append({"reason": "profile_order_balance_invalid", "block": block})
    if set(by_block) != {0, 1}:
        failures.append({"reason": "both_schedule_blocks_required"})
    for block, variants in by_block.items():
        if set(variants) != {"baseline", "candidate"}:
            failures.append({"reason": "both_schedule_variants_required", "block": block})
            continue
        baseline = variants["baseline"]
        candidate = variants["candidate"]
        for field in ("seed", "sha256", "entries", "profile_order_counts"):
            if baseline.get(field) != candidate.get(field):
                failures.append({"reason": "within_block_schedule_mismatch", "block": block, "field": field})
    if set(sequences.values()) != {
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    }:
        failures.append({"reason": "ab_ba_variant_reversal_required"})
    if set(by_block) == {0, 1}:
        for field in ("seed", "sha256", "entries"):
            left = next(iter(by_block[0].values())).get(field)
            right = next(iter(by_block[1].values())).get(field)
            if left == right:
                failures.append({"reason": "cross_block_schedule_variation_required", "field": field})
    return {
        "complete": not failures,
        "failures": failures,
        "blocks": sorted(by_block),
        "runner_pids": sorted(runner_pids),
    }


def _primary_comparison_coverage(
    evidence: Mapping[str, object], comparisons: Sequence[object]
) -> dict[str, object]:
    expected: set[tuple[object, ...]] = set()
    for variant, _artifact, validation in _production_artifacts(evidence):
        if variant != "candidate":
            continue
        records = validation.get("production_records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            topology = record.get("execution_topology")
            workload = record.get("workload")
            if not isinstance(topology, Mapping) or topology.get("worker_mode") != "default":
                continue
            expected.add(
                (
                    record.get("ab_block"),
                    record.get("input_policy"),
                    json.dumps(workload, sort_keys=True),
                    record.get("metric"),
                    record.get("profile"),
                )
            )
    observed: set[tuple[object, ...]] = set()
    for comparison in comparisons:
        if not isinstance(comparison, Mapping) or comparison.get("worker_mode") != "default":
            continue
        observed.add(
            (
                comparison.get("ab_block"),
                comparison.get("input_policy"),
                comparison.get("workload"),
                comparison.get("metric"),
                comparison.get("profile"),
            )
        )
    missing = sorted(expected - observed, key=repr)
    unexpected = sorted(observed - expected, key=repr)
    return {
        "complete": bool(expected) and not missing and not unexpected,
        "expected_primary_cell_count": len(expected),
        "observed_primary_cell_count": len(observed),
        "missing": [list(value) for value in missing],
        "unexpected": [list(value) for value in unexpected],
    }


def _published_claim_readiness(
    *, mode: str, diagnostic_ready: bool, stable_ready: bool
) -> dict[str, bool]:
    release_ready = bool(mode == "stable" and diagnostic_ready and stable_ready)
    return {
        "comparative_performance_claim_ready": release_ready,
        "performance_claim_ready": release_ready,
        "stable_release_ready": release_ready,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    expected_baseline = _full_sha(args.expected_baseline_sha, "--expected-baseline-sha")
    expected_candidate = _full_sha(args.expected_candidate_sha, "--expected-candidate-sha")
    baseline_manifest = args.baseline_manifest.resolve(strict=True)
    candidate_manifest = args.candidate_manifest.resolve(strict=True)
    evidence = _load_native_evidence_specs(
        (
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        )
    )
    expected_commits = {"baseline": expected_baseline, "candidate": expected_candidate}
    observed_commits = evidence.get("source_commits_by_variant")
    commit_binding_ok = observed_commits == expected_commits
    variants = (
        Variant("baseline", sys.executable, None, ()),
        Variant("candidate", sys.executable, None, ()),
    )
    schedule = _native_ab_schedule_readiness(evidence, variants, ("core",))
    comparisons = _native_ab_comparisons(
        evidence, variants, bootstrap_resamples=2_000, seed=0xC0DE_2026
    )
    result_snapshots = _result_snapshot_readiness(evidence)
    production_schedule = _production_schedule_readiness(evidence)
    primary_comparison_coverage = _primary_comparison_coverage(evidence, comparisons)
    raw_integrity_ready = bool(
        evidence.get("valid") is True
        and commit_binding_ok
        and all(
            item.get("complete") is True for item in _artifact_summary(evidence)
        )
    )
    diagnostic_comparative_ready = bool(
        raw_integrity_ready
        and schedule.get("complete") is True
        and production_schedule.get("complete") is True
        and result_snapshots.get("complete") is True
        and primary_comparison_coverage.get("complete") is True
        and comparisons
    )
    comparison_escalation = _comparison_escalation(comparisons)
    stable_ready = diagnostic_comparative_ready and all(
        item.get("complete") is True
        and item.get("performance_claim_ready") is True
        and item.get("raw_measurement_claim_ready") is True
        and item.get("measurement_mode") == "stable"
        for item in _artifact_summary(evidence)
    ) and comparison_escalation["stable_policy_ready"] is True
    published_claims = _published_claim_readiness(
        mode=args.mode,
        diagnostic_ready=diagnostic_comparative_ready,
        stable_ready=stable_ready,
    )
    payload = {
        "schema": "gafime.core-production-executor-comparison.v1",
        "mode": args.mode,
        "status": (
            "pass"
            if args.mode == "stable" and stable_ready
            else "informational_complete"
            if diagnostic_comparative_ready
            else "informational_incomplete_or_unvalidated"
        ),
        "exact_commit_binding": {
            "expected": expected_commits,
            "observed": observed_commits,
            "complete": commit_binding_ok,
        },
        "raw_artifact_integrity_ready": raw_integrity_ready,
        "diagnostic_comparative_readiness": diagnostic_comparative_ready,
        **published_claims,
        "native_evidence": {
            "valid": evidence.get("valid"),
            "arithmetic_claims_valid": evidence.get("arithmetic_claims_valid"),
            "failures": evidence.get("failures", []),
            "claim_failures": evidence.get("claim_failures", []),
            "artifacts": _artifact_summary(evidence),
        },
        "ab_ba_schedule": schedule,
        "production_cell_schedule": production_schedule,
        "result_snapshot_parity": result_snapshots,
        "primary_comparison_coverage": primary_comparison_coverage,
        "comparisons": comparisons,
        "comparison_escalation": comparison_escalation,
        "thread_scaling_claim_boundary": (
            "1/2/4/default speedup and efficiency tables are diagnostics; only "
            "default-worker cells are primary product-throughput results"
        ),
        "claim_boundary": {
            "informational": (
                "an informational workflow may publish raw integrity, schedule, and "
                "normalized comparison diagnostics but makes no release performance claim"
            ),
            "stable": (
                "stable mode requires complete exact-head/base production artifacts, "
                "paired A/B and B/A schedules, nonempty normalized comparisons, "
                "upper CI bounds at or below the one-percent regression margin, "
                "and no lower CI bound above the one/three-percent escalation margins"
            ),
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise ValueError("--output already exists; refuse to overwrite evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.mode == "stable" and not stable_ready:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Core production evidence validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
