#!/usr/bin/env python3
"""Evaluate explicit reviewed cost budgets for authenticated perf14 evidence.

This is a coarse same-host regression gate, not a speedup or release-readiness
claim. Budgets have no numeric defaults. Raw collector files remain authoritative;
source SHA fields are attestations, not a substitute for frozen provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

REPORT_SCHEMA = "gafime.public-ingestion-performance.v1"
BUDGET_SCHEMA = "gafime.public-ingestion-budget.v1"
PARSER_SCHEMA = "gafime.public-ingestion-parser.v1"
PARSER_CASE_KEYS = {"rows", "cols", "format"}
CONTEXT_KEYS = {
    "host",
    "python_version",
    "platform",
    "libc",
    "affinity",
    "cpu_count",
    "initial_allocator_policy",
    "thp",
}
ALLOCATOR_POLICY_KEYS = {
    "_RJEM_MALLOC_CONF",
    "POLARS_THP",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "MALLOC_CONF",
    "MALLOC_ARENA_MAX",
    "LD_PRELOAD",
}
CASE_KEYS = {
    "rows",
    "cols",
    "format",
    "precision",
    "workflow",
    "cache_state",
    "backend",
}
BOUND_KEYS = (
    "fixed_startup_ns",
    "direct_multiplier",
    "fixed_format_allowance_bytes",
    "extra_bytes_per_cell",
)
SNAPSHOT_KEYS = {
    "feature_names",
    "interactions",
    "stability",
    "permutations",
    "metric_rankings",
    "signal_detected",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, name: str, *, positive=False, integer=False):
    require(type(value) in (int, float), f"invalid numeric {name}")
    require(math.isfinite(value) and value >= 0, f"invalid numeric {name}")
    require(not positive or value > 0, f"non-positive {name}")
    require(not integer or type(value) is int, f"non-integer {name}")
    return value


def digest(value, width=64):
    require(
        isinstance(value, str)
        and len(value) == width
        and all(char in "0123456789abcdef" for char in value),
        "invalid identity digest",
    )
    return value


def case_key(case):
    require(set(case) == CASE_KEYS, "case must declare all exact collector fields")
    for field in ("rows", "cols"):
        number(case[field], field, positive=True, integer=True)
    require(
        all(isinstance(case[field], str) for field in CASE_KEYS - {"rows", "cols"}),
        "invalid case fields",
    )
    return json.dumps(case, sort_keys=True)


def peak(memory):
    require(memory["source"] == "linux_proc_self_status", "fresh Linux RSS unavailable")
    rss = number(memory["VmRSS_kib"], "VmRSS", integer=True)
    high = number(memory["VmHWM_kib"], "VmHWM", integer=True)
    require(high >= rss, "contradictory RSS high-water mark")
    return high * 1024


def producer_key(case):
    require(set(case) == PARSER_CASE_KEYS, "invalid parser case fields")
    for field in ("rows", "cols"):
        number(case[field], field, positive=True, integer=True)
    require(case["format"] in ("csv", "parquet", "ipc"), "invalid parser format")
    return json.dumps(case, sort_keys=True)


def measurement_context(value):
    """Require stable matching facts, not changing load or free-memory samples."""
    require(
        isinstance(value, dict) and set(value) == CONTEXT_KEYS,
        "invalid measurement context",
    )
    host = value["host"]
    require(
        isinstance(host, dict)
        and set(host) == {"hostname", "boot_id", "github_run_id", "github_run_attempt"},
        "invalid host context",
    )
    for field in ("hostname", "boot_id"):
        require(
            isinstance(host[field], str) and bool(host[field]), "missing host identity"
        )
    for field in ("github_run_id", "github_run_attempt"):
        require(
            host[field] is None or (isinstance(host[field], str) and bool(host[field])),
            "invalid Actions identity",
        )
    require(
        (host["github_run_id"] is None) == (host["github_run_attempt"] is None),
        "incomplete Actions identity",
    )
    require(
        isinstance(value["python_version"], str)
        and bool(value["python_version"])
        and isinstance(value["platform"], str)
        and value["platform"].startswith("Linux"),
        "invalid runtime context",
    )
    require(
        isinstance(value["libc"], list)
        and len(value["libc"]) == 2
        and all(isinstance(part, str) and bool(part) for part in value["libc"]),
        "invalid libc context",
    )
    number(value["cpu_count"], "cpu_count", positive=True, integer=True)
    affinity = value["affinity"]
    require(
        isinstance(affinity, list)
        and bool(affinity)
        and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
        and affinity == sorted(set(affinity)),
        "invalid CPU affinity",
    )
    policy = value["initial_allocator_policy"]
    require(
        isinstance(policy, dict)
        and set(policy) == ALLOCATOR_POLICY_KEYS
        and all(item is None or isinstance(item, str) for item in policy.values()),
        "invalid initial allocator policy",
    )
    thp = value["thp"]
    require(
        isinstance(thp, dict)
        and set(thp) == {"enabled", "defrag", "hpage_pmd_size"}
        and all(isinstance(item, str) and bool(item) for item in thp.values()),
        "missing THP context",
    )
    require(
        thp["hpage_pmd_size"].isdigit() and int(thp["hpage_pmd_size"]) > 0,
        "invalid THP page size",
    )
    return value


def parser_baselines(parser_report, selected, public_harness, cells, minimum):
    """Validate independent producer evidence before permitting any RSS baseline."""
    require(parser_report is not None, "independent parser report required")
    require(
        parser_report["schema"] == PARSER_SCHEMA, "unsupported parser report schema"
    )
    require(
        parser_report["status"] == "collected"
        and parser_report["collector_status"] == "passed",
        "incomplete parser collection",
    )
    records = parser_report["records"]
    number(
        parser_report["requested_jobs"],
        "parser requested_jobs",
        positive=True,
        integer=True,
    )
    require(
        len(records) == parser_report["requested_jobs"]
        and all(record["status"] == "passed" for record in records),
        "missing or failed parser samples",
    )
    variants = [v for v in parser_report["variants"] if v["label"] == selected["label"]]
    require(variants == [selected], "unmatched parser variant/source")
    require(
        digest(parser_report["public_harness"]["sha256"]) == public_harness,
        "unmatched parser public harness",
    )
    parser_harness = digest(parser_report["harness"]["sha256"])
    require(parser_harness != public_harness, "parser must use its independent worker")
    # Precision changes the native storage, not this Float64 file producer.
    # Sharing a producer between profile cells is allowed only when every
    # artifact, input, runtime and initial allocator policy agrees exactly.
    anchors = {}
    public_context = None
    for repeats in cells.values():
        for pair in repeats.values():
            for result in pair.values():
                key = producer_key(
                    {field: result["case"][field] for field in PARSER_CASE_KEYS}
                )
                context = measurement_context(result["measurement_context"])
                require(
                    public_context is None or public_context == context,
                    "mixed public measurement environment",
                )
                public_context = context
                require(
                    result["python"]["version"] == context["python_version"],
                    "contradictory Python context",
                )
                anchor = {
                    field: result[field]
                    for field in (
                        "identity",
                        "native",
                        "python",
                        "dependencies",
                        "fixture",
                        "measurement_context",
                    )
                }
                require(
                    key not in anchors or anchors[key] == anchor,
                    "mixed public producer identity",
                )
                anchors[key] = anchor
    producer_records = {}
    for record in records:
        if record["variant"] != selected["label"]:
            continue
        result = record["result"]
        require(
            result["schema"] == PARSER_SCHEMA + ".worker",
            "unsupported parser worker schema",
        )
        require(
            result["variant"] == selected and result["case"] == record["case"],
            "parser sample identity disagreement",
        )
        key = producer_key(record["case"])
        require(key in anchors, "unbudgeted parser cell")
        require(
            result["instrumented"] is False
            and result["treatment"] == "normal"
            and result["numerical_acquisition"] is False
            and result["frames_retained_until_snapshot"] is True,
            "invalid independent parser measurement scope",
        )
        require(
            digest(result["harness"]["sha256"]) == parser_harness
            and digest(result["public_harness"]["sha256"]) == public_harness,
            "mixed parser harness identity",
        )
        anchor = anchors[key]
        for field, value in anchor.items():
            require(result[field] == value, f"unmatched parser {field}")
        measurement_context(result["measurement_context"])
        fixture = result["fixture"]
        require(
            result["reader"]
            == {
                "name": "read_" + result["case"]["format"],
                "kwargs": {},
                "features": fixture["feature_names"],
                "target": "target",
            },
            "unmatched parser reader/projection",
        )
        number(
            result["timing"]["wall_nanoseconds"],
            "parser wall time",
            positive=True,
            integer=True,
        )
        for field in ("after_import", "after_call_before_snapshot"):
            peak(result["memory"][field])
        require(
            peak(result["memory"]["after_call_before_snapshot"])
            >= peak(result["memory"]["after_import"])
            and result["memory"]["inherited_resource_ru_maxrss_used"] is False,
            "invalid parser RSS measurement",
        )
        repeat = number(record["repeat"], "parser repeat", integer=True)
        repeats = producer_records.setdefault(key, {})
        require(repeat not in repeats, "duplicate parser sample")
        repeats[repeat] = result
    require(
        set(producer_records) == set(anchors), "missing declared parser cell coverage"
    )
    for repeats in producer_records.values():
        require(
            len(repeats) >= minimum and set(repeats) == set(range(len(repeats))),
            "missing parser repeat coverage",
        )
    return producer_records, parser_harness


def validate(report, budget, variant, parser_report=None):
    require(report["schema"] == REPORT_SCHEMA, "unsupported report schema")
    require(
        report["status"] == "collected" and report["collector_status"] == "passed",
        "incomplete collection",
    )
    require(
        report["public_numeric_parity"]["status"] == "passed", "public parity failed"
    )
    records = report["records"]
    number(report["requested_jobs"], "requested_jobs", positive=True, integer=True)
    require(
        len(records) == report["requested_jobs"]
        and all(r["status"] == "passed" for r in records),
        "missing or failed samples",
    )
    require(budget["schema"] == BUDGET_SCHEMA, "unsupported budget schema")
    model = budget.get("memory_baseline_model", "direct-v1")
    require(
        model in ("direct-v1", "independent-parser-v1"),
        "unsupported memory baseline model",
    )
    require(budget["source_dtype"] in ("float32", "float64"), "invalid source dtype")
    if model == "independent-parser-v1":
        require(
            budget["source_dtype"] == "float64",
            "parser model requires reviewed Float64 source",
        )
    minimum = number(budget["min_samples"], "min_samples", positive=True, integer=True)
    policies = {}
    for policy in budget["cells"]:
        key = case_key(policy["case"])
        require(key not in policies, "duplicate budget cell")
        for field in BOUND_KEYS:
            number(policy[field], field)
        policies[key] = policy
    require(policies, "empty budget coverage")
    selected = [v for v in report["variants"] if v["label"] == variant]
    require(len(selected) == 1, "missing or duplicate selected variant")
    selected = selected[0]
    require(
        type(budget["require_source_sha"]) is bool,
        "explicit source-binding policy required",
    )
    if selected["source_sha"] is not None or budget["require_source_sha"]:
        digest(selected["source_sha"], 40)
    harness = digest(report["harness"]["sha256"])
    identities, cells = set(), {}
    for record in records:
        if record["variant"] != variant:
            continue
        result = record["result"]
        require(type(result["instrumented"]) is bool, "invalid instrument flag")
        if result["instrumented"]:
            continue
        require(
            result["schema"] == REPORT_SCHEMA + ".worker", "unsupported worker schema"
        )
        require(
            result["case"] == record["case"] and result["variant"] == selected,
            "sample identity disagrees with collection",
        )
        identity = result["identity"]
        require(
            identity["byte_identical_installed_package"] is True
            and identity["expected_version"] == selected["expected_version"],
            "unauthenticated installed wheel",
        )
        require(
            identity["wheel"]["path"] == selected["wheel"], "wheel path disagreement"
        )
        native = result["native"]
        require(
            any(
                m["path"] == native["path"] and m["sha256"] == native["sha256"]
                for m in identity["verified_members"]
            ),
            "native is not authenticated wheel member",
        )
        require(
            digest(result["harness"]["sha256"]) == harness, "mixed harness identity"
        )
        require(
            result["fixture"]["source_dtype"] == budget["source_dtype"],
            "source dtype differs from reviewed budget",
        )
        identities.add(
            (
                digest(identity["wheel"]["sha256"]),
                digest(native["sha256"]),
                json.dumps(result["dependencies"], sort_keys=True),
                result["fixture"]["source_dtype"],
            )
        )
        case = dict(record["case"])
        route = case.pop("route")
        require(route in ("direct", "dataload"), "unknown public route")
        key = case_key(case)
        require(key in policies, "unbudgeted sample cell")
        repeat = number(record["repeat"], "repeat", integer=True)
        group = cells.setdefault(key, {}).setdefault(repeat, {})
        require(route not in group, "duplicate public-route sample")
        number(
            result["timing"]["wall_nanoseconds"],
            "wall time",
            positive=True,
            integer=True,
        )
        for memory_key in ("after_import", "after_call_before_snapshot"):
            peak(result["memory"][memory_key])
        require(
            peak(result["memory"]["after_call_before_snapshot"])
            >= peak(result["memory"]["after_import"]),
            "process RSS high-water mark decreased",
        )
        require(
            result["snapshot_generated_after_measurement"] is True
            and result["memory"]["inherited_resource_ru_maxrss_used"] is False,
            "invalid measurement ordering or RSS source",
        )
        numeric = result["snapshot"]["numeric"]
        require(SNAPSHOT_KEYS <= set(numeric), "incomplete numeric snapshot")
        encoded = json.dumps(
            numeric, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        require(
            hashlib.sha256(encoded).hexdigest()
            == digest(result["snapshot"]["numeric_sha256"]),
            "invalid numeric snapshot digest",
        )
        group[route] = result
    require(len(identities) == 1, "mixed or missing installed/source identities")
    require(set(cells) == set(policies), "missing declared cell coverage")
    producer_records, parser_harness = ({}, None)
    if model == "independent-parser-v1":
        producer_records, parser_harness = parser_baselines(
            parser_report, selected, harness, cells, minimum
        )
    comparisons = []
    for key, policy in policies.items():
        repeats = cells[key]
        require(
            len(repeats) >= minimum and set(repeats) == set(range(len(repeats))),
            "missing repeat coverage",
        )
        for pair in repeats.values():
            require(set(pair) == {"direct", "dataload"}, "missing matched public route")
            direct, loaded = pair["direct"], pair["dataload"]
            require(
                direct["snapshot"]["numeric"] == loaded["snapshot"]["numeric"],
                "numeric parity mismatch",
            )
            for field in ("fixture", "changed_fixture", "config"):
                require(
                    direct[field] == loaded[field], "unmatched fixture/configuration"
                )
        direct = [p["direct"] for p in repeats.values()]
        loaded = [p["dataload"] for p in repeats.values()]
        direct_time = statistics.median(r["timing"]["wall_nanoseconds"] for r in direct)
        load_time = statistics.median(r["timing"]["wall_nanoseconds"] for r in loaded)
        wall_limit = number(
            policy["fixed_startup_ns"] + policy["direct_multiplier"] * direct_time,
            "computed wall limit",
        )
        baseline_peak = max(peak(r["memory"]["after_import"]) for r in loaded)
        baseline_peak = max(
            baseline_peak,
            max(peak(r["memory"]["after_call_before_snapshot"]) for r in direct),
        )
        case = policy["case"]
        allowance = number(
            policy["fixed_format_allowance_bytes"]
            + policy["extra_bytes_per_cell"] * case["rows"] * case["cols"],
            "computed RSS allowance",
        )
        old_memory_limit = number(
            baseline_peak + allowance,
            "computed RSS limit",
        )
        memory_limit = old_memory_limit
        producer_costs = {}
        producer_failures = []
        if model == "independent-parser-v1":
            producer = producer_records[
                producer_key({field: case[field] for field in PARSER_CASE_KEYS})
            ].values()
            parser_peak = max(
                peak(r["memory"]["after_call_before_snapshot"]) for r in producer
            )
            parser_time = statistics.median(
                r["timing"]["wall_nanoseconds"] for r in producer
            )
            producer_failures = [
                name
                for name, actual, limit in (
                    ("parser_memory", parser_peak, old_memory_limit),
                    ("parser_latency", parser_time, wall_limit),
                )
                if actual > limit
            ]
            # A producer cannot normalize away its own cost. A failed control
            # never enlarges the public limit; both failures remain visible.
            if not producer_failures:
                memory_limit = number(
                    max(baseline_peak, parser_peak) + allowance,
                    "computed parser-aware RSS limit",
                )
            producer_costs = {
                "parser_samples": len(producer),
                "parser_peak_bytes": parser_peak,
                "parser_rss_limit_bytes": old_memory_limit,
                "median_parser_ns": parser_time,
                "parser_wall_limit_ns": wall_limit,
                "parser_baseline_applied": not producer_failures,
            }
        load_peak = max(peak(r["memory"]["after_call_before_snapshot"]) for r in loaded)
        failures = producer_failures + [
            name
            for name, actual, limit in (
                ("latency", load_time, wall_limit),
                ("memory", load_peak, memory_limit),
            )
            if actual > limit
        ]
        comparisons.append(
            {
                "case": case,
                "samples": len(repeats),
                "median_direct_ns": direct_time,
                "median_dataload_ns": load_time,
                "wall_limit_ns": wall_limit,
                "max_dataload_peak_bytes": load_peak,
                "old_rss_limit_bytes": old_memory_limit,
                "rss_limit_bytes": memory_limit,
                **producer_costs,
                "failures": failures,
            }
        )
    return {
        "schema": BUDGET_SCHEMA + ".verdict",
        "status": "failed" if any(c["failures"] for c in comparisons) else "passed",
        "release_ready": False,
        "variant": selected,
        "source_binding": "attested_commit"
        if selected["source_sha"]
        else "uncommitted_development_only",
        "authenticated_identity": list(next(iter(identities))),
        "harness_sha256": harness,
        "parser_harness_sha256": parser_harness,
        "memory_baseline_model": model,
        "cells": comparisons,
        "scope": "Same-host coarse public-call cost gate; not a universal speedup, independent numerical oracle, exact allocation count, or frozen-source provenance proof.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("report", "budget", "variant", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--parser-report")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; prior verdict will not be overwritten")
    verdict = {
        "schema": BUDGET_SCHEMA + ".verdict",
        "status": "failed",
        "release_ready": False,
    }
    inputs = {}
    try:
        inputs = {
            name: Path(getattr(args, name)).read_bytes()
            for name in ("report", "budget")
        }
        if args.parser_report:
            inputs["parser_report"] = Path(args.parser_report).read_bytes()
        verdict = validate(
            json.loads(inputs["report"]),
            json.loads(inputs["budget"]),
            args.variant,
            json.loads(inputs["parser_report"]) if "parser_report" in inputs else None,
        )
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        verdict["reason"] = str(error)
    verdict["inputs"] = {
        name: {
            "path": str(Path(getattr(args, name)).resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in inputs.items()
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(verdict, stream, indent=2, allow_nan=False)
    # The retained JSON remains the evidence record; expose the assessment in
    # CI too, so a failed budget is not just an unexplained exit after workers.
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "reason": verdict.get("reason"),
                "checked_cells": len(verdict.get("cells", [])),
                "failed_cells": [
                    cell for cell in verdict.get("cells", []) if cell["failures"]
                ],
                "release_ready": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if verdict["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
