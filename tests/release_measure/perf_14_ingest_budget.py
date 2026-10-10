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


def validate(report, budget, variant):
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
    require(budget["source_dtype"] in ("float32", "float64"), "invalid source dtype")
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
        memory_limit = number(
            baseline_peak
            + policy["fixed_format_allowance_bytes"]
            + policy["extra_bytes_per_cell"] * case["rows"] * case["cols"],
            "computed RSS limit",
        )
        load_peak = max(peak(r["memory"]["after_call_before_snapshot"]) for r in loaded)
        failures = [
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
                "rss_limit_bytes": memory_limit,
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
        "cells": comparisons,
        "scope": "Same-host coarse public-call cost gate; not a universal speedup, independent numerical oracle, exact allocation count, or frozen-source provenance proof.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("report", "budget", "variant", "output"):
        parser.add_argument("--" + field, required=True)
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
        verdict = validate(
            json.loads(inputs["report"]), json.loads(inputs["budget"]), args.variant
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
    return 0 if verdict["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
