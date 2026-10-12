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
    "JEMALLOC_CONF",
    "MALLOC_ARENA_MAX",
    "MALLOC_TRIM_THRESHOLD_",
    "MALLOC_MMAP_THRESHOLD_",
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


def file_descriptor(value, name):
    """Validate recorded identities without resolving retained runner paths."""
    require(
        isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"},
        f"incomplete {name} identity",
    )
    require(
        isinstance(value["path"], str) and bool(value["path"]), f"invalid {name} path"
    )
    digest(value["sha256"])
    number(value["size_bytes"], name + " size", positive=True, integer=True)


def dependency_provenance(value):
    require(
        isinstance(value, dict)
        and set(value)
        == {"numpy", "polars", "polars_native_members", "numpy_native_sha256"},
        "incomplete native dependency provenance",
    )
    for name in ("numpy", "polars"):
        require(
            isinstance(value[name], str) and bool(value[name]),
            "invalid dependency version",
        )
    digest(value["numpy_native_sha256"])
    members = value["polars_native_members"]
    require(
        isinstance(members, list) and bool(members), "missing Polars native members"
    )
    seen = set()
    for member in members:
        require(
            isinstance(member, dict)
            and set(member) == {"distribution", "member", "sha256", "size_bytes"},
            "incomplete Polars native member identity",
        )
        require(
            all(
                isinstance(member[name], str) and bool(member[name])
                for name in ("distribution", "member")
            ),
            "invalid Polars native member name",
        )
        digest(member["sha256"])
        number(
            member["size_bytes"],
            "Polars native member size",
            positive=True,
            integer=True,
        )
        key = (member["distribution"], member["member"])
        require(key not in seen, "duplicate Polars native member")
        seen.add(key)


def fixture_provenance(value, case):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "file",
            "features",
            "target",
            "feature_names",
            "parser_version",
            "source_dtype",
        },
        "incomplete parser fixture provenance",
    )
    for name in ("file", "features", "target"):
        file_descriptor(value[name], "fixture " + name)
    names = value["feature_names"]
    require(
        isinstance(names, list)
        and len(names) == case["cols"]
        and all(isinstance(name, str) and bool(name) for name in names)
        and len(names) == len(set(names))
        and "target" not in names,
        "invalid parser fixture feature names",
    )
    require(
        isinstance(value["parser_version"], str) and bool(value["parser_version"]),
        "missing fixture parser version",
    )
    require(
        value["source_dtype"] == "float64", "unreviewed parser fixture source dtype"
    )


def public_environment(value, context):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "platform",
            "affinity",
            "cpu_count",
            "RAYON_NUM_THREADS",
            "POLARS_MAX_THREADS",
            "auto_gpu_payloads_explicitly_disabled",
            "load_average",
        },
        "incomplete public environment provenance",
    )
    for name in ("platform", "affinity", "cpu_count"):
        require(value[name] == context[name], "contradictory public environment")
    require(
        type(value["cpu_count"]) is int
        and isinstance(value["affinity"], list)
        and all(type(cpu) is int for cpu in value["affinity"]),
        "invalid public CPU observation types",
    )
    for name in ("RAYON_NUM_THREADS", "POLARS_MAX_THREADS"):
        require(
            value[name] == context["initial_allocator_policy"][name],
            "contradictory public thread policy",
        )
    require(
        value["auto_gpu_payloads_explicitly_disabled"] is True,
        "undeclared CPU-only control",
    )
    load = value["load_average"]
    require(
        isinstance(load, list) and len(load) == 3, "missing public load observation"
    )
    for item in load:
        number(item, "public load average")


def host_observation(value, context):
    keys = {"cpu_info", "system_memory", "kernel", "scope"}
    require(
        isinstance(value, dict) and keys <= set(value) <= keys | {"process_status"},
        "incomplete host observation",
    )
    for name in ("cpu_info", "system_memory"):
        require(
            isinstance(value[name], str) and bool(value[name].strip()),
            "missing host " + name,
        )
    kernel = value["kernel"]
    require(
        isinstance(kernel, dict)
        and set(kernel)
        == {"system", "node", "release", "version", "machine", "processor"}
        and all(isinstance(part, str) for part in kernel.values())
        and all(
            bool(kernel[name])
            for name in ("system", "node", "release", "version", "machine")
        ),
        "incomplete kernel uname observation",
    )
    require(
        kernel["system"] == "Linux"
        and kernel["node"] == context["host"]["hostname"]
        and kernel["release"] in context["platform"]
        and kernel["machine"] in context["platform"],
        "contradictory host kernel observation",
    )
    require(
        isinstance(value["scope"], str)
        and value["scope"].startswith("post-measurement"),
        "host observation must follow measurement",
    )
    if "process_status" in value:
        require(
            isinstance(value["process_status"], (str, dict))
            and bool(value["process_status"]),
            "missing process status observation",
        )


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
        and set(thp) == {"enabled", "defrag", "shmem_enabled", "hpage_pmd_size"}
        and all(isinstance(item, str) and bool(item) for item in thp.values()),
        "missing THP context",
    )
    require(
        thp["hpage_pmd_size"].isdigit() and int(thp["hpage_pmd_size"]) > 0,
        "invalid THP page size",
    )
    return value


def runtime_observation(value, *, require_polars):
    """Check post-call facts without requiring an otherwise unused import."""
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "allocator_policy",
            "polars_loaded",
            "polars_thread_pool_size",
            "scope",
        },
        "invalid runtime observation",
    )
    policy = value["allocator_policy"]
    require(
        isinstance(policy, dict)
        and set(policy) == ALLOCATOR_POLICY_KEYS
        and all(item is None or isinstance(item, str) for item in policy.values()),
        "invalid effective allocator policy",
    )
    loaded = value["polars_loaded"]
    require(type(loaded) is bool, "invalid Polars loaded observation")
    require(not require_polars or loaded, "parser/dataload Polars observation required")
    if loaded:
        number(
            value["polars_thread_pool_size"],
            "Polars pool size",
            positive=True,
            integer=True,
        )
    else:
        require(
            value["polars_thread_pool_size"] is None,
            "unloaded Polars cannot report a pool",
        )
    require(
        isinstance(value["scope"], str)
        and value["scope"].startswith("post-measurement"),
        "runtime observation must follow measurement",
    )
    return {
        "allocator_policy": policy,
        "polars_thread_pool_size": value["polars_thread_pool_size"],
    }


def parser_baselines(parser_report, selected, public_harness, cells, minimum):
    """Validate independent producer evidence before permitting any RSS baseline."""
    require(parser_report is not None, "independent parser report required")
    # Structured validation can check this descriptor, not reproduce the
    # original JSON serialization. The CLI additionally checks the supplied
    # public report's actual bytes; retained runner paths stay portable.
    file_descriptor(parser_report["public_report"], "parser public report")
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
    global_runtime = None
    public_context = None
    for repeats in cells.values():
        for pair in repeats.values():
            for result in pair.values():
                key = producer_key(
                    {field: result["case"][field] for field in PARSER_CASE_KEYS}
                )
                context = measurement_context(result["measurement_context"])
                dependency_provenance(result["dependencies"])
                fixture_provenance(result["fixture"], result["case"])
                file_descriptor(result["identity"]["wheel"], "installed wheel")
                file_descriptor(result["native"], "native module")
                public_environment(result["environment"], context)
                host_observation(result["host_observation"], context)
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
                dataload = result["case"]["route"] == "dataload"
                runtime = runtime_observation(
                    result["runtime_observation"], require_polars=dataload
                )
                if result["runtime_observation"]["polars_loaded"]:
                    require(
                        global_runtime is None or global_runtime == runtime,
                        "mixed global effective allocator/pool observation",
                    )
                    global_runtime = runtime
                else:
                    require(
                        runtime["allocator_policy"]
                        == context["initial_allocator_policy"],
                        "unloaded Polars effective policy differs from initial policy",
                    )
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
        dependency_provenance(result["dependencies"])
        fixture_provenance(result["fixture"], result["case"])
        host_observation(result["host_observation"], result["measurement_context"])
        require(
            runtime_observation(result["runtime_observation"], require_polars=True)
            == global_runtime,
            "unmatched parser/dataload effective allocator/pool observation",
        )
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
        "public_report_binding": "metadata_only_use_cli_for_byte_binding"
        if model == "independent-parser-v1"
        else "not_required_by_legacy_model",
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
        parsed_budget = json.loads(inputs["budget"])
        parsed_parser = (
            json.loads(inputs["parser_report"]) if "parser_report" in inputs else None
        )
        if (
            parsed_budget.get("memory_baseline_model", "direct-v1")
            == "independent-parser-v1"
            and parsed_parser is not None
        ):
            binding = parsed_parser["public_report"]
            file_descriptor(binding, "parser public report")
            require(
                binding["sha256"] == hashlib.sha256(inputs["report"]).hexdigest()
                and binding["size_bytes"] == len(inputs["report"]),
                "parser source public report bytes do not match",
            )
        verdict = validate(
            json.loads(inputs["report"]),
            parsed_budget,
            args.variant,
            parsed_parser,
        )
        if verdict["memory_baseline_model"] == "independent-parser-v1":
            verdict["public_report_binding"] = "verified_cli_bytes_and_size"
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
