"""Synthetic cost-gate tests; numeric fixtures are not measured performance limits."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gafime_perf14_budget", Path(__file__).with_name("perf_14_ingest_budget.py")
)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def evidence():
    variant = {
        "label": "candidate",
        "wheel": "/artifacts/core.whl",
        "python": "/env/bin/python",
        "expected_version": "1.0.0rc3",
        "source_sha": "a" * 40,
    }
    numeric = {
        "feature_names": ["f0"],
        "interactions": [{"metric_bits": "0000000000000080"}],
        "stability": [],
        "permutations": [],
        "metric_rankings": {},
        "signal_detected": False,
    }
    digest = hashlib.sha256(
        json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    records, policies = [], []
    for cols in (20, 100):
        case = {
            "rows": 100_000,
            "cols": cols,
            "format": "parquet",
            "precision": "mixed",
            "workflow": "light",
            "cache_state": "miss",
            "backend": "auto-core",
        }
        policies.append(
            {
                "case": case,
                "fixed_startup_ns": 100,
                "direct_multiplier": 2,
                "fixed_format_allowance_bytes": 1024,
                "extra_bytes_per_cell": 1,
            }
        )
        for repeat in range(3):
            for route in ("direct", "dataload"):
                route_case = {**case, "route": route}
                native = {"path": "/env/gafime/gafime_py.so", "sha256": "b" * 64}
                result = {
                    "schema": gate.REPORT_SCHEMA + ".worker",
                    "instrumented": False,
                    "variant": variant,
                    "case": route_case,
                    "identity": {
                        "byte_identical_installed_package": True,
                        "expected_version": variant["expected_version"],
                        "wheel": {"path": variant["wheel"], "sha256": "c" * 64},
                        "verified_members": [native],
                    },
                    "native": native,
                    "harness": {"sha256": "d" * 64},
                    "dependencies": {"numpy": "test", "polars": "test"},
                    "fixture": {"source_dtype": "float64", "sha256": "e" * 64},
                    "changed_fixture": {"sha256": "f" * 64},
                    "config": {"precision": "mixed"},
                    "timing": {"wall_nanoseconds": 10 if route == "direct" else 20},
                    "memory": {
                        "after_import": memory(1000),
                        "after_call_before_snapshot": memory(
                            2000 if route == "direct" else 2500
                        ),
                        "inherited_resource_ru_maxrss_used": False,
                    },
                    "snapshot_generated_after_measurement": True,
                    "snapshot": {"numeric": numeric, "numeric_sha256": digest},
                }
                records.append(
                    {
                        "status": "passed",
                        "variant": "candidate",
                        "case": route_case,
                        "repeat": repeat,
                        "result": result,
                    }
                )
    report = {
        "schema": gate.REPORT_SCHEMA,
        "status": "collected",
        "collector_status": "passed",
        "public_numeric_parity": {"status": "passed"},
        "records": records,
        "requested_jobs": len(records),
        "variants": [variant],
        "harness": {"sha256": "d" * 64},
    }
    budget = {
        "schema": gate.BUDGET_SCHEMA,
        "min_samples": 3,
        "require_source_sha": True,
        "source_dtype": "float64",
        "cells": policies,
    }
    return deepcopy(report), deepcopy(budget)


def memory(high):
    return {
        "source": "linux_proc_self_status",
        "VmRSS_kib": high - 1,
        "VmHWM_kib": high,
    }


def loaded_wide(report):
    return [
        r["result"]
        for r in report["records"]
        if r["case"]["cols"] == 100 and r["case"]["route"] == "dataload"
    ]


def parser_evidence():
    """Six synthetic producer cells, not calibrated performance evidence."""
    original, budget = evidence()
    original_policies = deepcopy(budget["cells"])
    report = deepcopy(original)
    report["records"] = []
    budget["cells"] = []
    budget["memory_baseline_model"] = "independent-parser-v1"
    context = {
        "host": {
            "hostname": "synthetic-runner",
            "boot_id": "synthetic-boot",
            "github_run_id": "1",
            "github_run_attempt": "1",
        },
        "python_version": "3.11.17 synthetic",
        "platform": "Linux-synthetic",
        "libc": ["glibc", "2.39"],
        "affinity": [0, 1, 2, 3],
        "cpu_count": 4,
        "initial_allocator_policy": {key: None for key in gate.ALLOCATOR_POLICY_KEYS},
        "thp": {
            "enabled": "[always] madvise never",
            "defrag": "always defer defer+madvise [madvise] never",
            "shmem_enabled": "always within_size advise [never] deny force",
            "hpage_pmd_size": "2097152",
        },
    }
    for fmt in ("csv", "parquet", "ipc"):
        for policy in original_policies:
            clone = deepcopy(policy)
            clone["case"]["format"] = fmt
            budget["cells"].append(clone)
        for record in original["records"]:
            clone = deepcopy(record)
            result = clone["result"]
            clone["case"]["format"] = result["case"]["format"] = fmt
            shape = str(clone["case"]["cols"])
            fixture_digest = hashlib.sha256(f"{fmt}/{shape}".encode()).hexdigest()
            result["fixture"] = {
                "file": {
                    "path": f"/fixtures/{shape}.{fmt}",
                    "sha256": fixture_digest,
                    "size_bytes": 1024,
                },
                "features": {
                    "path": f"/fixtures/{shape}-{fmt}-X.npy",
                    "sha256": "e" * 64,
                    "size_bytes": 100_000 * clone["case"]["cols"] * 8 + 128,
                },
                "target": {
                    "path": f"/fixtures/{shape}-{fmt}-y.npy",
                    "sha256": "f" * 64,
                    "size_bytes": 800_128,
                },
                "feature_names": [f"f{i}" for i in range(clone["case"]["cols"])],
                "parser_version": "1.44.2",
                "source_dtype": "float64",
            }
            result["dependencies"] = {
                "numpy": "2.4.6",
                "polars": "1.44.2",
                "numpy_native_sha256": "9" * 64,
                "polars_native_members": [
                    {
                        "distribution": "polars-runtime-32",
                        "member": "_polars_runtime_32/_polars_runtime.abi3.so",
                        "sha256": "8" * 64,
                        "size_bytes": 180_192_520,
                    }
                ],
            }
            result["native"]["size_bytes"] = 1024
            result["identity"]["wheel"]["size_bytes"] = 2048
            result["identity"]["verified_members"][0]["size_bytes"] = 1024
            result["python"] = {
                "executable": result["variant"]["python"],
                "version": context["python_version"],
            }
            result["measurement_context"] = deepcopy(context)
            result["environment"] = {
                "platform": context["platform"],
                "affinity": deepcopy(context["affinity"]),
                "cpu_count": context["cpu_count"],
                "RAYON_NUM_THREADS": None,
                "POLARS_MAX_THREADS": None,
                "auto_gpu_payloads_explicitly_disabled": True,
                "load_average": [0.1, 0.2, 0.3],
            }
            result["host_observation"] = {
                "cpu_info": "synthetic CPU information",
                "system_memory": "MemTotal: 16000000 kB",
                "kernel": {
                    "system": "Linux",
                    "node": context["host"]["hostname"],
                    "release": "synthetic",
                    "version": "synthetic kernel",
                    "machine": "synthetic",
                    "processor": "",
                },
                "scope": "post-measurement snapshots; not constant free memory",
            }
            result["runtime_observation"] = {
                "allocator_policy": deepcopy(context["initial_allocator_policy"]),
                "polars_loaded": clone["case"]["route"] == "dataload",
                "polars_thread_pool_size": 4
                if clone["case"]["route"] == "dataload"
                else None,
                "scope": "post-measurement observation; no forced imports",
            }
            if result["runtime_observation"]["polars_loaded"]:
                result["runtime_observation"]["allocator_policy"][
                    "_RJEM_MALLOC_CONF"
                ] = "dirty_decay_ms:500,muzzy_decay_ms:1000"
            report["records"].append(clone)
    report["requested_jobs"] = len(report["records"])
    producer = {
        "schema": gate.PARSER_SCHEMA,
        "status": "collected",
        "collector_status": "passed",
        "variants": deepcopy(report["variants"]),
        "harness": {"sha256": "a" * 64},
        "public_harness": deepcopy(report["harness"]),
        "records": [],
    }
    for record in report["records"]:
        if record["case"]["route"] != "direct":
            continue
        result = deepcopy(record["result"])
        case = {field: record["case"][field] for field in gate.PARSER_CASE_KEYS}
        result.update(
            {
                "schema": gate.PARSER_SCHEMA + ".worker",
                "case": case,
                "harness": producer["harness"],
                "public_harness": producer["public_harness"],
                "instrumented": False,
                "treatment": "normal",
                "numerical_acquisition": False,
                "frames_retained_until_snapshot": True,
                "reader": {
                    "name": "read_" + case["format"],
                    "kwargs": {},
                    "features": result["fixture"]["feature_names"],
                    "target": "target",
                },
            }
        )
        result["memory"]["after_call_before_snapshot"] = memory(2200)
        result["runtime_observation"]["polars_loaded"] = True
        result["runtime_observation"]["polars_thread_pool_size"] = 4
        result["runtime_observation"]["allocator_policy"]["_RJEM_MALLOC_CONF"] = (
            "dirty_decay_ms:500,muzzy_decay_ms:1000"
        )
        producer["records"].append(
            {
                "status": "passed",
                "variant": "candidate",
                "case": case,
                "repeat": record["repeat"],
                "result": result,
            }
        )
    producer["requested_jobs"] = len(producer["records"])
    public_bytes = json.dumps(report).encode()
    producer["public_report"] = {
        "path": "/original-runner/public/report.json",
        "sha256": hashlib.sha256(public_bytes).hexdigest(),
        "size_bytes": len(public_bytes),
    }
    return report, budget, producer


def parser_case_results(producer, *, cols=20, fmt="parquet"):
    return [
        record["result"]
        for record in producer["records"]
        if record["case"]["cols"] == cols and record["case"]["format"] == fmt
    ]


def public_case_results(report, *, cols=20, fmt="parquet"):
    return [
        record["result"]
        for record in report["records"]
        if record["case"]["cols"] == cols
        and record["case"]["format"] == fmt
        and record["case"]["route"] == "dataload"
    ]


def revised_case(verdict, *, cols=20, fmt="parquet"):
    return next(
        cell
        for cell in verdict["cells"]
        if cell["case"]["cols"] == cols and cell["case"]["format"] == fmt
    )


def test_explicit_legacy_model_preserves_old_budget_without_parser():
    report, budget = evidence()
    absent = gate.validate(report, budget, "candidate")
    budget["memory_baseline_model"] = "direct-v1"
    explicit = gate.validate(report, budget, "candidate")
    assert explicit == absent


@pytest.mark.parametrize("model", [None, False, "unknown", [], {}])
def test_unknown_memory_model_fails_closed(model):
    report, budget = evidence()
    budget["memory_baseline_model"] = model
    with pytest.raises(ValueError, match="model"):
        gate.validate(report, budget, "candidate")


def test_parser_model_requires_independent_report():
    report, budget, _ = parser_evidence()
    with pytest.raises(ValueError, match="parser report required"):
        gate.validate(report, budget, "candidate")


def test_parser_aware_model_preserves_coefficients_and_exposes_old_failure():
    report, budget, producer = parser_evidence()
    for result in parser_case_results(producer):
        result["memory"]["after_call_before_snapshot"] = memory(3000)
    for result in public_case_results(report):
        result["memory"]["after_call_before_snapshot"] = memory(4800)
    revised = gate.validate(report, budget, "candidate", producer)
    budget["memory_baseline_model"] = "direct-v1"
    old = gate.validate(report, budget, "candidate")
    assert old["status"] == "failed" and revised["status"] == "passed"
    old_cell, new_cell = revised_case(old), revised_case(revised)
    assert old_cell["failures"] == ["memory"]
    assert new_cell["old_rss_limit_bytes"] == old_cell["rss_limit_bytes"]
    assert new_cell["rss_limit_bytes"] == 3000 * 1024 + 1024 + 2_000_000
    assert new_cell["parser_rss_limit_bytes"] == old_cell["rss_limit_bytes"]
    assert new_cell["parser_wall_limit_ns"] == old_cell["wall_limit_ns"]
    assert new_cell["parser_baseline_applied"] is True
    assert revised["release_ready"] is False
    assert len(revised["cells"]) == 6


def test_six_independent_producers_cover_all_eighteen_profile_cells():
    report, budget, producer = parser_evidence()
    for precision in ("fp32", "fp64"):
        for record in list(report["records"]):
            if record["case"]["precision"] != "mixed":
                continue
            clone = deepcopy(record)
            clone["case"]["precision"] = precision
            clone["result"]["case"]["precision"] = precision
            clone["result"]["config"]["precision"] = precision
            report["records"].append(clone)
        for policy in list(budget["cells"]):
            if policy["case"]["precision"] != "mixed":
                continue
            clone = deepcopy(policy)
            clone["case"]["precision"] = precision
            budget["cells"].append(clone)
    report["requested_jobs"] = len(report["records"])
    verdict = gate.validate(report, budget, "candidate", producer)
    assert verdict["status"] == "passed"
    assert len(verdict["cells"]) == 18
    assert len(producer["records"]) == 18
    assert all(cell["parser_samples"] == 3 for cell in verdict["cells"])


@pytest.mark.parametrize("failure", ["parser_memory", "parser_latency"])
def test_parser_cannot_self_normalize_its_memory_or_latency(failure):
    report, budget, producer = parser_evidence()
    for result in parser_case_results(producer):
        if failure == "parser_memory":
            result["memory"]["after_call_before_snapshot"] = memory(100_000)
        else:
            result["timing"]["wall_nanoseconds"] = 121
    for result in public_case_results(report):
        result["memory"]["after_call_before_snapshot"] = memory(4800)
    verdict = gate.validate(report, budget, "candidate", producer)
    cell = revised_case(verdict)
    assert verdict["status"] == "failed"
    assert cell["failures"] == [failure, "memory"]
    assert cell["rss_limit_bytes"] == cell["old_rss_limit_bytes"]
    assert cell["parser_baseline_applied"] is False


@pytest.mark.parametrize("boundary", ["rss", "latency"])
def test_parser_cap_equality_passes_and_one_unit_above_fails(boundary):
    report, budget, producer = parser_evidence()
    if boundary == "rss":
        for policy in budget["cells"]:
            policy["extra_bytes_per_cell"] = 0
        for result in producer["records"]:
            result["result"]["memory"]["after_call_before_snapshot"] = memory(2001)
        for record in report["records"]:
            if record["case"]["route"] == "dataload":
                record["result"]["memory"]["after_call_before_snapshot"] = memory(2002)
    else:
        for result in producer["records"]:
            result["result"]["timing"]["wall_nanoseconds"] = 120
    verdict = gate.validate(report, budget, "candidate", producer)
    assert verdict["status"] == "passed"
    for result in parser_case_results(producer):
        if boundary == "rss":
            result["memory"]["after_call_before_snapshot"] = memory(2002)
        else:
            result["timing"]["wall_nanoseconds"] = 121
    assert gate.validate(report, budget, "candidate", producer)["status"] == "failed"


@pytest.mark.parametrize(
    "failure", ["missing", "repeat", "duplicate", "failed", "unexpected", "count"]
)
def test_missing_or_corrupt_parser_coverage_fails_closed(failure):
    report, budget, producer = parser_evidence()
    if failure == "missing":
        producer["records"] = [
            record
            for record in producer["records"]
            if record["case"]["format"] != "ipc"
        ]
    elif failure == "repeat":
        producer["records"].pop(0)
    elif failure == "duplicate":
        producer["records"].append(deepcopy(producer["records"][0]))
    elif failure == "failed":
        producer["records"][0]["status"] = "failed"
    elif failure == "unexpected":
        producer["records"][0]["case"]["rows"] = 123
        producer["records"][0]["result"]["case"]["rows"] = 123
    producer["requested_jobs"] = len(producer["records"]) + (failure == "count")
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "native",
        "python",
        "dependencies",
        "fixture",
        "measurement_context",
        "variant",
        "harness",
        "public_harness",
        "reader",
    ],
)
def test_parser_identity_runtime_input_and_reader_must_match(field):
    report, budget, producer = parser_evidence()
    result = producer["records"][0]["result"]
    result[field] = (
        {**result[field], "sha256": "0" * 64}
        if field in ("harness", "public_harness")
        else {**result[field], "changed": True}
    )
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("instrumented", True),
        ("instrumented", 0),
        ("treatment", "decay-zero"),
        ("numerical_acquisition", True),
        ("frames_retained_until_snapshot", False),
    ],
)
def test_instrumented_or_nonindependent_parser_samples_never_qualify(field, value):
    report, budget, producer = parser_evidence()
    producer["records"][0]["result"][field] = value
    with pytest.raises(ValueError, match="scope"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("VmHWM_kib", float("nan")),
        ("VmHWM_kib", -1),
        ("VmRSS_kib", 100_000),
        ("source", "resource_ru_maxrss"),
    ],
)
def test_invalid_parser_memory_fails_closed(field, value):
    report, budget, producer = parser_evidence()
    producer["records"][0]["result"]["memory"]["after_call_before_snapshot"][field] = (
        value
    )
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate", producer)


def test_parser_does_not_waive_numeric_parity_or_inherited_rss():
    report, budget, producer = parser_evidence()
    producer["records"][0]["result"]["memory"]["inherited_resource_ru_maxrss_used"] = (
        True
    )
    with pytest.raises(ValueError, match="parser RSS"):
        gate.validate(report, budget, "candidate", producer)
    report["public_numeric_parity"]["status"] = "failed"
    with pytest.raises(ValueError, match="parity"):
        gate.validate(report, budget, "candidate", producer)


def test_parser_model_cannot_borrow_candidate_controls_for_rc2():
    report, budget, producer = parser_evidence()
    report["variants"][0]["source_sha"] = "f" * 40
    for record in report["records"]:
        record["result"]["variant"]["source_sha"] = "f" * 40
    with pytest.raises(ValueError, match="variant/source"):
        gate.validate(report, budget, "candidate", producer)


def test_large_materialization_regression_still_fails_revised_memory_and_latency():
    report, budget, producer = parser_evidence()
    for result in public_case_results(report):
        result["memory"]["after_call_before_snapshot"] = memory(100_000)
        result["timing"]["wall_nanoseconds"] = 10_000
    verdict = gate.validate(report, budget, "candidate", producer)
    assert revised_case(verdict)["failures"] == ["latency", "memory"]


def test_checked_budget_selects_new_model_without_numeric_allowance_changes():
    current = json.loads(
        Path(gate.__file__).with_name("ingest_cost_budget.json").read_text()
    )
    assert current["memory_baseline_model"] == "independent-parser-v1"
    assert len(current["cells"]) == 18
    for policy in current["cells"]:
        assert policy["fixed_startup_ns"] == 250_000_000
        assert policy["direct_multiplier"] == 4
        assert policy["fixed_format_allowance_bytes"] == 20 * 1024 * 1024
        assert policy["extra_bytes_per_cell"] == 12


@pytest.mark.parametrize("field", sorted(gate.CONTEXT_KEYS))
def test_matching_but_incomplete_context_cannot_qualify(field):
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        record["result"]["measurement_context"].pop(field)
    with pytest.raises(ValueError, match="context"):
        gate.validate(report, budget, "candidate", producer)


def test_public_context_must_match_across_all_producer_cells():
    report, budget, producer = parser_evidence()
    for record in report["records"]:
        if record["case"]["format"] == "ipc":
            record["result"]["measurement_context"]["host"]["boot_id"] = (
                "different-boot"
            )
    with pytest.raises(ValueError, match="environment"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize("field", sorted(gate.ALLOCATOR_POLICY_KEYS))
def test_every_diagnostic_allocator_policy_key_is_required(field):
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        record["result"]["measurement_context"]["initial_allocator_policy"].pop(field)
    with pytest.raises(ValueError, match="initial allocator policy"):
        gate.validate(report, budget, "candidate", producer)


def test_thp_shmem_configuration_cannot_be_omitted():
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        record["result"]["measurement_context"]["thp"].pop("shmem_enabled")
    with pytest.raises(ValueError, match="THP context"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("polars_loaded", False),
        ("polars_loaded", 1),
        ("polars_thread_pool_size", None),
        ("polars_thread_pool_size", 0),
        ("polars_thread_pool_size", -1),
        ("polars_thread_pool_size", float("nan")),
        ("polars_thread_pool_size", False),
        ("scope", "during measurement"),
    ],
)
def test_parser_effective_runtime_observation_must_be_sane(field, value):
    report, budget, producer = parser_evidence()
    producer["records"][0]["result"]["runtime_observation"][field] = value
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize("source", ["parser", "dataload"])
@pytest.mark.parametrize("field", ["allocator_policy", "polars_thread_pool_size"])
def test_parser_and_dataload_effective_policy_and_pool_must_match(source, field):
    report, budget, producer = parser_evidence()
    result = (
        parser_case_results(producer)[0]
        if source == "parser"
        else public_case_results(report)[0]
    )
    if field == "allocator_policy":
        result["runtime_observation"][field]["MALLOC_TRIM_THRESHOLD_"] = "1048576"
    else:
        result["runtime_observation"][field] = 2
    with pytest.raises(ValueError, match="allocator/pool"):
        gate.validate(report, budget, "candidate", producer)


def test_direct_route_may_stay_unloaded_but_must_match_if_polars_is_loaded():
    report, budget, producer = parser_evidence()
    for record in report["records"]:
        if record["case"]["route"] != "direct":
            continue
        observation = record["result"]["runtime_observation"]
        assert observation["polars_loaded"] is False
        assert observation["polars_thread_pool_size"] is None
    assert gate.validate(report, budget, "candidate", producer)["status"] == "passed"
    for record in report["records"]:
        if record["case"]["route"] == "direct":
            observation = record["result"]["runtime_observation"]
            observation["polars_loaded"] = True
            observation["polars_thread_pool_size"] = 2
            observation["allocator_policy"]["_RJEM_MALLOC_CONF"] = (
                "different-direct-policy"
            )
    with pytest.raises(ValueError, match="global effective"):
        gate.validate(report, budget, "candidate", producer)


def test_missing_effective_allocator_key_cannot_qualify():
    report, budget, producer = parser_evidence()
    producer["records"][0]["result"]["runtime_observation"]["allocator_policy"].pop(
        "JEMALLOC_CONF"
    )
    with pytest.raises(ValueError, match="effective allocator policy"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize(
    "field,key",
    [
        ("dependencies", key)
        for key in ("numpy", "polars", "polars_native_members", "numpy_native_sha256")
    ]
    + [
        ("fixture", key)
        for key in (
            "file",
            "features",
            "target",
            "feature_names",
            "parser_version",
            "source_dtype",
        )
    ]
    + [
        ("environment", key)
        for key in (
            "platform",
            "affinity",
            "cpu_count",
            "RAYON_NUM_THREADS",
            "POLARS_MAX_THREADS",
            "auto_gpu_payloads_explicitly_disabled",
            "load_average",
        )
    ]
    + [
        ("host_observation", key)
        for key in ("cpu_info", "system_memory", "kernel", "scope")
    ],
)
def test_equal_but_stripped_provenance_does_not_qualify(field, key):
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        if field in record["result"]:
            record["result"][field].pop(key)
    with pytest.raises((ValueError, KeyError)):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize("key", ["distribution", "member", "sha256", "size_bytes"])
def test_matching_incomplete_polars_member_identity_does_not_qualify(key):
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        record["result"]["dependencies"]["polars_native_members"][0].pop(key)
    with pytest.raises(ValueError, match="Polars native member"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize("key", ["path", "sha256", "size_bytes"])
@pytest.mark.parametrize("member", ["file", "features", "target"])
def test_matching_incomplete_fixture_identity_does_not_qualify(key, member):
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        record["result"]["fixture"][member].pop(key)
    with pytest.raises(ValueError, match="fixture"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize("mutation", ["duplicate", "target", "short", "blank"])
def test_matching_invalid_feature_names_do_not_qualify(mutation):
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        names = record["result"]["fixture"]["feature_names"]
        if mutation == "duplicate":
            names[0] = names[1]
        elif mutation == "target":
            names[0] = "target"
        elif mutation == "short":
            names.pop()
        else:
            names[0] = ""
    with pytest.raises(ValueError, match="feature names"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_count", 4.0),
        ("affinity", [0.0, 1, 2, 3]),
        ("load_average", [float("nan"), 0, 0]),
        ("auto_gpu_payloads_explicitly_disabled", 1),
    ],
)
def test_public_legacy_observation_types_are_checked(field, value):
    report, budget, producer = parser_evidence()
    report["records"][0]["result"]["environment"][field] = value
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize(
    "key", ["system", "node", "release", "version", "machine", "processor"]
)
def test_matching_incomplete_uname_does_not_qualify(key):
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        record["result"]["host_observation"]["kernel"].pop(key)
    with pytest.raises(ValueError, match="uname"):
        gate.validate(report, budget, "candidate", producer)


def test_changed_effective_policy_on_one_whole_cell_is_not_hidden_by_pair_matching():
    report, budget, producer = parser_evidence()
    for record in report["records"] + producer["records"]:
        if (
            record["case"]["format"] == "ipc"
            and record["result"]["runtime_observation"]["polars_loaded"]
        ):
            record["result"]["runtime_observation"]["allocator_policy"][
                "MALLOC_ARENA_MAX"
            ] = "1"
    with pytest.raises(ValueError, match="global effective"):
        gate.validate(report, budget, "candidate", producer)


def test_unloaded_direct_observation_must_preserve_initial_policy():
    report, budget, producer = parser_evidence()
    report["records"][0]["result"]["runtime_observation"]["allocator_policy"][
        "JEMALLOC_CONF"
    ] = "unexpected"
    with pytest.raises(ValueError, match="unloaded Polars effective policy"):
        gate.validate(report, budget, "candidate", producer)


@pytest.mark.parametrize("key", ["path", "sha256", "size_bytes"])
def test_structured_parser_validation_requires_public_report_identity(key):
    report, budget, producer = parser_evidence()
    producer["public_report"].pop(key)
    with pytest.raises(ValueError, match="parser public report identity"):
        gate.validate(report, budget, "candidate", producer)


def test_structured_validation_does_not_claim_raw_serialization_binding():
    report, budget, producer = parser_evidence()
    producer["public_report"]["sha256"] = "0" * 64
    verdict = gate.validate(report, budget, "candidate", producer)
    assert verdict["public_report_binding"] == "metadata_only_use_cli_for_byte_binding"


def test_parser_cli_requires_and_hashes_producer_report(tmp_path):
    report, budget, producer = parser_evidence()
    paths = {
        name: tmp_path / (name + ".json")
        for name in ("report", "budget", "parser_report")
    }
    for name, value in (
        ("report", report),
        ("budget", budget),
        ("parser_report", producer),
    ):
        paths[name].write_text(json.dumps(value))
    command = [
        sys.executable,
        "-I",
        str(gate.__file__),
        "--report",
        str(paths["report"]),
        "--budget",
        str(paths["budget"]),
        "--variant",
        "candidate",
    ]
    absent = tmp_path / "absent.json"
    result = subprocess.run(
        command + ["--output", str(absent)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "parser report required" in json.loads(absent.read_text())["reason"]
    complete = tmp_path / "complete.json"
    result = subprocess.run(
        command
        + ["--parser-report", str(paths["parser_report"]), "--output", str(complete)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads(complete.read_text())
    assert verdict["status"] == "passed" and verdict["release_ready"] is False
    assert verdict["public_report_binding"] == "verified_cli_bytes_and_size"
    assert producer["public_report"]["path"] != str(paths["report"])
    assert (
        verdict["inputs"]["parser_report"]["sha256"]
        == hashlib.sha256(paths["parser_report"].read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mismatch", ["hash", "size", "identical_object_different_bytes"]
)
def test_parser_cli_rejects_wrong_public_source_bytes(tmp_path, mismatch):
    report, budget, producer = parser_evidence()
    raw = json.dumps(report)
    if mismatch == "hash":
        producer["public_report"]["sha256"] = "0" * 64
    elif mismatch == "size":
        producer["public_report"]["size_bytes"] += 1
    else:
        raw = raw.replace(": ", ":\t", 1)
        assert json.loads(raw) == report
    paths = {
        name: tmp_path / (name + ".json")
        for name in ("report", "budget", "parser_report", "verdict")
    }
    paths["report"].write_text(raw)
    paths["budget"].write_text(json.dumps(budget))
    paths["parser_report"].write_text(json.dumps(producer))
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(gate.__file__),
            "--report",
            str(paths["report"]),
            "--budget",
            str(paths["budget"]),
            "--variant",
            "candidate",
            "--parser-report",
            str(paths["parser_report"]),
            "--output",
            str(paths["verdict"]),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    verdict = json.loads(paths["verdict"].read_text())
    assert verdict["status"] == "failed"
    assert "public report bytes do not match" in verdict["reason"]


def test_explicit_budget_passes_without_release_readiness_claim():
    report, budget = evidence()
    verdict = gate.validate(report, budget, "candidate")
    assert verdict["status"] == "passed"
    assert verdict["release_ready"] is False
    assert len(verdict["cells"]) == 2
    assert verdict["cells"][0]["wall_limit_ns"] == 120
    assert verdict["cells"][0]["rss_limit_bytes"] == 2000 * 1024 + 1024 + 2_000_000


@pytest.mark.parametrize("failure", ["latency", "memory"])
def test_exactly_one_declared_cell_can_fail_one_cost_boundary(failure):
    report, budget = evidence()
    for result in loaded_wide(report):
        if failure == "latency":
            result["timing"]["wall_nanoseconds"] = 121
        else:
            result["memory"]["after_call_before_snapshot"] = memory(20_000)
    verdict = gate.validate(report, budget, "candidate")
    assert verdict["status"] == "failed"
    assert [c["failures"] for c in verdict["cells"]] == [[], [failure]]


def test_large_import_floor_is_not_an_unsafe_small_cell_rss_ratio():
    report, budget = evidence()
    for result in loaded_wide(report):
        result["memory"]["after_import"] = memory(30_000)
        result["memory"]["after_call_before_snapshot"] = memory(30_000)
    assert gate.validate(report, budget, "candidate")["status"] == "passed"


@pytest.mark.parametrize("value", [None, False, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", gate.BOUND_KEYS)
def test_malformed_nonfinite_or_negative_cost_budgets_fail_closed(field, value):
    report, budget = evidence()
    budget["cells"][0][field] = value
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate")


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_sample_minimum_requires_positive_integer(value):
    report, budget = evidence()
    budget["min_samples"] = value
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate")


@pytest.mark.parametrize("mutation", ["wide", "route", "repeat", "duplicate"])
def test_missing_required_wide_or_matched_sample_coverage_fails(mutation):
    report, budget = evidence()
    if mutation == "wide":
        report["records"] = [r for r in report["records"] if r["case"]["cols"] == 20]
    elif mutation == "duplicate":
        report["records"].append(deepcopy(report["records"][0]))
    else:
        report["records"].pop(0)
    report["requested_jobs"] = len(report["records"])
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate")


def test_insufficient_paired_repeats_fail_even_when_collector_claims_complete():
    report, budget = evidence()
    report["records"] = [r for r in report["records"] if r["repeat"] < 2]
    report["requested_jobs"] = len(report["records"])
    with pytest.raises(ValueError, match="repeat"):
        gate.validate(report, budget, "candidate")


def test_unreviewed_source_dtype_cannot_bypass_memory_cost_budget():
    report, budget = evidence()
    budget["source_dtype"] = "float32"
    with pytest.raises(ValueError, match="dtype"):
        gate.validate(report, budget, "candidate")


def test_process_high_water_cannot_decrease_after_import():
    report, budget = evidence()
    loaded_wide(report)[0]["memory"]["after_import"] = memory(30_000)
    with pytest.raises(ValueError, match="decreased"):
        gate.validate(report, budget, "candidate")


@pytest.mark.parametrize("failure", ["failed", "missing", "changed_bits"])
def test_numeric_parity_is_rechecked_instead_of_trusting_passed_text(failure):
    report, budget = evidence()
    if failure == "failed":
        report["public_numeric_parity"]["status"] = "failed"
    elif failure == "missing":
        report.pop("public_numeric_parity")
    else:
        result = loaded_wide(report)[0]
        result["snapshot"]["numeric"] = {
            **result["snapshot"]["numeric"],
            "signal_detected": True,
        }
        result["snapshot"]["numeric_sha256"] = hashlib.sha256(
            json.dumps(
                result["snapshot"]["numeric"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    with pytest.raises((ValueError, KeyError)):
        gate.validate(report, budget, "candidate")


def test_matching_but_incomplete_numeric_snapshots_are_not_full_parity():
    report, budget = evidence()
    numeric = {"interactions": []}
    digest = hashlib.sha256(b'{"interactions":[]}').hexdigest()
    for record in report["records"]:
        record["result"]["snapshot"] = {
            "numeric": numeric,
            "numeric_sha256": digest,
        }
    with pytest.raises(ValueError, match="incomplete numeric"):
        gate.validate(report, budget, "candidate")


@pytest.mark.parametrize("field", ["fixture", "changed_fixture", "config"])
def test_unmatched_pair_inputs_or_configuration_fail_closed(field):
    report, budget = evidence()
    loaded_wide(report)[0][field] = {**loaded_wide(report)[0][field], "changed": True}
    with pytest.raises(ValueError, match="unmatched"):
        gate.validate(report, budget, "candidate")


@pytest.mark.parametrize("identity", ["native", "wheel", "harness", "variant"])
def test_mixed_native_wheel_harness_or_source_identity_fails(identity):
    report, budget = evidence()
    result = loaded_wide(report)[0]
    if identity == "variant":
        result["variant"] = {**result["variant"], "source_sha": "0" * 40}
    elif identity == "wheel":
        result["identity"]["wheel"]["sha256"] = "0" * 64
    else:
        result[identity] = {**result[identity], "sha256": "0" * 64}
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate")


def test_null_source_sha_is_only_explicitly_allowed_development_evidence():
    report, budget = evidence()
    report["variants"][0]["source_sha"] = None
    for record in report["records"]:
        record["result"]["variant"]["source_sha"] = None
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate")
    budget["require_source_sha"] = False
    verdict = gate.validate(report, budget, "candidate")
    assert verdict["status"] == "passed"
    assert verdict["source_binding"] == "uncommitted_development_only"
    assert verdict["release_ready"] is False


def test_instrumented_records_cannot_make_budget_or_sample_coverage_pass():
    report, budget = evidence()
    extra = deepcopy(report["records"][0])
    extra["result"]["instrumented"] = True
    extra["result"]["timing"]["wall_nanoseconds"] = 1e30
    report["records"].append(extra)
    report["requested_jobs"] += 1
    assert gate.validate(report, budget, "candidate")["status"] == "passed"
    for record in report["records"]:
        record["result"]["instrumented"] = True
    with pytest.raises(ValueError):
        gate.validate(report, budget, "candidate")


@pytest.mark.parametrize("source", ["unavailable", "resource_ru_maxrss"])
def test_unavailable_or_inherited_rss_is_not_a_passing_memory_gate(source):
    report, budget = evidence()
    loaded_wide(report)[0]["memory"]["after_import"]["source"] = source
    with pytest.raises(ValueError, match="RSS"):
        gate.validate(report, budget, "candidate")


def test_cli_preserves_failed_verdict_and_refuses_to_overwrite(tmp_path):
    report, budget = evidence()
    report["public_numeric_parity"]["status"] = "failed"
    report_path, budget_path, output = (
        tmp_path / "report.json",
        tmp_path / "budget.json",
        tmp_path / "verdict.json",
    )
    report_path.write_text(json.dumps(report))
    budget_path.write_text(json.dumps(budget))
    command = [
        sys.executable,
        "-I",
        str(gate.__file__),
        "--report",
        str(report_path),
        "--budget",
        str(budget_path),
        "--variant",
        "candidate",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    verdict = json.loads(output.read_text())
    assert verdict["status"] == "failed" and verdict["release_ready"] is False
    assert "parity" in verdict["reason"]
    summary = json.loads(result.stdout)
    assert summary == {
        "status": "failed",
        "reason": verdict["reason"],
        "checked_cells": 0,
        "failed_cells": [],
        "release_ready": False,
    }
    assert (
        verdict["inputs"]["report"]["sha256"]
        == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )
    previous = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and output.read_bytes() == previous


def test_cli_invalid_json_retains_failure_without_touching_collector(tmp_path):
    report_path, budget_path, output = (
        tmp_path / "report.json",
        tmp_path / "budget.json",
        tmp_path / "verdict.json",
    )
    report_path.write_text("{invalid")
    budget_path.write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(gate.__file__),
            "--report",
            str(report_path),
            "--budget",
            str(budget_path),
            "--variant",
            "candidate",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(output.read_text())["status"] == "failed"
    assert json.loads(result.stdout)["reason"]
    assert report_path.read_text() == "{invalid"


@pytest.mark.parametrize("fail_memory", [False, True])
def test_cli_prints_cost_failures_without_changing_limits(tmp_path, fail_memory):
    report, budget = evidence()
    if fail_memory:
        for result in loaded_wide(report):
            result["memory"]["after_call_before_snapshot"] = memory(100_000)
    report_path, budget_path, output = (
        tmp_path / "report.json",
        tmp_path / "budget.json",
        tmp_path / "verdict.json",
    )
    report_path.write_text(json.dumps(report))
    budget_path.write_text(json.dumps(budget))
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(gate.__file__),
            "--report",
            str(report_path),
            "--budget",
            str(budget_path),
            "--variant",
            "candidate",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    verdict, summary = json.loads(output.read_text()), json.loads(result.stdout)
    assert result.returncode == int(fail_memory)
    assert summary["status"] == verdict["status"]
    assert summary["checked_cells"] == len(verdict["cells"])
    assert summary["failed_cells"] == [
        cell for cell in verdict["cells"] if cell["failures"]
    ]
    assert summary["release_ready"] is False
    assert budget_path.read_text() == json.dumps(budget)
