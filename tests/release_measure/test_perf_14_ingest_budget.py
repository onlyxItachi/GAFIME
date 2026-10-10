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
