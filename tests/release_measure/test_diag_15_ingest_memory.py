"""Focused guards for the disposable diagnostic, not new product policy."""

import importlib.util
import os
from pathlib import Path

import pytest

PATH = Path(__file__).with_name("diag_15_ingest_memory.py")
SPEC = importlib.util.spec_from_file_location("diag15", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_proc_fields_keep_units_and_separate_counters():
    assert MODULE.proc_fields(
        "VmRSS: 100 kB\nThreads: 4\nRss: 96 kB\nName: worker\n"
    ) == {
        "VmRSS_kib": 100,
        "Threads": 4,
        "Rss_kib": 96,
    }


@pytest.mark.parametrize(
    "name", ["normal", "glibc-trim", "decay-zero", "thp-never", "thp-data", "thp-all"]
)
def test_treatments_are_child_only(name):
    before = os.environ.copy()
    prior = {"POLARS_THP": "1", "OTHER": "retained"}
    result = MODULE.treatment_env(name, prior)
    assert os.environ == before
    assert prior == {"POLARS_THP": "1", "OTHER": "retained"}
    assert result["OTHER"] == "retained"
    if name in {"normal", "glibc-trim"}:
        assert result == prior
    else:
        assert "POLARS_THP" not in result
        assert "_RJEM_MALLOC_CONF" in result


def record(route, numeric, treatment="normal"):
    return {
        "case": {"route": route, "rows": 100000, "cols": 20},
        "label": "candidate",
        "repeat": 0,
        "treatment": treatment,
        "parser_only": False,
        "result": {"snapshot": {"numeric": numeric}},
    }


def test_numeric_parity_fails_closed_and_separates_treatments():
    assert MODULE.paired_results([record("direct", {"a": 1})])[0]["passed"] is False
    assert (
        MODULE.paired_results(
            [record("direct", {"a": 1}), record("dataload", {"a": 2})]
        )[0]["passed"]
        is False
    )
    assert (
        MODULE.paired_results(
            [record("direct", {"a": 1}), record("dataload", {"a": 1})]
        )[0]["passed"]
        is True
    )
    separated = MODULE.paired_results(
        [record("direct", {"a": 1}), record("dataload", {"a": 1}, "glibc-trim")]
    )
    assert len(separated) == 2 and not any(item["passed"] for item in separated)
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.paired_results([record("direct", {}), record("direct", {})])


def test_control_case_matrix_is_bounded():
    cases = [
        {"rows": 100000, "cols": cols, "format": fmt, "precision": precision}
        for cols in (20, 100)
        for fmt in ("csv", "parquet", "ipc")
        for precision in ("fp32", "mixed", "fp64")
    ]
    assert sum(MODULE.selected(case) for case in cases) == 4
    assert sum(MODULE.selected(case, full=True) for case in cases) == 8


def test_workflow_cannot_publish_or_change_product_source():
    root = PATH.parents[2]
    workflow = (root / ".github/workflows/rc3_rss_diagnostic.yml").read_text()
    assert "branches: [diagnostic/rc3-rss-attribution]" in workflow
    assert "contents: read" in workflow and "actions: read" in workflow
    assert "id-token" not in workflow and "environment: pypi" not in workflow
    assert "509d68a99a70182d117dad84eb905dae04764ec1" in workflow
    assert "publish_release" not in workflow and "twine" not in workflow
    assert "if: always()" in workflow


def test_budget_failure_must_be_valid_not_an_orchestration_error():
    value = {
        "schema": "gafime.public-ingestion-budget.v1.verdict",
        "status": "failed",
        "release_ready": False,
        "cells": [{}] * 18,
    }
    MODULE.validate_cost_verdict(value, 1)
    for replacement in (
        {"reason": "invalid_report"},
        {"schema": "unknown"},
        {"cells": []},
        {"release_ready": True},
    ):
        with pytest.raises(ValueError):
            MODULE.validate_cost_verdict(value | replacement, 1)
    with pytest.raises(ValueError):
        MODULE.validate_cost_verdict(value, 0)
