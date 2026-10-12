"""Independent producer boundary tests; these are not performance measurements."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).with_name("perf_14_parser_control.py")
SPEC = importlib.util.spec_from_file_location("parser_control_test", PATH)
assert SPEC and SPEC.loader
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


@pytest.mark.parametrize("dimension", ["rows", "cols"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_bad_shape_rejected_before_import_or_file_read(dimension, value):
    case = {"rows": 100, "cols": 3, "format": "parquet", dimension: value}
    with pytest.raises(ValueError, match="positive integer"):
        control.worker({"case": case})


@pytest.mark.parametrize(
    "case",
    [
        {"rows": 1, "cols": 1, "format": "json"},
        {"rows": 1, "cols": 1, "format": "csv", "precision": "fp32"},
    ],
)
def test_producer_has_no_precision_or_unknown_format_policy(case):
    with pytest.raises(ValueError, match="producer case"):
        control.worker({"case": case})


def test_parser_does_not_call_product_loader_or_execution():
    source = PATH.read_text()
    tree = ast.parse(source)
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "worker"
    )
    attrs = {node.attr for node in ast.walk(worker) if isinstance(node, ast.Attribute)}
    assert (
        not {"dataload", "analyze", "compile", "_read_frame", "_acquire_arrow_input"}
        & attrs
    )
    assert "gafime.dataloader" not in source
    assert control.READERS == {
        "csv": "read_csv",
        "parquet": "read_parquet",
        "ipc": "read_ipc",
    }


def test_incomplete_public_report_cannot_supply_parser_baseline(tmp_path):
    public = tmp_path / "public.json"
    public.write_text(
        json.dumps({"schema": control.load_public_harness().SCHEMA, "status": "failed"})
    )
    args = SimpleNamespace(
        public_report=str(public),
        output=str(tmp_path / "parser"),
        repeats=3,
        timeout=10,
        max_time_seconds=10,
    )
    with pytest.raises(ValueError, match="complete public"):
        control.driver(args)
    assert not (tmp_path / "parser").exists()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_process_bounds_fail_before_output_creation(tmp_path, timeout):
    public = tmp_path / "public.json"
    public.write_text(
        json.dumps(
            {"schema": control.load_public_harness().SCHEMA, "status": "collected"}
        )
    )
    args = SimpleNamespace(
        public_report=str(public),
        output=str(tmp_path / "parser"),
        repeats=3,
        timeout=timeout,
        max_time_seconds=10,
    )
    with pytest.raises(ValueError, match="time bounds"):
        control.driver(args)
    assert not (tmp_path / "parser").exists()
