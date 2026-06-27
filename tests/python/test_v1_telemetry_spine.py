from __future__ import annotations

import csv
import json
import time

import pytest

import tools.telemetry as telemetry


def test_deep_telemetry_spine_writes_required_spans(tmp_path):
    rec = telemetry.new_record(
        worktree=".",
        dataset={"source": "synthetic", "name": "tiny", "rows": 4, "features": 2},
        config={"backend": "cpu", "gafime": {"measure": "deep_telemetry_spine"}},
    )
    start = telemetry.monotonic_ns()
    with telemetry.span(rec, "gafime_rust_orchestrator"):
        time.sleep(0)
    with telemetry.span(rec, "gafime_cpu_execution"):
        time.sleep(0)
    with telemetry.span(rec, "result_ranking"):
        time.sleep(0)
    with telemetry.span(rec, "result_materialization"):
        time.sleep(0)
    telemetry.record_kernel(
        rec,
        backend="cuda",
        name="continuous_arity1",
        duration_ns=123,
        rows=4,
        cols=2,
        arity=1,
        graph_replay=False,
    )
    telemetry.finalize_e2e_and_python_overhead(rec, start)
    telemetry.validate_deep_telemetry(rec)

    json_path, csv_path = telemetry.write_run(rec, tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in telemetry.DEEP_TELEMETRY_SPANS:
        assert isinstance(data["spans_ns"][key], int), key
    assert data["gpu_kernels"][0]["name"] == "continuous_arity1"

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "gafime_rust_orchestrator_ns" in rows[0]
    assert "result_materialization_ns" in rows[0]


def test_deep_telemetry_validation_rejects_missing_span():
    rec = telemetry.new_record(worktree=".")
    rec["spans_ns"].pop("gafime_gpu_execution", None)
    with pytest.raises(ValueError, match="Missing deep telemetry spans"):
        telemetry.validate_deep_telemetry(rec)
