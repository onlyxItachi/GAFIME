"""Bounded tests for perf14 plumbing; these need no native/GPU installation."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gafime_perf14", Path(__file__).with_name("perf_14_public_ingest.py")
)
assert _SPEC and _SPEC.loader
perf14 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = perf14
_SPEC.loader.exec_module(perf14)


def test_import_driver_does_not_import_native_or_array_packages() -> None:
    script = str(Path(perf14.__file__))
    code = (
        "import runpy,sys; runpy.run_path(sys.argv[1]); "
        "assert not any(name.split('.')[0] in {'gafime','numpy','polars'} "
        "for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_reviewed_cost_tripwire_covers_both_reference_shapes_and_all_formats_profiles() -> (
    None
):
    """Keep the compact regression budget tied to the declared public matrix."""
    budget = json.loads(Path(__file__).with_name("ingest_cost_budget.json").read_text())
    assert budget["require_source_sha"] is True
    assert budget["source_dtype"] == "float64"
    assert budget["min_samples"] >= 3
    expected = {
        (100_000, cols, file_format, profile)
        for cols in (20, 100)
        for file_format in perf14.FORMATS
        for profile in perf14.PRECISIONS
    }
    actual = {
        (
            cell["case"]["rows"],
            cell["case"]["cols"],
            cell["case"]["format"],
            cell["case"]["precision"],
        )
        for cell in budget["cells"]
    }
    assert actual == expected
    assert len(budget["cells"]) == len(expected)
    for cell in budget["cells"]:
        assert cell["case"]["workflow"] == "light"
        assert cell["case"]["cache_state"] == "miss"
        assert cell["case"]["backend"] == "auto-core"


def test_v1_ci_keeps_installed_wheel_cost_and_default_workflow_gates() -> None:
    """A resident benchmark must not silently replace public-boundary evidence."""
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/v1_contract_validation.yml").read_text()
    for required in (
        'pip wheel --no-deps . --wheel-dir "$RUNNER_TEMP/gafime-ingest-wheelhouse"',
        'pip install --no-deps "${core_wheels[0]}"',
        "source_sha=$(git rev-parse HEAD)",
        '--source-sha candidate="$source_sha"',
        "--shape 100000x20 --shape 100000x100 --format csv,parquet,ipc",
        "--precision fp32,mixed,fp64 --workflow light --cache-state miss",
        "tests/release_measure/perf_14_ingest_budget.py",
        "--budget tests/release_measure/ingest_cost_budget.json",
        "--workflow default --cache-state miss,hit",
        "Preserve public ingestion evidence even on failure",
        "name: public-ingestion-evidence",
    ):
        assert required in workflow, required


def test_proc_peak_uses_exec_image_high_water_mark() -> None:
    assert perf14.linux_memory(
        "Name:\tpython\nVmHWM:\t12000 kB\nVmRSS:\t8000 kB\n"
    ) == {"VmHWM_kib": 12000, "VmRSS_kib": 8000}


@pytest.mark.parametrize(
    "text",
    [
        "VmRSS:\t8 kB",
        "VmRSS:\t8 bytes\nVmHWM:\t10 kB",
        "VmRSS:\t-1 kB\nVmHWM:\t10 kB",
        "VmRSS:\t11 kB\nVmHWM:\t10 kB",
    ],
)
def test_proc_memory_rejects_missing_malformed_and_contradictory_values(
    text: str,
) -> None:
    with pytest.raises(ValueError):
        perf14.linux_memory(text)


def test_snapshot_preserves_float_bits_and_significance_without_transport_claims() -> (
    None
):
    @dataclass
    class Item:
        candidate_id: str
        value: float

    report = SimpleNamespace(
        feature_names=["f0"],
        interactions=[Item("id", -0.0)],
        stability=[Item("id", 0.125)],
        permutations=[Item("id", float("nan"))],
        decision=SimpleNamespace(signal_detected=True),
        warnings=["cap"],
        backend=None,
    )

    # Real decisions are dataclasses; use one here to exercise the same route.
    @dataclass
    class Decision:
        signal_detected: bool
        message: str

    report.decision = Decision(True, "Arrow-specific message")
    first = perf14.report_snapshot(report)
    assert first["numeric"]["interactions"][0]["value"] == {
        "float64_le_hex": "0000000000000080"
    }
    assert first["numeric"]["stability"]
    assert first["numeric"]["permutations"]
    assert first["scope"]["independent_numerical_oracle"] is False
    json.dumps(first, allow_nan=False)
    report.warnings = ["other transport warning"]
    report.decision = Decision(True, "ordinary report")
    assert perf14.report_snapshot(report)["numeric_sha256"] == first["numeric_sha256"]
    report.interactions = [Item("id", 0.0)]
    assert perf14.report_snapshot(report)["numeric_sha256"] != first["numeric_sha256"]


def test_snapshot_preserves_compact_decision_path_parameter_arrays() -> None:
    result = perf14.bits_snapshot(
        {"features": array("Q", [0, 7]), "thresholds": array("d", [-0.0, 1.25])}
    )
    assert result["features"] == {"array_typecode": "Q", "values": [0, 7]}
    assert result["thresholds"]["array_typecode"] == "d"
    assert result["thresholds"]["values"][0] == {"float64_le_hex": "0000000000000080"}
    json.dumps(result, allow_nan=False)


def _wheel(
    tmp_path: Path, *, version: str = "1.0.0rc2", native: bytes = b"native"
) -> tuple[Path, Path]:
    package = tmp_path / "installed" / "gafime"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"package")
    (package / "gafime_py.so").write_bytes(b"native")
    wheel = tmp_path / "core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "gafime.dist-info/METADATA", f"Name: gafime\nVersion: {version}\n"
        )
        archive.writestr("gafime/__init__.py", b"package")
        archive.writestr("gafime/gafime_py.so", native)
    return wheel, package


def test_runtime_identity_binds_both_python_and_native_wheel_members(
    tmp_path: Path,
) -> None:
    wheel, package = _wheel(tmp_path)
    identity = perf14.verify_wheel(wheel, "1.0.0rc2", package)
    assert identity["byte_identical_installed_package"] is True
    assert len(identity["verified_members"]) == 2
    assert len(identity["wheel"]["sha256"]) == 64
    (package / "__init__.py").write_bytes(b"source-shadow")
    with pytest.raises(ValueError, match="differs from frozen wheel"):
        perf14.verify_wheel(wheel, "1.0.0rc2", package)


def test_runtime_identity_rejects_native_mismatch_and_wrong_version(
    tmp_path: Path,
) -> None:
    wheel, package = _wheel(tmp_path, native=b"other-native")
    with pytest.raises(ValueError, match="differs"):
        perf14.verify_wheel(wheel, "1.0.0rc2", package)
    with pytest.raises(ValueError, match="version"):
        perf14.verify_wheel(wheel, "1.0.0rc3", package)


def test_runtime_identity_rejects_unsafe_archive_path(tmp_path: Path) -> None:
    wheel, package = _wheel(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("gafime/../outside.py", b"unrelated")
    with pytest.raises(ValueError, match="noncanonical"):
        perf14.verify_wheel(wheel, "1.0.0rc2", package)


def test_loaded_native_must_be_the_authenticated_wheel_member(tmp_path: Path) -> None:
    wheel, package = _wheel(tmp_path)
    authenticated = perf14.verify_wheel(wheel, "1.0.0rc2", package)
    perf14.verify_loaded_native(package / "gafime_py.so", authenticated)
    with pytest.raises(ValueError, match="not a byte-verified"):
        perf14.verify_loaded_native(tmp_path / "another-library.so", authenticated)


def test_workload_bounds_keep_representative_shapes_without_giant_default_significance() -> (
    None
):
    perf14.check_bounds(100_000, 20, "light")
    perf14.check_bounds(100_000, 100, "light")
    perf14.check_bounds(1_000_000, 20, "light")
    for workflow in ("configured", "time-series", "decision-path"):
        perf14.check_bounds(4096, 16, workflow)
        with pytest.raises(ValueError, match="bounded"):
            perf14.check_bounds(100_000, 20, workflow)
    perf14.check_bounds(256, 8, "default")
    with pytest.raises(ValueError, match="literal default"):
        perf14.check_bounds(100_000, 20, "default")
    with pytest.raises(ValueError, match="positive"):
        perf14.check_bounds(0, 8, "light")


def test_generated_families_are_not_falsely_labelled_resident_cache_hits() -> None:
    perf14.check_cache_states("light", ["miss", "hit", "disabled", "target-change"])
    for family in ("time-series", "decision-path"):
        perf14.check_cache_states(family, ["miss", "disabled"])
        with pytest.raises(ValueError, match="do not use"):
            perf14.check_cache_states(family, ["hit"])


def test_cache_observation_does_not_invent_a_hit_for_warmed_nonresident_calls() -> None:
    empty = {"available": True, "entries": []}
    assert (
        perf14.observed_cache_state("hit", empty, empty)["observed"]
        == "no_resident_state"
    )
    entry = {
        "entry_id": 1,
        "artifact_id": 2,
        "native_handle_id": 3,
        "target_digest": "old",
        "closed": False,
    }
    before = {"available": True, "entries": [entry]}
    assert (
        perf14.observed_cache_state("miss", empty, before)["observed"]
        == "new_resident_state"
    )
    assert (
        perf14.observed_cache_state("hit", before, before)["observed"]
        == "resident_identity_reuse"
    )
    after = {"available": True, "entries": [{**entry, "target_digest": "new"}]}
    assert (
        perf14.observed_cache_state("target-change", before, after)["observed"]
        == "resident_target_update"
    )
    assert (
        perf14.observed_cache_state("disabled", empty, empty)["observed"]
        == "disabled_no_resident_state"
    )


def test_cache_metadata_snapshot_retains_only_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    artifact = SimpleNamespace(native_handle=object(), _closed=False)
    entry = SimpleNamespace(artifact=artifact, target_digest=b"digest")
    adapter = types.ModuleType("gafime.v1_adapter")
    adapter._current_analyze_cache = lambda: {"key": entry}
    package = types.ModuleType("gafime")
    package.__path__ = []
    package.v1_adapter = adapter
    monkeypatch.setitem(sys.modules, "gafime", package)
    monkeypatch.setitem(sys.modules, "gafime.v1_adapter", adapter)
    observed = perf14.cache_identities()
    assert observed["entries"][0]["artifact_id"] == id(artifact)
    json.dumps(observed)


def test_failed_and_instrumented_samples_cannot_enter_performance_statistics() -> None:
    good = {
        "status": "passed",
        "result": {
            "variant": {"label": "candidate"},
            "case": {"route": "dataload"},
            "instrumented": False,
            "timing": {"wall_nanoseconds": 100},
            "memory": {"after_call_before_snapshot": {"VmHWM_kib": 200}},
        },
    }
    instrumented = json.loads(json.dumps(good))
    instrumented["result"]["instrumented"] = True
    instrumented["result"]["timing"]["wall_nanoseconds"] = 999999
    summary = perf14.summaries([good, instrumented, {"status": "failed"}])
    assert summary[0]["samples"] == 1
    assert summary[0]["median_wall_nanoseconds"] == 100
    assert summary[0]["median_peak_rss_kib"] == 200


def _parity_pair(variant: str, value: str) -> tuple[list[dict], list[tuple]]:
    records, jobs = [], []
    for route in ("direct", "dataload"):
        case = {"route": route, "precision": "mixed", "cache_state": "miss"}
        snapshot = {"numeric": {"metric_bits": value}, "numeric_sha256": value}
        records.append(
            {
                "status": "passed",
                "variant": variant,
                "case": case,
                "repeat": 0,
                "result": {"instrumented": False, "snapshot": snapshot},
            }
        )
        jobs.append(({"label": variant}, case, 0, False))
    return records, jobs


def test_same_variant_changed_metric_bits_fail_public_parity_gate() -> None:
    records, jobs = _parity_pair("candidate", "3ff0000000000000")
    records[1]["result"]["snapshot"] = {
        "numeric": {"metric_bits": "4000000000000000"},
        "numeric_sha256": "different",
    }
    report = perf14.public_numeric_parity(records, jobs)
    assert report["status"] == "failed"
    assert report["numeric_mismatch_count"] == 1


def test_missing_or_duplicate_public_route_is_not_passing_parity() -> None:
    records, jobs = _parity_pair("candidate", "same")
    for incomplete in (records[:1], records + [records[1]]):
        report = perf14.public_numeric_parity(incomplete, jobs)
        assert report["status"] == "failed"
        assert report["missing_or_duplicate_count"] == 1


def test_parity_gate_is_within_each_variant_not_cross_release_equality() -> None:
    baseline, baseline_jobs = _parity_pair("rc2", "old-correctness-result")
    candidate, candidate_jobs = _parity_pair("candidate", "corrected-result")
    report = perf14.public_numeric_parity(
        baseline + candidate, baseline_jobs + candidate_jobs
    )
    assert report["status"] == "passed"
    assert len(report["comparisons"]) == 2


@pytest.mark.parametrize("failure", ["exit", "timeout", "bad-json"])
def test_worker_failure_preserves_both_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                "command", 1, output=b"partial-output", stderr=b"partial-error"
            )
        return subprocess.CompletedProcess(
            "command", 3 if failure == "exit" else 0, "not-json", "error-detail"
        )

    monkeypatch.setattr(perf14.subprocess, "run", run)
    result = perf14.run_child(
        "python", "--worker", tmp_path / "spec.json", tmp_path / "sample", 1
    )
    assert result["status"] == "failed"
    assert (tmp_path / "sample.stdout.log").read_text()
    assert (tmp_path / "sample.stderr.log").read_text()


def test_successful_worker_requires_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        perf14.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess("command", 0, '{"value":1}', ""),
    )
    assert perf14.run_child(
        "python", "--worker", tmp_path / "spec", tmp_path / "sample", 1
    ) == {"status": "passed", "result": {"value": 1}}


def test_valid_json_with_wrong_structure_is_not_a_passing_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        perf14.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess("command", 0, "null", ""),
    )
    result = perf14.run_child(
        "python", "--worker", tmp_path / "spec", tmp_path / "sample", 1
    )
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_worker_structure"


def test_invalid_assignment_or_duplicate_choice_fails_closed() -> None:
    for values in (
        ["candidate"],
        ["=python"],
        ["candidate="],
        ["candidate=a", "candidate=b"],
    ):
        with pytest.raises(ValueError):
            perf14.assignments(values)
    with pytest.raises(ValueError):
        perf14.choices("csv,csv", perf14.FORMATS)


def test_interpreter_keeps_virtual_environment_entrypoint_symlink(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "venv" / "bin" / "python"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(sys.executable)
    assert perf14.interpreter_path(str(entrypoint)) == str(entrypoint)
    assert perf14.interpreter_path(str(entrypoint)) != str(entrypoint.resolve())


def test_driver_rejects_unbound_wheel_and_harness_before_generating_files(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        python=["candidate=python"],
        wheel=[],
        expected_version=["candidate=1.0.0rc3"],
        source_sha=[],
        harness_sha=None,
    )
    with pytest.raises(ValueError, match="matching wheel"):
        perf14.driver(args)
    args.wheel = ["candidate=wheel.whl"]
    args.harness_sha = "0" * 64
    with pytest.raises(ValueError, match="harness source digest"):
        perf14.driver(args)
