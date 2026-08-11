"""Focused tests for lane-isolated native GPU A/B orchestration."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


_SCRIPT = Path(__file__).with_name("run_gpu_native_ab.py")
_SPEC = importlib.util.spec_from_file_location("gafime_run_gpu_native_ab", _SCRIPT)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _variant(
    tmp_path: Path,
    name: str,
    helper: Path,
    *,
    lane_helpers: dict[str, Path] | None = None,
) -> runner.Variant:
    source = tmp_path / f"{name}-source"
    source.mkdir()
    payload = tmp_path / f"{name}.payload"
    wheel = tmp_path / f"{name}.whl"
    evidence = tmp_path / f"{name}-canonical.json"
    payload.write_bytes(f"payload-{name}".encode())
    wheel.write_bytes(f"wheel-{name}".encode())
    evidence.write_text("{}", encoding="utf-8")
    return runner.Variant(
        name=name,
        helper=helper,
        payload=payload,
        wheel=wheel,
        source_root=source,
        canonical_evidence=evidence,
        lane_helpers=lane_helpers or {},
    )


def _config(
    tmp_path: Path,
    baseline: runner.Variant,
    candidate: runner.Variant,
    *,
    backend: str = "rocm",
) -> runner.RunnerConfig:
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    available = runner._affinity()
    assert available
    return runner.RunnerConfig(
        backend=backend,
        workload="all-metrics",
        input_policies=("common-f64", "native"),
        baseline=baseline,
        candidate=candidate,
        harness_source_root=harness,
        output_dir=tmp_path / "out",
        loop_plan_script=Path(__file__).with_name("native_loop_plan.py"),
        rows=256,
        features=8,
        candidates=8,
        arity=1,
        mi_bins=32,
        top_k=2,
        warmups=10,
        repeats=30,
        order_repetitions=30,
        device=0,
        affinity=(available[0],),
    )


def test_dry_run_plans_exact_matrix_and_requires_lane_binaries(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "helper"
    baseline = _variant(
        tmp_path,
        "baseline",
        helper,
        lane_helpers={
            "canonical_payload_api": tmp_path / "baseline-canonical",
            "supplemental_host_phase": tmp_path / "baseline-host",
        },
    )
    candidate = _variant(
        tmp_path,
        "candidate",
        helper,
        lane_helpers={
            "canonical_payload_api": tmp_path / "candidate-canonical",
            "supplemental_host_phase": tmp_path / "candidate-host",
        },
    )
    config = _config(tmp_path, baseline, candidate)

    summary = runner.run(config, dry_run=True)

    assert summary["calibration_process_count"] == 12
    assert summary["recorded_process_count"] == 24
    assert summary["expected_recorded_process_count"] == 24
    assert len(summary["recorded_commands"]) == 24
    assert summary["publication"] == {"performed": False, "actions": []}
    assert all(
        command[command.index("--evidence-lane") + 1] in runner.LANES
        for command in summary["recorded_commands"]
    )
    for command in summary["recorded_commands"]:
        variant_name = command[command.index("--variant") + 1]
        expected = baseline if variant_name == "baseline" else candidate
        assert command[command.index("--canonical-evidence") + 1] == str(
            expected.canonical_evidence
        )

    bare_baseline = _variant(tmp_path, "bare-baseline", helper)
    bare_candidate = _variant(tmp_path, "bare-candidate", helper)
    with pytest.raises(runner.RunnerError, match="lane-specific helper"):
        runner.run(
            _config(tmp_path, bare_baseline, bare_candidate, backend="rocm"),
            dry_run=True,
        )


def _write_fake_helper(path: Path, marker: str = "") -> None:
    script = r"""#!/usr/bin/env python3
import json
import hashlib
import os
from pathlib import Path
import sys

args = sys.argv[1:]
def value(name):
    index = args.index(name)
    return args[index + 1]

output = Path(value("--json"))
backend = "rocm"
lane = value("--evidence-lane")
policy = value("--input-policy")
variant = value("--variant")
workload = value("--workload")
calibration = "--calibration-only" in args
with Path(os.environ["FAKE_HELPER_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "pid": os.getpid(),
        "variant": variant,
        "lane": lane,
        "policy": policy,
        "calibration": calibration,
        "payload": os.environ.get("GAFIME_ROCM_V1_LIB"),
    }) + "\n")

if calibration:
    source_commit = {"baseline": "a" * 40, "candidate": "b" * 40}[variant]
    payload = {
        "schema": "gafime.native-loop-calibration.v1",
        "status": "calibration_only",
        "backend": backend,
        "device": {"ordinal": 0},
        "variant": variant,
        "scope_id": "rocm|all-metrics|256|8|8|1|32|2|" + policy + "|" + lane + "|rocm_events|0",
        "artifact_kind": "rocm_events",
        "evidence_lane": lane,
        "payload_load_status": "loaded_canonical_lane_only" if lane == "canonical_payload_api" else "not_loaded_by_lane_contract",
        "payload_absence_attested": lane != "canonical_payload_api",
        "execution_mode": "canonical_payload" if lane == "canonical_payload_api" else lane,
        "payload_loaded": lane == "canonical_payload_api",
        "payload_not_loaded": lane != "canonical_payload_api",
        "payload_execution_mode": "canonical_payload" if lane == "canonical_payload_api" else "payload_not_loaded",
        "source_commit": source_commit,
        "product_source_commit": source_commit,
        "harness_source_commit": "3" * 40,
        "workload": {
            "name": workload, "rows": 256, "features": 8, "candidates": 8,
            "arity": 1, "mi_bins": 32, "top_k": 2,
        },
        "input_policy": policy,
        "input_identity": {"dataset": "fake"},
        "environment": {
            "GAFIME_WHEEL_PATH": value("--wheel"),
            "GAFIME_ROCM_V1_LIB": os.environ["GAFIME_ROCM_V1_LIB"],
            "GAFIME_NATIVE_RUNNER_INVOCATION_ID": os.environ["GAFIME_NATIVE_RUNNER_INVOCATION_ID"],
            "GAFIME_NATIVE_RUNNER_PID": os.environ["GAFIME_NATIVE_RUNNER_PID"],
        },
        "runner_invocation_id": os.environ["GAFIME_NATIVE_RUNNER_INVOCATION_ID"],
        "runner_pid": int(os.environ["GAFIME_NATIVE_RUNNER_PID"]),
        "process_id": os.getpid(),
        "affinity": {"current_cpu": min(os.sched_getaffinity(0)), "allowed_cpus": sorted(os.sched_getaffinity(0))},
        "command_line": sys.argv,
        "provenance": {
            "benchmark_binary": {"path": sys.argv[0], "sha256": hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest()},
            "payload": {"path": os.environ["GAFIME_ROCM_V1_LIB"], "sha256": hashlib.sha256(Path(os.environ["GAFIME_ROCM_V1_LIB"]).read_bytes()).hexdigest()},
            "wheel": {"path": value("--wheel"), "sha256": hashlib.sha256(Path(value("--wheel")).read_bytes()).hexdigest()},
        },
        "source_root": str(Path(value("--source-root")).resolve()),
        "product_source_root": str(Path(value("--source-root")).resolve()),
        "harness_source_root": str(Path(value("--harness-source-root")).resolve()),
        "source_tree_state": {"status": "clean", "entry_count": 0, "entries": []},
        "product_source_tree_state": {"status": "clean", "entry_count": 0, "entries": []},
        "harness_source_tree_state": {"status": "clean", "entry_count": 0, "entries": []},
        "source_blob": {
            "path": str(Path(value("--source-root")).resolve()),
            "relative_path": "tests/gpu/fake_helper.py",
            "source_sha256": "d" * 64,
            "current_git_blob": "e" * 40,
            "head_git_blob": "e" * 40,
        },
        "harness_source_blob": {
            "path": str(Path(value("--harness-source-root")).resolve()),
            "relative_path": "tests/release_measure/fake_helper.py",
            "source_sha256": "f" * 64,
            "current_git_blob": "1" * 40,
            "head_git_blob": "1" * 40,
        },
        "git": {
            "path": "/usr/bin/git",
            "sha256": hashlib.sha256(Path("/usr/bin/git").read_bytes()).hexdigest(),
            "version": "git version test",
            "git_dir": str(Path(value("--source-root")).resolve() / ".git"),
            "git_common_dir": str(Path(value("--source-root")).resolve() / ".git"),
            "removed_environment": [key for key in os.environ if key.startswith("GIT_")],
        },
        "entry_count": 1,
        "entries": [{"key": "payload/fp32/execute/pearson", "loop_count": 2}],
    }
else:
    sequence = value("--variant-sequence").split(",")
    plan = value("--loop-plan")
    payload = {
        "schema": "gafime.rocm.native_timing.v2",
        "status": "pass",
        "backend": backend,
        "artifact_kind": "rocm_events",
        "evidence_lane": lane,
        "payload_load_status": "loaded_canonical_lane_only" if lane == "canonical_payload_api" else "not_loaded_by_lane_contract",
        "execution_mode": "canonical_payload" if lane == "canonical_payload_api" else lane,
        "payload_loaded": lane == "canonical_payload_api",
        "payload_not_loaded": lane != "canonical_payload_api",
        "payload_execution_mode": "canonical_payload" if lane == "canonical_payload_api" else "payload_not_loaded",
        "lane_isolation": "fresh_helper_process_per_variant_trial_and_lane",
        "process_isolation": "fresh_helper_process_per_variant_trial",
        "input_policy": policy,
        "variant": variant,
        "ab_block": int(value("--ab-block")),
        "variant_sequence": sequence,
        "profiles": ["fp32", "mixed", "fp64"],
        "source_commit": {"baseline": "a" * 40, "candidate": "b" * 40}[variant],
        "product_source_commit": {"baseline": "a" * 40, "candidate": "b" * 40}[variant],
        "harness_source_commit": "3" * 40,
        "source_root": str(Path(value("--source-root")).resolve()),
        "product_source_root": str(Path(value("--source-root")).resolve()),
        "harness_source_root": str(Path(value("--harness-source-root")).resolve()),
        "workload": {
            "name": workload, "rows": 256, "features": 8, "candidates": 8,
            "arity": 1, "mi_bins": 32, "top_k": 2,
        },
        "device": {"id": 0, "name": "fake"},
        "scope_id": json.loads(Path(plan).read_text(encoding="utf-8"))["scope"]["scope_id"],
        "input_identity": {"dataset": "fake"},
        "environment": {
            "GAFIME_WHEEL_PATH": value("--wheel"),
            "GAFIME_ROCM_V1_LIB": os.environ["GAFIME_ROCM_V1_LIB"],
            "GAFIME_NATIVE_RUNNER_INVOCATION_ID": os.environ["GAFIME_NATIVE_RUNNER_INVOCATION_ID"],
            "GAFIME_NATIVE_RUNNER_PID": os.environ["GAFIME_NATIVE_RUNNER_PID"],
        },
        "runner_invocation_id": os.environ["GAFIME_NATIVE_RUNNER_INVOCATION_ID"],
        "runner_pid": int(os.environ["GAFIME_NATIVE_RUNNER_PID"]),
        "process_id": os.getpid(),
        "command_line": sys.argv,
        "loop_plan": {
            "mode": "immutable",
            "path": plan,
            "relative_path": Path(os.path.relpath(Path(plan).resolve(), start=output.resolve().parent)).as_posix(),
            "semantic_sha256": json.loads(Path(plan).read_text(encoding="utf-8"))["plan_sha256"],
            "file_sha256": hashlib.sha256(Path(plan).read_bytes()).hexdigest(),
        },
        "calibration_prepass": {
            "performed": True,
            "uses_shared_calibration_cache": True,
        },
        "self_checks": {
            "payload_absence_attested": lane != "canonical_payload_api",
        },
        "provenance": {
            "benchmark_binary": {"path": sys.argv[0], "sha256": hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest()},
            "payload": {"path": os.environ["GAFIME_ROCM_V1_LIB"], "sha256": __import__("hashlib").sha256(Path(os.environ["GAFIME_ROCM_V1_LIB"]).read_bytes()).hexdigest()},
            "wheel": {"path": value("--wheel"), "sha256": __import__("hashlib").sha256(Path(value("--wheel")).read_bytes()).hexdigest()},
        },
        "affinity": {"allowed_cpus": sorted(os.sched_getaffinity(0))},
        "records": [{
            "evidence_lane": lane,
            "samples_us": [1.0],
            "unique_output": str(output),
        }],
    }
output.write_text(json.dumps(payload), encoding="utf-8")
"""
    path.write_text(script + f"\n# lane-helper-marker: {marker}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_fake_helpers_execute_every_cell_in_fresh_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "fake_rocm_helper.py"
    canonical = tmp_path / "fake_rocm_canonical_helper.py"
    host = tmp_path / "fake_rocm_host_helper.py"
    candidate_helper = tmp_path / "fake_rocm_candidate_helper.py"
    candidate_canonical = tmp_path / "fake_rocm_candidate_canonical_helper.py"
    candidate_host = tmp_path / "fake_rocm_candidate_host_helper.py"
    log = tmp_path / "helper.log"
    _write_fake_helper(helper, "baseline-direct")
    _write_fake_helper(canonical, "baseline-canonical")
    _write_fake_helper(host, "baseline-host")
    _write_fake_helper(candidate_helper, "candidate-direct")
    _write_fake_helper(candidate_canonical, "candidate-canonical")
    _write_fake_helper(candidate_host, "candidate-host")
    monkeypatch.setenv("FAKE_HELPER_LOG", str(log))

    baseline = _variant(
        tmp_path,
        "baseline",
        helper,
        lane_helpers={
            "canonical_payload_api": canonical,
            "supplemental_host_phase": host,
        },
    )
    candidate = _variant(
        tmp_path,
        "candidate",
        candidate_helper,
        lane_helpers={
            "canonical_payload_api": candidate_canonical,
            "supplemental_host_phase": candidate_host,
        },
    )
    config = _config(tmp_path, baseline, candidate)

    def fake_source_identity(
        root: Path, label: str, *, dry_run: bool
    ) -> dict[str, object]:
        commit = "a" * 40 if "baseline" in str(root) else "b" * 40
        if "harness" in label:
            commit = "3" * 40
        return {
            "root": str(root),
            "status": "clean",
            "commit": commit,
            "tree": "4" * 40,
        }

    monkeypatch.setattr(runner, "_source_identity", fake_source_identity)
    summary = runner.run(config)

    assert summary["status"] == "pass"
    assert (config.output_dir / "calibration").is_dir()
    assert (config.output_dir / "artifacts").is_dir()
    assert summary["calibration_process_count"] == 12
    assert summary["recorded_process_count"] == 24
    assert len(summary["artifacts"]) == 24
    assert len(log.read_text(encoding="utf-8").splitlines()) == 36
    assert len({item["sha256"] for item in summary["artifacts"]}) == 24
    assert all(
        item["schedule"]["evidence_lane"] in runner.LANES
        for item in summary["artifacts"]
    )
    for manifest_path in summary["manifests"].values():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert manifest["process_isolation"] == "fresh_helper_process_per_variant_trial"
        assert len(manifest["artifacts"]) == 12

    observations = [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert {
        item["payload"] for item in observations if item["variant"] == "baseline"
    } == {str(baseline.payload)}
    assert {
        item["payload"] for item in observations if item["variant"] == "candidate"
    } == {str(candidate.payload)}


def test_rocm_lane_helpers_must_be_byte_distinct(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    canonical = tmp_path / "canonical"
    host = tmp_path / "host"
    for path in (direct, canonical, host):
        path.write_bytes(b"same-helper")
    variant = _variant(
        tmp_path,
        "baseline",
        direct,
        lane_helpers={
            "canonical_payload_api": canonical,
            "supplemental_host_phase": host,
        },
    )
    with pytest.raises(runner.RunnerError, match="byte-distinct"):
        variant.helper_for("rocm", "canonical_payload_api", dry_run=False)


def test_invalid_matrix_is_rejected_before_child_processes(tmp_path: Path) -> None:
    helper = tmp_path / "helper"
    baseline = _variant(tmp_path, "baseline", helper)
    candidate = _variant(tmp_path, "candidate", helper)
    config = _config(tmp_path, baseline, candidate)
    invalid = runner.RunnerConfig(**{**config.__dict__, "repeats": 2})
    with pytest.raises(runner.RunnerError, match="repeats"):
        runner.run(invalid, dry_run=True)


def test_strict_machine_json_rejects_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="non-finite"):
        runner._strict_load(path)


def test_git_identity_ignores_path_wrapper_and_git_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PATH ``git`` shim must not be able to forge calibration provenance."""

    source = tmp_path / "source"
    source.mkdir()
    git = runner._trusted_git_executable()
    clean_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    clean_environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "HOME": str(tmp_path),
        }
    )
    subprocess.run(
        [str(git), "init", "-q", str(source)], check=True, env=clean_environment
    )
    tracked = source / "tracked.txt"
    tracked.write_text("trusted git\n", encoding="utf-8")
    subprocess.run(
        [str(git), "-C", str(source), "add", "tracked.txt"],
        check=True,
        env=clean_environment,
    )
    subprocess.run(
        [
            str(git),
            "-C",
            str(source),
            "-c",
            "user.name=GAFIME test",
            "-c",
            "user.email=gafime-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
        env=clean_environment,
    )
    expected_commit = subprocess.run(
        [str(git), "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    ).stdout.strip()

    marker = tmp_path / "wrapper-used"
    wrapper = tmp_path / "git"
    wrapper.write_text(
        f"#!/bin/sh\nprintf forged > {marker}\nprintf '%s\\n' {'f' * 40}\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "forged-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "forged-work-tree"))

    identity = runner._source_identity(source, "test source", dry_run=False)

    assert identity["commit"] == expected_commit
    assert Path(str(identity["git"]["path"])).resolve() == git
    assert "GIT_DIR" in identity["git"]["removed_environment"]
    assert "GIT_WORK_TREE" in identity["git"]["removed_environment"]
    assert not marker.exists()
