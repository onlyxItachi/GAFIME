"""Adversarial unit tests for the perf13 native-evidence gate."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


_SCRIPT = Path(__file__).with_name("perf_13_precision_profiles.py")
_SPEC = importlib.util.spec_from_file_location("gafime_perf13", _SCRIPT)
assert _SPEC and _SPEC.loader
perf13 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = perf13
_SPEC.loader.exec_module(perf13)

_PLAN_SCRIPT = Path(__file__).with_name("native_loop_plan.py")
_PLAN_SPEC = importlib.util.spec_from_file_location(
    "gafime_native_loop_plan_perf13_test", _PLAN_SCRIPT
)
assert _PLAN_SPEC and _PLAN_SPEC.loader
native_loop_plan = importlib.util.module_from_spec(_PLAN_SPEC)
sys.modules[_PLAN_SPEC.name] = native_loop_plan
_PLAN_SPEC.loader.exec_module(native_loop_plan)


def _identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _native_plan_case(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    helper = tmp_path / "helper"
    helper.write_bytes(b"helper")
    git_path = (
        Path("/usr/bin/git") if Path("/usr/bin/git").is_file() else Path("/bin/git")
    )
    git_digest = hashlib.sha256(git_path.read_bytes()).hexdigest()
    clean_tree = {"status": "clean", "entry_count": 0, "entries": []}
    common = {
        "schema": native_loop_plan.CALIBRATION_SCHEMA,
        "status": "calibration_only",
        "backend": "cuda",
        "artifact_kind": "cuda_events",
        "evidence_lane": "supplemental_internal_kernel",
        "device": {"name": "test-gpu"},
        "scope_id": "cuda|native-plan-test",
        "workload": {
            "name": "native-plan-test",
            "rows": 256,
            "features": 8,
            "candidates": 8,
            "arity": 1,
            "mi_bins": 32,
            "top_k": 2,
        },
        "input_policy": "common-f64",
        "input_identity": {"matrix_sha256": "a" * 64},
        "harness_source_commit": "c" * 40,
        "command_line": [str(helper), "--calibration-only"],
        "provenance": {
            "benchmark_binary": {"path": str(helper), "sha256": "b" * 64},
            "payload": None,
            "wheel": None,
        },
        "source_root": str(tmp_path / "product"),
        "product_source_root": str(tmp_path / "product"),
        "harness_source_root": str(tmp_path / "harness"),
        "source_tree_state": clean_tree,
        "product_source_tree_state": clean_tree,
        "harness_source_tree_state": clean_tree,
        "source_blob": {
            "path": str(tmp_path / "product"),
            "relative_path": "tests/gpu/helper.cpp",
            "source_sha256": "d" * 64,
            "current_git_blob": "e" * 40,
            "head_git_blob": "e" * 40,
        },
        "harness_source_blob": {
            "path": str(tmp_path / "harness"),
            "relative_path": "tests/gpu/helper.cpp",
            "source_sha256": "f" * 64,
            "current_git_blob": "1" * 40,
            "head_git_blob": "1" * 40,
        },
        "git": {
            "path": str(git_path.resolve()),
            "sha256": git_digest,
            "version": "git version test",
            "git_dir": str(tmp_path / "product" / ".git"),
            "git_common_dir": str(tmp_path / "product" / ".git"),
            "removed_environment": ["GIT_DIR"],
        },
    }
    calibration_paths: list[Path] = []
    for variant, commit, counts in (
        ("baseline", "1" * 40, {"cell/a": 2, "cell/b": 4}),
        ("candidate", "2" * 40, {"cell/a": 3, "cell/b": 1}),
    ):
        path = tmp_path / f"{variant}.json"
        payload = {
            **common,
            "variant": variant,
            "source_commit": commit,
            "product_source_commit": commit,
            "entries": [
                {"key": key, "loop_count": count} for key, count in counts.items()
            ],
            "entry_count": len(counts),
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        calibration_paths.append(path)
    # Calibration artifacts and the immutable plan are siblings so their
    # portable bindings never need to escape the plan directory.
    plan_path = tmp_path / "plan.json"
    plan = native_loop_plan.make_plan(calibration_paths, plan_path=plan_path)
    native_loop_plan.write_plan(plan_path, plan)
    artifact_path = tmp_path / "artifacts" / "artifact.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text("{}", encoding="utf-8")
    loop_plan = {
        "mode": "immutable",
        "path": str(plan_path.resolve()),
        "relative_path": Path(
            os.path.relpath(plan_path.resolve(), start=artifact_path.parent.resolve())
        ).as_posix(),
        "semantic_sha256": plan["plan_sha256"],
        "file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    }
    payload = {
        "lane_isolation": "fresh_helper_process_per_variant_trial_and_lane",
        "evidence_lane": "supplemental_internal_kernel",
        "artifact_kind": "cuda_events",
        "scope_id": "cuda|native-plan-test",
        "input_policy": "common-f64",
        "loop_plan": loop_plan,
    }
    records = [
        {"calibration_key": item["key"], "loop_count_per_sample": item["loop_count"]}
        for item in plan["entries"]
    ]
    return plan_path, payload, records


def test_native_loop_plan_reauthenticates_semantic_and_file_digests(
    tmp_path: Path,
) -> None:
    plan_path, payload, records = _native_plan_case(tmp_path)
    artifact_path = tmp_path / "artifacts" / "artifact.json"
    assert (
        perf13._native_loop_plan_failures(artifact_path, payload, "cuda", records) == []
    )

    tampered = json.loads(json.dumps(payload))
    tampered["loop_plan"]["file_sha256"] = "0" * 64
    failures = perf13._native_loop_plan_failures(
        artifact_path, tampered, "cuda", records
    )
    assert "native_loop_plan_file_sha256_mismatch" in failures

    tampered = json.loads(json.dumps(payload))
    tampered["loop_plan"]["semantic_sha256"] = "f" * 64
    failures = perf13._native_loop_plan_failures(
        artifact_path, tampered, "cuda", records
    )
    assert "native_loop_plan_semantic_sha256_mismatch" in failures


def test_native_loop_plan_reauthenticates_calibration_files_and_binding_metadata(
    tmp_path: Path,
) -> None:
    plan_path, payload, records = _native_plan_case(tmp_path)
    artifact_path = tmp_path / "artifacts" / "artifact.json"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(baseline.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    failures = perf13._native_loop_plan_failures(
        artifact_path, payload, "cuda", records
    )
    assert "native_loop_plan_calibration_baseline_file_sha256_mismatch" in failures

    # Restore a valid calibration file, then alter a binding field while
    # retaining the old file hash.  The plan must not trust metadata copied
    # into its binding object over the authenticated payload.
    baseline.write_text(
        json.dumps(
            {
                **json.loads(baseline.read_text(encoding="utf-8")),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    # Rebuild the fixture so its binding/file hashes are coherent again.
    plan_path, payload, records = _native_plan_case(tmp_path / "rebuilt")
    artifact_path = plan_path.parents[1] / "artifacts" / "artifact.json"
    tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered_plan["bindings"][0]["input_policy"] = "native"
    tampered_plan["plan_sha256"] = native_loop_plan._plan_digest(tampered_plan)
    native_loop_plan.write_plan(plan_path, tampered_plan)
    payload["loop_plan"]["semantic_sha256"] = tampered_plan["plan_sha256"]
    payload["loop_plan"]["file_sha256"] = hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    failures = perf13._native_loop_plan_failures(
        artifact_path, payload, "cuda", records
    )
    assert (
        "native_loop_plan_calibration_baseline_binding_input_policy_mismatch"
        in failures
    )


def test_native_loop_plan_counts_must_be_derived_from_both_calibrations(
    tmp_path: Path,
) -> None:
    plan_path, payload, records = _native_plan_case(tmp_path)
    artifact_path = tmp_path / "artifacts" / "artifact.json"
    tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered_plan["entries"][0]["loop_count"] += 1
    tampered_plan["plan_sha256"] = native_loop_plan._plan_digest(tampered_plan)
    native_loop_plan.write_plan(plan_path, tampered_plan)
    payload["loop_plan"]["semantic_sha256"] = tampered_plan["plan_sha256"]
    payload["loop_plan"]["file_sha256"] = hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    failures = perf13._native_loop_plan_failures(
        artifact_path, payload, "cuda", records
    )
    assert "native_loop_plan_entry_not_derived_from_calibration" in failures


def test_native_loop_plan_paths_cannot_escape_explicit_evidence_root(
    tmp_path: Path,
) -> None:
    plan_path, payload, records = _native_plan_case(tmp_path)
    artifact_path = tmp_path / "artifacts" / "artifact.json"
    external_dir = tmp_path.parent / f"{tmp_path.name}-external"
    external_dir.mkdir()
    external_plan = external_dir / "plan.json"
    external_plan.write_bytes(plan_path.read_bytes())

    absolute_escape = json.loads(json.dumps(payload))
    absolute_escape["loop_plan"]["path"] = str(external_plan.resolve())
    failures = perf13._native_loop_plan_failures(
        artifact_path,
        absolute_escape,
        "cuda",
        records,
        evidence_root=tmp_path,
    )
    assert "native_loop_plan_path_outside_evidence_root" in failures

    traversal_escape = json.loads(json.dumps(payload))
    traversal_escape["loop_plan"]["path"] = Path(
        os.path.relpath(external_plan, start=artifact_path.parent)
    ).as_posix()
    failures = perf13._native_loop_plan_failures(
        artifact_path,
        traversal_escape,
        "cuda",
        records,
        evidence_root=tmp_path,
    )
    assert "native_loop_plan_path_outside_evidence_root" in failures

    external_calibration = external_dir / "baseline.json"
    external_calibration.write_bytes((tmp_path / "baseline.json").read_bytes())
    tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    baseline_binding = next(
        item for item in tampered_plan["bindings"] if item["variant"] == "baseline"
    )
    baseline_binding["path"] = str(external_calibration.resolve())
    tampered_plan["plan_sha256"] = native_loop_plan._plan_digest(tampered_plan)
    native_loop_plan.write_plan(plan_path, tampered_plan)
    payload["loop_plan"]["semantic_sha256"] = tampered_plan["plan_sha256"]
    payload["loop_plan"]["file_sha256"] = hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    failures = perf13._native_loop_plan_failures(
        artifact_path,
        payload,
        "cuda",
        records,
        evidence_root=tmp_path,
    )
    assert (
        "native_loop_plan_calibration_baseline_path_outside_evidence_root" in failures
    )


def test_independent_loop_plan_verifier_enforces_root_contract(tmp_path: Path) -> None:
    cases = (
        (
            "version",
            "native_loop_plan_version_mismatch",
            lambda plan: plan.__setitem__("version", 2),
        ),
        (
            "source-count",
            "native_loop_plan_source_count_must_be_two",
            lambda plan: plan.__setitem__("source_count", 1),
        ),
        (
            "variants",
            "native_loop_plan_baseline_and_candidate_variants_required",
            lambda plan: plan.__setitem__("variants", ["candidate", "baseline"]),
        ),
        (
            "distinct-commits",
            "native_loop_plan_distinct_full_source_commits_required",
            lambda plan: plan.__setitem__("source_commits", ["1" * 40, "1" * 40]),
        ),
        (
            "root-binding",
            "native_loop_plan_binding_source_commits_root_mismatch",
            lambda plan: plan.__setitem__("source_commits", ["3" * 40, "4" * 40]),
        ),
    )
    for name, expected, mutate in cases:
        case_root = tmp_path / name
        plan_path, payload, records = _native_plan_case(case_root)
        artifact_path = case_root / "artifacts" / "artifact.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        mutate(plan)
        plan["plan_sha256"] = native_loop_plan._plan_digest(plan)
        plan_path.write_text(
            json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        payload["loop_plan"]["semantic_sha256"] = plan["plan_sha256"]
        payload["loop_plan"]["file_sha256"] = hashlib.sha256(
            plan_path.read_bytes()
        ).hexdigest()
        failures = perf13._native_loop_plan_failures(
            artifact_path,
            payload,
            "cuda",
            records,
            evidence_root=case_root,
        )
        assert expected in failures


def test_public_git_provenance_rejects_path_and_git_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    git = perf13._trusted_git_executable()
    assert git is not None
    environment, _ = perf13._git_environment()
    subprocess.run([str(git), "init", "-q", str(source)], check=True, env=environment)
    tracked = source / "tracked.txt"
    tracked.write_text("trusted\n", encoding="utf-8")
    subprocess.run(
        [str(git), "-C", str(source), "add", "tracked.txt"],
        check=True,
        env=environment,
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
        env=environment,
    )
    marker = tmp_path / "wrapper-used"
    wrapper = tmp_path / "git"
    wrapper.write_text(
        f"#!/bin/sh\nprintf forged > {marker}\nprintf '%s\\n' {'f' * 40}\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    redirected = tmp_path / "redirected.git"
    redirected.mkdir()
    config = tmp_path / "hostile.gitconfig"
    config.write_text("[include]\npath = /definitely/not/trusted\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GIT_DIR", str(redirected))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path))

    provenance = perf13._git_provenance(source, source_path=tracked)

    assert provenance["status"] == "trusted"
    assert provenance["git"]["path"] == str(git)
    assert not marker.exists()
    removed = set(provenance["git"]["removed_environment"])
    assert {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    } <= removed
    assert perf13._public_git_provenance_failures(provenance, label="source") == []

    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = perf13._git_provenance(source, source_path=tracked)
    assert dirty["status"] == "invalid"
    assert (
        dirty["source_blob"]["current_git_blob"]
        != dirty["source_blob"]["head_git_blob"]
    )


def test_native_schedule_rejects_manifest_payload_mismatch(tmp_path: Path) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=True, schedule_in_manifest_only=False
    )
    manifest = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["schedule"] = {"ab_block": 1}
    baseline_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = perf13._load_native_evidence_specs(
        [("baseline", str(baseline_manifest)), ("candidate", str(candidate_manifest))]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))
    assert any(
        failure.get("reason") == "artifact_schedule_payload_mismatch"
        for failure in readiness["failures"]
        if isinstance(failure, dict)
    )


def test_native_schedule_input_policy_must_exactly_match_payload(
    tmp_path: Path,
) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=True
    )
    manifest = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["schedule"]["input_policy"] = "native"
    baseline_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = perf13._load_native_evidence_specs(
        [("baseline", str(baseline_manifest)), ("candidate", str(candidate_manifest))]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))
    reasons = {
        failure.get("reason")
        for failure in readiness["failures"]
        if isinstance(failure, dict)
    }
    assert "artifact_schedule_payload_mismatch" in reasons
    assert "native_schedule_input_policy_payload_mismatch" in reasons


def test_native_ab_rejects_identical_product_identities_without_public_results(
    tmp_path: Path,
) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=True, candidate_commit="a" * 40
    )
    evidence = perf13._load_native_evidence_specs(
        [("baseline", str(baseline_manifest)), ("candidate", str(candidate_manifest))]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))
    assert any(
        failure.get("reason")
        == "native_baseline_candidate_product_identities_must_differ"
        for failure in readiness["failures"]
        if isinstance(failure, dict)
    )


def _native_harness_fields(
    source: Path, *, product_commit: str, runner: Path | None = None
) -> dict[str, object]:
    source_identity = _identity(source)
    fields: dict[str, object] = {
        "product_source_commit": product_commit,
        "product_source_tree_state": {"status": "clean", "entries": []},
        "harness_source_commit": "c" * 40,
        "harness_source_tree_state": {"status": "clean", "entries": []},
        "harness_source_blob": {
            "relative_path": source.name,
            "source_sha256": source_identity["sha256"],
            "current_git_blob": "d" * 40,
            "head_git_blob": "d" * 40,
        },
    }
    if runner is not None:
        runner_identity = _identity(runner)
        fields["harness_runner_blob"] = {
            "relative_path": runner.name,
            "source_sha256": runner_identity["sha256"],
            "current_git_blob": "e" * 40,
            "head_git_blob": "e" * 40,
        }
    return fields


def _runtime_dependencies() -> dict[str, object]:
    return {
        name: {
            "status": "observed",
            "version": "test",
            "record": {
                "path": f"/{name}.dist-info/RECORD",
                "size_bytes": 10,
                "sha256": ("a" if name == "numpy" else "b") * 64,
            },
        }
        for name in perf13.BENCHMARK_RUNTIME_DISTRIBUTIONS
    }


def _write_manifest(
    path: Path,
    artifact: Path,
    *,
    backend: str = "core",
    variant: str = "candidate",
    source_commit: str = "a" * 40,
    kind: str | None = None,
) -> None:
    if kind is None:
        kind = "core_microbenchmark" if backend == "core" else f"{backend}_events"
    path.write_text(
        json.dumps(
            {
                "schema": perf13.NATIVE_EVIDENCE_SCHEMA,
                "status": "validated",
                "arithmetic_claims_valid": True,
                "source_commit": source_commit,
                "artifacts": [
                    {
                        "variant": variant,
                        "backend": backend,
                        "kind": kind,
                        "path": str(artifact),
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )


def _core_artifact(
    tmp_path: Path,
    profiles: tuple[str, ...],
    *,
    source_commit: str = "a" * 40,
) -> Path:
    source = tmp_path / "source"
    runner = tmp_path / "runner.py"
    binary = tmp_path / "binary"
    wheel = tmp_path / "gafime.whl"
    source.write_bytes(b"source")
    runner.write_bytes(b"runner")
    binary.write_bytes(b"binary")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "gafime-1.0.0b2.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gafime\nVersion: 1.0.0b2\n",
        )
        archive.writestr("gafime/gafime_py.so", b"core-native")
    repeats = (
        perf13.CORE_BALANCED_SCHEDULE_CYCLES
        * perf13.CORE_PROFILE_ORDER_COUNT
        * perf13.CORE_METRIC_ROTATION_COUNT
    )
    profile_orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    measured_schedule = [
        {
            "block_index": block_index,
            "balanced_cycle": cycle,
            "order_index": order_index,
            "profile_order": list(profile_orders[order_index]),
            "metric_rotation": metric_rotation,
        }
        for block_index, (cycle, order_index, metric_rotation) in enumerate(
            (
                (cycle, order_index, metric_rotation)
                for cycle in range(perf13.CORE_BALANCED_SCHEDULE_CYCLES)
                for order_index in range(perf13.CORE_PROFILE_ORDER_COUNT)
                for metric_rotation in range(perf13.CORE_METRIC_ROTATION_COUNT)
            )
        )
    ]
    clean_status = (
        "no_order_effect_above_one_percent_with_95_percent_familywise_confidence"
    )
    position_contrasts = [
        {
            "positions": list(positions),
            "observed_signed_percent": 0.0,
            "corrected_bootstrap_ci_percent": [-0.1, 0.1],
            "status": clean_status,
        }
        for positions in ((0, 1), (0, 2), (1, 2))
    ]
    sensitivity_cells = [
        {
            "profile": profile,
            "metric": metric,
            "order_position_median_ns": [200_000_000.0] * 3,
            "max_order_position_spread_percent": 0.0,
            "position_contrasts": position_contrasts,
            "corrected_per_contrast_confidence_level": 1.0
            - 0.05 / perf13.CORE_ORDER_TOTAL_COMPARISONS,
            "balanced_cycle_cluster_count": perf13.CORE_BALANCED_SCHEDULE_CYCLES,
            "observations_per_position": repeats // 3,
            "status": clean_status,
        }
        for profile in profiles
        for metric in perf13.ALL_METRICS
    ]
    records = []
    for profile in profiles:
        for metric in perf13.ALL_METRICS:
            records.append(
                {
                    "profile": profile,
                    "operation": "metric_kernel",
                    "metric": metric,
                    "samples_ns": [200_000_000.0] * repeats,
                    "raw_samples_ns": [200_000_000] * repeats,
                    "loop_count_per_sample": 1,
                    "sample_region_target_ns": perf13.CORE_MIN_MEASURED_REGION_NS,
                    "sample_region_min_observed_ns": 200_000_000,
                    "sample_region_target_met": True,
                }
            )
    raw_order = [
        {
            "profile": profile,
            "metric": metric,
            "block_index": block["block_index"],
            "balanced_cycle": block["balanced_cycle"],
            "order_index": block["order_index"],
            "metric_rotation": block["metric_rotation"],
            "position": block["profile_order"].index(profile),
            "profile_order": block["profile_order"],
            "precondition_iterations": 10,
            "precondition_duration_ns": 100_000_000,
            "duration_ns": 200_000_000,
        }
        for block in measured_schedule
        for profile in profiles
        for metric in perf13.ALL_METRICS
    ]
    artifact = tmp_path / "core.json"
    artifact_payload = {
        "schema": "gafime.core-native-arithmetic.v2",
        "status": "pass",
        "backend": "core",
        "profiles": list(profiles),
        "source_commit": source_commit,
        "source_tree_state": {"status": "clean", "entries": []},
        "input_policy": "common-f64",
        "input_identity": {
            "matrix_sha256": "1" * 64,
            "target_sha256": "2" * 64,
            "feature_names_sha256": "3" * 64,
        },
        "warmups": 10,
        "repeats": repeats,
        "per_sample_untimed_same_cell_preconditions": 10,
        "per_sample_untimed_precondition_min_ns": 100_000_000,
        "target_region_ns": perf13.CORE_MIN_MEASURED_REGION_NS,
        "calibration_target_region_ns": 200_000_000,
        "calibration_policy": ("fixed_loop_count_per_cell_no_per_sample_rescaling"),
        "metric_rotations": list(range(perf13.CORE_METRIC_ROTATION_COUNT)),
        "balanced_schedule_cycles": perf13.CORE_BALANCED_SCHEDULE_CYCLES,
        "profile_order_metric_rotation_pair_repetitions": (
            perf13.CORE_BALANCED_SCHEDULE_CYCLES
        ),
        "measured_schedule": measured_schedule,
        "all_six_profile_orders_covered": True,
        "all_profile_order_metric_rotation_pairs_covered": True,
        "order_sensitivity": {
            "threshold_percent": 1.0,
            "maximum_spread_percent": 0.0,
            "confirmed_contamination_cells": 0,
            "inconclusive_cells": 0,
            "bootstrap_resamples": perf13.CORE_ORDER_BOOTSTRAP_RESAMPLES,
            "familywise_confidence_level": 0.95,
            "multiple_comparison_correction": (
                "bonferroni_two_sided_across_profile_metric_cells_and_position_pair_contrasts"
            ),
            "comparison_cells": perf13.CORE_ORDER_COMPARISON_CELLS,
            "position_pair_contrasts_per_cell": (
                perf13.CORE_ORDER_POSITION_PAIR_CONTRASTS
            ),
            "total_comparisons": perf13.CORE_ORDER_TOTAL_COMPARISONS,
            "corrected_per_contrast_confidence_level": 1.0
            - 0.05 / perf13.CORE_ORDER_TOTAL_COMPARISONS,
            "bootstrap_stratification": "whole_balanced_cycle_cluster",
            "status": clean_status,
            "cells": sensitivity_cells,
        },
        "sample_region_gate": {
            "minimum_required_ns": perf13.CORE_MIN_MEASURED_REGION_NS,
            "under_target_cells": 0,
            "status": "all_raw_regions_meet_minimum",
        },
        "preemption_observation": {
            "status": "not_used_for_sample_filtering",
            "reason": (
                "no portable reliable per-region involuntary-context-switch "
                "counter; fixed long regions and whole-cycle cluster intervals "
                "retain scheduler effects"
            ),
        },
        "measurement_scope": "native_arithmetic_only",
        "decomposition_boundaries": {"candidate_materialization": "fused"},
        "compiler": {"rustc": "rustc-test"},
        "device": {"kind": "cpu", "identity": "test-cpu"},
        "process_affinity": [0],
        "command_line": [str(binary)],
        "clock": "steady_clock",
        "clock_and_power_state": {
            "before": {"cpu_governor": ["performance"]},
            "after": {"cpu_governor": ["performance"]},
        },
        "environment": {},
        "provenance": {
            "benchmark_source": _identity(source),
            "harness_source": _identity(source),
            "harness_runner": _identity(runner),
            "benchmark_binary": _identity(binary),
            "python_executable": _identity(Path(sys.executable)),
            "wheel": _identity(wheel),
        },
        "records": records,
        "raw_order": raw_order,
    }
    artifact_payload.update(
        _native_harness_fields(source, product_commit=source_commit, runner=runner)
    )
    artifact.write_text(json.dumps(artifact_payload))
    return artifact


def _cuda_artifact(
    tmp_path: Path,
    *,
    source_commit: str = "a" * 40,
    kind: str = "cuda_events",
    include_orders: bool = True,
    include_clocks: bool = True,
) -> Path:
    """Build a small schema-valid CUDA fixture for validator tests."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.cu"
    binary = tmp_path / "benchmark"
    payload = tmp_path / "libgafime_cuda.so"
    wheel = tmp_path / "gafime_cuda.whl"
    for path, contents in (
        (source, b"source"),
        (binary, b"binary"),
        (payload, b"payload"),
    ):
        path.write_bytes(contents)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gafime_cuda/libgafime_cuda.so", payload.read_bytes())

    records: list[dict[str, object]] = []
    host_operations = (
        "ingest_conversion",
        "planning",
        "allocation",
        "h2d_upload",
        "d2h_transfer",
        "report_construction",
    )
    device_operations = (
        "target_stat_preparation",
        "feature_stat_preparation",
        "candidate_materialization",
        "ranking_kernel",
        "ranking_topk",
        "selected_row_gather",
    )
    canonical_orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    order_cycles = [
        canonical_orders[cycle % len(canonical_orders) :]
        + canonical_orders[: cycle % len(canonical_orders)]
        for cycle in range(perf13.MIN_NATIVE_ORDER_REPETITIONS)
    ]
    orders = canonical_orders if include_orders else []

    def timing_record(
        profile: str,
        operation: str,
        metric: str,
        *,
        absolute_order_index: int,
        profile_order: tuple[str, ...],
    ) -> dict[str, object]:
        device_timed = operation in device_operations or operation == "metric_kernel"
        record: dict[str, object] = {
            "profile": profile,
            "operation": operation,
            "metric": metric,
            "evidence_lane": (
                "supplemental_internal_kernel"
                if device_timed
                else "supplemental_host_phase"
            ),
            "comparability": (
                "direct_kernel_only" if device_timed else "supplemental_host_control"
            ),
            "samples_us": [perf13.GPU_NATIVE_MIN_SAMPLE_REGION_US] * 30,
            "raw_samples_us": [perf13.GPU_NATIVE_MIN_SAMPLE_REGION_US] * 30,
            "loop_count_per_sample": 1,
            "loop_counts_per_sample": [1] * 30,
            "sample_region_target_us": perf13.GPU_NATIVE_MIN_SAMPLE_REGION_US,
            "sample_region_min_observed_us": perf13.GPU_NATIVE_MIN_SAMPLE_REGION_US,
            "sample_region_target_met": True,
            "precondition_iterations": 10,
            "precondition_duration_us": 100_000.0,
            "precondition_max_batch_iterations": 1,
            "precondition_clock": (
                "cuda_event_stream" if device_timed else "host_steady_clock"
            ),
        }
        if include_clocks:
            record.update(
                {
                    "clock": (
                        "cuda_event_stream_clock"
                        if device_timed
                        else "host_steady_clock"
                    ),
                    "synchronization": (
                        "cuda_event_synchronize" if device_timed else "host_monotonic"
                    ),
                    "timing_scope": "device_event" if device_timed else "host_only",
                }
            )
        if include_orders:
            record.update(
                {
                    "order_index": absolute_order_index,
                    "profile_order": list(profile_order),
                }
            )
        return record

    for cycle_index, cycle_orders in enumerate(order_cycles):
        for order_index, order in enumerate(cycle_orders):
            absolute_order_index = cycle_index * len(canonical_orders) + order_index
            for profile in order:
                records.extend(
                    timing_record(
                        profile,
                        operation,
                        "none",
                        absolute_order_index=absolute_order_index,
                        profile_order=order,
                    )
                    for operation in host_operations + device_operations
                )
                records.extend(
                    timing_record(
                        profile,
                        "metric_kernel",
                        metric,
                        absolute_order_index=absolute_order_index,
                        profile_order=order,
                    )
                    for metric in perf13.ALL_METRICS
                )
                canonical = timing_record(
                    profile,
                    "payload_execute",
                    "pearson",
                    absolute_order_index=absolute_order_index,
                    profile_order=order,
                )
                canonical.update(
                    {
                        "evidence_lane": "canonical_payload_api",
                        "comparability": "within_abi_surface_only",
                        "timing_scope": "host_synchronized_payload_api",
                        "synchronization": "cuda_device_synchronize",
                    }
                )
                records.append(canonical)

    schema = (
        "gafime.native-decomposition.v1"
        if kind == "native_decomposition"
        else "gafime.cuda.native_timing.v2"
    )
    artifact = tmp_path / "cuda-native.json"
    artifact_payload = {
        "schema": schema,
        "status": "pass",
        "backend": "cuda",
        "profiles": list(perf13.PROFILE_ORDER),
        "source_commit": source_commit,
        "source_root": str(tmp_path),
        "product_source_root": str(tmp_path),
        "source_tree_state": {"status": "clean", "entries": []},
        "linked_direct_kernel_product": {
            "root": str(tmp_path),
            "commit": source_commit,
            "precision_source_sha256": "4" * 64,
            "precision_header_sha256": "5" * 64,
            "continuous_unary_available": True,
        },
        "input_policy": "common-f64",
        "input_identity": {
            "matrix_sha256": "1" * 64,
            "target_sha256": "2" * 64,
            "feature_names_sha256": "3" * 64,
        },
        "warmups": 10,
        "repeats": 30,
        "sample_region_target_us": perf13.GPU_NATIVE_MIN_SAMPLE_REGION_US,
        "per_record_untimed_same_cell_preconditions": 10,
        "per_record_untimed_precondition_min_us": 100_000.0,
        "precondition_device_batch_target_us": 1_000.0,
        "max_precondition_batch_iterations": 4096,
        "calibration_policy": "fixed_loop_count_per_cell_no_per_sample_rescaling",
        "calibration_prepass": {
            "performed": True,
            "profile_order": list(perf13.PROFILE_ORDER),
            "records_discarded": len(perf13.PROFILE_ORDER)
            * (
                len(host_operations)
                + len(device_operations)
                + len(perf13.ALL_METRICS)
                + 1
            ),
            "samples_discarded": len(perf13.PROFILE_ORDER)
            * (
                len(host_operations)
                + len(device_operations)
                + len(perf13.ALL_METRICS)
                + 1
            )
            * 30,
            "calibrated_key_count": len(perf13.PROFILE_ORDER)
            * (
                len(host_operations)
                + len(device_operations)
                + len(perf13.ALL_METRICS)
                + 1
            ),
            "uses_shared_calibration_cache": True,
            "included_payload_api": True,
            "included_in_profile_order_cycles": False,
        },
        "order_schedule": "deterministic_per_cycle_shuffle_v1",
        "order_seed": 20260810,
        "order_repetitions": perf13.MIN_NATIVE_ORDER_REPETITIONS,
        "profile_order_cycles": (
            [[list(order) for order in cycle] for cycle in order_cycles]
            if include_orders
            else []
        ),
        "execution_mode": "canonical_payload",
        "canonical_payload_resolution": {
            "status": "resolved",
            "symbols": sorted(
                {
                    "gafime_gpu_matrix_alloc_v2",
                    "gafime_gpu_matrix_upload_v2",
                    "gafime_gpu_execute_v2",
                    "gafime_gpu_matrix_free_v2",
                }
            ),
        },
        "decomposition_boundaries": {
            "candidate_materialization": "fused into the measured path"
        },
        "compiler": {
            "nvcc_major": 13,
            "nvcc_minor": 3,
            "nvcc": {"status": "observed", "version": "nvcc-test"},
            "host_cxx": {"status": "observed", "version": "cxx-test"},
            "linker": {"status": "observed", "version": "ld-test"},
        },
        "device": {"name": "test-cuda", "runtime_version": 1},
        "process_affinity": [0],
        "command_line": [str(binary), "--profiles", "fp32,mixed,fp64"],
        "environment": {},
        "clock": {"host": "steady_clock", "device": "cudaEvent"},
        "clock_and_power_capture_point": (
            perf13.GPU_NATIVE_CLOCK_POWER_CAPTURE_BOUNDARY
        ),
        "clock_and_power_state": {
            "before": {
                "cpu_governor": {
                    "status": "observed",
                    "values": ["performance"],
                },
                "nvidia_smi": {
                    "status": "pass",
                    "source": "command",
                    "output": "gpu,p8,clock=100,power=10",
                },
            },
            "after": {
                "cpu_governor": {
                    "status": "observed",
                    "values": ["performance"],
                },
                "nvidia_smi": {
                    "status": "pass",
                    "source": "command",
                    "output": "gpu,p0,clock=200,power=20",
                },
            },
        },
        "profile_orders": ([list(order) for order in orders]),
        "provenance": {
            "benchmark_source": _identity(source),
            "harness_source": _identity(source),
            "benchmark_binary": _identity(binary),
            "payload": _identity(payload),
            "python_executable": _identity(Path(sys.executable)),
            "wheel": _identity(wheel),
        },
        "records": records,
    }
    artifact_payload.update(
        _native_harness_fields(source, product_commit=source_commit)
    )
    artifact.write_text(json.dumps(artifact_payload))
    return artifact


def _rocm_artifact(tmp_path: Path) -> Path:
    """Adapt the complete CUDA validator fixture to ROCm's schema/provenance."""

    seed = _cuda_artifact(tmp_path / "cuda-seed")
    payload = json.loads(seed.read_text())
    benchmark_binary = Path(payload["provenance"]["benchmark_binary"]["path"])
    rocm_payload = tmp_path / "libgafime_rocm.so"
    rocm_payload.write_bytes(b"rocm-payload")
    wheel = tmp_path / "gafime_rocm.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gafime_rocm/libgafime_rocm.so", rocm_payload.read_bytes())

    payload.update(
        {
            "schema": "gafime.rocm.native_timing.v2",
            "backend": "rocm",
            "abi_surface": "numeric-route-v2",
            "self_checks": {
                "abi_surface": "numeric-route-v2",
                "canonical_routes": True,
                "typed_precision_profiles": False,
                "canonical_symbols_authenticated": True,
            },
            "compiler": {
                "predefined_version": "rocm-test",
                "hipcc": {"status": "observed", "version": "hipcc-test"},
                "clangxx": {"status": "observed", "version": "clang-test"},
                "linker": {"status": "observed", "version": "ld-test"},
            },
            "device": {"name": "test-rocm", "gcn_arch": "gfx-test"},
            "clock": {"host": "steady_clock", "device": "hipEvent"},
            "command_line": [
                str(benchmark_binary),
                "--profiles",
                "fp32,mixed,fp64",
            ],
        }
    )
    for phase in ("before", "after"):
        payload["clock_and_power_state"][phase].pop("nvidia_smi")
        payload["clock_and_power_state"][phase]["rocm_smi"] = {
            "status": "pass",
            "source": "rocm-smi",
            "output": json.dumps(
                {
                    "card0": {
                        "sclk clock level": "100Mhz",
                        "Average Graphics Package Power (W)": 10,
                    }
                }
            ),
        }
    payload["provenance"]["payload"] = _identity(rocm_payload)
    payload["provenance"]["wheel"] = _identity(wheel)
    for record in payload["records"]:
        if record["operation"] == "ranking_kernel":
            record["operation"] = "ranking_target_ranks"
        if record.get("clock") == "cuda_event_stream_clock":
            record["clock"] = "hip_event_elapsed_after_synchronized_execute"
        if record.get("synchronization") == "cuda_event_synchronize":
            record["synchronization"] = "hipEventSynchronize"
        if record.get("precondition_clock") == "cuda_event_stream":
            record["precondition_clock"] = "hip_event_default_stream"
    artifact = tmp_path / "rocm-native.json"
    artifact.write_text(json.dumps(payload))
    return artifact


def test_perf13_rejects_arbitrary_hash_only_native_file(tmp_path: Path) -> None:
    artifact = tmp_path / "arbitrary.json"
    artifact.write_text('{"looks_like":"native evidence"}\n')
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any("backend_schema_mismatch" in failure for failure in loaded["failures"])


def test_perf13_requires_complete_profile_union_not_kind_intersection(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, ("fp32",))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, artifact)
    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is True
    readiness = perf13._native_evidence_backend_readiness(
        loaded,
        ("core",),
        (perf13.Variant("candidate", "python", None, ()),),
    )
    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "native_profile_coverage_incomplete"
        for failure in readiness["failures"]
    )


def test_perf13_rejects_core_report_claim_from_arithmetic_fixture(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload["records"].append(
        {
            "profile": "fp32",
            "operation": "report_construction",
            "metric": "none",
            "samples_us": [1.0] * 30,
        }
    )
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "core_native_report_construction_must_not_be_claimed" in failure
        for failure in loaded["failures"]
    )


def test_perf13_requires_core_precondition_policy_and_raw_evidence(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload.pop("per_sample_untimed_precondition_min_ns")
    payload.pop("raw_order")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "core_native_precondition_time_floor_required" in failure
        for failure in loaded["failures"]
    )
    assert any(
        "core_native_raw_precondition_evidence_required" in failure
        for failure in loaded["failures"]
    )


def test_perf13_rejects_core_raw_precondition_below_floor(tmp_path: Path) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload["raw_order"][0]["precondition_duration_ns"] = 99_999_999
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "core_native_raw_precondition_floor_not_met" in failure
        for failure in loaded["failures"]
    )


def test_perf13_rejects_any_core_raw_measured_region_below_floor(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload["raw_order"][0]["duration_ns"] = 99_999_999
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "under-target-raw-region-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "core_native_raw_region_below_100ms" in failure
        for failure in loaded["failures"]
    )


@pytest.mark.parametrize(
    ("loop_count", "expected_failure"),
    (
        (None, "core_fixed_loop_count_required"),
        (0, "core_fixed_loop_count_required"),
        (True, "core_fixed_loop_count_required"),
        (2, "core_duration_normalization_mismatch"),
    ),
)
def test_perf13_rejects_missing_invalid_or_inconsistent_core_loop_count(
    tmp_path: Path,
    loop_count: object,
    expected_failure: str,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    record = payload["records"][0]
    if loop_count is None:
        record.pop("loop_count_per_sample")
    else:
        record["loop_count_per_sample"] = loop_count
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "invalid-core-loop-count-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(expected_failure in failure for failure in loaded["failures"])


def test_perf13_preserves_inconclusive_core_evidence_but_blocks_claim(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    sensitivity = payload["order_sensitivity"]
    sensitivity["inconclusive_cells"] = 1
    sensitivity["status"] = "inconclusive_order_effect_requires_rerun"
    cell = sensitivity["cells"][0]
    cell["status"] = "inconclusive_order_effect_requires_rerun"
    cell["position_contrasts"][0]["corrected_bootstrap_ci_percent"] = [
        -0.5,
        1.2,
    ]
    cell["position_contrasts"][0]["status"] = "inconclusive_order_effect_requires_rerun"
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "inconclusive-core-order-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is True
    assert loaded["evidence_integrity_valid"] is True
    assert loaded["performance_claim_ready"] is False
    assert loaded["arithmetic_claims_valid"] is False
    assert any(
        "core_native_order_sensitivity_not_claim_ready" in failure
        for failure in loaded["claim_failures"]
    )


def test_metal_validator_handles_incomplete_artifact_without_cross_backend_state(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "metal.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "gafime.metal.native_timing.v1",
                "status": "pass",
                "backend": "metal",
                "profile": "fp32",
                "source_commit": "a" * 40,
                "precision_domains": {
                    "storage": "fp32",
                    "pointwise": "fp32",
                    "reduction": "fp32",
                    "result": "fp32",
                },
                "warmups": 10,
                "repeats": 30,
                "gpu_timing_supported": False,
                "records": [],
            }
        )
    )

    validated = perf13._validate_metal_native_timing_artifact(
        artifact, manifest_source_commit="a" * 40
    )

    assert validated["status"] == "invalid"
    assert "complete_gpu_timestamp_support_required" in validated["failures"]


def test_metal_timing_record_validator_enforces_round_trip_and_fixed_loops() -> None:
    record: dict[str, object] = {
        "samples_us": [1250.0] * 3,
        "raw_samples_us": [5000.0] * 3,
        "host_synchronized_samples_us": [1500.0] * 3,
        "raw_host_synchronized_samples_us": [6000.0] * 3,
        "gpu_timestamp_samples_us": [1250.0] * 3,
        "raw_gpu_timestamp_samples_us": [5000.0] * 3,
        "gpu_timestamp_valid_samples": 3,
        "loop_count_per_sample": 4,
        "loop_counts_per_sample": [4] * 3,
        "sample_region_target_us": 5000.0,
        "sample_region_min_observed_us": 5000.0,
        "sample_region_target_met": True,
    }

    assert (
        perf13._metal_timing_record_failures(
            record,
            prefix="record_0",
            repeats=3,
            expected_target_us=5000.0,
        )
        == []
    )

    mismatched = dict(record)
    mismatched["samples_us"] = [1250.01, 1250.0, 1250.0]
    failures = perf13._metal_timing_record_failures(
        mismatched,
        prefix="record_0",
        repeats=3,
        expected_target_us=5000.0,
    )
    assert "record_0_duration_normalization_mismatch" in failures

    variable_loops = dict(record)
    variable_loops["loop_counts_per_sample"] = [4, 8, 4]
    failures = perf13._metal_timing_record_failures(
        variable_loops,
        prefix="record_0",
        repeats=3,
        expected_target_us=5000.0,
    )
    assert "record_0_fixed_loop_count_required" in failures

    invalid_host = dict(record)
    invalid_host["host_synchronized_samples_us"] = [1500.0, float("inf"), 1500.0]
    failures = perf13._metal_timing_record_failures(
        invalid_host,
        prefix="record_0",
        repeats=3,
        expected_target_us=5000.0,
    )
    assert "record_0_host_timing_samples_invalid" in failures

    invalid_gpu = dict(record)
    invalid_gpu["gpu_timestamp_samples_us"] = [1250.01, 1250.0, 1250.0]
    failures = perf13._metal_timing_record_failures(
        invalid_gpu,
        prefix="record_0",
        repeats=3,
        expected_target_us=5000.0,
    )
    assert "record_0_gpu_timestamp_duration_normalization_mismatch" in failures

    missing_gpu = dict(record)
    missing_gpu.pop("raw_gpu_timestamp_samples_us")
    failures = perf13._metal_timing_record_failures(
        missing_gpu,
        prefix="record_0",
        repeats=3,
        expected_target_us=5000.0,
    )
    assert "record_0_gpu_timestamp_samples_invalid" in failures


def test_discrete_metal_managed_resource_sync_maps_to_d2h_transfer() -> None:
    assert perf13._native_operation_names(
        "d2h_managed_resource_synchronize", "none"
    ) == {"d2h_transfer"}


def test_metal_timing_record_validator_rejects_zero_repeats_without_crashing() -> None:
    failures = perf13._metal_timing_record_failures(
        {
            "samples_us": [],
            "raw_samples_us": [],
            "loop_count_per_sample": 1,
            "loop_counts_per_sample": [],
            "sample_region_target_us": 5000.0,
            "sample_region_target_met": True,
        },
        prefix="record_0",
        repeats=0,
        expected_target_us=5000.0,
    )

    assert "record_0_normalized_samples_invalid" in failures
    assert "record_0_raw_samples_invalid" in failures
    assert "record_0_fixed_loop_count_required" in failures
    assert "record_0_sample_region_gate_invalid" in failures


def test_metal_validator_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "metal-duplicate-key.json"
    artifact.write_text(
        '{"schema":"gafime.metal.native_timing.v1",'
        '"status":"pass","status":"validated"}',
        encoding="utf-8",
    )

    validated = perf13._validate_metal_native_timing_artifact(
        artifact, manifest_source_commit="a" * 40
    )

    assert validated["status"] == "invalid"
    assert any("duplicate_json_key:status" in item for item in validated["failures"])


def _minimal_metal_lifecycle_artifact(
    tmp_path: Path, lifecycle: dict[str, object], records: list[dict[str, object]]
) -> Path:
    artifact = tmp_path / "metal-lifecycle.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "gafime.metal.native_timing.v1",
                "status": "pass",
                "backend": "metal",
                "profile": "fp32",
                "source_commit": "a" * 40,
                "precision_domains": {
                    "storage": "fp32",
                    "pointwise": "fp32",
                    "reduction": "fp32",
                    "result": "fp32",
                },
                "warmups": 10,
                "repeats": 30,
                "sample_region_target_us": 5000.0,
                "gpu_timing_supported": False,
                "execution_mode": "supplemental_internal_kernel",
                "records": [],
                "canonical_payload_lifecycle": lifecycle,
                "canonical_payload_records": records,
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _minimal_metal_lifecycle() -> dict[str, object]:
    return {
        "status": "validated",
        "schema": "gafime.native-decomposition.v1",
        "execution_layer": "installed_payload_dylib",
        "abi": "1.1",
        "route_count": 1,
        "mixed_route_rejected": True,
        "fp64_route_rejected": True,
        "profile_mask": 0x1,
        "storage_dtype_mask": 0x1,
        "result_dtype_mask": 0x1,
        "route_query_status": 0,
        "route_fill_status": 0,
        "matrix_alloc_status": 0,
        "matrix_upload_status": 0,
        "execute_status": 0,
        "matrix_free_status": 0,
        "abi_surface": "numeric-route-v2",
        "symbols": sorted(perf13.CANONICAL_ABI_SYMBOLS_BY_SURFACE["numeric-route-v2"]),
        "optional_symbols": [],
        "records_field": "canonical_payload_records",
        "permutation_supported": True,
        "result_checksum": 1.0,
        "operation_status": {
            "matrix_update_target": {"status": "pass", "abi_status": 0},
            "execution_memory_peak": {
                "status": "pass",
                "abi_status": 0,
                "bytes": 1,
            },
            "permutation_memory_peak": {
                "status": "pass",
                "abi_status": 0,
                "bytes": 1,
            },
            "permutation_pvalues": {
                "status": "pass",
                "abi_status": 0,
                "row_count": 1,
            },
            "interaction_diagnostics": {"status": "pass", "abi_status": 0},
        },
    }


def test_metal_lifecycle_gate_authenticates_masks_statuses_and_fp64_rejection(
    tmp_path: Path,
) -> None:
    lifecycle = _minimal_metal_lifecycle()
    lifecycle.update(
        {
            "fp64_route_rejected": False,
            "profile_mask": 0x3,
            "storage_dtype_mask": 0x3,
            "result_dtype_mask": 0x3,
            "execute_status": -1,
        }
    )
    artifact = _minimal_metal_lifecycle_artifact(tmp_path, lifecycle, [])

    validated = perf13._validate_metal_native_timing_artifact(
        artifact, manifest_source_commit="a" * 40
    )

    assert "metal_fp64_route_rejection_required" in validated["failures"]
    assert "metal_fp32_profile_mask_required" in validated["failures"]
    assert "metal_fp32_storage_dtype_mask_required" in validated["failures"]
    assert "metal_fp32_result_dtype_mask_required" in validated["failures"]
    assert "metal_execute_status_must_be_ok" in validated["failures"]


def test_metal_generic_lifecycle_requires_permutation_and_unique_records(
    tmp_path: Path,
) -> None:
    lifecycle = _minimal_metal_lifecycle()
    lifecycle["permutation_supported"] = False
    symbols = lifecycle["symbols"]
    assert isinstance(symbols, list)
    symbols.append(symbols[0])
    lifecycle["optional_symbols"] = [
        "gafime_gpu_permutation_memory_peak_v2",
        "gafime_gpu_permutation_memory_peak_v2",
    ]
    operation_status = lifecycle["operation_status"]
    assert isinstance(operation_status, dict)
    operation_status["permutation_memory_peak"] = {
        "status": "unsupported",
        "abi_status": -2,
        "bytes": 0,
    }
    operation_status["permutation_pvalues"] = {
        "status": "unsupported",
        "abi_status": -2,
        "row_count": 0,
    }
    duplicate_record = {
        "operation": "matrix_allocation",
        "metric": "none",
        "samples_us": [5000.0] * 30,
        "raw_samples_us": [5000.0] * 30,
        "loop_count_per_sample": 1,
        "loop_counts_per_sample": [1] * 30,
        "sample_region_target_us": 5000.0,
        "sample_region_min_observed_us": 5000.0,
        "sample_region_target_met": True,
        "clock": "host_steady_clock_canonical_abi1_1",
        "synchronization": (
            "canonical_abi1_1_payload_call_returns_after_device_completion"
        ),
    }
    artifact = _minimal_metal_lifecycle_artifact(
        tmp_path, lifecycle, [duplicate_record, duplicate_record]
    )

    validated = perf13._validate_metal_native_timing_artifact(
        artifact, manifest_source_commit="a" * 40
    )

    assert "metal_permutation_support_marker_required" in validated["failures"]
    assert "canonical_payload_symbols_must_be_unique" in validated["failures"]
    assert "metal_optional_symbols_must_be_unique" in validated["failures"]
    assert (
        "metal_permutation_memory_peak_operation_status_must_be_ok"
        in validated["failures"]
    )
    assert (
        "metal_permutation_pvalues_operation_status_must_be_ok" in validated["failures"]
    )
    assert "canonical_record_1_duplicate_identity" in validated["failures"]
    assert "canonical_payload_record_count_mismatch" in validated["failures"]


def test_generic_native_validator_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "cuda-duplicate-key.json"
    artifact.write_text(
        '{"schema":"gafime.cuda.native_timing.v2",'
        '"status":"pass","status":"validated"}',
        encoding="utf-8",
    )

    validated = perf13._validate_native_artifact(
        artifact,
        backend="cuda",
        kind="cuda_events",
        manifest_source_commit="a" * 40,
    )

    assert validated["status"] == "invalid"
    assert any("duplicate_json_key:status" in item for item in validated["failures"])


def test_native_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "duplicate-manifest.json"
    manifest.write_text(
        '{"schema":"gafime.native-evidence-manifest.v1",'
        '"status":"validated","status":"not_collected",'
        '"arithmetic_claims_valid":false,"artifacts":[]}',
        encoding="utf-8",
    )

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["status"] == "invalid"
    assert loaded["evidence_integrity_valid"] is False
    assert any("duplicate_json_key:status" in item for item in loaded["failures"])


def test_metal_inline_lifecycle_authenticates_product_and_common_harness(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "libgafime_metal_v1.dylib"
    wheel_path = tmp_path / "gafime.whl"
    harness_path = tmp_path / "metal_precision_native_timing.mm"
    payload_path.write_bytes(b"metal-payload")
    harness_path.write_bytes(b"common-harness")
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(
            "gafime/_metal/libgafime_metal_v1.dylib", payload_path.read_bytes()
        )

    clean_tree = {"status": "clean", "entry_count": 0, "entries": []}
    product_binding = {
        "root": str(tmp_path / "product"),
        "commit": "a" * 40,
        "tree_state": clean_tree,
    }
    harness_identity = _identity(harness_path)
    harness_binding = {
        "root": str(tmp_path / "harness"),
        "commit": "b" * 40,
        "relative_path": harness_path.name,
        "sha256": harness_identity["sha256"],
        "source_sha256": harness_identity["sha256"],
        "current_git_blob": "c" * 40,
        "head_git_blob": "c" * 40,
        "tree_state": clean_tree,
    }
    harness_blob = {
        "relative_path": harness_path.name,
        "source_sha256": harness_identity["sha256"],
        "current_git_blob": "c" * 40,
        "head_git_blob": "c" * 40,
    }
    artifact = {
        "source_commit": "a" * 40,
        "source_root": product_binding["root"],
        "source_tree_state": clean_tree,
        "product_source_commit": "a" * 40,
        "product_source_root": product_binding["root"],
        "product_source_tree_state": clean_tree,
        "product_source_binding": product_binding,
        "harness_source_commit": "b" * 40,
        "harness_source_root": harness_binding["root"],
        "harness_source_tree_state": clean_tree,
        "harness_source_binding": harness_binding,
        "harness_source_blob": harness_blob,
        "provenance": {
            "payload": _identity(payload_path),
            "wheel": _identity(wheel_path),
            "harness_source": harness_identity,
        },
    }
    lifecycle = {
        **{
            key: artifact[key]
            for key in (
                "source_commit",
                "source_root",
                "source_tree_state",
                "product_source_commit",
                "product_source_root",
                "product_source_tree_state",
                "product_source_binding",
                "harness_source_commit",
                "harness_source_root",
                "harness_source_tree_state",
                "harness_source_binding",
                "harness_source_blob",
            )
        },
        "wheel_member": "gafime/_metal/libgafime_metal_v1.dylib",
        "wheel_member_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "provenance": artifact["provenance"],
    }

    assert perf13._metal_inline_lifecycle_provenance_failures(lifecycle, artifact) == []

    tampered = json.loads(json.dumps(lifecycle))
    tampered["harness_source_commit"] = "d" * 40
    failures = perf13._metal_inline_lifecycle_provenance_failures(tampered, artifact)
    assert "canonical_inline_harness_source_commit_mismatch" in failures


def test_metal_clock_power_provenance_accepts_honest_unavailable_telemetry() -> None:
    phase = {
        "cpu_governor": {
            "status": "unavailable",
            "detail": "sysfs CPU governors are not exposed by macOS",
        },
        "system_profiler": {
            "status": "pass",
            "source": "system_profiler SPDisplaysDataType -json",
            "output": '{"SPDisplaysDataType": [{"sppci_model": "Apple GPU"}]}',
        },
        "cpu_power_management": {
            "status": "pass",
            "source": "pmset -g custom",
            "output": "Battery Power: powermode 0",
        },
        "metal_gpu_clock_power": {
            "status": "unavailable",
            "detail": "Metal does not expose live GPU clock or power telemetry",
        },
    }
    payload = {
        "clock_and_power_capture_point": (
            "before and after all timed benchmark regions"
        ),
        "clock_and_power_state": {"before": phase, "after": phase},
    }

    assert perf13._gpu_clock_power_failures("metal", payload) == []


def test_metal_clock_power_provenance_rejects_missing_unavailable_detail() -> None:
    phase = {
        "cpu_governor": {"status": "unavailable", "detail": "not exposed"},
        "system_profiler": {"status": "pass", "output": "Apple GPU"},
        "cpu_power_management": {"status": "unavailable", "detail": "not exposed"},
        "metal_gpu_clock_power": {"status": "unavailable", "detail": "not exposed"},
    }
    after = json.loads(json.dumps(phase))
    after["metal_gpu_clock_power"].pop("detail")
    payload = {
        "clock_and_power_capture_point": (
            "before and after all timed benchmark regions"
        ),
        "clock_and_power_state": {"before": phase, "after": after},
    }

    assert (
        "metal_gpu_clock_power_after_unavailable_detail_required"
        in perf13._gpu_clock_power_failures("metal", payload)
    )


def _canonical_cuda_lifecycle(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = tmp_path / "libgafime_cuda.so"
    wheel = tmp_path / "gafime_cuda.whl"
    consumer = tmp_path / "abi-consumer"
    source = tmp_path / "abi_1_1_c_consumer.c"
    payload.write_bytes(b"payload")
    consumer.write_bytes(b"consumer")
    source.write_bytes(b"source")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gafime_cuda/libgafime_cuda.so", payload.read_bytes())
    operations = sorted(perf13.CANONICAL_ABI_LIFECYCLE_OPERATIONS)
    marker = {
        "schema": "gafime.abi-1.1-consumer-result.v1",
        "status": "pass",
        "abi_surface": "numeric-route-v2",
        "backend_kind": 2,
        "route_count": 3,
        "operations": operations,
    }
    provenance = {
        "payload": _identity(payload),
        "wheel": _identity(wheel),
        "consumer_binary": _identity(consumer),
        "consumer_source": _identity(source),
        "harness_source": _identity(source),
    }
    lifecycle = {
        "schema": "gafime.native-decomposition.v1",
        "status": "pass",
        "execution_mode": "canonical_payload",
        "execution_layer": "independent_abi_1_1_c_consumer",
        "abi": "1.1",
        "abi_surface": "numeric-route-v2",
        "contract_role": "candidate_canonical_numeric_route",
        "backend": "cuda",
        "profiles": ["fp32", "mixed", "fp64"],
        "source_commit": "a" * 40,
        "product_source_commit": "a" * 40,
        "source_tree_state": {"status": "clean", "entries": []},
        "harness_source_commit": "b" * 40,
        "harness_source_tree_state": {"status": "clean", "entries": []},
        "wheel_member": "gafime_cuda/libgafime_cuda.so",
        "wheel_member_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "operations": operations,
        "symbols": sorted(perf13.CANONICAL_ABI_SYMBOLS_BY_SURFACE["numeric-route-v2"]),
        "route_count": 3,
        "consumer_result": {
            "schema": "gafime.abi-1.1-consumer-result.v1",
            "status": "pass",
            "returncode": 0,
            "marker": marker,
        },
        "provenance": provenance,
    }
    return lifecycle, provenance


def test_canonical_lifecycle_requires_executed_independent_consumer(
    tmp_path: Path,
) -> None:
    lifecycle, provenance = _canonical_cuda_lifecycle(tmp_path)
    assert (
        perf13._canonical_lifecycle_failures(
            lifecycle,
            backend="cuda",
            source_commit="a" * 40,
            artifact_provenance=provenance,
        )
        == []
    )

    lifecycle.pop("consumer_result")
    lifecycle["execution_layer"] = "resolved_symbols_only"
    failures = perf13._canonical_lifecycle_failures(
        lifecycle,
        backend="cuda",
        source_commit="a" * 40,
        artifact_provenance=provenance,
    )
    assert "canonical_payload_lifecycle_independent_consumer_required" in failures
    assert "canonical_payload_lifecycle_consumer_result_required" in failures


def test_canonical_lifecycle_authenticates_typed_surface_and_common_harness(
    tmp_path: Path,
) -> None:
    lifecycle, provenance = _canonical_cuda_lifecycle(tmp_path)
    typed_operations = sorted(perf13.CANONICAL_ABI_TYPED_LIFECYCLE_OPERATIONS)
    typed_symbols = sorted(
        perf13.CANONICAL_ABI_SYMBOLS_BY_SURFACE["precision-typed-v1.1"]
    )
    lifecycle.update(
        {
            "abi_surface": "precision-typed-v1.1",
            "contract_role": "historical_pre_freeze_typed_baseline",
            "execution_layer": "independent_abi_1_1_typed_c_consumer",
            "operations": typed_operations,
            "symbols": typed_symbols,
        }
    )
    lifecycle["consumer_result"] = {
        "schema": "gafime.abi-1.1-typed-consumer-result.v1",
        "status": "pass",
        "returncode": 0,
        "marker": {
            "schema": "gafime.abi-1.1-typed-consumer-result.v1",
            "status": "pass",
            "abi_surface": "precision-typed-v1.1",
            "backend_kind": 2,
            "profile_count": 3,
            "operations": typed_operations,
        },
    }
    assert (
        perf13._canonical_lifecycle_failures(
            lifecycle,
            backend="cuda",
            source_commit="a" * 40,
            artifact_provenance=provenance,
        )
        == []
    )

    lifecycle["harness_source_commit"] = "c" * 40
    failures = perf13._canonical_lifecycle_failures(
        lifecycle,
        backend="cuda",
        source_commit="a" * 40,
        artifact_provenance=provenance,
    )
    assert failures == []


def test_canonical_lifecycle_requires_explicit_surface_and_product_commit(
    tmp_path: Path,
) -> None:
    lifecycle, provenance = _canonical_cuda_lifecycle(tmp_path)
    lifecycle.pop("abi_surface")
    lifecycle.pop("product_source_commit")
    failures = perf13._canonical_lifecycle_failures(
        lifecycle,
        backend="cuda",
        source_commit="a" * 40,
        artifact_provenance=provenance,
    )
    assert "canonical_payload_lifecycle_abi_surface_required" in failures
    assert "canonical_payload_lifecycle_source_commit_mismatch" in failures


def test_perf13_self_check_is_adversarially_executable() -> None:
    assert perf13._self_check() == 0


def test_variant_workers_share_frozen_canonical_harness(tmp_path: Path) -> None:
    baseline = perf13.Variant(
        "baseline", sys.executable, str(tmp_path / "baseline-source"), ()
    )
    candidate = perf13.Variant(
        "candidate", sys.executable, str(tmp_path / "candidate-source"), ()
    )

    expected = str(Path(perf13.__file__).resolve())
    assert perf13._variant_worker_script(baseline) == expected
    assert perf13._variant_worker_script(candidate) == expected


def test_two_variant_native_manifests_preserve_distinct_source_commits(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline_artifact = _core_artifact(
        baseline_root, perf13.PROFILE_ORDER, source_commit="a" * 40
    )
    candidate_artifact = _core_artifact(
        candidate_root, perf13.PROFILE_ORDER, source_commit="b" * 40
    )
    baseline_manifest = tmp_path / "baseline-manifest.json"
    candidate_manifest = tmp_path / "candidate-manifest.json"
    _write_manifest(
        baseline_manifest,
        baseline_artifact,
        variant="baseline",
        source_commit="a" * 40,
    )
    _write_manifest(
        candidate_manifest,
        candidate_artifact,
        variant="candidate",
        source_commit="b" * 40,
    )

    loaded = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )

    assert loaded["valid"] is True
    assert loaded["source_commits_by_variant"] == {
        "baseline": "a" * 40,
        "candidate": "b" * 40,
    }


def test_single_shared_native_manifest_preserves_source_commit(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER, source_commit="a" * 40)
    manifest = tmp_path / "native-evidence.json"
    _write_manifest(
        manifest,
        artifact,
        variant="current",
        source_commit="a" * 40,
    )

    loaded = perf13._load_native_evidence_specs(str(manifest))

    assert loaded["valid"] is True
    assert loaded["source_commits_by_variant"] == {"current": "a" * 40}


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    (
        (
            lambda payload: payload.__setitem__("product_source_commit", "f" * 40),
            "native_product_source_commit_mismatch",
        ),
        (
            lambda payload: payload["harness_source_blob"].__setitem__(
                "head_git_blob", "e" * 40
            ),
            "native_harness_git_blob_mismatch",
        ),
        (
            lambda payload: payload["harness_runner_blob"].__setitem__(
                "head_git_blob", "f" * 40
            ),
            "native_harness_runner_git_blob_mismatch",
        ),
        (
            lambda payload: payload["provenance"].pop("harness_runner"),
            "missing_provenance_harness_runner",
        ),
    ),
)
def test_native_helper_provenance_fails_closed(
    tmp_path: Path, mutation, expected_failure: str
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    mutation(payload)
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "native-provenance-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(expected_failure in failure for failure in loaded["failures"])


def test_rocm_native_route_gate_accepts_authenticated_typed_baseline() -> None:
    payload = {
        "abi_surface": "precision-typed-v1.1",
        "self_checks": {
            "abi_surface": "precision-typed-v1.1",
            "canonical_routes": False,
            "typed_precision_profiles": True,
            "canonical_symbols_authenticated": True,
        },
    }

    assert (
        perf13._native_payload_route_failures("rocm", payload, kind="rocm_events") == []
    )

    payload["self_checks"]["typed_precision_profiles"] = False
    assert perf13._native_payload_route_failures(
        "rocm", payload, kind="rocm_events"
    ) == ["native_generic_abi_route_unsupported"]


def test_native_ab_requires_one_common_helper_commit_and_hash(tmp_path: Path) -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    manifests = []
    for variant, commit in zip(variants, ("a" * 40, "b" * 40), strict=True):
        root = tmp_path / variant.name
        root.mkdir()
        artifact = _core_artifact(root, perf13.PROFILE_ORDER, source_commit=commit)
        if variant.name == "candidate":
            payload = json.loads(artifact.read_text())
            payload["harness_source_commit"] = "e" * 40
            artifact.write_text(json.dumps(payload))
        manifest = tmp_path / f"{variant.name}-native-provenance.json"
        _write_manifest(
            manifest,
            artifact,
            variant=variant.name,
            source_commit=commit,
        )
        manifests.append(manifest)
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(manifests[0])),
            ("candidate", str(manifests[1])),
        ]
    )

    readiness = perf13._native_evidence_backend_readiness(evidence, ("core",), variants)

    assert readiness["complete"] is False
    assert any(
        failure.get("reason") == "baseline_and_candidate_native_harness_mismatch"
        for failure in readiness["failures"]
    )


def test_core_native_ab_requires_one_common_runner_sha_and_blob(
    tmp_path: Path,
) -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    manifests = []
    for variant, commit in zip(variants, ("a" * 40, "b" * 40), strict=True):
        root = tmp_path / variant.name
        root.mkdir()
        artifact = _core_artifact(root, perf13.PROFILE_ORDER, source_commit=commit)
        if variant.name == "candidate":
            payload = json.loads(artifact.read_text())
            runner_path = Path(payload["provenance"]["harness_runner"]["path"])
            runner_path.write_bytes(b"different common runner")
            runner_identity = _identity(runner_path)
            payload["provenance"]["harness_runner"] = runner_identity
            payload["harness_runner_blob"].update(
                {
                    "source_sha256": runner_identity["sha256"],
                    "current_git_blob": "f" * 40,
                    "head_git_blob": "f" * 40,
                }
            )
            artifact.write_text(json.dumps(payload))
        manifest = tmp_path / f"{variant.name}-runner-provenance.json"
        _write_manifest(
            manifest,
            artifact,
            variant=variant.name,
            source_commit=commit,
        )
        manifests.append(manifest)
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(manifests[0])),
            ("candidate", str(manifests[1])),
        ]
    )

    assert evidence["valid"] is True
    readiness = perf13._native_evidence_backend_readiness(evidence, ("core",), variants)

    assert readiness["complete"] is False
    assert any(
        failure.get("reason") == "baseline_and_candidate_core_harness_runner_mismatch"
        for failure in readiness["failures"]
    )


def test_native_statistics_include_raw_distribution_and_bootstrap_fields(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))
    validation = loaded["artifacts"][0]["validation"]
    sample_statistics = validation["native_statistics"][0]["statistics"]

    assert loaded["valid"] is True
    assert len(sample_statistics["raw_durations"]) == 120
    assert {"median", "mad", "p05", "p95", "bootstrap_median_95_ci"} <= set(
        sample_statistics
    )
    assert sample_statistics["auto_scaling"]["status"] == "observed"


def test_native_statistics_use_normalized_samples_not_calibration_regions(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    repeats = payload["repeats"]
    for record in payload["records"]:
        record["samples_us"] = [1.0] * repeats
        record["raw_samples_us"] = [200_000.0] * repeats
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "normalized-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))
    statistics = loaded["artifacts"][0]["validation"]["native_statistics"][0][
        "statistics"
    ]

    assert loaded["valid"] is True
    assert statistics["statistics_scope"] == "normalized_per_call"
    assert statistics["median"] == 1.0
    assert statistics["normalized_durations"] == [1.0] * repeats
    assert statistics["raw_durations"] == [200_000.0] * repeats


def test_core_native_statistics_accept_normalized_and_raw_nanoseconds(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    repeats = payload["repeats"]
    for record in payload["records"]:
        record.pop("samples_us", None)
        record["samples_ns"] = [1_000_000] * repeats
        record["raw_samples_ns"] = [200_000_000] * repeats
        record["loop_count_per_sample"] = 200
        record["sample_region_target_ns"] = perf13.CORE_MIN_MEASURED_REGION_NS
        record["sample_region_target_met"] = True
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "nanoseconds-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))
    statistics = loaded["artifacts"][0]["validation"]["native_statistics"][0][
        "statistics"
    ]

    assert loaded["valid"] is True
    assert statistics["unit"] == "ns"
    assert statistics["median"] == 1_000_000.0
    assert statistics["raw_durations"] == [200_000_000.0] * repeats
    assert statistics["auto_scaling"]["target_unit"] == "ns"


def test_release_claim_sample_floor_is_hard_100ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_13_precision_profiles.py",
            "--native-evidence",
            "manifest.json",
            "--min-sample-ms",
            "99.999",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        perf13._parse_args()
    assert exc_info.value.code == 2


def test_public_cli_defaults_to_five_complete_order_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "native-evidence.json"
    manifest.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_13_precision_profiles.py",
            "--native-evidence",
            str(manifest),
        ],
    )

    args = perf13._parse_args()

    assert args.order_repetitions == perf13.MIN_PUBLIC_ORDER_REPETITIONS == 5


def test_public_cli_rejects_fewer_than_five_order_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "native-evidence.json"
    manifest.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_13_precision_profiles.py",
            "--native-evidence",
            str(manifest),
            "--order-repetitions",
            "4",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        perf13._parse_args()

    assert exc_info.value.code == 2


def test_readme_canonical_perf13_commands_use_five_order_cycles() -> None:
    readme = (Path(__file__).parent / "README.md").read_text()
    canonical_commands = readme.split(
        "The public matrix includes safe continuous-family", 1
    )[0]
    comparison_command = readme.split("For an exact-head comparison", 1)[1].split(
        "The JSON calls its rate", 1
    )[0]

    assert "--order-repetitions 5" in canonical_commands
    assert "--order-repetitions 5" in comparison_command


def test_native_ab_comparison_uses_normalized_samples(
    tmp_path: Path,
) -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    artifacts = []
    manifests = []
    for variant, normalized in ((variants[0], 1.0), (variants[1], 2.0)):
        root = tmp_path / variant.name
        root.mkdir()
        artifact = _core_artifact(
            root,
            perf13.PROFILE_ORDER,
            source_commit=("a" if variant.name == "baseline" else "b") * 40,
        )
        payload = json.loads(artifact.read_text())
        repeats = payload["repeats"]
        for record in payload["records"]:
            record["samples_us"] = [normalized] * repeats
            record["raw_samples_us"] = [200_000.0] * repeats
        artifact.write_text(json.dumps(payload))
        manifest = tmp_path / f"{variant.name}-manifest.json"
        _write_manifest(
            manifest,
            artifact,
            variant=variant.name,
            source_commit=("a" if variant.name == "baseline" else "b") * 40,
        )
        artifacts.append(artifact)
        manifests.append(manifest)

    evidence = perf13._load_native_evidence_specs(
        [("baseline", str(manifests[0])), ("candidate", str(manifests[1]))]
    )
    comparisons = perf13._native_ab_comparisons(
        evidence, variants, bootstrap_resamples=25, seed=7
    )

    assert comparisons
    assert all(
        comparison["candidate_latency_delta_percent"] == 100.0
        for comparison in comparisons
    )


def test_native_ab_does_not_pair_unsupported_generic_payload(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "unsupported-route")
    payload = json.loads(artifact.read_text())
    payload.pop("canonical_payload_resolution")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "unsupported-route-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    evidence = perf13._load_native_evidence(str(manifest))
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    assert evidence["valid"] is False
    assert any(
        "native_generic_abi_route_evidence_required" in failure
        for failure in evidence["failures"]
    )
    assert (
        perf13._native_ab_comparisons(
            evidence, variants, bootstrap_resamples=25, seed=7
        )
        == []
    )


def test_native_artifact_requires_explicit_input_policy_and_identity(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    manifest = tmp_path / "input-metadata-manifest.json"
    payload = json.loads(artifact.read_text())
    for field, reason in (
        ("input_policy", "input_policy_required"),
        ("input_identity", "input_identity_required"),
    ):
        payload = json.loads(artifact.read_text())
        payload.pop(field)
        artifact.write_text(json.dumps(payload))
        _write_manifest(manifest, artifact)
        loaded = perf13._load_native_evidence(str(manifest))
        assert loaded["valid"] is False
        assert any(reason in failure for failure in loaded["failures"])


def test_native_artifact_requires_clean_tree_and_complete_compiler_metadata(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload["source_tree_state"] = {"status": "dirty", "entries": ["M src"]}
    payload["compiler"] = {"rustc": None}
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "provenance-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "clean_source_tree_required" in failure for failure in loaded["failures"]
    )
    assert any(
        "compiler_rustc_version_required" in failure for failure in loaded["failures"]
    )


def test_rocm_compiler_validation_allows_optional_null_but_requires_observed_tools() -> (
    None
):
    compiler = {
        "predefined_version": "15.2.0",
        "clang_version": None,
        "hipcc": {"status": "observed", "version": "hipcc --version"},
        "clangxx": {"status": "observed", "version": "clang++ --version"},
        "linker": {"status": "observed", "version": "ld --version"},
    }

    assert perf13._compiler_provenance_failures("rocm", compiler) == []
    compiler.pop("hipcc")
    assert "compiler_hipcc_observed_version_required" in (
        perf13._compiler_provenance_failures("rocm", compiler)
    )


def test_native_payload_must_match_declared_wheel_member(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "wheel-mismatch")
    payload = json.loads(artifact.read_text())
    payload_path = Path(payload["provenance"]["payload"]["path"])
    payload_path.write_bytes(b"payload-not-from-wheel")
    payload["provenance"]["payload"] = _identity(payload_path)
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "wheel-mismatch-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "native_payload_not_from_declared_wheel" in failure
        for failure in loaded["failures"]
    )


def test_core_native_artifact_requires_a_real_core_wheel(tmp_path: Path) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload["provenance"].pop("wheel")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-core-wheel-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "missing_provenance_wheel" in failure
        or "core_native_wheel_path_required" in failure
        for failure in loaded["failures"]
    )


def test_core_native_wheel_must_match_public_variant(tmp_path: Path) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    manifest = tmp_path / "core-wheel-binding-manifest.json"
    _write_manifest(manifest, artifact)
    evidence = perf13._load_native_evidence(str(manifest))
    variant = perf13.Variant("candidate", sys.executable, None, ())
    public_results = (
        {
            "kind": "public",
            "status": "pass",
            "backend": "core",
            "native_binaries": [],
            "provenance": {
                "variant": "candidate",
                "source_commit": "a" * 40,
                "wheel_artifacts": [{"sha256": "f" * 64}],
            },
        },
    )

    readiness = perf13._native_evidence_backend_readiness(
        evidence, ("core",), (variant,), public_results
    )

    assert readiness["complete"] is False
    assert any(
        failure.get("reason") == "native_wheel_does_not_match_public_variant"
        for failure in readiness["failures"]
    )


def test_native_environment_normalizes_only_authenticated_variant_paths() -> None:
    def validation(root: str, environment: object) -> dict[str, object]:
        return {
            "environment": environment,
            "provenance": {
                "source_root": {
                    "path": f"{root}/src",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                },
                "benchmark_binary": {"path": f"{root}/bin/benchmark"},
                "payload": {"path": f"{root}/lib/libgafime_cuda.so"},
                "wheel": {"path": f"{root}/wheels/gafime_cuda.whl"},
                "python_executable": {
                    "path": f"{root}/venv/bin/python",
                    "sha256": "e" * 64,
                    "size_bytes": 100,
                },
            },
        }

    baseline = validation(
        "/baseline",
        [
            "CUDA_VISIBLE_DEVICES=0",
            "GAFIME_WHEEL_PATH=/baseline/wheels/gafime_cuda.whl",
            "GAFIME_CUDA_V1_LIB=/baseline/lib/libgafime_cuda.so",
            "VIRTUAL_ENV=/baseline/venv",
            "PATH=/baseline/venv/bin:/baseline/bin:/usr/bin",
            "PYTHONPATH=/baseline/src/python",
            "LD_LIBRARY_PATH=/baseline/venv/lib:/baseline/lib:/opt/cuda/lib64:/usr/lib",
        ],
    )
    candidate = validation(
        "/candidate",
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "GAFIME_WHEEL_PATH": "/candidate/wheels/gafime_cuda.whl",
            "GAFIME_CUDA_V1_LIB": "/candidate/lib/libgafime_cuda.so",
            "VIRTUAL_ENV": "/candidate/venv",
            "PATH": "/candidate/venv/bin:/candidate/bin:/usr/bin",
            "PYTHONPATH": "/candidate/src/python",
            "LD_LIBRARY_PATH": "/candidate/venv/lib:/candidate/lib:/opt/cuda/lib64:/usr/lib",
        },
    )

    assert perf13._native_environment_comparison_view(
        baseline
    ) == perf13._native_environment_comparison_view(candidate)

    candidate["environment"]["LD_LIBRARY_PATH"] = (
        "/candidate/lib:/opt/cuda-next/lib64:/usr/lib"
    )
    assert perf13._native_environment_comparison_view(
        baseline
    ) != perf13._native_environment_comparison_view(candidate)


def test_native_environment_rejects_unauthenticated_virtual_env_root() -> None:
    validation = {
        "environment": {
            "VIRTUAL_ENV": "/untrusted/venv",
            "PATH": "/untrusted/venv/bin:/usr/bin",
        },
        "provenance": {
            "python_executable": {
                "path": "/different/venv/bin/python",
                "sha256": "e" * 64,
                "size_bytes": 100,
            }
        },
    }

    assert perf13._native_environment_comparison_view(validation) is None


def test_native_environment_infers_authenticated_virtual_env_from_interpreter() -> None:
    def validation(root: str) -> dict[str, object]:
        return {
            "environment": {
                "PATH": f"{root}/venv/bin:/usr/bin",
                "PYTHONPATH": f"{root}/src/python:/opt/shared",
            },
            "source_root": f"{root}/src",
            "provenance": {
                "python_executable": {
                    "path": f"{root}/venv/bin/python",
                    "sha256": "e" * 64,
                    "size_bytes": 100,
                }
            },
        }

    baseline = perf13._native_environment_comparison_view(validation("/baseline"))
    candidate = perf13._native_environment_comparison_view(validation("/candidate"))

    assert baseline == candidate
    assert baseline is not None
    assert baseline["paths"]["PATH"].startswith("<virtual_env>/bin")


def test_native_environment_normalizes_windows_virtual_env_case() -> None:
    def validation(root: str, interpreter_root: str) -> dict[str, object]:
        return {
            "environment": {
                "VIRTUAL_ENV": root,
                "PATH": f"{root}\\Scripts;C:\\Windows\\System32",
            },
            "provenance": {
                "python_executable": {
                    "path": f"{interpreter_root}\\Scripts\\python.exe",
                    "sha256": "e" * 64,
                    "size_bytes": 100,
                }
            },
        }

    baseline = perf13._native_environment_comparison_view(
        validation("C:\\Baseline\\Venv", "c:\\baseline\\venv")
    )
    candidate = perf13._native_environment_comparison_view(
        validation("D:\\Candidate\\Venv", "d:\\candidate\\venv")
    )

    assert baseline == candidate


def test_native_environment_keeps_single_windows_search_paths_whole() -> None:
    def validation(drive: str, name: str) -> dict[str, object]:
        root = f"{drive}:\\{name}\\Venv"
        source = f"{drive}:\\{name}\\Source"
        return {
            "environment": {
                "PATH": f"{root}\\Scripts",
                "PYTHONPATH": f"{source}\\python",
            },
            "source_root": source,
            "provenance": {
                "python_executable": {
                    "path": f"{root}\\Scripts\\python.exe",
                    "sha256": "e" * 64,
                    "size_bytes": 100,
                }
            },
        }

    baseline = perf13._native_environment_comparison_view(validation("C", "Baseline"))
    candidate = perf13._native_environment_comparison_view(validation("D", "Candidate"))

    assert baseline == candidate
    assert baseline is not None
    assert baseline["paths"]["PATH"] == "<virtual_env>/scripts"
    assert baseline["paths"]["PYTHONPATH"] == "<source_root>/python"


def test_rocm_dynamic_snapshot_rejects_empty_power_file(tmp_path: Path) -> None:
    device = tmp_path / "card0" / "device"
    hwmon = device / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "device").write_text("0x150e\n")
    (device / "uevent").write_text("DRIVER=amdgpu\n")
    (hwmon / "power1_average").write_text("\n")

    snapshot = perf13._amd_sysfs_snapshot(dynamic=True, root=tmp_path)

    assert snapshot["status"] == "unavailable"
    assert perf13._rocm_dynamic_telemetry_fields(snapshot) == ()

    (device / "pp_dpm_sclk").write_text("0: 600Mhz\n1: 2200Mhz *\n")
    snapshot = perf13._amd_sysfs_snapshot(dynamic=True, root=tmp_path)

    assert snapshot["status"] == "pass"
    assert perf13._rocm_dynamic_telemetry_fields(snapshot)

    (device / "pp_dpm_sclk").unlink()
    (device / "power_dpm_state").write_text("performance\n")
    (device / "gpu_busy_percent").write_text("73\n")
    snapshot = perf13._amd_sysfs_snapshot(dynamic=True, root=tmp_path)

    assert snapshot["status"] == "unavailable"


def test_rocm_clock_power_gate_requires_nonempty_dynamic_field() -> None:
    empty_identity = {
        "status": "pass",
        "source": "rocm-smi",
        "output": json.dumps({"card0": {"Card series": "AMD Radeon"}}),
    }
    dynamic = {
        "status": "pass",
        "source": "rocm-smi",
        "output": json.dumps(
            {"card0": {"sclk clock speed:": "2200Mhz", "Card series": "AMD"}}
        ),
    }
    unsupported = {
        "status": "pass",
        "source": "rocm-smi",
        "output": json.dumps({"card0": {"Average Graphics Package Power (W)": "N/A"}}),
    }

    def payload(observation: dict[str, object]) -> dict[str, object]:
        phase = {
            "cpu_governor": {"status": "observed", "values": ["performance"]},
            "rocm_smi": observation,
        }
        return {
            "clock_and_power_capture_point": (
                perf13.GPU_NATIVE_CLOCK_POWER_CAPTURE_BOUNDARY
            ),
            "clock_and_power_state": {"before": phase, "after": phase},
        }

    assert any(
        "dynamic_clock_or_power_required" in failure
        for failure in perf13._gpu_clock_power_failures("rocm", payload(empty_identity))
    )
    assert any(
        "dynamic_clock_or_power_required" in failure
        for failure in perf13._gpu_clock_power_failures("rocm", payload(unsupported))
    )
    assert perf13._gpu_clock_power_failures("rocm", payload(dynamic)) == []


def test_public_environment_compares_extra_pythonpath_entries_exactly() -> None:
    def provenance(root: str) -> dict[str, object]:
        return {
            "source_root": f"{root}/src",
            "python_executable_identity": {
                "path": f"{root}/venv/bin/python",
                "sha256": "e" * 64,
                "size_bytes": 100,
            },
            "wheel_artifacts": [
                {
                    "path": f"{root}/wheels/gafime.whl",
                    "sha256": "f" * 64,
                    "size_bytes": 200,
                }
            ],
            "loaded_module_files": [],
        }

    baseline_provenance = provenance("/baseline")
    candidate_provenance = provenance("/candidate")
    baseline_environment = {
        "VIRTUAL_ENV": "/baseline/venv",
        "PATH": "/baseline/venv/bin:/usr/bin",
        "PYTHONPATH": "/baseline/src/python:/opt/shared",
    }
    candidate_environment = {
        "VIRTUAL_ENV": "/candidate/venv",
        "PATH": "/candidate/venv/bin:/usr/bin",
        "PYTHONPATH": "/candidate/src/python:/opt/shared",
    }

    assert "PATH" in perf13.RELEVANT_ENV_KEYS

    assert perf13._environment_comparison_view(
        baseline_environment, baseline_provenance
    ) == perf13._environment_comparison_view(
        candidate_environment, candidate_provenance
    )

    candidate_environment["PYTHONPATH"] = "/candidate/src/python:/opt/injected"
    assert perf13._environment_comparison_view(
        baseline_environment, baseline_provenance
    ) != perf13._environment_comparison_view(
        candidate_environment, candidate_provenance
    )

    candidate_environment["PYTHONPATH"] = "/candidate/src/python:/opt/shared"
    candidate_environment["PATH"] = "/candidate/venv/bin:/opt/injected/bin:/usr/bin"
    assert perf13._environment_comparison_view(
        baseline_environment, baseline_provenance
    ) != perf13._environment_comparison_view(
        candidate_environment, candidate_provenance
    )


def test_native_artifact_requires_python_executable_identity(tmp_path: Path) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload["provenance"].pop("python_executable")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-python-identity-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "missing_provenance_python_executable" in failure
        for failure in loaded["failures"]
    )


def test_core_native_artifact_requires_runtime_argv_bound_to_binary(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    payload.pop("command_line")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-core-command-line-manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "native_command_line_required" in failure for failure in loaded["failures"]
    )


def test_public_gpu_provenance_rejects_empty_successful_snapshot() -> None:
    result = {
        "kind": "public",
        "backend": "cuda",
        "status": "pass",
        "native_binaries": [{"sha256": "a" * 64}],
        "provenance": {
            "variant": "candidate",
            "source_commit": "a" * 40,
            "source_tree_state": {"status": "clean"},
            "wheel_artifacts": [{"sha256": "b" * 64}],
            "benchmark_script": {"sha256": "c" * 64},
            "benchmark_script_canonical": True,
            "wheel_runtime_binding": {"complete": True},
            "loaded_module_files": [{"sha256": "d" * 64}],
            "process_affinity": {"status": "observed", "cpus": [0]},
            "machine": "machine",
            "processor": "processor",
            "platform": "platform",
            "python_executable": sys.executable,
            "python_executable_identity": _identity(Path(sys.executable)),
            "python_version": sys.version,
            "environment": {},
            "runtime_dependencies": _runtime_dependencies(),
            "toolchains": {
                name: {"status": "pass"}
                for name in ("rustc", "cc", "cxx", "nvcc", "linker")
            },
            "clock_and_power_capture_point": (
                "before and after all timed benchmark regions"
            ),
            "clock_and_power_state": {
                phase: {
                    "cpu_governor": {
                        "status": "observed",
                        "values": ["performance"],
                    },
                    "nvidia_smi": {"status": "pass", "output": ""},
                }
                for phase in ("before", "after")
            },
            "device_identity": {"status": "pass", "output": "gpu"},
            "driver_command": "worker",
            "worker_command": "worker",
        },
    }

    readiness = perf13._provenance_readiness((result,))
    missing = set(readiness["failures"][0]["missing"])

    assert "nvidia_smi_before" in missing
    assert "nvidia_smi_after" in missing


def test_public_rocm_provenance_rejects_identity_or_placeholder_telemetry() -> None:
    observation = {
        "status": "pass",
        "source": "rocm-smi",
        "output": json.dumps(
            {
                "card0": {
                    "Card series": "AMD Radeon",
                    "Average Graphics Package Power (W)": "N/A",
                }
            }
        ),
    }
    phase = {
        "cpu_governor": {"status": "observed", "values": ["performance"]},
        "rocm_smi": observation,
    }
    result = {
        "kind": "public",
        "backend": "rocm",
        "status": "pass",
        "native_binaries": [{"sha256": "a" * 64}],
        "provenance": {
            "clock_and_power_capture_point": (
                "before and after all timed benchmark regions"
            ),
            "clock_and_power_state": {"before": phase, "after": phase},
        },
    }

    readiness = perf13._provenance_readiness((result,))
    missing = set(readiness["failures"][0]["missing"])

    assert "rocm_smi_before_dynamic_clock_or_power" in missing
    assert "rocm_smi_after_dynamic_clock_or_power" in missing


def test_native_decomposition_requires_validated_canonical_lifecycle(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(
        tmp_path / "native-decomposition", kind="native_decomposition"
    )
    manifest = tmp_path / "native-decomposition-manifest.json"
    _write_manifest(
        manifest,
        artifact,
        backend="cuda",
        kind="native_decomposition",
    )

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "native_decomposition_requires_validated_canonical_lifecycle" in failure
        for failure in loaded["failures"]
    )


def test_native_device_events_require_clock_and_synchronization_metadata(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-clocks", include_clocks=False)
    manifest = tmp_path / "missing-clocks-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "record_0_clock_and_synchronization_required" in failure
        for failure in loaded["failures"]
    )


def test_cuda_native_requires_authenticated_calibration_prepass(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-calibration-prepass")
    payload = json.loads(artifact.read_text())
    payload["calibration_prepass"]["included_in_profile_order_cycles"] = True
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-calibration-prepass-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "cuda_native_calibration_prepass_must_be_excluded_from_recorded_cycles"
        in failure
        for failure in loaded["failures"]
    )


def test_cuda_native_requires_post_prepass_clock_boundary(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "old-clock-boundary")
    payload = json.loads(artifact.read_text())
    payload["clock_and_power_capture_point"] = (
        "before and after all timed benchmark regions"
    )
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "old-clock-boundary-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert "clock_power_capture_boundary_required" in " ".join(loaded["failures"])


def test_cuda_native_binds_direct_kernel_binary_to_product_commit(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "linked-product")
    payload = json.loads(artifact.read_text())
    payload["linked_direct_kernel_product"]["commit"] = "f" * 40
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "linked-product-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "cuda_linked_direct_kernel_product_commit_mismatch" in failure
        for failure in loaded["failures"]
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("performed", False, "performed_required"),
        ("profile_order", ["mixed", "fp32", "fp64"], "profile_order_required"),
        ("records_discarded", 0, "records_discarded_required"),
        ("samples_discarded", 1, "sample_count_mismatch"),
        ("calibrated_key_count", 1, "key_coverage_mismatch"),
    ),
)
def test_cuda_native_calibration_prepass_metadata_is_authenticated(
    tmp_path: Path,
    field: str,
    value: object,
    reason: str,
) -> None:
    artifact = _cuda_artifact(tmp_path / f"bad-calibration-{field}")
    payload = json.loads(artifact.read_text())
    payload["calibration_prepass"][field] = value
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / f"bad-calibration-{field}-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        f"cuda_native_calibration_prepass_{reason}" in failure
        for failure in loaded["failures"]
    )


def test_native_ab_excludes_cells_with_different_fixed_loop_counts(
    tmp_path: Path,
) -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    artifacts = []
    for variant, loop_count in (("baseline", 1), ("candidate", 2)):
        artifact = tmp_path / f"loop-count-{variant}.json"
        artifact.write_text(
            json.dumps(
                {
                    "backend": "cuda",
                    "input_policy": "common-f64",
                    "input_identity": {"matrix_sha256": "1" * 64},
                    "records": [
                        {
                            "profile": "fp32",
                            "operation": "metric_kernel",
                            "metric": "pearson",
                            "evidence_lane": "canonical_payload_api",
                            "comparability": "within_abi_surface_only",
                            "samples_us": [100.0] * 30,
                            "loop_count_per_sample": loop_count,
                            "order_index": 0,
                            "profile_order": list(perf13.PROFILE_ORDER),
                            "clock": "cuda_event_stream_clock",
                            "timing_scope": "device_event",
                        }
                    ],
                }
            )
        )
        artifacts.append(
            {
                "variant": variant,
                "backend": "cuda",
                "path": str(artifact),
                "validation": {
                    "complete": True,
                    "performance_claim_ready": True,
                },
            }
        )
    evidence = {"valid": True, "artifacts": artifacts}

    comparisons = perf13._native_ab_comparisons(
        evidence, variants, bootstrap_resamples=25, seed=7
    )
    failures = perf13._native_ab_loop_count_failures(evidence, variants, ("cuda",))

    assert comparisons == []
    assert failures and failures[0]["reason"] == (
        "native_ab_loop_count_mismatch_incomparable"
    )


def test_cuda_native_provenance_requires_validator_relative_path(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-relative-path")
    payload = json.loads(artifact.read_text())
    payload["harness_source_blob"].pop("relative_path")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-relative-path-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "native_harness_relative_path_invalid" in failure
        for failure in loaded["failures"]
    )


def test_cuda_native_rejects_missing_runtime_argv_or_affinity(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-runtime-identity")
    payload = json.loads(artifact.read_text())
    payload.pop("command_line")
    payload.pop("process_affinity")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-runtime-identity-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    failures = " ".join(loaded["failures"])
    assert "native_command_line_required" in failures
    assert "process_affinity_provenance_required" in failures


def test_cuda_native_rejects_unbounded_precondition_batch(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "unbounded-precondition")
    payload = json.loads(artifact.read_text())
    payload["records"][0]["precondition_max_batch_iterations"] = 4097
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "unbounded-precondition-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "record_0_cuda_precondition_batch_bound_invalid" in failure
        for failure in loaded["failures"]
    )


def test_cuda_native_rejects_invalid_precondition_batch_cap(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "invalid-precondition-cap")
    payload = json.loads(artifact.read_text())
    payload["max_precondition_batch_iterations"] = 0
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "invalid-precondition-cap-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    failures = " ".join(loaded["failures"])
    assert "cuda_native_precondition_batch_cap_required" in failures
    assert "record_0_cuda_precondition_batch_bound_invalid" in failures


def test_cuda_native_rejects_per_sample_loop_rescaling(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "rescaled-loop-count")
    payload = json.loads(artifact.read_text())
    payload["records"][0]["loop_counts_per_sample"][0] = 2
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "rescaled-loop-count-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "record_0_cuda_fixed_loop_count_required" in failure
        for failure in loaded["failures"]
    )


def test_cuda_native_rejects_missing_sample_region_metadata(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-sample-region")
    payload = json.loads(artifact.read_text())
    payload.pop("sample_region_target_us")
    for record in payload["records"]:
        record.pop("sample_region_target_us")
        record.pop("sample_region_min_observed_us")
        record.pop("sample_region_target_met")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-sample-region-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    failures = " ".join(loaded["failures"])

    assert loaded["valid"] is False
    assert "cuda_native_sample_region_target_us_required" in failures
    assert "record_0_cuda_sample_region_target_metadata_required" in failures
    assert "record_0_cuda_sample_region_target_not_met" in failures
    assert "record_0_cuda_sample_region_minimum_invalid" in failures


def test_cuda_native_rejects_one_microsecond_sample_regions(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "one-microsecond-region")
    payload = json.loads(artifact.read_text())
    payload["sample_region_target_us"] = 1.0
    for record in payload["records"]:
        record["sample_region_target_us"] = 1.0
        record["sample_region_min_observed_us"] = 1.0
        record["sample_region_target_met"] = True
        record["raw_samples_us"] = [1.0] * 30
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "one-microsecond-region-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    failures = " ".join(loaded["failures"])

    assert loaded["valid"] is False
    assert "cuda_native_sample_region_target_us_required" in failures
    assert "record_0_cuda_sample_region_target_metadata_required" in failures
    assert "record_0_cuda_raw_region_below_declared_target" in failures
    assert "record_0_cuda_sample_region_minimum_invalid" in failures


def test_rocm_native_rejects_false_sample_region_target_marker(
    tmp_path: Path,
) -> None:
    artifact = _rocm_artifact(tmp_path / "false-sample-target")
    payload = json.loads(artifact.read_text())
    payload["records"][0]["sample_region_target_met"] = False
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "false-sample-target-manifest.json"
    _write_manifest(manifest, artifact, backend="rocm", kind="rocm_events")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "record_0_rocm_sample_region_target_not_met" in failure
        for failure in loaded["failures"]
    )


def test_cuda_native_rejects_raw_normalized_duration_mismatch(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "duration-normalization-mismatch")
    payload = json.loads(artifact.read_text())
    payload["records"][0]["samples_us"][0] -= 1.0
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "duration-normalization-mismatch-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert loaded["evidence_integrity_status"] == "invalid"
    assert any(
        "record_0_cuda_duration_normalization_mismatch" in failure
        for failure in loaded["failures"]
    )


def test_rocm_native_requires_bounded_precondition_and_fixed_loop_count(
    tmp_path: Path,
) -> None:
    artifact = _rocm_artifact(tmp_path / "rocm-fixed-methodology")
    manifest = tmp_path / "rocm-fixed-methodology-manifest.json"
    _write_manifest(manifest, artifact, backend="rocm", kind="rocm_events")

    loaded = perf13._load_native_evidence(str(manifest))
    assert loaded["valid"] is True, loaded["failures"]

    payload = json.loads(artifact.read_text())
    payload["max_precondition_batch_iterations"] = 0
    payload["calibration_policy"] = "per-sample-rescaling"
    payload["records"][0]["precondition_duration_us"] = 1.0
    payload["records"][0]["loop_counts_per_sample"][0] = 2
    artifact.write_text(json.dumps(payload))
    _write_manifest(manifest, artifact, backend="rocm", kind="rocm_events")

    loaded = perf13._load_native_evidence(str(manifest))
    failures = " ".join(loaded["failures"])
    assert loaded["valid"] is False
    assert "rocm_native_precondition_batch_cap_required" in failures
    assert "rocm_native_fixed_calibration_policy_required" in failures
    assert "record_0_rocm_same_cell_precondition_floor_not_met" in failures
    assert "record_0_rocm_fixed_loop_count_required" in failures


def test_cuda_native_artifact_requires_before_after_clock_power_state(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-power-state")
    payload = json.loads(artifact.read_text())
    payload.pop("clock_and_power_state")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-power-state-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "clock_and_power_state_required" in failure for failure in loaded["failures"]
    )


def test_native_gpu_artifact_requires_all_six_profile_orders(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-orders", include_orders=False)
    manifest = tmp_path / "missing-orders-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "all_six_native_profile_orders_required" in failure
        for failure in loaded["failures"]
    )
    failures = " ".join(loaded["failures"])
    assert "native_profile_order_cycles_required" in failures
    assert "native_record_order_cycle_coverage_required" in failures


def test_native_gpu_artifact_accepts_recorded_all_six_orders(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "complete-orders", include_orders=True)
    manifest = tmp_path / "complete-orders-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is True
    validation = loaded["artifacts"][0]["validation"]
    observed = validation["observed_profile_orders"]
    assert len(observed) == 6
    assert validation["native_order_sensitivity"]["status"] == (
        perf13.ORDER_EFFECT_CLEAN_STATUS
    )
    assert len(validation["native_order_sensitivity"]["profile_order_cycles"]) == (
        perf13.MIN_NATIVE_ORDER_REPETITIONS
    )
    assert (
        validation["native_order_sensitivity"]["distinct_profile_order_cycle_count"]
        >= 2
    )
    assert validation["evidence_integrity_status"] == "valid"
    assert validation["performance_claim_ready"] is True
    families = validation["native_order_claim_families"]["families"]
    assert families["supplemental_internal_kernel"]["status"] == (
        perf13.ORDER_EFFECT_CLEAN_STATUS
    )
    assert families["supplemental_internal_kernel"]["claim_ready"] is True
    assert families["supplemental_host_phase"]["status"] == (
        perf13.ORDER_EFFECT_CLEAN_STATUS
    )
    assert families["supplemental_host_phase"]["claim_ready"] is True
    assert families["canonical_payload_api"]["status"] == (
        perf13.ORDER_EFFECT_CLEAN_STATUS
    )
    assert families["canonical_payload_api"]["claim_ready"] is True


def test_native_order_inconclusive_preserves_structurally_valid_evidence(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "inconclusive-host-order")
    payload = json.loads(artifact.read_text())
    for record in payload["records"]:
        if (
            record["evidence_lane"] != "supplemental_host_phase"
            or record["operation"] != "planning"
            or record["profile"] != "fp64"
        ):
            continue
        cycle = record["order_index"] // 6
        position = record["profile_order"].index("fp64")
        value = 100.0 + (position * 10.0 if cycle < 12 else 0.0)
        raw_value = value * 64.0
        record["samples_us"] = [value] * 30
        record["raw_samples_us"] = [raw_value] * 30
        record["loop_count_per_sample"] = 64
        record["loop_counts_per_sample"] = [64] * 30
        record["sample_region_min_observed_us"] = raw_value
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "inconclusive-host-order-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    validation = loaded["artifacts"][0]["validation"]
    families = validation["native_order_claim_families"]["families"]

    assert loaded["valid"] is True
    assert loaded["status"] == "validated"
    assert loaded["evidence_integrity_status"] == "valid"
    assert loaded["failures"] == []
    assert loaded["performance_claim_ready"] is False
    assert loaded["arithmetic_claims_valid"] is False
    assert validation["complete"] is True
    assert validation["status"] == "pass"
    assert validation["performance_claim_ready"] is False
    assert families["supplemental_host_phase"]["status"] == (
        perf13.ORDER_EFFECT_INCONCLUSIVE_STATUS
    )
    assert families["supplemental_host_phase"]["claim_ready"] is False
    assert families["supplemental_internal_kernel"]["claim_ready"] is True
    assert any(
        "native_order_claim_family_supplemental_host_phase_not_claim_ready" in failure
        for failure in loaded["claim_failures"]
    )


def test_supplemental_contamination_does_not_invalidate_clean_canonical_family(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "supplemental-host-contamination")
    payload = json.loads(artifact.read_text())
    for record in payload["records"]:
        if (
            record["evidence_lane"] == "supplemental_host_phase"
            and record["operation"] == "planning"
            and record["profile"] == "fp64"
        ):
            position = record["profile_order"].index("fp64")
            value = 100.0 + position * 2.0
            raw_value = value * 64.0
            record["samples_us"] = [value] * 30
            record["raw_samples_us"] = [raw_value] * 30
            record["loop_count_per_sample"] = 64
            record["loop_counts_per_sample"] = [64] * 30
            record["sample_region_min_observed_us"] = raw_value
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "supplemental-host-contamination-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    validation = loaded["artifacts"][0]["validation"]
    family_summary = validation["native_order_claim_families"]
    families = family_summary["families"]

    assert loaded["valid"] is True
    assert loaded["evidence_integrity_status"] == "valid"
    assert loaded["failures"] == []
    assert loaded["arithmetic_claims_valid"] is False
    assert validation["complete"] is True
    assert validation["performance_claim_ready"] is False
    assert family_summary["claim_ready"] is False
    assert families["canonical_payload_api"]["status"] == (
        perf13.ORDER_EFFECT_CLEAN_STATUS
    )
    assert families["canonical_payload_api"]["claim_ready"] is True
    assert families["supplemental_internal_kernel"]["claim_ready"] is True
    assert families["supplemental_host_phase"]["status"] == (
        perf13.ORDER_EFFECT_CONTAMINATED_STATUS
    )
    assert families["supplemental_host_phase"]["claim_ready"] is False
    assert any(
        contrast["simultaneous_lower_absolute_bound_percent"] > 1.0
        for cell in families["supplemental_host_phase"]["cells"]
        for contrast in cell["position_contrasts"]
        if contrast["status"] == perf13.ORDER_EFFECT_CONTAMINATED_STATUS
    )


def test_unknown_native_evidence_lane_is_valid_but_never_claim_ready(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "unknown-evidence-lane")
    payload = json.loads(artifact.read_text())
    unclassified_count = 0
    for record in payload["records"]:
        if record["operation"] == "planning" and record["profile"] == "fp32":
            record["evidence_lane"] = "future_unreviewed_lane"
            unclassified_count += 1
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "unknown-evidence-lane-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    validation = loaded["artifacts"][0]["validation"]
    family_summary = validation["native_order_claim_families"]

    assert loaded["valid"] is True
    assert loaded["evidence_integrity_status"] == "valid"
    assert loaded["performance_claim_ready"] is False
    assert family_summary["claim_ready"] is False
    assert family_summary["unclassified_record_count"] == unclassified_count
    assert family_summary["unclassified_evidence_lanes"] == ["future_unreviewed_lane"]
    assert any(
        "native_order_evidence_lane_unclassified" in failure
        for failure in loaded["claim_failures"]
    )


def test_missing_predeclared_native_family_is_valid_but_never_claim_ready(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-canonical-family")
    payload = json.loads(artifact.read_text())
    payload["records"] = [
        record
        for record in payload["records"]
        if record["evidence_lane"] != "canonical_payload_api"
    ]
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-canonical-family-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    validation = loaded["artifacts"][0]["validation"]
    family_summary = validation["native_order_claim_families"]

    assert loaded["valid"] is True
    assert loaded["evidence_integrity_status"] == "valid"
    assert loaded["performance_claim_ready"] is False
    assert family_summary["missing_required_families"] == ["canonical_payload_api"]
    assert any(
        "native_order_claim_family_canonical_payload_api_not_claim_ready:not_present"
        in failure
        for failure in loaded["claim_failures"]
    )


def test_native_gpu_artifact_rejects_reused_complete_cycle_sequence(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "reused-order-cycle")
    payload = json.loads(artifact.read_text())
    canonical_orders = [
        list(order) for order in itertools.permutations(perf13.PROFILE_ORDER)
    ]
    payload["profile_order_cycles"] = [
        canonical_orders for _ in range(perf13.MIN_NATIVE_ORDER_REPETITIONS)
    ]
    for record in payload["records"]:
        record["profile_order"] = canonical_orders[record["order_index"] % 6]
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "reused-order-cycle-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    failures = " ".join(loaded["failures"])

    assert loaded["valid"] is False
    assert loaded["evidence_integrity_status"] == "invalid"
    assert "native_profile_order_cycle_variation_required" in failures
    assert "native_adjacent_profile_order_cycle_reuse_forbidden" in failures


def test_native_gpu_artifact_requires_machine_readable_order_seed(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-order-seed")
    payload = json.loads(artifact.read_text())
    payload.pop("order_seed")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-order-seed-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert loaded["evidence_integrity_status"] == "invalid"
    assert any(
        "native_order_seed_required" in failure for failure in loaded["failures"]
    )


def test_native_gpu_artifact_rejects_missing_cycle_schedule_metadata(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-order-cycle-metadata")
    payload = json.loads(artifact.read_text())
    payload.pop("order_schedule")
    payload.pop("order_repetitions")
    payload.pop("profile_order_cycles")
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-order-cycle-metadata-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))
    failures = " ".join(loaded["failures"])

    assert loaded["valid"] is False
    assert "native_deterministic_per_cycle_schedule_required" in failures
    assert "native_order_repetitions_required" in failures
    assert "native_profile_order_cycles_required" in failures


def test_native_gpu_artifact_cross_checks_declared_and_recorded_cycles(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "declared-recorded-cycle-mismatch")
    payload = json.loads(artifact.read_text())
    payload["profile_order_cycles"][0][0], payload["profile_order_cycles"][0][1] = (
        payload["profile_order_cycles"][0][1],
        payload["profile_order_cycles"][0][0],
    )
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "declared-recorded-cycle-mismatch-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "native_declared_record_order_cycles_mismatch" in failure
        for failure in loaded["failures"]
    )


def test_cuda_native_artifact_requires_separate_direct_stat_preparation_records(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "missing-stat-preparation")
    payload = json.loads(artifact.read_text())
    payload["records"] = [
        record
        for record in payload["records"]
        if record["operation"] != "feature_stat_preparation"
    ]
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "missing-stat-preparation-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any(
        "complete_decomposition_profile_coverage_required" in failure
        for failure in loaded["failures"]
    )
    incomplete = loaded["artifacts"][0]["validation"]["incomplete_profiles"]
    assert all("feature_stat_preparation" in missing for missing in incomplete.values())


def test_native_ab_key_retains_workload_order_clock_and_boundary(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "native-key-baseline"
    candidate_root = tmp_path / "native-key-candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline_artifact = _cuda_artifact(
        baseline_root, source_commit="a" * 40, include_orders=True
    )
    candidate_artifact = _cuda_artifact(
        candidate_root, source_commit="b" * 40, include_orders=True
    )
    baseline_manifest = tmp_path / "native-key-baseline-manifest.json"
    candidate_manifest = tmp_path / "native-key-candidate-manifest.json"
    _write_manifest(
        baseline_manifest,
        baseline_artifact,
        backend="cuda",
        variant="baseline",
        source_commit="a" * 40,
    )
    _write_manifest(
        candidate_manifest,
        candidate_artifact,
        backend="cuda",
        variant="candidate",
        source_commit="b" * 40,
    )
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    comparisons = perf13._native_ab_comparisons(
        evidence, variants, bootstrap_resamples=25, seed=7
    )

    assert comparisons
    assert all(
        json.loads(comparison["workload"]) == {"order_seed": 20260810}
        for comparison in comparisons
    )
    assert all(isinstance(comparison["order_index"], int) for comparison in comparisons)
    assert all(comparison["profile_order"] for comparison in comparisons)
    assert all(comparison["clock"] for comparison in comparisons)
    assert all(comparison["timing_boundary"] for comparison in comparisons)


def test_native_ab_comparisons_keep_only_claim_ready_order_families(
    tmp_path: Path,
) -> None:
    artifacts = []
    family_assessments = {
        "canonical_payload_api": {"claim_ready": True},
        "supplemental_internal_kernel": {"claim_ready": True},
        "supplemental_host_phase": {"claim_ready": False},
    }
    for variant, value in (("baseline", 100.0), ("candidate", 99.0)):
        artifact_path = tmp_path / f"{variant}-native.json"
        records = [
            {
                "profile": "mixed",
                "operation": operation,
                "metric": "pearson",
                "samples_us": [value] * 30,
                "order_index": 0,
                "profile_order": list(perf13.PROFILE_ORDER),
                "clock": "host_steady_clock",
                "timing_scope": lane,
                "evidence_lane": lane,
                "comparability": comparability,
            }
            for operation, lane, comparability in (
                (
                    "payload_execute",
                    "canonical_payload_api",
                    "within_abi_surface_only",
                ),
                (
                    "planning",
                    "supplemental_host_phase",
                    "supplemental_host_control",
                ),
            )
        ]
        artifact_path.write_text(
            json.dumps(
                {
                    "input_policy": "common-f64",
                    "input_identity": {"matrix_sha256": "1" * 64},
                    "records": records,
                }
            )
        )
        artifacts.append(
            {
                "variant": variant,
                "backend": "cuda",
                "path": str(artifact_path),
                "validation": {
                    "complete": True,
                    "performance_claim_ready": False,
                    "native_order_claim_families": {"families": family_assessments},
                },
            }
        )
    evidence = {
        "valid": True,
        "arithmetic_claims_valid": False,
        "artifacts": artifacts,
    }
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    comparisons = perf13._native_ab_comparisons(
        evidence, variants, bootstrap_resamples=25, seed=7
    )

    assert comparisons
    assert {comparison["measurement_category"] for comparison in comparisons} == {
        "canonical_payload_api"
    }
    assert all(
        comparison["operation"] == "payload_execute" for comparison in comparisons
    )


def test_provenance_gate_requires_toolchain_and_before_after_clock_records() -> None:
    result = {
        "kind": "public",
        "backend": "core",
        "native_binaries": [{"sha256": "a" * 64}],
        "provenance": {
            "variant": "candidate",
            "source_commit": "a" * 40,
            "source_tree_state": {"status": "clean"},
            "wheel_artifacts": [{"sha256": "a" * 64}],
            "benchmark_script": {"sha256": "a" * 64},
            "benchmark_script_canonical": True,
            "wheel_runtime_binding": {"complete": True},
            "loaded_module_files": [{"sha256": "a" * 64}],
            "process_affinity": {"status": "observed", "cpus": [0]},
            "machine": "machine",
            "processor": "processor",
            "platform": "platform",
            "python_executable": sys.executable,
            "python_version": "3",
            "environment": {},
        },
    }

    readiness = perf13._provenance_readiness((result,))
    missing = set(readiness["failures"][0]["missing"])

    assert readiness["complete"] is False
    assert "toolchains" in missing
    assert "clock_and_power_state" in missing


def test_provenance_gate_accepts_documented_unobservable_darwin_affinity() -> None:
    result = {
        "kind": "public",
        "backend": "metal",
        "provenance": {
            "variant": "candidate",
            "platform": "macOS-26.4-arm64-arm-64bit",
            "process_affinity": {
                "status": "unavailable",
                "cpus": None,
                "detail": "os.sched_getaffinity is unavailable on this platform",
            },
        },
    }

    readiness = perf13._provenance_readiness((result,))
    missing = set(readiness["failures"][0]["missing"])

    assert "observed_process_affinity" not in missing
    assert "nonempty_process_affinity" not in missing


def test_toolchain_snapshot_uses_apple_linker_version_flag(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def command_output(command: tuple[str, ...]) -> dict[str, object]:
        calls.append(command)
        if command == ("ld", "--version"):
            return {"status": "error", "output": "", "returncode": 1}
        return {"status": "pass", "output": "observed", "returncode": 0}

    monkeypatch.setattr(perf13.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(perf13, "_command_output", command_output)

    snapshot = perf13._toolchain_snapshot()

    assert ("ld", "--version") in calls
    assert ("ld", "-v") in calls
    assert snapshot["linker"]["status"] == "pass"


def test_comparative_input_gate_rejects_different_dataset_identities() -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    results = []
    for variant, wheel_hash, matrix_hash in (
        (variants[0], "a" * 64, "1" * 64),
        (variants[1], "b" * 64, "2" * 64),
    ):
        results.append(
            {
                "kind": "public",
                "status": "pass",
                "backend": "core",
                "input_policy": "common-f64",
                "workload": {"name": "small-latency"},
                "provenance": {
                    "variant": variant.name,
                    "source_commit": "a" * 40
                    if variant.name == "baseline"
                    else "b" * 40,
                    "wheel_artifacts": [{"sha256": wheel_hash}],
                    "machine": "machine",
                    "processor": "processor",
                    "platform": "platform",
                    "python_version": "3",
                    "benchmark_script": {"sha256": "f" * 64},
                    "environment": {},
                    "process_affinity": {"cpus": [0]},
                },
                "cells": [
                    {
                        "status": "pass",
                        "surface": "one_shot",
                        "profile": "fp32",
                        "input_identity": {
                            "matrix_sha256": matrix_hash,
                            "target_sha256": "3" * 64,
                            "feature_names_sha256": "4" * 64,
                        },
                    }
                ],
            }
        )

    readiness = perf13._comparative_input_readiness(results, variants)

    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "baseline_and_candidate_input_identity_mismatch"
        for failure in readiness["failures"]
    )


def test_comparative_gate_rejects_different_benchmark_script_hashes() -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    results = []
    for variant, source_commit, wheel_hash, script_hash in (
        (variants[0], "a" * 40, "a" * 64, "1" * 64),
        (variants[1], "b" * 40, "b" * 64, "2" * 64),
    ):
        results.append(
            {
                "kind": "public",
                "status": "pass",
                "provenance": {
                    "variant": variant.name,
                    "source_commit": source_commit,
                    "wheel_artifacts": [{"sha256": wheel_hash}],
                    "benchmark_script": {"sha256": script_hash},
                },
            }
        )

    readiness = perf13._comparative_input_readiness(results, variants)

    assert any(
        failure["reason"] == "baseline_and_candidate_benchmark_script_mismatch"
        for failure in readiness["failures"]
    )


def test_comparative_gate_accepts_isolated_paths_and_dynamic_clock_changes() -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    def result(
        variant: perf13.Variant, *, python: str, clock_output: str
    ) -> dict[str, object]:
        commit = "a" * 40 if variant.name == "baseline" else "b" * 40
        wheel = "1" * 64 if variant.name == "baseline" else "2" * 64
        snapshot = {"status": "pass", "output": clock_output}
        return {
            "kind": "public",
            "status": "pass",
            "backend": "cuda",
            "provenance": {
                "variant": variant.name,
                "source_commit": commit,
                "wheel_artifacts": [{"sha256": wheel}],
                "machine": "same-machine",
                "processor": "same-processor",
                "platform": "same-platform",
                "python_executable": python,
                "python_executable_identity": {
                    "path": python,
                    "size_bytes": 10,
                    "sha256": "e" * 64,
                },
                "python_version": "3.14",
                "benchmark_script": {"sha256": "f" * 64},
                "environment": {
                    "VIRTUAL_ENV": str(Path(python).parent.parent),
                    "OMP_NUM_THREADS": "1",
                },
                "runtime_dependencies": _runtime_dependencies(),
                "process_affinity": {"cpus": [0]},
                "device_identity": {
                    "status": "pass",
                    "output": "same-gpu,same-driver,same-bus",
                },
                "clock_and_power_state": {
                    "before": {
                        "cpu_governor": {
                            "status": "observed",
                            "values": ["performance"],
                        },
                        "nvidia_smi": snapshot,
                    },
                    "after": {
                        "cpu_governor": {
                            "status": "observed",
                            "values": ["performance"],
                        },
                        "nvidia_smi": snapshot,
                    },
                },
            },
        }

    readiness = perf13._comparative_input_readiness(
        [
            result(variants[0], python="/usr/bin/python3", clock_output="clock=100"),
            result(
                variants[1],
                python="/opt/candidate/bin/python",
                clock_output="clock=200",
            ),
        ],
        variants,
    )

    assert readiness["complete"] is True, readiness["failures"]


def test_comparative_gate_checks_every_backend_and_requires_gpu_identity() -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    results = []
    for backend in ("core", "cuda"):
        for variant in variants:
            clock_state = {
                "before": {
                    "cpu_governor": {"status": "observed", "values": ["performance"]}
                },
                "after": {
                    "cpu_governor": {"status": "observed", "values": ["performance"]}
                },
            }
            if backend == "cuda":
                clock_state["before"]["nvidia_smi"] = {"status": "pass", "output": "p8"}
                clock_state["after"]["nvidia_smi"] = {"status": "pass", "output": "p0"}
            results.append(
                {
                    "kind": "public",
                    "status": "pass",
                    "backend": backend,
                    "provenance": {
                        "variant": variant.name,
                        "source_commit": ("a" if variant.name == "baseline" else "b")
                        * 40,
                        "wheel_artifacts": [
                            {
                                "sha256": ("1" if variant.name == "baseline" else "2")
                                * 64
                            }
                        ],
                        "machine": "same-machine",
                        "processor": "same-processor",
                        "platform": "same-platform",
                        "python_version": "3.14",
                        "python_executable_identity": {
                            "path": f"/{variant.name}/python",
                            "size_bytes": 10,
                            "sha256": "e" * 64,
                        },
                        "benchmark_script": {"sha256": "f" * 64},
                        "environment": {},
                        "runtime_dependencies": _runtime_dependencies(),
                        "toolchains": {},
                        "process_affinity": {"cpus": [0]},
                        "device_identity": (
                            {"status": "not_applicable", "output": "CPU"}
                            if backend == "core"
                            else None
                        ),
                        "clock_and_power_state": clock_state,
                    },
                }
            )

    readiness = perf13._comparative_input_readiness(results, variants)

    assert readiness["complete"] is False
    assert any(
        failure.get("backend") == "cuda"
        and failure.get("reason") == "baseline_and_candidate_device_identity_required"
        for failure in readiness["failures"]
    )


def test_comparative_gate_rejects_interpreter_or_semantic_environment_mismatch() -> (
    None
):
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    def result(
        variant: perf13.Variant, *, digest: str, threads: str
    ) -> dict[str, object]:
        return {
            "kind": "public",
            "status": "pass",
            "backend": "core",
            "provenance": {
                "variant": variant.name,
                "source_commit": ("a" if variant.name == "baseline" else "b") * 40,
                "wheel_artifacts": [
                    {"sha256": ("1" if variant.name == "baseline" else "2") * 64}
                ],
                "machine": "same-machine",
                "processor": "same-processor",
                "platform": "same-platform",
                "python_version": "3.14",
                "python_executable_identity": {
                    "path": f"/{variant.name}/python",
                    "size_bytes": 10,
                    "sha256": digest,
                },
                "benchmark_script": {"sha256": "f" * 64},
                "environment": {"OMP_NUM_THREADS": threads},
                "process_affinity": {"cpus": [0]},
                "device_identity": {
                    "status": "not_applicable",
                    "output": "CPU backend",
                },
                "clock_and_power_state": {
                    "before": {
                        "cpu_governor": {
                            "status": "observed",
                            "values": ["performance"],
                        }
                    },
                    "after": {
                        "cpu_governor": {
                            "status": "observed",
                            "values": [
                                "powersave"
                                if variant.name == "candidate"
                                else "performance"
                            ],
                        }
                    },
                },
            },
        }

    readiness = perf13._comparative_input_readiness(
        [
            result(variants[0], digest="d" * 64, threads="1"),
            result(variants[1], digest="e" * 64, threads="2"),
        ],
        variants,
    )

    reasons = {failure["reason"] for failure in readiness["failures"]}
    assert "baseline_and_candidate_runtime_mismatch" in reasons
    assert "baseline_and_candidate_environment_mismatch" in reasons
    assert "cpu_governor_changed_during_worker" in reasons


def test_threshold_gate_rejects_unresolved_interleaved_control() -> None:
    readiness = perf13._threshold_readiness(
        [],
        [],
        interleaved_order_sensitivity=[{"status": "unacceptable_until_investigated"}],
    )

    assert readiness["complete"] is False
    assert readiness["failures"][0]["kind"] == "interleaved_order_sensitivity"


def test_cold_and_native_ab_comparisons_use_independent_samples(
    tmp_path: Path,
) -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    cold = perf13._cold_comparisons(
        [
            {
                "variant": "baseline",
                "backend": "core",
                "profile": "fp32",
                "input_policy": "common-f64",
                "workload": "small-latency",
                "raw_durations_ns": [100.0] * 30,
            },
            {
                "variant": "candidate",
                "backend": "core",
                "profile": "fp32",
                "input_policy": "common-f64",
                "workload": "small-latency",
                "raw_durations_ns": [101.0] * 30,
            },
        ],
        variants,
        bootstrap_resamples=25,
        seed=7,
    )
    assert cold[0]["pairing"] == "independent_worker_distributions"

    phase_results = []
    for variant, offset in (("baseline", 0), ("candidate", 1)):
        phase_results.append(
            {
                "kind": "cold",
                "status": "pass",
                "backend": "core",
                "profile": "fp32",
                "input_policy": "common-f64",
                "workload": {"name": "small-latency"},
                "cold_region_duration_ns": 100 + offset,
                "phases": {
                    "python_import": {
                        "status": "observed",
                        "duration_ns": 10.0 + offset,
                    },
                    "combined_phase": {
                        "status": "combined_not_separately_observable",
                        "duration_ns": None,
                    },
                },
                "provenance": {"variant": variant},
            }
        )
    phase_summaries = perf13._cold_summaries(
        phase_results, bootstrap_resamples=25, seed=7
    )
    phase_names = {summary["phase"] for summary in phase_summaries}
    assert phase_names == {"overall_cold_interval", "python_import"}
    phase_comparisons = perf13._cold_comparisons(
        phase_summaries,
        variants,
        bootstrap_resamples=25,
        seed=7,
    )
    assert {comparison["phase"] for comparison in phase_comparisons} == {
        "overall_cold_interval",
        "python_import",
    }

    baseline_root = tmp_path / "native-baseline"
    candidate_root = tmp_path / "native-candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline_artifact = _core_artifact(
        baseline_root, perf13.PROFILE_ORDER, source_commit="a" * 40
    )
    candidate_artifact = _core_artifact(
        candidate_root, perf13.PROFILE_ORDER, source_commit="b" * 40
    )
    baseline_manifest = tmp_path / "native-baseline-manifest.json"
    candidate_manifest = tmp_path / "native-candidate-manifest.json"
    _write_manifest(
        baseline_manifest,
        baseline_artifact,
        variant="baseline",
        source_commit="a" * 40,
    )
    _write_manifest(
        candidate_manifest,
        candidate_artifact,
        variant="candidate",
        source_commit="b" * 40,
    )
    native_evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )
    native = perf13._native_ab_comparisons(
        native_evidence,
        variants,
        bootstrap_resamples=25,
        seed=7,
    )

    assert native
    assert all(
        comparison["pairing"] == "independent_worker_distributions"
        for comparison in native
    )


def test_perf13_emits_independent_ab_delta_interval() -> None:
    variants = (
        perf13.Variant("baseline", "python", None, ()),
        perf13.Variant("candidate", "python", None, ()),
    )
    results = []
    for variant, values in ((variants[0], [100.0] * 30), (variants[1], [101.0] * 30)):
        results.append(
            {
                "kind": "public",
                "status": "pass",
                "backend": "core",
                "profile_order": ["fp32", "mixed", "fp64"],
                "input_policy": "common-f64",
                "ab_block": 0,
                "order_repeat": 0,
                "workload": {"name": "small-latency"},
                "provenance": {"variant": variant.name},
                "cells": [
                    {
                        "status": "pass",
                        "surface": "one_shot",
                        "profile": "fp32",
                        "distribution": {
                            "median_ns": values[0],
                            "raw_per_call_duration_ns": values,
                        },
                    }
                ],
            }
        )
    comparison = perf13._ab_comparisons(
        results, variants, bootstrap_resamples=25, seed=7
    )[0]
    assert comparison["effective_comparison_sample_count"] == 30
    assert comparison["pairing"] == "independent_worker_distributions"
    assert comparison["sample_count_baseline"] == 30
    assert comparison["sample_count_candidate"] == 30
    assert isinstance(
        comparison["bootstrap_candidate_latency_delta_95_ci_percent"], list
    )


def test_strict_evidence_json_rejects_nonfinite_constants() -> None:
    with pytest.raises(ValueError):
        perf13._strict_json_loads('{"duration": NaN}')
    with pytest.raises(ValueError):
        perf13._strict_json_loads('{"duration": Infinity}')
    with pytest.raises(ValueError):
        perf13._strict_json_loads('{"duration": -Infinity}')


def test_public_result_matrix_requires_exact_schedule_keys() -> None:
    schedule = {
        "kind": "public",
        "variant": "candidate",
        "backend": "core",
        "profile_order": ["fp32"],
        "surface_order": ["one_shot"],
        "workload": "small-latency",
        "input_policy": "common-f64",
        "order_repeat": 0,
        "order_index": 0,
        "ab_block": 0,
        "variant_sequence": ["candidate"],
        "interleaved_control": False,
    }
    result = {
        **schedule,
        "workload": perf13._workload_payload(perf13.WORKLOADS["small-latency"]),
        "status": "pass",
        "cells": [{"status": "pass", "surface": "one_shot", "profile": "fp32"}],
        "interleaved_controls": [],
    }
    readiness = perf13._result_matrix_readiness(
        [result, result],
        expected_public_result_count=1,
        expected_schedule=[schedule],
    )
    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "public_schedule_result_key_coverage_mismatch"
        for failure in readiness["failures"]
    )


def test_distribution_consistency_recomputes_declared_statistics() -> None:
    distribution = {
        "raw_per_call_duration_ns": [10.0, 20.0, 30.0],
        "median_ns": 999.0,
    }
    failures = perf13._distribution_consistency_failures(
        distribution, require_full=False
    )
    assert "median_ns_does_not_match_raw_samples" in failures
    assert "raw_per_call_duration_ns_must_be_finite_positive" not in failures

    nonfinite = dict(distribution)
    nonfinite["raw_per_call_duration_ns"] = [10.0, float("nan"), 30.0]
    assert (
        "raw_per_call_duration_ns_must_be_finite_positive"
        in perf13._distribution_consistency_failures(nonfinite, require_full=False)
    )


def test_parent_recomputes_expected_input_binding() -> None:
    workload = perf13._workload_payload(perf13.WORKLOADS["small-latency"])
    job = {"input_policy": "native", "seed": 1234}
    expected = perf13._expected_input_binding(job, precision="mixed", workload=workload)
    result = {"input_policy": "native", "seed": 1234, "workload": workload}
    cell = {
        "profile": "mixed",
        "input_binding": {**expected, "binding_sha256": "0" * 64},
        "input_identity": perf13._expected_input_identity_shape(expected),
    }
    failures = perf13._public_input_binding_failures(result, cell)
    assert any(
        failure["reason"] == "public_input_binding_mismatch" for failure in failures
    )


def test_regression_escalation_requires_ci_to_exclude_zero() -> None:
    inconclusive = perf13._comparison_classification(8.0, [-1.0, 12.0])
    assert inconclusive == {
        "review_status": "inconclusive_ci_crosses_zero",
        "confidence_interpretation": "no_direction_confirmed",
        "repeatable_regression": False,
        "escalation": "none_inconclusive",
    }
    one_percent = perf13._comparison_classification(2.0, [0.1, 3.8])
    assert one_percent["review_status"] == "confirmed_regression_above_one_percent"
    assert one_percent["escalation"] == "investigate"
    three_percent = perf13._comparison_classification(4.0, [0.2, 7.8])
    assert three_percent["review_status"] == "confirmed_regression_above_three_percent"
    assert three_percent["escalation"] == "maintainer_approval_required"
    improvement = perf13._comparison_classification(-2.0, [-3.0, -0.1])
    assert improvement["review_status"] == "confirmed_improvement"


def test_ci_crossing_zero_does_not_trigger_threshold_escalation() -> None:
    readiness = perf13._threshold_readiness(
        [],
        [
            {
                "sample_count_baseline": 30,
                "sample_count_candidate": 30,
                "candidate_latency_delta_percent": 8.0,
                "bootstrap_candidate_latency_delta_95_ci_percent": [-1.0, 12.0],
                "review_status": "inconclusive_ci_crosses_zero",
            }
        ],
    )
    assert readiness["complete"] is True


def test_threshold_gate_derives_classification_from_ci_not_declared_label() -> None:
    readiness = perf13._threshold_readiness(
        [],
        [
            {
                "sample_count_baseline": 30,
                "sample_count_candidate": 30,
                "candidate_latency_delta_percent": 4.0,
                "bootstrap_candidate_latency_delta_95_ci_percent": [2.0, 6.0],
                # A persisted artifact must not be able to suppress escalation
                # by supplying a label that contradicts its own bootstrap CI.
                "review_status": "inconclusive_ci_crosses_zero",
            }
        ],
    )

    assert readiness["complete"] is False
    statuses = {failure["status"] for failure in readiness["failures"]}
    assert "bootstrap_regression_classification_mismatch" in statuses
    assert "confirmed_regression_above_three_percent" in statuses


def _scheduled_native_manifests(
    tmp_path: Path,
    *,
    include_reverse: bool,
    schedule_in_manifest_only: bool = False,
    candidate_commit: str = "b" * 40,
) -> tuple[Path, Path]:
    artifacts_by_variant: dict[str, list[dict[str, object]]] = {
        "baseline": [],
        "candidate": [],
    }
    commits = (("baseline", "a" * 40), ("candidate", candidate_commit))
    for variant, commit in commits:
        root = tmp_path / variant
        root.mkdir()
        base_artifact = _core_artifact(root, perf13.PROFILE_ORDER, source_commit=commit)
        base_payload = json.loads(base_artifact.read_text())
        base_payload["workload"] = {
            "name": "native-small",
            "class": "native_backend_decomposition",
            "rows": 4096,
            "features": 8,
            "candidates": 8,
            "arity": 1,
            "mi_bins": 64,
            "top_k": 2,
            "metric_set": list(perf13.ALL_METRICS),
        }
        for block, sequence in (
            (0, ["baseline", "candidate"]),
            (1, ["candidate", "baseline"]),
        ):
            if block == 1 and not include_reverse:
                continue
            payload = dict(base_payload)
            payload["native_trial_id"] = f"{variant}-block-{block}"
            schedule = {
                "variant": variant,
                "ab_block": block,
                "variant_sequence": sequence,
                "input_policy": base_payload["input_policy"],
                "process_isolation": perf13.NATIVE_AB_PROCESS_ISOLATION,
            }
            if not schedule_in_manifest_only:
                payload.update(schedule)
            artifact = root / f"core-block-{block}.json"
            artifact.write_text(json.dumps(payload))
            artifact_entry = {
                "variant": variant,
                "backend": "core",
                "kind": "core_microbenchmark",
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "schedule": schedule,
            }
            artifacts_by_variant[variant].append(artifact_entry)
    manifests = []
    for variant, commit in commits:
        manifest = tmp_path / f"{variant}-scheduled-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": perf13.NATIVE_EVIDENCE_SCHEMA,
                    "status": "validated",
                    "arithmetic_claims_valid": True,
                    "source_commit": commit,
                    "artifacts": artifacts_by_variant[variant],
                }
            )
        )
        manifests.append(manifest)
    return manifests[0], manifests[1]


def test_native_ab_schedule_requires_fresh_ab_and_reversed_ba_blocks(
    tmp_path: Path,
) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=True
    )
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))

    assert readiness["complete"] is True
    assert len(readiness["schedule"]) == 4
    assert readiness["input_policy_coverage"] == {"core": ["common-f64"]}
    assert {tuple(entry["variant_sequence"]) for entry in readiness["schedule"]} == {
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    }
    for entry in readiness["schedule"]:
        assert entry["binary"]
        assert entry["wheel"]
        assert entry["harness"]
        assert entry["product"]
        assert entry["environment"] is not None
        assert entry["workload"]
        assert entry["input_identity"]
    comparisons = perf13._native_ab_comparisons(
        evidence, variants, bootstrap_resamples=25, seed=7
    )
    assert comparisons
    assert {comparison["ab_block"] for comparison in comparisons} == {0, 1}
    assert {tuple(comparison["variant_sequence"]) for comparison in comparisons} == {
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    }


def test_native_ab_schedule_rejects_missing_reversed_block(tmp_path: Path) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=False
    )
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )
    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))

    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "both_native_ab_and_ba_blocks_required"
        for failure in readiness["failures"]
    )


def test_native_ab_schedule_rejects_different_process_affinity(
    tmp_path: Path,
) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=True
    )
    candidate_payload = json.loads(candidate_manifest.read_text())
    for item in candidate_payload["artifacts"]:
        artifact = Path(item["path"])
        payload = json.loads(artifact.read_text())
        payload["process_affinity"] = [1]
        artifact.write_text(json.dumps(payload))
        item["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    candidate_manifest.write_text(json.dumps(candidate_payload))
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))

    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "native_process_affinity_mismatch"
        for failure in readiness["failures"]
    )


def test_native_ab_schedule_rejects_different_core_command_line(
    tmp_path: Path,
) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=True
    )
    candidate_payload = json.loads(candidate_manifest.read_text())
    for item in candidate_payload["artifacts"]:
        artifact = Path(item["path"])
        payload = json.loads(artifact.read_text())
        payload["command_line"].extend(["--rows", "8192"])
        artifact.write_text(json.dumps(payload))
        item["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    candidate_manifest.write_text(json.dumps(candidate_payload))
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))

    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "native_command_line_mismatch"
        for failure in readiness["failures"]
    )


def test_native_command_line_validation_and_normalization_are_fail_closed() -> None:
    assert perf13._native_command_line_view([]) is None
    assert perf13._native_command_line_view(["binary", "bad\x00argument"]) is None
    assert perf13._native_process_affinity_view({"allowed_cpus": []}) is None
    assert perf13._native_process_affinity_view("unobservable") is None

    baseline = [
        "/evidence/baseline/helper",
        "--payload",
        "/evidence/baseline/payload.so",
        "--wheel",
        "/evidence/baseline/payload.whl",
        "--source-root",
        "/source/baseline",
        "--source-commit",
        "a" * 40,
        "--metallib",
        "/evidence/baseline/gafime.metallib",
        "--shader-source",
        "/source/baseline/src/metal/shader.metal",
        "--canonical-evidence",
        "/evidence/baseline/canonical.json",
        "--harness-source-root",
        "/source/common-harness",
        "--harness-source-commit",
        "c" * 40,
        "--variant",
        "baseline",
        "--ab-block",
        "0",
        "--variant-sequence",
        "baseline,candidate",
        "--input-policy",
        "common-f64",
        "--rows",
        "4096",
        "--warmups",
        "10",
        "--repeats",
        "30",
        "--json",
        "/evidence/baseline.json",
    ]
    candidate = list(baseline)
    replacements = {
        "/evidence/baseline/helper": "/evidence/candidate/helper",
        "/evidence/baseline/payload.so": "/evidence/candidate/payload.so",
        "/evidence/baseline/payload.whl": "/evidence/candidate/payload.whl",
        "/source/baseline": "/source/candidate",
        "a" * 40: "b" * 40,
        "/evidence/baseline/gafime.metallib": "/evidence/candidate/gafime.metallib",
        "/source/baseline/src/metal/shader.metal": "/source/candidate/src/metal/shader.metal",
        "/evidence/baseline/canonical.json": "/evidence/candidate/canonical.json",
        "baseline": "candidate",
        "0": "1",
        "baseline,candidate": "candidate,baseline",
        "/evidence/baseline.json": "/evidence/candidate.json",
    }
    candidate = [replacements.get(argument, argument) for argument in candidate]

    baseline_view = perf13._native_command_line_comparison_view(baseline)
    candidate_view = perf13._native_command_line_comparison_view(candidate)
    assert baseline_view == candidate_view

    changed_workload = list(candidate)
    changed_workload[changed_workload.index("4096")] = "8192"
    assert (
        perf13._native_command_line_comparison_view(changed_workload) != baseline_view
    )

    changed_harness = list(candidate)
    changed_harness[changed_harness.index("/source/common-harness")] = "/source/other"
    assert perf13._native_command_line_comparison_view(changed_harness) != baseline_view

    changed_harness_commit = list(candidate)
    changed_harness_commit[changed_harness_commit.index("c" * 40)] = "d" * 40
    assert (
        perf13._native_command_line_comparison_view(changed_harness_commit)
        != baseline_view
    )


def test_native_ab_comparisons_use_hash_bound_manifest_schedule_metadata(
    tmp_path: Path,
) -> None:
    baseline_manifest, candidate_manifest = _scheduled_native_manifests(
        tmp_path, include_reverse=True, schedule_in_manifest_only=True
    )
    evidence = perf13._load_native_evidence_specs(
        [
            ("baseline", str(baseline_manifest)),
            ("candidate", str(candidate_manifest)),
        ]
    )
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    readiness = perf13._native_ab_schedule_readiness(evidence, variants, ("core",))
    comparisons = perf13._native_ab_comparisons(
        evidence, variants, bootstrap_resamples=25, seed=7
    )

    assert readiness["complete"] is True
    assert comparisons
    assert {comparison["ab_block"] for comparison in comparisons} == {0, 1}
    assert {tuple(comparison["variant_sequence"]) for comparison in comparisons} == {
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    }


def test_native_helpers_record_policy_schedule_and_profile_order_source_gates() -> None:
    cuda_source = (
        Path(__file__).parents[2] / "tests/gpu/cuda_precision_native_timing.cu"
    )
    rocm_source = Path(__file__).parents[2] / "tests/gpu/rocm_native_timing.cpp"
    metal_source = (
        Path(__file__).parents[2] / "tests/gpu/metal_precision_native_timing.mm"
    )
    cuda_text = cuda_source.read_text()
    cuda_cmake_text = (
        Path(__file__).parents[2] / "src/cuda/CMakeLists.txt"
    ).read_text()
    rocm_text = rocm_source.read_text()
    metal_text = metal_source.read_text()
    core_text = (
        Path(__file__).parents[2]
        / "tests/release_measure/core_precision_native_benchmark.rs"
    ).read_text()

    assert "canonical_requested" in cuda_text
    assert "GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT" in cuda_cmake_text
    assert "GAFIME_CUDA_BENCHMARK_LINKED_PRODUCT_COMMIT" in cuda_cmake_text
    assert "GAFIME_CUDA_BENCHMARK_PRODUCT_SOURCE_SHA256" in cuda_cmake_text
    assert "linked_direct_kernel_product" in cuda_text
    assert "all six canonical profile orders" in cuda_text
    assert "relative_path" in cuda_text
    assert "command_line" in cuda_text
    assert "command_line" in rocm_text
    assert "affinity" in rocm_text
    assert "fixed_loop_count_per_cell_no_per_sample_rescaling" in cuda_text
    assert "calibration_prepass" in cuda_text
    assert perf13.GPU_NATIVE_CLOCK_POWER_CAPTURE_BOUNDARY in cuda_text
    assert "calibration_prepass_records_discarded" in cuda_text
    assert "kPerRecordUntimedSameCellPreconditions" in cuda_text
    assert "kMaxPreconditionBatchIterations" in cuda_text
    assert "cudaEventElapsedTime(precondition)" in cuda_text
    assert "std::setprecision(12)" not in cuda_text
    assert cuda_text.count("std::setprecision(17)") >= 3
    assert "std::setprecision(12)" not in metal_text
    assert "std::numeric_limits<double>::max_digits10" in metal_text
    assert metal_text.count('", \\"loop_count_per_sample\\": "') == 1
    assert ".map(f64::to_string)" in core_text
    assert 'format!("{value:.17}")' not in core_text
    assert "fixed_loop_count_per_cell_no_per_sample_rescaling" in rocm_text
    assert "calibration_prepass" in rocm_text
    assert perf13.GPU_NATIVE_CLOCK_POWER_CAPTURE_BOUNDARY in rocm_text
    assert "kPerRecordUntimedSameCellPreconditions" in rocm_text
    assert "kMaxPreconditionBatchIterations" in rocm_text
    assert "hipEventElapsedTime(precondition)" in rocm_text
    for marker in (
        "--input-policy",
        "--variant-sequence",
        "fresh_helper_process_per_variant_trial",
    ):
        assert marker in cuda_text
        assert marker in rocm_text
    for marker in (
        "--input-policy",
        "common-f64",
        "native_fp32.v1",
        "GPUStartTime/GPUEndTime",
    ):
        assert marker in metal_text
    for operation in ("target_stat_preparation", "feature_stat_preparation"):
        assert operation in cuda_text
        assert operation in perf13.NATIVE_REQUIRED_OPERATIONS_BY_BACKEND["cuda"]
        assert operation in perf13.NATIVE_DEVICE_TIMED_OPERATIONS_BY_BACKEND["cuda"]
    assert (
        "payload-private target/feature-stat preparation separated from execute"
        in cuda_text
    )
    assert '\\"profile_order\\"' in rocm_text
    assert "order_repetitions" in cuda_text
    assert "order_repetitions" in rocm_text
    assert "kMinimumOrderRepetitions = 30" in cuda_text
    assert "kMinimumOrderRepetitions = 30" in rocm_text
    assert "supplemental_host_phase" in cuda_text
    assert "supplemental_host_phase" in rocm_text
    for order_schedule_marker in (
        "canonical_orders",
        "std::shuffle(orders.begin(), orders.end(), order_generator);",
        "previous_orders",
        "deterministic_per_cycle_shuffle_v1",
        "profile_order_cycles",
    ):
        assert order_schedule_marker in cuda_text
        assert order_schedule_marker in rocm_text


def test_native_order_sensitivity_detects_true_shift_with_lower_bound() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(perf13.MIN_NATIVE_ORDER_REPETITIONS):
        offset = cycle % len(orders)
        cycle_orders = orders[offset:] + orders[:offset]
        for order_index, order in enumerate(cycle_orders):
            for profile in perf13.PROFILE_ORDER:
                position = order.index(profile)
                value = 100.0 + position * 2.0
                records.append(
                    {
                        "profile": profile,
                        "order_index": cycle * 6 + order_index,
                        "profile_order": list(order),
                        "operation": "metric_kernel",
                        "metric": "pearson",
                        "samples_us": [value] * 30,
                        "raw_samples_us": [value * 30.0] * 30,
                        "loop_counts_per_sample": [30] * 30,
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=perf13.MIN_NATIVE_ORDER_REPETITIONS
    )

    assert sensitivity["status"] == perf13.ORDER_EFFECT_CONTAMINATED_STATUS
    assert sensitivity["raw_per_order_data"] is True
    contaminated = [
        contrast
        for cell in sensitivity["cells"]
        for contrast in cell["position_contrasts"]
        if contrast["status"] == perf13.ORDER_EFFECT_CONTAMINATED_STATUS
    ]
    assert contaminated
    assert (
        max(
            contrast["simultaneous_lower_absolute_bound_percent"]
            for contrast in contaminated
        )
        > 1.0
    )


def test_native_order_sensitivity_marks_heterogeneous_cycles_inconclusive() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(perf13.MIN_NATIVE_ORDER_REPETITIONS):
        offset = cycle % len(orders)
        cycle_orders = orders[offset:] + orders[:offset]
        for order_index, order in enumerate(cycle_orders):
            for profile in perf13.PROFILE_ORDER:
                value = 100.0
                if cycle < 12:
                    value += order.index(profile) * 10.0
                records.append(
                    {
                        "profile": profile,
                        "order_index": cycle * 6 + order_index,
                        "profile_order": list(order),
                        "operation": "metric_kernel",
                        "metric": "pearson",
                        "samples_us": [value] * 30,
                        "raw_samples_us": [value * 30.0] * 30,
                        "loop_counts_per_sample": [30] * 30,
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=perf13.MIN_NATIVE_ORDER_REPETITIONS
    )

    assert sensitivity["status"] == perf13.ORDER_EFFECT_INCONCLUSIVE_STATUS
    assert any(
        contrast["simultaneous_lower_absolute_bound_percent"]
        <= 1.0
        < contrast["simultaneous_upper_absolute_bound_percent"]
        for cell in sensitivity["cells"]
        for contrast in cell["position_contrasts"]
    )


def test_native_order_sensitivity_marks_uncertain_shift_inconclusive() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(perf13.MIN_NATIVE_ORDER_REPETITIONS):
        spread = 4.0 if cycle % 5 in (0, 4) else 0.5
        offset = cycle % len(orders)
        cycle_orders = orders[offset:] + orders[:offset]
        for order_index, order in enumerate(cycle_orders):
            for profile in perf13.PROFILE_ORDER:
                value = 100.0 + order.index(profile) * spread
                records.append(
                    {
                        "profile": profile,
                        "order_index": cycle * 6 + order_index,
                        "profile_order": list(order),
                        "operation": "target_update",
                        "metric": "spearman",
                        "samples_us": [value] * 30,
                        "raw_samples_us": [value * 30.0] * 30,
                        "loop_counts_per_sample": [30] * 30,
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=perf13.MIN_NATIVE_ORDER_REPETITIONS
    )

    assert sensitivity["status"] == perf13.ORDER_EFFECT_INCONCLUSIVE_STATUS
    assert sensitivity["decision_rule"].startswith("clean only when every")


def test_native_order_sensitivity_proves_simultaneous_equivalence() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(perf13.MIN_NATIVE_ORDER_REPETITIONS):
        offset = cycle % len(orders)
        cycle_orders = orders[offset:] + orders[:offset]
        for order_index, order in enumerate(cycle_orders):
            for profile in perf13.PROFILE_ORDER:
                # Cycle drift is shared by every position and therefore is not
                # misclassified as an order effect.
                value = 100.0 + cycle * 0.2
                records.append(
                    {
                        "profile": profile,
                        "order_index": cycle * 6 + order_index,
                        "profile_order": list(order),
                        "operation": "metric_kernel",
                        "metric": "mutual_info",
                        "samples_us": [value] * 30,
                        "raw_samples_us": [value * 16.0] * 30,
                        "loop_counts_per_sample": [16] * 30,
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=perf13.MIN_NATIVE_ORDER_REPETITIONS
    )

    assert sensitivity["status"] == perf13.ORDER_EFFECT_CLEAN_STATUS
    assert sensitivity["total_comparisons"] == 9
    assert sensitivity["multiple_comparison_correction"].startswith(
        "joint_max_standardized_complete_cycle_cluster_bootstrap"
    )
    assert "never treated as independent" in sensitivity["raw_sample_clustering"]
    assert all(
        contrast["simultaneous_upper_absolute_bound_percent"] <= 1.0
        for cell in sensitivity["cells"]
        for contrast in cell["position_contrasts"]
    )


def test_native_order_sensitivity_rejects_incomplete_cycle_and_variable_loops() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(perf13.MIN_NATIVE_ORDER_REPETITIONS):
        offset = cycle % len(orders)
        cycle_orders = orders[offset:] + orders[:offset]
        for order_index, order in enumerate(cycle_orders):
            for profile in perf13.PROFILE_ORDER:
                if (
                    cycle == perf13.MIN_NATIVE_ORDER_REPETITIONS - 1
                    and order_index == 5
                ):
                    continue
                loop_count = (
                    8 if cycle < perf13.MIN_NATIVE_ORDER_REPETITIONS - 1 else 16
                )
                records.append(
                    {
                        "profile": profile,
                        "order_index": cycle * 6 + order_index,
                        "profile_order": list(order),
                        "operation": "metric_kernel",
                        "metric": "r2",
                        "samples_us": [100.0] * 30,
                        "raw_samples_us": [100.0 * loop_count] * 30,
                        "loop_counts_per_sample": [loop_count] * 30,
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=perf13.MIN_NATIVE_ORDER_REPETITIONS
    )

    assert sensitivity["status"] == "insufficient_complete_order_cycle_evidence"
    assert sensitivity["incomplete_cells"]


def test_native_order_sensitivity_rejects_one_reused_cycle_sequence() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(perf13.MIN_NATIVE_ORDER_REPETITIONS):
        for order_index, order in enumerate(orders):
            for profile in perf13.PROFILE_ORDER:
                records.append(
                    {
                        "profile": profile,
                        "order_index": cycle * 6 + order_index,
                        "profile_order": list(order),
                        "operation": "metric_kernel",
                        "metric": "pearson",
                        "samples_us": [100.0] * 30,
                        "raw_samples_us": [3_000.0] * 30,
                        "loop_counts_per_sample": [30] * 30,
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=perf13.MIN_NATIVE_ORDER_REPETITIONS
    )

    assert sensitivity["status"] == "insufficient_complete_order_cycle_evidence"
    assert (
        "native_profile_order_cycle_variation_required"
        in sensitivity["order_cycle_schedule_failures"]
    )
    assert (
        "native_adjacent_profile_order_cycle_reuse_forbidden"
        in sensitivity["order_cycle_schedule_failures"]
    )


def test_missing_native_order_evidence_is_never_threshold_ready() -> None:
    sensitivity = perf13._native_order_sensitivity([], order_repetitions=None)
    readiness = perf13._threshold_readiness(
        [], [], native_order_sensitivity=[sensitivity]
    )

    assert sensitivity["status"] == "not_evaluated_order_repetitions_missing"
    assert readiness["complete"] is False
    assert readiness["failures"][0]["kind"] == "native_order_sensitivity"


def test_native_order_claim_requires_predeclared_thirty_complete_cycles() -> None:
    assert perf13.MIN_NATIVE_ORDER_REPETITIONS == 30

    sensitivity = perf13._native_order_sensitivity(
        [], order_repetitions=perf13.MIN_NATIVE_ORDER_REPETITIONS - 1
    )

    assert sensitivity["status"] == "insufficient_order_repeatability"
    assert sensitivity["required_order_repetitions"] == 30


def _public_order_results(
    position_effect: Callable[[int, int], float],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(5):
        for order_index, order in enumerate(orders):
            position = order.index("fp32")
            value = float(position_effect(cycle, position))
            results.append(
                {
                    "kind": "public",
                    "status": "pass",
                    "backend": "cuda",
                    "input_policy": "common-f64",
                    "workload": {"name": "small-latency"},
                    "profile_order": list(order),
                    "order_repeat": cycle,
                    "order_index": order_index,
                    "ab_block": 0,
                    "provenance": {"variant": "candidate"},
                    "cells": [
                        {
                            "status": "pass",
                            "surface": "compiled",
                            "profile": "fp32",
                            "profile_order_ordinal": position,
                            "distribution": {
                                "median_ns": value,
                                "raw_per_call_duration_ns": [value] * 30,
                            },
                        }
                    ],
                }
            )
    return results


def test_public_order_sensitivity_uses_same_equivalence_and_uncertainty_rule() -> None:
    clean = perf13._order_sensitivity(
        _public_order_results(lambda cycle, _position: 100.0 + cycle * 0.1)
    )
    inconclusive = perf13._order_sensitivity(
        _public_order_results(
            lambda cycle, position: 100.0 + (2.0 * position if cycle < 3 else 0.0)
        )
    )
    contaminated = perf13._order_sensitivity(
        _public_order_results(lambda _cycle, position: 100.0 + 2.0 * position)
    )

    assert clean[0]["status"] == perf13.ORDER_EFFECT_CLEAN_STATUS
    assert inconclusive[0]["status"] == perf13.ORDER_EFFECT_INCONCLUSIVE_STATUS
    assert inconclusive[0]["max_position_median_spread_percent"] > 1.0
    assert contaminated[0]["status"] == perf13.ORDER_EFFECT_CONTAMINATED_STATUS
    assert any(
        contrast["simultaneous_lower_absolute_bound_percent"]
        <= 1.0
        < contrast["simultaneous_upper_absolute_bound_percent"]
        for contrast in inconclusive[0]["position_contrasts"]
    )


def _interleaved_order_results() -> list[dict[str, object]]:
    orders = list(itertools.permutations(perf13.PROFILE_ORDER)) * 5
    profiles: dict[str, object] = {}
    for profile in perf13.PROFILE_ORDER:
        values = [
            100.0 + (10.0 * order.index(profile) if sample_index < 6 else 0.0)
            for sample_index, order in enumerate(orders)
        ]
        profiles[profile] = {
            "distribution": {
                "raw_per_call_duration_ns": values,
                "measured_repetitions": 30,
            }
        }
    return [
        {
            "kind": "public",
            "status": "pass",
            "backend": "rocm",
            "input_policy": "native",
            "workload": {"name": "medium-mixed-overhead"},
            "interleaved_controls": [
                {
                    "status": "pass",
                    "surface": "resident",
                    "profile_block_orders": [list(order) for order in orders],
                    "profiles": profiles,
                }
            ],
        }
    ]


def test_interleaved_order_sensitivity_preserves_six_order_cycle_clusters() -> None:
    results = _interleaved_order_results()

    summaries = perf13._interleaved_order_sensitivity(results)

    assert len(summaries) == 3
    assert all(
        summary["status"] == perf13.ORDER_EFFECT_INCONCLUSIVE_STATUS
        for summary in summaries
    )
    readiness = perf13._threshold_readiness(
        [], [], interleaved_order_sensitivity=summaries
    )
    assert readiness["complete"] is False


def test_interleaved_order_sensitivity_requires_all_backend_profiles() -> None:
    results = _interleaved_order_results()
    profiles = results[0]["interleaved_controls"][0]["profiles"]
    assert isinstance(profiles, dict)
    profiles.pop("fp64")

    summaries = perf13._interleaved_order_sensitivity(results)

    assert {summary["profile"] for summary in summaries} == set(
        perf13.BACKEND_PROFILES["rocm"]
    )
    assert all(
        summary["status"] == "insufficient_complete_order_cycle_evidence"
        for summary in summaries
    )


def test_metal_workflow_runs_typed_consumer_against_historical_baseline_only() -> None:
    workflow = (
        Path(__file__).parents[2]
        / ".github"
        / "workflows"
        / "metal_beast_benchmark.yml"
    ).read_text()

    assert "-DGAFIME_ABI_CONSUMER_ENABLE_TYPED_BASELINE=ON" in workflow
    assert (
        "installed-payload-baseline/gafime/_metal/libgafime_metal_v1.dylib" in workflow
    )
    assert "gafime_abi_1_1_typed_c_consumer_metal_baseline" in workflow
    typed_consumer_block = workflow.split(
        'baseline_payload_path="$METAL_RESULTS_DIR/installed-payload-baseline/', 1
    )[1].split("for input_policy in common-f64 native; do", 1)[0]
    assert '-DGAFIME_ABI_CONSUMER_PAYLOAD_PATH="$baseline_payload_path"' in (
        typed_consumer_block
    )
    assert "installed-payload-candidate/gafime/_metal/libgafime_metal_v1.dylib" not in (
        typed_consumer_block
    )


def test_native_operation_aliases_accept_explicit_payload_supplemental_records() -> (
    None
):
    expected = {
        "payload_allocation": "supplemental:payload_allocation",
        "payload_h2d_upload": "supplemental:payload_h2d_upload",
        "payload_update_target": "supplemental:payload_update_target",
        "payload_execution_memory_peak": "supplemental:payload_execution_memory_peak",
        "payload_execute": "supplemental:payload_execute",
        "target_update": "supplemental:target_update",
        "execution_memory_forecast": "supplemental:execution_memory_forecast",
    }

    assert perf13.NATIVE_SUPPLEMENTAL_OPERATION_ALIASES == expected
    for operation, canonical in expected.items():
        assert perf13._native_operation_names(operation, "pearson") == {canonical}
    assert (
        perf13._native_operation_names("payload_unvalidated_operation", "pearson")
        == set()
    )


def test_cuda_native_artifact_accepts_payload_supplemental_records(
    tmp_path: Path,
) -> None:
    artifact = _cuda_artifact(tmp_path / "payload-supplementals")
    payload = json.loads(artifact.read_text())
    templates = [
        record for record in payload["records"] if record["operation"] == "allocation"
    ]
    for operation in perf13.NATIVE_SUPPLEMENTAL_OPERATION_ALIASES:
        payload["records"].extend(
            {**template, "operation": operation} for template in templates
        )
    artifact.write_text(json.dumps(payload))
    manifest = tmp_path / "payload-supplementals-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is True
    validation = loaded["artifacts"][0]["validation"]
    for operation in perf13.NATIVE_SUPPLEMENTAL_OPERATION_ALIASES.values():
        assert operation in validation["operations_by_profile"]["fp32"]


def test_metal_native_route_gate_accepts_exact_typed_or_generic_surface_only() -> None:
    for surface in ("precision-typed-v1.1", "numeric-route-v2"):
        symbols = sorted(perf13.CANONICAL_ABI_SYMBOLS_BY_SURFACE[surface])
        payload = {
            "canonical_payload_lifecycle": {
                "status": "validated",
                "abi_surface": surface,
                "symbols": symbols,
            }
        }
        assert (
            perf13._native_payload_route_failures("metal", payload, kind="metal_events")
            == []
        )

        partial = json.loads(json.dumps(payload))
        partial["canonical_payload_lifecycle"]["symbols"].pop()
        assert perf13._native_payload_route_failures(
            "metal", partial, kind="metal_events"
        ) == ["native_generic_abi_route_unsupported"]

        mixed = json.loads(json.dumps(payload))
        other = (
            "numeric-route-v2"
            if surface == "precision-typed-v1.1"
            else "precision-typed-v1.1"
        )
        mixed["canonical_payload_lifecycle"]["symbols"].append(
            next(
                symbol
                for symbol in perf13.CANONICAL_ABI_SYMBOLS_BY_SURFACE[other]
                if symbol not in symbols
            )
        )
        assert perf13._native_payload_route_failures(
            "metal", mixed, kind="metal_events"
        ) == ["native_generic_abi_route_unsupported"]


def test_metal_input_policies_bind_distinct_source_and_fp32_execution_identity() -> (
    None
):
    common_identity = {
        "algorithm": "gafime.metal.native_timing.dataset.v2",
        "input_policy": "common-f64",
        "generator": "deterministic_integer_modulus.common_f64.v1",
        "source_dtype": "float64",
        "matrix_dtype": "float64",
        "target_dtype": "float64",
        "execution_dtype": "float32",
        "execution_matrix_dtype": "float32",
        "execution_target_dtype": "float32",
        "layout": "row_major",
        "matrix_sha256": "1" * 64,
        "target_sha256": "2" * 64,
        "execution_matrix_sha256": "3" * 64,
        "execution_target_sha256": "4" * 64,
    }

    assert perf13._metal_input_policy_failures("common-f64", common_identity) == []
    native_identity = dict(common_identity)
    native_identity.update(
        {
            "input_policy": "native",
            "generator": "deterministic_integer_modulus.native_fp32.v1",
            "source_dtype": "float32",
            "matrix_dtype": "float32",
            "target_dtype": "float32",
            "matrix_sha256": native_identity["execution_matrix_sha256"],
            "target_sha256": native_identity["execution_target_sha256"],
        }
    )
    assert perf13._metal_input_policy_failures("native", native_identity) == []


def _strict_lane_payload(
    lane: str,
    *,
    backend: str = "cuda",
    profiles: tuple[str, ...] = ("fp32",),
    payload_not_loaded: bool | None = None,
    payload_execution_mode: str | None = None,
    records_lane: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Small lane-only fixture for adversarial contract tests."""

    operations = sorted(perf13.NATIVE_LANE_REQUIRED_OPERATIONS[lane])
    if lane == "supplemental_internal_kernel" and backend == "rocm":
        operations = [
            operation
            for operation in operations
            if operation not in {"target_stat_preparation", "feature_stat_preparation"}
        ]
    raw_operations = {
        "supplemental:payload_execute": ("payload_execute", "pearson"),
    }
    records: list[dict[str, object]] = []
    for profile in profiles:
        for operation in operations:
            raw_operation, metric = raw_operations.get(
                operation,
                (
                    "metric_kernel",
                    operation.split(":", 1)[1]
                    if operation.startswith("metric:")
                    else "none",
                ),
            )
            if operation == "ranking_target_ranks":
                raw_operation, metric = "ranking_target_ranks", "spearman"
            elif operation == "ranking_topk":
                raw_operation, metric = "ranking_topk", "pearson"
            elif operation == "selected_row_gather":
                raw_operation, metric = "selected_row_gather", "pearson"
            elif operation == "candidate_materialization":
                raw_operation, metric = "candidate_materialization", "none"
            elif operation in {
                "target_stat_preparation",
                "feature_stat_preparation",
            }:
                raw_operation, metric = operation, "none"
            elif operation in {
                "ingest_conversion",
                "planning",
                "allocation",
                "h2d_upload",
                "d2h_transfer",
                "report_construction",
            }:
                raw_operation, metric = operation, "none"
            records.append(
                {
                    "profile": profile,
                    "operation": raw_operation,
                    "metric": metric,
                    "evidence_lane": records_lane or lane,
                }
            )
    payload: dict[str, object] = {
        "backend": backend,
        "evidence_lane": lane,
        "lane_isolation": perf13.NATIVE_LANE_ISOLATION,
        "execution_mode": {
            "canonical_payload_api": "canonical_payload",
            "supplemental_internal_kernel": "supplemental_internal_kernel",
            "supplemental_host_phase": "supplemental_host_phase",
        }[lane],
    }
    if lane == "canonical_payload_api":
        payload["canonical_payload_resolution"] = {
            "status": "resolved",
            "symbols": ["gafime_gpu_execute_v2"],
        }
    else:
        if payload_not_loaded is not None:
            payload["payload_not_loaded"] = payload_not_loaded
        if payload_execution_mode is not None:
            payload["payload_execution_mode"] = payload_execution_mode
    return payload, records


def test_native_lane_contract_rejects_missing_artifact_lane() -> None:
    payload, records = _strict_lane_payload("supplemental_internal_kernel")
    payload.pop("evidence_lane")
    failures, lane, _ = perf13._native_lane_contract_failures(
        payload, records, backend="cuda", profiles={"fp32"}
    )
    assert lane is None
    assert "native_evidence_lane_required_and_known" in failures


def test_native_lane_contract_rejects_cross_lane_records() -> None:
    payload, records = _strict_lane_payload("supplemental_internal_kernel")
    payload.update({"payload_not_loaded": True, "payload_execution_mode": "not_loaded"})
    records[0]["evidence_lane"] = "supplemental_host_phase"
    failures, lane, _ = perf13._native_lane_contract_failures(
        payload, records, backend="cuda", profiles={"fp32"}
    )
    assert lane == "supplemental_internal_kernel"
    assert any("record_0_evidence_lane_mismatch" in failure for failure in failures)


@pytest.mark.parametrize(
    "lane", ["supplemental_internal_kernel", "supplemental_host_phase"]
)
def test_native_lane_contract_rejects_payload_loaded_supplemental_artifacts(
    lane: str,
) -> None:
    payload, records = _strict_lane_payload(
        lane,
        payload_not_loaded=False,
        payload_execution_mode="canonical_payload_api",
    )
    payload["payload_loaded"] = True
    failures, _, _ = perf13._native_lane_contract_failures(
        payload, records, backend="cuda", profiles={"fp32"}
    )
    assert "native_supplemental_payload_not_loaded_required" in failures
    assert "native_supplemental_payload_loaded_forbidden" in failures
    assert "native_supplemental_payload_execution_marker_required" in failures
    assert "native_supplemental_payload_resolution_forbidden" not in failures


def test_native_lane_contract_accepts_external_canonical_lifecycle_binding() -> None:
    payload, records = _strict_lane_payload(
        "supplemental_internal_kernel",
        payload_not_loaded=True,
        payload_execution_mode="payload_not_loaded",
    )
    payload["canonical_payload_lifecycle"] = {
        "status": "validated",
        "schema": "gafime.native-decomposition.v1",
        "binding": "external_canonical_evidence",
        "path": "/evidence/canonical.json",
        "sha256": "a" * 64,
    }
    failures, lane, _ = perf13._native_lane_contract_failures(
        payload, records, backend="cuda", profiles={"fp32"}
    )
    assert lane == "supplemental_internal_kernel"
    assert "native_supplemental_payload_lifecycle_forbidden" not in failures


def test_perf13_source_enforces_external_canonical_lifecycle_reauthentication() -> None:
    source = Path(perf13.__file__).read_text(encoding="utf-8")

    assert (
        'canonical_lifecycle.get("binding") != "external_canonical_evidence"' in source
    )
    assert "canonical_payload_lifecycle_external_binding_required" in source
    assert "canonical_payload_lifecycle_absolute_path_required" in source
    assert "_sha256(lifecycle_file) != lifecycle_sha" in source
    assert "_strict_json_loads(" in source


def test_native_route_validation_is_strictly_lane_aware() -> None:
    canonical, _ = _strict_lane_payload("canonical_payload_api")
    canonical["canonical_payload_resolution"] = {
        "status": "resolved",
        "abi_surface": "numeric-route-v2",
        "symbols": sorted(perf13.CANONICAL_ABI_SYMBOLS_BY_SURFACE["numeric-route-v2"]),
    }
    assert (
        perf13._native_payload_route_failures("cuda", canonical, kind="cuda_events")
        == []
    )
    partial = json.loads(json.dumps(canonical))
    partial["canonical_payload_resolution"]["symbols"].pop()
    assert perf13._native_payload_route_failures(
        "cuda", partial, kind="cuda_events"
    ) == ["native_generic_abi_route_unsupported"]

    supplemental, _ = _strict_lane_payload(
        "supplemental_internal_kernel",
        payload_not_loaded=True,
        payload_execution_mode="payload_not_loaded",
    )
    supplemental["canonical_payload_lifecycle"] = {
        "status": "validated",
        "binding": "external_canonical_evidence",
        "path": "/evidence/canonical.json",
        "sha256": "a" * 64,
    }
    assert (
        perf13._native_payload_route_failures("cuda", supplemental, kind="cuda_events")
        == []
    )
    live_resolution = json.loads(json.dumps(supplemental))
    live_resolution["canonical_payload_resolution"] = {"status": "resolved"}
    assert perf13._native_payload_route_failures(
        "cuda", live_resolution, kind="cuda_events"
    ) == ["native_supplemental_live_payload_resolution_forbidden"]
    supplemental.pop("canonical_payload_lifecycle")
    assert perf13._native_payload_route_failures(
        "cuda", supplemental, kind="cuda_events"
    ) == ["native_supplemental_external_canonical_lifecycle_required"]


def _tracked_rocm_direct_sources(
    tmp_path: Path,
) -> tuple[Path, Path, str, dict[str, object]]:
    git = perf13._trusted_git_executable()
    assert git is not None
    environment, _ = perf13._git_environment()
    product = tmp_path / "product"
    harness = tmp_path / "harness"
    sources = (
        (product, "src/rocm/kernels.hip", b"// product kernels\n"),
        (product, "src/rocm/kernels.hpp", b"// product declarations\n"),
        (harness, "tests/gpu/rocm_native_direct_lane.hip", b"// direct lane\n"),
    )
    for root in (product, harness):
        root.mkdir()
        subprocess.run([str(git), "init", "-q", str(root)], check=True, env=environment)
    for root, relative, content in sources:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for root in (product, harness):
        subprocess.run(
            [str(git), "-C", str(root), "add", "."], check=True, env=environment
        )
        subprocess.run(
            [
                str(git),
                "-C",
                str(root),
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
            env=environment,
        )
    commit = perf13._git_commit(str(product))
    assert commit is not None
    source_blob = perf13._git_source_blob(product, product / "src/rocm/kernels.hip")
    assert source_blob["status"] == "tracked_at_head"
    return product, harness, commit, source_blob


def test_rocm_lane_authenticates_compiled_lane_and_direct_product_identity(
    tmp_path: Path,
) -> None:
    product_root, harness_root, source_commit, source_blob = (
        _tracked_rocm_direct_sources(tmp_path)
    )
    payload = {
        "source_root": str(product_root),
        "product_source_root": str(product_root),
        "harness_source_root": str(harness_root),
        "source_blob": source_blob,
        "compiled_lane": "supplemental_internal_kernel",
        "direct_kernel_product": {
            "compiled": True,
            "root": str(product_root),
            "commit": source_commit,
            "kernels_sha256": source_blob["source_sha256"],
            "kernels_header_sha256": perf13._sha256(
                product_root / "src/rocm/kernels.hpp"
            ),
            "direct_source_sha256": perf13._sha256(
                harness_root / "tests/gpu/rocm_native_direct_lane.hip"
            ),
        },
    }
    assert (
        perf13._rocm_direct_kernel_product_failures(
            payload,
            evidence_lane="supplemental_internal_kernel",
            source_commit=source_commit,
        )
        == []
    )

    adversaries = (
        (
            "compiled-lane",
            lambda value: value.__setitem__("compiled_lane", "canonical_payload_api"),
            "rocm_compiled_lane_evidence_lane_mismatch",
        ),
        (
            "compiled-marker",
            lambda value: value["direct_kernel_product"].__setitem__("compiled", False),
            "rocm_direct_kernel_product_compiled_marker_mismatch",
        ),
        (
            "root",
            lambda value: value["direct_kernel_product"].__setitem__("root", "/other"),
            "rocm_direct_kernel_product_root_mismatch",
        ),
        (
            "commit",
            lambda value: value["direct_kernel_product"].__setitem__(
                "commit", "b" * 40
            ),
            "rocm_direct_kernel_product_commit_mismatch",
        ),
        (
            "kernels",
            lambda value: value["direct_kernel_product"].__setitem__(
                "kernels_sha256", "invalid"
            ),
            "rocm_direct_kernel_product_kernels_sha256_required",
        ),
        (
            "header",
            lambda value: value["direct_kernel_product"].__setitem__(
                "kernels_header_sha256", "invalid"
            ),
            "rocm_direct_kernel_product_kernels_header_sha256_required",
        ),
        (
            "direct-source",
            lambda value: value["direct_kernel_product"].__setitem__(
                "direct_source_sha256", "invalid"
            ),
            "rocm_direct_kernel_product_direct_source_sha256_required",
        ),
        (
            "kernels-same-shape",
            lambda value: value["direct_kernel_product"].__setitem__(
                "kernels_sha256", "1" * 64
            ),
            "rocm_direct_kernel_product_kernels_source_blob_mismatch",
        ),
        (
            "header-same-shape",
            lambda value: value["direct_kernel_product"].__setitem__(
                "kernels_header_sha256", "2" * 64
            ),
            "rocm_direct_kernel_product_kernels_header_sha256_mismatch",
        ),
        (
            "direct-source-same-shape",
            lambda value: value["direct_kernel_product"].__setitem__(
                "direct_source_sha256", "3" * 64
            ),
            "rocm_direct_kernel_product_direct_source_sha256_mismatch",
        ),
        (
            "attested-source-blob",
            lambda value: value["source_blob"].__setitem__("source_sha256", "4" * 64),
            "rocm_direct_kernel_product_kernels_source_blob_mismatch",
        ),
    )
    for _, mutate, expected in adversaries:
        tampered = json.loads(json.dumps(payload))
        mutate(tampered)
        assert expected in perf13._rocm_direct_kernel_product_failures(
            tampered,
            evidence_lane="supplemental_internal_kernel",
            source_commit=source_commit,
        )

    for path, digest_field, expected in (
        (
            product_root / "src/rocm/kernels.hpp",
            "kernels_header_sha256",
            "rocm_direct_kernel_product_kernels_header_tracked_binding_required",
        ),
        (
            harness_root / "tests/gpu/rocm_native_direct_lane.hip",
            "direct_source_sha256",
            "rocm_direct_kernel_product_direct_source_tracked_binding_required",
        ),
    ):
        original = path.read_bytes()
        path.write_bytes(original + b"// dirty current bytes\n")
        dirty = json.loads(json.dumps(payload))
        dirty["direct_kernel_product"][digest_field] = perf13._sha256(path)
        assert expected in perf13._rocm_direct_kernel_product_failures(
            dirty,
            evidence_lane="supplemental_internal_kernel",
            source_commit=source_commit,
        )
        path.write_bytes(original)

    for control_lane in ("canonical_payload_api", "supplemental_host_phase"):
        control = {
            "source_root": str(product_root),
            "product_source_root": str(product_root),
            "harness_source_root": str(harness_root),
            "compiled_lane": control_lane,
            "direct_kernel_product": {
                "compiled": False,
                "root": "",
                "commit": "",
                "kernels_sha256": "",
                "kernels_header_sha256": "",
                "direct_source_sha256": "",
            },
        }
        assert (
            perf13._rocm_direct_kernel_product_failures(
                control,
                evidence_lane=control_lane,
                source_commit=source_commit,
            )
            == []
        )
        control["direct_kernel_product"]["root"] = str(product_root)
        assert "rocm_control_direct_kernel_product_identity_present" in (
            perf13._rocm_direct_kernel_product_failures(
                control,
                evidence_lane=control_lane,
                source_commit=source_commit,
            )
        )


def _strict_matrix_validation(
    lane: str,
    policy: str,
    block: int,
    index: int,
) -> dict[str, object]:
    operations = sorted(perf13.NATIVE_LANE_REQUIRED_OPERATIONS[lane])
    if lane == "supplemental_internal_kernel":
        # This synthetic matrix uses the CUDA contract.
        pass
    return {
        "complete": True,
        "lane_contract_active": True,
        "evidence_lane": lane,
        "profiles": list(perf13.PROFILE_ORDER),
        "input_policy": policy,
        "operations_by_profile_lane": {
            profile: operations for profile in perf13.PROFILE_ORDER
        },
        "operations_by_profile": {
            profile: operations for profile in perf13.PROFILE_ORDER
        },
        "runner_pid": 1000,
        "process_id": 2000 + index,
        "runner_invocation_id": f"{index + 1:032x}",
        "provenance": {},
    }


def _strict_matrix_artifacts() -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    index = 0
    for lane in perf13.NATIVE_ORDER_EVIDENCE_LANES:
        for policy in perf13.INPUT_POLICIES:
            for block in (0, 1):
                artifacts.append(
                    {
                        "variant": "candidate",
                        "backend": "cuda",
                        "kind": "cuda_events",
                        "path": f"/evidence/{index}.json",
                        "sha256": f"{index + 1:064x}",
                        "schedule": {
                            "evidence_lane": lane,
                            "input_policy": policy,
                            "ab_block": block,
                        },
                        "validation": _strict_matrix_validation(
                            lane, policy, block, index
                        ),
                    }
                )
                index += 1
    return artifacts


def _strict_matrix_evidence(
    artifacts: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "valid": True,
        "arithmetic_claims_valid": True,
        "artifacts": artifacts,
        "source_commits_by_variant": {},
    }


def _strict_matrix_readiness(artifacts: list[dict[str, object]]) -> dict[str, object]:
    return perf13._native_evidence_backend_readiness(
        _strict_matrix_evidence(artifacts),
        ("cuda",),
        (perf13.Variant("candidate", sys.executable, None, ()),),
    )


def test_native_backend_readiness_rejects_missing_lane_policy_block_cell() -> None:
    artifacts = _strict_matrix_artifacts()
    artifacts.pop(0)
    readiness = _strict_matrix_readiness(artifacts)
    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "native_lane_matrix_cell_missing"
        and failure["lane"] == "canonical_payload_api"
        and failure["input_policy"] == perf13.INPUT_POLICIES[0]
        and failure["ab_block"] == 0
        for failure in readiness["failures"]
    )


def test_native_backend_readiness_does_not_pool_operation_coverage_across_lanes() -> (
    None
):
    artifacts = _strict_matrix_artifacts()
    for artifact in artifacts:
        validation = artifact["validation"]
        if validation["evidence_lane"] != "canonical_payload_api":
            continue
        for profile in perf13.PROFILE_ORDER:
            validation["operations_by_profile_lane"][profile] = []
            validation["operations_by_profile"][profile] = []
    readiness = _strict_matrix_readiness(artifacts)
    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "native_lane_operation_coverage_incomplete"
        and "canonical_payload_api" in failure["missing_operations_by_lane_profile"]
        for failure in readiness["failures"]
    )


def test_native_backend_readiness_rejects_duplicate_lane_policy_block_cell() -> None:
    artifacts = _strict_matrix_artifacts()
    duplicate = dict(artifacts[0])
    duplicate["path"] = artifacts[0]["path"]
    duplicate["sha256"] = artifacts[0]["sha256"]
    duplicate["validation"] = dict(artifacts[0]["validation"])
    duplicate["validation"]["process_id"] = 9999
    artifacts.append(duplicate)
    readiness = _strict_matrix_readiness(artifacts)
    assert readiness["complete"] is False
    reasons = {failure["reason"] for failure in readiness["failures"]}
    assert "native_lane_matrix_cell_duplicate" in reasons
