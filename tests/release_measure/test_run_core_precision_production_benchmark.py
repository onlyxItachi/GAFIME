"""Focused contract tests for the Core production benchmark runner.

These are deliberately runner-only tests: they never compile product code or
execute the performance matrix.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).with_name("run_core_precision_production_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("gafime_core_production_runner", _SCRIPT)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _arguments(*extra: str) -> list[str]:
    return [
        "--product-source-root",
        "/product",
        "--harness-source-root",
        "/harness",
        "--product-rlib",
        "/product/libgafime_cpu-test.rlib",
        "--orchestrator-rlib",
        "/product/libgafime_orchestrator-test.rlib",
        "--types-rlib",
        "/product/libgafime_types-test.rlib",
        "--rayon-rlib",
        "/product/librayon-test.rlib",
        "--wheel",
        "/product/gafime-test.whl",
        "--binary",
        "/evidence/benchmark",
        "--output",
        "/evidence/report.json",
        *extra,
    ]


def test_defaults_preserve_full_profiles_metrics_matrix_and_scaling_labels() -> None:
    args = runner._parser().parse_args(_arguments())
    profiles, metrics, workloads, policies, workers = runner._validate_arguments(args)

    assert profiles == ("fp32", "mixed", "fp64")
    assert metrics == ("pearson", "spearman", "mutual_info", "r2")
    assert workloads == ("latency", "medium", "kernel")
    assert policies == ("common-f64", "native")
    assert workers == ("1", "2", "4", "default")


def test_stable_mode_rejects_candidate_only_sequence() -> None:
    args = runner._parser().parse_args(
        _arguments(
            "--mode",
            "stable",
            "--variant",
            "candidate",
            "--variant-sequence",
            "candidate",
        )
    )
    with pytest.raises(ValueError, match="candidate-only raw artifact"):
        runner._validate_arguments(args)


def test_stable_mode_accepts_ordered_two_variant_block() -> None:
    args = runner._parser().parse_args(
        _arguments(
            "--mode",
            "stable",
            "--variant",
            "candidate",
            "--variant-sequence",
            "baseline,candidate",
        )
    )
    runner._validate_arguments(args)


def test_stable_mode_rejects_reduced_matrix_but_informational_accepts_it() -> None:
    reduced = _arguments("--workloads", "latency", "--input-policies", "common-f64")
    informational = runner._parser().parse_args(reduced)
    assert runner._validate_arguments(informational)[2:4] == (
        ("latency",),
        ("common-f64",),
    )
    stable = runner._parser().parse_args(
        [
            *reduced,
            "--mode",
            "stable",
            "--variant-sequence",
            "baseline,candidate",
        ]
    )
    with pytest.raises(ValueError, match="complete canonical"):
        runner._validate_arguments(stable)


def test_scaling_plan_skips_oversubscribing_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _pid: {7})

    plan = runner._scaling_execution_plan(("1", "2", "4", "default"))

    assert plan.allowed_cpu_count == 1
    assert plan.executed_worker_modes == ("1", "default")
    assert {item["worker_mode"] for item in plan.skipped_worker_modes} == {"2", "4"}
    assert all(
        item["reason"] == "allowed_cpu_count_below_requested_workers"
        for item in plan.skipped_worker_modes
    )


def _valid_child(
    *, worker_mode: str, effective_workers: int, allowed: int
) -> dict[str, object]:
    worker_ids = list(range(effective_workers))
    snapshot: dict[str, object] = {
        "result_dtype": "f32",
        "row_count": 1,
        "max_arity": 1,
        "metric_count": 1,
        "result_flags": 0,
        "metric_ids": [1],
        "combo_indices": [0],
        "ranks": [0],
        "families": [0],
        "candidate_ids": [0],
        "row_flags": [0],
        "metric_value_bits": [0],
        "metric_value_text": ["0.0"],
        "metric_value_classes": ["finite"],
    }
    structural_digest, metric_digest = runner._snapshot_digests(snapshot)
    return {
        "profile": "fp32",
        "metric": "pearson",
        "input_policy": "common-f64",
        "workload": {"name": "latency"},
        "variant_sequence": ["baseline", "candidate"],
        "runner_pid": 10,
        "process_id": 11,
        "measurement_mode": "informational",
        "candidate_family_scope": "ranked_unary_candidates_only",
        "process_affinity": f"0-{allowed - 1}",
        "cell_schedule": {
            "index": 0,
            "seed": 7,
            "sha256": "a" * 64,
            "profile_order": ["fp32", "mixed", "fp64"],
        },
        "raw_samples_ns": [100_000_000],
        "samples_ns": [100_000_000.0],
        "loop_count_per_sample": 1,
        "target_region_ns": 100_000_000,
        "sample_region_min_observed_ns": 100_000_000,
        "sample_region_target_met": True,
        "environment": {"PATH": "/bin"},
        "result": {
            "rows_written": 1,
            "candidate_digest": structural_digest,
            "visible_score_bits": metric_digest,
            "digest_scope": (
                "all_visible_result_metadata_structural_arrays_and_metric_bits"
            ),
            "untimed_snapshot": snapshot,
        },
        "clock_and_power_capture_point": "before and after all timed benchmark regions",
        "clock_and_power_state": {"before": {}, "after": {}},
        "device": {"logical_cpu_count": 8, "physical_cpu_count": 4},
        "execution_topology": {
            "measurement_role": (
                "primary_default_worker_production_result"
                if worker_mode == "default"
                else "thread_scaling_diagnostic"
            ),
            "worker_mode": worker_mode,
            "effective_rayon_workers": effective_workers,
            "allowed_parallelism": allowed,
            "allowed_parallelism_source": "std::thread::available_parallelism",
            "process_affinity": f"0-{allowed - 1}",
            "process_affinity_cardinality": allowed,
            "affinity_matches_allowed_parallelism": True,
            "pool_start_worker_ids": worker_ids,
            "pool_start_worker_count": effective_workers,
            "pool_start_evidence_scope": (
                "dedicated_pool_construction_only_not_candidate_work_participation"
            ),
            "worker_os_cpu_ticks": [
                {
                    "worker_id": worker_id,
                    "os_tid": 100 + worker_id,
                    "cpu_ticks_before": 1,
                    "cpu_ticks_after": 2,
                    "work_ticks": 1,
                }
                for worker_id in worker_ids
            ],
            "worker_cpu_tick_status": "all_effective_workers_positive",
            "worker_cpu_ticks_observable": True,
            "every_effective_worker_positive_work_ticks": True,
        },
    }


def _schedule_entry(worker_mode: str = "default") -> object:
    return runner.CellScheduleEntry(
        schedule_index=0,
        input_policy="common-f64",
        workload="latency",
        metric="pearson",
        profile="fp32",
        worker_mode=worker_mode,
        profile_order=("fp32", "mixed", "fp64"),
        profile_order_ordinal=0,
    )


def _validate_child(child: dict[str, object], *, worker_mode: str) -> None:
    runner._validate_child_contract(
        child,
        worker_mode=worker_mode,
        variant_sequence=("baseline", "candidate"),
        runner_pid=10,
        mode="informational",
        schedule_entry=_schedule_entry(worker_mode),
        schedule_seed=7,
        schedule_sha256="a" * 64,
        repetitions=1,
        runner_allowed_cpu_count=4,
    )


def test_child_contract_requires_cpuset_aware_default_and_exact_explicit_workers() -> (
    None
):
    _validate_child(
        _valid_child(worker_mode="default", effective_workers=4, allowed=4),
        worker_mode="default",
    )
    explicit = _valid_child(worker_mode="2", effective_workers=2, allowed=4)
    entry = _schedule_entry("2")
    explicit["cell_schedule"]["profile_order"] = list(entry.profile_order)
    _validate_child(explicit, worker_mode="2")

    with pytest.raises(RuntimeError, match="full allowed CPU set"):
        _validate_child(
            _valid_child(worker_mode="default", effective_workers=2, allowed=4),
            worker_mode="default",
        )
    with pytest.raises(RuntimeError, match="exactly its requested Rayon count"):
        _validate_child(
            _valid_child(worker_mode="4", effective_workers=2, allowed=4),
            worker_mode="4",
        )


def test_child_contract_requires_pool_start_count_to_match_effective_workers() -> None:
    child = _valid_child(worker_mode="2", effective_workers=2, allowed=4)
    topology = child["execution_topology"]
    assert isinstance(topology, dict)
    topology["pool_start_worker_count"] = 1

    with pytest.raises(RuntimeError, match="pool-start evidence"):
        _validate_child(child, worker_mode="2")


def test_child_environment_scrubs_inherited_rayon_global_override(
    tmp_path: Path,
) -> None:
    identity = runner.RepositoryIdentity(
        root=tmp_path,
        commit="a" * 40,
        tree="b" * 40,
        git_dir=tmp_path,
        git_common_dir=tmp_path,
    )
    source = runner.SourceIdentity(
        path=tmp_path / "source.rs",
        relative_path=Path("source.rs"),
        sha256="c" * 64,
        git_blob="d" * 40,
    )
    file_identity = runner.FileIdentity(tmp_path / "artifact", "e" * 64, 1)
    environment = runner._child_environment(
        base={"RAYON_NUM_THREADS": "99", "PATH": "/bin"},
        product=identity,
        harness=identity,
        source=source,
        runner_source=source,
        product_rlib=file_identity,
        orchestrator_rlib=file_identity,
        types_rlib=file_identity,
        rayon_rlib=file_identity,
        wheel=file_identity,
        binary=file_identity,
        output=tmp_path / "child.json",
        profile="fp32",
        metric="pearson",
        workload="latency",
        policy="common-f64",
        workers="default",
        warmups=10,
        repetitions=30,
        variant="candidate",
        ab_block=0,
        variant_sequence=("candidate",),
        runner_invocation_id="x",
        runner_pid=10,
        mode="informational",
        schedule_index=0,
        schedule_seed=7,
        schedule_sha256="a" * 64,
        profile_order=("fp32", "mixed", "fp64"),
    )

    assert "RAYON_NUM_THREADS" not in environment
    assert environment["GAFIME_PRODUCTION_RUNNER_PID"] == "10"


def test_benchmark_environment_preserves_real_tool_path_but_scrubs_git_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/opt/rust/bin:/usr/bin")
    monkeypatch.setenv("GIT_DIR", "/attacker")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")

    environment = runner._benchmark_environment()

    assert environment["PATH"] == "/opt/rust/bin:/usr/bin"
    assert "GIT_DIR" not in environment
    assert "GIT_CONFIG_COUNT" not in environment


def test_schedule_is_balanced_shared_within_block_and_varies_across_blocks() -> None:
    kwargs = {
        "profiles": ("fp32", "mixed", "fp64"),
        "metrics": ("pearson", "spearman", "mutual_info", "r2"),
        "workloads": ("latency", "medium", "kernel"),
        "policies": ("common-f64", "native"),
        "worker_modes": ("1", "2", "4", "default"),
        "seed": 91,
    }
    first, first_seed, first_hash, first_counts = runner._cell_schedule(
        **kwargs, ab_block=0
    )
    replay, replay_seed, replay_hash, replay_counts = runner._cell_schedule(
        **kwargs, ab_block=0
    )
    second, second_seed, second_hash, _ = runner._cell_schedule(**kwargs, ab_block=1)

    assert first == replay
    assert (first_seed, first_hash, first_counts) == (
        replay_seed,
        replay_hash,
        replay_counts,
    )
    assert max(first_counts.values()) - min(first_counts.values()) <= 1
    assert first_seed != second_seed
    assert first_hash != second_hash
    assert first != second


def test_child_contract_rejects_runner_pid_reuse_and_forged_normalization() -> None:
    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    child["process_id"] = 10
    with pytest.raises(RuntimeError, match="process identity"):
        _validate_child(child, worker_mode="default")

    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    child["samples_ns"] = [99_000_000.0]
    with pytest.raises(RuntimeError, match="normalized timings"):
        _validate_child(child, worker_mode="default")


def test_stable_child_requires_every_worker_to_have_positive_cpu_ticks() -> None:
    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    child["measurement_mode"] = "stable"
    topology = child["execution_topology"]
    assert isinstance(topology, dict)
    topology["worker_os_cpu_ticks"][2]["work_ticks"] = 0
    topology["every_effective_worker_positive_work_ticks"] = False
    topology["worker_cpu_tick_status"] = "observable_but_one_or_more_workers_zero"

    with pytest.raises(RuntimeError, match="positive Linux CPU ticks"):
        runner._validate_child_contract(
            child,
            worker_mode="default",
            variant_sequence=("baseline", "candidate"),
            runner_pid=10,
            mode="stable",
            schedule_entry=_schedule_entry(),
            schedule_seed=7,
            schedule_sha256="a" * 64,
            repetitions=1,
            runner_allowed_cpu_count=4,
        )


def test_child_contract_rejects_python_rust_affinity_disagreement() -> None:
    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    with pytest.raises(RuntimeError, match="Python affinity-derived"):
        runner._validate_child_contract(
            child,
            worker_mode="default",
            variant_sequence=("baseline", "candidate"),
            runner_pid=10,
            mode="informational",
            schedule_entry=_schedule_entry(),
            schedule_seed=7,
            schedule_sha256="a" * 64,
            repetitions=1,
            runner_allowed_cpu_count=3,
        )


def test_child_contract_binds_timed_digest_to_every_snapshot_array() -> None:
    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    snapshot = child["result"]["untimed_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["row_flags"] = [1]
    with pytest.raises(RuntimeError, match="authenticate untimed snapshot"):
        _validate_child(child, worker_mode="default")

    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    snapshot = child["result"]["untimed_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["result_flags"] = 1
    with pytest.raises(RuntimeError, match="authenticate untimed snapshot"):
        _validate_child(child, worker_mode="default")

    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    snapshot = child["result"]["untimed_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["metric_ids"] = [2]
    with pytest.raises(RuntimeError, match="full result snapshot"):
        _validate_child(child, worker_mode="default")


def test_child_contract_rejects_text_or_class_not_derived_from_timed_bits() -> None:
    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    snapshot = child["result"]["untimed_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["metric_value_text"] = ["1.0"]
    with pytest.raises(RuntimeError, match="full result snapshot"):
        _validate_child(child, worker_mode="default")

    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    snapshot = child["result"]["untimed_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["metric_value_classes"] = ["nan"]
    with pytest.raises(RuntimeError, match="full result snapshot"):
        _validate_child(child, worker_mode="default")


def test_static_clock_power_view_ignores_samples_but_preserves_policy() -> None:
    def phase(current: str) -> dict[str, object]:
        return {
            "cpu_governor": "performance",
            "policy_clock_state": [
                {
                    "policy": "policy0",
                    "scaling_cur_freq_khz": current,
                    "scaling_min_freq_khz": "400000",
                    "scaling_max_freq_khz": "5100000",
                    "cpuinfo_min_freq_khz": "400000",
                    "cpuinfo_max_freq_khz": "5100000",
                    "energy_performance_preference": "performance",
                }
            ],
            "platform_power_profile": "performance",
            "macos_pmset_custom": None,
            "power_interface": "linux-cpufreq",
        }

    view = runner._static_clock_power_view(
        {"before": phase("4700000"), "after": phase("5050000")}
    )

    assert view["before"] == view["after"]
    policy = view["before"]["policy_clock_state"][0]
    assert "scaling_cur_freq_khz" not in policy
    assert policy["scaling_max_freq_khz"] == "5100000"


def test_static_clock_power_view_detects_policy_drift() -> None:
    before = {
        "cpu_governor": "performance",
        "policy_clock_state": [
            {"policy": "policy0", "scaling_max_freq_khz": "5100000"}
        ],
        "platform_power_profile": "performance",
        "macos_pmset_custom": None,
        "power_interface": "linux-cpufreq",
    }
    after = {**before, "platform_power_profile": "balanced"}

    view = runner._static_clock_power_view({"before": before, "after": after})

    assert view["before"] != view["after"]


def test_canonical_environment_is_nonempty_and_rayon_is_scrubbed() -> None:
    child = _valid_child(worker_mode="default", effective_workers=4, allowed=4)
    canonical = runner._canonical_child_environment(child)

    assert canonical["PATH"] == "/bin"
    assert canonical["RAYON_NUM_THREADS"] == "<scrubbed>"
    child["environment"] = {}
    with pytest.raises(RuntimeError, match="nonempty PATH"):
        runner._canonical_child_environment(child)


def test_compiler_command_links_exact_product_and_route_dependencies(
    tmp_path: Path,
) -> None:
    command = runner._compiler_command(
        rustup="rustup",
        toolchain="1.97.1",
        source=tmp_path / "production.rs",
        product_rlib=tmp_path / "libgafime_cpu-product.rlib",
        orchestrator_rlib=tmp_path / "libgafime_orchestrator-product.rlib",
        types_rlib=tmp_path / "libgafime_types-product.rlib",
        rayon_rlib=tmp_path / "librayon-product.rlib",
        dependency_dirs=(tmp_path / "deps",),
        binary=tmp_path / "benchmark",
    )

    assert "cargo" not in command
    assert "-Copt-level=3" in command
    assert "-Clto=fat" in command
    for expected in (
        "gafime_cpu=",
        "gafime_orchestrator=",
        "gafime_types=",
        "rayon=",
    ):
        assert any(expected in item for item in command)


def test_manifest_uses_portable_path_within_one_evidence_root(tmp_path: Path) -> None:
    artifact = tmp_path / "aggregate.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    product = runner.RepositoryIdentity(
        root=tmp_path,
        commit="a" * 40,
        tree="b" * 40,
        git_dir=tmp_path,
        git_common_dir=tmp_path,
    )

    runner._write_manifest(
        manifest,
        output=runner._file_identity(artifact),
        product=product,
        variant="candidate",
        ab_block=0,
        variant_sequence=("baseline", "candidate"),
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifacts"][0]["path"] == "aggregate.json"
    assert payload["artifacts"][0]["relative_path"] == "aggregate.json"


def test_leaf_and_production_claim_boundaries_are_structurally_distinct() -> None:
    leaf = (
        Path(__file__)
        .with_name("core_precision_native_benchmark.rs")
        .read_text(encoding="utf-8")
    )
    production = (
        Path(__file__)
        .with_name("core_precision_production_benchmark.rs")
        .read_text(encoding="utf-8")
    )

    assert "gafime.core-leaf-kernel-diagnostic.v1" in leaf
    assert "eligible_for_core_production_throughput" in leaf
    assert "supplemental_single_core_leaf_kernel_diagnostic" in leaf
    assert "gafime.core-production-executor.child.v1" in production
    assert "production_executor_metric" in production
    assert (
        "planner_protocol_resident_precision_compute_backend_ranked_result"
        in production
    )
    assert "pool_start_evidence_scope" in production


def test_workflow_uses_the_compatible_before_fix_precision_head_and_tracks_the_runner() -> (
    None
):
    workflow = (
        Path(__file__).parents[2]
        / ".github"
        / "workflows"
        / "core_precision_production_benchmark.yml"
    ).read_text(encoding="utf-8")

    assert "d52199f44aa80ab8ef50c18db95dd1630961cdaf" in workflow
    assert (
        "github.event.pull_request.base.sha || inputs.expected_baseline_sha"
        not in workflow
    )
    assert 'live.get("base", {}).get("ref") != "main"' in workflow
    assert 'git", "merge-base", "--is-ancestor", baseline, candidate' in workflow
    assert (
        "tests/release_measure/run_core_precision_production_benchmark.py" in workflow
    )
    assert (
        "tests/release_measure/test_run_core_precision_production_benchmark.py"
        in workflow
    )
    assert "tests/release_measure/test_perf_13_precision_profiles.py" in workflow
    assert "tests/release_measure/v1_architecture_gate.py" in workflow
    assert "gafime-core-stable" in workflow
    assert (
        'fromJSON(\'["self-hosted","linux","x64","gafime-core-stable"]\')' in workflow
    )
    assert "run_variant 0 baseline baseline,candidate" in workflow
    assert "run_variant 1 candidate candidate,baseline" in workflow
    assert "maturin build --release --locked" in workflow
    assert "'maturin==1.9.6'" in workflow
    assert (
        workflow.count('RUSTUP_TOOLCHAIN="$RUST_VERSION" CARGO_TARGET_DIR="$target"')
        == 2
    )
    assert (
        "precision_executor_parallelism_contract_covers_every_profile_and_rank_mode"
        in workflow
    )
    assert '--binary "$CORE_PRODUCTION_RESULTS/core-production-$variant"' in workflow
    assert (
        '--binary "$CORE_PRODUCTION_RESULTS/core-production-$variant-ab$block"'
        not in workflow
    )
