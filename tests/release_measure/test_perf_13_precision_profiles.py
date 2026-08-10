"""Adversarial unit tests for the perf13 native-evidence gate."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import zipfile

import pytest


_SCRIPT = Path(__file__).with_name("perf_13_precision_profiles.py")
_SPEC = importlib.util.spec_from_file_location("gafime_perf13", _SCRIPT)
assert _SPEC and _SPEC.loader
perf13 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = perf13
_SPEC.loader.exec_module(perf13)


def _identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


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
        kind = (
            "core_microbenchmark"
            if backend == "core"
            else f"{backend}_events"
        )
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
    records = []
    for profile in profiles:
        for metric in perf13.ALL_METRICS:
            records.append(
                {
                    "profile": profile,
                    "operation": "metric_kernel",
                    "metric": metric,
                    "samples_us": [1.0] * 30,
                }
            )
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
                "repeats": 30,
                "target_region_ns": 5_000_000,
                "measurement_scope": "native_arithmetic_only",
                "decomposition_boundaries": {"candidate_materialization": "fused"},
                "compiler": {"rustc": "rustc-test"},
                "device": {"kind": "cpu", "identity": "test-cpu"},
                "process_affinity": [0],
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
    }
    artifact_payload.update(
        _native_harness_fields(
            source, product_commit=source_commit, runner=runner
        )
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
    orders = list(itertools.permutations(perf13.PROFILE_ORDER)) if include_orders else []
    for profile in perf13.PROFILE_ORDER:
        for operation in host_operations + device_operations:
            record: dict[str, object] = {
                "profile": profile,
                "operation": operation,
                "metric": "none",
                "samples_us": [1.0] * 30,
            }
            if include_clocks:
                record.update(
                    {
                        "clock": (
                            "cuda_event_stream_clock"
                            if operation in device_operations
                            else "host_steady_clock"
                        ),
                        "synchronization": (
                            "cuda_event_synchronize"
                            if operation in device_operations
                            else "host_monotonic"
                        ),
                        "timing_scope": (
                            "device_event"
                            if operation in device_operations
                            else "host_only"
                        ),
                    }
                )
            if orders:
                order_index = len(records) % len(orders)
                record.update(
                    {
                        "order_index": order_index,
                        "profile_order": list(orders[order_index]),
                    }
                )
            records.append(record)
        for metric in perf13.ALL_METRICS:
            record = {
                "profile": profile,
                "operation": "metric_kernel",
                "metric": metric,
                "samples_us": [1.0] * 30,
            }
            if include_clocks:
                record.update(
                    {
                        "clock": "cuda_event_stream_clock",
                        "synchronization": "cuda_event_synchronize",
                        "timing_scope": "device_event",
                    }
                )
            if orders:
                order_index = len(records) % len(orders)
                record.update(
                    {
                        "order_index": order_index,
                        "profile_order": list(orders[order_index]),
                    }
                )
            records.append(record)

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
                "source_tree_state": {"status": "clean", "entries": []},
                "input_policy": "common-f64",
                "input_identity": {
                    "matrix_sha256": "1" * 64,
                    "target_sha256": "2" * 64,
                    "feature_names_sha256": "3" * 64,
                },
                "warmups": 10,
                "repeats": 30,
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
                "environment": {},
                "clock": {"host": "steady_clock", "device": "cudaEvent"},
                "clock_and_power_capture_point": (
                    "before and after all timed benchmark regions"
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
                "profile_orders": (
                    [list(order) for order in orders]
                ),
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


def test_perf13_rejects_arbitrary_hash_only_native_file(tmp_path: Path) -> None:
    artifact = tmp_path / "arbitrary.json"
    artifact.write_text('{"looks_like":"native evidence"}\n')
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, artifact)

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is False
    assert any("backend_schema_mismatch" in failure for failure in loaded["failures"])


def test_perf13_requires_complete_profile_union_not_kind_intersection(tmp_path: Path) -> None:
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


def test_perf13_rejects_core_report_claim_from_arithmetic_fixture(tmp_path: Path) -> None:
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
        **{key: artifact[key] for key in (
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
        )},
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


def _canonical_cuda_lifecycle(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
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
    assert perf13._canonical_lifecycle_failures(
        lifecycle,
        backend="cuda",
        source_commit="a" * 40,
        artifact_provenance=provenance,
    ) == []

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
    typed_operations = sorted(
        perf13.CANONICAL_ABI_TYPED_LIFECYCLE_OPERATIONS
    )
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
    assert perf13._canonical_lifecycle_failures(
        lifecycle,
        backend="cuda",
        source_commit="a" * 40,
        artifact_provenance=provenance,
    ) == []

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
    artifact = _core_artifact(
        tmp_path, perf13.PROFILE_ORDER, source_commit="a" * 40
    )
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

    assert perf13._native_payload_route_failures(
        "rocm", payload, kind="rocm_events"
    ) == []

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

    readiness = perf13._native_evidence_backend_readiness(
        evidence, ("core",), variants
    )

    assert readiness["complete"] is False
    assert any(
        failure.get("reason")
        == "baseline_and_candidate_native_harness_mismatch"
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
    readiness = perf13._native_evidence_backend_readiness(
        evidence, ("core",), variants
    )

    assert readiness["complete"] is False
    assert any(
        failure.get("reason")
        == "baseline_and_candidate_core_harness_runner_mismatch"
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
    assert len(sample_statistics["raw_durations"]) == 30
    assert {"median", "mad", "p05", "p95", "bootstrap_median_95_ci"} <= set(
        sample_statistics
    )
    assert sample_statistics["auto_scaling"]["status"] == (
        "not_observed_in_native_artifact"
    )


def test_native_statistics_use_normalized_samples_not_calibration_regions(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    for record in payload["records"]:
        record["raw_samples_us"] = [100.0] * 30
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
    assert statistics["normalized_durations"] == [1.0] * 30
    assert statistics["raw_durations"] == [100.0] * 30


def test_core_native_statistics_accept_normalized_and_raw_nanoseconds(
    tmp_path: Path,
) -> None:
    artifact = _core_artifact(tmp_path, perf13.PROFILE_ORDER)
    payload = json.loads(artifact.read_text())
    for record in payload["records"]:
        record.pop("samples_us", None)
        record["samples_ns"] = [1_000_000] * 30
        record["raw_samples_ns"] = [5_000_000] * 30
        record["loop_count_per_sample"] = 5
        record["sample_region_target_ns"] = 5_000_000
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
    assert statistics["raw_durations"] == [5_000_000.0] * 30
    assert statistics["auto_scaling"]["target_unit"] == "ns"


def test_release_claim_sample_floor_is_hard_100ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["perf_13_precision_profiles.py", "--native-evidence", "manifest.json", "--min-sample-ms", "99.999"],
    )
    with pytest.raises(SystemExit) as exc_info:
        perf13._parse_args()
    assert exc_info.value.code == 2


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
        artifact = _core_artifact(root, perf13.PROFILE_ORDER, source_commit=("a" if variant.name == "baseline" else "b") * 40)
        payload = json.loads(artifact.read_text())
        for record in payload["records"]:
            record["samples_us"] = [normalized] * 30
            record["raw_samples_us"] = [100.0] * 30
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
    assert perf13._native_ab_comparisons(
        evidence, variants, bootstrap_resamples=25, seed=7
    ) == []


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
    assert any("clean_source_tree_required" in failure for failure in loaded["failures"])
    assert any("compiler_rustc_version_required" in failure for failure in loaded["failures"])


def test_rocm_compiler_validation_allows_optional_null_but_requires_observed_tools() -> None:
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

    baseline = perf13._native_environment_comparison_view(
        validation("C", "Baseline")
    )
    candidate = perf13._native_environment_comparison_view(
        validation("D", "Candidate")
    )

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
        "output": json.dumps(
            {"card0": {"Average Graphics Package Power (W)": "N/A"}}
        ),
    }

    def payload(observation: dict[str, object]) -> dict[str, object]:
        phase = {
            "cpu_governor": {"status": "observed", "values": ["performance"]},
            "rocm_smi": observation,
        }
        return {
            "clock_and_power_capture_point": (
                "before and after all timed benchmark regions"
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

    candidate_environment["PYTHONPATH"] = (
        "/candidate/src/python:/opt/injected"
    )
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
    artifact = _cuda_artifact(tmp_path / "native-decomposition", kind="native_decomposition")
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
        "clock_and_power_state_required" in failure
        for failure in loaded["failures"]
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


def test_native_gpu_artifact_accepts_recorded_all_six_orders(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "complete-orders", include_orders=True)
    manifest = tmp_path / "complete-orders-manifest.json"
    _write_manifest(manifest, artifact, backend="cuda")

    loaded = perf13._load_native_evidence(str(manifest))

    assert loaded["valid"] is True
    observed = loaded["artifacts"][0]["validation"]["observed_profile_orders"]
    assert len(observed) == 6


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
    assert all(
        "feature_stat_preparation" in missing for missing in incomplete.values()
    )


def test_native_ab_key_retains_workload_order_clock_and_boundary(tmp_path: Path) -> None:
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
    assert all(comparison["workload"] == "{}" for comparison in comparisons)
    assert all(isinstance(comparison["order_index"], int) for comparison in comparisons)
    assert all(comparison["profile_order"] for comparison in comparisons)
    assert all(comparison["clock"] for comparison in comparisons)
    assert all(comparison["timing_boundary"] for comparison in comparisons)


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
                    "source_commit": "a" * 40 if variant.name == "baseline" else "b" * 40,
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

    def result(variant: perf13.Variant, *, python: str, clock_output: str) -> dict[str, object]:
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
            result(variants[1], python="/opt/candidate/bin/python", clock_output="clock=200"),
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
                        "source_commit": (
                            "a" if variant.name == "baseline" else "b"
                        )
                        * 40,
                        "wheel_artifacts": [
                            {
                                "sha256": (
                                    "1" if variant.name == "baseline" else "2"
                                )
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
        and failure.get("reason")
        == "baseline_and_candidate_device_identity_required"
        for failure in readiness["failures"]
    )


def test_comparative_gate_rejects_interpreter_or_semantic_environment_mismatch() -> None:
    variants = (
        perf13.Variant("baseline", sys.executable, None, ()),
        perf13.Variant("candidate", sys.executable, None, ()),
    )

    def result(variant: perf13.Variant, *, digest: str, threads: str) -> dict[str, object]:
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
        interleaved_order_sensitivity=[
            {"status": "unacceptable_until_investigated"}
        ],
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
    tmp_path: Path, *, include_reverse: bool, schedule_in_manifest_only: bool = False
) -> tuple[Path, Path]:
    artifacts_by_variant: dict[str, list[dict[str, object]]] = {
        "baseline": [],
        "candidate": [],
    }
    for variant, commit in (("baseline", "a" * 40), ("candidate", "b" * 40)):
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
            }
            if schedule_in_manifest_only:
                artifact_entry["schedule"] = schedule
            artifacts_by_variant[variant].append(artifact_entry)
    manifests = []
    for variant, commit in (("baseline", "a" * 40), ("candidate", "b" * 40)):
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
    assert {
        tuple(entry["variant_sequence"]) for entry in readiness["schedule"]
    } == {("baseline", "candidate"), ("candidate", "baseline")}
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
    assert {
        tuple(comparison["variant_sequence"]) for comparison in comparisons
    } == {("baseline", "candidate"), ("candidate", "baseline")}


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
    assert {
        tuple(comparison["variant_sequence"]) for comparison in comparisons
    } == {("baseline", "candidate"), ("candidate", "baseline")}


def test_native_helpers_record_policy_schedule_and_profile_order_source_gates() -> None:
    cuda_source = Path(__file__).parents[2] / "tests/gpu/cuda_precision_native_timing.cu"
    rocm_source = Path(__file__).parents[2] / "tests/gpu/rocm_native_timing.cpp"
    metal_source = Path(__file__).parents[2] / "tests/gpu/metal_precision_native_timing.mm"
    cuda_text = cuda_source.read_text()
    rocm_text = rocm_source.read_text()
    metal_text = metal_source.read_text()

    assert "canonical_requested" in cuda_text
    assert "all six canonical profile orders" in cuda_text
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


def test_native_order_sensitivity_requires_repeatable_raw_six_order_effect() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(5):
        for order_index, order in enumerate(orders):
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
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=5
    )

    assert sensitivity["status"] == "confirmed_order_contamination_above_three_percent"
    assert sensitivity["raw_per_order_data"] is True
    assert sensitivity["max_repeatable_order_position_spread_percent"] > 1.0
    assert sensitivity["max_repeatable_order_position_spread_percent"] > 3.0


def test_native_order_sensitivity_does_not_gate_one_noisy_cycle() -> None:
    records: list[dict[str, object]] = []
    orders = list(itertools.permutations(perf13.PROFILE_ORDER))
    for cycle in range(5):
        for order_index, order in enumerate(orders):
            for profile in perf13.PROFILE_ORDER:
                value = 100.0
                if cycle == 0:
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
                    }
                )

    sensitivity = perf13._native_order_sensitivity(
        records, order_repetitions=5
    )

    assert sensitivity["status"] == "no_repeatable_order_effect_above_one_percent_observed"


def test_metal_workflow_runs_typed_consumer_against_historical_baseline_only() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github" / "workflows" / "metal_beast_benchmark.yml"
    ).read_text()

    assert "-DGAFIME_ABI_CONSUMER_ENABLE_TYPED_BASELINE=ON" in workflow
    assert "installed-payload-baseline/gafime/_metal/libgafime_metal_v1.dylib" in workflow
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


def test_native_operation_aliases_accept_explicit_payload_supplemental_records() -> None:
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
    assert perf13._native_operation_names("payload_unvalidated_operation", "pearson") == set()


def test_cuda_native_artifact_accepts_payload_supplemental_records(tmp_path: Path) -> None:
    artifact = _cuda_artifact(tmp_path / "payload-supplementals")
    payload = json.loads(artifact.read_text())
    template = next(
        record for record in payload["records"] if record["operation"] == "allocation"
    )
    for operation in perf13.NATIVE_SUPPLEMENTAL_OPERATION_ALIASES:
        payload["records"].append({**template, "operation": operation})
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
        assert perf13._native_payload_route_failures(
            "metal", payload, kind="metal_events"
        ) == []

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


def test_metal_input_policies_bind_distinct_source_and_fp32_execution_identity() -> None:
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
