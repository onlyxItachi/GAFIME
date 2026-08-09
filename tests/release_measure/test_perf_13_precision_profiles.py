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
    binary = tmp_path / "binary"
    source.write_bytes(b"source")
    binary.write_bytes(b"binary")
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
    artifact.write_text(
        json.dumps(
            {
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
                "process_affinity": [0],
                "clock": "steady_clock",
                "provenance": {
                    "benchmark_source": _identity(source),
                    "benchmark_binary": _identity(binary),
                },
                "records": records,
            }
        )
    )
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
    artifact.write_text(
        json.dumps(
            {
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
                "profile_orders": (
                    [list(order) for order in orders]
                ),
                "provenance": {
                    "benchmark_source": _identity(source),
                    "benchmark_binary": _identity(binary),
                    "payload": _identity(payload),
                    "wheel": _identity(wheel),
                },
                "records": records,
            }
        )
    )
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
        "backend_kind": 2,
        "route_count": 3,
        "operations": operations,
    }
    provenance = {
        "payload": _identity(payload),
        "wheel": _identity(wheel),
        "consumer_binary": _identity(consumer),
        "consumer_source": _identity(source),
    }
    lifecycle = {
        "schema": "gafime.native-decomposition.v1",
        "status": "pass",
        "execution_mode": "canonical_payload",
        "execution_layer": "independent_abi_1_1_c_consumer",
        "abi": "canonical_1.1",
        "backend": "cuda",
        "profiles": ["fp32", "mixed", "fp64"],
        "source_commit": "a" * 40,
        "source_tree_state": {"status": "clean", "entries": []},
        "wheel_member": "gafime_cuda/libgafime_cuda.so",
        "wheel_member_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "operations": operations,
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


def test_comparative_gate_rejects_runtime_and_clock_snapshot_mismatch() -> None:
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
                "python_version": "3.14",
                "benchmark_script": {"sha256": "f" * 64},
                "environment": {},
                "process_affinity": {"cpus": [0]},
                "clock_and_power_state": {
                    "before": {"nvidia_smi": snapshot},
                    "after": {"nvidia_smi": snapshot},
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

    assert readiness["complete"] is False
    assert any(
        failure["reason"] == "baseline_and_candidate_runtime_mismatch"
        and failure["field"] == "python_executable"
        for failure in readiness["failures"]
    )
    assert any(
        failure["reason"] == "baseline_and_candidate_clock_power_snapshot_mismatch"
        and failure["backend"] == "cuda"
        for failure in readiness["failures"]
    )


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
