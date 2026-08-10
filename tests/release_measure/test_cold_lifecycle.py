"""Focused contract tests for the isolated cold-lifecycle benchmark."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import sys
import zipfile

import pytest


_SCRIPT = Path(__file__).with_name("cold_lifecycle.py")
_SPEC = importlib.util.spec_from_file_location("gafime_cold_lifecycle", _SCRIPT)
assert _SPEC and _SPEC.loader
cold = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cold
_SPEC.loader.exec_module(cold)


def test_ctypes_layout_matches_published_abi_11_prefix() -> None:
    cold._abi_layout_self_check()


def test_route_contract_is_complete_for_each_public_profile() -> None:
    expected = {
        "fp32": (cold.DTYPE_F32,) * 4,
        "mixed": (cold.DTYPE_F32, cold.DTYPE_F32, cold.DTYPE_F64, cold.DTYPE_F64),
        "fp64": (cold.DTYPE_F64,) * 4,
    }
    for profile, domains in expected.items():
        route = cold._Route()
        route.abi_version = cold.ABI_VERSION
        route.struct_size = cold.ctypes.sizeof(cold._Route)
        route.route_id = cold.PROFILE_IDS[profile]
        route.profile = cold.PROFILE_IDS[profile]
        (
            route.storage_dtype,
            route.pointwise_dtype,
            route.reduction_dtype,
            route.result_dtype,
        ) = domains
        route.overflow_policy = 1
        assert cold._route_ok(route, profile)

        route.result_dtype = (
            cold.DTYPE_F64 if domains[-1] == cold.DTYPE_F32 else cold.DTYPE_F32
        )
        assert not cold._route_ok(route, profile)


def test_route_validation_rejects_bad_prefix_reserved_and_duplicates() -> None:
    route = cold._route_for_profile("fp32")
    route.abi_version = 2 << 16
    assert not cold._route_ok(route, "fp32")
    route = cold._route_for_profile("fp32")
    route.abi_version = 1 << 16
    assert not cold._route_ok(route, "fp32")
    route = cold._route_for_profile("fp32")
    route.struct_size = cold._Route.reserved.offset - 1
    assert not cold._route_ok(route, "fp32")
    route = cold._route_for_profile("fp32")
    route.reserved[0] = 1
    assert not cold._route_ok(route, "fp32")

    def duplicate_routes(_device, _abi, _record_size, records, _capacity, count):
        count_out = cold.ctypes.cast(count, cold.ctypes.POINTER(cold.ctypes.c_uint32))
        count_out.contents.value = 2
        if records is not None:
            _write_route_records(
                records,
                [cold._route_for_profile("fp32"), cold._route_for_profile("fp32")],
            )
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": duplicate_routes},
    )
    funcs = cold._set_prototypes(payload)
    with pytest.raises(RuntimeError, match="duplicate route record"):
        cold._select_route(funcs, "cuda", "fp32")


def test_generic_route_selection_validates_secondary_routes_and_requires_known_set() -> (
    None
):
    def malformed_secondary(_device, _abi, _record_size, records, _capacity, count):
        count_out = cold.ctypes.cast(count, cold.ctypes.POINTER(cold.ctypes.c_uint32))
        count_out.contents.value = 2
        if records is not None:
            mixed = cold._route_for_profile("mixed")
            mixed.result_dtype = cold.DTYPE_F32
            _write_route_records(records, [cold._route_for_profile("fp32"), mixed])
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": malformed_secondary},
    )
    funcs = cold._set_prototypes(payload)
    with pytest.raises(RuntimeError, match="malformed mixed numeric route"):
        cold._select_route(funcs, "cuda", "fp32")

    def sparse_fp64(_device, _abi, _record_size, records, _capacity, count):
        count_out = cold.ctypes.cast(count, cold.ctypes.POINTER(cold.ctypes.c_uint32))
        count_out.contents.value = 1
        if records is not None:
            _write_route_records(records, [cold._route_for_profile("fp64")])
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": sparse_fp64},
    )
    funcs = cold._set_prototypes(payload)
    with pytest.raises(RuntimeError, match="expected known route set"):
        cold._select_route(funcs, "cuda", "fp64")


def test_generic_route_selection_skips_unknown_future_route_and_accepts_larger_known_records() -> (
    None
):
    def future_routes(_device, _abi, record_size, records, _capacity, count):
        count_out = cold.ctypes.cast(count, cold.ctypes.POINTER(cold.ctypes.c_uint32))
        count_out.contents.value = 4
        if records is not None:
            assert record_size == cold.ctypes.sizeof(cold._RouteRecord)
            output = cold.ctypes.cast(records, cold.ctypes.POINTER(cold._RouteRecord))
            for index, profile in enumerate(cold.PROFILE_ORDER):
                route = cold._route_for_profile(profile)
                route.abi_version = (1 << 16) | 2
                route.struct_size = cold.ctypes.sizeof(cold._RouteRecord) + 64
                output[index].known = route
            output[3] = _unknown_future_route()
            output[3].known.struct_size = cold.ctypes.sizeof(cold._RouteRecord) + 64
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": future_routes},
    )
    funcs = cold._set_prototypes(payload)
    route, status, _detail, metadata = cold._select_route(funcs, "cuda", "mixed")
    assert status == cold.STATUS_OK
    assert route is not None and route.struct_size == cold.ctypes.sizeof(cold._Route)
    assert cold._route_ok(route, "mixed")
    assert metadata["profile_mask"] == 0x7


def test_generic_route_selection_rejects_unknown_duplicates_and_required_flags() -> (
    None
):
    def duplicate_unknown(_device, _abi, _record_size, records, _capacity, count):
        count_out = cold.ctypes.cast(count, cold.ctypes.POINTER(cold.ctypes.c_uint32))
        count_out.contents.value = 5
        if records is not None:
            output = cold.ctypes.cast(records, cold.ctypes.POINTER(cold._RouteRecord))
            for index, profile in enumerate(cold.PROFILE_ORDER):
                output[index].known = cold._route_for_profile(profile)
            output[3] = _unknown_future_route()
            output[4] = _unknown_future_route()
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": duplicate_unknown},
    )
    funcs = cold._set_prototypes(payload)
    with pytest.raises(RuntimeError, match="duplicate route record"):
        cold._select_route(funcs, "cuda", "fp32")

    def required_flag(_device, _abi, _record_size, records, _capacity, count):
        count_out = cold.ctypes.cast(count, cold.ctypes.POINTER(cold.ctypes.c_uint32))
        count_out.contents.value = 4
        if records is not None:
            output = cold.ctypes.cast(records, cold.ctypes.POINTER(cold._RouteRecord))
            for index, profile in enumerate(cold.PROFILE_ORDER):
                output[index].known = cold._route_for_profile(profile)
            output[3] = _unknown_future_route()
            output[3].known.flags = 1
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": required_flag},
    )
    funcs = cold._set_prototypes(payload)
    with pytest.raises(RuntimeError, match="invalid route prefix/id"):
        cold._select_route(funcs, "cuda", "fp32")


def test_stats_retain_raw_samples_and_reproducible_confidence_interval() -> None:
    values = [11, 13, 17, 19, 23] * 6
    first = cold._stats(values, cold._stable_seed(7, "phase"), 500)
    second = cold._stats(values, cold._stable_seed(7, "phase"), 500)
    assert first == second
    assert first["count"] == 30
    assert first["raw_duration_ns"] == values
    assert len(first["bootstrap_median_95_ci_ns"]) == 2


def test_wheel_payload_binding_requires_exact_member_bytes(tmp_path: Path) -> None:
    payload = tmp_path / "libgafime_cuda.so"
    payload.write_bytes(b"exact-payload")
    wheel = tmp_path / "gafime_cuda.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gafime_cuda/libgafime_cuda.so", payload.read_bytes())

    binding = cold._wheel_payload_binding("cuda", payload, wheel)
    assert binding["status"] == "verified"
    assert binding["member"] == "gafime_cuda/libgafime_cuda.so"

    payload.write_bytes(b"different-payload")
    with pytest.raises(ValueError, match="do not match"):
        cold._wheel_payload_binding("cuda", payload, wheel)


def test_amd_sysfs_identity_is_stable_and_skips_non_amd_cards(tmp_path: Path) -> None:
    amd = tmp_path / "card1" / "device"
    amd.mkdir(parents=True)
    (amd / "vendor").write_text("0x1002\n")
    (amd / "device").write_text("0x150e\n")
    (amd / "uevent").write_text("PCI_SLOT_NAME=0000:c5:00.0\nDRIVER=amdgpu\n")
    (amd / "unique_id").write_text("stable-id\n")
    other = tmp_path / "card0" / "device"
    other.mkdir(parents=True)
    (other / "vendor").write_text("0x10de\n")

    identity = cold._amd_sysfs_identity(tmp_path)

    assert identity["status"] == "pass"
    assert identity["source"] == "linux_drm_sysfs"
    parsed = json.loads(str(identity["output"]))
    assert parsed == {
        "devices": [
            {
                "card": "card1",
                "device": "0x150e",
                "uevent": ["DRIVER=amdgpu", "PCI_SLOT_NAME=0000:c5:00.0"],
                "unique_id": "stable-id",
            }
        ],
        "driver": {"kernel_release": cold.platform.release(), "name": "amdgpu"},
    }


def test_rocm_device_identity_falls_back_when_rocm_smi_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = {
        "status": "pass",
        "output": '[{"device":"0x150e"}]',
        "source": "linux_drm_sysfs",
    }
    monkeypatch.setattr(
        cold,
        "_command",
        lambda *_args, **_kwargs: {"status": "error", "detail": "missing"},
    )
    monkeypatch.setattr(cold, "_amd_sysfs_identity", lambda: fallback)

    assert cold._device_identity("rocm") == fallback


def test_rocm_device_identity_rejects_header_only_rocm_smi_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = {
        "status": "pass",
        "output": '[{"device":"0x150e"}]',
        "source": "linux_drm_sysfs",
    }
    monkeypatch.setattr(
        cold,
        "_command",
        lambda *_args, **_kwargs: {
            "status": "pass",
            "output": "ROCm System Management Interface",
        },
    )
    monkeypatch.setattr(cold, "_amd_sysfs_identity", lambda: fallback)

    assert cold._device_identity("rocm") == fallback


def test_rocm_device_identity_accepts_complete_rocm_smi_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "status": "pass",
        "output": "GPU[0] : Card series: AMD Radeon\nDriver version: 7.0.0",
    }
    monkeypatch.setattr(cold, "_command", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(
        cold,
        "_amd_sysfs_identity",
        lambda: pytest.fail("complete ROCm SMI identity must not fall back"),
    )

    assert cold._device_identity("rocm") == identity


def _cold_artifact(
    *,
    variant: str,
    block: int,
    sequence: tuple[str, str],
    sample_ns: int,
) -> dict[str, object]:
    phase_summaries = {
        "fp32": {
            phase: {
                "comparability": cold.PHASE_COMPARABILITY[phase],
                "status_counts": {"observed": 30}
                if phase == "python_import"
                else {"not_observable": 30},
                "observed": {
                    "count": 30,
                    "raw_duration_ns": [sample_ns] * 30,
                }
                if phase == "python_import"
                else None,
                "observed_combined": None,
            }
            for phase in cold.REQUIRED_PHASES
        }
    }
    source_commit = ("a" if variant == "baseline" else "b") * 40
    artifact_hash = "1" * 64 if variant == "baseline" else "2" * 64
    return {
        "schema": cold.SCHEMA,
        "status": "pass",
        "backend": "metal",
        "variant": variant,
        "ab_block": block,
        "variant_sequence": list(sequence),
        "process_isolation": "fresh_worker_process_per_profile_sample",
        "fresh_subprocess_per_sample": True,
        "repetitions_per_profile": 30,
        "profiles": ["fp32"],
        "workload": {"rows": 4, "cols": 2, "candidate_count": 1, "metric": "pearson"},
        "phase_comparability": cold.PHASE_COMPARABILITY,
        "phase_summaries": phase_summaries,
        "provenance": {
            "benchmark_script": {"sha256": "f" * 64},
            "device": {"status": "pass", "output": "Apple GPU"},
            "toolchain": {
                "compiler": {"status": "pass", "output": "metal fixture"},
                "linker": {"status": "pass", "output": "ld fixture"},
            },
            "process_affinity": {"status": "unavailable", "detail": "fixture"},
            "clock_and_power_state": {
                "before": {"accelerator": {"status": "pass"}},
                "after": {"accelerator": {"status": "pass"}},
            },
            "source": {
                "commit": source_commit,
                "git_status": {"status": "clean", "entries": []},
            },
            "payload_wheel_binding": {"status": "verified"},
            "payload": {"sha256": artifact_hash},
            "wheel": {"sha256": artifact_hash[::-1]},
        },
    }


def _cold_comparison_manifest(
    tmp_path: Path, *, baseline_ns: int, candidate_ns: int
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sequences = {
        0: ("baseline", "candidate"),
        1: ("candidate", "baseline"),
    }
    entries = []
    for block in (0, 1):
        for variant in sequences[block]:
            payload = _cold_artifact(
                variant=variant,
                block=block,
                sequence=sequences[block],
                sample_ns=baseline_ns if variant == "baseline" else candidate_ns,
            )
            path = tmp_path / f"cold-{block}-{variant}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            entries.append(
                {
                    "path": path.name,
                    "sha256": cold._sha256(path),
                    "variant": variant,
                    "ab_block": block,
                    "variant_sequence": list(sequences[block]),
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": cold.COMPARISON_MANIFEST_SCHEMA,
                "artifacts": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_cold_comparison_accepts_ab_ba_and_rejects_repeatable_regression(
    tmp_path: Path,
) -> None:
    passing = cold._cold_comparison(
        _cold_comparison_manifest(
            tmp_path / "passing", baseline_ns=100, candidate_ns=99
        ),
        seed=7,
        resamples=500,
    )
    assert passing["status"] == "pass"
    assert passing["valid_for_canonical_cold_lifecycle_claims"] is True

    failing = cold._cold_comparison(
        _cold_comparison_manifest(
            tmp_path / "failing", baseline_ns=100, candidate_ns=105
        ),
        seed=7,
        resamples=500,
    )
    assert failing["status"] == "regression_or_contamination_detected"
    assert failing["valid_for_canonical_cold_lifecycle_claims"] is False
    assert any(
        failure["reason"] == "repeatable_regression_above_three_percent"
        for failure in failing["failures"]
    )


def test_required_cold_boundaries_are_declared() -> None:
    assert "runtime_context_initialization" in cold.REQUIRED_PHASES
    assert "code_object_or_module_registration" in cold.REQUIRED_PHASES
    assert "planning" in cold.REQUIRED_PHASES
    assert "process_exit_cleanup" in cold.REQUIRED_PHASES
    assert cold.REQUIRED_PHASES.index("planning") < cold.REQUIRED_PHASES.index(
        "first_execution"
    )
    assert cold.REQUIRED_PHASES.index("explicit_cleanup") < cold.REQUIRED_PHASES.index(
        "process_exit_cleanup"
    )


def test_profile_permutations_cover_all_orders() -> None:
    expected = set(itertools.permutations(cold.PROFILE_ORDER))
    schedule = cold._profile_order_schedule(cold.PROFILE_ORDER, 12, 20260809)
    assert set(schedule[:6]) == expected
    assert set(schedule[6:12]) == expected
    assert schedule == cold._profile_order_schedule(cold.PROFILE_ORDER, 12, 20260809)
    assert schedule != cold._profile_order_schedule(cold.PROFILE_ORDER, 12, 20260810)


class _FakeSymbol:
    def __init__(self, callback=None):
        self.callback = callback
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args) if self.callback else cold.STATUS_OK


class _FakePayload:
    def __init__(self, names, callbacks=None):
        callbacks = callbacks or {}
        for name in names:
            setattr(self, name, _FakeSymbol(callbacks.get(name)))


_TYPED_SYMBOLS = (
    "gafime_gpu_precision_capabilities",
    "gafime_gpu_matrix_alloc_v2",
    "gafime_gpu_matrix_upload_f32_v2",
    "gafime_gpu_matrix_upload_f64_v2",
    "gafime_gpu_matrix_update_target_f32_v2",
    "gafime_gpu_matrix_update_target_f64_v2",
    "gafime_gpu_execute_f32_v2",
    "gafime_gpu_execute_f64_v2",
    "gafime_gpu_execution_memory_peak_v2",
    "gafime_gpu_matrix_free",
)


_GENERIC_SYMBOLS = (
    "gafime_gpu_numeric_routes_v2",
    "gafime_gpu_matrix_alloc_v2",
    "gafime_gpu_matrix_upload_v2",
    "gafime_gpu_matrix_update_target_v2",
    "gafime_gpu_execute_v2",
    "gafime_gpu_execution_memory_peak_v2",
    "gafime_gpu_permutation_memory_peak_v2",
    "gafime_gpu_permutation_pvalues_v2",
    "gafime_gpu_interaction_diagnostics_v2",
    "gafime_gpu_matrix_free_v2",
)


def _write_route_records(records, routes):
    output = cold.ctypes.cast(records, cold.ctypes.POINTER(cold._RouteRecord))
    for index, route in enumerate(routes):
        output[index].known = route


def _unknown_future_route():
    record = cold._RouteRecord()
    record.known.abi_version = (1 << 16) | 2
    record.known.struct_size = cold.ctypes.sizeof(cold._RouteRecord)
    record.known.route_id = 0x10001
    record.known.profile = 0x10001
    record.known.storage_dtype = 0x10001
    record.known.pointwise_dtype = 0x10001
    record.known.reduction_dtype = 0x10001
    record.known.result_dtype = 0x10001
    record.known.overflow_policy = 0x10001
    record.known.flags = cold.ABI_IGNORABLE_FLAG_MASK
    record.future_fields[0] = 0x123456789ABCDEF0
    return record


def _fill_typed_capabilities(_device_id, output):
    capabilities = cold.ctypes.cast(
        output, cold.ctypes.POINTER(cold._PrecisionCapabilities)
    ).contents
    capabilities.abi_version = cold.ABI_VERSION
    capabilities.backend_kind = cold.BACKEND_KINDS["cuda"]
    capabilities.profile_mask = 0x7
    capabilities.storage_dtype_mask = 0x3
    capabilities.result_dtype_mask = 0x3
    return cold.STATUS_OK


def test_frozen_typed_surface_is_selected_without_generic_route_symbol() -> None:
    payload = _FakePayload(
        _TYPED_SYMBOLS,
        {"gafime_gpu_precision_capabilities": _fill_typed_capabilities},
    )
    funcs = cold._set_prototypes(payload)

    assert funcs["abi_surface"] == "precision-typed-v1.1"
    assert funcs["route_source"] == "precision_capabilities"
    assert funcs["free_returns_status"] is False
    route, status, detail, metadata = cold._select_route(funcs, "cuda", "mixed")
    assert status == cold.STATUS_OK
    assert route is not None
    assert cold._route_ok(route, "mixed")
    assert "typed precision capability-mask" in detail
    assert metadata["route_synthesized"] is True


def test_generic_surface_remains_preferred_when_route_symbol_exists() -> None:
    payload = _FakePayload(_GENERIC_SYMBOLS)
    funcs = cold._set_prototypes(payload)

    assert funcs["abi_surface"] == "numeric-route-v2"
    assert funcs["route_source"] == "numeric_routes_v2"
    assert funcs["free_returns_status"] is True


def test_generic_surface_requires_all_ten_symbols_before_lifecycle() -> None:
    for missing in _GENERIC_SYMBOLS:
        payload = _FakePayload(name for name in _GENERIC_SYMBOLS if name != missing)
        with pytest.raises(RuntimeError, match="missing required ABI symbol"):
            cold._set_prototypes(payload)


def test_typed_capability_profile_without_dtype_mask_fails_closed() -> None:
    def incomplete_capabilities(_device_id, output):
        capabilities = cold.ctypes.cast(
            output, cold.ctypes.POINTER(cold._PrecisionCapabilities)
        ).contents
        capabilities.abi_version = cold.ABI_VERSION
        capabilities.backend_kind = cold.BACKEND_KINDS["cuda"]
        capabilities.profile_mask = 0x7
        capabilities.storage_dtype_mask = 0x1
        capabilities.result_dtype_mask = 0x1
        return cold.STATUS_OK

    payload = _FakePayload(
        _TYPED_SYMBOLS,
        {"gafime_gpu_precision_capabilities": incomplete_capabilities},
    )
    funcs = cold._set_prototypes(payload)
    with pytest.raises(RuntimeError, match="storage dtype mask"):
        cold._select_route(funcs, "cuda", "fp64")


def test_phase_comparability_never_claims_d2h_or_cross_surface_identity() -> None:
    assert set(cold.PHASE_COMPARABILITY) == set(cold.REQUIRED_PHASES)
    assert cold.PHASE_COMPARABILITY_LIMITS["first_result_materialization"][
        "status"
    ] == ("host_only_d2h_unobservable")
    assert cold.PHASE_COMPARABILITY_LIMITS["first_capability_query"]["status"] == (
        "not_comparable"
    )
    phase = cold._phase(
        "observed",
        17,
        "caller-owned host read",
        comparability="host_only_d2h_unobservable",
    )
    assert phase["comparability"] == "host_only_d2h_unobservable"

    phases = {
        name: cold._phase(
            "not_observable",
            None,
            "fixture",
            comparability=cold._phase_comparability(name),
        )
        for name in cold.REQUIRED_PHASES
    }
    cold._validate_phase_comparability(phases)
    phases["planning"]["comparability"] = "direct"
    with pytest.raises(RuntimeError, match="planning comparability drift"):
        cold._validate_phase_comparability(phases)


def test_missing_typed_common_symbol_is_not_silently_treated_as_legacy() -> None:
    payload = _FakePayload(_TYPED_SYMBOLS[1:])
    with pytest.raises(RuntimeError, match="gafime_gpu_precision_capabilities"):
        cold._set_prototypes(payload)
