"""Focused contract tests for the isolated cold-lifecycle benchmark."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys

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
        route.storage_dtype, route.pointwise_dtype, route.reduction_dtype, route.result_dtype = domains
        route.overflow_policy = 1
        assert cold._route_ok(route, profile)

        route.result_dtype = cold.DTYPE_F64 if domains[-1] == cold.DTYPE_F32 else cold.DTYPE_F32
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
            records[0] = cold._route_for_profile("fp32")
            records[1] = cold._route_for_profile("fp32")
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": duplicate_routes},
    )
    funcs = cold._set_prototypes(payload)
    with pytest.raises(RuntimeError, match="duplicate numeric route ids"):
        cold._select_route(funcs, "cuda", "fp32")


def test_generic_route_selection_validates_secondary_routes_and_allows_sparse_sets() -> None:
    def malformed_secondary(_device, _abi, _record_size, records, _capacity, count):
        count_out = cold.ctypes.cast(count, cold.ctypes.POINTER(cold.ctypes.c_uint32))
        count_out.contents.value = 2
        if records is not None:
            records[0] = cold._route_for_profile("fp32")
            records[1] = cold._route_for_profile("mixed")
            records[1].result_dtype = cold.DTYPE_F32
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
            records[0] = cold._route_for_profile("fp64")
        return cold.STATUS_OK

    payload = _FakePayload(
        _GENERIC_SYMBOLS,
        {"gafime_gpu_numeric_routes_v2": sparse_fp64},
    )
    funcs = cold._set_prototypes(payload)
    route, status, _detail, _metadata = cold._select_route(funcs, "cuda", "fp64")
    assert status == cold.STATUS_OK
    assert route is not None and cold._route_ok(route, "fp64")


def test_stats_retain_raw_samples_and_reproducible_confidence_interval() -> None:
    values = [11, 13, 17, 19, 23] * 6
    first = cold._stats(values, cold._stable_seed(7, "phase"), 500)
    second = cold._stats(values, cold._stable_seed(7, "phase"), 500)
    assert first == second
    assert first["count"] == 30
    assert first["raw_duration_ns"] == values
    assert len(first["bootstrap_median_95_ci_ns"]) == 2


def test_required_cold_boundaries_are_declared() -> None:
    assert "runtime_context_initialization" in cold.REQUIRED_PHASES
    assert "code_object_or_module_registration" in cold.REQUIRED_PHASES
    assert "planning" in cold.REQUIRED_PHASES
    assert "process_exit_cleanup" in cold.REQUIRED_PHASES
    assert cold.REQUIRED_PHASES.index("planning") < cold.REQUIRED_PHASES.index("first_execution")
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
    "gafime_gpu_matrix_free_v2",
)


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
    assert cold.PHASE_COMPARABILITY_LIMITS["first_result_materialization"]["status"] == (
        "host_only_d2h_unobservable"
    )
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
            "not_observable", None, "fixture", comparability=cold._phase_comparability(name)
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
