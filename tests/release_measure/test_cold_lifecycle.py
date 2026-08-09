"""Focused contract tests for the isolated cold-lifecycle benchmark."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys


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
        route.route_id = cold.PROFILE_IDS[profile]
        route.profile = cold.PROFILE_IDS[profile]
        route.storage_dtype, route.pointwise_dtype, route.reduction_dtype, route.result_dtype = domains
        route.overflow_policy = 1
        assert cold._route_ok(route, profile)

        route.result_dtype = cold.DTYPE_F64 if domains[-1] == cold.DTYPE_F32 else cold.DTYPE_F32
        assert not cold._route_ok(route, profile)


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
