from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1" and str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime  # noqa: E402
from gafime import capabilities  # noqa: E402


def _boundary(snapshot):
    def runtime_capabilities(*, backend, device_id, probe):
        assert backend == snapshot["configured_backend"]
        assert device_id == 0
        assert probe is snapshot["probe_performed"]
        return snapshot

    return SimpleNamespace(
        BOUNDARY_NAME="fake-gafime-py",
        __version__="1.0.0a0",
        runtime_capabilities=runtime_capabilities,
    )


def test_family_capabilities_separate_generation_from_scoring():
    families = {family.name: family for family in gafime.available_families()}

    assert families["continuous"].generation_placement == "native_continuous"
    assert families["continuous"].scoring_backends == (
        "gafime_cpu",
        "cuda",
        "rocm",
        "metal",
    )
    for name in ("time_series", "decision_path"):
        family = families[name]
        assert family.generation_placement == "gafime_cpu"
        assert family.graph_scope == "continuous_scoring_only"
        assert family.scoring_backends == (
            "gafime_cpu",
            "cuda",
            "rocm",
            "metal",
        )
        # Legacy fields remain scoring aliases, never generation-kernel claims.
        assert family.cuda_kernel and family.rocm_kernel and family.metal_kernel


def test_runtime_capability_values_come_from_native_probe(monkeypatch):
    snapshot = {
        "configured_backend": "cuda",
        "selected_backend": "cuda",
        "status": "available",
        "detail": "explicit backend passed the runtime ABI probe",
        "probe_performed": True,
        "runtime": {
            "device": {"name": "test device", "total_global_mem_bytes": 1234},
            "graph": {"supported": True, "mode": "stream_capture"},
            "significance": {"permutation_pvalues_abi": True},
            "rt": {
                "available": True,
                "decision_path_membership_abi": True,
                "decision_path_score_abi": True,
            },
        },
        "candidates": {"cuda": {"status": "available"}},
    }
    monkeypatch.setattr(capabilities, "_load_boundary", lambda _backend: _boundary(snapshot))

    value = gafime.backend_capabilities("cuda", probe=True, mi_bins=20)

    assert value.selected_backend == "cuda"
    assert value.graph_support.source == "runtime"
    assert value.graph_support.value["mode"] == "stream_capture"
    assert value.device_significance.source == "runtime"
    assert value.device_significance.value is True
    assert value.rt_availability.source == "runtime"
    assert value.rt_availability.value["available"] is True
    assert value.device.source == "runtime"
    assert value.device.value["name"] == "test device"
    assert value.mi_estimator.value == "fixed_equal_width_adaptive_template"
    assert value.mi_bin_ceiling.value["effective_template_ceiling"] == 16
    assert value.host_significance_fallback.value == "gafime_cpu"


def test_unprobed_gpu_fields_are_unknown_not_invented(monkeypatch):
    snapshot = {
        "configured_backend": "cuda",
        "selected_backend": None,
        "status": "not_probed",
        "detail": "runtime payload probing is disabled",
        "probe_performed": False,
        "runtime": None,
        "candidates": {},
    }
    monkeypatch.setattr(capabilities, "_load_boundary", lambda _backend: _boundary(snapshot))

    value = gafime.backend_capabilities("cuda", probe=False)

    assert value.selected_backend is None
    assert value.graph_support.source == "unknown"
    assert value.device_significance.source == "unknown"
    assert value.rt_availability.source == "unknown"
    assert value.device.source == "unknown"
    assert value.mi_bin_ceiling.source == "static"
    assert value.arrow_ingest_mode.value["zero_copy_into_compute"] is False


def test_native_unprobed_explicit_backend_is_configured_but_not_selected():
    value = gafime.backend_capabilities("cuda", probe=False)

    assert value.configured_backend == "cuda"
    assert value.selected_backend is None
    assert value.selection_status == "not_probed"


def test_core_static_capabilities_do_not_require_device_data(monkeypatch):
    snapshot = {
        "configured_backend": "core",
        "selected_backend": "core",
        "status": "available",
        "detail": "Core is built into the native boundary.",
        "probe_performed": False,
        "runtime": None,
        "candidates": {},
    }
    monkeypatch.setattr(capabilities, "_load_boundary", lambda _backend: _boundary(snapshot))

    value = gafime.backend_capabilities("core")

    assert value.graph_support.value is False
    assert value.graph_support.source == "static"
    assert value.device_significance.value is False
    assert value.rt_availability.value is False
    assert value.mi_estimator.value == "adaptive_quantile"
    assert value.mi_bin_ceiling.value["backend_max"] == 96
    assert value.to_dict()["configured_backend"] == "core"


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal", "core"])
def test_backend_alias_normalization_is_publicly_stable(backend, monkeypatch):
    snapshot = {
        "configured_backend": backend,
        "selected_backend": backend,
        "status": "not_probed",
        "detail": None,
        "probe_performed": False,
        "runtime": None,
        "candidates": {},
    }
    monkeypatch.setattr(capabilities, "_load_boundary", lambda _backend: _boundary(snapshot))
    assert gafime.backend_capabilities(backend).configured_backend == backend


def test_report_backend_info_uses_native_selection_and_placement():
    from gafime.v1_adapter import _backend_info

    native = SimpleNamespace(
        backend_name="v1-cuda-cabi",
        device="cuda",
        is_gpu=True,
        selected_backend="cuda",
        execution_placement="cuda",
    )

    info = _backend_info(native)

    assert info.selected_backend == "cuda"
    assert info.execution_placement == "cuda"
