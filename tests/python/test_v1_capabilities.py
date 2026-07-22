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
    assert families["decision_path"].native_compact_scoring == ("cuda_rt_optional",)
    assert families["decision_path"].significance_support.permutation is False
    assert families["decision_path"].significance_support.stability is True
    assert "rediscovery" in families["decision_path"].significance_support.detail
    for name in ("continuous", "time_series"):
        support = families[name].significance_support
        assert support.permutation is True
        assert support.stability is True
    assert families["time_series"].native_compact_scoring == ()


def test_runtime_capability_values_come_from_native_probe(monkeypatch):
    snapshot = {
        "configured_backend": "cuda",
        "selected_backend": "cuda",
        "status": "available",
        "detail": "explicit backend passed the runtime ABI probe",
        "probe_performed": True,
        "runtime": {
            "device": {"name": "test device", "total_global_mem_bytes": 1234},
            "graph": {
                "supported": True,
                "mode": "stream_capture",
                "supports_device_ranking": True,
            },
            "significance": {"permutation_pvalues_abi": True},
            "rt": {
                "available": True,
                "decision_path_membership_abi": True,
                "decision_path_score_abi": True,
            },
            "precision": {
                "storage_dtypes": ["float32"],
                "compute_policies": ["stable"],
                "interaction_arithmetic": "float32",
                "accumulators": {
                    "pearson": "float32",
                    "r2": "float32",
                    "spearman": "float64",
                    "mutual_info": "float64",
                },
                "result_dtype": "float32",
                "scale_normalization": "adaptive_high_dynamic",
                "compensated_summation": False,
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
    assert value.permutation_significance.value == {
        "placement": "cuda",
        "static_family": "native_fixed_plan_abi_or_ranked_replay",
        "adaptive_or_generated_family": "ranked_replay",
    }
    assert value.stability_significance.value == {
        "placement": "gafime_cpu",
        "mode": "selected_candidate_bootstrap",
    }
    assert value.rt_availability.source == "runtime"
    assert value.rt_availability.value["available"] is True
    assert value.device.source == "runtime"
    assert value.device.value["name"] == "test device"
    assert value.mi_estimator.value == "fixed_equal_width_adaptive_template"
    assert value.mi_bin_ceiling.value["effective_template_ceiling"] == 16
    assert value.host_significance_fallback.value == "gafime_cpu"
    assert value.precision_contract.source == "runtime"
    assert value.precision_contract.value["effective"] == {
        "storage_dtype": "float32",
        "compute_policy": "stable",
    }
    assert value.precision_contract.value["accumulators"]["mutual_info"] == "float64"
    decision_path = next(
        family
        for family in value.to_dict()["families"]
        if family["name"] == "decision_path"
    )
    assert decision_path["significance_support"] == {
        "permutation": False,
        "stability": True,
        "detail": (
            "Permutation significance is unavailable because every permuted target "
            "requires decision-path rediscovery; selected-candidate bootstrap "
            "stability is supported."
        ),
    }


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
    assert value.permutation_significance.source == "unknown"
    assert value.stability_significance.value["placement"] == "gafime_cpu"
    assert value.rt_availability.source == "unknown"
    assert value.device.source == "unknown"
    assert value.mi_bin_ceiling.source == "static"
    assert value.arrow_ingest_mode.value["zero_copy_into_compute"] is False
    assert value.precision_contract.value["effective"] is None
    assert value.precision_contract.value["accumulators"]["mutual_info"] is None


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
    assert value.permutation_significance.value["placement"] == "gafime_cpu"
    assert value.stability_significance.value["placement"] == "gafime_cpu"
    assert value.rt_availability.value is False
    assert value.mi_estimator.value == "adaptive_quantile"
    assert value.mi_bin_ceiling.value["backend_max"] == 96
    assert value.precision_contract.value["request_supported"] is True
    assert value.precision_contract.value["accumulators"] == {
        "pearson": "float64",
        "r2": "float64",
        "spearman": "float64",
        "mutual_info": "float64",
    }
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
        storage_dtype="float32",
        compute_policy="stable",
        interaction_arithmetic="float32",
        result_dtype="float32",
        mi_accumulation_dtype="float64",
    )

    info = _backend_info(native)

    assert info.selected_backend == "cuda"
    assert info.execution_placement == "cuda"
    assert info.requested_storage_dtype == "float32"
    assert info.effective_storage_dtype == "float32"
    assert info.metric_accumulators["mutual_info"] == "float64"
    assert info.metric_accumulators["spearman"] == "float64"


@pytest.mark.parametrize(
    ("storage_dtype", "compute_policy", "reason"),
    [
        ("float64", "exact", "no current Core, CUDA, ROCm, or Metal"),
        ("float32", "exact", "true f64 ingest"),
        ("float32", "fast", "high-dynamic normalization guard"),
    ],
)
def test_precision_capability_reports_unsupported_requests(
    storage_dtype, compute_policy, reason, monkeypatch
):
    snapshot = {
        "configured_backend": "core",
        "selected_backend": "core",
        "status": "available",
        "detail": None,
        "probe_performed": False,
        "runtime": None,
        "candidates": {},
    }
    monkeypatch.setattr(capabilities, "_load_boundary", lambda _backend: _boundary(snapshot))

    value = gafime.backend_capabilities(
        "core",
        storage_dtype=storage_dtype,
        compute_policy=compute_policy,
    ).precision_contract.value

    assert value["request_supported"] is False
    assert value["effective"] is None
    assert reason in value["rejection_reason"]


def test_precision_capability_rejects_unknown_names():
    with pytest.raises(ValueError, match="storage_dtype"):
        gafime.backend_capabilities("core", storage_dtype="binary128")
    with pytest.raises(ValueError, match="compute_policy"):
        gafime.backend_capabilities("core", compute_policy="magic")


@pytest.mark.parametrize("backend", ["rocm", "metal"])
def test_ranked_gpu_replay_is_reported_as_device_significance(backend, monkeypatch):
    snapshot = {
        "configured_backend": backend,
        "selected_backend": backend,
        "status": "available",
        "detail": None,
        "probe_performed": True,
        "runtime": {
            "graph": {"supports_device_ranking": True},
            "significance": {"permutation_pvalues_abi": False},
        },
        "candidates": {},
    }
    monkeypatch.setattr(capabilities, "_load_boundary", lambda _backend: _boundary(snapshot))

    value = gafime.backend_capabilities(backend, probe=True)

    assert value.device_significance.value is True
    assert value.permutation_significance.value["placement"] == backend
    assert value.permutation_significance.value["static_family"] == "ranked_replay"
