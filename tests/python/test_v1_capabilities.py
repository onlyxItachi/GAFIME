from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime  # noqa: E402
from gafime import capabilities  # noqa: E402


def _boundary(snapshot):
    def runtime_capabilities(*, backend, device_id, probe, precision):
        assert backend == snapshot["configured_backend"]
        assert device_id == 0
        assert probe is snapshot["probe_performed"]
        assert precision in {"fp32", "mixed", "fp64"}
        return snapshot

    return SimpleNamespace(
        BOUNDARY_NAME="fake-gafime-py",
        __version__="1.0.0rc1",
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
    assert families["decision_path"].native_compact_scoring == ()
    assert families["decision_path"].significance_support.permutation is True
    assert families["decision_path"].significance_support.stability is True
    assert "rediscovery" in families["decision_path"].significance_support.detail
    for name in ("continuous", "time_series", "decision_path"):
        support = families[name].significance_support
        assert support.stability is True
        assert "conditional on selection" in support.detail
        assert "not out-of-sample" in support.detail
    for name in ("continuous", "time_series", "decision_path"):
        assert families[name].significance_support.permutation is True
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
            "precision": {
                "profiles": ["fp32", "mixed", "fp64"],
                "profile_domains": {
                    "mixed": {
                        "storage_dtype": "float32",
                        "interaction_arithmetic": "float32",
                        "reduction_dtype": "float64",
                        "accumulators": {
                            "pearson": "float64",
                            "r2": "float64",
                            "spearman": "float64",
                            "mutual_info": "float64",
                        },
                        "result_dtype": "float64",
                    }
                },
                "scale_normalization": "adaptive_high_dynamic",
                "compensated_summation": False,
                "interaction_overflow_diagnostics": True,
            },
        },
        "candidates": {"cuda": {"status": "available"}},
    }
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )
    monkeypatch.setattr(
        capabilities,
        "installed_payload_build_policy",
        lambda _backend: (
            {"optix_rt": "off", "cuda_tuning_policy": "runtime-device-class"},
            "installed gafime-cuda test policy",
        ),
    )

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
    assert "conditional on selection" in value.stability_significance.detail
    assert "does not correct selection bias" in value.stability_significance.detail
    assert value.device.source == "runtime"
    assert value.device.value["name"] == "test device"
    assert value.mi_estimator.value == "fixed_equal_width_adaptive_template"
    assert value.mi_bin_ceiling.value["effective_template_ceiling"] == 16
    assert value.host_significance_fallback.value == "gafime_cpu"
    assert value.precision_contract.source == "runtime"
    assert value.precision_contract.value["effective"] == "mixed"
    assert value.precision_contract.value["storage_dtype"] == "float32"
    assert value.precision_contract.value["result_dtype"] == "float64"
    assert value.precision_contract.value["accumulators"]["mutual_info"] == "float64"
    assert value.precision_contract.value["interaction_overflow_diagnostics"] is True
    assert value.payload_build_policy.source == "package"
    assert value.payload_build_policy.value["optix_rt"] == "off"
    decision_path = next(
        family
        for family in value.to_dict()["families"]
        if family["name"] == "decision_path"
    )
    decision_significance = decision_path["significance_support"]
    assert decision_significance["permutation"] is True
    assert decision_significance["stability"] is True
    assert "decision-path rediscovery" in decision_significance["detail"]
    assert "conditional on selection" in decision_significance["detail"]
    assert "not out-of-sample" in decision_significance["detail"]


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
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )

    value = gafime.backend_capabilities("cuda", probe=False)

    assert value.selected_backend is None
    assert value.graph_support.source == "unknown"
    assert value.device_significance.source == "unknown"
    assert value.permutation_significance.source == "unknown"
    assert value.stability_significance.value["placement"] == "gafime_cpu"
    assert value.device.source == "unknown"
    assert value.mi_bin_ceiling.source == "static"
    assert value.arrow_ingest_mode.value["zero_copy_into_compute"] is False
    assert value.precision_contract.value["effective"] is None
    assert value.precision_contract.value["accumulators"]["mutual_info"] == "float64"
    assert value.precision_contract.value["interaction_overflow_diagnostics"] is False


def test_native_unprobed_explicit_backend_is_configured_but_not_selected(monkeypatch):
    snapshot = {
        "configured_backend": "cuda",
        "selected_backend": None,
        "status": "not_probed",
        "detail": "runtime payload probing is disabled",
        "probe_performed": False,
        "runtime": None,
        "candidates": {},
    }
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )
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
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )

    value = gafime.backend_capabilities("core")

    assert value.graph_support.value is False
    assert value.graph_support.source == "static"
    assert value.device_significance.value is False
    assert value.permutation_significance.value["placement"] == "gafime_cpu"
    assert value.stability_significance.value["placement"] == "gafime_cpu"
    assert value.mi_estimator.value == "adaptive_quantile"
    assert value.mi_bin_ceiling.value["backend_max"] == 96
    assert value.precision_contract.value["request_supported"] is True
    assert value.precision_contract.value["accumulators"] == {
        "pearson": "float64",
        "r2": "float64",
        "spearman": "float64",
        "mutual_info": "float64",
    }
    assert value.precision_contract.value["interaction_overflow_diagnostics"] is True
    assert value.payload_build_policy.source == "static"
    assert value.payload_build_policy.value is None
    assert value.to_dict()["configured_backend"] == "core"


def test_payload_policy_errors_are_reported_as_unknown(monkeypatch):
    snapshot = {
        "configured_backend": "rocm",
        "selected_backend": "rocm",
        "status": "available",
        "detail": None,
        "probe_performed": True,
        "runtime": {},
        "candidates": {},
    }
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )

    def invalid_policy(_backend):
        raise RuntimeError("malformed policy")

    monkeypatch.setattr(capabilities, "installed_payload_build_policy", invalid_policy)

    value = gafime.backend_capabilities("rocm", probe=True)

    assert value.payload_build_policy.source == "unknown"
    assert value.payload_build_policy.value is None
    assert "malformed policy" in value.payload_build_policy.detail


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
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )
    precision = "fp32" if backend == "metal" else "mixed"
    assert (
        gafime.backend_capabilities(backend, precision=precision).configured_backend
        == backend
    )


def test_report_backend_info_uses_native_selection_and_placement():
    from gafime.v1_adapter import _backend_info

    native = SimpleNamespace(
        backend_name="v1-cuda-cabi",
        device="cuda",
        is_gpu=True,
        selected_backend="cuda",
        execution_placement="cuda",
        precision="mixed",
        storage_dtype="float32",
        interaction_arithmetic="float32",
        result_dtype="float64",
        mi_accumulation_dtype="float64",
    )

    info = _backend_info(native)

    assert info.selected_backend == "cuda"
    assert info.execution_placement == "cuda"
    assert info.requested_precision == "mixed"
    assert info.effective_precision == "mixed"
    assert info.storage_dtype == "float32"
    assert info.result_dtype == "float64"
    assert info.metric_accumulators["mutual_info"] == "float64"
    assert info.metric_accumulators["spearman"] == "float64"


@pytest.mark.parametrize(
    ("storage_dtype", "compute_policy", "expected"),
    [
        ("float32", "fast", "fp32"),
        ("float32", "stable", "mixed"),
        ("float64", "exact", "fp64"),
    ],
)
def test_precision_capability_legacy_pairs_map_to_profiles(
    storage_dtype, compute_policy, expected, monkeypatch
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
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )

    with pytest.warns(DeprecationWarning, match="precision profile"):
        value = gafime.backend_capabilities(
            "core",
            storage_dtype=storage_dtype,
            compute_policy=compute_policy,
        ).precision_contract.value

    assert value["requested"] == expected
    assert value["request_supported"] is True
    assert value["effective"] == expected


@pytest.mark.parametrize(
    ("storage_dtype", "compute_policy"),
    [("float32", "exact"), ("float64", "stable"), ("float64", "fast")],
)
def test_precision_capability_rejects_unsupported_legacy_pairs(
    storage_dtype, compute_policy
):
    with pytest.raises(ValueError, match="unsupported legacy precision pair"):
        gafime.backend_capabilities(
            "core",
            storage_dtype=storage_dtype,
            compute_policy=compute_policy,
        )


def test_precision_capability_rejects_unknown_names():
    with pytest.raises(ValueError, match="storage_dtype"):
        gafime.backend_capabilities(
            "core", storage_dtype="binary128", compute_policy="stable"
        )
    with pytest.raises(ValueError, match="compute_policy"):
        gafime.backend_capabilities(
            "core", storage_dtype="float32", compute_policy="magic"
        )


def test_precision_capability_rejects_explicit_profile_with_legacy_pair():
    with pytest.raises(TypeError, match="precision cannot be combined"):
        gafime.backend_capabilities(
            "core",
            precision="mixed",
            storage_dtype="float32",
            compute_policy="fast",
        )


@pytest.mark.parametrize("precision", ["mixed", "fp64"])
def test_static_metal_precision_rejection_is_reported_without_payload_discovery(
    precision, monkeypatch
):
    def fail_discovery(_backend):
        raise AssertionError("unsupported Metal profile reached payload discovery")

    monkeypatch.setattr(capabilities, "_load_boundary", fail_discovery)
    value = gafime.backend_capabilities(
        "metal", probe=False, precision=precision
    ).precision_contract.value

    assert value["requested"] == precision
    assert value["effective"] is None
    assert value["request_supported"] is False
    assert "Metal supports precision='fp32' only" in value["rejection_reason"]


def test_probed_metal_unsupported_precision_fails_before_payload_discovery(monkeypatch):
    def fail_discovery(_backend):
        raise AssertionError("unsupported Metal profile reached payload discovery")

    monkeypatch.setattr(capabilities, "_load_boundary", fail_discovery)
    with pytest.raises(ValueError, match="Metal supports precision='fp32' only"):
        gafime.backend_capabilities("metal", probe=True, precision="mixed")


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
    monkeypatch.setattr(
        capabilities, "_load_boundary", lambda _backend: _boundary(snapshot)
    )

    precision = "fp32" if backend == "metal" else "mixed"
    value = gafime.backend_capabilities(backend, probe=True, precision=precision)

    assert value.device_significance.value is True
    assert value.permutation_significance.value["placement"] == backend
    assert value.permutation_significance.value["static_family"] == "ranked_replay"
