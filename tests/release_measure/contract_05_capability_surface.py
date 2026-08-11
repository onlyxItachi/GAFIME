#!/usr/bin/env python3
"""Public capability/version contract for an installed GAFIME v1 package.

Run from an installed wheel environment. The command must resolve ``gafime``
from that environment, not by relying on a checkout-local private boundary.
"""

from __future__ import annotations

import importlib

import gafime


def main() -> None:
    capabilities = gafime.backend_capabilities("core", probe=True, precision="mixed")
    if capabilities.configured_backend != "core":
        raise AssertionError("core capability contract lost the configured backend")
    if (
        capabilities.selected_backend != "core"
        or capabilities.selection_status != "available"
    ):
        raise AssertionError(
            "core capability contract did not report the selected Core backend"
        )
    if capabilities.graph_support.value is not False:
        raise AssertionError("Core graph support must remain explicitly unavailable")
    if capabilities.device_significance.value is not False:
        raise AssertionError(
            "Core device significance must remain explicitly unavailable"
        )
    if capabilities.host_significance_fallback.value != "gafime_cpu":
        raise AssertionError("host significance fallback placement changed")
    if capabilities.permutation_significance.value["placement"] != "gafime_cpu":
        raise AssertionError("Core permutation significance placement changed")
    if capabilities.stability_significance.value["placement"] != "gafime_cpu":
        raise AssertionError("Core stability significance placement changed")
    stability_detail = capabilities.stability_significance.detail or ""
    if (
        "conditional on selection" not in stability_detail
        or "not out-of-sample" not in stability_detail
        or "does not correct selection bias" not in stability_detail
    ):
        raise AssertionError("Core stability significance scope is undisclosed")
    if capabilities.arrow_ingest_mode.value["zero_copy_into_compute"] is not False:
        raise AssertionError("Arrow ingest must not claim zero-copy compute ownership")
    precision = capabilities.precision_contract.value
    if precision["requested"] != "mixed":
        raise AssertionError("Core precision request contract changed")
    if precision["effective"] != "mixed":
        raise AssertionError("Core precision request was not reported as effective")
    if tuple(precision["supported_profiles"]) != ("fp32", "mixed", "fp64"):
        raise AssertionError("Core must advertise all three distributed profiles")
    if (
        precision["storage_dtype"] != "float32"
        or precision["interaction_arithmetic"] != "float32"
        or precision["reduction_dtype"] != "float64"
        or precision["result_dtype"] != "float64"
    ):
        raise AssertionError("Core mixed profile domains changed")
    if set(precision["accumulators"].values()) != {"float64"}:
        raise AssertionError("Core accumulator disclosure changed")
    if precision["interaction_overflow_diagnostics"] is not True:
        raise AssertionError("Core interaction-overflow diagnostics were not disclosed")

    expected_domains = {
        "fp32": ("float32", "float32", "float32", "float32"),
        "mixed": ("float32", "float32", "float64", "float64"),
        "fp64": ("float64", "float64", "float64", "float64"),
    }
    for profile, expected in expected_domains.items():
        profile_capabilities = gafime.backend_capabilities(
            "core",
            probe=True,
            precision=profile,
        ).precision_contract.value
        actual = (
            profile_capabilities["storage_dtype"],
            profile_capabilities["interaction_arithmetic"],
            profile_capabilities["reduction_dtype"],
            profile_capabilities["result_dtype"],
        )
        if actual != expected:
            raise AssertionError(
                f"Core precision={profile!r} domains {actual!r} != {expected!r}"
            )
        if (
            profile_capabilities["requested"] != profile
            or profile_capabilities["effective"] != profile
            or not profile_capabilities["request_supported"]
        ):
            raise AssertionError(
                f"Core precision={profile!r} request/effective identity changed"
            )

    expected_backend_profiles = {
        "core": ("fp32", "mixed", "fp64"),
        "cuda": ("fp32", "mixed", "fp64"),
        "rocm": ("fp32", "mixed", "fp64"),
        "metal": ("fp32",),
    }
    for backend, expected in expected_backend_profiles.items():
        advertised = tuple(
            gafime.backend_capabilities(
                backend,
                probe=backend == "core",
                precision="fp32",
            ).precision_contract.value["supported_profiles"]
        )
        if advertised != expected:
            raise AssertionError(
                f"{backend} advertised profiles {advertised!r} != {expected!r}"
            )
    for unsupported in ("mixed", "fp64"):
        metal_precision = gafime.backend_capabilities(
            "metal",
            probe=False,
            precision=unsupported,
        ).precision_contract.value
        if metal_precision["request_supported"]:
            raise AssertionError(
                f"Metal unexpectedly advertised precision={unsupported!r}"
            )
        if "Metal supports precision='fp32' only" not in str(
            metal_precision["rejection_reason"]
        ):
            raise AssertionError("Metal precision rejection is not actionable")

    families = {family.name: family for family in capabilities.families}
    for name in ("time_series", "decision_path"):
        family = families[name]
        if family.generation_placement != "gafime_cpu":
            raise AssertionError(f"{name} generation placement is not gafime_cpu")
        if family.graph_scope != "continuous_scoring_only":
            raise AssertionError(f"{name} graph scope claims generation capture")
    if families["decision_path"].native_compact_scoring:
        raise AssertionError("decision_path must use the standard continuous scorer")
    decision_significance = families["decision_path"].significance_support
    if decision_significance.permutation is not True:
        raise AssertionError("decision_path permutation significance support was lost")
    if decision_significance.stability is not True:
        raise AssertionError("decision_path bootstrap stability support was lost")
    if "rediscovery" not in decision_significance.detail:
        raise AssertionError(
            "decision_path permutation rediscovery contract is undisclosed"
        )
    if (
        "conditional on selection" not in decision_significance.detail
        or "not out-of-sample" not in decision_significance.detail
    ):
        raise AssertionError("decision_path bootstrap stability scope is undisclosed")

    native = importlib.import_module("gafime.gafime_py")
    if native.__version__ != gafime.__version__:
        raise AssertionError("native/package public versions diverged")
    if native.native_version() != gafime.__version__:
        raise AssertionError("native version function diverged from package version")
    if not callable(getattr(native, "runtime_capabilities", None)):
        raise AssertionError(
            "native boundary lacks the public runtime capability query"
        )
    print(
        "capability contract passed "
        f"version={gafime.__version__} backend={capabilities.selected_backend}"
    )


if __name__ == "__main__":
    main()
