#!/usr/bin/env python3
"""Public capability/version contract for an installed GAFIME v1 package.

Run from an installed wheel environment. The command must resolve ``gafime``
from that environment, not by relying on a checkout-local private boundary.
"""

from __future__ import annotations

import importlib

import gafime


def main() -> None:
    capabilities = gafime.backend_capabilities("core", probe=True)
    if capabilities.configured_backend != "core":
        raise AssertionError("core capability contract lost the configured backend")
    if capabilities.selected_backend != "core" or capabilities.selection_status != "available":
        raise AssertionError("core capability contract did not report the selected Core backend")
    if capabilities.graph_support.value is not False:
        raise AssertionError("Core graph support must remain explicitly unavailable")
    if capabilities.device_significance.value is not False:
        raise AssertionError("Core device significance must remain explicitly unavailable")
    if capabilities.host_significance_fallback.value != "gafime_cpu":
        raise AssertionError("host significance fallback placement changed")
    if capabilities.permutation_significance.value["placement"] != "gafime_cpu":
        raise AssertionError("Core permutation significance placement changed")
    if capabilities.stability_significance.value["placement"] != "gafime_cpu":
        raise AssertionError("Core stability significance placement changed")
    if capabilities.arrow_ingest_mode.value["zero_copy_into_compute"] is not False:
        raise AssertionError("Arrow ingest must not claim zero-copy compute ownership")
    precision = capabilities.precision_contract.value
    if precision["requested"] != {
        "storage_dtype": "float32",
        "compute_policy": "stable",
    }:
        raise AssertionError("Core precision request contract changed")
    if precision["effective"] != precision["requested"]:
        raise AssertionError("Core precision request was not reported as effective")
    if precision["supported_storage_dtypes"] != ("float32",):
        raise AssertionError("Core must not advertise unimplemented f64 storage")
    if set(precision["accumulators"].values()) != {"float64"}:
        raise AssertionError("Core accumulator disclosure changed")

    families = {family.name: family for family in capabilities.families}
    for name in ("time_series", "decision_path"):
        family = families[name]
        if family.generation_placement != "gafime_cpu":
            raise AssertionError(f"{name} generation placement is not gafime_cpu")
        if family.graph_scope != "continuous_scoring_only":
            raise AssertionError(f"{name} graph scope claims generation capture")
    if families["decision_path"].native_compact_scoring != ("cuda_rt_optional",):
        raise AssertionError("decision_path compact CUDA RT placement changed")
    decision_significance = families["decision_path"].significance_support
    if decision_significance.permutation is not False:
        raise AssertionError(
            "decision_path must disclose unavailable permutation significance"
        )
    if decision_significance.stability is not True:
        raise AssertionError("decision_path bootstrap stability support was lost")
    if "rediscovery" not in decision_significance.detail:
        raise AssertionError("decision_path permutation exclusion lost its reason")

    native = importlib.import_module("gafime.gafime_py")
    if native.__version__ != gafime.__version__:
        raise AssertionError("native/package public versions diverged")
    if native.native_version() != gafime.__version__:
        raise AssertionError("native version function diverged from package version")
    if not callable(getattr(native, "runtime_capabilities", None)):
        raise AssertionError("native boundary lacks the public runtime capability query")
    print(
        "capability contract passed "
        f"version={gafime.__version__} backend={capabilities.selected_backend}"
    )


if __name__ == "__main__":
    main()
