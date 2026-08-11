from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import warnings
from typing import Any, Mapping

from ._precision import (
    SUPPORTED_PRECISIONS,
    backend_precision_error,
    normalize_precision,
    precision_from_legacy_pair,
)
from .config import _LEGACY_PRECISION_WARNING
from ._payloads import installed_payload_build_policy
from .families import BOOTSTRAP_STABILITY_SCOPE, FamilyCapability, available_families


_BACKEND_ALIASES = {
    "auto": "auto",
    "cpu": "core",
    "core": "core",
    "rust": "core",
    "v1-rust-cpu": "core",
    "cuda": "cuda",
    "rocm": "rocm",
    "hip": "rocm",
    "metal": "metal",
}
_MI_TEMPLATE_LEVELS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96)


class _DefaultPrecision(str):
    """String-compatible sentinel that distinguishes an omitted default."""


_DEFAULT_PRECISION = _DefaultPrecision("mixed")


@dataclass(frozen=True)
class CapabilityValue:
    """A capability value and the evidence behind it.

    ``runtime`` means the loaded C ABI reported the value. ``package`` means it
    was read from an installed distribution without loading its native library.
    ``static`` means it follows checked-in Core policy. ``unknown`` deliberately
    makes no claim because no compatible observation is available.
    """

    value: Any
    source: str
    detail: str | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    """Public backend placement and capability snapshot.

    The object contains only Python values, never backend-native handles. A
    probe loads a configured payload and calls its identity/capability ABI, but
    does not allocate a matrix or execute a scoring workload.
    """

    configured_backend: str
    selected_backend: str | None
    selection_status: str
    selection_detail: str | None
    probe_performed: bool
    native_boundary: CapabilityValue
    native_version: CapabilityValue
    families: tuple[FamilyCapability, ...]
    graph_support: CapabilityValue
    device_significance: CapabilityValue
    host_significance_fallback: CapabilityValue
    permutation_significance: CapabilityValue
    stability_significance: CapabilityValue
    mi_estimator: CapabilityValue
    mi_bin_ceiling: CapabilityValue
    precision_contract: CapabilityValue
    payload_build_policy: CapabilityValue
    arrow_ingest_mode: CapabilityValue
    generated_family_graph_limit: CapabilityValue
    device: CapabilityValue
    probe_details: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def backend_capabilities(
    backend: str = "auto",
    device_id: int = 0,
    *,
    probe: bool = False,
    mi_bins: int = 96,
    mi_approximate: bool = False,
    precision: str = _DEFAULT_PRECISION,
    **legacy_precision: object,
) -> BackendCapabilities:
    """Return truthful static and, when requested, runtime backend facts.

    ``probe=False`` is side-effect free with respect to GPU payload loading.
    ``probe=True`` follows the same payload loader used by the native resolver;
    explicit backends never become another backend, while ``auto`` reports its
    CPU fallback and every failed candidate explicitly.
    """

    configured = _normalize_backend(backend)
    if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
        raise ValueError("device_id must be a non-negative integer.")
    if not isinstance(mi_bins, int) or isinstance(mi_bins, bool) or mi_bins < 2:
        raise ValueError("mi_bins must be an integer greater than or equal to 2.")
    if legacy_precision:
        unexpected = set(legacy_precision) - {"storage_dtype", "compute_policy"}
        if unexpected:
            raise TypeError(
                f"unexpected precision keyword(s): {', '.join(sorted(unexpected))}"
            )
        if set(legacy_precision) != {"storage_dtype", "compute_policy"}:
            raise TypeError(
                "deprecated storage_dtype and compute_policy must be supplied together."
            )
        if precision is not _DEFAULT_PRECISION:
            raise TypeError(
                "precision cannot be combined with deprecated storage_dtype/compute_policy."
            )
        precision = precision_from_legacy_pair(
            legacy_precision["storage_dtype"], legacy_precision["compute_policy"]
        )
        warnings.warn(_LEGACY_PRECISION_WARNING, DeprecationWarning, stacklevel=2)
    precision = normalize_precision(precision)
    unsupported_reason = backend_precision_error(configured, precision)
    if unsupported_reason is not None and probe:
        raise ValueError(
            f"unsupported precision request precision={precision!r}: {unsupported_reason}"
        )
    if unsupported_reason is not None:
        # A static capability query must be able to report a rejected request,
        # but it must not load the native boundary or discover payloads merely
        # to learn that the profile/backend pair is impossible.
        snapshot = {
            "configured_backend": configured,
            "selected_backend": None,
            "status": "unsupported",
            "detail": unsupported_reason,
            "probe_performed": False,
            "runtime": None,
            "candidates": {
                configured: {"status": "unsupported", "detail": unsupported_reason}
            },
        }
        boundary = None
        boundary_error = (
            "native boundary was not loaded for an unsupported precision request"
        )
    else:
        snapshot, boundary, boundary_error = _runtime_snapshot(
            configured, device_id, probe, precision
        )
    selected = _string_or_none(snapshot.get("selected_backend"))
    selection_status = str(snapshot.get("status", "unknown"))
    selection_detail = _string_or_none(snapshot.get("detail"))
    effective_backend = selected or (configured if configured != "auto" else None)
    runtime = _mapping_or_empty(snapshot.get("runtime"))

    native_boundary = (
        CapabilityValue(
            getattr(
                boundary,
                "BOUNDARY_NAME",
                getattr(boundary, "__name__", type(boundary).__name__),
            ),
            "runtime",
        )
        if boundary is not None
        else CapabilityValue(None, "unknown", boundary_error)
    )
    native_version = (
        CapabilityValue(getattr(boundary, "__version__", None), "runtime")
        if boundary is not None
        else CapabilityValue(None, "unknown", boundary_error)
    )

    return BackendCapabilities(
        configured_backend=configured,
        selected_backend=selected,
        selection_status=selection_status,
        selection_detail=selection_detail,
        probe_performed=bool(snapshot.get("probe_performed", probe)),
        native_boundary=native_boundary,
        native_version=native_version,
        families=available_families(),
        graph_support=_graph_support(effective_backend, runtime),
        device_significance=_device_significance(effective_backend, runtime),
        host_significance_fallback=CapabilityValue(
            "gafime_cpu",
            "static",
            "Core significance and GPU bootstrap stability use the retained CPU "
            "matrix. GPU permutation maxT uses same-device ranking when available.",
        ),
        permutation_significance=_permutation_significance(effective_backend, runtime),
        stability_significance=_stability_significance(effective_backend),
        mi_estimator=_mi_estimator(effective_backend, mi_approximate),
        mi_bin_ceiling=_mi_bin_ceiling(effective_backend, mi_bins),
        precision_contract=_precision_contract(
            effective_backend,
            selected,
            runtime,
            precision,
        ),
        payload_build_policy=_payload_build_policy(effective_backend),
        arrow_ingest_mode=CapabilityValue(
            {
                "protocol": "Arrow C stream",
                "record_batches": "exactly one required",
                "compute_buffer": (
                    "native row-major f64 copy"
                    if precision == "fp64"
                    else "native row-major f32 copy"
                ),
                "zero_copy_into_compute": False,
            },
            "static",
            "The Arrow boundary avoids Python-object materialization but owns a "
            "row-major compute buffer after validation.",
        ),
        generated_family_graph_limit=CapabilityValue(
            {
                "time_series": "gafime_cpu generation is outside graph capture; "
                "only later continuous scoring may use a runtime-supported graph",
                "decision_path": "gafime_cpu generation is outside graph capture; "
                "only later continuous scoring may use a runtime-supported graph",
            },
            "static",
        ),
        device=(
            CapabilityValue(runtime["device"], "runtime")
            if "device" in runtime
            else CapabilityValue(
                None,
                "unknown",
                "No validated device-info ABI result is available for this selection.",
            )
        ),
        probe_details=_mapping_or_empty(snapshot.get("candidates")),
    )


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string.")
    try:
        return _BACKEND_ALIASES[backend.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown backend {backend!r}") from exc


def _runtime_snapshot(
    backend: str,
    device_id: int,
    probe: bool,
    precision: str,
) -> tuple[Mapping[str, object], object | None, str | None]:
    try:
        boundary = _load_boundary(backend)
    except Exception as exc:  # Boundary failures must become observable facts.
        base_boundary = _load_base_boundary()
        if base_boundary is not None:
            detail = f"payload discovery failed: {exc}"
            candidate = backend if backend != "auto" else "payload_discovery"
            return (
                {
                    "configured_backend": backend,
                    "status": "unavailable",
                    "detail": detail,
                    "probe_performed": probe,
                    "candidates": {
                        candidate: {"status": "unavailable", "detail": detail}
                    },
                },
                base_boundary,
                None,
            )
        return (
            {
                "configured_backend": backend,
                "status": "unavailable",
                "detail": f"native boundary import failed: {exc}",
                "probe_performed": False,
            },
            None,
            f"native boundary import failed: {exc}",
        )

    runtime_capabilities = getattr(boundary, "runtime_capabilities", None)
    if not callable(runtime_capabilities):
        return (
            {
                "configured_backend": backend,
                "status": "unknown",
                "detail": "native boundary does not expose runtime_capabilities",
                "probe_performed": False,
            },
            boundary,
            None,
        )
    try:
        snapshot = runtime_capabilities(
            backend=backend,
            device_id=device_id,
            probe=probe,
            precision=precision,
        )
    except Exception as exc:
        return (
            {
                "configured_backend": backend,
                "status": "unavailable",
                "detail": f"native capability query failed: {exc}",
                "probe_performed": probe,
            },
            boundary,
            None,
        )
    if not isinstance(snapshot, Mapping):
        return (
            {
                "configured_backend": backend,
                "status": "unknown",
                "detail": "native runtime_capabilities returned a non-mapping result",
                "probe_performed": probe,
            },
            boundary,
            None,
        )
    return snapshot, boundary, None


def _load_boundary(backend: str) -> object:
    # Reuse the public adapter's boundary/discovery seam. Payload-discovery work
    # only needs to keep this native module entrypoint available.
    from .v1_adapter import _load_boundary_for_backend as load_boundary

    return load_boundary(backend)


def _load_base_boundary() -> object | None:
    """Read only the Core boundary after a payload-discovery failure.

    This does not select or execute a backend. It preserves package/native
    version reporting while the returned selection status remains unavailable.
    """

    for name in ("gafime.gafime_py", "gafime_py"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _graph_support(
    backend: str | None, runtime: Mapping[str, object]
) -> CapabilityValue:
    graph = _mapping_or_empty(runtime.get("graph"))
    if graph:
        return CapabilityValue(graph, "runtime")
    if backend == "core":
        return CapabilityValue(
            False, "static", "Core has no graph capture/replay path."
        )
    return CapabilityValue(
        None, "unknown", "Graph support requires a validated payload probe."
    )


def _payload_build_policy(backend: str | None) -> CapabilityValue:
    if backend == "core":
        return CapabilityValue(
            None,
            "static",
            "core is carried by the base gafime distribution, not a separate "
            "vendor payload wheel.",
        )
    if backend == "metal":
        return CapabilityValue(
            {
                "distribution_identity": "gafime",
                "packaging": "embedded-in-macos-arm64-core-wheel",
                "library": "libgafime_metal_v1.dylib",
                "metallib": "gafime_metal_v1.metallib",
            },
            "static",
            "Metal is embedded only in the Apple Silicon gafime core wheel.",
        )
    if backend not in {"cuda", "rocm"}:
        return CapabilityValue(
            None,
            "unknown",
            "Payload build policy depends on the selected vendor backend.",
        )
    try:
        policy, detail = installed_payload_build_policy(backend)
    except Exception as exc:
        return CapabilityValue(
            None,
            "unknown",
            f"installed payload policy could not be validated: {exc}",
        )
    return CapabilityValue(
        policy, "package" if policy is not None else "unknown", detail
    )


def _device_significance(
    backend: str | None,
    runtime: Mapping[str, object],
) -> CapabilityValue:
    graph = _mapping_or_empty(runtime.get("graph"))
    significance = _mapping_or_empty(runtime.get("significance"))
    if backend in {"cuda", "rocm", "metal"} and graph:
        supported = bool(graph.get("supports_device_ranking"))
        native_cuda = backend == "cuda" and bool(
            significance.get("permutation_pvalues_abi")
        )
        mode = (
            "the optional native CUDA fixed-plan ABI plus same-device ranked replay"
            if native_cuda
            else "Rust-orchestrated same-device ranked replay"
        )
        return CapabilityValue(
            supported,
            "runtime",
            f"Permutation maxT uses {mode}; bootstrap stability remains on CPU."
            if supported
            else "This payload does not advertise device ranking, so device "
            "permutation significance is unavailable.",
        )
    if backend == "core":
        return CapabilityValue(
            False,
            "static",
            "Core computes permutation and stability significance on CPU.",
        )
    return CapabilityValue(
        None,
        "unknown",
        "Device significance requires a validated GPU payload probe.",
    )


def _permutation_significance(
    backend: str | None,
    runtime: Mapping[str, object],
) -> CapabilityValue:
    if backend == "core":
        return CapabilityValue(
            {"placement": "gafime_cpu", "mode": "family_wise_maxT"},
            "static",
        )
    if backend in {"cuda", "rocm", "metal"}:
        graph = _mapping_or_empty(runtime.get("graph"))
        if not graph:
            return CapabilityValue(
                None,
                "unknown",
                "Permutation placement requires a validated GPU payload probe.",
            )
        if not bool(graph.get("supports_device_ranking")):
            return CapabilityValue(
                {"placement": None, "mode": "unavailable"},
                "runtime",
                "The loaded payload does not advertise device top-k ranking.",
            )
        significance = _mapping_or_empty(runtime.get("significance"))
        static_mode = (
            "native_fixed_plan_abi_or_ranked_replay"
            if backend == "cuda" and bool(significance.get("permutation_pvalues_abi"))
            else "ranked_replay"
        )
        return CapabilityValue(
            {
                "placement": backend,
                "static_family": static_mode,
                "adaptive_or_generated_family": "ranked_replay",
            },
            "runtime",
            "Rust owns family-wise exceedance counts; each null family is scored "
            "and reduced on the observed GPU backend.",
        )
    return CapabilityValue(
        None,
        "unknown",
        "Permutation placement depends on the selected backend.",
    )


def _stability_significance(backend: str | None) -> CapabilityValue:
    if backend is None:
        return CapabilityValue(
            None,
            "unknown",
            "Stability placement depends on the selected backend. "
            f"{BOOTSTRAP_STABILITY_SCOPE}",
        )
    return CapabilityValue(
        {"placement": "gafime_cpu", "mode": "selected_candidate_bootstrap"},
        "static",
        "GPU observations retain backend-compatible MI settings, but bootstrap "
        f"resampling currently executes on CPU. {BOOTSTRAP_STABILITY_SCOPE}",
    )


def _mi_estimator(backend: str | None, mi_approximate: bool) -> CapabilityValue:
    if backend is None:
        return CapabilityValue(
            None, "unknown", "Estimator depends on the selected backend."
        )
    if backend == "core" and not mi_approximate:
        return CapabilityValue("adaptive_quantile", "static")
    return CapabilityValue("fixed_equal_width_adaptive_template", "static")


def _mi_bin_ceiling(backend: str | None, requested: int) -> CapabilityValue:
    if backend is None:
        return CapabilityValue(
            None, "unknown", "Bin ceiling depends on the selected backend."
        )
    backend_ceiling = 48 if backend == "metal" else 96
    capped = min(max(requested, 2), backend_ceiling)
    effective = max(level for level in _MI_TEMPLATE_LEVELS if level <= capped)
    return CapabilityValue(
        {
            "configured_max": requested,
            "backend_max": backend_ceiling,
            "effective_template_ceiling": effective,
            "templates": _MI_TEMPLATE_LEVELS,
        },
        "static",
        "Sample count selects a template at or below this ceiling.",
    )


def _precision_contract(
    backend: str | None,
    selected_backend: str | None,
    runtime: Mapping[str, object],
    precision: str,
) -> CapabilityValue:
    requested = precision
    rejection_reason = backend_precision_error(backend or "auto", precision)
    runtime_precision = _mapping_or_empty(runtime.get("precision"))

    if backend == "core":
        supported_profiles = SUPPORTED_PRECISIONS
        interaction_arithmetic = "float64" if precision == "fp64" else "float32"
        reduction_dtype = "float32" if precision == "fp32" else "float64"
        accumulators = {
            metric: reduction_dtype
            for metric in ("pearson", "r2", "spearman", "mutual_info")
        }
        result_dtype = reduction_dtype
        scale_normalization = f"centered_{reduction_dtype}_reduction"
        compensated_summation = False
        interaction_overflow_diagnostics = True
        source = "static"
    elif backend in {"cuda", "rocm", "metal"}:
        default_profiles = ("fp32",) if backend == "metal" else SUPPORTED_PRECISIONS
        supported_profiles = tuple(runtime_precision.get("profiles", default_profiles))
        profile_domains = _mapping_or_empty(runtime_precision.get("profile_domains"))
        requested_domains = _mapping_or_empty(profile_domains.get(precision))
        reduction_dtype = "float32" if precision == "fp32" else "float64"
        interaction_arithmetic = requested_domains.get(
            "interaction_arithmetic",
            runtime_precision.get(
                "interaction_arithmetic",
                "float64" if precision == "fp64" else "float32",
            ),
        )
        if requested_domains:
            accumulators = dict(
                _mapping_or_empty(requested_domains.get("accumulators"))
            )
            source = "runtime"
        elif runtime_precision.get("accumulators"):
            accumulators = dict(
                _mapping_or_empty(runtime_precision.get("accumulators"))
            )
            source = "runtime"
        else:
            accumulators = {
                metric: reduction_dtype
                for metric in ("pearson", "r2", "spearman", "mutual_info")
            }
            source = "static"
        reduction_dtype = requested_domains.get("reduction_dtype", reduction_dtype)
        result_dtype = requested_domains.get(
            "result_dtype", runtime_precision.get("result_dtype", reduction_dtype)
        )
        scale_normalization = runtime_precision.get(
            "scale_normalization", "adaptive_high_dynamic"
        )
        compensated_summation = bool(
            runtime_precision.get("compensated_summation", False)
        )
        interaction_overflow_diagnostics = bool(
            runtime_precision.get("interaction_overflow_diagnostics", False)
        )
    else:
        supported_profiles = SUPPORTED_PRECISIONS
        interaction_arithmetic = None
        reduction_dtype = None
        accumulators = {}
        result_dtype = None
        scale_normalization = None
        compensated_summation = False
        interaction_overflow_diagnostics = False
        source = "unknown"

    if rejection_reason is None and precision not in supported_profiles:
        rejection_reason = (
            f"backend={backend!r} does not advertise precision={precision!r}; "
            f"supported profiles are {', '.join(supported_profiles)}."
        )
    request_supported = rejection_reason is None
    effective = (
        precision if request_supported and selected_backend is not None else None
    )
    detail = rejection_reason
    if detail is None and selected_backend is None:
        detail = "Effective precision requires a selected backend."
    return CapabilityValue(
        {
            "requested": requested,
            "effective": effective,
            "request_supported": request_supported,
            "rejection_reason": rejection_reason,
            "supported_profiles": supported_profiles,
            "storage_dtype": "float64" if precision == "fp64" else "float32",
            "interaction_arithmetic": interaction_arithmetic,
            "reduction_dtype": reduction_dtype,
            "accumulators": accumulators,
            "result_dtype": result_dtype,
            "scale_normalization": scale_normalization,
            "compensated_summation": compensated_summation,
            "interaction_overflow_diagnostics": interaction_overflow_diagnostics,
        },
        source,
        detail,
    )


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
