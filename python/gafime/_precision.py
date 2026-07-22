from __future__ import annotations


SUPPORTED_STORAGE_DTYPES = ("float32",)
SUPPORTED_COMPUTE_POLICIES = ("stable",)

_STORAGE_DTYPE_ALIASES = {
    "f32": "float32",
    "fp32": "float32",
    "float32": "float32",
    "f64": "float64",
    "fp64": "float64",
    "float64": "float64",
}
_COMPUTE_POLICY_ALIASES = {
    "fast": "fast",
    "stable": "stable",
    "exact": "exact",
}


def normalize_storage_dtype(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("storage_dtype must be a string.")
    try:
        return _STORAGE_DTYPE_ALIASES[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "storage_dtype must be one of: float32, float64."
        ) from exc


def normalize_compute_policy(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("compute_policy must be a string.")
    try:
        return _COMPUTE_POLICY_ALIASES[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "compute_policy must be one of: fast, stable, exact."
        ) from exc


def unsupported_precision_reason(storage_dtype: str, compute_policy: str) -> str | None:
    if storage_dtype == "float64":
        return (
            "float64 storage is reserved in the v1 ABI but no current Core, CUDA, "
            "ROCm, or Metal execution path accepts an f64 matrix upload"
        )
    if compute_policy == "exact":
        return (
            "the exact compute policy requires a true f64 ingest, interaction, "
            "reduction, and result contract, which is not implemented"
        )
    if compute_policy == "fast":
        return (
            "the stable policy already selects the tuned fast kernel for safe "
            "input ranges; disabling its high-dynamic normalization guard is not "
            "a supported numerical contract"
        )
    return None
