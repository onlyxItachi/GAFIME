from __future__ import annotations


SUPPORTED_PRECISIONS = ("fp32", "mixed", "fp64")

_LEGACY_PAIR_TO_PROFILE = {
    ("float32", "fast"): "fp32",
    ("float32", "stable"): "mixed",
    ("float64", "exact"): "fp64",
}


def normalize_precision(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("precision must be a string.")
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_PRECISIONS:
        raise ValueError("precision must be one of: fp32, mixed, fp64.")
    return normalized


def precision_from_legacy_pair(storage_dtype: object, compute_policy: object) -> str:
    if not isinstance(storage_dtype, str) or not isinstance(compute_policy, str):
        raise TypeError("legacy storage_dtype and compute_policy must both be strings.")
    storage = storage_dtype.strip().lower()
    if storage not in {"float32", "float64"}:
        raise ValueError("legacy storage_dtype must be float32 or float64.")
    policy = compute_policy.strip().lower()
    if policy not in {"fast", "stable", "exact"}:
        raise ValueError("legacy compute_policy must be fast, stable, or exact.")
    try:
        return _LEGACY_PAIR_TO_PROFILE[(storage, policy)]
    except KeyError as exc:
        raise ValueError(
            "unsupported legacy precision pair; accepted mappings are "
            "float32+fast -> fp32, float32+stable -> mixed, and "
            "float64+exact -> fp64."
        ) from exc


def backend_precision_error(backend: str, precision: str) -> str | None:
    normalized_backend = backend.strip().lower()
    normalized_precision = normalize_precision(precision)
    if normalized_backend == "metal" and normalized_precision != "fp32":
        return (
            f"Metal supports precision='fp32' only; precision={normalized_precision!r} "
            "requires native fp64 arithmetic and must use Core, CUDA, or ROCm"
        )
    return None
