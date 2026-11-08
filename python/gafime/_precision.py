from __future__ import annotations


SUPPORTED_PRECISIONS = ("fp32", "mixed", "fp64")


def normalize_precision(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("precision must be a string.")
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_PRECISIONS:
        raise ValueError("precision must be one of: fp32, mixed, fp64.")
    return normalized


def backend_precision_error(backend: str, precision: str) -> str | None:
    normalized_backend = backend.strip().lower()
    normalized_precision = normalize_precision(precision)
    if normalized_backend == "metal" and normalized_precision != "fp32":
        return (
            f"Metal supports precision='fp32' only; precision={normalized_precision!r} "
            "requires native fp64 arithmetic and must use Core, CUDA, or ROCm"
        )
    return None
