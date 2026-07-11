"""Shared backend selection and classification for release-measure gates."""
from __future__ import annotations

import os
from collections.abc import Iterable


PAYLOAD_ENVS = {
    "cuda": "GAFIME_CUDA_V1_LIB",
    "rocm": "GAFIME_ROCM_V1_LIB",
    "metal": "GAFIME_METAL_V1_LIB",
}

_ALIASES = {
    "cpu": "core",
    "hip": "rocm",
}

_EXPECTED_RESOLVED = {
    "core": "v1-rust-cpu",
    "cuda": "v1-cuda-cabi",
    "rocm": "v1-rocm-cabi",
    "metal": "v1-metal-cabi",
}

_VALID_BACKENDS = frozenset((*_EXPECTED_RESOLVED, "auto"))


def normalize_backend(name: str) -> str:
    backend = _ALIASES.get(name.strip().lower(), name.strip().lower())
    if backend not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS | set(_ALIASES)))
        raise ValueError(f"unsupported backend {name!r}; expected one of: {valid}")
    return backend


def selected_backends(default: Iterable[str]) -> tuple[str, ...]:
    raw = os.environ.get("GAFIME_BACKENDS")
    if raw is None:
        raw = os.environ.get("GAFIME_BACKEND")
    names = (
        list(default)
        if raw is None
        else [part for part in raw.split(",") if part.strip()]
    )
    if not names:
        raise ValueError("backend selection is empty")

    selected = []
    for name in names:
        backend = normalize_backend(name)
        if backend not in selected:
            selected.append(backend)
    return tuple(selected)


def payload_env(backend: str) -> str | None:
    return PAYLOAD_ENVS.get(normalize_backend(backend))


def payload_is_configured(backend: str) -> bool:
    env_name = payload_env(backend)
    return env_name is None or bool(os.environ.get(env_name))


def is_unconfigured_payload_error(backend: str, exc: BaseException) -> bool:
    env_name = payload_env(backend)
    return (
        env_name is not None
        and not os.environ.get(env_name)
        and isinstance(exc, ValueError)
        and str(exc) == f"v1 GPU boundary error: {env_name} is not set"
    )


def assert_resolved_backend(requested: str, info: object) -> None:
    backend = normalize_backend(requested)
    if info is None:
        raise AssertionError(f"{backend} analysis did not report backend information")
    resolved = str(getattr(info, "name", "") or "")
    if backend == "auto":
        allowed = set(_EXPECTED_RESOLVED.values())
        if resolved not in allowed:
            raise AssertionError(
                f"auto resolved to unexpected backend {resolved!r}; "
                f"expected one of {sorted(allowed)}"
            )
        return
    expected = _EXPECTED_RESOLVED[backend]
    if resolved != expected:
        raise AssertionError(
            f"explicit {backend} request resolved to {resolved!r}, expected {expected!r}"
        )
