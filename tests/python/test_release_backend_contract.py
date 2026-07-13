from __future__ import annotations

from pathlib import Path
import sys

import pytest


_RELEASE_MEASURE = Path(__file__).resolve().parents[1] / "release_measure"
if str(_RELEASE_MEASURE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_MEASURE))

from _backend_contract import is_unconfigured_payload_error  # noqa: E402


@pytest.mark.parametrize(
    ("backend", "env_name"),
    [
        ("cuda", "GAFIME_CUDA_V1_LIB"),
        ("rocm", "GAFIME_ROCM_V1_LIB"),
        ("metal", "GAFIME_METAL_V1_LIB"),
    ],
)
def test_only_exact_unconfigured_payload_value_error_skips(
    monkeypatch, backend, env_name
):
    monkeypatch.delenv(env_name, raising=False)
    exact = ValueError(f"v1 GPU boundary error: {env_name} is not set")
    assert is_unconfigured_payload_error(backend, exact)

    assert not is_unconfigured_payload_error(backend, RuntimeError(str(exact)))
    assert not is_unconfigured_payload_error(
        backend, AssertionError(f"unexpected regression mentions {env_name}")
    )
    assert not is_unconfigured_payload_error(
        backend, ValueError(f"prefix: {env_name} is not set")
    )

    monkeypatch.setenv(env_name, "/configured/payload")
    assert not is_unconfigured_payload_error(backend, exact)


def test_core_errors_never_classify_as_missing_payload():
    assert not is_unconfigured_payload_error(
        "core", ValueError("v1 GPU boundary error: GAFIME_CUDA_V1_LIB is not set")
    )
