import platform

import numpy as np
import pytest

from gafime.discrete import (
    DiscreteFunctionCandidate,
    score_discrete_candidates,
    score_discrete_selection_candidates,
)
from gafime.metrics import MetricSuite


def _discrete_fixture(seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1024, 5)).astype(np.float32)
    y = (
        0.8 * (1.0 / (1.0 + np.exp(-12.0 * (X[:, 0] - 0.1))))
        + 0.5 * X[:, 2] * (1.0 / (1.0 + np.exp(-12.0 * (X[:, 1] + 0.2))))
        + 0.1 * rng.normal(size=X.shape[0])
    ).astype(np.float32)
    candidates = [
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(0,),
            thresholds=(0.1,),
            direction="ge",
            scales=(1.0,),
            candidate_id="threshold-ge",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(1,),
            thresholds=(-0.2,),
            direction="le",
            scales=(1.0,),
            candidate_id="threshold-le",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_interval",
            feature_indices=(0,),
            intervals=((-0.5, 0.5),),
            scales=(1.0,),
            candidate_id="interval",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_value_gated_threshold",
            feature_indices=(1,),
            thresholds=(-0.2,),
            direction="ge",
            value_feature=2,
            scales=(1.0,),
            candidate_id="value-threshold",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_soft_rectangle",
            feature_indices=(0, 1),
            intervals=((-0.5, 0.5), (-0.25, 0.75)),
            scales=(1.0, 1.0),
            candidate_id="rectangle",
        ),
        DiscreteFunctionCandidate(
            kind="discrete_function_value_in_soft_rectangle",
            feature_indices=(0, 1),
            intervals=((-0.5, 0.5), (-0.25, 0.75)),
            value_feature=3,
            scales=(1.0, 1.0),
            candidate_id="value-rectangle",
        ),
    ]
    return X, y, candidates


def _assert_native_matches_python(backend):
    X, y, candidates = _discrete_fixture()
    suite = MetricSuite(("pearson",))
    native = backend.score_discrete_candidates(X, y, candidates, suite)
    expected = score_discrete_candidates(X, y, candidates, suite)

    for candidate in candidates:
        assert native[candidate]["pearson"] == pytest.approx(
            expected[candidate]["pearson"], abs=3e-4
        )


def _assert_native_selection_matches_python(backend):
    X, y, candidates = _discrete_fixture()
    baseline_pred = 0.35 * X[:, 0] - 0.2 * X[:, 2]
    native = backend.score_discrete_selection_candidates(
        X,
        y,
        candidates,
        baseline_pred=baseline_pred,
        mi_bins=16,
    )
    expected = score_discrete_selection_candidates(
        X,
        y,
        candidates,
        baseline_pred=baseline_pred,
        mi_bins=16,
    )

    for candidate in candidates:
        for name in (
            "mutual_info",
            "variance_reduction",
            "residual_abs_corr",
            "residual_r2_gain",
        ):
            assert native[candidate][name] == pytest.approx(
                expected[candidate][name], abs=2e-3
            )


def test_cuda_soft_discrete_kernel_matches_python():
    from gafime.backends.native_cuda_backend import NativeCudaBackend

    try:
        backend = NativeCudaBackend()
    except Exception as exc:
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if not getattr(backend, "_has_discrete_soft_api", False):
        pytest.skip("CUDA discrete soft native API is not built.")

    _assert_native_matches_python(backend)


def test_cuda_soft_discrete_selection_kernel_matches_python():
    from gafime.backends.native_cuda_backend import NativeCudaBackend

    try:
        backend = NativeCudaBackend()
    except Exception as exc:
        pytest.skip(f"CUDA backend unavailable: {exc}")
    if not getattr(backend, "_has_discrete_selection_api", False):
        pytest.skip("CUDA discrete selection native API is not built.")

    _assert_native_selection_matches_python(backend)


def test_metal_soft_discrete_kernel_matches_python():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal backend requires macOS arm64.")

    from gafime.backends.native_metal_backend import NativeMetalBackend

    try:
        backend = NativeMetalBackend()
    except Exception as exc:
        pytest.skip(f"Metal backend unavailable: {exc}")
    if not getattr(backend.lib, "_gafime_has_discrete_soft_api", False):
        pytest.skip("Metal discrete soft native API is not built.")

    _assert_native_matches_python(backend)
