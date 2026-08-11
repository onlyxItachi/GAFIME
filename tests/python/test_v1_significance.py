"""P-A: native permutation-test + stability significance, end to end.

Requires the built native boundary (`gafime.gafime_py`); skips cleanly otherwise.
Self-paths to the wheel source (`python/`) so it exercises the shipped package,
not the legacy `./gafime/` at the repo root. Determinism comes from a fixed data
seed plus the native seeded permutation stream (`random_seed`), so the stochastic
assertions are reproducible.
"""

from __future__ import annotations

import random
import os
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

# Resolve the v1 wheel source (python/), not the legacy ./gafime/ at the repo root.
_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))

# The significance path is native; skip when the wheel is not built.
pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine, compile as gafime_compile  # noqa: E402


def _dataset(target_fn, *, n=160, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        f0 = rng.gauss(0.0, 1.0)
        f1 = rng.gauss(0.0, 1.0)
        f2 = rng.gauss(0.0, 1.0)
        X.append([f0, f1, f2])
        y.append(target_fn(f0, f1, f2, rng))
    return X, y


def _config():
    return EngineConfig(
        metric_names=("pearson", "r2"),
        permutation_tests=50,
        num_repeats=5,
        random_seed=7,
        budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=100),
    )


_U64_MASK = (1 << 64) - 1


def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & _U64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return state, value ^ (value >> 31)


def _permutation_target(target, base_seed, permutation_index):
    state = (
        base_seed
        ^ ((0xA5A5A5A5 * 0x9E3779B97F4A7C15) & _U64_MASK)
        ^ ((permutation_index * 0xD1B54A32D192ED03) & _U64_MASK)
    )
    _, seed = _splitmix64(state)
    shuffled = list(target)
    state = seed
    for index in range(len(shuffled) - 1, 0, -1):
        state, value = _splitmix64(state)
        other = value % (index + 1)
        shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
    return shuffled


def _extremeness(metric, value):
    if not np.isfinite(value):
        return float("-inf")
    return abs(value) if metric in {"pearson", "spearman"} else value


def test_strong_signal_is_detected_with_low_pvalue_and_stability():
    X, y = _dataset(lambda a, b, c, rng: 3.0 * a + 0.01 * rng.gauss(0.0, 1.0))
    report = GafimeEngine(_config()).analyze(X, y, feature_names=["f0", "f1", "f2"])

    assert len(report.permutations) > 0, "permutation results must be populated"
    assert len(report.stability) > 0, "stability results must be populated"
    assert report.decision.signal_detected is True

    # Every candidate carries a per-metric p-value dict and a stability std dict.
    for perm, stab in zip(report.permutations, report.stability):
        assert set(perm.p_values) == {"pearson", "r2"}
        assert set(stab.metrics_std) == {"pearson", "r2"}

    # The driving feature reaches the permutation floor 1/(50+1) ~= 0.0196.
    best_pearson_p = min(perm.p_values["pearson"] for perm in report.permutations)
    assert best_pearson_p <= 0.05, f"strong-signal p={best_pearson_p}"


@pytest.mark.parametrize(
    ("precision", "typecode"),
    [("fp32", "f"), ("mixed", "d"), ("fp64", "d")],
)
def test_native_significance_vectors_preserve_public_result_dtype(precision, typecode):
    X, y = _dataset(lambda a, b, c, rng: a + 0.1 * b, n=32)
    config = replace(
        _config(),
        precision=precision,
        permutation_tests=3,
        num_repeats=2,
    )
    report = GafimeEngine(config).analyze(X, y, feature_names=["f0", "f1", "f2"])
    native = report.interactions.native_handle

    for getter_name in (
        "significance_pvalues",
        "significance_means",
        "significance_stds",
    ):
        rows = getattr(native, getter_name)()
        assert rows
        assert all(row.typecode == typecode for row in rows)


def test_pure_noise_is_not_detected_under_family_wise_maxt():
    X, y = _dataset(lambda a, b, c, rng: rng.gauss(0.0, 1.0))
    report = GafimeEngine(_config()).analyze(X, y, feature_names=["f0", "f1", "f2"])

    # Permutation/stability still computed and reported...
    assert len(report.permutations) > 0
    # ...but the family-wise (maxT) correction must reject noise: searching many
    # candidates must not manufacture a significant hit.
    assert report.decision.signal_detected is False, report.decision.message


def test_permutation_tests_zero_skips_significance_but_still_analyzes():
    cfg = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )
    X, y = _dataset(lambda a, b, c, rng: 3.0 * a, n=32)
    report = GafimeEngine(cfg).analyze(X, y, feature_names=["f0", "f1", "f2"])

    assert len(report.interactions) > 0
    assert list(report.permutations) == []
    assert list(report.stability) == []
    # Decision still resolves (interactions-based) without significance.
    assert report.decision is not None


def test_significance_report_cap_is_independent_of_higher_order_feature_cap():
    cfg = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=3,
        num_repeats=1,
        significance_top_n=3,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=8,
            top_features_for_higher_k=1,
        ),
    )
    X, y = _dataset(lambda a, b, c, rng: 2.0 * a + b, n=48)

    report = GafimeEngine(cfg).analyze(X, y, feature_names=["f0", "f1", "f2"])

    assert [item.combo for item in report.interactions] == [(0,), (1,), (2,)]
    assert len(report.permutations) == 3


def test_adaptive_maxt_rescreens_higher_order_shortlist_per_permutation():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(32, 18)).astype(np.float32)
    y = rng.normal(size=32).astype(np.float32)
    cfg = EngineConfig(
        backend="cpu",
        metric_names=("pearson",),
        permutation_tests=199,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=5_000,
            top_features_for_higher_k=5,
        ),
    )

    report = GafimeEngine(cfg).analyze(X, y)
    unary_shortlist = sorted(
        (item for item in report.interactions if len(item.combo) == 1),
        key=lambda item: abs(item.metrics["pearson"]),
        reverse=True,
    )[:5]
    candidate = next(item for item in report.interactions if item.combo == (0, 3))
    permutation = next(item for item in report.permutations if item.combo == (0, 3))

    assert [item.combo[0] for item in unary_shortlist] == [0, 14, 3, 12, 1]
    assert candidate.metrics["pearson"] == pytest.approx(0.524196982, abs=1e-7)
    assert permutation.p_values["pearson"] == pytest.approx(0.060, abs=1e-7)
    assert permutation.p_values["pearson"] > cfg.permutation_p_threshold


def test_adaptive_maxt_replay_respects_effective_candidate_feature_count():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(64, 100)).astype(np.float32)
    y = (
        0.8 * X[:, 0]
        - 0.55 * X[:, 3]
        + 0.65 * X[:, 1] * X[:, 7]
        + rng.normal(scale=0.35, size=64)
    ).astype(np.float32)
    permutations = 17
    seed = 37
    budget = ComputeBudget(
        max_comb_size=2,
        max_combinations_per_k=50,
        top_features_for_higher_k=20,
        max_feature_candidate=20,
    )
    tested_config = EngineConfig(
        backend="cpu",
        metric_names=("pearson",),
        permutation_tests=permutations,
        num_repeats=1,
        random_seed=seed,
        significance_top_n=50,
        budget=budget,
    )

    tested_artifact = gafime_compile(X, y, config=tested_config)
    try:
        observed_report = tested_artifact.analyze()
    finally:
        tested_artifact.close()

    replay_config = EngineConfig(
        backend="cpu",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=seed,
        budget=budget,
    )
    replay_artifact = gafime_compile(X, y, config=replay_config)
    null_maxima = []
    try:
        for permutation_index in range(permutations):
            replay_artifact.update_target(
                _permutation_target(y, seed, permutation_index)
            )
            replay_report = replay_artifact.analyze()
            null_maxima.append(
                max(
                    _extremeness("pearson", item.metrics["pearson"])
                    for item in replay_report.interactions
                )
            )
    finally:
        replay_artifact.close()

    observed_by_combo = {
        item.combo: item.metrics["pearson"] for item in observed_report.interactions
    }
    assert len(observed_report.permutations) == 50
    assert all(
        feature < 20 for item in observed_report.interactions for feature in item.combo
    )
    for item in observed_report.permutations:
        observed = _extremeness("pearson", observed_by_combo[item.combo])
        exceedances = sum(maximum >= observed for maximum in null_maxima)
        expected = (exceedances + 1) / (permutations + 1)
        assert item.p_values["pearson"] == pytest.approx(expected, abs=1.0e-7)


def test_cuda_adaptive_maxt_uses_device_shortlist_and_restores_target():
    if not os.environ.get("GAFIME_CUDA_V1_LIB"):
        pytest.skip("GAFIME_CUDA_V1_LIB is not configured")

    rng = random.Random(16)
    X = [[float(rng.randrange(11)) for _ in range(8)] for _ in range(64)]
    y = [float(rng.randrange(11)) for _ in range(64)]
    cfg = EngineConfig(
        backend="cuda",
        metric_names=("mutual_info",),
        permutation_tests=3,
        num_repeats=1,
        random_seed=17,
        mi_bins=16,
        mi_approximate=True,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=100,
            top_features_for_higher_k=2,
            keep_in_vram=False,
        ),
    )

    artifact = gafime_compile(X, y, config=cfg)
    try:
        first = artifact.analyze()
        second = artifact.analyze()
    finally:
        artifact.close()

    first_higher = [item.combo for item in first.interactions if len(item.combo) > 1]
    second_higher = [item.combo for item in second.interactions if len(item.combo) > 1]
    assert first_higher == [(7, 5)]
    assert second_higher == first_higher
    assert [item.p_values for item in second.permutations] == [
        item.p_values for item in first.permutations
    ]


@pytest.mark.parametrize(
    ("backend", "payload_env"),
    [
        ("cuda", "GAFIME_CUDA_V1_LIB"),
        ("rocm", "GAFIME_ROCM_V1_LIB"),
        ("metal", "GAFIME_METAL_V1_LIB"),
    ],
)
def test_gpu_adaptive_maxt_matches_exhaustive_same_device_oracle(backend, payload_env):
    if not os.environ.get(payload_env):
        pytest.skip(f"{payload_env} is not configured")

    rng = random.Random(29)
    X = []
    y = []
    for row_index in range(48):
        row = [rng.uniform(-1.5, 1.5) for _ in range(6)]
        X.append(row)
        y.append(
            -2.4 * row[0]
            + 1.1 * row[1] * row[2]
            - 0.7 * row[2] * row[3] * row[4]
            + 0.03 * ((row_index % 7) - 3)
        )

    metrics = ("pearson", "spearman", "mutual_info", "r2")
    permutations = 5
    seed = 41
    budget = ComputeBudget(
        max_comb_size=3,
        max_combinations_per_k=100,
        top_features_for_higher_k=4,
        keep_in_vram=False,
    )
    tested_config = EngineConfig(
        backend=backend,
        metric_names=metrics,
        permutation_tests=permutations,
        num_repeats=1,
        random_seed=seed,
        significance_top_n=12,
        mi_bins=8,
        mi_approximate=True,
        budget=budget,
    )
    tested_artifact = gafime_compile(X, y, config=tested_config)
    try:
        observed_report = tested_artifact.analyze()
        replay_report = tested_artifact.analyze()
    finally:
        tested_artifact.close()

    observed_by_combo = {
        item.combo: item.metrics for item in observed_report.interactions
    }
    assert any(len(combo) == 3 for combo in observed_by_combo)
    assert observed_by_combo[(0,)]["pearson"] < -0.8

    oracle_config = EngineConfig(
        backend=backend,
        metric_names=metrics,
        permutation_tests=0,
        num_repeats=1,
        random_seed=seed,
        mi_bins=8,
        mi_approximate=True,
        budget=budget,
    )
    oracle_artifact = gafime_compile(X, y, config=oracle_config)
    null_maxima = []
    try:
        for permutation_index in range(permutations):
            oracle_artifact.update_target(
                _permutation_target(y, seed, permutation_index)
            )
            null_report = oracle_artifact.analyze()
            null_maxima.append(
                {
                    metric: max(
                        _extremeness(metric, item.metrics[metric])
                        for item in null_report.interactions
                    )
                    for metric in metrics
                }
            )
    finally:
        oracle_artifact.close()

    for item in observed_report.permutations:
        observed_metrics = observed_by_combo[item.combo]
        for metric in metrics:
            observed = _extremeness(metric, observed_metrics[metric])
            exceedances = sum(maximum[metric] >= observed for maximum in null_maxima)
            expected = (exceedances + 1) / (permutations + 1)
            assert item.p_values[metric] == pytest.approx(expected, abs=1.0e-7)

    assert [item.p_values for item in replay_report.permutations] == [
        item.p_values for item in observed_report.permutations
    ]


@pytest.mark.parametrize(
    ("backend", "payload_env"),
    [
        ("cuda", "GAFIME_CUDA_V1_LIB"),
        ("rocm", "GAFIME_ROCM_V1_LIB"),
        ("metal", "GAFIME_METAL_V1_LIB"),
    ],
)
def test_gpu_static_maxt_preserves_observed_metrics_and_resident_target(
    backend, payload_env
):
    if not os.environ.get(payload_env):
        pytest.skip(f"{payload_env} is not configured")

    X, y = _dataset(lambda a, b, c, rng: 2.0 * a - b, n=32, seed=13)
    budget = ComputeBudget(
        max_comb_size=1,
        max_combinations_per_k=8,
        top_features_for_higher_k=1,
        keep_in_vram=False,
    )
    baseline = GafimeEngine(
        EngineConfig(
            backend=backend,
            metric_names=("pearson", "mutual_info"),
            permutation_tests=0,
            num_repeats=1,
            random_seed=19,
            mi_approximate=True,
            budget=budget,
        )
    ).analyze(X, y)
    cfg = EngineConfig(
        backend=backend,
        metric_names=("pearson", "mutual_info"),
        permutation_tests=3,
        num_repeats=1,
        random_seed=19,
        mi_approximate=True,
        budget=budget,
    )

    artifact = gafime_compile(X, y, config=cfg)
    try:
        first = artifact.analyze()
        second = artifact.analyze()
    finally:
        artifact.close()

    baseline_metrics = [item.metrics for item in baseline.interactions]
    assert [item.metrics for item in first.interactions] == baseline_metrics
    assert [item.metrics for item in second.interactions] == baseline_metrics
    assert [item.p_values for item in second.permutations] == [
        item.p_values for item in first.permutations
    ]
    assert all(
        0.0 < value <= 1.0
        for item in first.permutations
        for value in item.p_values.values()
    )
