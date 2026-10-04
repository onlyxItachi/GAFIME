"""Independent temporal maxT and public compiled-lifecycle regressions.

Core is the default. A separately qualified native lane can be selected with
GAFIME_TEMPORAL_TEST_BACKEND; selecting it explicitly never permits fallback.
The scalar oracle below does not call GAFIME generation, selection, scoring,
planning, or significance. Native eager/compiled parity is a separate check.
"""

from __future__ import annotations

import math
import os
import random
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1":
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402
import gafime.v1_adapter as adapter  # noqa: E402

_MASK = (1 << 64) - 1
_BACKEND = os.environ.get("GAFIME_TEMPORAL_TEST_BACKEND", "core")


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & _MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return state, (value ^ (value >> 31)) & _MASK


def _permuted(target, seed, index):
    state = (
        seed
        ^ ((0xA5A5_A5A5 * 0x9E3779B97F4A7C15) & _MASK)
        ^ ((index * 0xD1B54A32D192ED03) & _MASK)
    )
    _, state = _splitmix(state)
    result = list(target)
    for position in range(len(result) - 1, 0, -1):
        state, value = _splitmix(state)
        swap = value % (position + 1)
        result[position], result[swap] = result[swap], result[position]
    return result


def _pearson(column, target):
    pairs = [
        (x, y) for x, y in zip(column, target) if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return math.nan
    mx = math.fsum(x for x, _ in pairs) / len(pairs)
    my = math.fsum(y for _, y in pairs) / len(pairs)
    covariance = math.fsum((x - mx) * (y - my) for x, y in pairs)
    variance = math.fsum((x - mx) ** 2 for x, _ in pairs) * math.fsum(
        (y - my) ** 2 for _, y in pairs
    )
    return covariance / math.sqrt(variance) if variance > 0 else math.nan


def _strength(score):
    return abs(score) if math.isfinite(score) else -math.inf


def _unary_order(count, cap, seed):
    order = list(range(count))
    if count > cap:
        random.Random(seed).shuffle(order)
        del order[cap:]
    return order


def _reference_expansion(X, target, config):
    """Reference for lag=1/no-window tests, including both independent caps."""
    cast = float if config.precision == "fp64" else _f32
    columns = [[cast(value) for value in column] for column in zip(*X)]
    target = [cast(value) for value in target]
    count = len(columns)
    cap = config.budget.max_combinations_per_k
    eligible = _unary_order(count, cap, config.random_seed)
    sources = sorted(
        eligible, key=lambda j: (-_strength(_pearson(columns[j], target)), j)
    )[: config.budget.top_k_features_for_time_series]
    generated, labels = [], []
    for source in sources:
        column = columns[source]
        lag = [math.nan, *column[:-1]]
        delta = [math.nan] + [
            cast(column[i] - column[i - 1]) for i in range(1, len(column))
        ]
        acceleration = [math.nan, math.nan] + [
            cast(cast(column[i] - cast(2 * column[i - 1])) + column[i - 2])
            for i in range(2, len(column))
        ]
        generated.extend([lag, delta, delta, acceleration])
        labels.extend(
            f"f{source}_{op}1" for op in ("lag", "delta", "velocity", "acceleration")
        )
    limit = config.budget.max_time_series_candidates
    return (
        columns + generated[:limit],
        [f"f{j}" for j in range(count)] + labels[:limit],
        sources,
    )


def _fixture():
    rng = random.Random(116)
    X = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(32)]
    target = [rng.gauss(0, 1) for _ in X]
    return X, target


def _config(precision, **changes):
    if _BACKEND == "metal" and precision != "fp32":
        pytest.skip("Metal's supported temporal precision is fp32 only")
    return replace(
        EngineConfig(
            backend=_BACKEND,
            precision=precision,
            random_seed=7,
            metric_names=("pearson",),
            permutation_tests=31,
            num_repeats=1,
            enable_time_series_functions=True,
            time_series_lags=(1,),
            time_series_windows=(),
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=64,
                top_k_features_for_time_series=2,
                max_time_series_candidates=6,
            ),
        ),
        **changes,
    )


def _assert_same_report(actual, expected):
    assert actual.feature_names == expected.feature_names
    assert [
        (row.candidate_id, row.combo, row.expression, row.family, row.params)
        for row in actual.interactions
    ] == [
        (row.candidate_id, row.combo, row.expression, row.family, row.params)
        for row in expected.interactions
    ]
    for left, right in zip(actual.interactions, expected.interactions):
        assert left.metrics == pytest.approx(right.metrics, nan_ok=True)
    assert actual.permutations == expected.permutations
    assert actual.stability == expected.stability


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize(
    "top_k,generated_cap,significance_top_n", [(2, 6, 50), (8, 1, 1)]
)
def test_time_series_maxt_matches_independent_adaptive_reference(
    precision, top_k, generated_cap, significance_top_n
):
    X, target = _fixture()
    config = _config(precision, significance_top_n=significance_top_n)
    config = replace(
        config,
        budget=replace(
            config.budget,
            top_k_features_for_time_series=top_k,
            max_time_series_candidates=generated_cap,
        ),
    )
    cast = float if precision == "fp64" else _f32
    stored_target = [cast(value) for value in target]
    columns, names, sources = _reference_expansion(X, stored_target, config)
    report = GafimeEngine(config).analyze(X, target)
    assert report.feature_names == names
    maxima, changed = [], 0
    for index in range(config.permutation_tests):
        permuted = _permuted(stored_target, config.random_seed, index)
        null_columns, null_names, null_sources = _reference_expansion(
            X, permuted, config
        )
        changed += null_sources != sources and null_names != names
        maxima.append(
            max(_strength(_pearson(column, permuted)) for column in null_columns)
        )
    assert changed > 0, "fixture must change the adaptive generated family"
    scores = {}
    for row in report.interactions:
        expected = _pearson(columns[row.combo[0]], stored_target)
        assert row.metrics["pearson"] == pytest.approx(expected, abs=3e-6)
        scores[row.candidate_id] = _strength(expected)
    assert report.permutations
    for result in report.permutations:
        observed = scores[result.candidate_id]
        # Exact count comparisons must not depend on fp32 rounding near a tie.
        assert min(abs(value - observed) for value in maxima) > 3e-6
        expected = (1 + sum(value >= observed for value in maxima)) / (len(maxima) + 1)
        assert result.p_values["pearson"] == (
            cast(expected) if precision == "fp32" else expected
        )


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("generated_cap", [1, 6])
def test_time_series_update_reselects_sources_and_matches_fresh_compile(
    precision, generated_cap
):
    X, _ = _fixture()
    first = [row[0] + 0.2 * row[1] for row in X]
    second = [row[6] + 0.2 * row[7] for row in X]
    config = _config(precision, permutation_tests=7, num_repeats=3)
    config = replace(
        config, budget=replace(config.budget, max_time_series_candidates=generated_cap)
    )
    artifact = GafimeEngine(config).compile(X, first)
    try:
        before = artifact.analyze()
        old_plan = artifact.scenario_plan
        assert artifact.update_target(second) is artifact
        fresh = GafimeEngine(config).analyze(X, second)
        assert artifact.feature_names == fresh.feature_names
        assert artifact.feature_names != before.feature_names
        assert artifact.scenario_plan is not old_plan
        assert artifact.scenario_plan.n_features == len(X[0])
        _assert_same_report(artifact.analyze(), fresh)
        _assert_same_report(artifact.analyze(), fresh)
    finally:
        artifact.close()


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_time_series_none_seed_rebuilds_seed_capped_sources(monkeypatch, precision):
    X, target = _fixture()
    config = _config(precision, random_seed=None, permutation_tests=7, num_repeats=3)
    config = replace(config, budget=replace(config.budget, max_combinations_per_k=3))
    # Compilation consumes the first seed; each analyze consumes another.
    seeds = iter([7, 11, 23, 11])
    monkeypatch.setattr(adapter, "_fresh_random_seed", lambda: next(seeds))
    artifact = GafimeEngine(config).compile(X, target)
    try:
        names, reports = [], []
        for seed in [11, 23, 11]:
            actual = artifact.analyze()
            fixed = replace(config, random_seed=seed)
            _, expected_names, _ = _reference_expansion(X, target, fixed)
            assert artifact.feature_names == expected_names
            expected = GafimeEngine(fixed).analyze(X, target)
            _assert_same_report(actual, expected)
            names.append(actual.feature_names)
            reports.append(actual)
        assert names[0] != names[1], "seed cap fixture must select different sources"
        _assert_same_report(reports[0], reports[2])
    finally:
        artifact.close()


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_time_series_nan_only_maxima_do_not_create_significance(precision):
    config = _config(precision, permutation_tests=3)
    X = [[math.nan, math.inf] for _ in range(8)]
    report = GafimeEngine(config).analyze(X, list(range(8)))
    assert report.permutations
    assert all(result.p_values["pearson"] == 1.0 for result in report.permutations)
