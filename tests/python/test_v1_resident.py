"""P-D: resident-session reuse — compile once, swap the target, re-analyze.

The compiled artifact keeps the feature matrix resident (uploaded on GPU / held
on CPU); update_target replaces only y, so a re-analyze reuses the resident
matrix and must match a fresh compile+analyze with the same y.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402

# Import the compile *function* from the subpackage: `gafime.compile` is both a
# function and a submodule, and once the submodule is imported elsewhere it
# shadows the function on the top-level package.
from gafime.compile import compile as gafime_compile  # noqa: E402


def _metrics_by_combo(report):
    return {tuple(it.combo): it.metrics["pearson"] for it in report.interactions}


def test_update_target_reuses_resident_matrix_and_matches_fresh():
    X = [[float(i), float((i * i) % 5)] for i in range(30)]
    y1 = [float(i) for i in range(30)]           # tracks feature 0
    y2 = [float((i * i) % 5) for i in range(30)]  # tracks feature 1
    cfg = EngineConfig(
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )

    artifact = gafime_compile(X, y1, ["a", "b"], config=cfg)
    m1 = _metrics_by_combo(artifact.analyze())

    artifact.update_target(y2)
    m2 = _metrics_by_combo(artifact.analyze())

    fresh = _metrics_by_combo(gafime_compile(X, y2, ["a", "b"], config=cfg).analyze())

    # Reused-with-new-target must equal a fresh compile+analyze with that target.
    assert set(m2) == set(fresh)
    for combo in m2:
        assert m2[combo] == pytest.approx(fresh[combo], abs=1e-6)
    # And the swap actually changed the result vs the original target.
    assert any(abs(m1[c] - m2[c]) > 1e-6 for c in m1)


def test_update_target_invalidates_cached_interaction_diagnostics():
    X = [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]]
    y = [1.0, 2.0, 3.0, 4.0]
    cfg = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )

    artifact = gafime_compile(X, y, ["a", "b"], config=cfg)
    try:
        before = artifact.analyze()
        assert all(not item.source_nonfinite for item in before.interactions)

        artifact.update_target([float("nan"), 2.0, 3.0, 4.0])
        after = artifact.analyze()
        assert all(item.source_nonfinite for item in after.interactions)
    finally:
        artifact.close()


def test_update_target_validates_length_and_closed():
    X = [[float(i)] for i in range(8)]
    y = [float(i) for i in range(8)]
    cfg = EngineConfig(metric_names=("pearson",), budget=ComputeBudget(max_comb_size=1))
    artifact = gafime_compile(X, y, ["a"], config=cfg)
    with pytest.raises(Exception):
        artifact.update_target([1.0, 2.0])  # wrong length
    artifact.close()
    with pytest.raises(Exception):
        artifact.update_target(y)  # closed


def test_legacy_seeded_screening_matches_published_candidate_order():
    y = [float(index) for index in range(8)]
    tied = [[value, value, value, value] for value in y]
    cfg = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(
            max_comb_size=3,
            max_combinations_per_k=10,
            top_features_for_higher_k=3,
            keep_in_vram=False,
        ),
    )

    report = GafimeEngine(cfg).analyze(tied, y, ["a", "b", "c", "d"])

    assert [item.combo for item in report.interactions] == [
        (0,),
        (1,),
        (2,),
        (3,),
        (2, 0),
        (2, 1),
        (0, 1),
        (2, 0, 1),
    ]


def test_legacy_unary_cap_and_top_one_screening_are_honored():
    y = [float(index) for index in range(8)]
    six_features = [
        [float(row + feature) for feature in range(6)] for row in range(8)
    ]
    capped = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(
            max_comb_size=1,
            max_combinations_per_k=3,
            top_features_for_higher_k=6,
            keep_in_vram=False,
        ),
    )
    capped_report = GafimeEngine(capped).analyze(six_features, y)
    assert [item.combo for item in capped_report.interactions] == [(4,), (0,), (5,)]

    top_one = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=10,
            top_features_for_higher_k=1,
            keep_in_vram=False,
        ),
    )
    tied = [[value, value, value, value] for value in y]
    top_one_report = GafimeEngine(top_one).analyze(tied, y)
    assert [item.combo for item in top_one_report.interactions] == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (-7, [6, 7, 2, 4]),
        (2**64 + 7, [5, 6, 7, 4]),
        (2**130 + 12345, [1, 0, 3, 2]),
    ],
)
def test_legacy_python_integer_seed_domain_is_preserved(seed, expected):
    X = [[float(row + feature) for feature in range(8)] for row in range(12)]
    y = [float(row) for row in range(12)]
    cfg = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=seed,
        budget=ComputeBudget(
            max_comb_size=1,
            max_combinations_per_k=4,
            top_features_for_higher_k=4,
            keep_in_vram=False,
        ),
    )

    report = GafimeEngine(cfg).analyze(X, y)

    assert [item.combo[0] for item in report.interactions] == expected


def test_legacy_core_ties_follow_scheduler_feature_order():
    y = [float(index) for index in range(12)]
    X = [[value] * 5 for value in y]
    cfg = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(
            max_comb_size=3,
            max_combinations_per_k=3,
            top_features_for_higher_k=3,
            keep_in_vram=False,
        ),
    )

    report = GafimeEngine(cfg).analyze(X, y)

    assert [item.combo for item in report.interactions] == [
        (4,),
        (0,),
        (3,),
        (3, 4),
        (3, 0),
        (4, 0),
        (3, 4, 0),
    ]


def test_update_target_rebuilds_screened_candidate_plan():
    X = [
        [
            float(index),
            float(index % 2),
            float((index * 7) % 11),
            float((index * index) % 13),
            float(-index),
        ]
        for index in range(32)
    ]
    y1 = [row[0] for row in X]
    y2 = [row[2] for row in X]
    cfg = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=100,
            top_features_for_higher_k=2,
        ),
    )

    artifact = gafime_compile(X, y1, config=cfg)
    before = artifact.analyze()
    artifact.update_target(y2)
    after = artifact.analyze()
    fresh = gafime_compile(X, y2, config=cfg).analyze()

    assert [item.combo for item in before.interactions][-1] == (0, 4)
    assert [item.combo for item in after.interactions][-1] == (2, 1)
    assert [item.combo for item in after.interactions] == [
        item.combo for item in fresh.interactions
    ]
    assert [item.metrics for item in after.interactions] == [
        item.metrics for item in fresh.interactions
    ]


@pytest.mark.parametrize(
    ("backend", "payload_env"),
    [
        ("cuda", "GAFIME_CUDA_V1_LIB"),
        ("rocm", "GAFIME_ROCM_V1_LIB"),
        ("metal", "GAFIME_METAL_V1_LIB"),
    ],
)
def test_gpu_update_target_invalidates_resident_screened_descriptors(
    backend, payload_env
):
    if not os.environ.get(payload_env):
        pytest.skip(f"{payload_env} is not configured")

    X = [
        [
            float(index),
            float(index % 2),
            float((index * 7) % 11),
            float((index * index) % 13),
            float(-index),
        ]
        for index in range(32)
    ]
    y1 = [row[0] for row in X]
    y2 = [row[2] for row in X]
    cfg = EngineConfig(
        backend=backend,
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=100,
            top_features_for_higher_k=2,
        ),
    )

    artifact = gafime_compile(X, y1, config=cfg)
    artifact.analyze()
    artifact.update_target(y2)
    after = artifact.analyze()
    fresh = gafime_compile(X, y2, config=cfg).analyze()

    assert [item.combo for item in after.interactions][-1] == (2, 1)
    assert [item.combo for item in after.interactions] == [
        item.combo for item in fresh.interactions
    ]
    assert [item.metrics for item in after.interactions] == [
        item.metrics for item in fresh.interactions
    ]
