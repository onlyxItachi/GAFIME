"""P-C: native decision_path family, end to end (requires the built wheel).

Enables `enable_decision_path_functions` and checks that the engine discovers
depth-k GBDT conjunction paths, appends their membership columns, and mines them
through the continuous path — recovering an AND-structured signal that a single
raw feature cannot express.
"""

from __future__ import annotations

import math
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))

pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402
from gafime.decision_path import (  # noqa: E402
    DecisionPathCandidate,
    decision_path_candidate_from_result,
    evaluate_decision_path_candidate,
)


def _and_dataset(per_quadrant=20):
    """y is high only where f0 AND f1 are both high — a pure interaction that no
    single raw feature separates."""
    X, y = [], []
    for q0 in (0, 1):
        for q1 in (0, 1):
            for k in range(per_quadrant):
                f0 = (0.2 if q0 == 0 else 0.8) + 0.001 * k
                f1 = (0.2 if q1 == 0 else 0.8) + 0.001 * k
                X.append([f0, f1])
                y.append(5.0 if (q0 == 1 and q1 == 1) else 0.0)
    return X, y


def _config(**overrides) -> EngineConfig:
    base = EngineConfig(
        enable_decision_path_functions=True,
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=64),
        decision_path_max_depth=2,
        decision_path_rounds=1,
        decision_path_max_paths=8,
        decision_path_min_leaf=5,
        decision_path_learning_rate=1.0,
    )
    return replace(base, **overrides) if overrides else base


def test_decision_path_discovers_and_conjunction_feature():
    X, y = _and_dataset()
    report = GafimeEngine(_config()).analyze(X, y, feature_names=["f0", "f1"])

    # Path membership columns are appended after the two base features.
    assert len(report.feature_names) > 2
    path_names = [n for n in report.feature_names if n.startswith("path[")]
    assert path_names, f"expected a path[...] feature, got {report.feature_names}"
    assert report.warnings and "decision_path" in report.warnings[0]

    # The AND region's membership indicator matches y exactly -> near-perfect
    # pearson, which no single raw feature (f0 or f1 alone) can reach.
    best_pearson = max(abs(item.metrics["pearson"]) for item in report.interactions)
    assert best_pearson >= 0.9, (
        f"path feature should strongly separate y, got {best_pearson}"
    )


def test_native_decision_path_result_params_preserve_original_path_nodes():
    X, y = _and_dataset()
    report = GafimeEngine(_config()).analyze(X, y, feature_names=["f0", "f1"])
    result = next(
        item
        for item in report.interactions
        if item.family == "decision_path" and len(item.combo) == 1
    )

    assert set(result.params) == {
        "kind",
        "features",
        "thresholds",
        "signs",
        "gain",
        "support",
        "round_id",
        "native_candidate_id",
        "candidate_id",
    }
    candidate = decision_path_candidate_from_result(result)
    assert candidate.features
    assert all(feature < 2 for feature in candidate.features)
    assert candidate.features != result.combo
    expected_label = "path[{}]".format(
        " & ".join(
            f"{['f0', 'f1'][feature]}{'<=' if sign < 0 else '>'}{threshold:.4f}"
            for feature, threshold, sign in zip(
                candidate.features, candidate.thresholds, candidate.signs
            )
        )
    )
    assert result.feature_names == (expected_label,)
    assert candidate.candidate_id == result.candidate_id
    assert candidate.native_candidate_id == int(result.candidate_id.rsplit(":", 1)[-1])
    assert isinstance(candidate.support, int)
    native_params = report.interactions.native_handle.decision_path_params(
        result.combo[0]
    )
    assert native_params["thresholds"].typecode == "f"
    assert isinstance(native_params["support"], int)
    assert len(evaluate_decision_path_candidate(X, candidate)) == len(X)


def test_portable_decision_path_preserves_native_nan_membership_semantics():
    candidate = DecisionPathCandidate(
        features=(0, 1),
        thresholds=(0.5, 0.5),
        signs=(1, 1),
    )

    membership = evaluate_decision_path_candidate(
        [[float("nan"), 1.0], [float("nan"), 0.0], [1.0, 1.0], [0.0, 1.0]],
        candidate,
    )

    assert math.isnan(membership[0])
    assert membership[1:] == [0.0, 1.0, 0.0]


def test_mixed_raw_and_path_result_is_not_decoded_as_one_path():
    X, y = _and_dataset()
    config = _config(budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=64))
    report = GafimeEngine(config).analyze(X, y, feature_names=["f0", "f1"])
    result = next(
        item
        for item in report.interactions
        if any(index < 2 for index in item.combo)
        and any(index >= 2 for index in item.combo)
    )

    assert result.family == "decision_path"
    assert result.params == {}
    with pytest.raises(ValueError, match="exactly one generated path-membership"):
        decision_path_candidate_from_result(result)


def test_decision_path_compile_returns_expanded_resident_artifact():
    X, y = _and_dataset()
    artifact = GafimeEngine(_config()).compile(X, y, feature_names=["f0", "f1"])
    try:
        assert artifact.scenario_plan.n_features == 2
        assert artifact.scenario_plan.feature_candidate_count == 2
        report = artifact.analyze()
        assert report.feature_names == artifact.feature_names
        assert any(name.startswith("path[") for name in artifact.feature_names)
        assert report.warnings and "decision_path discovered" in report.warnings[0]
    finally:
        artifact.close()


def test_decision_path_compiled_and_eager_preserve_full_unary_order():
    X, y = _and_dataset()
    config = _config(backend="cpu")
    eager = GafimeEngine(config).analyze(X, y, feature_names=["f0", "f1"])
    artifact = GafimeEngine(config).compile(X, y, feature_names=["f0", "f1"])
    try:
        compiled = artifact.analyze()
        assert [
            (item.combo, item.candidate_id, item.family, item.metrics, item.params)
            for item in compiled.interactions
        ] == [
            (item.combo, item.candidate_id, item.family, item.metrics, item.params)
            for item in eager.interactions
        ]
    finally:
        artifact.close()


def _float32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _float64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


@pytest.mark.parametrize(
    ("precision", "lower", "upper", "threshold_typecode"),
    [
        (
            "fp32",
            _float32_from_bits(0x3F800000),
            _float32_from_bits(0x3F800001),
            "f",
        ),
        (
            "fp32",
            _float32_from_bits(0x3F800001),
            _float32_from_bits(0x3F800002),
            "f",
        ),
        (
            "mixed",
            _float32_from_bits(0x3F800000),
            _float32_from_bits(0x3F800001),
            "f",
        ),
        (
            "mixed",
            _float32_from_bits(0x3F800001),
            _float32_from_bits(0x3F800002),
            "f",
        ),
        (
            "fp64",
            _float64_from_bits(0x3FF0000000000000),
            _float64_from_bits(0x3FF0000000000001),
            "d",
        ),
        (
            "fp64",
            _float64_from_bits(0x3FF0000000000001),
            _float64_from_bits(0x3FF0000000000002),
            "d",
        ),
    ],
)
def test_adjacent_float_decision_path_split_survives_eager_and_compiled(
    precision, lower, upper, threshold_typecode
):
    inputs = [[lower], [lower], [upper], [upper]]
    target = [0.0, 0.0, 1.0, 1.0]
    config = _config(
        precision=precision,
        backend="core",
        decision_path_max_depth=1,
        decision_path_rounds=1,
        decision_path_max_paths=1,
        decision_path_max_bins=0,
        decision_path_min_leaf=1,
        decision_path_top_k_features=1,
    )

    eager = GafimeEngine(config).analyze(inputs, target, feature_names=["value"])
    artifact = GafimeEngine(config).compile(inputs, target, feature_names=["value"])
    try:
        compiled = artifact.analyze()
    finally:
        artifact.close()

    for report in (eager, compiled):
        path = next(
            item
            for item in report.interactions
            if item.family == "decision_path" and item.params
        )
        threshold = float(path.params["thresholds"][0])
        assert lower <= threshold < upper
        assert path.params["thresholds"].typecode == threshold_typecode
        assert path.params["signs"] == [1]
        assert path.params["support"] == 2
        assert float(path.params["gain"]) == 2.0
        assert float(path.metrics["pearson"]) == 1.0
        candidate = decision_path_candidate_from_result(path)
        assert evaluate_decision_path_candidate(inputs, candidate) == [
            0.0,
            0.0,
            1.0,
            1.0,
        ]
        assert any(name.startswith("path[") for name in report.feature_names)

    assert compiled.feature_names == eager.feature_names
    assert [
        (item.combo, item.candidate_id, item.family, item.metrics, item.params)
        for item in compiled.interactions
    ] == [
        (item.combo, item.candidate_id, item.family, item.metrics, item.params)
        for item in eager.interactions
    ]


@pytest.mark.parametrize(
    ("max_paths", "top_k_features", "expected_path_limit"),
    [(8, 0, 0), (1, 1, 1)],
)
def test_decision_path_generation_caps_bound_executed_candidates(
    max_paths, top_k_features, expected_path_limit
):
    X, y = _and_dataset()
    report = GafimeEngine(
        _config(
            decision_path_max_paths=max_paths,
            decision_path_top_k_features=top_k_features,
        )
    ).analyze(X, y, feature_names=["f0", "f1"])

    path_names = [name for name in report.feature_names if name.startswith("path[")]
    assert len(path_names) <= expected_path_limit
    assert len(report.interactions) == len(report.feature_names)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("decision_path_max_depth", 0, "max_depth must be >= 1"),
        ("decision_path_rounds", 0, "rounds must be >= 1"),
        ("decision_path_max_paths", 0, "max_paths must be >= 1"),
        ("decision_path_max_bins", -1, "max_bins must be >= 0"),
        ("decision_path_min_leaf", 0, "min_leaf must be >= 1"),
        ("decision_path_learning_rate", 0.0, "learning_rate must be > 0"),
        ("decision_path_top_k_features", -1, "top_k_features must be >= 0"),
    ],
)
def test_decision_path_config_preserves_v05_validation(setting, value, message):
    X, y = _and_dataset()
    with pytest.raises(ValueError, match=message):
        GafimeEngine(_config(**{setting: value})).analyze(X, y)


def test_decision_path_max_bins_caps_split_boundaries_end_to_end():
    X = [[float(value)] for value in range(10)]
    y = [float(value >= 2) for value in range(10)]
    common = {
        "decision_path_max_depth": 1,
        "decision_path_rounds": 1,
        "decision_path_max_paths": 1,
        "decision_path_min_leaf": 1,
        "decision_path_top_k_features": 1,
    }

    exact = GafimeEngine(_config(decision_path_max_bins=0, **common)).analyze(
        X, y, feature_names=["value"]
    )
    capped = GafimeEngine(_config(decision_path_max_bins=1, **common)).analyze(
        X, y, feature_names=["value"]
    )

    exact_path = next(
        item
        for item in exact.interactions
        if item.family == "decision_path" and item.params
    )
    capped_path = next(
        item
        for item in capped.interactions
        if item.family == "decision_path" and item.params
    )
    assert tuple(exact_path.params["thresholds"]) == (1.5,)
    assert tuple(capped_path.params["thresholds"]) == (4.5,)


def test_decision_path_source_selection_uses_unary_strength_and_original_index():
    X = [[0.0, 1.0, float(row >= 6)] for row in range(12)]
    y = [float(row >= 6) for row in range(12)]
    names = ["prefix_zero", "prefix_one", "signal"]
    config = _config(
        decision_path_max_depth=1,
        decision_path_rounds=1,
        decision_path_max_paths=1,
        decision_path_min_leaf=2,
        decision_path_top_k_features=1,
    )

    eager = GafimeEngine(config).analyze(X, y, feature_names=names)
    artifact = GafimeEngine(config).compile(X, y, feature_names=names)
    try:
        compiled = artifact.analyze()
        eager_path = next(
            name for name in eager.feature_names if name.startswith("path[")
        )
        compiled_path = next(
            name for name in compiled.feature_names if name.startswith("path[")
        )
        assert "signal" in eager_path
        assert eager_path == compiled_path
        eager_result = next(
            item
            for item in eager.interactions
            if item.family == "decision_path" and len(item.combo) == 1
        )
        compiled_result = next(
            item
            for item in compiled.interactions
            if item.family == "decision_path" and len(item.combo) == 1
        )
        assert tuple(eager_result.params["features"]) == (2,)
        assert compiled_result.params == eager_result.params
    finally:
        artifact.close()


def test_compiled_decision_path_update_target_rediscovers_atomically():
    X, y = _and_dataset()
    artifact = GafimeEngine(_config()).compile(X, y, feature_names=["f0", "f1"])
    try:
        before = artifact.analyze()
        assert artifact.update_target(list(reversed(y))) is artifact
        after = artifact.analyze()
        assert after.feature_names == artifact.feature_names
        assert any(name.startswith("path[") for name in after.feature_names)
        assert after.feature_names != before.feature_names or [
            item.metrics for item in after.interactions
        ] != [item.metrics for item in before.interactions]
        assert artifact.native_handle.closed is False
    finally:
        artifact.close()


def test_decision_path_permutation_rediscovers_each_null_target():
    X, y = _and_dataset()
    config = _config(permutation_tests=5)

    eager = GafimeEngine(config).analyze(X, y, feature_names=["f0", "f1"])
    artifact = GafimeEngine(config).compile(X, y, feature_names=["f0", "f1"])
    try:
        compiled = artifact.analyze()
    finally:
        artifact.close()
    assert eager.permutations
    assert compiled.permutations
    assert [item.candidate_id for item in eager.permutations] == [
        item.candidate_id for item in compiled.permutations
    ]


def test_decision_path_opt_in_supports_the_default_permutation_request():
    X, y = _and_dataset()
    config = EngineConfig(
        enable_decision_path_functions=True,
        metric_names=("pearson",),
    )

    assert config.permutation_tests == 25
    report = GafimeEngine(config).analyze(X, y, feature_names=["f0", "f1"])
    assert report.permutations


def test_decision_path_carries_stability_when_requested():
    X, y = _and_dataset()
    report = GafimeEngine(_config(permutation_tests=0, num_repeats=5)).analyze(
        X, y, feature_names=["f0", "f1"]
    )
    assert len(report.permutations) == 0
    assert len(report.stability) > 0
    # A perfect AND-membership feature is a real, stable signal.
    assert report.decision.signal_detected is True
