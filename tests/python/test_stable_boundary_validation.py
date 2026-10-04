"""Bounded stable input checks against the real installed Rust boundary."""

from dataclasses import replace
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1":
    sys.path.insert(0, str(ROOT / "python"))

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402


@pytest.fixture
def config():
    # These tests intentionally require the actual native boundary: a fake
    # adapter cannot establish where validation happens or the exception type.
    pytest.importorskip("gafime.gafime_py")
    return EngineConfig(
        backend="core",
        metric_names=("pearson",),
        num_repeats=1,
        permutation_tests=0,
        budget=ComputeBudget(max_comb_size=1, keep_in_vram=False),
    )


def _analyze(config):
    return GafimeEngine(config).analyze(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]],
        [1.0, 2.0, 3.0, 4.0],
        feature_names=["a", "b"],
    )


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize(
    "field", ["stability_std_threshold", "permutation_p_threshold"]
)
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_thresholds_are_value_errors(config, precision, field, value):
    with pytest.raises(ValueError, match=field):
        _analyze(replace(config, precision=precision, **{field: value}))


def test_threshold_must_fit_result_lane(config):
    with pytest.raises(ValueError, match="stability_std_threshold"):
        _analyze(replace(config, precision="fp32", stability_std_threshold=1e100))
    _analyze(replace(config, precision="mixed", stability_std_threshold=1e100))


@pytest.mark.parametrize(
    "field",
    ["num_repeats", "permutation_tests", "significance_top_n", "mi_bins", "device_id"],
)
def test_negative_counts_are_value_errors(config, field):
    with pytest.raises(ValueError, match=field):
        _analyze(replace(config, **{field: -1}))


@pytest.mark.parametrize(
    "field",
    [
        "max_comb_size",
        "max_combinations_per_k",
        "top_features_for_higher_k",
        "max_generated_features",
        "vram_budget_mb",
    ],
)
def test_negative_budget_counts_are_value_errors(config, field):
    with pytest.raises(ValueError, match=field):
        _analyze(replace(config, budget=replace(config.budget, **{field: -1})))


@pytest.mark.parametrize("compiled", [False, True])
def test_conflicting_families_fail_with_actionable_error(config, compiled):
    engine = GafimeEngine(
        replace(
            config,
            enable_time_series_functions=True,
            enable_decision_path_functions=True,
        )
    )
    operation = engine.compile if compiled else engine.analyze
    with pytest.raises(ValueError, match="mutually exclusive"):
        operation([[1.0], [2.0], [3.0], [4.0]], [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("backend", ["Core", "CORE", "CPU", "RuSt", "V1-RUST-CPU"])
def test_backend_case_is_normalized_for_execution(config, backend):
    report = _analyze(replace(config, backend=backend))
    assert report.backend is not None
    assert report.backend.device == "cpu"
