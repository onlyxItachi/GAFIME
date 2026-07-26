from __future__ import annotations

import os
import sys
import threading
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


def test_continuous_report_owned_table_is_safe_to_read_from_another_thread():
    report = GafimeEngine(
        EngineConfig(
            backend="cpu",
            metric_names=("pearson",),
            permutation_tests=0,
            num_repeats=1,
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=8,
                keep_in_vram=False,
            ),
        )
    ).analyze(
        [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]],
        [1.0, 2.0, 3.0, 4.0],
        feature_names=["up", "down"],
    )
    native_report = report.interactions.native_handle
    outcomes: list[object] = []

    def read_report() -> None:
        try:
            outcomes.append(
                (
                    len(native_report),
                    native_report.combo(0),
                    native_report.metric_values(0),
                    native_report.ranked_indices(limit=1),
                )
            )
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=read_report)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcomes == [(2, [0], [1.0], [0])]


def test_fp32_interaction_overflow_is_counted_without_changing_candidate_identity():
    config = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(
            max_comb_size=5,
            max_combinations_per_k=64,
            top_features_for_higher_k=5,
            keep_in_vram=False,
        ),
    )
    magnitudes = (-1.0e8, -100.0, 100.0, 1.0e8)
    report = GafimeEngine(config).analyze(
        [[value] * 5 for value in magnitudes],
        [0.0, 1.0, 2.0, 3.0],
        feature_names=[f"x{index}" for index in range(5)],
    )

    arity_five = next(item for item in report.interactions if len(item.combo) == 5)
    assert set(arity_five.combo) == {0, 1, 2, 3, 4}
    assert arity_five.interaction_overflow_rows == 2
    assert arity_five.interaction_overflow_ratio == 0.5
    assert arity_five.source_nonfinite is False
    assert arity_five.precision_diagnostics_available is True
    assert report.backend is not None
    assert report.backend.interaction_diagnostics_available is True
    assert len(report.warnings) == 1
    assert "worst candidate lost 2 of 4 sample rows" in report.warnings[0]


def test_overflow_diagnostics_match_resident_eager_and_compiled_execution(monkeypatch):
    from gafime import v1_adapter

    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    v1_adapter._clear_analyze_cache_for_tests()
    config = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(
            max_comb_size=5,
            max_combinations_per_k=64,
            top_features_for_higher_k=5,
            keep_in_vram=True,
        ),
    )
    magnitudes = (-1.0e8, -100.0, 100.0, 1.0e8)
    features = [[value] * 5 for value in magnitudes]
    target = [0.0, 1.0, 2.0, 3.0]
    names = [f"x{index}" for index in range(5)]

    def signature(report):
        item = next(result for result in report.interactions if len(result.combo) == 5)
        return (
            item.candidate_id,
            item.interaction_overflow_rows,
            item.interaction_overflow_ratio,
            item.source_nonfinite,
            item.precision_diagnostics_available,
        )

    try:
        engine = GafimeEngine(config)
        first = engine.analyze(features, target, feature_names=names)
        cached = engine.analyze(features, target, feature_names=names)
        artifact = engine.compile(features, target, feature_names=names)
        try:
            compiled = artifact.analyze()
        finally:
            artifact.close()
        assert signature(first) == signature(cached) == signature(compiled)
        assert signature(first)[1:] == (2, 0.5, False, True)
    finally:
        v1_adapter._clear_analyze_cache_for_tests()


def test_source_nonfinite_is_reported_separately_without_overflow_warning():
    config = EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=8,
            top_features_for_higher_k=2,
            keep_in_vram=False,
        ),
    )
    report = GafimeEngine(config).analyze(
        [[float("nan"), -1.0], [1.0, 1.0], [2.0, 2.0]],
        [0.0, 1.0, 2.0],
        feature_names=["nonfinite", "finite"],
    )

    affected = next(item for item in report.interactions if 0 in item.combo)
    assert affected.source_nonfinite is True
    assert affected.interaction_overflow_rows == 0
    assert report.warnings == []


def test_unranked_public_report_rejects_pathological_plan_via_storage_admission():
    cols = 20_000
    config = EngineConfig(
        backend="cpu",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        significance_top_n=1,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=100_000_000,
            top_features_for_higher_k=cols,
            keep_in_vram=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="unranked continuous candidate storage exceeds the host-memory budget",
    ):
        GafimeEngine(config).compile(
            [[0.0] * cols, [1.0] * cols],
            [0.0, 1.0],
        )


def test_v047_power_user_unranked_plan_above_one_million_rows_is_preserved():
    cols = 1_450
    pair_rows = 1_050_525
    config = EngineConfig(
        backend="cpu",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=pair_rows,
            top_features_for_higher_k=cols,
            keep_in_vram=False,
        ),
    )
    X = [[float(row)] * cols for row in range(4)]

    artifact = GafimeEngine(config).compile(X, [0.0, 1.0, 2.0, 3.0])
    assert pair_rows == cols * (cols - 1) // 2
    assert cols + pair_rows == 1_051_975
    assert artifact.native_handle.cols == cols
    assert artifact.native_handle.max_arity == 2
    artifact.close()
