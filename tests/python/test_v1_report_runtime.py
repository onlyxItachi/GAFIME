from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
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
