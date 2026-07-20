from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime import EngineConfig  # noqa: E402
from gafime.reporting import DiagnosticReport, InteractionResult  # noqa: E402
from gafime.reporting.report import NativeContinuousInteractions  # noqa: E402


def test_manually_constructed_report_preserves_v05_sequence_api() -> None:
    report = DiagnosticReport(
        config=EngineConfig(),
        feature_names=["a", "b"],
        interactions=[
            InteractionResult(
                combo=(0,), feature_names=("a",), metrics={"pearson": 0.25}
            ),
            InteractionResult(
                combo=(1,), feature_names=("b",), metrics={"pearson": -0.75}
            ),
        ],
    )

    assert report.interactions.is_native_backed is False
    assert report.interactions.native_handle is None
    assert report.interactions.native_indices is None
    assert report.interactions.top_k(1)[0].combo == (1,)


class _FakeNativeReport:
    _metrics = ((0.99,), (0.4,), (0.8,))

    def __len__(self) -> int:
        return len(self._metrics)

    def metric_values(self, index: int) -> tuple[float, ...]:
        return self._metrics[index]

    def candidate_id(self, index: int) -> int:
        return index

    def interaction_components(self, index: int):
        return [index], list(self._metrics[index]), index

    def ranked_indices(self, **_kwargs):
        raise AssertionError("a ranked view must not rerank the full native table")


def test_ranking_native_view_stays_within_existing_indices() -> None:
    view = NativeContinuousInteractions(
        _FakeNativeReport(),
        ("a", "b", "c"),
        ("pearson",),
        indices=(1, 2),
    )

    reranked = view.top_k(1)

    assert reranked.native_indices == (2,)
    assert reranked[0].combo == (2,)


def test_native_family_uses_generated_column_boundary_not_feature_spelling() -> None:
    view = NativeContinuousInteractions(
        _FakeNativeReport(),
        ("sales_lag1", "path[raw]", "sales_lag1"),
        ("pearson",),
        indices=(0, 1, 2),
        generated_feature_start=2,
        generated_family="time_series",
    )

    assert [item.family for item in view] == [
        "interaction",
        "interaction",
        "time_series",
    ]
    assert view.top_k(3)[0].family == "interaction"


class _EqualScoreNativeReport(_FakeNativeReport):
    _metrics = ((0.5,), (0.5,))
    _candidate_ids = (10, 2)

    def candidate_id(self, index: int) -> int:
        return self._candidate_ids[index]

    def interaction_components(self, index: int):
        return [index], list(self._metrics[index]), self._candidate_ids[index]


def test_reranking_native_view_breaks_score_ties_by_numeric_candidate_id() -> None:
    view = NativeContinuousInteractions(
        _EqualScoreNativeReport(),
        ("ten", "two"),
        ("pearson",),
        indices=(0, 1),
    )

    reranked = view.ranked(metric_name="pearson")

    assert reranked.native_indices == (1, 0)
    assert [item.candidate_id for item in reranked] == ["interaction:2", "interaction:10"]


def test_reranking_native_view_rejects_unknown_metric_name() -> None:
    view = NativeContinuousInteractions(
        _EqualScoreNativeReport(),
        ("ten", "two"),
        ("pearson",),
        indices=(0, 1),
    )

    with pytest.raises(ValueError):
        view.ranked(metric_name="unknown")
