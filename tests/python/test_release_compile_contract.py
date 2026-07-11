from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_RELEASE_MEASURE = Path(__file__).resolve().parents[1] / "release_measure"
if str(_RELEASE_MEASURE) not in sys.path:
    sys.path.insert(0, str(_RELEASE_MEASURE))

from compile_02_compiled_vs_eager import assert_report_parity  # noqa: E402


def _report(*, metric=1.0, pvalue=0.1, stable_mean=1.0, decision=True):
    identity = {
        "candidate_id": "interaction:0",
        "family": "interaction",
        "combo": (0,),
        "expression": "x",
        "params": {},
    }
    return SimpleNamespace(
        interactions=[
            SimpleNamespace(
                **identity,
                feature_names=("x",),
                metrics={"pearson": metric},
            )
        ],
        permutations=[SimpleNamespace(**identity, p_values={"pearson": pvalue})],
        stability=[
            SimpleNamespace(
                **identity,
                metrics_mean={"pearson": stable_mean},
                metrics_std={"pearson": 0.01},
            )
        ],
        decision=SimpleNamespace(signal_detected=decision, message="decision"),
    )


def test_compiled_parity_rejects_significance_value_drift():
    with pytest.raises(AssertionError, match="permutations"):
        assert_report_parity(_report(), _report(pvalue=0.9))


def test_compiled_parity_rejects_decision_drift():
    with pytest.raises(AssertionError, match="decision differs"):
        assert_report_parity(_report(), _report(decision=False))


def test_compiled_parity_rejects_interaction_value_drift():
    with pytest.raises(AssertionError, match="pearson"):
        assert_report_parity(_report(), _report(metric=999.0))
