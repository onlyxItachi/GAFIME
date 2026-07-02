"""P-A: native permutation-test + stability significance, end to end.

Requires the built native boundary (`gafime.gafime_py`); skips cleanly otherwise.
Self-paths to the wheel source (`python/`) so it exercises the shipped package,
not the legacy `./gafime/` at the repo root. Determinism comes from a fixed data
seed plus the native seeded permutation stream (`random_seed`), so the stochastic
assertions are reproducible.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Resolve the v1 wheel source (python/), not the legacy ./gafime/ at the repo root.
_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

# The significance path is native; skip when the wheel is not built.
pytest.importorskip("gafime.gafime_py")

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402


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
