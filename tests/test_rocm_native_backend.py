import math
import unittest

from gafime import ComputeBudget, EngineConfig, GafimeEngine
from gafime.backends import resolve_backend
from gafime.backends.policy import PlatformProfile, backend_priority
from gafime.discrete import DiscreteFunctionCandidate
from gafime.metrics import MetricSuite
from gafime.native_data import coerce_inputs
from gafime.time_series import TimeSeriesCandidate


def _dataset(n=192, p=6):
    X = []
    for i in range(n):
        row = [
            ((i * 7 + j * 11) % 37 - 18) / 9.0 + 0.01 * j
            for j in range(p)
        ]
        X.append(row)
    y = [
        row[0] * row[1] + 0.5 * row[2] - 0.25 * row[3]
        for row in X
    ]
    return X, y


def _rocm_backend_or_skip(testcase):
    X, y = _dataset(n=16, p=4)
    Xn, yn, _ = coerce_inputs(X, y)
    try:
        backend, warnings = resolve_backend(EngineConfig(backend="rocm"), Xn, yn)
    except Exception as exc:
        testcase.skipTest(f"ROCm/HIP backend unavailable: {exc}")
    return backend, warnings


class RocmNativeBackendTests(unittest.TestCase):
    def test_explicit_rocm_policy_does_not_change_auto_priority(self):
        self.assertEqual(
            backend_priority("rocm", PlatformProfile("linux", "x86_64")),
            ["rocm"],
        )
        self.assertEqual(
            backend_priority("rocm", PlatformProfile("windows", "amd64")),
            ["rocm"],
        )
        self.assertEqual(
            backend_priority("hip", PlatformProfile("linux", "x86_64")),
            ["rocm"],
        )
        self.assertEqual(
            backend_priority("auto", PlatformProfile("linux", "x86_64")),
            ["cuda", "core"],
        )
        with self.assertRaisesRegex(RuntimeError, "not supported on macOS"):
            backend_priority("hip", PlatformProfile("darwin", "arm64"))

    def test_rocm_continuous_scores_match_core_for_arity_1_to_5(self):
        rocm, _warnings = _rocm_backend_or_skip(self)
        X, y = _dataset()
        Xn, yn, _ = coerce_inputs(X, y)
        core, _ = resolve_backend(EngineConfig(backend="core"), Xn, yn)
        combos = [
            (0,),
            (0, 1),
            (0, 1, 2),
            (0, 1, 2, 3),
            (0, 1, 2, 3, 4),
        ]
        suite = MetricSuite(("pearson", "r2"))

        rocm_scores = rocm.score_combos(Xn, yn, combos, suite)
        core_scores = core.score_combos(Xn, yn, combos, suite)

        for combo in combos:
            for metric in ("pearson", "r2"):
                self.assertTrue(math.isfinite(rocm_scores[combo][metric]))
                self.assertAlmostEqual(
                    rocm_scores[combo][metric],
                    core_scores[combo][metric],
                    places=4,
                    msg=f"{combo} {metric}",
                )

    def test_rocm_discrete_soft_and_selector_smoke(self):
        rocm, _warnings = _rocm_backend_or_skip(self)
        X, y = _dataset()
        Xn, yn, _ = coerce_inputs(X, y)
        candidates = [
            DiscreteFunctionCandidate(
                kind="discrete_function_soft_threshold",
                feature_indices=(0,),
                thresholds=(0.0,),
                mode="soft",
                direction="ge",
            ),
            DiscreteFunctionCandidate(
                kind="discrete_function_soft_rectangle",
                feature_indices=(0, 1),
                intervals=((-1.0, 1.0), (-0.5, 1.5)),
                mode="soft",
            ),
        ]

        report_scores = rocm.score_discrete_candidates(
            Xn,
            yn,
            candidates,
            MetricSuite(("pearson", "r2")),
        )
        selector_scores = rocm.score_discrete_selection_candidates(
            Xn,
            yn,
            candidates,
            mi_bins=32,
        )

        self.assertEqual(set(report_scores), set(candidates))
        self.assertEqual(set(selector_scores), set(candidates))
        for candidate in candidates:
            self.assertIn("pearson", report_scores[candidate])
            self.assertIn("mutual_info", selector_scores[candidate])
            self.assertTrue(math.isfinite(selector_scores[candidate]["mutual_info"]))

    def test_rocm_time_series_bucket_scores_match_core_completion(self):
        rocm, _warnings = _rocm_backend_or_skip(self)
        X, y = _dataset()
        Xn, yn, _ = coerce_inputs(X, y)
        core, _ = resolve_backend(EngineConfig(backend="core"), Xn, yn)
        candidates = [
            TimeSeriesCandidate("time_series_lag", feature_index=0, lag=2),
            TimeSeriesCandidate("time_series_delta", feature_index=1, lag=3),
            TimeSeriesCandidate("time_series_rolling_mean", feature_index=2, window=5),
        ]
        suite = MetricSuite(("pearson", "r2"))

        rocm_scores = rocm.score_time_series_candidates(Xn, yn, candidates, suite)
        core_scores = core.score_time_series_candidates(Xn, yn, candidates, suite)

        for candidate in candidates:
            for metric in ("pearson", "r2"):
                self.assertTrue(math.isfinite(rocm_scores[candidate][metric]))
                self.assertAlmostEqual(
                    rocm_scores[candidate][metric],
                    core_scores[candidate][metric],
                    places=4,
                    msg=f"{candidate} {metric}",
                )

    def test_rocm_rejects_hard_discrete_mode(self):
        rocm, _warnings = _rocm_backend_or_skip(self)
        X, y = _dataset()
        Xn, yn, _ = coerce_inputs(X, y)
        candidate = DiscreteFunctionCandidate(
            kind="discrete_function_soft_threshold",
            feature_indices=(0,),
            thresholds=(0.0,),
            mode="hard",
            direction="ge",
        )
        with self.assertRaisesRegex(ValueError, "GPU feature engineering with discrete hard mode is not supported"):
            rocm.score_discrete_candidates(Xn, yn, [candidate], MetricSuite(("pearson",)))

    def test_rocm_engine_smoke_finds_pair_interaction(self):
        _rocm_backend_or_skip(self)
        X, y = _dataset(n=256, p=6)
        report = GafimeEngine(
            EngineConfig(
                backend="rocm",
                metric_names=("pearson", "r2"),
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=64),
                permutation_tests=2,
                num_repeats=2,
            )
        ).analyze(X, y, [f"f{i}" for i in range(6)])
        self.assertEqual(report.backend.name, "rocm-native")
        self.assertTrue(any(result.combo == (0, 1) for result in report.interactions))


if __name__ == "__main__":
    unittest.main()
