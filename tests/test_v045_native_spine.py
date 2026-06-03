import unittest

import gafime
from gafime import ComputeBudget, EngineConfig, GafimeEngine, gafime_core, subfunctions
from gafime.backends import resolve_backend
from gafime.utils.arrays import coerce_inputs


def _dataset(n=160):
    X = [
        [
            (i - 50) / 20.0,
            ((i * 7) % 23 - 11) / 6.0,
            ((i * 5) % 17 - 8) / 5.0,
            ((i * i) % 29 - 14) / 7.0,
        ]
        for i in range(n)
    ]
    y = [row[0] * row[1] + 0.25 * row[2] for row in X]
    return X, y


class NativeSpineTests(unittest.TestCase):
    def test_version_metadata_is_v045(self):
        self.assertEqual(gafime.__version__, "0.4.5")
        self.assertEqual(getattr(subfunctions, "__version__", None), "0.4.5")


    def test_native_buffers_own_fp32_memory_and_core_has_single_scorer_surface(self):
        X, y, _ = coerce_inputs([[1, 2], [3, 4], [5, 8]], [1, 2, 3])

        self.assertEqual(gafime_core.precision_name(), "float32")
        self.assertEqual(memoryview(X.buffer).format, "f")
        self.assertEqual(memoryview(y.buffer).format, "f")
        self.assertEqual(X.nbytes, X.n_samples * X.n_features * 4)
        self.assertEqual(y.nbytes, len(y) * 4)

        feature_major = X.feature_major_buffer()
        self.assertEqual(memoryview(feature_major).format, "f")
        self.assertEqual(len(memoryview(feature_major)), X.n_samples * X.n_features)

        self.assertTrue(hasattr(gafime_core, "score_combos_buffer"))
        self.assertFalse(hasattr(gafime_core, "score_combos"))
        self.assertFalse(hasattr(gafime_core, "score_combos_flat"))

        self.assertIn(
            gafime_core.cpu_dispatch_target(),
            {"AVX512", "AVX2", "SSE4.2", "NEON", "Default"},
        )


    def test_unknown_backend_name_is_not_accepted(self):
        X, y, _ = coerce_inputs([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            resolve_backend(EngineConfig(backend="not-a-native-backend"), X, y)


    def test_core_engine_scores_native_continuous_interactions(self):
        X, y = _dataset()
        report = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                budget=ComputeBudget(max_comb_size=3, max_combinations_per_k=32),
                permutation_tests=2,
                num_repeats=2,
            )
        ).analyze(X, y, ["a", "b", "c", "d"])
        self.assertEqual(report.backend.name, "core")
        self.assertTrue(any(len(result.combo) == 3 for result in report.interactions))
        self.assertTrue(
            all("pearson" in result.metrics and "r2" in result.metrics for result in report.interactions)
        )


    def test_discrete_and_time_series_candidate_families_are_engine_integrated(self):
        X, y = _dataset()
        report = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                enable_discrete_functions=True,
                enable_time_series_functions=True,
                time_series_lags=(1, 2),
                time_series_windows=(3,),
                budget=ComputeBudget(
                    max_comb_size=2,
                    max_combinations_per_k=16,
                    max_discrete_candidates=48,
                    max_time_series_candidates=24,
                ),
                permutation_tests=2,
                num_repeats=2,
            )
        ).analyze(X, y, ["a", "b", "c", "d"])
        families = {result.family for result in report.interactions}
        self.assertIn("discrete_function", families)
        self.assertIn("time_series_function", families)


    def test_cuda_report_metric_names_are_not_rejected_by_policy_guard(self):
        from gafime.backends.native_cuda_backend import CUDA_REPORT_METRICS, CUDA_STATS_METRICS

        self.assertEqual(CUDA_STATS_METRICS, ("pearson", "r2"))
        self.assertEqual(CUDA_REPORT_METRICS, ("pearson", "spearman", "mutual_info", "r2"))
        X, y, _ = coerce_inputs([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
        try:
            resolve_backend(EngineConfig(backend="cuda", metric_names=("mutual_info",)), X, y)
        except Exception as exc:
            self.assertNotIn("unsupported metrics", str(exc))
            self.assertNotIn("supports report metrics", str(exc))


    def test_rust_batch_scheduler_returns_homogeneous_arity_batches(self):
        scheduler = subfunctions.BatchScheduler(max_blocks=4)
        batches = scheduler.create_batches(
            candidate_kinds=[0, 0, 0],
            feature_sets=[[0, 1], [0, 1, 2], [3, 4]],
            op_sets=[[0, 0], [0, 0, 0], [0, 0]],
            interaction_sets=[[0], [0, 0], [0]],
            ts_params=[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
        )
        self.assertTrue(batches)
        self.assertEqual({batch[5] for batch in batches}, {2, 3})
        for kinds, indices, ops, interact, ts_params, arity, size in batches:
            self.assertEqual(len(kinds), size)
            self.assertEqual(len(indices), size * arity)
            self.assertEqual(len(ops), size * arity)
            self.assertEqual(len(interact), size * max(arity - 1, 1))
            self.assertEqual(len(ts_params), size * 4)


if __name__ == "__main__":
    unittest.main()
