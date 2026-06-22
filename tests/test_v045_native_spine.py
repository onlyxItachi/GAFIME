import unittest
import json
import random
import warnings
from unittest.mock import patch

import gafime
from gafime import engine as engine_module
from gafime import ComputeBudget, EngineConfig, GafimeEngine, gafime_core, subfunctions
from gafime.backends import resolve_backend
from gafime.backends.policy import PlatformProfile, backend_priority
from gafime.decision_path import DecisionPathCandidate, evaluate_decision_path_candidate
from gafime.metrics import MetricSuite
from gafime.time_series import TIME_SERIES_KIND_CODES, TimeSeriesCandidate, score_time_series_candidates
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


def _decision_path_dataset(n=160):
    X = []
    y = []
    for i in range(n):
        x0 = ((i * 37) % 101) / 100.0
        x1 = ((i * 19 + 7) % 97) / 96.0
        x2 = ((i * 11 + 3) % 89) / 88.0
        planted = 1.0 if x0 > 0.55 and x1 > 0.45 else 0.0
        X.append([x0, x1, x2])
        y.append(planted + 0.05 * x2)
    return X, y


class NativeSpineTests(unittest.TestCase):
    def test_version_metadata_is_consistent(self):
        self.assertRegex(gafime.__version__, r"^\d+\.\d+\.\d+")
        self.assertEqual(getattr(subfunctions, "__version__", None), gafime.__version__)


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
        self.assertIn("Default", gafime_core.available_cpu_dispatch_targets())


    def test_unknown_backend_name_is_not_accepted(self):
        X, y, _ = coerce_inputs([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            resolve_backend(EngineConfig(backend="not-a-native-backend"), X, y)

    def test_platform_backend_priority_is_explicit(self):
        self.assertEqual(
            backend_priority("auto", PlatformProfile("darwin", "arm64")),
            ["metal", "core"],
        )
        self.assertEqual(
            backend_priority("auto", PlatformProfile("linux", "x86_64")),
            ["core"],
        )
        self.assertEqual(
            backend_priority("auto", PlatformProfile("windows", "amd64")),
            ["core"],
        )
        with patch("gafime.backends.policy._payload_available", side_effect=lambda name: name == "gafime_cuda"):
            self.assertEqual(
                backend_priority("auto", PlatformProfile("linux", "x86_64")),
                ["cuda", "core"],
            )
            self.assertEqual(
                backend_priority("auto", PlatformProfile("windows", "amd64")),
                ["cuda", "core"],
            )
        with patch("gafime.backends.policy._payload_available", side_effect=lambda name: name == "gafime_rocm"):
            self.assertEqual(
                backend_priority("auto", PlatformProfile("windows", "amd64")),
                ["rocm", "core"],
            )
            self.assertEqual(
                backend_priority("auto", PlatformProfile("linux", "x86_64")),
                ["rocm", "core"],
            )
        self.assertEqual(
            backend_priority("auto", PlatformProfile("linux", "aarch64")),
            ["core"],
        )
        with self.assertRaisesRegex(RuntimeError, "Metal backend is only supported on macOS"):
            backend_priority("metal", PlatformProfile("linux", "x86_64"))
        with self.assertRaisesRegex(RuntimeError, "CUDA backend is not supported on macOS"):
            backend_priority("cuda", PlatformProfile("darwin", "arm64"))
        with self.assertRaisesRegex(RuntimeError, "ARM wheels"):
            backend_priority("cuda", PlatformProfile("linux", "aarch64"))


    def test_metal_backend_resolves_or_fails_cleanly(self):
        X, y, _ = coerce_inputs([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
        try:
            backend, _warnings = resolve_backend(EngineConfig(backend="metal"), X, y)
        except Exception as exc:
            self.assertNotIn("known issues in GAFIME v0.4.5", str(exc))
            self.assertIn("metal", str(exc).lower())
        else:
            info = backend.info()
            self.assertEqual(info.name, "metal-native")
            self.assertTrue(info.is_gpu)


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
        self.assertTrue(getattr(report.interactions, "is_native_backed", False))
        with self.assertWarns(DeprecationWarning):
            json.dumps(report.to_dict())

    def test_native_report_top_k_is_native_backed_index_view(self):
        X, y = _dataset()
        report = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=24),
                permutation_tests=0,
                num_repeats=1,
            )
        ).analyze(X, y, ["a", "b", "c", "d"])

        top = report.interactions.top_k(3, metric_name="r2")
        self.assertTrue(top.is_native_backed)
        self.assertIs(top.native_handle, report.interactions.native_handle)
        self.assertEqual(len(top), 3)
        self.assertIsNotNone(top.native_indices)
        top_values = [item.metrics["r2"] for item in top]
        all_values = sorted(
            (item.metrics["r2"] for item in report.interactions),
            reverse=True,
        )
        self.assertEqual(top_values, all_values[:3])

    def test_continuous_metric_cache_matches_core_for_actual_and_permuted_y(self):
        X_raw, y_raw = _dataset()
        X, y, _ = coerce_inputs(X_raw, y_raw)
        combos = [(0,), (1,), (0, 1), (2,), (2, 3)]
        metric_names = ("pearson", "spearman", "mutual_info", "r2")

        cache = gafime_core.build_continuous_metric_cache(
            X.buffer,
            combos,
            metric_names,
            32,
            100_000_000,
        )
        self.assertIsNotNone(cache)
        self.assertGreater(cache.bytes, 0)
        self.assertEqual([tuple(combo) for combo in cache.combos()], combos)

        y_perm = y.shuffled(random.Random(11))
        for target in (y, y_perm):
            cached = gafime_core.score_continuous_metric_cache(cache, target.buffer)
            expected = gafime_core.score_combos_buffer(
                X.buffer,
                target.buffer,
                combos,
                metric_names,
                32,
            )
            self.assertEqual(len(cached), len(expected))
            max_diff = max(
                abs(float(left) - float(right))
                for cached_row, expected_row in zip(cached, expected)
                for left, right in zip(cached_row, expected_row)
            )
            self.assertLessEqual(max_diff, 1e-6)

        tiny = gafime_core.build_continuous_metric_cache(
            X.buffer,
            combos,
            metric_names,
            32,
            1,
        )
        self.assertIsNone(tiny)

    def test_time_series_metric_cache_matches_core_for_actual_and_permuted_y(self):
        X_raw, y_raw = _dataset()
        X, y, _ = coerce_inputs(X_raw, y_raw)
        candidates = [
            TimeSeriesCandidate("time_series_lag", feature_index=0, lag=1),
            TimeSeriesCandidate("time_series_delta", feature_index=0, lag=2),
            TimeSeriesCandidate("time_series_rolling_mean", feature_index=1, window=3),
            TimeSeriesCandidate("time_series_rolling_std", feature_index=1, window=4),
        ]
        metric_names = ("pearson", "spearman", "mutual_info", "r2")

        cache = gafime_core.build_time_series_metric_cache(
            X.buffer,
            [TIME_SERIES_KIND_CODES[candidate.kind] for candidate in candidates],
            [candidate.feature_index for candidate in candidates],
            [candidate.lag for candidate in candidates],
            [candidate.window for candidate in candidates],
            metric_names,
            32,
            100_000_000,
        )
        self.assertIsNotNone(cache)
        self.assertGreater(cache.bytes, 0)
        self.assertEqual(
            [tuple(combo) for combo in cache.combos()],
            [candidate.combo for candidate in candidates],
        )

        metric_suite = MetricSuite(metric_names, mi_bins=32)
        y_perm = y.shuffled(random.Random(11))
        for target in (y, y_perm):
            cached = gafime_core.score_continuous_metric_cache(cache, target.buffer)
            expected_by_candidate = score_time_series_candidates(
                X,
                target,
                candidates,
                metric_suite,
            )
            expected = [
                [expected_by_candidate[candidate][name] for name in metric_names]
                for candidate in candidates
            ]
            self.assertEqual(len(cached), len(expected))
            max_diff = max(
                abs(float(left) - float(right))
                for cached_row, expected_row in zip(cached, expected)
                for left, right in zip(cached_row, expected_row)
            )
            self.assertLessEqual(max_diff, 1e-5)

        tiny = gafime_core.build_time_series_metric_cache(
            X.buffer,
            [TIME_SERIES_KIND_CODES[candidate.kind] for candidate in candidates],
            [candidate.feature_index for candidate in candidates],
            [candidate.lag for candidate in candidates],
            [candidate.window for candidate in candidates],
            metric_names,
            32,
            1,
        )
        self.assertIsNone(tiny)

    def test_native_decision_path_finder_recovers_recreatable_records(self):
        X_raw, y_raw = _decision_path_dataset()
        X, y, _ = coerce_inputs(X_raw, y_raw)
        records = gafime_core.find_decision_path_candidates(
            X.buffer,
            y.buffer,
            None,
            2,
            8,
            0,
            8,
            1,
            1.0,
        )
        self.assertGreater(len(records), 0)
        first = records[0]
        self.assertEqual(first.candidate_id, 0)
        self.assertEqual(len(first.features), len(first.thresholds))
        self.assertEqual(len(first.features), len(first.signs))
        self.assertTrue(all(sign in (-1, 1) for sign in first.signs))

        candidate = DecisionPathCandidate(
            features=tuple(first.features),
            thresholds=tuple(first.thresholds),
            signs=tuple(first.signs),
            gain=float(first.gain),
            support=float(first.support),
            round_id=int(first.round_id),
            native_candidate_id=int(first.candidate_id),
            candidate_id=f"decision_path:{first.candidate_id}",
        )
        values = evaluate_decision_path_candidate(X, candidate)
        self.assertEqual(len(values), X.n_samples)
        self.assertGreater(sum(values), 0.0)

    def test_decision_path_family_is_engine_integrated(self):
        X, y = _decision_path_dataset()
        report = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                enable_decision_path_functions=True,
                decision_path_max_depth=2,
                decision_path_rounds=1,
                decision_path_max_paths=8,
                decision_path_max_bins=0,
                decision_path_min_leaf=8,
                decision_path_top_k_features=3,
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=16),
                permutation_tests=2,
                num_repeats=2,
            )
        ).analyze(X, y, ["x0", "x1", "x2"])

        decision_paths = [
            result for result in report.interactions
            if result.family == "decision_path"
        ]
        self.assertTrue(decision_paths)
        top = decision_paths[0]
        self.assertEqual(top.candidate_id, top.params["candidate_id"])
        self.assertEqual(top.params["kind"], "decision_path")
        self.assertIn("decision_path(", top.expression)
        self.assertTrue(set(top.params["signs"]).issubset({-1, 1}))
        self.assertTrue(getattr(report.interactions, "is_native_backed", False))

        stability_families = {result.family for result in report.stability}
        permutation_families = {result.family for result in report.permutations}
        self.assertIn("decision_path", stability_families)
        self.assertIn("decision_path", permutation_families)

    def test_gpu_backend_alias_is_deprecated(self):
        X, y, _ = coerce_inputs([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                resolve_backend(EngineConfig(backend="gpu"), X, y)
            except RuntimeError:
                pass
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))


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

    def test_native_ridge_baseline_matches_python_fallback_and_missing_symbol_is_safe(self):
        X_raw, y_raw = _dataset()
        X, y, _ = coerce_inputs(X_raw, y_raw)
        scores = {
            (0,): {"pearson": 0.90},
            (1,): {"pearson": 0.80},
            (0, 1): {"pearson": 0.70},
            (2,): {"pearson": 0.60},
            (0, 2): {"pearson": 0.50},
        }

        native_pred = engine_module._continuous_baseline_prediction(X, y, scores)
        with patch.object(engine_module, "_native_ridge_baseline", return_value=None) as fallback:
            python_pred = engine_module._continuous_baseline_prediction(X, y, scores)

        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(len(native_pred), len(python_pred))
        self.assertLess(
            max(abs(left - right) for left, right in zip(native_pred, python_pred)),
            1e-5,
        )

        with patch("importlib.import_module", return_value=object()):
            self.assertIsNone(
                engine_module._native_ridge_baseline(X, y, [(0,), (1,)], alpha=1.0)
            )


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
