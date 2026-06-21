import ctypes
import os
import unittest
from unittest.mock import patch

import gafime
from gafime import CompileFlags, ComputeBudget, EngineConfig, GafimeEngine
from gafime.backends.base import BackendInfo
from gafime.backends.core_backend import CoreBackend
from gafime.compile.sessions import BackendSession, ResidentContinuousMatrixSession
from gafime.metrics import MetricSuite
from gafime.time_series import TimeSeriesCandidate


def _dataset(n=96):
    X = [
        [
            (i - 20) / 10.0,
            ((i * 7) % 19 - 9) / 5.0,
            ((i * 5) % 13 - 6) / 4.0,
        ]
        for i in range(n)
    ]
    y = [row[0] * row[1] + 0.2 * row[2] for row in X]
    return X, y, ["a", "b", "c"]


class CompileApiTests(unittest.TestCase):
    def _config(self):
        return EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=16),
            permutation_tests=0,
            num_repeats=1,
        )

    def test_public_compile_api_exists(self):
        X, y, names = _dataset()
        artifact = gafime.compile(X, y, names, config=self._config())
        try:
            self.assertEqual(artifact.flags, CompileFlags())
            self.assertEqual(artifact.backend.name, "core")
            report = artifact.analyze()
            self.assertEqual(report.backend.name, "core")
            self.assertTrue(getattr(report.interactions, "is_native_backed", False))
            self.assertTrue(report.interactions)
        finally:
            artifact.close()

    def test_engine_compile_api_exists(self):
        X, y, names = _dataset()
        artifact = GafimeEngine(self._config()).compile(X, y, names)
        try:
            self.assertEqual(artifact.backend.name, "core")
            self.assertEqual(artifact.scenario_plan.n_features, 3)
        finally:
            artifact.close()

    def test_compiled_default_matches_legacy_output(self):
        X, y, names = _dataset()
        cfg = self._config()
        compiled = GafimeEngine(cfg).analyze(X, y, names)
        legacy = GafimeEngine(cfg)._analyze_legacy(X, y, names)
        self.assertEqual(
            [(item.combo, item.metrics) for item in compiled.interactions],
            [(item.combo, item.metrics) for item in legacy.interactions],
        )
        self.assertEqual(compiled.warnings, legacy.warnings)

    def test_compiled_all_families_match_legacy_output(self):
        X, y, names = _dataset()
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            enable_discrete_functions=True,
            enable_time_series_functions=True,
            discrete_quantiles=(0.25, 0.5, 0.75),
            time_series_lags=(1, 2),
            time_series_windows=(3,),
            budget=ComputeBudget(
                max_comb_size=2,
                max_combinations_per_k=10,
                max_discrete_candidates=16,
                max_thresholds_per_feature=2,
                top_k_features_for_discrete=2,
                max_time_series_candidates=12,
                top_k_features_for_time_series=2,
            ),
            permutation_tests=0,
            num_repeats=1,
        )
        compiled = GafimeEngine(cfg).analyze(X, y, names)
        legacy = GafimeEngine(cfg)._analyze_legacy(X, y, names)
        self.assertEqual(
            [
                (item.family, item.candidate_id, item.combo, item.metrics)
                for item in compiled.interactions
            ],
            [
                (item.family, item.candidate_id, item.combo, item.metrics)
                for item in legacy.interactions
            ],
        )
        self.assertEqual(compiled.warnings, legacy.warnings)

    def test_legacy_env_fallback(self):
        X, y, names = _dataset()
        cfg = self._config()
        old = os.environ.get("GAFIME_USE_LEGACY_ENGINE")
        os.environ["GAFIME_USE_LEGACY_ENGINE"] = "1"
        try:
            report = GafimeEngine(cfg).analyze(X, y, names)
        finally:
            if old is None:
                os.environ.pop("GAFIME_USE_LEGACY_ENGINE", None)
            else:
                os.environ["GAFIME_USE_LEGACY_ENGINE"] = old
        self.assertEqual(report.backend.name, "core")

    def test_compiled_analyze_scores_through_session(self):
        X, y, names = _dataset()

        class CountingSession(BackendSession):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.score_combos_calls = 0

            def score_combos(self, X_arg, y_arg, combos, metric_suite):
                self.score_combos_calls += 1
                return super().score_combos(X_arg, y_arg, combos, metric_suite)

        def compile_session(backend, X_arg, y_arg, scenario_plan, metric_suite, flags):
            return CountingSession(backend, X_arg, y_arg, scenario_plan, metric_suite, flags)

        with patch.object(CoreBackend, "compile_session", compile_session):
            artifact = GafimeEngine(self._config()).compile(X, y, names)
            try:
                report = artifact.analyze()
                self.assertTrue(report.interactions)
                self.assertGreaterEqual(artifact._session.score_combos_calls, 2)
            finally:
                artifact.close()

    def test_resident_session_updates_target_for_changed_y(self):
        class FakeData:
            def __init__(self, shape):
                self.buffer = object()
                self.shape = shape

        class FakeBackend:
            def __init__(self, supports_updates=True):
                self.fallback_calls = 0
                self.supports_updates = supports_updates
                self.update_calls = []

            def info(self):
                return BackendInfo("fake", "cpu", False, None, None)

            def supports_resident_target_update(self):
                return self.supports_updates

            def update_resident_target(self, matrix, y_arg):
                self.update_calls.append((matrix.value, y_arg.buffer))

            def score_combos(self, X_arg, y_arg, combos, metric_suite):
                self.fallback_calls += 1
                return {
                    tuple(combo): {name: 42.0 for name in metric_suite.metric_names}
                    for combo in combos
                }

        launch_calls = []

        def allocate_matrix(X_arg, y_arg):
            return ctypes.c_void_p(123), []

        def scheduler_batches(combos):
            arity = len(combos[0])
            indices = [idx for combo in combos for idx in combo]
            return [(None, indices, None, None, None, arity, len(combos))]

        def launch_global_batch(matrix, indices, arity, batch_size):
            launch_calls.append((matrix.value, tuple(indices), arity, batch_size))
            return [[7.0] for _ in range(batch_size)]

        backend = FakeBackend()
        X = FakeData((8, 2))
        y = FakeData((8,))
        metric_suite = MetricSuite(("pearson",))
        session = ResidentContinuousMatrixSession(
            backend,
            X,
            y,
            scenario_plan=None,
            metric_suite=metric_suite,
            flags=CompileFlags(),
            allocate_matrix=allocate_matrix,
            free_matrix=lambda matrix: None,
            launch_global_batch=launch_global_batch,
            scheduler_batches=scheduler_batches,
            stats_metric_names=lambda names: tuple(names),
            stats_to_metrics=lambda row, names: {
                name: float(row[idx]) for idx, name in enumerate(names)
            },
            complete_report_metrics=lambda X_arg, y_arg, combos, suite, scores: None,
            max_arity=2,
        )
        try:
            resident_scores = session.score_combos(X, y, [(0,)], metric_suite)
            self.assertEqual(resident_scores[(0,)]["pearson"], 7.0)
            self.assertEqual(backend.fallback_calls, 0)
            self.assertEqual(len(launch_calls), 1)

            permuted_y = FakeData((8,))
            permuted_scores = session.score_combos(X, permuted_y, [(0,)], metric_suite)
            self.assertEqual(permuted_scores[(0,)]["pearson"], 7.0)
            self.assertEqual(backend.update_calls, [(123, permuted_y.buffer)])
            self.assertEqual(backend.fallback_calls, 0)
            self.assertEqual(len(launch_calls), 2)

            restored_scores = session.score_combos(X, y, [(0,)], metric_suite)
            self.assertEqual(restored_scores[(0,)]["pearson"], 7.0)
            self.assertEqual(
                backend.update_calls,
                [(123, permuted_y.buffer), (123, y.buffer)],
            )
            self.assertEqual(backend.fallback_calls, 0)
            self.assertEqual(len(launch_calls), 3)

            sampled_scores = session.score_combos(FakeData((4, 2)), y, [(0,)], metric_suite)
            self.assertEqual(sampled_scores[(0,)]["pearson"], 42.0)
            self.assertEqual(backend.fallback_calls, 1)
            self.assertEqual(len(launch_calls), 3)
        finally:
            session.close()

    def test_resident_session_falls_back_when_target_update_unsupported(self):
        class FakeData:
            def __init__(self, shape):
                self.buffer = object()
                self.shape = shape

        class PredicateFalseBackend:
            def __init__(self):
                self.fallback_calls = 0
                self.update_calls = 0

            def info(self):
                return BackendInfo("fake", "cpu", False, None, None)

            def supports_resident_target_update(self):
                return False

            def update_resident_target(self, matrix, y_arg):
                self.update_calls += 1

            def score_combos(self, X_arg, y_arg, combos, metric_suite):
                self.fallback_calls += 1
                return {
                    tuple(combo): {name: 42.0 for name in metric_suite.metric_names}
                    for combo in combos
                }

        class MissingUpdaterBackend:
            def __init__(self):
                self.fallback_calls = 0

            def info(self):
                return BackendInfo("fake", "cpu", False, None, None)

            def supports_resident_target_update(self):
                return True

            def score_combos(self, X_arg, y_arg, combos, metric_suite):
                self.fallback_calls += 1
                return {
                    tuple(combo): {name: 43.0 for name in metric_suite.metric_names}
                    for combo in combos
                }

        def make_session(backend):
            return ResidentContinuousMatrixSession(
                backend,
                X,
                y,
                scenario_plan=None,
                metric_suite=metric_suite,
                flags=CompileFlags(),
                allocate_matrix=lambda X_arg, y_arg: (ctypes.c_void_p(123), []),
                free_matrix=lambda matrix: None,
                launch_global_batch=lambda matrix, indices, arity, batch_size: [[7.0]],
                scheduler_batches=lambda combos: [(None, [0], None, None, None, 1, 1)],
                stats_metric_names=lambda names: tuple(names),
                stats_to_metrics=lambda row, names: {
                    name: float(row[idx]) for idx, name in enumerate(names)
                },
                complete_report_metrics=lambda X_arg, y_arg, combos, suite, scores: None,
                max_arity=2,
            )

        X = FakeData((8, 2))
        y = FakeData((8,))
        y2 = FakeData((8,))
        metric_suite = MetricSuite(("pearson",))

        predicate_false = PredicateFalseBackend()
        session = make_session(predicate_false)
        try:
            scores = session.score_combos(X, y2, [(0,)], metric_suite)
            self.assertEqual(scores[(0,)]["pearson"], 42.0)
            self.assertEqual(predicate_false.fallback_calls, 1)
            self.assertEqual(predicate_false.update_calls, 0)
        finally:
            session.close()

        missing_updater = MissingUpdaterBackend()
        session = make_session(missing_updater)
        try:
            scores = session.score_combos(X, y2, [(0,)], metric_suite)
            self.assertEqual(scores[(0,)]["pearson"], 43.0)
            self.assertEqual(missing_updater.fallback_calls, 1)
        finally:
            session.close()

    def test_resident_session_scores_time_series_with_target_swaps(self):
        class FakeData:
            def __init__(self, shape):
                self.buffer = object()
                self.shape = shape

        class FakeBackend:
            def __init__(self):
                self.fallback_calls = 0
                self.resident_calls = 0
                self.update_calls = []

            def info(self):
                return BackendInfo("fake", "cpu", False, None, None)

            def supports_resident_target_update(self):
                return True

            def update_resident_target(self, matrix, y_arg):
                self.update_calls.append((matrix.value, y_arg.buffer))

            def score_time_series_candidates(self, X_arg, y_arg, candidates, metric_suite):
                self.fallback_calls += 1
                return {
                    candidate: {name: 42.0 for name in metric_suite.metric_names}
                    for candidate in candidates
                }

            def score_time_series_candidates_resident(self, matrix, X_arg, y_arg, candidates, metric_suite):
                self.resident_calls += 1
                return {
                    candidate: {name: 7.0 for name in metric_suite.metric_names}
                    for candidate in candidates
                }

        def make_session(backend, X, y, metric_suite):
            return ResidentContinuousMatrixSession(
                backend,
                X,
                y,
                scenario_plan=None,
                metric_suite=metric_suite,
                flags=CompileFlags(),
                allocate_matrix=lambda X_arg, y_arg: (ctypes.c_void_p(123), []),
                free_matrix=lambda matrix: None,
                launch_global_batch=lambda matrix, indices, arity, batch_size: [],
                scheduler_batches=lambda combos: [],
                stats_metric_names=lambda names: tuple(names),
                stats_to_metrics=lambda row, names: {},
                complete_report_metrics=lambda X_arg, y_arg, combos, suite, scores: None,
                max_arity=2,
            )

        backend = FakeBackend()
        X = FakeData((8, 2))
        y = FakeData((8,))
        y2 = FakeData((8,))
        metric_suite = MetricSuite(("pearson",))
        candidate = TimeSeriesCandidate("time_series_lag", feature_index=0, lag=1)
        session = make_session(backend, X, y, metric_suite)
        try:
            resident_scores = session.score_time_series_candidates(X, y, [candidate], metric_suite)
            self.assertEqual(resident_scores[candidate]["pearson"], 7.0)
            self.assertEqual(backend.resident_calls, 1)
            self.assertEqual(backend.update_calls, [])
            self.assertEqual(backend.fallback_calls, 0)

            swapped_scores = session.score_time_series_candidates(X, y2, [candidate], metric_suite)
            self.assertEqual(swapped_scores[candidate]["pearson"], 7.0)
            self.assertEqual(backend.resident_calls, 2)
            self.assertEqual(backend.update_calls, [(123, y2.buffer)])
            self.assertEqual(backend.fallback_calls, 0)

            fallback_scores = session.score_time_series_candidates(
                FakeData((8, 2)),
                y2,
                [candidate],
                metric_suite,
            )
            self.assertEqual(fallback_scores[candidate]["pearson"], 42.0)
            self.assertEqual(backend.resident_calls, 2)
            self.assertEqual(backend.fallback_calls, 1)
        finally:
            session.close()

    def test_resident_session_time_series_fallback_does_not_swap_without_resident_scorer(self):
        class FakeData:
            def __init__(self, shape):
                self.buffer = object()
                self.shape = shape

        class MissingResidentScorerBackend:
            def __init__(self):
                self.fallback_calls = 0
                self.update_calls = 0

            def info(self):
                return BackendInfo("fake", "cpu", False, None, None)

            def supports_resident_target_update(self):
                return True

            def update_resident_target(self, matrix, y_arg):
                self.update_calls += 1

            def score_time_series_candidates(self, X_arg, y_arg, candidates, metric_suite):
                self.fallback_calls += 1
                return {
                    candidate: {name: 42.0 for name in metric_suite.metric_names}
                    for candidate in candidates
                }

        backend = MissingResidentScorerBackend()
        X = FakeData((8, 2))
        y = FakeData((8,))
        y2 = FakeData((8,))
        metric_suite = MetricSuite(("pearson",))
        candidate = TimeSeriesCandidate("time_series_lag", feature_index=0, lag=1)
        session = ResidentContinuousMatrixSession(
            backend,
            X,
            y,
            scenario_plan=None,
            metric_suite=metric_suite,
            flags=CompileFlags(),
            allocate_matrix=lambda X_arg, y_arg: (ctypes.c_void_p(123), []),
            free_matrix=lambda matrix: None,
            launch_global_batch=lambda matrix, indices, arity, batch_size: [],
            scheduler_batches=lambda combos: [],
            stats_metric_names=lambda names: tuple(names),
            stats_to_metrics=lambda row, names: {},
            complete_report_metrics=lambda X_arg, y_arg, combos, suite, scores: None,
            max_arity=2,
        )
        try:
            scores = session.score_time_series_candidates(X, y2, [candidate], metric_suite)
            self.assertEqual(scores[candidate]["pearson"], 42.0)
            self.assertEqual(backend.fallback_calls, 1)
            self.assertEqual(backend.update_calls, 0)
        finally:
            session.close()

    def test_compiled_permutations_swap_resident_target_instead_of_fallback(self):
        X, y, names = _dataset()

        class CountingResidentBackend:
            def __init__(self):
                self.fallback_calls = 0
                self.update_calls = 0

            def info(self):
                return BackendInfo("resident-test", "cpu", False, None, None)

            def metric_suite(self, config):
                return MetricSuite(config.metric_names, mi_bins=config.mi_bins)

            def to_device(self, data):
                return data

            def to_host(self, data):
                return data

            def sample_indices(self, n_samples, rng):
                return list(range(n_samples))

            def permute(self, y_arg, rng):
                return y_arg.shuffled(rng)

            def supports_resident_target_update(self):
                return True

            def update_resident_target(self, matrix, y_arg):
                self.update_calls += 1

            def score_combos(self, X_arg, y_arg, combos, metric_suite):
                self.fallback_calls += 1
                return {
                    tuple(combo): {name: 0.25 for name in metric_suite.metric_names}
                    for combo in combos
                }

        def scheduler_batches(combos):
            batches = []
            for arity in sorted({len(combo) for combo in combos}):
                arity_combos = [combo for combo in combos if len(combo) == arity]
                indices = [idx for combo in arity_combos for idx in combo]
                batches.append((None, indices, None, None, None, arity, len(arity_combos)))
            return batches

        def compile_session(backend, X_arg, y_arg, scenario_plan, metric_suite, flags):
            resident_backend = CountingResidentBackend()
            session = ResidentContinuousMatrixSession(
                resident_backend,
                X_arg,
                y_arg,
                scenario_plan,
                metric_suite,
                flags,
                allocate_matrix=lambda X_native, y_native: (ctypes.c_void_p(123), []),
                free_matrix=lambda matrix: None,
                launch_global_batch=lambda matrix, indices, arity, batch_size: [
                    [1.0] for _ in range(batch_size)
                ],
                scheduler_batches=scheduler_batches,
                stats_metric_names=lambda metric_names: tuple(metric_names),
                stats_to_metrics=lambda row, metric_names: {
                    name: float(row[idx]) for idx, name in enumerate(metric_names)
                },
                complete_report_metrics=lambda X_arg, y_arg, combos, suite, scores: None,
                max_arity=2,
            )
            session.counting_backend = resident_backend
            return session

        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson",),
            budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
            permutation_tests=4,
            num_repeats=2,
        )
        with patch.object(CoreBackend, "compile_session", compile_session):
            artifact = GafimeEngine(cfg).compile(X, y, names)
            try:
                report = artifact.analyze()
                self.assertTrue(report.interactions)
                backend = artifact._session.counting_backend
                self.assertEqual(backend.update_calls, cfg.permutation_tests)
                self.assertEqual(backend.fallback_calls, cfg.num_repeats)
            finally:
                artifact.close()

    def test_compiled_time_series_permutations_reuse_resident_target_swaps(self):
        X, y, names = _dataset()

        class CountingResidentBackend:
            def __init__(self):
                self.combo_fallback_calls = 0
                self.ts_fallback_calls = 0
                self.ts_resident_calls = 0
                self.update_calls = 0

            def info(self):
                return BackendInfo("resident-ts-test", "cpu", False, None, None)

            def metric_suite(self, config):
                return MetricSuite(config.metric_names, mi_bins=config.mi_bins)

            def to_device(self, data):
                return data

            def to_host(self, data):
                return data

            def sample_indices(self, n_samples, rng):
                return list(range(n_samples))

            def permute(self, y_arg, rng):
                return y_arg.shuffled(rng)

            def supports_resident_target_update(self):
                return True

            def update_resident_target(self, matrix, y_arg):
                self.update_calls += 1

            def score_combos(self, X_arg, y_arg, combos, metric_suite):
                self.combo_fallback_calls += 1
                return {
                    tuple(combo): {name: 0.25 for name in metric_suite.metric_names}
                    for combo in combos
                }

            def score_time_series_candidates(self, X_arg, y_arg, candidates, metric_suite):
                self.ts_fallback_calls += 1
                return {
                    candidate: {name: 0.25 for name in metric_suite.metric_names}
                    for candidate in candidates
                }

            def score_time_series_candidates_resident(self, matrix, X_arg, y_arg, candidates, metric_suite):
                self.ts_resident_calls += 1
                return {
                    candidate: {name: 1.0 for name in metric_suite.metric_names}
                    for candidate in candidates
                }

        def scheduler_batches(combos):
            batches = []
            for arity in sorted({len(combo) for combo in combos}):
                arity_combos = [combo for combo in combos if len(combo) == arity]
                indices = [idx for combo in arity_combos for idx in combo]
                batches.append((None, indices, None, None, None, arity, len(arity_combos)))
            return batches

        def compile_session(backend, X_arg, y_arg, scenario_plan, metric_suite, flags):
            resident_backend = CountingResidentBackend()
            session = ResidentContinuousMatrixSession(
                resident_backend,
                X_arg,
                y_arg,
                scenario_plan,
                metric_suite,
                flags,
                allocate_matrix=lambda X_native, y_native: (ctypes.c_void_p(123), []),
                free_matrix=lambda matrix: None,
                launch_global_batch=lambda matrix, indices, arity, batch_size: [
                    [1.0] for _ in range(batch_size)
                ],
                scheduler_batches=scheduler_batches,
                stats_metric_names=lambda metric_names: tuple(metric_names),
                stats_to_metrics=lambda row, metric_names: {
                    name: float(row[idx]) for idx, name in enumerate(metric_names)
                },
                complete_report_metrics=lambda X_arg, y_arg, combos, suite, scores: None,
                max_arity=1,
            )
            session.counting_backend = resident_backend
            return session

        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson",),
            enable_time_series_functions=True,
            time_series_lags=(1,),
            time_series_windows=(),
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=8,
                max_time_series_candidates=4,
                top_k_features_for_time_series=1,
            ),
            permutation_tests=4,
            num_repeats=2,
        )
        with patch.object(CoreBackend, "compile_session", compile_session):
            artifact = GafimeEngine(cfg).compile(X, y, names)
            try:
                report = artifact.analyze()
                self.assertIn("time_series_function", {item.family for item in report.interactions})
                backend = artifact._session.counting_backend
                self.assertEqual(backend.ts_resident_calls, 1 + cfg.permutation_tests)
                self.assertEqual(backend.ts_fallback_calls, cfg.num_repeats)
                self.assertEqual(backend.combo_fallback_calls, cfg.num_repeats)
                self.assertGreaterEqual(backend.update_calls, cfg.permutation_tests)
            finally:
                artifact.close()

    def test_compiled_discrete_candidates_use_session_descriptor_table(self):
        X, y, names = _dataset()
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            enable_discrete_functions=True,
            discrete_quantiles=(0.25, 0.5, 0.75),
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=8,
                max_discrete_candidates=12,
                max_thresholds_per_feature=2,
                top_k_features_for_discrete=2,
            ),
            permutation_tests=0,
            num_repeats=1,
        )
        artifact = GafimeEngine(cfg).compile(X, y, names)
        try:
            report = artifact.analyze()
            self.assertIn("discrete_function", {item.family for item in report.interactions})
            table = artifact._session.candidate_table_handle
            self.assertIsNotNone(table)
            self.assertEqual(table.family, "discrete")
            self.assertGreater(len(table), 0)
        finally:
            artifact.close()

    def test_compiled_time_series_candidates_use_session_descriptor_table(self):
        X, y, names = _dataset()
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            enable_time_series_functions=True,
            time_series_lags=(1, 2),
            time_series_windows=(3,),
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=8,
                max_time_series_candidates=12,
                top_k_features_for_time_series=2,
            ),
            permutation_tests=0,
            num_repeats=1,
        )
        artifact = GafimeEngine(cfg).compile(X, y, names)
        try:
            report = artifact.analyze()
            self.assertIn("time_series_function", {item.family for item in report.interactions})
            table = artifact._session.time_series_candidate_table_handle
            self.assertIsNotNone(table)
            self.assertEqual(table.family, "time_series")
            self.assertGreater(len(table), 0)
        finally:
            artifact.close()

    def test_compiled_report_uses_native_sequences_for_all_result_families(self):
        X, y, names = _dataset()
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
            permutation_tests=2,
            num_repeats=2,
        )
        report = GafimeEngine(cfg).analyze(X, y, names)
        self.assertTrue(getattr(report.interactions, "is_native_backed", False))
        self.assertTrue(getattr(report.stability, "is_native_backed", False))
        self.assertTrue(getattr(report.permutations, "is_native_backed", False))

    def test_export_handles_are_tied_to_compiled_artifact_lifetime(self):
        X, y, names = _dataset()
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            enable_discrete_functions=True,
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=8,
                max_discrete_candidates=8,
                top_k_features_for_discrete=1,
            ),
            permutation_tests=0,
            num_repeats=1,
        )
        artifact = GafimeEngine(cfg).compile(X, y, names, flags=CompileFlags(export=True))
        try:
            before = artifact.exports
            self.assertEqual(before.backend_name, "core")
            self.assertIsNotNone(before.feature_matrix_handle)
            self.assertIsNone(before.result_table_handle)
            artifact.analyze()
            after = artifact.exports
            self.assertIsNotNone(after.result_table_handle)
            self.assertIsNotNone(after.candidate_table_handle)
        finally:
            artifact.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = artifact.exports

    def test_exports_require_export_flag(self):
        X, y, names = _dataset()
        artifact = GafimeEngine(self._config()).compile(X, y, names)
        try:
            with self.assertRaisesRegex(RuntimeError, "export handles are not available"):
                _ = artifact.exports
        finally:
            artifact.close()

    def test_repeated_compiled_analyze_reuses_rust_continuous_combo_plan(self):
        X, y, names = _dataset()
        artifact = GafimeEngine(self._config()).compile(X, y, names)
        try:
            first = artifact.analyze()
            second = artifact.analyze()
            self.assertEqual(
                [(item.combo, item.metrics) for item in first.interactions],
                [(item.combo, item.metrics) for item in second.interactions],
            )
            self.assertGreaterEqual(artifact._session.continuous_combo_cache_hits, 1)
        finally:
            artifact.close()

    def test_repeated_compiled_analyze_reuses_family_candidate_plans(self):
        X, y, names = _dataset()
        cfg = EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            enable_discrete_functions=True,
            enable_time_series_functions=True,
            discrete_quantiles=(0.25, 0.5, 0.75),
            time_series_lags=(1, 2),
            time_series_windows=(3,),
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=8,
                max_discrete_candidates=12,
                max_thresholds_per_feature=2,
                top_k_features_for_discrete=2,
                max_time_series_candidates=12,
                top_k_features_for_time_series=2,
            ),
            permutation_tests=0,
            num_repeats=1,
        )
        artifact = GafimeEngine(cfg).compile(X, y, names)
        try:
            first = artifact.analyze()
            second = artifact.analyze()
            self.assertEqual(
                [(item.candidate_id, item.combo, item.metrics) for item in first.interactions],
                [(item.candidate_id, item.combo, item.metrics) for item in second.interactions],
            )
            self.assertGreaterEqual(artifact._session.discrete_plan_cache_hits, 1)
            self.assertGreaterEqual(artifact._session.time_series_plan_cache_hits, 1)
            self.assertGreaterEqual(artifact._session.candidate_table_cache_hits, 2)
        finally:
            artifact.close()


if __name__ == "__main__":
    unittest.main()
