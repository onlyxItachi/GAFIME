import os
import unittest
from unittest.mock import patch

import gafime
from gafime import CompileFlags, ComputeBudget, EngineConfig, GafimeEngine
from gafime.backends.core_backend import CoreBackend
from gafime.compile.sessions import BackendSession


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
