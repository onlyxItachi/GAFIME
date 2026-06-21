import os
import unittest

import gafime
from gafime import CompileFlags, ComputeBudget, EngineConfig, GafimeEngine


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


if __name__ == "__main__":
    unittest.main()
