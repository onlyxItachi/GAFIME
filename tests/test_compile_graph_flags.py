import unittest

from gafime import CompileFlags, ComputeBudget, EngineConfig, GafimeEngine


class CompileGraphFlagTests(unittest.TestCase):
    def test_graph_flag_falls_back_without_breaking_core_analysis(self):
        X = [[float(i), float(i % 3)] for i in range(24)]
        y = [row[0] * row[1] for row in X]
        artifact = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
                permutation_tests=0,
                num_repeats=1,
            )
        ).compile(X, y, flags=CompileFlags(graph=True))
        try:
            self.assertEqual(artifact._session.graph_status, "fallback")
            self.assertTrue(any("graph capture requested" in item for item in artifact.warnings))
            report = artifact.analyze()
            self.assertTrue(report.interactions)
        finally:
            artifact.close()


if __name__ == "__main__":
    unittest.main()
