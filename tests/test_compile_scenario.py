import unittest

from gafime import CompileFlags, ComputeBudget, EngineConfig
from gafime import subfunctions
from gafime.compile.scenario import UINT128_MAX, build_scenario_plan
from gafime.utils.arrays import coerce_inputs


def _matrix(n_features, n_samples=8):
    X = [[float(i + j) for j in range(n_features)] for i in range(n_samples)]
    y = [float(i) for i in range(n_samples)]
    return coerce_inputs(X, y)[0]


class CompileScenarioPlanTests(unittest.TestCase):
    def test_rust_compile_plan_builder_is_exposed(self):
        self.assertTrue(hasattr(subfunctions, "CompilePlanBuilder"))
        native = subfunctions.CompilePlanBuilder().build(
            8,
            6,
            True,
            3,
            10,
            4,
            100_000,
            9,
            12,
            500,
            50,
            100_000,
            50,
            -2,
            False,
            False,
            [0.25, 0.5, 0.75],
            [1, 2],
            [3],
            1024,
        )
        self.assertEqual(native.n_features, 6)
        self.assertEqual(int(native.continuous_descriptors()[1]["universe_count"]), 6)

    def test_continuous_descriptors_use_caps_without_materializing(self):
        X = _matrix(6)
        plan = build_scenario_plan(
            X,
            EngineConfig(
                budget=ComputeBudget(
                    max_comb_size=3,
                    max_combinations_per_k=10,
                    top_features_for_higher_k=4,
                )
            ),
            CompileFlags(plan=True),
        )
        self.assertEqual([item.arity for item in plan.continuous], [1, 2, 3])
        self.assertEqual(plan.continuous[0].universe_count, 6)
        self.assertEqual(plan.continuous[1].universe_count, 6)
        self.assertEqual(plan.continuous[2].universe_count, 4)
        self.assertEqual(plan.continuous_count, 16)
        self.assertEqual(plan.continuous[1].offset, 6)

    def test_known_power_user_counts_wide_feature_space(self):
        X = _matrix(128)
        plan = build_scenario_plan(
            X,
            EngineConfig(
                budget=ComputeBudget(
                    max_feature_candidate=-1,
                    max_comb_size=5,
                    max_combinations_per_k=2,
                    top_features_for_higher_k=128,
                )
            ),
        )
        self.assertEqual(plan.feature_candidate_count, 128)
        self.assertFalse(plan.warnings)
        self.assertEqual(plan.continuous[4].universe_count, 264566400)
        self.assertEqual(plan.continuous[4].planned_count, 2)

    def test_unknown_power_user_gets_safety_cap_and_warning(self):
        X = _matrix(4096, n_samples=2)
        plan = build_scenario_plan(
            X,
            EngineConfig(budget=ComputeBudget(max_feature_candidate=-1)),
        )
        self.assertEqual(plan.feature_candidate_count, 1024)
        self.assertTrue(any("practical safety cap" in item for item in plan.warnings))

    def test_saturates_high_arity_counts(self):
        X = _matrix(10000, n_samples=1)
        plan = build_scenario_plan(
            X,
            EngineConfig(
                budget=ComputeBudget(
                    max_feature_candidate=-1,
                    max_comb_size=64,
                    max_combinations_per_k=-1,
                    top_features_for_higher_k=10000,
                )
            ),
        )
        self.assertEqual(plan.continuous[-1].universe_count, UINT128_MAX)
        self.assertTrue(plan.continuous[-1].saturated)
        self.assertEqual(plan.continuous[-1].offset_end, (1 << 64) - 1)

    def test_discrete_and_time_series_descriptor_counts(self):
        X = _matrix(5)
        cfg = EngineConfig(
            enable_discrete_functions=True,
            enable_time_series_functions=True,
            discrete_quantiles=(0.25, 0.5, 0.75),
            time_series_lags=(1, 2),
            time_series_windows=(3,),
            budget=ComputeBudget(
                max_comb_size=1,
                max_combinations_per_k=20,
                top_k_features_for_discrete=3,
                max_thresholds_per_feature=2,
                max_intervals_per_feature=1,
                max_feature_pairs_for_rectangles=2,
                max_discrete_candidates=100,
                top_k_features_for_time_series=4,
                max_time_series_candidates=100,
            ),
        )
        plan = build_scenario_plan(X, cfg)
        self.assertIsNotNone(plan.discrete)
        self.assertIsNotNone(plan.time_series)
        self.assertEqual(plan.discrete.threshold_count, 2)
        self.assertEqual(plan.discrete.interval_count, 1)
        self.assertEqual(plan.discrete.planned_count, 33)
        self.assertEqual(plan.time_series.template_count, 11)
        self.assertEqual(plan.time_series.planned_count, 44)


if __name__ == "__main__":
    unittest.main()
