"""Emit deterministic installed-Core compatibility bits for cross-head comparison.

Run outside the checkout with the intended installed wheel. Compare stdout from
the base and candidate environments, and from one/default Rayon worker policy.
This is a regression snapshot, not a timing or feature-quality benchmark.
"""

from __future__ import annotations

import json

import numpy as np
from gafime import ComputeBudget, EngineConfig, GafimeEngine


def metric_bits(values: dict[str, float]) -> dict[str, str]:
    return {name: float(value).hex() for name, value in values.items()}


def main() -> None:
    rng = np.random.default_rng(7393)
    matrix = rng.normal(size=(48, 8))
    target = matrix[:, 0] * matrix[:, 3] + 0.25 * matrix[:, 5]
    results = []
    for profile in ("fp32", "mixed", "fp64"):
        for seed in (0, 7):
            config = EngineConfig(
                backend="core",
                precision=profile,
                metric_names=("pearson", "spearman", "mutual_info", "r2"),
                budget=ComputeBudget(
                    max_comb_size=3,
                    max_combinations_per_k=7,
                    top_features_for_higher_k=4,
                ),
                random_seed=seed,
                permutation_tests=5,
                num_repeats=2,
                significance_top_n=4,
            )
            report = GafimeEngine(config).analyze(matrix, target)
            results.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "interactions": [
                        [list(item.combo), metric_bits(item.metrics)]
                        for item in report.interactions
                    ],
                    "permutations": [
                        [list(item.combo), metric_bits(item.p_values)]
                        for item in report.permutations
                    ],
                    "stability": [
                        [
                            list(item.combo),
                            metric_bits(item.metrics_mean),
                            metric_bits(item.metrics_std),
                        ]
                        for item in report.stability
                    ],
                    "signal_detected": report.decision.signal_detected,
                }
            )
    print(json.dumps(results, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
