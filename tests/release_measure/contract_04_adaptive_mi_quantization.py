#!/usr/bin/env python3
"""Validate sample-adaptive MI templates against a large-sample reference."""

from __future__ import annotations

import numpy as np

from gafime import ComputeBudget, EngineConfig, GafimeEngine


MAX_ROWS = 73_728
FEATURES = 12
SEEDS = (11, 29, 47, 71, 101)
CASES = (
    (1_152, 8, 12),
    (4_608, 16, 24),
    (18_432, 32, 48),
)


def mi_scores(matrix: np.ndarray, target: np.ndarray, maximum_bins: int):
    config = EngineConfig(
        backend="core",
        metric_names=("mutual_info",),
        num_repeats=1,
        permutation_tests=0,
        mi_bins=maximum_bins,
        mi_approximate=True,
        budget=ComputeBudget(
            max_comb_size=2,
            max_combinations_per_k=10_000,
        ),
    )
    report = GafimeEngine(config=config).analyze(
        matrix.tolist(),
        target.tolist(),
        feature_names=[f"x{index}" for index in range(matrix.shape[1])],
    )
    scores = {}
    for interaction in report.interactions:
        combo = tuple(sorted(interaction.combo))
        if combo in scores:
            raise AssertionError(f"duplicate MI candidate: {combo}")
        value = float(interaction.metrics["mutual_info"])
        if not np.isfinite(value):
            raise AssertionError(f"non-finite MI score for {combo}: {value}")
        scores[combo] = value
    return scores


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def agreement(observed, reference) -> tuple[float, float, float, int]:
    if observed.keys() != reference.keys():
        raise AssertionError(
            "MI candidate identities changed across sample/template runs"
        )
    keys = sorted(reference)
    observed_values = np.asarray([observed[key] for key in keys])
    reference_values = np.asarray([reference[key] for key in keys])
    rank_correlation = float(
        np.corrcoef(
            average_ranks(observed_values),
            average_ranks(reference_values),
        )[0, 1]
    )
    score_correlation = float(np.corrcoef(observed_values, reference_values)[0, 1])
    if not np.isfinite(rank_correlation) or not np.isfinite(score_correlation):
        raise AssertionError(
            "MI agreement correlation is non-finite: "
            f"rank={rank_correlation} score={score_correlation}"
        )
    top_count = 12
    observed_top = set(np.argsort(observed_values, kind="stable")[-top_count:])
    reference_top = set(np.argsort(reference_values, kind="stable")[-top_count:])
    top_overlap = len(observed_top & reference_top) / top_count
    corrected_zeros = int(np.count_nonzero(observed_values == 0.0))
    return rank_correlation, score_correlation, top_overlap, corrected_zeros


def dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((MAX_ROWS, FEATURES), dtype=np.float32)
    noise = rng.standard_normal(MAX_ROWS, dtype=np.float32) * np.float32(0.35)
    target = (
        np.float32(0.9) * matrix[:, 0]
        + np.float32(0.7) * (matrix[:, 1] * matrix[:, 2])
        + np.float32(0.6) * (matrix[:, 3] > np.float32(0.25))
        + np.float32(0.45) * np.sin(matrix[:, 4] * np.float32(1.7))
        + np.float32(0.25) * (matrix[:, 5] * matrix[:, 6])
        + noise
    ).astype(np.float32)
    return matrix, target


def main() -> None:
    records: list[tuple[int, tuple[float, ...], tuple[float, ...]]] = []
    for seed in SEEDS:
        matrix, target = dataset(seed)
        reference = mi_scores(matrix, target, 96)
        for rows, coarse_bins, adaptive_bins in CASES:
            coarse = mi_scores(matrix[:rows], target[:rows], coarse_bins)
            adaptive = mi_scores(matrix[:rows], target[:rows], 96)
            explicit = mi_scores(matrix[:rows], target[:rows], adaptive_bins)
            if adaptive != explicit:
                raise AssertionError(
                    f"n={rows}: adaptive maximum 96 did not resolve to {adaptive_bins}"
                )
            coarse_agreement = agreement(coarse, reference)
            adaptive_agreement = agreement(adaptive, reference)
            if adaptive_agreement[0] <= coarse_agreement[0]:
                raise AssertionError(
                    f"n={rows} seed={seed}: adaptive rank stability regressed"
                )
            if adaptive_agreement[1] <= coarse_agreement[1]:
                raise AssertionError(
                    f"n={rows} seed={seed}: adaptive score stability regressed"
                )
            records.append((rows, coarse_agreement, adaptive_agreement))

    for rows, coarse_bins, adaptive_bins in CASES:
        selected = [record for record in records if record[0] == rows]
        coarse_rank = float(np.median([record[1][0] for record in selected]))
        adaptive_rank = float(np.median([record[2][0] for record in selected]))
        coarse_score = float(np.median([record[1][1] for record in selected]))
        adaptive_score = float(np.median([record[2][1] for record in selected]))
        coarse_top = float(np.median([record[1][2] for record in selected]))
        adaptive_top = float(np.median([record[2][2] for record in selected]))
        if adaptive_top < coarse_top:
            raise AssertionError(f"n={rows}: median top-12 overlap regressed")
        print(
            f"n={rows:>5} bins={coarse_bins:>2}->{adaptive_bins:>2} "
            f"rank={coarse_rank:.5f}->{adaptive_rank:.5f} "
            f"score={coarse_score:.5f}->{adaptive_score:.5f} "
            f"top12={coarse_top:.3f}->{adaptive_top:.3f}"
        )

    print("adaptive MI quantization contract: PASS")


if __name__ == "__main__":
    main()
