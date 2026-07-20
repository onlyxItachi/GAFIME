#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
import struct

import numpy as np

import gafime


def f32(value: float) -> np.float32:
    return np.float32(value)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(np.float32(value))))[0]


def column_means_f32(matrix: np.ndarray) -> list[np.float32]:
    means: list[np.float32] = []
    for col in range(matrix.shape[1]):
        total = 0.0
        for row in range(matrix.shape[0]):
            total += float(np.float32(matrix[row, col]))
        means.append(f32(total / matrix.shape[0]))
    return means


def interaction_signal(matrix: np.ndarray, combo: tuple[int, ...]) -> list[np.float32]:
    if len(combo) == 1:
        return [f32(value) for value in matrix[:, combo[0]]]

    means = column_means_f32(matrix)
    out = [f32(1.0) for _ in range(matrix.shape[0])]
    for col in combo:
        mean = means[col]
        for row in range(matrix.shape[0]):
            delta = f32(f32(matrix[row, col]) - mean)
            out[row] = f32(out[row] * delta)
    return out


def pearson_scalar_f32(signal: list[np.float32], target: list[np.float32]) -> np.float32:
    if len(signal) != len(target) or not signal:
        return f32(0.0)

    sx = 0.0
    sy = 0.0
    n = 0
    for x_value, y_value in zip(signal, target):
        x = float(np.float32(x_value))
        y = float(np.float32(y_value))
        if np.isfinite(x) and np.isfinite(y):
            sx += x
            sy += y
            n += 1
    if n == 0:
        return f32(0.0)

    mean_x = sx / n
    mean_y = sy / n
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for x_value, y_value in zip(signal, target):
        x = float(np.float32(x_value))
        y = float(np.float32(y_value))
        if np.isfinite(x) and np.isfinite(y):
            dx = x - mean_x
            dy = y - mean_y
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy

    denom = max(sxx * syy, 0.0) ** 0.5
    if denom <= 0.0:
        return f32(0.0)
    return f32(min(1.0, max(-1.0, sxy / denom)))


def r2_scalar_f32(signal: list[np.float32], target: list[np.float32]) -> np.float32:
    corr = pearson_scalar_f32(signal, target)
    return f32(min(1.0, max(0.0, float(f32(corr * corr)))))


def numpy_reference(matrix: np.ndarray, target: np.ndarray) -> dict[tuple[int, ...], dict[str, np.float32]]:
    target_values = [f32(value) for value in target]
    out: dict[tuple[int, ...], dict[str, np.float32]] = {}
    for arity in (1, 2):
        for combo in combinations(range(matrix.shape[1]), arity):
            signal = interaction_signal(matrix, combo)
            pearson = pearson_scalar_f32(signal, target_values)
            out[combo] = {
                "pearson": pearson,
                "r2": r2_scalar_f32(signal, target_values),
            }
    return out


def api_report_map(report: object) -> dict[tuple[int, ...], dict[str, float]]:
    mapped: dict[tuple[int, ...], dict[str, float]] = {}
    for item in report.interactions:
        # This oracle checks unary and pairwise math. The candidate-identity
        # gates separately preserve the legacy seeded tuple orientation.
        combo = tuple(sorted(int(value) for value in item.combo))
        if combo in mapped:
            raise AssertionError(f"duplicate canonical pair identity: {combo}")
        mapped[combo] = {
            str(name): float(value) for name, value in item.metrics.items()
        }
    return mapped


def assert_bit_equal(name: str, combo: tuple[int, ...], actual: float, expected: np.float32) -> None:
    actual_bits = f32_bits(actual)
    expected_bits = f32_bits(expected)
    if actual_bits != expected_bits:
        raise AssertionError(
            f"{combo} {name} bit mismatch: actual={actual:.9g} 0x{actual_bits:08x}, "
            f"expected={float(expected):.9g} 0x{expected_bits:08x}"
        )


def main() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 2.0],
            [2.0, 2.0, -1.0],
            [4.0, 5.0, 0.5],
        ],
        dtype=np.float32,
    )
    target = np.array([0.5, 1.5, 4.0], dtype=np.float32)
    feature_names = ["a", "b", "c"]

    cfg = gafime.EngineConfig(
        backend="core",
        metric_names=("pearson", "r2"),
        budget=gafime.ComputeBudget(max_comb_size=2, max_combinations_per_k=64),
        permutation_tests=0,
        num_repeats=1,
    )
    report = gafime.GafimeEngine(cfg).analyze(matrix.tolist(), target.tolist(), feature_names)
    actual = api_report_map(report)
    expected = numpy_reference(matrix, target)

    if set(actual) != set(expected):
        raise AssertionError(f"combo set mismatch: actual={sorted(actual)}, expected={sorted(expected)}")

    for combo, expected_metrics in expected.items():
        actual_metrics = actual[combo]
        for metric_name, expected_value in expected_metrics.items():
            assert_bit_equal(metric_name, combo, actual_metrics[metric_name], expected_value)

    print(f"top-level API NumPy bit parity verified for {len(expected)} candidates")


if __name__ == "__main__":
    main()
