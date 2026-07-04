#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import tempfile

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeRegressor

import gafime

from contract_01_top_level_numpy_parity import (
    api_report_map,
    f32,
    f32_bits,
    numpy_reference,
    pearson_scalar_f32,
    r2_scalar_f32,
)


def assert_close(actual: float, expected: float, label: str, tol: float = 1.0e-6) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label}: actual={actual:.9g}, expected={expected:.9g}, tol={tol:g}")


def assert_bit_equal(actual: float, expected: np.float32, label: str) -> None:
    left = f32_bits(actual)
    right = f32_bits(expected)
    if left != right:
        raise AssertionError(
            f"{label}: actual={actual:.9g} 0x{left:08x}, expected={float(expected):.9g} 0x{right:08x}"
        )


def metric_reference_for_columns(matrix: np.ndarray, target: np.ndarray) -> dict[int, dict[str, np.float32]]:
    target_values = [f32(value) for value in target]
    out: dict[int, dict[str, np.float32]] = {}
    for col in range(matrix.shape[1]):
        signal = [f32(value) for value in matrix[:, col]]
        pearson = pearson_scalar_f32(signal, target_values)
        out[col] = {
            "pearson": pearson,
            "r2": r2_scalar_f32(signal, target_values),
        }
    return out


def expand_time_series_numpy(
    matrix: np.ndarray,
    base_names: list[str],
    lags: tuple[int, ...],
    windows: tuple[int, ...],
    velocity: bool,
) -> tuple[np.ndarray, list[str]]:
    rows, cols = matrix.shape
    generated: list[np.ndarray] = []
    names = list(base_names)
    for col in range(cols):
        values = matrix[:, col].astype(np.float32)
        for lag in lags:
            if lag == 0 or lag >= rows:
                continue
            feature = np.full(rows, np.nan, dtype=np.float32)
            feature[lag:] = values[:-lag]
            generated.append(feature)
            names.append(f"{base_names[col]}_lag{lag}")
            if velocity:
                delta = np.full(rows, np.nan, dtype=np.float32)
                delta[lag:] = values[lag:] - values[:-lag]
                generated.append(delta.astype(np.float32))
                names.append(f"{base_names[col]}_delta{lag}")
                velocity_feature = np.full(rows, np.nan, dtype=np.float32)
                velocity_feature[lag:] = (values[lag:] - values[:-lag]) / np.float32(lag)
                generated.append(velocity_feature.astype(np.float32))
                names.append(f"{base_names[col]}_velocity{lag}")
                if 2 * lag < rows:
                    acceleration = np.full(rows, np.nan, dtype=np.float32)
                    acceleration[2 * lag :] = (
                        values[2 * lag :]
                        - np.float32(2.0) * values[lag : rows - lag]
                        + values[: rows - 2 * lag]
                    ) / np.float32(lag * lag)
                    generated.append(acceleration.astype(np.float32))
                    names.append(f"{base_names[col]}_acceleration{lag}")
        for window in windows:
            if window < 2 or window > rows:
                continue
            mean_feature = np.full(rows, np.nan, dtype=np.float32)
            std_feature = np.full(rows, np.nan, dtype=np.float32)
            sum_feature = np.full(rows, np.nan, dtype=np.float32)
            for row in range(window - 1, rows):
                total = 0.0
                total2 = 0.0
                valid = True
                for item in values[row - window + 1 : row + 1]:
                    value = float(np.float32(item))
                    if not np.isfinite(value):
                        valid = False
                        break
                    total += value
                    total2 += value * value
                if valid:
                    mean = total / window
                    mean_feature[row] = f32(mean)
                    std_feature[row] = f32(max(0.0, total2 / window - mean * mean) ** 0.5)
                    sum_feature[row] = f32(total)
            generated.append(mean_feature)
            names.append(f"{base_names[col]}_rollmean{window}")
            generated.append(std_feature)
            names.append(f"{base_names[col]}_rollstd{window}")
            generated.append(sum_feature)
            names.append(f"{base_names[col]}_rollsum{window}")
    if generated:
        expanded = np.column_stack([matrix.astype(np.float32), *generated]).astype(np.float32)
    else:
        expanded = matrix.astype(np.float32)
    return expanded, names


_PATH_TERM = re.compile(r"^(?P<name>.+?)(?P<op><=|>)(?P<threshold>-?\d+(?:\.\d+)?)$")


def mask_from_path_label(label: str, matrix: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if not label.startswith("path[") or not label.endswith("]"):
        raise AssertionError(f"not a path label: {label}")
    body = label[len("path[") : -1]
    mask = np.ones(matrix.shape[0], dtype=bool)
    for raw_term in body.split(" & "):
        match = _PATH_TERM.match(raw_term)
        if match is None:
            raise AssertionError(f"cannot parse path term {raw_term!r} from {label!r}")
        name = match.group("name")
        col = feature_names.index(name)
        threshold = float(match.group("threshold"))
        if match.group("op") == "<=":
            mask &= matrix[:, col] <= threshold
        else:
            mask &= matrix[:, col] > threshold
    return mask


def verify_continuous_and_compile() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 2.0],
            [2.0, 2.0, -1.0],
            [4.0, 5.0, 0.5],
        ],
        dtype=np.float32,
    )
    target = np.array([0.5, 1.5, 4.0], dtype=np.float32)
    names = ["a", "b", "c"]
    cfg = gafime.EngineConfig(
        backend="core",
        metric_names=("pearson", "r2"),
        budget=gafime.ComputeBudget(max_comb_size=2, max_combinations_per_k=64),
        permutation_tests=0,
        num_repeats=1,
    )

    eager = gafime.GafimeEngine(cfg).analyze(matrix.tolist(), target.tolist(), names)
    compiled = gafime.GafimeEngine(cfg).compile(matrix.tolist(), target.tolist(), names)
    try:
        from_compiled = compiled.analyze()
    finally:
        compiled.close()

    expected = numpy_reference(matrix, target)
    eager_map = api_report_map(eager)
    compiled_map = api_report_map(from_compiled)
    if eager_map.keys() != expected.keys() or compiled_map.keys() != expected.keys():
        raise AssertionError("continuous compile/eager combo set does not match NumPy reference")
    if eager_map != compiled_map:
        raise AssertionError("compiled analyze output differs from eager analyze output")
    for combo, metrics in expected.items():
        for metric_name, expected_value in metrics.items():
            assert_bit_equal(eager_map[combo][metric_name], expected_value, f"continuous {combo} {metric_name}")
    print("continuous base/interaction generation and compile parity verified")


def verify_time_series_generation() -> None:
    matrix = np.array(
        [
            [1.0, 10.0],
            [2.0, 7.0],
            [4.0, 3.0],
        ],
        dtype=np.float32,
    )
    target = np.array([0.0, 1.0, 4.0], dtype=np.float32)
    base_names = ["sales", "cost"]
    lags = (1, 2)
    windows = (2,)
    cfg = gafime.EngineConfig(
        backend="core",
        enable_time_series_functions=True,
        time_series_lags=lags,
        time_series_windows=windows,
        metric_names=("pearson", "r2"),
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=64),
        permutation_tests=0,
        num_repeats=1,
    )
    report = gafime.GafimeEngine(cfg).analyze(matrix.tolist(), target.tolist(), base_names)
    compiled = gafime.GafimeEngine(cfg).compile(matrix.tolist(), target.tolist(), base_names)
    try:
        compiled_report = compiled.analyze()
    finally:
        compiled.close()
    expanded, expected_names = expand_time_series_numpy(matrix, base_names, lags, windows, True)
    if report.feature_names != expected_names:
        raise AssertionError(f"time-series feature names mismatch: {report.feature_names} != {expected_names}")
    if compiled_report.feature_names != expected_names:
        raise AssertionError(
            f"compiled time-series feature names mismatch: {compiled_report.feature_names} != {expected_names}"
        )

    expected = metric_reference_for_columns(expanded, target)
    actual = api_report_map(report)
    compiled_actual = api_report_map(compiled_report)
    if actual != compiled_actual:
        raise AssertionError("compiled time-series analyze output differs from eager analyze output")
    expected_combos = {(idx,) for idx in range(expanded.shape[1])}
    if set(actual) != expected_combos:
        raise AssertionError(f"time-series combo set mismatch: {sorted(actual)} != {sorted(expected_combos)}")
    for col, metrics in expected.items():
        combo = (col,)
        for metric_name, expected_value in metrics.items():
            assert_bit_equal(actual[combo][metric_name], expected_value, f"time-series {expected_names[col]} {metric_name}")

    signal_matrix = np.asarray([[float(i)] for i in range(80)], dtype=np.float32)
    signal_target = np.asarray([0.0] + [float(i - 1) for i in range(1, 80)], dtype=np.float32)
    sig_cfg = gafime.EngineConfig(
        backend="core",
        enable_time_series_functions=True,
        time_series_lags=(1,),
        time_series_windows=(),
        metric_names=("pearson",),
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=16),
        permutation_tests=50,
        num_repeats=5,
        permutation_p_threshold=0.05,
        stability_std_threshold=0.10,
    )
    sig_report = gafime.GafimeEngine(sig_cfg).analyze(
        signal_matrix.tolist(),
        signal_target.tolist(),
        ["x"],
    )
    if not sig_report.permutations or not sig_report.stability:
        raise AssertionError("time-series significance did not populate permutations/stability")
    if not sig_report.decision.signal_detected:
        raise AssertionError("time-series significance did not detect the lag signal")
    expected_ops = (
        "_lag1",
        "_delta1",
        "_velocity1",
        "_acceleration1",
        "_rollmean2",
        "_rollstd2",
        "_rollsum2",
    )
    for suffix in expected_ops:
        if not any(name.endswith(suffix) for name in report.feature_names):
            raise AssertionError(f"time-series feature type {suffix} was not generated: {report.feature_names}")

    print(f"time-series full lag/rolling generation verified for {len(expected_names) - len(base_names)} features")


def verify_decision_path_generation() -> None:
    rows: list[list[float]] = []
    target: list[float] = []
    for q0 in (0, 1):
        for q1 in (0, 1):
            for k in range(6):
                f0 = (0.2 if q0 == 0 else 0.8) + 0.001 * k
                f1 = (0.2 if q1 == 0 else 0.8) + 0.001 * k
                rows.append([f0, f1])
                target.append(5.0 if q0 == 1 and q1 == 1 else 0.0)
    matrix = np.asarray(rows, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    names = ["f0", "f1"]

    cfg = gafime.EngineConfig(
        backend="core",
        enable_decision_path_functions=True,
        metric_names=("pearson", "r2"),
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=128),
        permutation_tests=0,
        num_repeats=1,
        decision_path_max_depth=2,
        decision_path_rounds=1,
        decision_path_max_paths=8,
        decision_path_min_leaf=4,
        decision_path_learning_rate=1.0,
    )
    report = gafime.GafimeEngine(cfg).analyze(matrix.tolist(), y.tolist(), names)
    compiled = gafime.GafimeEngine(cfg).compile(matrix.tolist(), y.tolist(), names)
    try:
        compiled_report = compiled.analyze()
    finally:
        compiled.close()
    path_labels = [name for name in report.feature_names if name.startswith("path[")]
    if not path_labels:
        raise AssertionError(f"decision-path analysis generated no path features: {report.feature_names}")
    if compiled_report.feature_names != report.feature_names:
        raise AssertionError("compiled decision-path feature names differ from eager analyze")
    if api_report_map(compiled_report) != api_report_map(report):
        raise AssertionError("compiled decision-path analyze output differs from eager analyze output")

    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=4, random_state=0)
    tree.fit(matrix, y)
    leaf_ids = tree.apply(matrix)
    best_leaf = max(np.unique(leaf_ids), key=lambda leaf: float(y[leaf_ids == leaf].mean()))
    sklearn_mask = leaf_ids == best_leaf

    matching_label = None
    matching_mask = None
    for label in path_labels:
        mask = mask_from_path_label(label, matrix, names)
        if np.array_equal(mask, sklearn_mask):
            matching_label = label
            matching_mask = mask
            break
    if matching_label is None or matching_mask is None:
        raise AssertionError(
            f"no GAFIME path label matched sklearn depth-2 high-signal leaf; labels={path_labels}"
        )

    path_index = report.feature_names.index(matching_label)
    actual = api_report_map(report)[(path_index,)]
    signal = [f32(1.0 if value else 0.0) for value in matching_mask]
    expected_pearson = pearson_scalar_f32(signal, [f32(value) for value in y])
    expected_r2 = r2_scalar_f32(signal, [f32(value) for value in y])
    assert_close(actual["pearson"], expected_pearson, f"decision-path {matching_label} pearson")
    assert_close(actual["r2"], expected_r2, f"decision-path {matching_label} r2")
    print(f"decision-path generation verified against sklearn leaf: {matching_label}")


def verify_dataload_arrow_matches_direct_api() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 2.0],
            [2.0, 2.0, -1.0],
            [4.0, 5.0, 0.5],
        ],
        dtype=np.float32,
    )
    target = np.array([0.5, 1.5, 4.0], dtype=np.float32)
    names = ["a", "b", "c"]
    cfg = gafime.EngineConfig(
        backend="core",
        metric_names=("pearson", "r2"),
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=64),
        permutation_tests=0,
        num_repeats=1,
    )
    direct = gafime.GafimeEngine(cfg).analyze(matrix.tolist(), target.tolist(), names)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "features.csv"
        frame = pl.DataFrame({name: matrix[:, idx] for idx, name in enumerate(names)})
        frame = frame.with_columns(pl.Series("target", target))
        frame.write_csv(path)
        loaded = gafime.dataload(path, target="target", config=cfg)

    if loaded.feature_names != names:
        raise AssertionError(f"dataload feature names mismatch: {loaded.feature_names} != {names}")
    if api_report_map(loaded) != api_report_map(direct):
        raise AssertionError("dataload Arrow/native ingest output differs from direct top-level API output")
    print("dataload Arrow/native ingest matches direct top-level API")


def main() -> None:
    verify_continuous_and_compile()
    verify_time_series_generation()
    verify_decision_path_generation()
    verify_dataload_arrow_matches_direct_api()
    print("all top-level feature-generation reference checks passed")


if __name__ == "__main__":
    main()
