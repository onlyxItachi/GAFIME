"""Compare a legacy GAFIME distribution with a current v1 wheel.

The worker is intentionally import-compatible with both releases. Dataset
fetching and preprocessing happen in a separate ``prepare`` command, so timed
runs cover only GAFIME planning, upload, scoring, and result construction.

Example:

  python perf_08_v047_distribution_ab.py prepare --output-dir build/v047-ab/data
  <isolated-python> perf_08_v047_distribution_ab.py worker \
    --dataset build/v047-ab/data/openml_cpu_act_197.npz \
    --backend cuda --cache-size 0 --include-compiled \
    --top-features-for-higher-k 12 \
    --output build/v047-ab/results/current-cuda-cpu-act.json

Use Pearson and R2 for strict cross-version comparisons. MI and Spearman have
different backend/estimator routing in v0.4.7 and must be reported separately.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import struct
import sys
import time
from typing import Any, Iterable


DEFAULT_METRICS = ("pearson", "r2")
DEFAULT_SEED = 7


def _standardize_finite(values):
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        finite = np.isfinite(array)
        fill = float(np.median(array[finite])) if bool(finite.any()) else 0.0
        array = np.where(finite, array, fill)
        scale = float(array.std())
        if not math.isfinite(scale) or scale == 0.0:
            scale = 1.0
        return np.ascontiguousarray(
            (array - float(array.mean())) / scale, dtype=np.float32
        )

    if array.ndim != 2:
        raise ValueError(f"expected a 1D or 2D array, received shape={array.shape}")
    for column in range(array.shape[1]):
        values_column = array[:, column]
        finite = np.isfinite(values_column)
        fill = float(np.median(values_column[finite])) if bool(finite.any()) else 0.0
        array[:, column] = np.where(finite, values_column, fill)
    means = array.mean(axis=0)
    scales = array.std(axis=0)
    scales[~np.isfinite(scales) | (scales == 0.0)] = 1.0
    return np.ascontiguousarray((array - means) / scales, dtype=np.float32)


def _save_dataset(
    output_dir: Path,
    name: str,
    features,
    target,
    feature_names: Iterable[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    features = _standardize_finite(features)
    target = _standardize_finite(target)
    names = np.asarray([str(value) for value in feature_names], dtype=np.str_)
    if features.shape[0] != target.shape[0]:
        raise ValueError(f"{name}: feature and target row counts differ")
    if features.shape[1] != names.shape[0]:
        raise ValueError(f"{name}: feature-name count differs")
    if not bool(np.isfinite(features).all()) or not bool(np.isfinite(target).all()):
        raise ValueError(f"{name}: preprocessing left a non-finite value")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npz"
    np.savez(
        path,
        X=features,
        y=target,
        feature_names=names,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "bytes": int(features.nbytes + target.nbytes),
        **metadata,
    }


def prepare_datasets(output_dir: Path) -> list[dict[str, Any]]:
    import numpy as np
    from sklearn.datasets import fetch_openml, make_friedman1

    manifest: list[dict[str, Any]] = []
    features, target = make_friedman1(
        n_samples=32_768,
        n_features=24,
        noise=1.0,
        random_state=DEFAULT_SEED,
    )
    manifest.append(
        _save_dataset(
            output_dir,
            "sklearn_friedman1_32768x24",
            features,
            target,
            [f"x{index}" for index in range(features.shape[1])],
            {
                "source": "sklearn.make_friedman1",
                "seed": DEFAULT_SEED,
                "preprocessing": "median finite fill plus per-column z-score",
            },
        )
    )

    for data_id, file_name in (
        (197, "openml_cpu_act_197"),
        (574, "openml_house_16h_574"),
    ):
        dataset = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
        try:
            features = dataset.data.astype(float, copy=False)
            target = dataset.target.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OpenML {data_id} is not fully numeric") from exc
        if features.shape[1] == 0:
            raise ValueError(f"OpenML {data_id} has no numeric features")
        manifest.append(
            _save_dataset(
                output_dir,
                file_name,
                features,
                target,
                dataset.feature_names,
                {
                    "source": "OpenML",
                    "openml_id": data_id,
                    "openml_name": str(dataset.details.get("name", file_name)),
                    "openml_version": int(dataset.details.get("version", 0)),
                    "preprocessing": "numeric array, median finite fill, per-column z-score",
                },
            )
        )
        if data_id == 197:
            rng = np.random.default_rng(DEFAULT_SEED)
            subset = np.sort(rng.choice(features.shape[0], size=1_024, replace=False))
            manifest.append(
                _save_dataset(
                    output_dir,
                    "openml_cpu_act_197_1024",
                    features[subset],
                    target[subset],
                    dataset.feature_names,
                    {
                        "source": "OpenML deterministic subset",
                        "openml_id": data_id,
                        "openml_name": str(dataset.details.get("name", file_name)),
                        "openml_version": int(dataset.details.get("version", 0)),
                        "subset_rows": 1_024,
                        "seed": DEFAULT_SEED,
                        "preprocessing": (
                            "deterministic row subset, numeric array, median finite fill, "
                            "per-column z-score"
                        ),
                    },
                )
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _load_dataset(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        features = np.ascontiguousarray(payload["X"], dtype=np.float32)
        target = np.ascontiguousarray(payload["y"], dtype=np.float32)
        names = [str(value) for value in payload["feature_names"].tolist()]
        metadata = json.loads(str(payload["metadata"].item()))
    return features, target, names, metadata


def _candidate_count(
    features: int,
    max_arity: int,
    max_per_arity: int,
    top_features_for_higher_k: int,
) -> int:
    unary_count = min(features, max_per_arity)
    higher_features = min(unary_count, top_features_for_higher_k)
    higher_count = sum(
        min(math.comb(higher_features, arity), max_per_arity)
        for arity in range(2, min(higher_features, max_arity) + 1)
    )
    return unary_count + higher_count


def _memory_status() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmPeak"}:
                values[f"{key.lower()}_kib"] = int(value.strip().split()[0])
    except OSError:
        pass
    values["ru_maxrss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return values


def _snapshot(report, metric_names: tuple[str, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in report.interactions:
        combo = tuple(int(value) for value in item.combo)
        metrics = tuple(float(item.metrics[name]) for name in metric_names)
        if not all(math.isfinite(value) for value in metrics):
            raise AssertionError(f"non-finite metric for combo={combo}: {metrics}")
        rows.append(
            {
                "combo": combo,
                "family": str(getattr(item, "family", "interaction") or "interaction"),
                "candidate_id": str(getattr(item, "candidate_id", "") or ""),
                "metrics": metrics,
            }
        )
    if len({row["combo"] for row in rows}) != len(rows):
        raise AssertionError("candidate identities are not unique")
    populated_ids = [row["candidate_id"] for row in rows if row["candidate_id"]]
    if populated_ids and len(populated_ids) != len(rows):
        raise AssertionError("candidate ids are only partially populated")
    if len(set(populated_ids)) != len(populated_ids):
        raise AssertionError("candidate ids are not unique")

    digest = hashlib.sha256()
    identity_digest = hashlib.sha256()
    candidate_id_digest = hashlib.sha256()
    for row_index, row in enumerate(rows):
        combo = row["combo"]
        values = row["metrics"]
        family = row["family"].encode("utf-8")
        candidate_id = row["candidate_id"].encode("utf-8")
        packed_combo = struct.pack(f"<{len(combo)}I", *combo)
        identity_digest.update(struct.pack("<Q", row_index))
        identity_digest.update(struct.pack("<I", len(combo)))
        identity_digest.update(packed_combo)
        identity_digest.update(struct.pack("<I", len(family)))
        identity_digest.update(family)
        candidate_id_digest.update(struct.pack("<I", len(candidate_id)))
        candidate_id_digest.update(candidate_id)
        digest.update(struct.pack("<I", len(combo)))
        digest.update(packed_combo)
        digest.update(struct.pack(f"<{len(values)}f", *values))

    primary = metric_names.index("pearson") if "pearson" in metric_names else 0
    ranked = sorted(
        enumerate(rows),
        key=lambda value: (-abs(value[1]["metrics"][primary]), value[0]),
    )
    stability = _significance_snapshot(
        getattr(report, "stability", []) or [],
        metric_names,
        ("metrics_mean", "metrics_std"),
    )
    permutations = _significance_snapshot(
        getattr(report, "permutations", []) or [],
        metric_names,
        ("p_values",),
    )
    decision = getattr(report, "decision", None)
    return {
        "candidate_identity_contract": "report-order-feature-tuples-families-v3",
        "candidate_count": len(rows),
        "candidate_identity_sha256": identity_digest.hexdigest(),
        "candidate_id_contract": "stable-unique" if populated_ids else "legacy-empty",
        "candidate_id_sha256": candidate_id_digest.hexdigest(),
        "metric_sha256_f32": digest.hexdigest(),
        "top20": [
            {
                "combo": list(row["combo"]),
                "family": row["family"],
                "candidate_id": row["candidate_id"],
                "metrics": list(row["metrics"]),
            }
            for _, row in ranked[:20]
        ],
        "scores": [
            {
                "combo": list(row["combo"]),
                "family": row["family"],
                "candidate_id": row["candidate_id"],
                "metrics": list(row["metrics"]),
            }
            for row in rows
        ],
        "stability": stability,
        "permutations": permutations,
        "warnings": [str(value) for value in (getattr(report, "warnings", []) or [])],
        "decision": None
        if decision is None
        else {
            "signal_detected": bool(decision.signal_detected),
            "message": str(decision.message),
        },
    }


def _significance_snapshot(
    items: Any,
    metric_names: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        values: dict[str, list[float]] = {}
        for field in value_fields:
            mapping = getattr(item, field)
            field_values = [float(mapping[name]) for name in metric_names]
            if not all(math.isfinite(value) for value in field_values):
                raise AssertionError(f"non-finite {field} values: {field_values}")
            values[field] = field_values
        rows.append(
            {
                "combo": [int(value) for value in item.combo],
                "family": str(getattr(item, "family", "interaction") or "interaction"),
                "candidate_id": str(getattr(item, "candidate_id", "") or ""),
                **values,
            }
        )
    return rows


def _mi_estimator_contract(
    version: str,
    backend: dict[str, Any],
    metric_names: tuple[str, ...],
    requested_approximation: bool,
) -> str | None:
    if "mutual_info" not in metric_names:
        return None
    if backend["is_gpu"]:
        if version == "0.4.7":
            return "host-adaptive-quantile-completion"
        return "device-fixed-bin-adaptive-template"
    if requested_approximation:
        return "cpu-fixed-bin-adaptive-template"
    return "cpu-adaptive-quantile"


def _snapshot_max_abs_deltas(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    metric_names: tuple[str, ...],
    *,
    cross_distribution: bool = False,
) -> dict[str, float]:
    if candidate.get("candidate_identity_contract") != reference.get(
        "candidate_identity_contract"
    ):
        raise AssertionError("candidate identity contracts differ")
    if candidate["candidate_identity_sha256"] != reference["candidate_identity_sha256"]:
        raise AssertionError("candidate identities changed")
    _assert_candidate_id_contract(
        reference,
        candidate,
        cross_distribution=cross_distribution,
    )
    if candidate.get("warnings", []) != reference.get("warnings", []):
        raise AssertionError("public warnings changed")
    reference_decision = reference.get("decision")
    candidate_decision = candidate.get("decision")
    if (reference_decision is None) != (candidate_decision is None):
        raise AssertionError("decision presence changed")
    if reference_decision is not None and (
        reference_decision["signal_detected"]
        != candidate_decision["signal_detected"]
    ):
        raise AssertionError("decision signal changed")
    if (
        not cross_distribution
        and reference_decision is not None
        and reference_decision["message"] != candidate_decision["message"]
    ):
        raise AssertionError("decision message changed")
    reference_scores = reference["scores"]
    candidate_scores = candidate["scores"]
    if len(reference_scores) != len(candidate_scores):
        raise AssertionError("candidate score counts changed")

    deltas = {name: 0.0 for name in metric_names}
    for reference_row, candidate_row in zip(reference_scores, candidate_scores):
        _assert_result_identity(
            reference_row,
            candidate_row,
            cross_distribution=cross_distribution,
        )
        for metric_index, metric_name in enumerate(metric_names):
            delta = abs(
                float(reference_row["metrics"][metric_index])
                - float(candidate_row["metrics"][metric_index])
            )
            deltas[metric_name] = max(deltas[metric_name], delta)
    for collection_name, value_fields in (
        ("stability", ("metrics_mean", "metrics_std")),
        ("permutations", ("p_values",)),
    ):
        reference_rows = reference.get(collection_name, [])
        candidate_rows = candidate.get(collection_name, [])
        if len(reference_rows) != len(candidate_rows):
            raise AssertionError(f"{collection_name} row count changed")
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            _assert_result_identity(
                reference_row,
                candidate_row,
                cross_distribution=cross_distribution,
            )
            for value_field in value_fields:
                for metric_index, metric_name in enumerate(metric_names):
                    delta_key = f"{collection_name}.{value_field}.{metric_name}"
                    delta = abs(
                        float(reference_row[value_field][metric_index])
                        - float(candidate_row[value_field][metric_index])
                    )
                    deltas[delta_key] = max(deltas.get(delta_key, 0.0), delta)
    return deltas


def _assert_candidate_id_contract(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    cross_distribution: bool,
) -> None:
    reference_contract = reference.get("candidate_id_contract", "legacy-empty")
    candidate_contract = candidate.get("candidate_id_contract", "legacy-empty")
    if (
        cross_distribution
        and reference_contract == "legacy-empty"
        and candidate_contract == "stable-unique"
    ):
        return
    if reference_contract != candidate_contract:
        raise AssertionError("candidate id population changed")
    if reference.get("candidate_id_sha256") != candidate.get("candidate_id_sha256"):
        raise AssertionError("candidate ids changed")


def _assert_result_identity(
    reference_row: dict[str, Any],
    candidate_row: dict[str, Any],
    *,
    cross_distribution: bool,
) -> None:
    for field in ("combo", "family"):
        if candidate_row[field] != reference_row[field]:
            raise AssertionError(f"candidate report order or {field} changed")
    reference_id = reference_row.get("candidate_id", "")
    candidate_id = candidate_row.get("candidate_id", "")
    if not (cross_distribution and not reference_id and candidate_id):
        if candidate_id != reference_id:
            raise AssertionError("candidate id changed")


def _normalized_work(result: dict[str, Any]) -> dict[str, Any]:
    work = copy.deepcopy(result["work"])
    work.setdefault(
        "top_features_for_higher_k", int(result["dataset"]["features"])
    )
    work.setdefault("num_repeats", 1)
    work.setdefault("permutation_tests", 0)
    work.setdefault("random_seed", DEFAULT_SEED)
    return work


def _run_once(call, metric_names: tuple[str, ...]):
    start = time.perf_counter_ns()
    report = call()
    report_ns = time.perf_counter_ns() - start
    snapshot = _snapshot(report, metric_names)
    total_ns = time.perf_counter_ns() - start
    return (
        report,
        snapshot,
        {
            "report_ns": report_ns,
            "materialize_ns": total_ns - report_ns,
            "total_ns": total_ns,
        },
    )


def _summarize_samples(
    samples: list[dict[str, int]], candidate_pairs: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": samples}
    for field in ("report_ns", "materialize_ns", "total_ns"):
        values = [int(sample[field]) for sample in samples]
        median_ns = int(statistics.median(values))
        result[field] = {
            "min": min(values),
            "median": median_ns,
            "max": max(values),
        }
        if field != "materialize_ns":
            result[field]["candidate_sample_gevals_per_second"] = (
                candidate_pairs / median_ns
            )
    return result


def _package_provenance(gafime_module) -> dict[str, Any]:
    import numpy as np
    import sklearn

    package_file = Path(gafime_module.__file__).resolve()
    distributions: dict[str, Any] = {}
    for distribution_name in ("gafime", "gafime-cuda", "gafime-rocm"):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        native_files = []
        for relative_path in distribution.files or ():
            path = Path(distribution.locate_file(relative_path))
            if path.suffix not in {".so", ".dylib", ".pyd"} or not path.is_file():
                continue
            native_files.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        distributions[distribution_name] = {
            "version": distribution.version,
            "native_files": native_files,
        }

    payloads: dict[str, Any] = {}
    for name, environment_name in (
        ("cuda", "GAFIME_CUDA_V1_LIB"),
        ("rocm", "GAFIME_ROCM_V1_LIB"),
        ("metal", "GAFIME_METAL_V1_LIB"),
    ):
        raw_path = os.environ.get(environment_name)
        if raw_path:
            path = Path(raw_path).resolve()
            payloads[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    provenance = {
        "gafime_version": str(gafime_module.__version__),
        "gafime_package": str(package_file),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "numpy": str(np.__version__),
        "sklearn": str(sklearn.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_head": os.environ.get("GAFIME_BENCH_GIT_HEAD"),
        "distributions": distributions,
        "payloads": payloads,
    }
    try:
        from gafime import gafime_core

        provenance["native_cpu_dispatch"] = str(gafime_core.cpu_dispatch_target())
        provenance["available_cpu_dispatch"] = list(
            gafime_core.available_cpu_dispatch_targets()
        )
    except (ImportError, AttributeError):
        provenance["native_cpu_dispatch"] = None
        provenance["available_cpu_dispatch"] = []
    return provenance


def run_worker(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.cache_size is not None:
        os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = str(arguments.cache_size)

    import gafime
    from gafime import ComputeBudget, EngineConfig, GafimeEngine

    provenance = _package_provenance(gafime)
    features, target, names, dataset_metadata = _load_dataset(arguments.dataset)
    rows, columns = (int(features.shape[0]), int(features.shape[1]))
    if arguments.input_format == "lists":
        benchmark_features = features.tolist()
        benchmark_target = target.tolist()
    else:
        benchmark_features = features
        benchmark_target = target
    top_features_for_higher_k = (
        columns
        if arguments.top_features_for_higher_k is None
        else arguments.top_features_for_higher_k
    )
    expected_candidates = _candidate_count(
        columns,
        arguments.max_arity,
        arguments.max_combinations_per_arity,
        top_features_for_higher_k,
    )
    candidate_pairs = expected_candidates * rows
    metric_names = tuple(arguments.metrics)
    budget = ComputeBudget(
        max_comb_size=arguments.max_arity,
        max_combinations_per_k=arguments.max_combinations_per_arity,
        top_features_for_higher_k=top_features_for_higher_k,
        keep_in_vram=True,
    )
    config_kwargs: dict[str, Any] = {
        "budget": budget,
        "metric_names": metric_names,
        "num_repeats": arguments.num_repeats,
        "permutation_tests": arguments.permutation_tests,
        "random_seed": arguments.random_seed,
        "backend": arguments.backend,
    }
    if "mi_approximate" in getattr(EngineConfig, "__dataclass_fields__", {}):
        config_kwargs["mi_approximate"] = bool(arguments.mi_approximate)
    config = EngineConfig(**config_kwargs)
    engine = GafimeEngine(config)

    memory_before = _memory_status()
    gc.collect()
    first_report, first_snapshot, first_timing = _run_once(
        lambda: engine.analyze(
            benchmark_features, benchmark_target, feature_names=names
        ),
        metric_names,
    )
    if first_snapshot["candidate_count"] != expected_candidates:
        raise AssertionError(
            f"expected {expected_candidates} candidates, received "
            f"{first_snapshot['candidate_count']}"
        )
    if (
        arguments.expect_candidate_identity_sha256 is not None
        and first_snapshot["candidate_identity_sha256"]
        != arguments.expect_candidate_identity_sha256
    ):
        raise AssertionError(
            "candidate identity differs from --expect-candidate-identity-sha256"
        )

    repeated_samples: list[dict[str, int]] = []
    repeated_snapshot_max_abs = {name: 0.0 for name in metric_names}
    last_report = first_report
    last_snapshot = first_snapshot
    for _ in range(arguments.repeats):
        gc.collect()
        time.sleep(arguments.pause_seconds)
        last_report, last_snapshot, timing = _run_once(
            lambda: engine.analyze(
                benchmark_features, benchmark_target, feature_names=names
            ),
            metric_names,
        )
        deltas = _snapshot_max_abs_deltas(first_snapshot, last_snapshot, metric_names)
        for metric_name, delta in deltas.items():
            repeated_snapshot_max_abs[metric_name] = max(
                repeated_snapshot_max_abs[metric_name], delta
            )
        if max(deltas.values(), default=0.0) > 1e-5:
            raise AssertionError("repeated eager scores differ beyond the 1e-5 guard")
        repeated_samples.append(timing)

    backend_info = getattr(last_report, "backend", None)
    backend_payload = {
        "name": str(getattr(backend_info, "name", "unknown")),
        "device": str(getattr(backend_info, "device", "unknown")),
        "is_gpu": bool(getattr(backend_info, "is_gpu", False)),
    }
    mi_estimator = _mi_estimator_contract(
        provenance["gafime_version"],
        backend_payload,
        metric_names,
        bool(arguments.mi_approximate),
    )
    result: dict[str, Any] = {
        "schema": "gafime.legacy-current-ab.v3",
        "provenance": provenance,
        "dataset": {
            "path": str(arguments.dataset.resolve()),
            "sha256": hashlib.sha256(arguments.dataset.read_bytes()).hexdigest(),
            "rows": rows,
            "features": columns,
            "bytes": int(features.nbytes + target.nbytes),
            **dataset_metadata,
        },
        "work": {
            "max_arity": arguments.max_arity,
            "max_combinations_per_arity": arguments.max_combinations_per_arity,
            "top_features_for_higher_k": top_features_for_higher_k,
            "metrics": list(metric_names),
            "candidates": expected_candidates,
            "candidate_sample_pairs": candidate_pairs,
            "full_python_materialization": True,
            "input_format": arguments.input_format,
            "mi_approximate_requested": bool(arguments.mi_approximate),
            "mi_estimator": mi_estimator,
            "num_repeats": arguments.num_repeats,
            "permutation_tests": arguments.permutation_tests,
            "random_seed": arguments.random_seed,
        },
        "backend": backend_payload,
        "cache_size": arguments.cache_size,
        "requested_eager_cache_policy": (
            "package-default"
            if arguments.cache_size is None
            else "disabled"
            if arguments.cache_size == 0
            else "resident-lru"
        ),
        "first_eager": {
            **first_timing,
            "report_candidate_sample_gevals_per_second": candidate_pairs
            / first_timing["report_ns"],
            "total_candidate_sample_gevals_per_second": candidate_pairs
            / first_timing["total_ns"],
        },
        "repeated_eager": _summarize_samples(repeated_samples, candidate_pairs),
        "repeated_snapshot_max_abs": repeated_snapshot_max_abs,
        "snapshot": last_snapshot,
        "memory": {
            "before": memory_before,
            "after_eager": _memory_status(),
        },
    }

    if arguments.include_compiled:
        if not hasattr(engine, "compile"):
            raise AssertionError(
                "--include-compiled requested, but this distribution has no compile API"
            )
        gc.collect()
        compile_start = time.perf_counter_ns()
        artifact = engine.compile(
            benchmark_features, benchmark_target, feature_names=names
        )
        compile_ns = time.perf_counter_ns() - compile_start
        try:
            compiled_samples: list[dict[str, int]] = []
            compiled_snapshot = None
            for _ in range(arguments.repeats):
                gc.collect()
                time.sleep(arguments.pause_seconds)
                _, compiled_snapshot, timing = _run_once(artifact.analyze, metric_names)
                deltas = _snapshot_max_abs_deltas(
                    first_snapshot, compiled_snapshot, metric_names
                )
                if max(deltas.values(), default=0.0) > 1e-5:
                    raise AssertionError(
                        "compiled scores differ from eager beyond the 1e-5 guard"
                    )
                compiled_samples.append(timing)
            result["compiled"] = {
                "compile_ns": compile_ns,
                "compile_candidate_sample_gevals_per_second": candidate_pairs
                / compile_ns,
                "replay": _summarize_samples(compiled_samples, candidate_pairs),
                "snapshot": compiled_snapshot,
                "metric_max_abs_vs_eager": _snapshot_max_abs_deltas(
                    first_snapshot, compiled_snapshot, metric_names
                ),
                "continuous_metric_cache_hits": int(
                    getattr(artifact, "continuous_metric_cache_hits", 0)
                ),
                "continuous_metric_cache_builds": int(
                    getattr(artifact, "continuous_metric_cache_builds", 0)
                ),
                "candidate_table_cache_hits": int(
                    getattr(artifact, "candidate_table_cache_hits", 0)
                ),
            }
        finally:
            artifact.close()
        result["memory"]["after_compiled"] = _memory_status()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _print_worker_summary(result: dict[str, Any]) -> None:
    work = result["work"]
    repeated = result["repeated_eager"]["total_ns"]
    print(
        f"GAFIME {result['provenance']['gafime_version']} "
        f"backend={result['backend']['name']} rows={result['dataset']['rows']:,} "
        f"candidates={work['candidates']:,} pairs={work['candidate_sample_pairs']:,}"
    )
    print(
        f"  repeated eager full median={repeated['median'] / 1e6:.3f}ms "
        f"rate={repeated['candidate_sample_gevals_per_second']:.3f}GEval/s"
    )
    if "compiled" in result:
        replay = result["compiled"]["replay"]["total_ns"]
        print(
            f"  compiled full median={replay['median'] / 1e6:.3f}ms "
            f"rate={replay['candidate_sample_gevals_per_second']:.3f}GEval/s"
        )


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for offset in range(start, end):
            ranks[ordered[offset]] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must have equal non-zero length")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_norm * right_norm)


def compare_results(
    baseline_path: Path,
    candidate_path: Path,
    *,
    max_metric_abs: float | None = None,
    min_eager_full_speedup: float | None = None,
    min_compiled_full_speedup: float | None = None,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for key in ("python", "numpy", "sklearn", "platform", "machine"):
        if baseline["provenance"][key] != candidate["provenance"][key]:
            raise AssertionError(f"baseline and candidate environments differ at {key}")
    baseline_work = _normalized_work(baseline)
    candidate_work = _normalized_work(candidate)
    if baseline_work != candidate_work:
        raise AssertionError("baseline and candidate work definitions differ")
    baseline_snapshot = baseline["snapshot"]
    candidate_snapshot = candidate["snapshot"]
    if baseline_snapshot.get("candidate_identity_contract") != candidate_snapshot.get(
        "candidate_identity_contract"
    ):
        raise AssertionError("baseline and candidate identity contracts differ")
    if (
        baseline_snapshot["candidate_identity_sha256"]
        != candidate_snapshot["candidate_identity_sha256"]
    ):
        raise AssertionError("baseline and candidate identities differ")
    report_value_deltas = _snapshot_max_abs_deltas(
        baseline_snapshot,
        candidate_snapshot,
        tuple(baseline["work"]["metrics"]),
        cross_distribution=True,
    )

    baseline_scores = baseline_snapshot["scores"]
    candidate_scores = candidate_snapshot["scores"]
    if len(baseline_scores) != len(candidate_scores):
        raise AssertionError("baseline and candidate score counts differ")
    metric_names = baseline["work"]["metrics"]
    deltas: dict[str, Any] = {}
    for metric_index, metric_name in enumerate(metric_names):
        baseline_values = []
        candidate_values = []
        absolute_deltas = []
        for baseline_row, candidate_row in zip(baseline_scores, candidate_scores):
            baseline_value = float(baseline_row["metrics"][metric_index])
            candidate_value = float(candidate_row["metrics"][metric_index])
            baseline_values.append(baseline_value)
            candidate_values.append(candidate_value)
            absolute_deltas.append(abs(baseline_value - candidate_value))
        deltas[metric_name] = {
            "max_abs": max(absolute_deltas, default=0.0),
            "median_abs": statistics.median(absolute_deltas)
            if absolute_deltas
            else 0.0,
            "mean_abs": statistics.fmean(absolute_deltas) if absolute_deltas else 0.0,
            "rank_correlation": _pearson(
                _average_ranks([abs(value) for value in baseline_values]),
                _average_ranks([abs(value) for value in candidate_values]),
            ),
        }

    baseline_top = {tuple(row["combo"]) for row in baseline_snapshot["top20"]}
    candidate_top = {tuple(row["combo"]) for row in candidate_snapshot["top20"]}
    baseline_total = baseline["repeated_eager"]["total_ns"]["median"]
    candidate_total = candidate["repeated_eager"]["total_ns"]["median"]
    baseline_report = baseline["repeated_eager"]["report_ns"]["median"]
    candidate_report = candidate["repeated_eager"]["report_ns"]["median"]
    comparison: dict[str, Any] = {
        "schema": "gafime.legacy-current-comparison.v3",
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "baseline_version": baseline["provenance"]["gafime_version"],
        "candidate_version": candidate["provenance"]["gafime_version"],
        "dataset": candidate["dataset"],
        "backend": {
            "baseline": baseline["backend"],
            "candidate": candidate["backend"],
        },
        "work": candidate_work,
        "candidate_identity_match": True,
        "candidate_id_contract": {
            "baseline": baseline_snapshot.get("candidate_id_contract", "legacy-empty"),
            "candidate": candidate_snapshot.get("candidate_id_contract", "legacy-empty"),
        },
        "warnings_match": True,
        "decision_signal_match": True,
        "decision_message_match": baseline_snapshot.get("decision")
        == candidate_snapshot.get("decision"),
        "metric_deltas": deltas,
        "report_value_max_abs": max(report_value_deltas.values(), default=0.0),
        "top20_overlap": len(baseline_top & candidate_top),
        "top20_union": len(baseline_top | candidate_top),
        "repeated_eager_report_speedup": baseline_report / candidate_report,
        "repeated_eager_full_speedup": baseline_total / candidate_total,
        "baseline_repeated_eager_full_ns": baseline_total,
        "candidate_repeated_eager_full_ns": candidate_total,
    }
    compiled = candidate.get("compiled")
    if compiled is not None:
        compiled_total = compiled["replay"]["total_ns"]["median"]
        compiled_report = compiled["replay"]["report_ns"]["median"]
        comparison["compiled_full_speedup_vs_baseline_eager"] = (
            baseline_total / compiled_total
        )
        comparison["compiled_report_speedup_vs_baseline_eager"] = (
            baseline_report / compiled_report
        )
        comparison["candidate_compiled_full_ns"] = compiled_total
    observed_max_metric_abs = max(report_value_deltas.values(), default=0.0)
    if max_metric_abs is not None and observed_max_metric_abs > max_metric_abs:
        raise AssertionError(
            f"metric drift {observed_max_metric_abs:.9g} exceeds {max_metric_abs:.9g}"
        )
    if (
        min_eager_full_speedup is not None
        and comparison["repeated_eager_full_speedup"] < min_eager_full_speedup
    ):
        raise AssertionError(
            "candidate eager path is slower than the required baseline ratio: "
            f"{comparison['repeated_eager_full_speedup']:.6f} < "
            f"{min_eager_full_speedup:.6f}"
        )
    if min_compiled_full_speedup is not None:
        compiled_speedup = comparison.get("compiled_full_speedup_vs_baseline_eager")
        if compiled_speedup is None:
            raise AssertionError("compiled speedup gate requested without compiled results")
        if compiled_speedup < min_compiled_full_speedup:
            raise AssertionError(
                "candidate compiled path is slower than the required baseline ratio: "
                f"{compiled_speedup:.6f} < {min_compiled_full_speedup:.6f}"
            )
    return comparison


def aggregate_results(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("aggregate requires at least one input")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = results[0]
    metric_names = reference["work"]["metrics"]
    for result in results[1:]:
        for key in ("dataset", "backend", "cache_size"):
            if result[key] != reference[key]:
                raise AssertionError(f"aggregate input differs at {key}")
        if _normalized_work(result) != _normalized_work(reference):
            raise AssertionError("aggregate input differs at work")
        if result["provenance"] != reference["provenance"]:
            raise AssertionError("aggregate provenance differs")
        if (
            result["snapshot"].get("candidate_identity_contract")
            != reference["snapshot"].get("candidate_identity_contract")
        ):
            raise AssertionError("aggregate candidate identity contracts differ")
        if (
            result["snapshot"]["candidate_identity_sha256"]
            != reference["snapshot"]["candidate_identity_sha256"]
        ):
            raise AssertionError("aggregate candidate identities differ")
        _snapshot_max_abs_deltas(
            reference["snapshot"],
            result["snapshot"],
            tuple(metric_names),
        )

    snapshot_process_max_abs = {name: 0.0 for name in metric_names}
    snapshots = [result["snapshot"]["scores"] for result in results]
    if any(len(snapshot) != len(snapshots[0]) for snapshot in snapshots[1:]):
        raise AssertionError("aggregate metric snapshot lengths differ")
    for row_index, reference_row in enumerate(snapshots[0]):
        process_rows = [snapshot[row_index] for snapshot in snapshots]
        if any(
            (row["combo"], row["family"], row["candidate_id"])
            != (
                reference_row["combo"],
                reference_row["family"],
                reference_row["candidate_id"],
            )
            for row in process_rows[1:]
        ):
            raise AssertionError("aggregate candidate report order differs")
        for metric_index, metric_name in enumerate(metric_names):
            values = [float(row["metrics"][metric_index]) for row in process_rows]
            snapshot_process_max_abs[metric_name] = max(
                snapshot_process_max_abs[metric_name], max(values) - min(values)
            )
    if max(snapshot_process_max_abs.values(), default=0.0) > 1e-5:
        raise AssertionError("aggregate metric snapshots differ beyond the 1e-5 guard")

    aggregate = copy.deepcopy(reference)
    candidate_pairs = int(reference["work"]["candidate_sample_pairs"])
    eager_samples = [
        sample for result in results for sample in result["repeated_eager"]["samples"]
    ]
    aggregate["repeated_eager"] = _summarize_samples(eager_samples, candidate_pairs)
    aggregate["first_eager_samples"] = [result["first_eager"] for result in results]
    aggregate["aggregate_inputs"] = [str(path.resolve()) for path in paths]
    aggregate["aggregate_processes"] = len(paths)
    aggregate["snapshot_process_max_abs"] = snapshot_process_max_abs
    if all("repeated_snapshot_max_abs" in result for result in results):
        aggregate["repeated_snapshot_max_abs"] = {
            metric_name: max(
                result["repeated_snapshot_max_abs"][metric_name] for result in results
            )
            for metric_name in metric_names
        }
    aggregate["memory_processes"] = [result["memory"] for result in results]
    aggregate["memory_max"] = {
        phase: {
            key: max(
                result["memory"][phase][key]
                for result in results
                if phase in result["memory"] and key in result["memory"][phase]
            )
            for key in sorted(
                {
                    key
                    for result in results
                    if phase in result["memory"]
                    for key in result["memory"][phase]
                }
            )
        }
        for phase in sorted({phase for result in results for phase in result["memory"]})
    }

    compiled_results = [result.get("compiled") for result in results]
    if any(compiled is not None for compiled in compiled_results):
        if not all(compiled is not None for compiled in compiled_results):
            raise AssertionError("compiled coverage differs across aggregate inputs")
        compiled_values = [
            compiled for compiled in compiled_results if compiled is not None
        ]
        compiled_samples = [
            sample
            for compiled in compiled_values
            for sample in compiled["replay"]["samples"]
        ]
        aggregate["compiled"]["replay"] = _summarize_samples(
            compiled_samples, candidate_pairs
        )
        aggregate["compiled"]["compile_samples_ns"] = [
            int(compiled["compile_ns"]) for compiled in compiled_values
        ]
        aggregate["compiled"]["compile_ns"] = int(
            statistics.median(aggregate["compiled"]["compile_samples_ns"])
        )
        aggregate["compiled"]["compile_candidate_sample_gevals_per_second"] = (
            candidate_pairs / aggregate["compiled"]["compile_ns"]
        )
        if all("metric_max_abs_vs_eager" in compiled for compiled in compiled_values):
            aggregate["compiled"]["metric_max_abs_vs_eager"] = {
                metric_name: max(
                    compiled["metric_max_abs_vs_eager"][metric_name]
                    for compiled in compiled_values
                )
                for metric_name in metric_names
            }
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--dataset", type=Path, required=True)
    worker.add_argument(
        "--backend", choices=("core", "cuda", "rocm", "metal"), required=True
    )
    worker.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    worker.add_argument("--input-format", choices=("numpy", "lists"), default="numpy")
    worker.add_argument("--max-arity", type=int, default=3)
    worker.add_argument("--max-combinations-per-arity", type=int, default=100_000)
    worker.add_argument("--top-features-for-higher-k", type=int)
    worker.add_argument("--num-repeats", type=int, default=1)
    worker.add_argument("--permutation-tests", type=int, default=0)
    worker.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    worker.add_argument("--repeats", type=int, default=7)
    worker.add_argument("--pause-seconds", type=float, default=0.05)
    worker.add_argument("--cache-size", type=int)
    worker.add_argument("--mi-approximate", action="store_true")
    worker.add_argument("--include-compiled", action="store_true")
    worker.add_argument("--expect-candidate-identity-sha256")
    worker.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--max-metric-abs", type=float)
    compare.add_argument("--min-eager-full-speedup", type=float)
    compare.add_argument("--min-compiled-full-speedup", type=float)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--inputs", nargs="+", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "prepare":
        manifest = prepare_datasets(arguments.output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if arguments.command == "compare":
        comparison = compare_results(
            arguments.baseline,
            arguments.candidate,
            max_metric_abs=arguments.max_metric_abs,
            min_eager_full_speedup=arguments.min_eager_full_speedup,
            min_compiled_full_speedup=arguments.min_compiled_full_speedup,
        )
        rendered = json.dumps(comparison, indent=2, sort_keys=True)
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return
    if arguments.command == "aggregate":
        aggregate = aggregate_results(arguments.inputs)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
        )
        _print_worker_summary(aggregate)
        return
    result = run_worker(arguments)
    _print_worker_summary(result)


if __name__ == "__main__":
    main()
