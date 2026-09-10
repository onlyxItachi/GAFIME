"""Bounded installed-public-API lifecycle samples; not a throughput claim.

Run outside the checkout with a release-built installed wheel. Native payloads
must be explicitly staged for GPU runs. Record host interference separately;
these short samples detect pathological overhead, not general speedups.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pyarrow as pa


def _assert_installed(module: object, source_root: Path) -> Path:
    path = Path(getattr(module, "__file__", "")).resolve(strict=True)
    if not path.is_file() or path.is_relative_to(source_root):
        raise AssertionError(
            f"module did not come from an external installation: {path}"
        )
    return path


def _load_installed(source_root: Path) -> tuple[Path, Path]:
    # PYTHONPATH and the script directory must not turn a wheel diagnostic into
    # an editable-checkout test. Reject already-imported checkout modules too.
    sys.path[:] = [
        entry
        for entry in sys.path
        if not Path(entry or Path.cwd()).resolve().is_relative_to(source_root)
    ]
    package = importlib.import_module("gafime")
    extension = importlib.import_module("gafime.gafime_py")
    package_path = _assert_installed(package, source_root)
    extension_path = _assert_installed(extension, source_root)
    assert any(
        str(extension_path).endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ), "the installed boundary must be a compiled extension"
    return package_path, extension_path


def file_record(path: str) -> dict[str, str]:
    source = Path(path).resolve(strict=True)
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def sample_run(backend: str, precision: str, repetitions: int) -> dict:
    from gafime import EngineConfig, semantic

    rng = np.random.default_rng(7396)
    dtype = np.float64 if precision == "fp64" else np.float32
    data = rng.normal(size=(512, 8)).astype(dtype)
    features = [f"x{i}" for i in range(data.shape[1])]
    keys = list(range(len(data)))
    started = perf_counter_ns()
    with semantic.TabularSession(
        data,
        feature_names=features,
        row_keys=keys,
        row_domain="training",
        provenance="fixed synthetic lifecycle fixture; not learner efficacy",
        config=EngineConfig(backend=backend, precision=precision),
    ) as session:
        creation_ns = perf_counter_ns() - started
        session.begin_round()
        started = perf_counter_ns()
        candidates = session.propose_centered_interactions(limit=16)
        fit_ns = perf_counter_ns() - started
        reference = session.source("x0")
        pearson = semantic.Evidence.reference("pearson", reference)
        channels = [
            pearson,
            semantic.Evidence.reference("rank", reference, statistic="spearman"),
            semantic.Evidence.reference(
                "dependence", reference, statistic="fixed_nmi", bins=4
            ),
        ]
        policy = semantic.SelectionPolicy(pearson, limit=len(candidates))
        warmup = session.evaluate(candidates, channels)
        expected = [
            [warmup.value(candidates[i], channel) for channel in channels]
            for i in range(len(candidates))
        ]
        samples = []
        for repetition in range(repetitions):
            session.clear_materializations()
            for mode in ("cold-materialization", "accepted-resident"):
                before = session.diagnostics
                started = perf_counter_ns()
                report = session.evaluate(candidates, channels)
                evaluation_ns = perf_counter_ns() - started
                actual = [
                    [report.value(candidates[i], channel) for channel in channels]
                    for i in range(len(candidates))
                ]
                assert actual == expected, "resident reuse changed evidence"
                after = session.diagnostics
                started = perf_counter_ns()
                accepted = session.select(report, policy)
                acceptance_ns = perf_counter_ns() - started
                assert len(accepted) == len(candidates)
                samples.append(
                    dict(
                        repetition=repetition,
                        mode=mode,
                        evaluation_ns=evaluation_ns,
                        acceptance_ns=acceptance_ns,
                        before=before,
                        after=after,
                    )
                )

        inference_keys = list(range(10000, 10128))
        inference = session.snapshot(
            rng.normal(size=(128, 8)).astype(dtype),
            feature_names=features,
            row_keys=inference_keys,
            row_domain="inference",
            provenance="unlabeled fresh rows",
        )
        started = perf_counter_ns()
        delivered = session.transform(accepted, inference)
        inference_ns = perf_counter_ns() - started
        array = pa.array(delivered)
        assert len(array) == 128 and len(delivered.feature_names) == len(accepted)
        assert delivered.row_keys == inference_keys
        assert [field.name for field in array.type] == [
            "__gafime_row_key__",
            *delivered.feature_names,
        ]
        assert array.field("__gafime_row_key__").to_pylist() == inference_keys
        feature_type = pa.float64() if precision == "fp64" else pa.float32()
        assert all(
            array.field(name).type == feature_type for name in delivered.feature_names
        ), "Arrow feature types must retain the profile's pointwise domain"
        output_hash = hashlib.sha256(
            b"".join(
                array.field(name).to_numpy().tobytes()
                for name in delivered.feature_names
            )
        ).hexdigest()
        return dict(
            backend=report.backend,
            precision=report.precision,
            rows=512,
            candidates=len(candidates),
            channels=3,
            creation_ns=creation_ns,
            fit_ns=fit_ns,
            samples=samples,
            inference_ns=inference_ns,
            inference_values_sha256=output_hash,
            capabilities=session.capabilities,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("core", "cuda", "rocm", "metal"), required=True
    )
    parser.add_argument("--precision", choices=("fp32", "mixed", "fp64"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10:
        parser.error("repetitions must be in [1,10]")
    root = Path(__file__).resolve().parents[2]
    tracked_dirty = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=root, text=True
    ).strip()
    if tracked_dirty:
        parser.error("record samples only from a committed tracked source tree")
    package_path, extension_path = _load_installed(root)
    payload_names = {
        "cuda": ("GAFIME_CUDA_V1_LIB",),
        "rocm": ("GAFIME_ROCM_V1_LIB",),
        "metal": ("GAFIME_METAL_V1_LIB", "GAFIME_METAL_V1_METALLIB"),
    }
    payloads = {}
    for name in payload_names.get(args.backend, ()):
        if not os.environ.get(name):
            parser.error(f"{name} must identify the exact staged payload")
        payloads[name] = file_record(os.environ[name])
    output = dict(
        schema="gafime.semantic-discovery-sanity.v1",
        scope="bounded unisolated public lifecycle diagnostic",
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        platform=platform.platform(),
        python=platform.python_version(),
        rayon_num_threads=os.environ.get("RAYON_NUM_THREADS", "default"),
        package=file_record(str(package_path)),
        extension=file_record(str(extension_path)),
        payloads=payloads,
        result=sample_run(args.backend, args.precision, args.repetitions),
    )
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
