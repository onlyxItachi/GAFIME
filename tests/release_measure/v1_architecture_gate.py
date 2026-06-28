#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import types


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_RUNTIME_MODULES = {
    "gafime.backends",
    "gafime.backends.core_backend",
    "gafime.backends.native_cuda_backend",
    "gafime.backends.native_rocm_backend",
    "gafime.compile.scenario",
    "gafime.compile.sessions",
    "gafime.engine",
    "gafime.native_data",
}
FORBIDDEN_RUNTIME_STRINGS = (
    "GAFIME_V1_ENGINE",
    "GAFIME_USE_LEGACY_ENGINE",
    "gafime_core",
    "native_cuda_backend",
    "native_rocm_backend",
    "compile.sessions",
    "compile.scenario",
)
FORBIDDEN_LOCAL_RUNTIME_PATHS = (
    "gafime.egg-info",
    "gafime/backends",
    "gafime/metrics",
    "gafime/native_data.py",
    "gafime/optimizer",
    "gafime/planning",
    "gafime/preprocessors",
    "gafime/utils",
    "gafime/validation",
    "gafime_core/build",
    "python/gafime/backends",
    "python/gafime/metrics",
    "python/gafime/native_data.py",
    "python/gafime/planning",
    "python/gafime/validation",
)
FORBIDDEN_LOCAL_RUNTIME_GLOBS = (
    "gafime/_native*.so",
    "gafime/gafime_core*.so",
    "gafime/gafime_cpu*.so",
    "python/gafime/_native*.so",
    "python/gafime/gafime_core*.so",
    "python/gafime/gafime_cpu*.so",
)


class FakeRecord:
    def __init__(self, combo, metrics, candidate_id):
        self.combo = combo
        self.metrics = metrics
        self.candidate_id = candidate_id


class FakeReport:
    rows = 4
    cols = 2
    max_arity = 1
    metric_ids = [1, 4]

    def __init__(self, length=2):
        self.length = int(length)

    def __len__(self):
        return self.length

    def record(self, index):
        return FakeRecord(self.combo(index), self.metric_values(index), self.candidate_id(index))

    def combo(self, index):
        return [int(index % 2)]

    def metric_values(self, index):
        return [1.0 if index % 2 == 0 else -1.0, 1.0]

    def candidate_id(self, index):
        return int(index)

    def ranked_indices(self, *, metric_index=None, descending=True, limit=None):
        count = self.length if limit is None else min(int(limit), self.length)
        return list(range(count))

    def records(self):
        raise AssertionError("v1 gate forbids normal-path Python report list materialization")


class FakeArtifact:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __init__(self, calls, length=2):
        self.calls = calls
        self.closed = False
        self.length = length

    def analyze(self):
        return FakeReport(self.length)

    def close(self):
        self.closed = True


def install_fake_boundary(length=2):
    module = types.ModuleType("_v1_gate_fake_boundary")
    calls = []

    def compile_continuous(config, features, target, *, rows, cols):
        calls.append(
            {
                "config": config,
                "features": features,
                "target": target,
                "rows": rows,
                "cols": cols,
            }
        )
        return FakeArtifact(calls, length=length)

    module.compile_continuous = compile_continuous
    module.BOUNDARY_NAME = "fake-gafime-py"
    module.calls = calls
    sys.modules[module.__name__] = module
    os.environ["GAFIME_V1_BOUNDARY_MODULE"] = module.__name__
    return module


def check_packaging() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["build-system"]["build-backend"] == "maturin"
    assert pyproject["tool"]["maturin"]["manifest-path"] == "crates/gafime-py/Cargo.toml"
    assert pyproject["tool"]["maturin"]["module-name"] == "gafime.gafime_py"
    assert pyproject["tool"]["maturin"]["python-source"] == "python"
    build_requires = " ".join(pyproject["build-system"].get("requires", []))
    assert "setuptools" not in build_requires
    assert "pybind11" not in build_requires
    assert "cmake" not in build_requires

    setup_text = (ROOT / "setup.py").read_text()
    assert "no longer builds runtime artifacts" in setup_text
    manifest_text = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include gafime_core" not in manifest_text
    assert "RustCompileSrc" not in manifest_text

    packaged_files = {path.relative_to(ROOT / "python").as_posix() for path in (ROOT / "python").rglob("*") if path.is_file()}
    forbidden_paths = {
        "gafime/backends",
        "gafime/native_data.py",
        "gafime/compile/sessions.py",
        "gafime/compile/scenario.py",
    }
    for forbidden in forbidden_paths:
        assert all(not item.startswith(forbidden) for item in packaged_files), forbidden

    package_text = "\n".join(path.read_text() for path in (ROOT / "python" / "gafime").rglob("*.py"))
    for forbidden in FORBIDDEN_RUNTIME_STRINGS:
        assert forbidden not in package_text, forbidden


def check_no_local_legacy_runtime_artifacts() -> None:
    offenders = []
    for relative in FORBIDDEN_LOCAL_RUNTIME_PATHS:
        path = ROOT / relative
        if path.exists():
            offenders.append(relative)
    for pattern in FORBIDDEN_LOCAL_RUNTIME_GLOBS:
        offenders.extend(path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern))
    assert not offenders, f"legacy local runtime artifacts are present: {sorted(offenders)}"


def check_runtime_surface() -> None:
    sys.path.insert(0, str(ROOT))
    fake = install_fake_boundary()
    os.environ["GAFIME_USE_LEGACY_ENGINE"] = "1"
    try:
        import gafime

        before_modules = set(sys.modules)
        cfg = gafime.EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        report = gafime.GafimeEngine(cfg).analyze(
            [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]],
            [1.0, 2.0, 3.0, 4.0],
            ["a", "b"],
        )
        families = {family.name: family for family in gafime.available_families()}
    finally:
        os.environ.pop("GAFIME_USE_LEGACY_ENGINE", None)

    assert fake.calls, "public analyze did not use the v1 boundary"
    assert fake.calls[0]["features"] == [1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]
    assert report.backend.name == "v1-rust-cpu"
    assert report.interactions.is_native_backed
    assert report.interactions[0].metrics == {"pearson": 1.0, "r2": 1.0}
    assert report.interactions.top_k(1)[0].combo == (0,)
    assert families["continuous"].supported
    assert not families["decision_path"].supported
    assert not families["time_series"].supported
    assert all(not family.python_candidate_loop for family in families.values())
    loaded_forbidden = FORBIDDEN_RUNTIME_MODULES.intersection(set(sys.modules) - before_modules)
    assert not loaded_forbidden, sorted(loaded_forbidden)


def check_no_source_opt_in_or_fallback() -> None:
    source_text = "\n".join(path.read_text() for path in (ROOT / "gafime").rglob("*.py"))
    assert "GAFIME_V1_ENGINE" not in source_text
    assert "GAFIME_USE_LEGACY_ENGINE" not in source_text


def check_report_scale_view() -> None:
    sys.path.insert(0, str(ROOT))
    fake = install_fake_boundary(length=10_000_000)
    import gafime

    cfg = gafime.EngineConfig(
        metric_names=("pearson", "r2"),
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=10_000_000),
        permutation_tests=0,
        num_repeats=1,
    )
    report = gafime.GafimeEngine(cfg).analyze([[1.0, 0.0], [2.0, 1.0]], [1.0, 2.0], ["a", "b"])
    assert fake.calls
    assert len(report.interactions) == 10_000_000
    top = report.interactions.top_k(3, metric_name="pearson")
    assert len(top) == 3
    assert [item.combo for item in top] == [(0,), (1,), (0,)]


def run_cargo(include_gpu: bool) -> None:
    if include_gpu:
        required = (
            "/tmp/libgafime_cuda_v1.so",
            "/tmp/libgafime_rocm_v1.so",
            "/tmp/cuda_v1_abi_smoke",
            "/tmp/rocm_v1_abi_smoke",
        )
        missing = [path for path in required if not Path(path).exists()]
        if missing:
            raise AssertionError(f"missing optional GPU payloads or smokes: {missing}")
    subprocess.run(["cargo", "test", "--workspace"], cwd=ROOT, check=True)
    if include_gpu:
        subprocess.run(["cargo", "test", "-p", "gafime-gpu-sys"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--skip-cargo", action="store_true")
    args = parser.parse_args()

    check_no_local_legacy_runtime_artifacts()
    check_packaging()
    check_no_source_opt_in_or_fallback()
    check_runtime_surface()
    check_report_scale_view()
    if not args.skip_cargo:
        run_cargo(args.include_gpu)
    print("v1 architecture gate passed")


if __name__ == "__main__":
    main()
