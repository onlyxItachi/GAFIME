from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402
from gafime import v1_adapter  # noqa: E402
from gafime.errors import V1UnsupportedError  # noqa: E402


class _FakeReport:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __len__(self):
        return 0


class _FakeHandle:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __init__(self):
        self.closed = False
        self.events = []
        self.updated_targets = []

    def reseed(self, seed):
        self.events.append(("reseed", seed))

    def analyze(self):
        self.events.append(("analyze", None))
        return _FakeReport()

    def update_target(self, target):
        values = list(target)
        self.events.append(("update_target", values))
        self.updated_targets.append(values)

    def close(self):
        self.closed = True


def _fake_boundary(*, nested_rows: bool = False, handle_factory=_FakeHandle):
    boundary = types.SimpleNamespace(
        BOUNDARY_NAME="adapter-compatibility-fake",
        analyze_calls=[],
        analyze_row_calls=[],
        compile_calls=[],
        compile_row_calls=[],
        handles=[],
    )

    def analyze_continuous(config, features, target, *, rows, cols):
        boundary.analyze_calls.append(
            {
                "config": config,
                "features": list(features),
                "target": list(target),
                "rows": rows,
                "cols": cols,
            }
        )
        return _FakeReport()

    def compile_continuous(config, features, target, *, rows, cols):
        handle = handle_factory()
        boundary.compile_calls.append(
            {
                "config": config,
                "features": list(features),
                "target": list(target),
                "rows": rows,
                "cols": cols,
            }
        )
        boundary.handles.append(handle)
        return handle

    boundary.analyze_continuous = analyze_continuous
    boundary.compile_continuous = compile_continuous
    if nested_rows:

        def analyze_continuous_rows(config, features, target):
            boundary.analyze_row_calls.append((config, features, target))
            return _FakeReport()

        def compile_continuous_rows(config, features, target):
            handle = handle_factory()
            boundary.compile_row_calls.append((config, features, target))
            boundary.handles.append(handle)
            return handle

        boundary.analyze_continuous_rows = analyze_continuous_rows
        boundary.compile_continuous_rows = compile_continuous_rows
    return boundary


def _config(
    *,
    keep_in_vram: bool,
    random_seed: int | None = 7,
    budget: ComputeBudget | None = None,
) -> EngineConfig:
    return EngineConfig(
        backend="core",
        metric_names=("pearson",),
        permutation_tests=0,
        num_repeats=1,
        random_seed=random_seed,
        budget=budget
        or ComputeBudget(
            max_comb_size=1,
            max_combinations_per_k=8,
            keep_in_vram=keep_in_vram,
        ),
    )


@pytest.fixture(autouse=True)
def _clear_resident_cache(monkeypatch):
    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    v1_adapter._clear_analyze_cache_for_tests()
    yield
    v1_adapter._clear_analyze_cache_for_tests()


def test_nested_list_ingest_accepts_nonfinite_and_preserves_validation(monkeypatch):
    boundary = _fake_boundary(nested_rows=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    engine = GafimeEngine(_config(keep_in_vram=False))
    features = [[float("nan"), float("inf")], [float("-inf"), 1.0]]
    target = [float("inf"), float("nan")]

    engine.analyze(features, target, ["a", "b"])

    assert len(boundary.analyze_row_calls) == 1
    assert boundary.analyze_row_calls[0][1] is features
    assert boundary.analyze_row_calls[0][2] is target

    with pytest.raises(ValueError, match="outside fp32 range"):
        engine.analyze([[1.0e39]], [0.0], ["a"])
    with pytest.raises(ValueError, match="non-numeric"):
        engine.analyze([[object()]], [0.0], ["a"])
    with pytest.raises(ValueError, match="row 1 has length 1; expected 2"):
        engine.analyze([[1.0, 2.0], [3.0]], [0.0, 1.0], ["a", "b"])
    with pytest.raises(ValueError, match="same number of samples"):
        engine.analyze([[1.0], [2.0]], [0.0], ["a"])

    assert len(boundary.analyze_row_calls) == 1


@pytest.mark.parametrize("input_kind", ["list", "numpy"])
def test_nonfinite_resident_cache_and_target_digest_paths(monkeypatch, input_kind):
    boundary = _fake_boundary()
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    if input_kind == "numpy":
        np = pytest.importorskip("numpy")
        features = np.asarray(
            [[float("nan"), float("inf")], [float("-inf"), 1.0]],
            dtype=np.float64,
        )
        target = np.asarray([float("inf"), float("nan")], dtype=np.float64)
        replacement = np.asarray([float("-inf"), float("nan")], dtype=np.float64)
        overflow_features = np.asarray([[1.0e39], [0.0]], dtype=np.float64)
        overflow_target = np.asarray([1.0e39, 0.0], dtype=np.float64)
    else:
        features = [[float("nan"), float("inf")], [float("-inf"), 1.0]]
        target = [float("inf"), float("nan")]
        replacement = [float("-inf"), float("nan")]
        overflow_features = [[1.0e39], [0.0]]
        overflow_target = [1.0e39, 0.0]

    engine = GafimeEngine(_config(keep_in_vram=True))
    engine.analyze(features, target, ["a", "b"])
    engine.analyze(features, target, ["a", "b"])

    assert len(boundary.compile_calls) == 1
    assert math.isnan(boundary.compile_calls[0]["features"][0])
    assert boundary.compile_calls[0]["features"][1:3] == [float("inf"), float("-inf")]
    assert boundary.handles[0].updated_targets == []

    engine.analyze(features, replacement, ["a", "b"])

    assert len(boundary.compile_calls) == 1
    updated = boundary.handles[0].updated_targets
    assert len(updated) == 1
    assert updated[0][0] == float("-inf")
    assert math.isnan(updated[0][1])

    with pytest.raises(ValueError, match="outside fp32 range"):
        engine.analyze(overflow_features, [0.0, 1.0], ["a"])
    with pytest.raises(ValueError, match="outside fp32 range"):
        engine.analyze(features, overflow_target, ["a", "b"])
    assert len(updated) == 1


@pytest.mark.parametrize("path", ["one-shot", "resident-cache", "explicit-compile"])
def test_legacy_continuous_cap_warnings_propagate_on_every_path(monkeypatch, path):
    boundary = _fake_boundary(nested_rows=path != "resident-cache")
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    keep_in_vram = path == "resident-cache"
    budget = ComputeBudget(
        max_comb_size=4,
        max_combinations_per_k=5,
        top_features_for_higher_k=6,
        keep_in_vram=keep_in_vram,
    )
    engine = GafimeEngine(_config(keep_in_vram=keep_in_vram, budget=budget))
    features = [[float(row + col) for col in range(6)] for row in range(3)]
    target = [0.0, 1.0, 2.0]

    if path == "explicit-compile":
        artifact = engine.compile(features, target)
        try:
            report = artifact.analyze()
        finally:
            artifact.close()
    else:
        report = engine.analyze(features, target)

    assert report.warnings == [
        "Unary combinations capped by max_combinations_per_k.",
        "k=2 combinations capped by max_combinations_per_k.",
        "k=3 combinations capped by max_combinations_per_k.",
        "k=4 combinations capped by max_combinations_per_k.",
    ]
    if path == "one-shot":
        assert len(boundary.analyze_row_calls) == 1
    elif path == "resident-cache":
        assert len(boundary.compile_calls) == 1
    else:
        assert len(boundary.compile_row_calls) == 1


def test_none_seed_explicit_artifact_reseeds_before_every_analyze(monkeypatch):
    boundary = _fake_boundary()
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    seeds = iter([(1 << 255) + 1, (1 << 255) + 2, (1 << 255) + 3])
    monkeypatch.setattr(v1_adapter, "_fresh_random_seed", lambda: next(seeds))
    engine = GafimeEngine(_config(keep_in_vram=False, random_seed=None))

    artifact = engine.compile([[1.0], [2.0]], [0.0, 1.0], ["a"])
    try:
        artifact.analyze()
        artifact.analyze()
    finally:
        artifact.close()

    assert boundary.compile_calls[0]["config"]["random_seed"] == (1 << 255) + 1
    assert boundary.handles[0].events == [
        ("reseed", (1 << 255) + 2),
        ("analyze", None),
        ("reseed", (1 << 255) + 3),
        ("analyze", None),
    ]


def test_none_seed_explicit_artifact_requires_native_reseed(monkeypatch):
    def handle_without_reseed():
        handle = _FakeHandle()
        handle.reseed = None
        return handle

    boundary = _fake_boundary(handle_factory=handle_without_reseed)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    monkeypatch.setattr(v1_adapter, "_fresh_random_seed", lambda: 123)
    engine = GafimeEngine(_config(keep_in_vram=False, random_seed=None))
    artifact = engine.compile([[1.0], [2.0]], [0.0, 1.0], ["a"])

    try:
        with pytest.raises(
            V1UnsupportedError, match=r"random_seed=None.*reseed\(seed\)"
        ):
            artifact.analyze()
        assert boundary.handles[0].events == []
    finally:
        artifact.close()
