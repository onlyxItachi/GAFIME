from __future__ import annotations

import math
from pathlib import Path
import struct
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

    def update_target_buffer(self, target):
        values = list(struct.unpack(f"<{len(target) // 4}f", target))
        self.events.append(("update_target_buffer", values))
        self.updated_targets.append(values)

    def close(self):
        self.closed = True


def _fake_boundary(
    *, nested_rows: bool = False, buffers: bool = False, handle_factory=_FakeHandle
):
    boundary = types.SimpleNamespace(
        BOUNDARY_NAME="adapter-compatibility-fake",
        analyze_calls=[],
        analyze_row_calls=[],
        compile_calls=[],
        compile_row_calls=[],
        analyze_buffer_calls=[],
        compile_buffer_calls=[],
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
    if buffers:

        def analyze_continuous_buffers(config, features, target, *, rows, cols):
            boundary.analyze_buffer_calls.append(
                (config, bytes(features), bytes(target), rows, cols)
            )
            return _FakeReport()

        def compile_continuous_buffers(config, features, target, *, rows, cols):
            handle = handle_factory()
            boundary.compile_buffer_calls.append(
                (config, bytes(features), bytes(target), rows, cols)
            )
            boundary.handles.append(handle)
            return handle

        boundary.analyze_continuous_buffers = analyze_continuous_buffers
        boundary.compile_continuous_buffers = compile_continuous_buffers
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


def test_engine_config_preserves_legacy_positional_mi_bins_slot():
    config = EngineConfig(
        ComputeBudget(),
        ("mutual_info",),
        3,
        25,
        7,
        0.10,
        0.05,
        32,
    )

    assert config.mi_bins == 32
    assert config.significance_top_n == 50
    assert config.mi_approximate is False


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


@pytest.mark.parametrize("path", ["one-shot", "resident-cache", "explicit-compile"])
def test_current_boundary_uses_contiguous_f32_bytes_without_float_lists(monkeypatch, path):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    keep_in_vram = path == "resident-cache"
    engine = GafimeEngine(_config(keep_in_vram=keep_in_vram))
    features = [[1.0, float("nan")], [float("inf"), -2.0]]
    target = [0.5, float("-inf")]

    if path == "explicit-compile":
        artifact = engine.compile(features, target, ["a", "b"])
        try:
            artifact.analyze()
            artifact.update_target([1.5, 2.5])
        finally:
            artifact.close()
        call = boundary.compile_buffer_calls[0]
        assert boundary.handles[0].events[-1] == (
            "update_target_buffer",
            [1.5, 2.5],
        )
    elif path == "resident-cache":
        engine.analyze(features, target, ["a", "b"])
        engine.analyze(features, [1.5, 2.5], ["a", "b"])
        call = boundary.compile_buffer_calls[0]
        assert boundary.handles[0].events[-2:] == [
            ("update_target_buffer", [1.5, 2.5]),
            ("analyze", None),
        ]
    else:
        engine.analyze(features, target, ["a", "b"])
        call = boundary.analyze_buffer_calls[0]

    _, feature_bytes, target_bytes, rows, cols = call
    decoded_features = struct.unpack("<4f", feature_bytes)
    decoded_target = struct.unpack("<2f", target_bytes)
    assert (rows, cols) == (2, 2)
    assert decoded_features[0] == 1.0
    assert math.isnan(decoded_features[1])
    assert decoded_features[2] == float("inf")
    assert decoded_target[0] == 0.5
    assert decoded_target[1] == float("-inf")
    assert boundary.analyze_calls == []
    assert boundary.compile_calls == []


@pytest.mark.parametrize(
    ("path", "expected_digest_calls"),
    [("one-shot", 0), ("resident-cache", 2), ("explicit-compile", 0)],
)
def test_only_resident_cache_computes_content_digests(
    monkeypatch, path, expected_digest_calls
):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    original_digest = v1_adapter._f32_buffer_digest
    digest_calls = []

    def counted_digest(count, data):
        digest_calls.append((count, len(data)))
        return original_digest(count, data)

    monkeypatch.setattr(v1_adapter, "_f32_buffer_digest", counted_digest)
    engine = GafimeEngine(_config(keep_in_vram=path == "resident-cache"))
    features = [[1.0, 2.0], [3.0, 4.0]]
    target = [0.0, 1.0]

    if path == "explicit-compile":
        artifact = engine.compile(features, target, ["a", "b"])
        artifact.close()
    else:
        engine.analyze(features, target, ["a", "b"])

    assert len(digest_calls) == expected_digest_calls


def test_zero_cache_capacity_evicts_existing_resident_artifacts(monkeypatch):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "1")
    engine = GafimeEngine(_config(keep_in_vram=True))
    features = [[1.0], [2.0]]
    target = [0.0, 1.0]

    engine.analyze(features, target, ["a"])
    resident_handle = boundary.handles[0]
    assert not resident_handle.closed

    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "0")
    engine.analyze(features, target, ["a"])

    assert resident_handle.closed
    assert not v1_adapter._ANALYZE_CACHE
    assert len(boundary.analyze_buffer_calls) == 1


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


@pytest.mark.parametrize("path", ["one-shot", "resident-cache", "explicit-compile"])
def test_legacy_budget_validation_warnings_propagate_on_every_path(monkeypatch, path):
    boundary = _fake_boundary(nested_rows=path != "resident-cache")
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    keep_in_vram = path == "resident-cache"
    budget = ComputeBudget(
        max_comb_size=3,
        max_combinations_per_k=8,
        top_features_for_higher_k=0,
        keep_in_vram=keep_in_vram,
    )
    engine = GafimeEngine(_config(keep_in_vram=keep_in_vram, budget=budget))
    features = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
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
        "top_features_for_higher_k < 1; higher-order combos will be empty.",
        "max_comb_size exceeds feature count; will cap to n_features.",
    ]


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
