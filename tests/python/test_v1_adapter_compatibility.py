from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
import struct
import sys
import threading
import types

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime import CompileFlags, ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402
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


class _ClosingAnalyzeHandle(_FakeHandle):
    def analyze(self):
        self.closed = True
        raise RuntimeError("native analyze failed closed")


class _ClosingUpdateHandle(_FakeHandle):
    def update_target_buffer(self, target):
        self.closed = True
        raise RuntimeError("native target update failed closed")


class _ClosingReseedHandle(_FakeHandle):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    def reseed(self, seed):
        self.events.append(("reseed", seed))
        self.closed = True
        raise RuntimeError("native reseed failed closed")

    def close(self):
        self.close_calls += 1
        super().close()


class _FailingCloseHandle(_FakeHandle):
    def close(self):
        self.closed = True
        raise RuntimeError("native close failed after teardown")


class _PrimaryAndCleanupFailureHandle(_FakeHandle):
    def analyze(self):
        self.closed = True
        raise ValueError("primary analyze failure")

    def close(self):
        raise RuntimeError("cleanup close failure")


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


def test_compute_budget_preserves_first_six_positional_fields():
    budget = ComputeBudget(
        3,
        4_000,
        25,
        750,
        False,
        2_048,
        max_time_series_candidates=321,
        top_k_features_for_time_series=17,
        max_feature_candidate=99,
    )

    assert budget.max_comb_size == 3
    assert budget.max_combinations_per_k == 4_000
    assert budget.top_features_for_higher_k == 25
    assert budget.max_generated_features == 750
    assert budget.keep_in_vram is False
    assert budget.vram_budget_mb == 2_048
    assert budget.max_time_series_candidates == 321
    assert budget.top_k_features_for_time_series == 17
    assert budget.max_feature_candidate == 99
    assert (
        inspect.signature(ComputeBudget).parameters["max_time_series_candidates"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.parametrize(
    "ambiguous_tail",
    [
        (100_000,),
        (100_000, 50, None),
    ],
)
def test_compute_budget_rejects_ambiguous_later_positional_fields(ambiguous_tail):
    shared_prefix = (2, 5_000, 50, 0, True, 6_144)

    with pytest.raises(
        TypeError,
        match=r"argument 7 and later are ambiguous across v0\.4\.7 and v0\.5",
    ):
        ComputeBudget(*shared_prefix, *ambiguous_tail)


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


def test_engine_config_preserves_shared_v047_positional_prefix():
    config = EngineConfig(
        ComputeBudget(max_comb_size=3),
        ("pearson",),
        4,
        12,
        99,
        0.2,
        0.01,
        48,
        "cuda",
        2,
    )

    assert config.budget.max_comb_size == 3
    assert config.metric_names == ("pearson",)
    assert config.num_repeats == 4
    assert config.permutation_tests == 12
    assert config.random_seed == 99
    assert config.stability_std_threshold == 0.2
    assert config.permutation_p_threshold == 0.01
    assert config.mi_bins == 48
    assert config.backend == "cuda"
    assert config.device_id == 2


def test_engine_config_preserves_origin_main_positional_mi_approximate_layout():
    config = EngineConfig(
        ComputeBudget(max_comb_size=3),
        ("pearson",),
        4,
        12,
        99,
        0.2,
        0.01,
        48,
        True,
        "cuda",
        2,
    )

    assert config.mi_bins == 48
    assert config.mi_approximate is True
    assert config.backend == "cuda"
    assert config.device_id == 2


def test_engine_config_rejects_duplicate_positional_mi_approximate():
    with pytest.raises(TypeError, match="provided both positionally and by keyword"):
        EngineConfig(
            ComputeBudget(),
            ("pearson",),
            3,
            25,
            7,
            0.10,
            0.05,
            96,
            True,
            mi_approximate=False,
        )


def test_engine_config_rejects_legacy_discrete_positional_slot_with_migration():
    shared_prefix = (
        ComputeBudget(),
        ("pearson",),
        3,
        25,
        7,
        0.10,
        0.05,
        96,
        "core",
        0,
    )

    with pytest.raises(
        TypeError,
        match="positional argument 11 was enable_discrete_functions",
    ):
        EngineConfig(*shared_prefix, True)

    with pytest.raises(
        TypeError, match="discrete family is not part of the v1 runtime"
    ):
        EngineConfig(enable_discrete_functions=True)


def test_engine_config_migrates_disabled_legacy_discrete_option_with_warning():
    shared_prefix = (
        ComputeBudget(),
        ("pearson",),
        3,
        25,
        7,
        0.10,
        0.05,
        96,
        "core",
        0,
    )

    with pytest.warns(DeprecationWarning, match="ignored by the v1 runtime"):
        positional = EngineConfig(*shared_prefix, False)
    with pytest.warns(DeprecationWarning, match="ignored by the v1 runtime"):
        keyword = EngineConfig(enable_discrete_functions=False)

    assert positional.enable_time_series_functions is False
    assert positional.enable_decision_path_functions is False
    assert keyword.enable_time_series_functions is False
    assert keyword.enable_decision_path_functions is False


def test_current_family_switches_are_keyword_only():
    signature = inspect.signature(EngineConfig)

    assert (
        signature.parameters["enable_time_series_functions"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        signature.parameters["enable_decision_path_functions"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    config = EngineConfig(enable_time_series_functions=True)
    assert config.enable_time_series_functions is True


def test_engine_config_precision_contract_is_keyword_only_and_truthful():
    signature = inspect.signature(EngineConfig)

    assert signature.parameters["storage_dtype"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["compute_policy"].kind is inspect.Parameter.KEYWORD_ONLY
    config = EngineConfig()
    assert config.storage_dtype == "float32"
    assert config.compute_policy == "stable"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EngineConfig(storage_dtype="float64"), "no current Core, CUDA, ROCm, or Metal"),
        (EngineConfig(compute_policy="exact"), "true f64 ingest"),
        (EngineConfig(compute_policy="fast"), "high-dynamic normalization guard"),
    ],
)
def test_unsupported_precision_requests_fail_before_discovery_or_coercion(
    config, message, monkeypatch
):
    def tripwire(*_args, **_kwargs):
        raise AssertionError("precision validation ran too late")

    for name in (
        "_load_boundary_for_backend",
        "_coerce_row_major_f32",
        "_coerce_row_major_f32_for_cache",
        "_validate_arrow_target_frame",
    ):
        monkeypatch.setattr(v1_adapter, name, tripwire)

    operations = (
        lambda: v1_adapter.analyze_with_v1_boundary(config, [[1.0]], [1.0]),
        lambda: v1_adapter.analyze_time_series_with_v1_boundary(
            config, [[1.0]], [1.0]
        ),
        lambda: v1_adapter.analyze_decision_path_with_v1_boundary(
            config, [[1.0]], [1.0]
        ),
        lambda: v1_adapter.compile_with_v1_boundary(config, [[1.0]], [1.0]),
        lambda: v1_adapter.analyze_arrow_with_v1_boundary(config, None, None, []),
    )
    for operation in operations:
        with pytest.raises(V1UnsupportedError, match=message):
            operation()


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
def test_current_boundary_uses_contiguous_f32_bytes_without_float_lists(
    monkeypatch, path
):
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
    ("handle_factory", "operation"),
    [(_ClosingAnalyzeHandle, "analyze"), (_ClosingUpdateHandle, "update")],
)
def test_native_fail_closed_state_closes_python_artifact(
    monkeypatch, handle_factory, operation
):
    boundary = _fake_boundary(buffers=True, handle_factory=handle_factory)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    artifact = GafimeEngine(_config(keep_in_vram=False)).compile(
        [[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0], ["a", "b"]
    )

    with pytest.raises(RuntimeError, match="failed closed"):
        if operation == "analyze":
            artifact.analyze()
        else:
            artifact.update_target([1.0, 0.0])

    assert artifact._closed is True
    with pytest.raises(RuntimeError, match="closed"):
        artifact.analyze()


def test_none_seed_native_reseed_failure_closes_python_artifact(monkeypatch):
    boundary = _fake_boundary(handle_factory=_ClosingReseedHandle)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    seeds = iter((101, 102))
    monkeypatch.setattr(v1_adapter, "_fresh_random_seed", lambda: next(seeds))
    artifact = GafimeEngine(_config(keep_in_vram=False, random_seed=None)).compile(
        [[1.0], [2.0]], [0.0, 1.0], ["a"]
    )

    with pytest.raises(RuntimeError, match="native reseed failed closed"):
        artifact.analyze()

    handle = boundary.handles[0]
    assert boundary.compile_calls[0]["config"]["random_seed"] == 101
    assert handle.events == [("reseed", 102)]
    assert handle.close_calls == 1
    assert handle.closed is True
    assert artifact._closed is True
    with pytest.raises(RuntimeError, match="closed"):
        artifact.analyze()


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


def test_unset_seed_reuses_resident_artifact_but_reseeds_each_analysis(monkeypatch):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    seeds = iter((101, 102, 103, 104))
    monkeypatch.setattr(v1_adapter, "_fresh_random_seed", lambda: next(seeds))
    engine = GafimeEngine(_config(keep_in_vram=True, random_seed=None))
    features = [[1.0], [2.0]]
    target = [0.0, 1.0]

    engine.analyze(features, target, ["a"])
    engine.analyze(features, target, ["a"])

    assert len(boundary.compile_buffer_calls) == 1
    assert boundary.compile_buffer_calls[0][0]["random_seed"] == 101
    assert [event for event in boundary.handles[0].events if event[0] == "reseed"] == [
        ("reseed", 102),
        ("reseed", 104),
    ]


def test_resident_cache_identity_includes_selected_payload_paths(monkeypatch):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    monkeypatch.setenv("GAFIME_CUDA_V1_LIB", "/payload/standard.so")
    engine = GafimeEngine(
        EngineConfig(
            backend="cuda",
            metric_names=("pearson",),
            permutation_tests=0,
            num_repeats=1,
            budget=ComputeBudget(max_comb_size=1, keep_in_vram=True),
        )
    )
    features = [[1.0], [2.0]]
    target = [0.0, 1.0]

    engine.analyze(features, target, ["a"])
    monkeypatch.setenv("GAFIME_CUDA_V1_LIB", "/payload/rt.so")
    engine.analyze(features, target, ["a"])

    assert len(boundary.compile_buffer_calls) == 2
    assert len(v1_adapter._current_analyze_cache()) == 2


def test_resident_entries_do_not_serialize_unrelated_analyses(monkeypatch):
    barrier = threading.Barrier(2)
    should_block = threading.Event()

    class _ConcurrentHandle(_FakeHandle):
        def analyze(self):
            if should_block.is_set():
                barrier.wait(timeout=2.0)
            return super().analyze()

    boundary = _fake_boundary(buffers=True, handle_factory=_ConcurrentHandle)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    engine = GafimeEngine(_config(keep_in_vram=True))
    first_features = [[1.0], [2.0]]
    second_features = [[3.0], [4.0]]
    target = [0.0, 1.0]
    engine.analyze(first_features, target, ["a"])
    engine.analyze(second_features, target, ["a"])
    should_block.set()
    errors = []

    def run(features):
        try:
            engine.analyze(features, target, ["a"])
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(first_features,)),
        threading.Thread(target=run, args=(second_features,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(boundary.compile_buffer_calls) == 4
    assert all(handle.closed for handle in boundary.handles[2:])


def test_resident_base_exception_evicts_and_rebuilds_cache_entry(monkeypatch):
    class _NativePanic(BaseException):
        pass

    class _PanickingHandle(_FakeHandle):
        def __init__(self):
            super().__init__()
            self.analyze_calls = 0

        def analyze(self):
            self.analyze_calls += 1
            if self.analyze_calls == 2:
                raise _NativePanic("native panic")
            return super().analyze()

    boundary = _fake_boundary(buffers=True, handle_factory=_PanickingHandle)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    engine = GafimeEngine(_config(keep_in_vram=True))
    features = [[1.0], [2.0]]
    target = [0.0, 1.0]

    engine.analyze(features, target, ["a"])
    with pytest.raises(_NativePanic, match="native panic"):
        engine.analyze(features, target, ["a"])

    assert not v1_adapter._current_analyze_cache()
    assert boundary.handles[0].closed

    engine.analyze(features, target, ["a"])
    assert len(boundary.compile_buffer_calls) == 2
    assert len(v1_adapter._current_analyze_cache()) == 1


def test_explicit_native_artifact_rejects_cross_thread_use_before_boundary_call(
    monkeypatch,
):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    artifact = GafimeEngine(_config(keep_in_vram=False)).compile(
        [[1.0], [2.0]], [0.0, 1.0], ["a"]
    )
    errors = []

    def run():
        try:
            artifact.analyze()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "thread-affine" in str(errors[0])
    assert boundary.handles[0].events == []
    artifact.close()


def test_real_native_resident_artifacts_are_compiled_per_thread(monkeypatch):
    native_boundary = pytest.importorskip("gafime.gafime_py")
    compile_calls = []
    compile_calls_lock = threading.Lock()

    def compile_continuous_buffers(config, features, target, *, rows, cols):
        handle = native_boundary.compile_continuous_buffers(
            config,
            features,
            target,
            rows=rows,
            cols=cols,
        )
        with compile_calls_lock:
            compile_calls.append(threading.get_ident())
        return handle

    boundary = types.SimpleNamespace(
        BOUNDARY_NAME="real-thread-affinity-test",
        compile_continuous=native_boundary.compile_continuous,
        compile_continuous_buffers=compile_continuous_buffers,
    )
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    engine = GafimeEngine(_config(keep_in_vram=True))
    features = [[0.0, 1.0], [1.0, 3.0], [2.0, 2.0], [3.0, 4.0]]
    target = [0.0, 1.0, 1.5, 3.0]

    def snapshot():
        report = engine.analyze(features, target, ["a", "b"])
        return [(item.combo, dict(item.metrics)) for item in report.interactions]

    main_snapshot = snapshot()
    assert snapshot() == main_snapshot
    worker_snapshots = []
    worker_errors = []

    def run():
        try:
            worker_snapshots.extend((snapshot(), snapshot()))
        except BaseException as exc:  # pragma: no cover - asserted below
            worker_errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=10.0)

    assert not thread.is_alive()
    assert worker_errors == []
    assert worker_snapshots == [main_snapshot, main_snapshot]
    assert len(compile_calls) == 2
    assert len(set(compile_calls)) == 2


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
    assert not v1_adapter._current_analyze_cache()
    assert len(boundary.analyze_buffer_calls) == 1


def test_lower_positive_cache_capacity_is_enforced_on_hit(monkeypatch):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    engine = GafimeEngine(_config(keep_in_vram=True))
    target = [0.0, 1.0]
    first = [[1.0], [2.0]]
    second = [[3.0], [4.0]]

    engine.analyze(first, target, ["a"])
    engine.analyze(second, target, ["a"])
    first_handle, second_handle = boundary.handles
    assert len(v1_adapter._current_analyze_cache()) == 2

    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "1")
    engine.analyze(second, target, ["a"])

    assert first_handle.closed
    assert not second_handle.closed
    assert len(v1_adapter._current_analyze_cache()) == 1
    assert len(boundary.compile_buffer_calls) == 2


def test_compiled_artifact_restores_v05_flags_and_exports(monkeypatch):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    artifact = GafimeEngine(_config(keep_in_vram=False)).compile(
        [[1.0], [2.0]],
        [0.0, 1.0],
        ["a"],
        flags=CompileFlags(plan=False, export=True),
    )

    assert artifact.flags == CompileFlags(plan=False, export=True)
    with pytest.warns(DeprecationWarning, match="export_arrow"):
        before = artifact.exports
    assert before.backend_name == "core"
    assert before.feature_matrix_handle is boundary.handles[0]
    assert before.result_table_handle is None
    assert before.candidate_table_handle is None

    artifact.analyze()
    with pytest.warns(DeprecationWarning, match="export_arrow"):
        after = artifact.exports
    assert after.result_table_handle is not None
    artifact.close()


def test_compiled_artifact_legacy_exports_require_export_flag(monkeypatch):
    boundary = _fake_boundary(buffers=True)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    artifact = GafimeEngine(_config(keep_in_vram=False)).compile(
        [[1.0], [2.0]], [0.0, 1.0], ["a"]
    )
    try:
        with pytest.warns(DeprecationWarning, match="export_arrow"):
            with pytest.raises(
                V1UnsupportedError, match="export handles are not available"
            ):
                _ = artifact.exports
    finally:
        artifact.close()


def test_compiled_close_failure_marks_wrapper_closed(monkeypatch):
    boundary = _fake_boundary(buffers=True, handle_factory=_FailingCloseHandle)
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    artifact = GafimeEngine(_config(keep_in_vram=False)).compile(
        [[1.0], [2.0]], [0.0, 1.0], ["a"]
    )

    with pytest.raises(RuntimeError, match="native close failed"):
        artifact.close()
    with pytest.raises(RuntimeError, match="closed"):
        artifact.analyze()
    artifact.close()


def test_cleanup_failure_does_not_mask_primary_native_error(monkeypatch):
    boundary = _fake_boundary(
        buffers=True,
        handle_factory=_PrimaryAndCleanupFailureHandle,
    )
    monkeypatch.setattr(
        v1_adapter, "_load_boundary_for_backend", lambda _backend: boundary
    )
    artifact = GafimeEngine(_config(keep_in_vram=False)).compile(
        [[1.0], [2.0]], [0.0, 1.0], ["a"]
    )

    with pytest.raises(ValueError, match="primary analyze failure"):
        artifact.analyze()
    assert artifact._closed is True
    with pytest.raises(RuntimeError, match="closed"):
        artifact.analyze()


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
