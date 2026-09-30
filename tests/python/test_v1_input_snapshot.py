"""Owned input staging keeps resident identity and execution on the same bytes."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(ROOT / "python") not in sys.path
):
    sys.path.insert(0, str(ROOT / "python"))

from gafime import ComputeBudget, EngineConfig, GafimeEngine  # noqa: E402
from gafime import v1_adapter  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_resident_cache(monkeypatch):
    monkeypatch.setenv("GAFIME_V1_ANALYZE_CACHE_SIZE", "2")
    v1_adapter._clear_analyze_cache_for_tests()
    yield
    v1_adapter._clear_analyze_cache_for_tests()


def _metrics(report):
    return {tuple(item.combo): item.metrics for item in report.interactions}


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("mutation", ["features", "target"])
@pytest.mark.parametrize("input_kind", ["array", "memmap"])
def test_resident_hash_and_execution_survive_external_write(
    monkeypatch, tmp_path, precision, mutation, input_kind
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("gafime.gafime_py")
    dtype = "<f8" if precision == "fp64" else "<f4"
    original_target = np.arange(1, 17, dtype=dtype)
    original_features = np.column_stack((original_target, original_target % 3))
    features, target = original_features.copy(), original_target.copy()
    if input_kind == "memmap":
        features = np.memmap(
            tmp_path / "features.bin",
            mode="w+",
            dtype=dtype,
            shape=original_features.shape,
        )
        target = np.memmap(
            tmp_path / "target.bin",
            mode="w+",
            dtype=dtype,
            shape=original_target.shape,
        )
        features[:] = original_features
        target[:] = original_target
    config = EngineConfig(
        backend="core",
        precision=precision,
        metric_names=("pearson", "r2"),
        permutation_tests=0,
        num_repeats=1,
        random_seed=7,
        budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )
    engine = GafimeEngine(config)
    eager_config = replace(config, budget=replace(config.budget, keep_in_vram=False))
    expected = _metrics(
        GafimeEngine(eager_config).analyze(original_features, original_target)
    )
    if mutation == "target":
        engine.analyze(original_features, -original_target)

    start_write = threading.Event()
    write_done = threading.Event()
    writer_errors = []

    def writer():
        try:
            if not start_write.wait(timeout=5):
                raise AssertionError("analysis did not reach its digest")
            if mutation == "features":
                features[:, 0] = 0
            else:
                target[:] = 0
        except BaseException as exc:
            writer_errors.append(exc)
        finally:
            write_done.set()

    digest = v1_adapter._numeric_buffer_digest
    digest_calls = 0

    def hash_then_allow_write(count, data, profile):
        nonlocal digest_calls
        result = digest(count, data, profile)
        digest_calls += 1
        # Schedule a real external writer after the relevant digest, but before
        # compile/update consumes the data. No sleeps or timing luck are needed.
        if digest_calls == (1 if mutation == "features" else 2):
            start_write.set()
            assert write_done.wait(timeout=5)
        return result

    monkeypatch.setattr(v1_adapter, "_numeric_buffer_digest", hash_then_allow_write)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        first = _metrics(engine.analyze(features, target))
        entry = next(iter(v1_adapter._current_analyze_cache().values()))
        later = _metrics(engine.analyze(original_features, original_target))
        assert next(iter(v1_adapter._current_analyze_cache().values())) is entry
        assert len(v1_adapter._current_analyze_cache()) == 1
        assert not writer_errors
        # The next call is stable; it must not reuse content poisoned by the
        # earlier overlap merely because the original fingerprint matches.
        assert later == expected
        assert first == expected
    finally:
        start_write.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "big-endian"])
def test_resident_snapshot_is_readonly_private_and_byte_stable(precision, layout):
    np = pytest.importorskip("numpy")
    dtype = "<f8" if precision == "fp64" else "<f4"
    features = np.asarray(
        [[0.0, -0.0], [float("nan"), float("inf")], [-float("inf"), 1.25]],
        dtype=dtype,
    )
    target = np.asarray([1.0, -0.0, float("nan")], dtype=dtype)
    if layout == "fortran":
        features = np.asfortranarray(features)
    elif layout == "strided":
        features = features[:, ::-1]
        target = target[::-1]
    elif layout == "big-endian":
        big_dtype = ">f8" if precision == "fp64" else ">f4"
        features = features.astype(big_dtype)
        target = target.astype(big_dtype)

    expected_features = features.astype(dtype).tobytes(order="C")
    expected_target = target.astype(dtype).tobytes(order="C")
    coerced = v1_adapter._coerce_row_major_f32_for_cache(
        features, target, None, precision=precision, include_digests=True
    )
    assert not np.shares_memory(coerced.features, features)
    assert not np.shares_memory(coerced.target, target)
    assert not coerced.features.flags.writeable
    assert not coerced.target.flags.writeable
    assert features.flags.writeable and target.flags.writeable
    with pytest.raises(ValueError):
        coerced.features.setflags(write=True)
    with pytest.raises(ValueError):
        coerced.target.setflags(write=True)

    features[:] = 7
    target[:] = 9
    assert coerced.feature_bytes() == expected_features
    assert coerced.target_bytes() == expected_target
    assert coerced.feature_bytes() is coerced.feature_bytes()
    assert coerced.target_bytes() is coerced.target_bytes()
    assert coerced.feature_digest == v1_adapter._numeric_buffer_digest(
        6, expected_features, precision
    )
    assert coerced.target_digest == v1_adapter._numeric_buffer_digest(
        3, expected_target, precision
    )


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
@pytest.mark.parametrize("source_dtype", ["bool", "int64", "float32", "float64"])
def test_resident_snapshot_preserves_selected_dtype_conversion(precision, source_dtype):
    np = pytest.importorskip("numpy")
    dtype = "<f8" if precision == "fp64" else "<f4"
    features = np.asarray([[0, 1], [2, 3]], dtype=source_dtype)
    target = np.asarray([0, 1], dtype=source_dtype)
    coerced = v1_adapter._coerce_row_major_f32_for_cache(
        features, target, None, precision=precision, include_digests=True
    )
    assert coerced.feature_bytes() == features.astype(dtype).tobytes()
    assert coerced.target_bytes() == target.astype(dtype).tobytes()
    assert not coerced.features.flags.writeable
    assert not coerced.target.flags.writeable


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_resident_snapshot_validates_owned_source_range(precision):
    np = pytest.importorskip("numpy")
    storage_dtype = np.float64 if precision == "fp64" else np.float32
    value = np.longdouble(np.finfo(storage_dtype).max) * np.longdouble(2)
    if not np.isfinite(value):
        pytest.skip("longdouble has no wider finite range on this platform")
    storage_range = "fp64" if precision == "fp64" else "fp32"
    with pytest.raises(ValueError, match=f"outside {storage_range} range"):
        v1_adapter._coerce_row_major_f32_for_cache(
            np.asarray([[value]], dtype=np.longdouble),
            np.asarray([0.0], dtype=np.longdouble),
            None,
            precision=precision,
            include_digests=True,
        )
