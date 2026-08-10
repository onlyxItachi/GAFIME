from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


_SCRIPT = Path(__file__).with_name("canonical_abi_lifecycle_evidence.py")
_SPEC = importlib.util.spec_from_file_location("gafime_canonical_abi_evidence", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
canonical = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(canonical)


@pytest.mark.parametrize(
    ("surface", "schema"),
    (
        ("numeric-route-v2", canonical.RESULT_SCHEMA),
        ("precision-typed-v1.1", canonical.TYPED_RESULT_SCHEMA),
    ),
)
def test_produce_separates_product_and_harness_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    schema: str,
) -> None:
    product_root = tmp_path / "product"
    harness_root = tmp_path / "harness"
    source_relative = Path(canonical.ABI_SURFACES[surface]["consumer_source"])
    consumer_source = harness_root / source_relative
    consumer_source.parent.mkdir(parents=True)
    consumer_source.write_text("/* external harness */\n", encoding="utf-8")
    product_root.mkdir()
    consumer = tmp_path / "consumer"
    consumer.write_bytes(b"consumer-binary")
    payload = tmp_path / "libgafime_cuda.so"
    payload.write_bytes(b"payload-bytes")
    wheel = tmp_path / "gafime_cuda.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gafime_cuda/libgafime_cuda.so", payload.read_bytes())

    product_commit = "a" * 40
    harness_commit = "b" * 40

    def fake_git(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return product_commit if root == product_root else harness_commit
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return source_relative.as_posix()
        raise AssertionError((root, arguments))

    marker = {
        "schema": schema,
        "status": "pass",
        "abi_surface": surface,
        "backend_kind": 2,
        "route_count": 3,
        "operations": list(canonical.ABI_SURFACES[surface]["operations"]),
    }
    monkeypatch.setattr(canonical, "_git_output", fake_git)
    monkeypatch.setattr(
        canonical.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(marker) + "\n",
            stderr="",
        ),
    )

    evidence = canonical.produce(
        backend="cuda",
        consumer=consumer,
        payload=payload,
        wheel=wheel,
        source_root=product_root,
        abi_surface=surface,
        harness_source_root=harness_root,
    )

    assert evidence["source_commit"] == product_commit
    assert evidence["product_source_commit"] == product_commit
    assert evidence["harness_source_commit"] == harness_commit
    assert evidence["abi_surface"] == surface
    assert evidence["route_count"] == 3
    assert evidence["provenance"]["consumer_source"]["sha256"] == evidence[
        "provenance"
    ]["harness_source"]["sha256"]


def test_produce_rejects_consumer_source_outside_harness_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product_root = tmp_path / "product"
    harness_root = tmp_path / "harness"
    product_root.mkdir()
    harness_root.mkdir()
    consumer_source = tmp_path / "outside.c"
    consumer_source.write_text("/* outside */\n", encoding="utf-8")
    consumer = tmp_path / "consumer"
    consumer.write_bytes(b"consumer-binary")
    payload = tmp_path / "libgafime_cuda.so"
    payload.write_bytes(b"payload-bytes")
    wheel = tmp_path / "gafime_cuda.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gafime_cuda/libgafime_cuda.so", payload.read_bytes())

    def fake_git(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return ("a" if root == product_root else "b") * 40
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        raise AssertionError((root, arguments))

    marker = {
        "schema": canonical.RESULT_SCHEMA,
        "status": "pass",
        "abi_surface": "numeric-route-v2",
        "backend_kind": 2,
        "route_count": 3,
        "operations": list(canonical.REQUIRED_OPERATIONS),
    }
    monkeypatch.setattr(canonical, "_git_output", fake_git)
    monkeypatch.setattr(
        canonical.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(marker) + "\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="inside the harness tree"):
        canonical.produce(
            backend="cuda",
            consumer=consumer,
            payload=payload,
            wheel=wheel,
            source_root=product_root,
            abi_surface="numeric-route-v2",
            consumer_source=consumer_source,
            harness_source_root=harness_root,
        )
