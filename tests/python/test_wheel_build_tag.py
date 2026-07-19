from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import runpy
from types import ModuleType
import zipfile

from packaging.utils import parse_wheel_filename
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / ".github" / "scripts" / "retag_wheel_build.py"
ARTIFACT_GATE_PATH = (
    ROOT / "tests" / "release_measure" / "artifact_01_release_composition.py"
)


def _load_retag_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("retag_wheel_build", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retag_wheel_build = _load_retag_module()
read_release_wheel = runpy.run_path(str(ARTIFACT_GATE_PATH))["_read_wheel"]


def _record_row(name: str, data: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return [name, f"sha256={digest.decode('ascii')}", str(len(data))]


def _write_wheel(
    tmp_path: Path, filename: str = "demo_pkg-1.0-py3-none-any.whl"
) -> Path:
    wheel = tmp_path / filename
    dist_info = "demo_pkg-1.0.dist-info"
    entries = {
        "demo_pkg/__init__.py": b'__version__ = "1.0"\n',
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.0\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: wheel-build-tag-test\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
    }
    record_name = f"{dist_info}/RECORD"
    record_buffer = io.StringIO()
    writer = csv.writer(record_buffer, lineterminator="\n")
    for name, data in entries.items():
        writer.writerow(_record_row(name, data))
    writer.writerow([record_name, "", ""])

    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        archive.writestr(record_name, record_buffer.getvalue().encode("utf-8"))
    return wheel


@pytest.mark.parametrize(
    ("build_tag", "expected_build"),
    (
        ("1", (1, "")),
        ("1abc", (1, "abc")),
        ("12_cuda13_3", (12, "_cuda13_3")),
        ("0A_Z9", (0, "A_Z9")),
    ),
)
def test_retag_wheel_accepts_conservative_build_tags(
    tmp_path: Path,
    build_tag: str,
    expected_build: tuple[int, str],
) -> None:
    source = _write_wheel(tmp_path)
    source_bytes = source.read_bytes()

    output = retag_wheel_build.retag_wheel(
        source,
        build_tag,
        remove_original=False,
    )

    assert output.name == f"demo_pkg-1.0-{build_tag}-py3-none-any.whl"
    assert parse_wheel_filename(output.name)[2] == expected_build
    assert source.read_bytes() == source_bytes
    with zipfile.ZipFile(output) as archive:
        wheel_metadata = archive.read("demo_pkg-1.0.dist-info/WHEEL").decode("utf-8")
        assert wheel_metadata.count(f"Build: {build_tag}\n") == 1
        record = archive.read("demo_pkg-1.0.dist-info/RECORD").decode("utf-8")
        assert "demo_pkg-1.0.dist-info/WHEEL,sha256=" in record
        assert record.endswith("demo_pkg-1.0.dist-info/RECORD,,\n")


def test_release_gate_rejects_non_numeric_wheel_build_tag(tmp_path: Path) -> None:
    source = _write_wheel(tmp_path)
    invalid = source.with_name("demo_pkg-1.0-abc-py3-none-any.whl")
    source.rename(invalid)

    with pytest.raises(AssertionError, match="invalid wheel filename"):
        read_release_wheel(invalid)


@pytest.mark.parametrize(
    "build_tag",
    (
        "abc",
        "1-hyphen",
        "1 whitespace",
        "",
        "1nonascii\N{LATIN SMALL LETTER E WITH ACUTE}",
    ),
)
def test_retag_wheel_rejects_invalid_build_tags_without_touching_source(
    tmp_path: Path,
    build_tag: str,
) -> None:
    source = _write_wheel(tmp_path)
    source_bytes = source.read_bytes()

    with pytest.raises(ValueError, match="invalid wheel build tag"):
        retag_wheel_build.retag_wheel(source, build_tag, remove_original=True)

    assert source.read_bytes() == source_bytes
    assert [path.name for path in tmp_path.iterdir()] == [source.name]


def test_retag_wheel_validates_source_before_replacing_output(tmp_path: Path) -> None:
    source = tmp_path / "demo_pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("demo_pkg-1.0.dist-info/RECORD", "")
    source_bytes = source.read_bytes()
    output = tmp_path / "demo_pkg-1.0-1-py3-none-any.whl"
    output.write_bytes(b"existing output")

    with pytest.raises(RuntimeError, match=r"does not contain a \.dist-info/WHEEL"):
        retag_wheel_build.retag_wheel(source, "1", remove_original=True)

    assert source.read_bytes() == source_bytes
    assert output.read_bytes() == b"existing output"
