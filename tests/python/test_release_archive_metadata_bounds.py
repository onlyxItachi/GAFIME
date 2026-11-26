from __future__ import annotations

import io
from pathlib import Path
import runpy
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_GATE_PATH = (
    ROOT / "tests" / "release_measure" / "artifact_01_release_composition.py"
)
ARTIFACT_GATE = runpy.run_path(str(ARTIFACT_GATE_PATH))
open_bounded_sdist = ARTIFACT_GATE["_open_bounded_sdist"]
open_bounded_wheel = ARTIFACT_GATE["_open_bounded_wheel"]
GATE_GLOBALS = ARTIFACT_GATE["_assert_wheel_metadata_admission"].__globals__


def _write_standard_pax_sdist(path: Path) -> str:
    long_component = "release_metadata_" + "a" * 120
    member_name = f"gafime-1.0.0/{long_component}/module.py"
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        directory = tarfile.TarInfo("gafime-1.0.0")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        member = tarfile.TarInfo(member_name)
        member.size = 0
        member.pax_headers = {"comment": "ordinary release metadata"}
        archive.addfile(member, io.BytesIO())
    return member_name


def test_bounded_wheel_admission_accepts_standard_deflate_and_comment(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "gafime-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = b"reviewed standard ZIP release metadata"
        archive.writestr("gafime/__init__.py", b"__version__ = '1.0.0'\n")

    with open_bounded_wheel(wheel) as archive:
        assert archive.namelist() == ["gafime/__init__.py"]
        assert archive.read("gafime/__init__.py") == b"__version__ = '1.0.0'\n"


def test_bounded_sdist_admission_accepts_standard_pax_path(tmp_path: Path) -> None:
    sdist = tmp_path / "gafime-1.0.0.tar.gz"
    member_name = _write_standard_pax_sdist(sdist)

    with open_bounded_sdist(sdist) as archive:
        assert member_name in {member.name for member in archive}


def test_wheel_filename_budget_rejects_before_zipfile_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "gafime-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("gafime/__init__.py", b"")

    constructed = False
    original_zipfile = zipfile.ZipFile

    def tracking_zipfile(*args: object, **kwargs: object) -> zipfile.ZipFile:
        nonlocal constructed
        constructed = True
        return original_zipfile(*args, **kwargs)

    monkeypatch.setitem(GATE_GLOBALS, "ARCHIVE_MAX_ZIP_FILENAME_BYTES", 4)
    monkeypatch.setattr(zipfile, "ZipFile", tracking_zipfile)

    with pytest.raises(
        AssertionError, match="oversized ZIP central-directory metadata"
    ):
        with open_bounded_wheel(wheel):
            pass
    assert not constructed


def test_pax_padded_budget_rejects_before_tarfile_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdist = tmp_path / "gafime-1.0.0.tar.gz"
    _write_standard_pax_sdist(sdist)

    constructed = False
    original_tarfile_open = tarfile.open

    def tracking_tarfile_open(*args: object, **kwargs: object) -> tarfile.TarFile:
        nonlocal constructed
        constructed = True
        return original_tarfile_open(*args, **kwargs)

    monkeypatch.setitem(
        GATE_GLOBALS,
        "ARCHIVE_MAX_TAR_SPECIAL_METADATA_TOTAL_BYTES",
        tarfile.BLOCKSIZE - 1,
    )
    monkeypatch.setattr(tarfile, "open", tracking_tarfile_open)

    with pytest.raises(AssertionError, match="excessive TAR extension metadata"):
        with open_bounded_sdist(sdist):
            pass
    assert not constructed


def test_production_archive_parsers_have_pre_admission_gates() -> None:
    source = ARTIFACT_GATE_PATH.read_text(encoding="utf-8")
    wheel_gateway = source[source.index("def _open_bounded_wheel") :]
    sdist_gateway = source[source.index("def _open_bounded_sdist") :]

    assert source.count("with zipfile.ZipFile(") == 1
    assert source.count("with tarfile.open(") == 1
    assert source.index("zipfile._EndRecData(stream)") < source.index(
        "def _open_bounded_wheel"
    )
    assert wheel_gateway.index("_assert_wheel_metadata_admission(path, raw)") < (
        wheel_gateway.index("with zipfile.ZipFile(raw)")
    )
    assert sdist_gateway.index("_assert_tar_metadata_admission(") < (
        sdist_gateway.index('with tarfile.open(fileobj=bounded_expanded, mode="r:")')
    )
    assert "ARCHIVE_MAX_ZIP_FILENAME_BYTES = 512" in source
    assert "tarfile.GNUTYPE_SPARSE" in source
    assert 'keyword != b"size"' in source
    assert "has nonzero TAR extension metadata padding" in source
