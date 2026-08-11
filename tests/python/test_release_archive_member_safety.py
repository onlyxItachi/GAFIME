from __future__ import annotations

import io
from pathlib import Path
import runpy
import stat
import tarfile
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_GATE_PATH = (
    ROOT / "tests" / "release_measure" / "artifact_01_release_composition.py"
)
ARTIFACT_GATE = runpy.run_path(str(ARTIFACT_GATE_PATH))
read_release_sdist = ARTIFACT_GATE["_read_sdist"]
read_release_wheel = ARTIFACT_GATE["_read_wheel"]

METADATA = b"Metadata-Version: 2.4\nName: gafime\nVersion: 1.0.0\n\n"
WHEEL = (
    b"Wheel-Version: 1.0\n"
    b"Generator: release-archive-safety-test\n"
    b"Root-Is-Purelib: true\n"
    b"Tag: cp311-cp311-manylinux_2_28_x86_64\n\n"
)


def _tar_file(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _tar_special(
    name: str, member_type: bytes, *, linkname: str = ""
) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.linkname = linkname
    return info, b""


def _write_sdist(tmp_path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> Path:
    path = tmp_path / "gafime-1.0.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for info, data in members:
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return path


def _write_wheel(tmp_path: Path, members: list[tuple[zipfile.ZipInfo, bytes]]) -> Path:
    path = tmp_path / "gafime-1.0.0-cp311-cp311-manylinux_2_28_x86_64.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("gafime-1.0.0.dist-info/METADATA", METADATA)
            archive.writestr("gafime-1.0.0.dist-info/WHEEL", WHEEL)
            for info, data in members:
                archive.writestr(info, data)
    return path


def test_release_gate_accepts_canonical_regular_archive_members(tmp_path: Path) -> None:
    directory = tarfile.TarInfo("gafime-1.0.0/gafime")
    directory.type = tarfile.DIRTYPE
    sdist = _write_sdist(
        tmp_path,
        [
            _tar_special("gafime-1.0.0", tarfile.DIRTYPE),
            _tar_file("gafime-1.0.0/PKG-INFO", METADATA),
            (directory, b""),
            _tar_file("gafime-1.0.0/gafime/__init__.py", b""),
        ],
    )
    wheel_member = zipfile.ZipInfo("gafime/__init__.py")
    wheel = _write_wheel(tmp_path, [(wheel_member, b"")])

    assert read_release_sdist(sdist).members == {
        "PKG-INFO",
        "gafime",
        "gafime/__init__.py",
    }
    assert "gafime/__init__.py" in read_release_wheel(wheel).members


@pytest.mark.parametrize(
    ("members", "message"),
    (
        ([_tar_file("../PKG-INFO", METADATA)], "non-canonical"),
        ([_tar_file("/PKG-INFO", METADATA)], "non-canonical"),
        ([_tar_file("C:/PKG-INFO", METADATA)], "non-canonical"),
        (
            [
                _tar_file("root/PKG-INFO", METADATA),
                _tar_file(r"root/sub\..\..\escape.py", b"pass\n"),
            ],
            "non-canonical",
        ),
        (
            [
                _tar_file("root/PKG-INFO", METADATA),
                _tar_file("root/./escape.py", b"pass\n"),
            ],
            "non-canonical",
        ),
        (
            [
                _tar_file("root/PKG-INFO", METADATA),
                _tar_file("root//escape.py", b"pass\n"),
            ],
            "non-canonical",
        ),
        (
            [
                _tar_file("root/PKG-INFO", METADATA),
                _tar_file("root/setup.py", b"reviewed\n"),
                _tar_file("root/setup.py", b"installed\n"),
            ],
            "duplicate",
        ),
        (
            [
                _tar_file("root/PKG-INFO", METADATA),
                _tar_special(
                    "root/module.py", tarfile.SYMTYPE, linkname="../../outside"
                ),
            ],
            "regular files or directories",
        ),
        (
            [
                _tar_file("root/PKG-INFO", METADATA),
                _tar_special("root/pipe", tarfile.FIFOTYPE),
            ],
            "regular files or directories",
        ),
    ),
)
def test_release_gate_rejects_unsafe_sdist_members(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
    message: str,
) -> None:
    sdist = _write_sdist(tmp_path, members)

    with pytest.raises(AssertionError, match=message):
        read_release_sdist(sdist)


@pytest.mark.parametrize(
    ("member_name", "message"),
    (
        ("../escape.py", "non-canonical"),
        ("/escape.py", "non-canonical"),
        ("C:/escape.py", "non-canonical"),
        (r"gafime\..\escape.py", "non-canonical"),
        ("gafime/./escape.py", "non-canonical"),
        ("gafime//escape.py", "non-canonical"),
    ),
)
def test_release_gate_rejects_unsafe_wheel_member_paths(
    tmp_path: Path, member_name: str, message: str
) -> None:
    wheel = _write_wheel(tmp_path, [(zipfile.ZipInfo(member_name), b"pass\n")])

    with pytest.raises(AssertionError, match=message):
        read_release_wheel(wheel)


def test_release_gate_rejects_duplicate_wheel_members(tmp_path: Path) -> None:
    name = "gafime/module.py"
    wheel = _write_wheel(
        tmp_path,
        [
            (zipfile.ZipInfo(name), b"reviewed\n"),
            (zipfile.ZipInfo(name), b"installed\n"),
        ],
    )

    with pytest.raises(AssertionError, match="duplicate"):
        read_release_wheel(wheel)


def test_release_gate_rejects_nul_truncated_wheel_member(tmp_path: Path) -> None:
    placeholder = b"gafime/module.pyXhidden.py"
    unsafe_name = b"gafime/module.py\x00hidden.py"
    wheel = _write_wheel(
        tmp_path,
        [(zipfile.ZipInfo(placeholder.decode("ascii")), b"pass\n")],
    )
    raw = wheel.read_bytes()
    assert raw.count(placeholder) == 2
    wheel.write_bytes(raw.replace(placeholder, unsafe_name))

    with pytest.raises(AssertionError, match="non-canonical"):
        read_release_wheel(wheel)


def test_release_gate_rejects_sdist_root_mismatched_with_filename(
    tmp_path: Path,
) -> None:
    sdist = _write_sdist(
        tmp_path,
        [_tar_file("different-1.0.0/PKG-INFO", METADATA)],
    )

    with pytest.raises(AssertionError, match="canonical archive filename stem"):
        read_release_sdist(sdist)


@pytest.mark.parametrize("file_type", (stat.S_IFLNK, stat.S_IFIFO))
def test_release_gate_rejects_special_wheel_members(
    tmp_path: Path, file_type: int
) -> None:
    info = zipfile.ZipInfo("gafime/special")
    info.create_system = 3
    info.external_attr = (file_type | 0o777) << 16
    wheel = _write_wheel(tmp_path, [(info, b"../../outside")])

    with pytest.raises(AssertionError, match="regular files or directories"):
        read_release_wheel(wheel)
