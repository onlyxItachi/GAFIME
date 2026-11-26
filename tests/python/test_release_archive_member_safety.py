from __future__ import annotations

import io
from pathlib import Path
import runpy
import stat
import sys
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


def _write_sdist(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
    *,
    pax_format: bool = False,
) -> Path:
    path = tmp_path / "gafime-1.0.0.tar.gz"
    with tarfile.open(
        path,
        "w:gz",
        format=tarfile.PAX_FORMAT if pax_format else tarfile.DEFAULT_FORMAT,
    ) as archive:
        for info, data in members:
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return path


def _write_wheel(
    tmp_path: Path,
    members: list[tuple[zipfile.ZipInfo, bytes]],
    *,
    platform: str = "manylinux_2_28_x86_64",
) -> Path:
    path = tmp_path / f"gafime-1.0.0-cp311-cp311-{platform}.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("gafime-1.0.0.dist-info/METADATA", METADATA)
            archive.writestr(
                "gafime-1.0.0.dist-info/WHEEL",
                WHEEL.replace(b"manylinux_2_28_x86_64", platform.encode("ascii")),
            )
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


def test_release_gate_accepts_pax_metadata_through_bounded_stream(
    tmp_path: Path,
) -> None:
    package = _tar_file("gafime-1.0.0/gafime/__init__.py", b"")
    package[0].pax_headers = {"comment": "release metadata"}
    sdist = _write_sdist(
        tmp_path,
        [
            _tar_special("gafime-1.0.0", tarfile.DIRTYPE),
            _tar_file("gafime-1.0.0/PKG-INFO", METADATA),
            package,
        ],
        pax_format=True,
    )

    assert "gafime/__init__.py" in read_release_sdist(sdist).members


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


@pytest.mark.parametrize(
    "alias",
    (
        "gafime-1.0.0.data/purelib/gafime/__init__.py",
        "gafime-1.0.0.data/platlib/gafime/__init__.py",
        "GAFIME/__init__.py",
        "gafime/__init__.py.",
        "gafime/__init__.py ",
    ),
)
def test_release_gate_rejects_wheel_install_destination_aliases(
    tmp_path: Path, alias: str
) -> None:
    wheel = _write_wheel(
        tmp_path,
        [
            (zipfile.ZipInfo("gafime/__init__.py"), b"reviewed\n"),
            (zipfile.ZipInfo(alias), b"different\n"),
        ],
    )

    with pytest.raises(AssertionError, match="same install destination"):
        read_release_wheel(wheel)


@pytest.mark.parametrize("reverse", (False, True))
def test_release_gate_rejects_wheel_file_parent_destination_alias(
    tmp_path: Path, reverse: bool
) -> None:
    members = [
        (zipfile.ZipInfo("gafime/native"), b"file\n"),
        (zipfile.ZipInfo("gafime/native/detail.py"), b"child\n"),
    ]
    if reverse:
        members.reverse()

    with pytest.raises(
        AssertionError, match="file/parent install-destination conflict"
    ):
        read_release_wheel(_write_wheel(tmp_path, members))


@pytest.mark.parametrize("platform", ("win_amd64", "win_arm64"))
def test_release_gate_accepts_canonical_windows_wheel_paths(
    tmp_path: Path, platform: str
) -> None:
    wheel = _write_wheel(
        tmp_path,
        [
            (zipfile.ZipInfo("gafime/__init__.py"), b""),
            (zipfile.ZipInfo("gafime/compile/flags.py"), b""),
        ],
        platform=platform,
    )
    assert "gafime/compile/flags.py" in read_release_wheel(wheel).members


@pytest.mark.parametrize(
    "name",
    (
        "gafime/extra.py",
        "gafime/EXAMPL~1/extra.py",
        "gafime/alternate/extra.py",
        "gafime-1.0.0.data/purelib/gafime/compile/flags.py",
        "GAFIME/compile/flags.py",
    ),
)
def test_release_gate_rejects_unapproved_windows_wheel_paths(
    tmp_path: Path, name: str
) -> None:
    wheel = _write_wheel(tmp_path, [(zipfile.ZipInfo(name), b"")], platform="win_amd64")
    with pytest.raises(AssertionError, match="unapproved Windows wheel destination"):
        read_release_wheel(wheel)


@pytest.mark.skipif(sys.platform != "win32", reason="requires the Windows filesystem")
@pytest.mark.parametrize(
    "filenames",
    (
        (
            "gafime-1.0.0-cp311-cp311-win_amd64.whl",
            "gafime_cuda-1.0.0-cp311-cp311-win_amd64.whl",
        ),
        ("gafime-1.0.0-cp311-cp311-win_arm64.whl",),
    ),
)
def test_approved_wheel_paths_are_distinct_on_windows(
    tmp_path: Path, filenames: tuple[str, ...]
) -> None:
    files = []
    directories = set()
    for filename in filenames:
        wheel_files, wheel_dirs = ARTIFACT_GATE["_windows_wheel_layout"](Path(filename))
        files.extend(wheel_files)
        directories.update(wheel_dirs)
    for name in sorted(directories, key=lambda name: (name.count("/"), name)):
        (tmp_path / name).mkdir()
    for name in sorted(files):
        with (tmp_path / name).open("xb") as destination:
            destination.write(name.encode("ascii"))
    for name in files:
        assert (tmp_path / name).read_bytes() == name.encode("ascii")


def test_release_gate_accepts_noncolliding_purelib_relocation(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path,
        [
            (zipfile.ZipInfo("gafime/__init__.py"), b"package\n"),
            (
                zipfile.ZipInfo("gafime-1.0.0.data/purelib/companion.py"),
                b"helper\n",
            ),
        ],
    )

    artifact = read_release_wheel(wheel)

    assert "companion.py" in artifact.members
    assert dict(artifact.member_sources)["companion.py"] == (
        "gafime-1.0.0.data/purelib/companion.py"
    )


def test_release_gate_rejects_ambiguous_data_scheme_casing(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path,
        [
            (
                zipfile.ZipInfo("gafime-1.0.0.DATA/purelib/companion.py"),
                b"helper\n",
            ),
        ],
    )

    with pytest.raises(AssertionError, match="ambiguous wheel .data casing"):
        read_release_wheel(wheel)


def test_release_gate_rejects_unicode_equivalent_install_destinations(
    tmp_path: Path,
) -> None:
    wheel = _write_wheel(
        tmp_path,
        [
            (zipfile.ZipInfo("gafime/caf\u00e9.py"), b"composed\n"),
            (zipfile.ZipInfo("gafime/cafe\u0301.py"), b"decomposed\n"),
        ],
    )

    with pytest.raises(AssertionError, match="same install destination"):
        read_release_wheel(wheel)


def test_release_gate_rejects_empty_portable_destination_component(
    tmp_path: Path,
) -> None:
    wheel = _write_wheel(
        tmp_path,
        [(zipfile.ZipInfo("gafime/ . /module.py"), b"safe\n")],
    )

    with pytest.raises(AssertionError, match="empty portable install destination"):
        read_release_wheel(wheel)


def test_release_gate_bounds_member_metadata_before_content_read(
    tmp_path: Path,
) -> None:
    wheel = _write_wheel(
        tmp_path,
        [(zipfile.ZipInfo("gafime/__init__.py"), b"safe\n")],
    )

    with zipfile.ZipFile(wheel) as archive:
        info = archive.getinfo("gafime/__init__.py")
        info.file_size = ARTIFACT_GATE["ARCHIVE_MAX_MEMBER_BYTES"] + 1
        with pytest.raises(AssertionError, match="expanded size"):
            ARTIFACT_GATE["_validated_wheel_members"](wheel, archive)


def test_release_gate_scans_normal_member_with_bounded_reads() -> None:
    chunk = ARTIFACT_GATE["ARCHIVE_SCAN_CHUNK_BYTES"]

    class BoundedStream:
        def __init__(self, data: bytes) -> None:
            self.data = data
            self.offset = 0

        def read(self, size: int) -> bytes:
            assert 0 < size <= chunk
            result = self.data[self.offset : self.offset + size]
            self.offset += len(result)
            return result

    nested, findings = ARTIFACT_GATE["_scan_archive_member"](
        Path("normal.whl"),
        "gafime/module.py",
        BoundedStream(b"safe source\n" * (chunk // 3 + 3)),
    )

    assert nested is False
    assert findings == ()


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
