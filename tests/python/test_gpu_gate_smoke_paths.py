"""Developer smoke selection only: no Cargo, GPU or fixture executable runs."""

import importlib.util
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "smoke_path_gate", ROOT / "tests/release_measure/v1_architecture_gate.py"
)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
SMOKES = ("GAFIME_CUDA_ABI_SMOKE", "GAFIME_CUDA_RT_ABI_SMOKE", "GAFIME_ROCM_ABI_SMOKE")
PAYLOADS = ("GAFIME_CUDA_V1_LIB", "GAFIME_CUDA_RT_V1_LIB", "GAFIME_ROCM_V1_LIB")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX developer GPU gate; non-POSIX fails closed")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    paths = {}
    for key in (*SMOKES, *PAYLOADS):
        path = private / key
        path.write_text("benign fixture, never executed\n")
        path.chmod(0o700)
        monkeypatch.setenv(key, str(path))
        paths[key] = path
    calls = []
    monkeypatch.setattr(GATE.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    return paths, calls


def test_private_explicit_paths_preserve_order_and_environment(setup):
    paths, calls = setup
    GATE.run_cargo(True)
    assert [call[0][0] for call in calls[:3]] == [[str(paths[key])] for key in SMOKES]
    assert calls[3][0][0] == ["cargo", "test", "--workspace", "--", "--test-threads=1"]
    assert calls[1][1]["env"]["GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP"] == "1"
    assert calls[0][1]["env"]["GAFIME_CUDA_V1_LIB"] == str(paths[PAYLOADS[0]])


@pytest.mark.parametrize("key", SMOKES)
def test_missing_explicit_smoke_never_uses_shared_defaults(setup, monkeypatch, key):
    _, calls = setup
    monkeypatch.delenv(key)
    with pytest.raises(AssertionError, match="missing configured"):
        GATE.run_cargo(True)
    assert not calls


@pytest.mark.parametrize("key", SMOKES)
@pytest.mark.parametrize("unsafe", ("parent", "file", "nonexec", "foreign_owner"))
def test_all_smoke_selectors_fail_closed(setup, monkeypatch, key, unsafe):
    paths, calls = setup
    path = paths[key]
    if unsafe == "parent":
        path.parent.chmod(0o777)
    elif unsafe == "file":
        path.chmod(0o722)
    elif unsafe == "nonexec":
        path.chmod(0o600)
    else:
        original = Path.lstat

        def foreign(self, *args, **kwargs):
            info = original(self, *args, **kwargs)
            if self == path:
                return SimpleNamespace(
                    st_mode=info.st_mode, st_uid=os.geteuid() + 10000
                )
            return info

        # Same-UID emulation of an ownership failure, not a cross-UID exploit.
        monkeypatch.setattr(Path, "lstat", foreign)
    with pytest.raises(AssertionError, match="GPU smoke"):
        GATE.run_cargo(True)
    assert not calls


def test_symlink_executes_only_checked_private_target(setup, monkeypatch):
    paths, calls = setup
    alias = paths[SMOKES[0]].parent / "alias"
    alias.symlink_to(paths[SMOKES[0]])
    monkeypatch.setenv(SMOKES[0], str(alias))
    GATE.run_cargo(True)
    assert calls[0][0][0] == [str(paths[SMOKES[0]])]


def test_root_sticky_ancestor_allowed_but_not_direct_shared_parent(setup, monkeypatch):
    paths, _ = setup
    path = paths[SMOKES[0]]
    shared = path.parent.parent
    original = Path.lstat

    def sticky(self, *args, **kwargs):
        if self == shared:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR | stat.S_ISVTX | 0o777, st_uid=0
            )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", sticky)
    assert GATE.trusted_smoke_path("fixture", str(path)) == path
    # The same mode is forbidden if it is the immediate executable parent.
    paths[SMOKES[0]].parent.chmod(0o777)
    with pytest.raises(AssertionError, match="writable GPU smoke parent"):
        GATE.trusted_smoke_path("fixture", str(path))


def test_non_gpu_does_not_read_smoke_paths(setup, monkeypatch):
    _, calls = setup
    for key in (*SMOKES, *PAYLOADS):
        monkeypatch.delenv(key)
    GATE.run_cargo(False)
    assert len(calls) == 1 and calls[0][0][0][0] == "cargo"


def test_non_posix_selection_fails_closed(setup, monkeypatch):
    paths, calls = setup
    monkeypatch.setattr(GATE.os, "name", "nt")
    with pytest.raises(AssertionError, match="POSIX ownership"):
        GATE.trusted_smoke_path("fixture", str(paths[SMOKES[0]]))
    assert not calls
