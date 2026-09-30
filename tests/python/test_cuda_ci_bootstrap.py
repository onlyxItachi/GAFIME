from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
HELPER = "provision_cuda_13_3_rpms.sh"
MANIFEST = "cuda_13_3_rpms.sha256"
KEY_SHA256 = "27e46a2d43e125859fb8a62c3b75bf798aeb95fa6f7d9bf790c1167ed9a0b39c"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux CI bootstrap")


def _fixture_bytes(name: str) -> bytes:
    return f"CUDA bootstrap fixture: {name}".encode()


@pytest.fixture
def bootstrap(tmp_path: Path):
    # No command in this harness can install packages or write system paths.
    scripts = tmp_path / "checkout with spaces" / ".github" / "scripts"
    scripts.mkdir(parents=True)
    helper = (
        (SCRIPTS / HELPER)
        .read_text()
        .replace(KEY_SHA256, hashlib.sha256(_fixture_bytes("D42D0685.pub")).hexdigest())
    )
    (scripts / HELPER).write_text(helper)
    names = [line.split()[1] for line in (SCRIPTS / MANIFEST).read_text().splitlines()]
    (scripts / MANIFEST).write_text(
        "".join(
            f"{hashlib.sha256(_fixture_bytes(name)).hexdigest()}  {name}\n"
            for name in names
        )
    )
    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    mock = mock_bin / "mock-command"
    mock.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json
import os
from pathlib import Path
import sys

command = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CUDA_TEST_EVENTS"], "a") as stream:
    stream.write(json.dumps({"command": command, "args": args}) + "\n")
failure = os.environ.get("CUDA_TEST_FAIL", "")
if failure == command or (failure == "signature" and command == "rpm" and "--checksig" in args):
    sys.exit(7)
if command == "curl":
    name = args[-1].rsplit("/", 1)[-1]
    assert args[-1].startswith("https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/")
    data = f"CUDA bootstrap fixture: {name}".encode()
    if os.environ.get("CUDA_TEST_CORRUPT") == name:
        data += b"corruption"
    Path(args[args.index("--output") + 1]).write_bytes(data)
elif command == "sha256sum":
    os.execv(os.environ["CUDA_TEST_REAL_SHA256SUM"], ["sha256sum", *args])
elif command == "rpm" and "--checksig" in args:
    if failure == "unsigned":
        print("Payload SHA256 digest: OK")
    else:
        key = "deadbeef" if failure == "wrong_signer" else "d42d0685"
        print(f"Header V4 RSA/SHA512 Signature, key ID {key}: OK")
elif command not in ("rpm", "dnf", "ln"):
    raise AssertionError(command)
"""
    )
    mock.chmod(0o755)
    for command in ("curl", "sha256sum", "rpm", "dnf", "ln"):
        (mock_bin / command).symlink_to(mock)
    events = tmp_path / "events.jsonl"
    checks = tmp_path / "file-checks.txt"
    download_root = tmp_path / "downloads"
    download_root.mkdir()
    env = dict(
        os.environ, PATH=f"{mock_bin}:{os.environ['PATH']}", TMPDIR=str(download_root)
    )
    env.update(
        CUDA_TEST_EVENTS=str(events),
        CUDA_TEST_FILE_CHECKS=str(checks),
        CUDA_TEST_REAL_SHA256SUM=shutil.which("sha256sum") or "/usr/bin/sha256sum",
    )
    wrapper = r"""
test() {
    if [[ "${2:-}" == /usr/local/cuda-13.3/* ]]; then
        printf '%s\n' "$2" >> "$CUDA_TEST_FILE_CHECKS"
        [[ "$2" != "${CUDA_TEST_MISSING_COMPONENT:-}" ]]
    else
        builtin test "$@"
    fi
}
export -f test
exec bash "$1"
"""

    def run(**overrides: str):
        result = subprocess.run(
            ["bash", "-c", wrapper, "bootstrap-test", str(scripts / HELPER)],
            env=dict(env, **overrides),
            capture_output=True,
            text=True,
            timeout=30,
        )
        calls = (
            [json.loads(line) for line in events.read_text().splitlines()]
            if events.exists()
            else []
        )
        assert not list(download_root.iterdir()), (
            "bootstrap must clean its own temporary downloads"
        )
        return result, calls

    return scripts, names, checks, run


def test_bootstrap_verifies_every_package_before_explicit_local_install(bootstrap):
    _scripts, names, checks, run = bootstrap
    result, calls = run()
    assert result.returncode == 0, result.stderr
    assert len(names) == 13
    downloads = [call for call in calls if call["command"] == "curl"]
    assert {call["args"][-1].rsplit("/", 1)[-1] for call in downloads} == {
        *names,
        "D42D0685.pub",
    }
    hashes = [i for i, call in enumerate(calls) if call["command"] == "sha256sum"]
    imports = [
        i
        for i, call in enumerate(calls)
        if call["command"] == "rpm" and "--import" in call["args"]
    ]
    signatures = [
        i
        for i, call in enumerate(calls)
        if call["command"] == "rpm" and "--checksig" in call["args"]
    ]
    installs = [i for i, call in enumerate(calls) if call["command"] == "dnf"]
    assert (
        len(hashes) == 2
        and len(imports) == 1
        and len(signatures) == 13
        and len(installs) == 1
    )
    assert max(hashes) < imports[0] < min(signatures) <= max(signatures) < installs[0]
    args = calls[installs[0]]["args"]
    assert args[:4] == [
        "--disablerepo=cuda*",
        "--setopt=localpkg_gpgcheck=1",
        "install",
        "-y",
    ]
    assert len(args[4:]) == 13 and {Path(path).name for path in args[4:]} == set(names)
    assert all(Path(path).is_absolute() for path in args[4:])
    assert len(checks.read_text().splitlines()) == 12
    assert calls[-1] == {
        "command": "ln",
        "args": ["-sfn", "/usr/local/cuda-13.3", "/usr/local/cuda"],
    }


@pytest.mark.parametrize(
    "failure", ["curl", "rpm", "signature", "unsigned", "wrong_signer"]
)
def test_bootstrap_does_not_install_unverified_inputs(bootstrap, failure):
    _scripts, _names, _checks, run = bootstrap
    result, calls = run(CUDA_TEST_FAIL=failure)
    assert result.returncode != 0
    assert not any(call["command"] in ("dnf", "ln") for call in calls)


@pytest.mark.parametrize(
    "filename", ["D42D0685.pub", "cuda-nvcc-13-3-13.3.73-1.x86_64.rpm"]
)
def test_bootstrap_rejects_checksum_mismatch_before_key_import(bootstrap, filename):
    _scripts, _names, _checks, run = bootstrap
    result, calls = run(CUDA_TEST_CORRUPT=filename)
    assert result.returncode != 0
    assert not any(call["command"] in ("rpm", "dnf", "ln") for call in calls)


@pytest.mark.parametrize(
    "change", ["missing", "empty", "duplicate", "path", "hash", "extra"]
)
def test_bootstrap_rejects_bad_manifest_before_download(bootstrap, change):
    scripts, _names, _checks, run = bootstrap
    manifest = scripts / MANIFEST
    lines = manifest.read_text().splitlines()
    if change == "missing":
        lines.pop()
    elif change == "empty":
        lines = []
    elif change == "duplicate":
        lines[-1] = lines[0]
    elif change == "path":
        lines[0] = lines[0].replace("  cccl", "  ../cccl")
    elif change == "hash":
        lines[0] = "not-a-sha256  " + lines[0].split()[1]
    else:
        lines[0] += " extra-field"
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""))
    result, calls = run()
    assert result.returncode != 0
    assert calls == []


def test_bootstrap_propagates_install_failure_without_toolkit_link(bootstrap):
    _scripts, _names, _checks, run = bootstrap
    result, calls = run(CUDA_TEST_FAIL="dnf")
    assert result.returncode != 0
    assert not any(call["command"] == "ln" for call in calls)


def test_bootstrap_missing_manifest_fails_before_download(bootstrap):
    scripts, _names, _checks, run = bootstrap
    (scripts / MANIFEST).unlink()
    result, calls = run()
    assert result.returncode != 0
    assert "missing RPM checksum manifest" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    "component", ["bin/nvdisasm", "include/crt/host_config.h", "lib64/libcudart.so.13"]
)
def test_bootstrap_rejects_missing_installed_components(bootstrap, component):
    _scripts, _names, _checks, run = bootstrap
    result, calls = run(CUDA_TEST_MISSING_COMPONENT=f"/usr/local/cuda-13.3/{component}")
    assert result.returncode != 0
    assert "missing " in result.stderr
    assert not any(call["command"] == "ln" for call in calls)


def test_release_gate_requires_full_pinned_bootstrap(tmp_path: Path):
    namespace = runpy.run_path(
        str(ROOT / "tests/release_measure/artifact_01_release_composition.py")
    )
    check = namespace["_assert_cuda_ci_bootstrap"]
    check(ROOT)
    scripts = tmp_path / ".github" / "scripts"
    scripts.mkdir(parents=True)
    for name in (HELPER, MANIFEST):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    manifest = scripts / MANIFEST
    manifest.write_text("\n".join(manifest.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(AssertionError, match="complete 13-RPM"):
        check(tmp_path)


@pytest.mark.parametrize(
    "removed", ["rpm --checksig --verbose", "--setopt=localpkg_gpgcheck=1", KEY_SHA256]
)
def test_release_gate_rejects_bootstrap_verification_drift(
    tmp_path: Path, removed: str
):
    namespace = runpy.run_path(
        str(ROOT / "tests/release_measure/artifact_01_release_composition.py")
    )
    scripts = tmp_path / ".github" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPTS / MANIFEST, scripts / MANIFEST)
    (scripts / HELPER).write_text(
        (SCRIPTS / HELPER).read_text().replace(removed, "REMOVED")
    )
    with pytest.raises(AssertionError, match="lost verified provisioning"):
        namespace["_assert_cuda_ci_bootstrap"](tmp_path)
