"""Exercise macOS CI bootstrap without network, real toolchains or caller state."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    os.name != "posix" or BASH is None, reason="POSIX CI shell fixture requires bash"
)


def _script() -> str:
    workflow = (ROOT / ".github/workflows/native_platform_validation.yml").read_text()
    metal = workflow.split("  metal_macos:\n", 1)[1].split("\n  arm_linux:", 1)[0]
    step = metal.split("      - name: Install build tooling\n", 1)[1]
    step = step.split("      - name:", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


def _executable(path: Path, content: str) -> None:
    path.write_text("#!/bin/sh\n" + content)
    path.chmod(0o755)


def _run(tmp_path: Path, *, home_rustup: bool, global_rustup: bool, fail: bool = False):
    home = tmp_path / "home"
    home.mkdir()
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "calls.log"
    github_path = tmp_path / "github-path"
    payload = tmp_path / "rustup-payload"
    _executable(payload, 'printf "home:%s\\n" "$*" >> "$CALL_LOG"\n')
    _executable(tools / "python", 'printf "pip\\n" >> "$CALL_LOG"\n')
    _executable(
        tools / "curl",
        'printf "installer\\n" >> "$CALL_LOG"\n'
        'if [ "$FAIL_INSTALLER" = 1 ]; then printf "exit 17\\n"; exit 0; fi\n'
        "printf '%s\\n' '/bin/mkdir -p \"$HOME/.cargo/bin\"' "
        '\'/bin/cp "$RUSTUP_PAYLOAD" "$HOME/.cargo/bin/rustup"\'\n',
    )
    (tools / "sh").symlink_to("/bin/sh")
    if global_rustup:
        _executable(tools / "rustup", 'printf "global\\n" >> "$CALL_LOG"\n')
    if home_rustup:
        target = home / ".cargo/bin/rustup"
        target.parent.mkdir(parents=True)
        shutil.copyfile(payload, target)
        target.chmod(0o755)
    result = subprocess.run(
        [BASH, "-c", _script()],
        cwd=tmp_path,
        env={
            "HOME": str(home),
            "PATH": str(tools),
            "GITHUB_PATH": str(github_path),
            "CALL_LOG": str(log),
            "RUSTUP_PAYLOAD": str(payload),
            "FAIL_INSTALLER": str(int(fail)),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, log.read_text().splitlines(), github_path, home


@pytest.mark.parametrize(
    "home_rustup,global_rustup", [(True, False), (False, True), (False, False)]
)
def test_macos_bootstrap_requires_the_executable_it_will_invoke(
    tmp_path: Path, home_rustup: bool, global_rustup: bool
) -> None:
    result, calls, github_path, home = _run(
        tmp_path, home_rustup=home_rustup, global_rustup=global_rustup
    )
    assert result.returncode == 0, result.stderr
    assert ("installer" in calls) is not home_rustup
    assert "global" not in calls
    assert "home:toolchain install 1.97.1 --profile minimal" in calls
    assert "home:default 1.97.1" in calls
    assert github_path.read_text() == f"{home}/.cargo/bin\n"


def test_macos_bootstrap_installer_failure_does_not_pass(tmp_path: Path) -> None:
    result, calls, github_path, _home = _run(
        tmp_path, home_rustup=False, global_rustup=True, fail=True
    )
    assert result.returncode == 17
    assert not any(call.startswith("home:") for call in calls)
    assert not github_path.exists()
