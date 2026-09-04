from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "verify_release_ref_identity.py"
TAG = "v1.0.0-rc.1"
BRANCH = f"release/{TAG}"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, *, annotated_tag: bool = False) -> tuple[Path, str]:
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "--initial-branch=main", str(work))
    _git(work, "config", "user.name", "GAFIME Test")
    _git(work, "config", "user.email", "test@example.invalid")
    (work / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(work, "add", "tracked.txt")
    _git(work, "commit", "-m", "initial")
    source_sha = _git(work, "rev-parse", "HEAD")
    _git(work, "branch", BRANCH)
    if annotated_tag:
        _git(work, "tag", "--annotate", TAG, "--message", "candidate")
    else:
        _git(work, "tag", TAG)
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "origin", "main", BRANCH, TAG)
    return work, source_sha


def _verify(work: Path, source_sha: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            "python",
            str(SCRIPT),
            "--release-tag",
            TAG,
            "--source-sha",
            source_sha,
        ),
        cwd=work,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("annotated_tag", [False, True])
def test_release_ref_identity_accepts_matching_lightweight_and_annotated_tags(
    tmp_path: Path, annotated_tag: bool
) -> None:
    work, source_sha = _repository(tmp_path, annotated_tag=annotated_tag)

    result = _verify(work, source_sha)

    assert result.returncode == 0, result.stderr
    assert f"source={source_sha}" in result.stdout


def test_release_ref_identity_rejects_remote_tag_movement(tmp_path: Path) -> None:
    work, source_sha = _repository(tmp_path)
    (work / "tracked.txt").write_text("tag moved\n", encoding="utf-8")
    _git(work, "commit", "--all", "-m", "move tag")
    _git(work, "tag", "--force", TAG)
    _git(work, "push", "--force", "origin", f"refs/tags/{TAG}")
    _git(work, "checkout", "--detach", source_sha)

    result = _verify(work, source_sha)

    assert result.returncode != 0
    assert "remote tag" in result.stderr


def test_release_ref_identity_rejects_remote_branch_movement(tmp_path: Path) -> None:
    work, source_sha = _repository(tmp_path)
    (work / "tracked.txt").write_text("branch moved\n", encoding="utf-8")
    _git(work, "commit", "--all", "-m", "move branch")
    _git(work, "push", "origin", f"HEAD:refs/heads/{BRANCH}")
    _git(work, "checkout", "--detach", source_sha)

    result = _verify(work, source_sha)

    assert result.returncode != 0
    assert "remote branch" in result.stderr
