from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / ".github" / "scripts" / "release_bundle.py"
SPEC = importlib.util.spec_from_file_location("gafime_release_bundle", BUNDLE_PATH)
assert SPEC is not None and SPEC.loader is not None
release_bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_bundle
SPEC.loader.exec_module(release_bundle)

TRACKED_FIXTURE_FILES = (
    Path(".gitattributes"),
    Path(".github/release-artifacts.json"),
    Path(".github/scripts/release_bundle.py"),
    Path("tests/release_measure/release_manifest.py"),
)


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _git_show_bytes(repository: Path, revision_and_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", revision_and_path],
        cwd=repository,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"git show failed ({completed.returncode}): {revision_and_path}\n"
            f"stderr:\n{completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def _tracked_fixture_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "source"
    repository.mkdir()
    for relative in TRACKED_FIXTURE_FILES:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    _run(["git", "init", "--initial-branch=main"], cwd=repository)
    _run(["git", "config", "user.name", "GAFIME fixture"], cwd=repository)
    _run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repository)
    _run(["git", "config", "core.autocrlf", "false"], cwd=repository)
    _run(["git", "add", *map(str, TRACKED_FIXTURE_FILES)], cwd=repository)
    _run(["git", "commit", "-m", "tracked release manifest fixture"], cwd=repository)
    return repository


def _checkout_with_autocrlf(repository: Path, destination: Path, autocrlf: str) -> Path:
    # Configure the clone before materializing the worktree. Reconfiguring an
    # already checked-out path would not prove Git's checkout conversion rule.
    _run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--no-local",
            str(repository),
            str(destination),
        ],
        cwd=repository.parent,
    )
    _run(["git", "config", "core.autocrlf", autocrlf], cwd=destination)
    _run(["git", "checkout", "--force", "HEAD"], cwd=destination)
    return destination


def _receipt_fields(output: str) -> dict[str, str]:
    prefix = "RELEASE BUNDLE FIXTURE RECEIPT: PASS "
    line = next((line for line in output.splitlines() if line.startswith(prefix)), None)
    assert line is not None, output
    fields = dict(field.split("=", 1) for field in line[len(prefix) :].split())
    assert set(fields) == {
        "manifest_sha256",
        "provenance_sha256",
        "sha256sums_sha256",
        "packages",
    }
    return fields


def test_release_manifest_digest_rejects_carriage_return_bytes(tmp_path: Path) -> None:
    manifest = tmp_path / ".github" / "release-artifacts.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"{}\r\n")

    with pytest.raises(ValueError, match="LF-only bytes"):
        release_bundle._manifest_digest(tmp_path)


def test_tracked_manifest_and_fixture_receipt_are_deterministic_across_autocrlf(
    tmp_path: Path,
) -> None:
    source = _tracked_fixture_repository(tmp_path)
    tracked_bytes = _git_show_bytes(source, "HEAD:.github/release-artifacts.json")
    expected_manifest_sha256 = hashlib.sha256(tracked_bytes).hexdigest()
    expected_package_count = str(
        release_bundle.load_release_manifest(ROOT).standard_artifact_count
    )
    assert b"\r" not in tracked_bytes

    receipts: dict[str, dict[str, str]] = {}
    for autocrlf in ("false", "true"):
        checkout = _checkout_with_autocrlf(
            source, tmp_path / f"checkout-{autocrlf}", autocrlf
        )
        assert (
            _run(
                ["git", "config", "--get", "core.autocrlf"], cwd=checkout
            ).stdout.strip()
            == autocrlf
        )
        attributes = _run(
            [
                "git",
                "check-attr",
                "text",
                "eol",
                "--",
                ".github/release-artifacts.json",
            ],
            cwd=checkout,
        ).stdout
        assert ".github/release-artifacts.json: text: set" in attributes
        assert ".github/release-artifacts.json: eol: lf" in attributes

        manifest_bytes = (checkout / ".github/release-artifacts.json").read_bytes()
        assert manifest_bytes == tracked_bytes
        assert b"\r" not in manifest_bytes

        receipt = _run(
            [sys.executable, ".github/scripts/release_bundle.py", "--fixture-receipt"],
            cwd=checkout,
        )
        fields = _receipt_fields(receipt.stdout)
        assert fields["manifest_sha256"] == expected_manifest_sha256
        assert fields["packages"] == expected_package_count
        receipts[autocrlf] = fields
        print(
            "WINDOWS-PORTABLE RELEASE RECEIPT "
            f"autocrlf={autocrlf} "
            f"manifest_sha256={fields['manifest_sha256']} "
            f"provenance_sha256={fields['provenance_sha256']} "
            f"sha256sums_sha256={fields['sha256sums_sha256']}"
        )

    assert receipts["false"] == receipts["true"]
