#!/usr/bin/env python3
"""Fail closed unless local, tag, and release-branch commits still match."""

from __future__ import annotations

import argparse
import re
import subprocess

from release_version import ReleaseVersion, VersionPolicyError


_SHA1_RE = re.compile(r"[0-9a-f]{40}")


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_release_ref_identity(release_tag: str, source_sha: str) -> None:
    """Verify the checked-out source and live release refs share one commit."""

    try:
        version = ReleaseVersion.from_tag(release_tag)
    except VersionPolicyError as exc:
        raise ValueError(str(exc)) from exc
    if version.tag != release_tag:
        raise ValueError(f"release tag {release_tag!r} is not canonical")
    if _SHA1_RE.fullmatch(source_sha) is None:
        raise ValueError("source SHA must be one lowercase 40-character Git SHA-1")

    local_sha = _git("rev-parse", "--verify", "HEAD^{commit}")
    if local_sha != source_sha:
        raise RuntimeError(
            f"checked-out source resolves to {local_sha}, expected {source_sha}"
        )

    tag_ref = f"refs/tags/{release_tag}"
    branch_ref = f"refs/heads/release/{release_tag}"
    output = _git("ls-remote", "origin", tag_ref, f"{tag_ref}^{{}}", branch_ref)
    refs: dict[str, str] = {}
    for line in output.splitlines():
        sha, ref = line.split("\t", 1)
        if ref in refs:
            raise RuntimeError(f"remote returned duplicate release ref {ref}")
        refs[ref] = sha

    remote_tag_sha = refs.get(f"{tag_ref}^{{}}", refs.get(tag_ref))
    remote_branch_sha = refs.get(branch_ref)
    if remote_tag_sha != source_sha:
        raise RuntimeError(
            f"remote tag {tag_ref} resolves to {remote_tag_sha}, expected {source_sha}"
        )
    if remote_branch_sha != source_sha:
        raise RuntimeError(
            f"remote branch {branch_ref} resolves to {remote_branch_sha}, "
            f"expected {source_sha}"
        )

    print(
        f"release refs verified: source={source_sha} tag={tag_ref} branch={branch_ref}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    try:
        verify_release_ref_identity(args.release_tag, args.source_sha)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
