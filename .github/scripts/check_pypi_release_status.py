#!/usr/bin/env python3
"""Verify missing or fully yanked releases through PyPI's JSON API."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


PROJECT_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*")
USER_AGENT = "GAFIME-release-status-check/1"


class ReleaseStatusError(RuntimeError):
    """Raised when live PyPI state does not match the requested contract."""


@dataclass(frozen=True)
class ReleaseSpec:
    project: str
    version: str

    @property
    def display(self) -> str:
        return f"{self.project}=={self.version}"


def parse_release_spec(value: str) -> ReleaseSpec:
    project, separator, version = value.partition("==")
    if (
        separator != "=="
        or PROJECT_PATTERN.fullmatch(project) is None
        or VERSION_PATTERN.fullmatch(version) is None
    ):
        raise argparse.ArgumentTypeError(
            f"invalid release {value!r}; expected PROJECT==VERSION"
        )
    return ReleaseSpec(project=project, version=version)


def release_json_url(spec: ReleaseSpec) -> str:
    project = urllib.parse.quote(spec.project, safe="")
    version = urllib.parse.quote(spec.version, safe="")
    return f"https://pypi.org/pypi/{project}/{version}/json"


def fetch_release(spec: ReleaseSpec, timeout: float) -> dict[str, Any] | None:
    request = urllib.request.Request(
        release_json_url(spec),
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ReleaseStatusError(
            f"{spec.display}: PyPI returned HTTP {exc.code}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseStatusError(
            f"{spec.display}: could not read PyPI release metadata: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReleaseStatusError(f"{spec.display}: PyPI returned invalid JSON")
    return payload


def require_missing(spec: ReleaseSpec, payload: dict[str, Any] | None) -> None:
    if payload is not None:
        raise ReleaseStatusError(
            f"{spec.display}: release exists but was required to be absent"
        )


def require_yanked(
    spec: ReleaseSpec,
    payload: dict[str, Any] | None,
    reason_contains: str | None,
) -> tuple[int, tuple[str, ...]]:
    if payload is None:
        raise ReleaseStatusError(
            f"{spec.display}: release is absent but was required to be yanked"
        )
    files = payload.get("urls")
    if not isinstance(files, list) or not files:
        raise ReleaseStatusError(f"{spec.display}: release contains no files")

    unyanked: list[str] = []
    missing_reasons: list[str] = []
    reasons: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ReleaseStatusError(
                f"{spec.display}: release contains invalid file metadata"
            )
        filename = str(item.get("filename", "<unknown>"))
        if item.get("yanked") is not True:
            unyanked.append(filename)
            continue
        reason = item.get("yanked_reason")
        if not isinstance(reason, str) or not reason.strip():
            missing_reasons.append(filename)
            continue
        reason = reason.strip()
        reasons.add(reason)
        if (
            reason_contains is not None
            and reason_contains.casefold() not in reason.casefold()
        ):
            raise ReleaseStatusError(
                f"{spec.display}: yank reason for {filename!r} does not contain "
                f"{reason_contains!r}"
            )

    if unyanked:
        raise ReleaseStatusError(
            f"{spec.display}: unyanked files remain: {', '.join(unyanked)}"
        )
    if missing_reasons:
        raise ReleaseStatusError(
            f"{spec.display}: yanked files lack a reason: {', '.join(missing_reasons)}"
        )
    return len(files), tuple(sorted(reasons))


def self_test() -> None:
    spec = parse_release_spec("gafime-cuda==1.0.0b1")
    assert spec.display == "gafime-cuda==1.0.0b1"
    assert release_json_url(spec).endswith("/gafime-cuda/1.0.0b1/json")

    payload = {
        "urls": [
            {
                "filename": "gafime_cuda-1.0.0b1.tar.gz",
                "yanked": True,
                "yanked_reason": "Matching Core package was not published.",
            },
            {
                "filename": "gafime_cuda-1.0.0b1-cp310-abi3.whl",
                "yanked": True,
                "yanked_reason": "Matching Core package was not published.",
            },
        ]
    }
    count, reasons = require_yanked(spec, payload, "Core package")
    assert count == 2
    assert reasons == ("Matching Core package was not published.",)
    require_missing(ReleaseSpec("gafime", "1.0.0b1"), None)

    broken = {"urls": [{**payload["urls"][0], "yanked": False}]}
    try:
        require_yanked(spec, broken, None)
    except ReleaseStatusError as exc:
        assert "unyanked files remain" in str(exc)
    else:
        raise AssertionError("unyanked release unexpectedly passed")

    print("PyPI release status self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify missing and fully yanked PyPI releases."
    )
    parser.add_argument(
        "--expect-missing",
        action="append",
        default=[],
        type=parse_release_spec,
        metavar="PROJECT==VERSION",
    )
    parser.add_argument(
        "--expect-yanked",
        action="append",
        default=[],
        type=parse_release_spec,
        metavar="PROJECT==VERSION",
    )
    parser.add_argument(
        "--reason-contains",
        help="Require every yank reason to contain this text, case-insensitively.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.expect_missing and not args.expect_yanked:
        parser.error("at least one expected release state is required")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    for spec in args.expect_missing:
        require_missing(spec, fetch_release(spec, args.timeout))
        print(f"PASS missing {spec.display}")
    for spec in args.expect_yanked:
        count, reasons = require_yanked(
            spec,
            fetch_release(spec, args.timeout),
            args.reason_contains,
        )
        print(
            f"PASS yanked {spec.display} files={count} "
            f"reasons={json.dumps(reasons)}"
        )
    print(
        "PYPI RELEASE STATUS: PASS "
        f"missing={len(args.expect_missing)} yanked={len(args.expect_yanked)}"
    )


if __name__ == "__main__":
    try:
        main()
    except ReleaseStatusError as exc:
        raise SystemExit(f"PYPI RELEASE STATUS: FAIL {exc}") from exc
