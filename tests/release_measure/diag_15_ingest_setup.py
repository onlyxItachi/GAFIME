#!/usr/bin/env python3
"""Retain public PyPI RC2 metadata and authenticate the negative-control wheel."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rc2-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = "https://pypi.org/pypi/gafime/1.0.0rc2/json"
    with urlopen(url, timeout=30) as response:
        data = json.load(response)
    matches = [item for item in data["urls"] if item["filename"] == args.rc2_wheel.name]
    if len(matches) != 1:
        raise ValueError("RC2 filename absent or duplicated in public PyPI metadata")
    with args.rc2_wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != matches[0]["digests"]["sha256"]:
        raise ValueError("RC2 wheel differs from public PyPI digest")
    args.output.write_text(
        json.dumps({"url": url, "wheel_sha256": digest, "metadata": data}, indent=2)
    )


if __name__ == "__main__":
    main()
