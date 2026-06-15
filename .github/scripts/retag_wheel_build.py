from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path


def _hash_record(data: bytes) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}", str(len(data))


def _with_build_tag(filename: str, build_tag: str) -> str:
    if not filename.endswith(".whl"):
        raise ValueError(f"not a wheel filename: {filename}")
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) == 5:
        dist, version, py_tag, abi_tag, platform_tag = parts
        return f"{dist}-{version}-{build_tag}-{py_tag}-{abi_tag}-{platform_tag}.whl"
    if len(parts) == 6:
        dist, version, _old_build, py_tag, abi_tag, platform_tag = parts
        return f"{dist}-{version}-{build_tag}-{py_tag}-{abi_tag}-{platform_tag}.whl"
    raise ValueError(f"unsupported wheel filename shape: {filename}")


def _add_build_field(wheel_text: str, build_tag: str) -> str:
    lines = wheel_text.splitlines()
    out: list[str] = []
    inserted = False
    replaced = False
    for line in lines:
        if line.startswith("Build:"):
            out.append(f"Build: {build_tag}")
            replaced = True
            continue
        out.append(line)
        if not inserted and not replaced and line.startswith("Wheel-Version:"):
            out.append(f"Build: {build_tag}")
            inserted = True
    if not inserted and not replaced:
        out.insert(0, f"Build: {build_tag}")
    return "\n".join(out) + "\n"


def retag_wheel(path: Path, build_tag: str, remove_original: bool) -> Path:
    output = path.with_name(_with_build_tag(path.name, build_tag))
    if output.exists():
        output.unlink()

    entries: list[tuple[zipfile.ZipInfo, bytes]] = []
    record_name: str | None = None
    wheel_seen = False

    with zipfile.ZipFile(path, "r") as source:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(".dist-info/RECORD"):
                record_name = info.filename
                continue
            if info.filename.endswith(".dist-info/WHEEL"):
                data = _add_build_field(data.decode("utf-8"), build_tag).encode("utf-8")
                wheel_seen = True
            entries.append((info, data))

    if record_name is None:
        raise RuntimeError(f"{path} does not contain a .dist-info/RECORD file")
    if not wheel_seen:
        raise RuntimeError(f"{path} does not contain a .dist-info/WHEEL file")

    record_buffer = io.StringIO()
    writer = csv.writer(record_buffer, lineterminator="\n")
    for info, data in entries:
        digest, size = _hash_record(data)
        writer.writerow([info.filename, digest, size])
    writer.writerow([record_name, "", ""])
    record_data = record_buffer.getvalue().encode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".whl", dir=str(path.parent)) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for info, data in entries:
                target.writestr(info, data)
            target.writestr(record_name, record_data)
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if remove_original and output != path:
        path.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-tag", required=True)
    parser.add_argument("--remove-original", action="store_true")
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    for wheel in args.wheels:
        output = retag_wheel(wheel, args.build_tag, args.remove_original)
        print(f"{wheel.name} -> {output.name}")


if __name__ == "__main__":
    main()
