from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
METAL_LIBRARY = "libgafime_metal_v1.dylib"
METALLIB = "gafime_metal_v1.metallib"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def stage_metal_payload(output: Path, build_dir: Path) -> tuple[Path, Path]:
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("the bundled Metal payload can only be staged on macOS arm64")

    output = output.resolve()
    build_dir = build_dir.resolve()
    library_dir = build_dir / "library"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    library_dir.mkdir(parents=True)

    run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "src" / "metal"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={library_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={library_dir}",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "gafime_metal_v1",
        ]
    )

    library = library_dir / METAL_LIBRARY
    metallib = build_dir / METALLIB
    for artifact in (library, metallib):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Metal build did not produce {artifact}")
    run(["lipo", "-verify_arch", "arm64", str(library)])

    output.mkdir(parents=True, exist_ok=True)
    staged_library = output / METAL_LIBRARY
    staged_metallib = output / METALLIB
    for source, destination in ((library, staged_library), (metallib, staged_metallib)):
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return staged_library, staged_metallib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "python" / "gafime" / "_metal",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPO_ROOT / "build" / "metal-payload",
    )
    args = parser.parse_args()
    library, metallib = stage_metal_payload(args.output, args.build_dir)
    print(f"staged Metal payload: {library} {metallib}")


if __name__ == "__main__":
    main()
