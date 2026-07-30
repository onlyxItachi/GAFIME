from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import sysconfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install exactly one locally built GAFIME core wheel."
    )
    parser.add_argument("wheelhouse", type=Path)
    args = parser.parse_args()

    wheelhouse = args.wheelhouse.resolve()
    cache_tag = sys.implementation.cache_tag
    if cache_tag is None or not cache_tag.startswith("cpython-"):
        parser.error(f"unsupported Python implementation tag: {cache_tag!r}")
    python_tag = f"cp{cache_tag.removeprefix('cpython-')}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    wheels = sorted(
        path
        for path in wheelhouse.glob(f"gafime-*-{python_tag}-{python_tag}-*.whl")
        if path.is_file() and not path.name.startswith("gafime_")
    )
    if len(wheels) != 1:
        parser.error(
            f"expected exactly one {python_tag} core wheel for {platform_tag} in "
            f"{wheelhouse}, found "
            f"{[path.name for path in wheels]}"
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", str(wheels[0])],
        check=True,
    )


if __name__ == "__main__":
    main()
