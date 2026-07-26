from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install exactly one locally built GAFIME core wheel."
    )
    parser.add_argument("wheelhouse", type=Path)
    args = parser.parse_args()

    wheelhouse = args.wheelhouse.resolve()
    wheels = sorted(
        path
        for path in wheelhouse.glob("gafime-*.whl")
        if path.is_file() and not path.name.startswith("gafime_")
    )
    if len(wheels) != 1:
        parser.error(
            f"expected exactly one core wheel in {wheelhouse}, found "
            f"{[path.name for path in wheels]}"
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", str(wheels[0])],
        check=True,
    )


if __name__ == "__main__":
    main()
