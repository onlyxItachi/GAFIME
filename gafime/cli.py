from __future__ import annotations

import argparse
import sys

from gafime import __version__
from gafime.tutorial import generate_tutorial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GAFIME - GPU-Accelerated Feature Interaction Mining Engine"
    )
    parser.add_argument("-V", "--version", action="version", version=f"gafime {__version__}")
    parser.add_argument("-i", "--init", action="store_true", help="Generate a starter notebook")
    parser.add_argument("-o", "--output", type=str, default="gafime_tutorial.ipynb")
    parser.add_argument("-c", "--check", action="store_true", help="Check native compute backends")
    args = parser.parse_args()

    if args.check:
        _check_backends()
    elif args.init:
        generate_tutorial(output_path=args.output)
    else:
        parser.print_help()
        sys.exit(0)


def _check_backends() -> None:
    from gafime.backends import resolve_backend
    from gafime.config import EngineConfig
    from gafime.native_data import coerce_inputs

    print(f"GAFIME v{__version__}")
    print("=" * 40)
    X, y, _ = coerce_inputs([[0.0, 0.0], [1.0, 1.0]], [0.0, 1.0])

    for name, backend_name in [
        ("CUDA", "cuda"),
        ("Metal", "metal"),
        ("C++ Core", "core"),
        ("Auto", "auto"),
    ]:
        try:
            backend, warnings = resolve_backend(EngineConfig(backend=backend_name), X, y)
            info = backend.info()
            print(f"    {name:12s}: available ({info.name}, {info.device})")
            for warning in warnings:
                print(f"      warning: {warning}")
        except Exception as exc:
            print(f"    {name:12s}: not available ({exc})")


if __name__ == "__main__":
    main()
