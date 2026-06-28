from __future__ import annotations

import argparse

from gafime import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="GAFIME v1 native runtime")
    parser.add_argument("-V", "--version", action="version", version=f"gafime {__version__}")
    parser.add_argument("--check", action="store_true", help="Check the v1 Python boundary import")
    args = parser.parse_args()

    if args.check:
        _check_v1_boundary()
        return
    parser.print_help()


def _check_v1_boundary() -> None:
    from gafime.v1_adapter import _load_boundary

    boundary = _load_boundary()
    print(f"GAFIME v{__version__}")
    print(f"v1 boundary: {getattr(boundary, 'BOUNDARY_NAME', boundary.__name__)}")


if __name__ == "__main__":
    main()
