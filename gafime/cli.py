import argparse
import sys

from gafime import __version__
from gafime.tutorial import generate_tutorial


def main():
    parser = argparse.ArgumentParser(
        description="GAFIME — GPU-Accelerated Feature Interaction Mining Engine"
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"gafime {__version__}"
    )
    parser.add_argument(
        "-i", "--init", action="store_true",
        help="Generate an interactive starter Jupyter notebook"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="gafime_tutorial.ipynb",
        help="Output path for the generated tutorial notebook"
    )
    parser.add_argument(
        "-c", "--check", action="store_true",
        help="Check available compute backends"
    )

    args = parser.parse_args()

    if args.check:
        _check_backends()
    elif args.init:
        generate_tutorial(output_path=args.output)
    else:
        parser.print_help()
        sys.exit(0)


def _check_backends():
    """Check and report available compute backends."""
    import numpy as np
    from gafime.config import EngineConfig
    from gafime.backends import resolve_backend

    print(f"GAFIME v{__version__}")
    print("=" * 40)

    X = np.zeros((10, 2), dtype=np.float64)
    y = np.zeros(10, dtype=np.float64)

    config = EngineConfig(backend="auto")
    backend, warnings = resolve_backend(config, X, y)
    info = backend.info()

    print(f"  Selected backend: {info.name}")
    print(f"  Device: {info.device}")
    print(f"  GPU: {'Yes' if info.is_gpu else 'No'}")

    if warnings:
        print(f"\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    print(f"\n  Backend availability:")
    for name, bstr in [("CUDA", "cuda"), ("Metal", "metal"), ("C++ Core", "core"), ("NumPy", "numpy")]:
        try:
            cfg = EngineConfig(backend=bstr)
            b, w = resolve_backend(cfg, X, y)
            binfo = b.info()
            # Verify the resolved backend matches what was requested
            if bstr != "numpy" and binfo.name == "numpy":
                print(f"    {name:12s}: not available (fell back to numpy)")
            else:
                print(f"    {name:12s}: available ({binfo.name})")
        except Exception as e:
            print(f"    {name:12s}: not available ({e})")


if __name__ == "__main__":
    main()
