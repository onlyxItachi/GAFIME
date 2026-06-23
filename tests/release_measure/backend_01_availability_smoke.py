"""backend_01 | BACKEND coverage: probe each backend (core / cuda / rocm / metal),
report which load natively vs fall back, and dump their capability info. The
honest "every backend" status snapshot for the release notes.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python backend_01_availability_smoke.py
"""
import numpy as np

from gafime.config import EngineConfig

try:
    from gafime.backends import resolve_backend
    from gafime.utils.arrays import coerce_inputs
except Exception:  # pragma: no cover
    resolve_backend = None
    coerce_inputs = None


def main():
    Xraw = np.random.default_rng(0).random((64, 6))
    yraw = (Xraw[:, 0] > 0.5).astype(float)
    # coerce to the engine's native inputs, exactly as CompiledGafime.from_engine does
    X, y, _ = coerce_inputs(Xraw.tolist(), yraw.tolist(), None) if coerce_inputs else (Xraw, yraw, None)
    print(f"{'requested':<10}{'resolved':<16}{'native?':<9}{'notes'}")
    for name in ("core", "cuda", "rocm", "metal", "auto"):
        try:
            cfg = EngineConfig(backend=name)
            if resolve_backend is None:
                print(f"{name:<10}{'<no resolver>':<16}")
                continue
            backend, warnings = resolve_backend(cfg, X, y)
            info = backend.info()
            native = "native" in getattr(info, "name", "").lower() or "numpy" not in getattr(info, "name", "").lower()
            print(f"{name:<10}{getattr(info,'name','?'):<16}{str(native):<9}{(warnings or [''])[0][:50]}")
        except Exception as exc:
            print(f"{name:<10}{'<unavailable>':<16}{'False':<9}{type(exc).__name__}: {str(exc)[:40]}")
    print("\nrecord which backends resolve native on THIS host (4060 sm_89 / gfx1150).")


if __name__ == "__main__":
    main()
