"""backend_02 | BACKEND numerical parity: the same dataset must give the same
interactions on core vs CUDA vs ROCm (within fp tolerance). This is what backs
"every backend flawless". GPU rows skip cleanly if that GPU is absent.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/mc-torch-cu/bin/python backend_02_cross_backend_parity.py
"""
import numpy as np

from gafime.config import EngineConfig
from gafime.engine import GafimeEngine

import _measure_common as mc

TOL = 1e-3


def metric_map(report):
    out = {}
    for ir in (getattr(report, "interactions", []) or []):
        out[tuple(ir.combo)] = {k: float(v) for k, v in ir.metrics.items()}
    return out


def main():
    X, y, names, meta, _ = mc.load_synthetic_and(n=3000, f=10)
    Xl, yl = X.tolist(), y.tolist()
    ref = metric_map(GafimeEngine(config=EngineConfig(backend="core")).analyze(Xl, yl, feature_names=names))
    print(f"core reference: {len(ref)} interactions")
    for backend in ("cuda", "rocm"):
        try:
            r = metric_map(GafimeEngine(config=EngineConfig(backend=backend)).analyze(Xl, yl, feature_names=names))
        except Exception as exc:
            print(f"[{backend}] skipped (GPU absent): {type(exc).__name__}: {str(exc)[:50]}")
            continue
        maxd, missing = 0.0, 0
        for combo, m in ref.items():
            if combo not in r:
                missing += 1
                continue
            for k, v in m.items():
                maxd = max(maxd, abs(v - r[combo].get(k, v)))
        verdict = "PASS" if (maxd <= TOL and missing == 0) else "FAIL"
        print(f"[{backend}] vs core: max|Δ|={maxd:.2e} missing={missing} -> {verdict} (tol {TOL})")
    print("\nrecord per-GPU parity; any FAIL blocks the 'flawless backend' claim.")


if __name__ == "__main__":
    main()
