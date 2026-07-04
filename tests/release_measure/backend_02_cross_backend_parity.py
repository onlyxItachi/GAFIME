"""backend_02 | BACKEND numerical parity: the same dataset must give the same
interactions on core vs CUDA vs ROCm (within fp tolerance). This is what backs
"every backend flawless". GPU rows skip cleanly if that GPU is absent.

Mutual information has two approved CPU modes: default adaptive quantile bins
and fixed equal-width bins. CUDA/ROCm implement the fixed-bin estimator, so this
gate validates exact metrics in default mode and validates MI only through the
explicit fixed-bin parity path (`mi_approximate=True`).

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 backend_02_cross_backend_parity.py
"""
import numpy as np

from gafime.config import EngineConfig
from gafime import GafimeEngine

import _measure_common as mc

TOL = 1e-3
EXACT_METRICS = ("pearson", "spearman", "r2")
FIXED_BIN_METRICS = ("pearson", "spearman", "mutual_info", "r2")


def metric_map(report):
    out = {}
    for ir in (getattr(report, "interactions", []) or []):
        out[tuple(sorted(ir.combo))] = {k: float(v) for k, v in ir.metrics.items()}
    return out


def run_case(metrics, *, mi_approximate):
    X, y, names, meta, _ = mc.load_synthetic_and(n=3000, f=10)
    Xl, yl = X.tolist(), y.tolist()
    ref = metric_map(
        GafimeEngine(
            config=EngineConfig(
                backend="core",
                metric_names=metrics,
                mi_approximate=mi_approximate,
            )
        ).analyze(Xl, yl, feature_names=names)
    )
    print(
        f"core reference: {len(ref)} interactions "
        f"metrics={','.join(metrics)} mi_approximate={mi_approximate}"
    )
    failures = []
    for backend in ("cuda", "rocm"):
        try:
            r = metric_map(
                GafimeEngine(
                    config=EngineConfig(
                        backend=backend,
                        metric_names=metrics,
                        mi_approximate=mi_approximate,
                    )
                ).analyze(Xl, yl, feature_names=names)
            )
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
        if verdict != "PASS":
            failures.append(f"{backend}: max_delta={maxd:.6g} missing={missing}")
    return failures


def main():
    failures = []
    failures.extend(run_case(EXACT_METRICS, mi_approximate=False))
    failures.extend(run_case(FIXED_BIN_METRICS, mi_approximate=True))
    print("\nrecord per-GPU parity; any FAIL blocks the 'flawless backend' claim.")
    if failures:
        raise AssertionError("backend parity failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
