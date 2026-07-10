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

import math
import os

from gafime import GafimeEngine
from gafime.config import EngineConfig

import _measure_common as mc

TOL = 1e-3
EXACT_METRICS = ("pearson", "spearman", "r2")
FIXED_BIN_METRICS = ("pearson", "spearman", "mutual_info", "r2")


def metric_map(report):
    out = {}
    for ir in getattr(report, "interactions", []) or []:
        combo = tuple(sorted(ir.combo))
        if combo in out:
            raise AssertionError(f"duplicate interaction combo in report: {combo}")
        out[combo] = {k: float(v) for k, v in ir.metrics.items()}
    return out


def run_case(metrics, *, mi_approximate, mi_bins=96, rows=3000):
    X, y, names, _meta, _ = mc.load_synthetic_and(n=rows, f=10)
    Xl, yl = X.tolist(), y.tolist()
    ref = metric_map(
        GafimeEngine(
            config=EngineConfig(
                backend="core",
                metric_names=metrics,
                mi_approximate=mi_approximate,
                mi_bins=mi_bins,
            )
        ).analyze(Xl, yl, feature_names=names)
    )
    print(
        f"core reference: {len(ref)} interactions "
        f"metrics={','.join(metrics)} rows={rows} "
        f"mi_approximate={mi_approximate} mi_bins={mi_bins}"
    )
    failures = []
    for backend in ("cuda", "rocm"):
        payload_env = {
            "cuda": "GAFIME_CUDA_V1_LIB",
            "rocm": "GAFIME_ROCM_V1_LIB",
        }[backend]
        try:
            r = metric_map(
                GafimeEngine(
                    config=EngineConfig(
                        backend=backend,
                        metric_names=metrics,
                        mi_approximate=mi_approximate,
                        mi_bins=mi_bins,
                    )
                ).analyze(Xl, yl, feature_names=names)
            )
        except Exception as exc:
            if os.environ.get(payload_env):
                raise AssertionError(
                    f"configured {backend} payload failed parity setup"
                ) from exc
            print(
                f"[{backend}] skipped (GPU absent): {type(exc).__name__}: {str(exc)[:50]}"
            )
            continue
        reference_combos = set(ref)
        backend_combos = set(r)
        missing_combos = reference_combos - backend_combos
        extra_combos = backend_combos - reference_combos
        missing_metrics = 0
        nonfinite_metrics = 0
        maxd = 0.0
        worst = None
        for combo in sorted(reference_combos & backend_combos):
            m = ref[combo]
            for k, v in m.items():
                if k not in r[combo]:
                    missing_metrics += 1
                    continue
                backend_value = r[combo][k]
                if not math.isfinite(v) or not math.isfinite(backend_value):
                    nonfinite_metrics += 1
                    continue
                delta = abs(v - backend_value)
                if delta > maxd:
                    maxd = delta
                    worst = (combo, k, v, backend_value)
        verdict = (
            "PASS"
            if maxd <= TOL
            and not missing_combos
            and not extra_combos
            and missing_metrics == 0
            and nonfinite_metrics == 0
            else "FAIL"
        )
        print(
            f"[{backend}] vs core: max|Δ|={maxd:.2e} "
            f"missing_combos={len(missing_combos)} extra_combos={len(extra_combos)} "
            f"missing_metrics={missing_metrics} nonfinite_metrics={nonfinite_metrics} "
            f"worst={worst} -> {verdict} (tol {TOL})"
        )
        if verdict != "PASS":
            failures.append(
                f"{backend}: max_delta={maxd:.6g} "
                f"missing_combos={len(missing_combos)} "
                f"extra_combos={len(extra_combos)} "
                f"missing_metrics={missing_metrics} "
                f"nonfinite_metrics={nonfinite_metrics}"
            )
    return failures


def main():
    failures = []
    failures.extend(run_case(EXACT_METRICS, mi_approximate=False))
    failures.extend(run_case(FIXED_BIN_METRICS, mi_approximate=True, mi_bins=96))
    for rows in (1152, 4608, 18432):
        failures.extend(
            run_case(
                ("mutual_info",),
                mi_approximate=True,
                mi_bins=96,
                rows=rows,
            )
        )
    print("\nrecord per-GPU parity; any FAIL blocks the 'flawless backend' claim.")
    if failures:
        raise AssertionError("backend parity failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
