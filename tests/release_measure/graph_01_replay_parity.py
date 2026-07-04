"""graph_01 | CUDA/HIP GRAPH parity: capturing+replaying the launch sequence must
not change results. Compares interaction metrics with graph=True vs graph=False
on the same data (within fp atomic-ordering tolerance). GPU-gated: requires a
CUDA or ROCm backend; skips on core.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 graph_01_replay_parity.py   # CUDA
"""
import gafime
from gafime import CompileFlags, EngineConfig

import _measure_common as mc

TOL = 1e-4  # fp32 atomic-ordering noise budget (per graph-track doc ~1e-6..1e-4)


def metrics_vector(report):
    out = []
    for ir in (getattr(report, "interactions", []) or []):
        out.append((tuple(ir.combo), tuple(sorted((k, round(float(v), 6)) for k, v in ir.metrics.items()))))
    return sorted(out)


def run(backend):
    X, y, names, meta, _ = mc.load_synthetic_and(n=4000, f=12)
    Xl, yl = X.tolist(), y.tolist()
    cfg = dict(backend=backend, permutation_tests=20)  # repeated same-shape launches exercise graph
    base = gafime.compile(Xl, yl, names, config=EngineConfig(**cfg),
                          flags=CompileFlags(plan=True, graph=False))
    g = gafime.compile(Xl, yl, names, config=EngineConfig(**cfg),
                       flags=CompileFlags(plan=True, graph=True))
    rb, rg = base.analyze(), g.analyze()
    base.close(); g.close()

    vb, vg = metrics_vector(rb), metrics_vector(rg)
    if len(vb) != len(vg):
        print(f"[{backend}] FAIL: result count differs {len(vb)} vs {len(vg)}")
        return
    maxd = 0.0
    for (cb, mb), (cgc, mg) in zip(vb, vg):
        for (_, a), (_, b) in zip(mb, mg):
            maxd = max(maxd, abs(a - b))
    print(f"[{backend}] results={len(vb)} max|Δmetric| graph-vs-plain = {maxd:.2e} "
          f"-> {'PASS' if maxd <= TOL else 'FAIL'} (tol {TOL})")


def main():
    import os
    backend = os.environ.get("GAFIME_GRAPH_BACKEND", "cuda")  # set to "rocm" on the AMD box
    try:
        run(backend)
    except Exception as exc:
        print(f"[{backend}] skipped/error (needs that GPU): {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
