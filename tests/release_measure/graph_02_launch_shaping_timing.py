"""graph_02 | CUDA/HIP graph launch-shaping timing.

Times a permutation-heavy run with graph=True vs graph=False on the same GPU;
the gap is launch-overhead removed by the backend payload. GPU-gated. Logged.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 graph_02_launch_shaping_timing.py
"""
import os

import gafime
from gafime import CompileFlags, EngineConfig

import _measure_common as mc

REPS = 5


def timed(Xl, yl, names, backend, graph):
    tel = mc.telemetry()
    best = None
    for _ in range(REPS):
        c = gafime.compile(Xl, yl, names, config=EngineConfig(backend=backend, permutation_tests=50),
                           flags=CompileFlags(plan=True, graph=graph))
        t0 = tel.monotonic_ns()
        c.analyze()
        dt = tel.monotonic_ns() - t0
        c.close()
        best = dt if best is None else min(best, dt)
    return best


def main():
    backend = os.environ.get("GAFIME_GRAPH_BACKEND", "cuda")
    X, y, names, meta, _ = mc.load_synthetic_and(n=6000, f=16)
    Xl, yl = X.tolist(), y.tolist()
    tel = mc.telemetry()
    try:
        plain = timed(Xl, yl, names, backend, False)
        graph = timed(Xl, yl, names, backend, True)
    except Exception as exc:
        print(f"[{backend}] skipped/error (needs that GPU): {type(exc).__name__}: {str(exc)[:80]}")
        return
    speedup = plain / graph if graph else float("nan")
    print(f"[{backend}] plain={plain/1e6:.2f}ms  graph={graph/1e6:.2f}ms  speedup={speedup:.3f}x")
    rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                         config={"backend": backend, "gafime": {"measure": "graph_launch_shaping"}})
    rec["results"].update({"status": "pass", "plain_launch_ns": int(plain),
                           "graph_replay_ns": int(graph), "graph_speedup": round(speedup, 4)})
    tel.write_run(rec, mc.OUTDIR)
    print(f"logged real speedup (may be ~1.0 — no claim beyond the artifact). {mc.OUTDIR}")


if __name__ == "__main__":
    main()
