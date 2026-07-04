"""perf_01 | PERF HARDENING: resident session benefit. Compiling once and reusing
the resident session across repeated analyze should beat re-compiling per run
(the matrix stays resident; on GPU it stays on-device). Times reuse vs fresh.
GPU-preferred (set backend=cuda); meaningful but smaller on core. Logged.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 perf_01_residency_session_benefit.py
"""
import os

import gafime
from gafime import CompileFlags, EngineConfig
from gafime import GafimeEngine

import _measure_common as mc

REPS = 10


def main():
    backend = os.environ.get("GAFIME_BACKEND", "core")
    X, y, names, meta, _ = mc.load_synthetic_and(n=4000, f=12)
    Xl, yl = X.tolist(), y.tolist()
    tel = mc.telemetry()
    cfg = EngineConfig(backend=backend)

    # fresh: full pipeline every call
    t0 = tel.monotonic_ns()
    for _ in range(REPS):
        GafimeEngine(config=cfg).analyze(Xl, yl, feature_names=names)
    fresh = (tel.monotonic_ns() - t0) / REPS

    # resident: compile once, reuse session
    compiled = gafime.compile(Xl, yl, names, config=cfg, flags=CompileFlags(plan=True))
    t1 = tel.monotonic_ns()
    for _ in range(REPS):
        compiled.analyze()
    resident = (tel.monotonic_ns() - t1) / REPS
    compiled.close()

    speedup = fresh / resident if resident else float("nan")
    print(f"[{backend}] fresh/call={fresh/1e6:.2f}ms  resident/call={resident/1e6:.2f}ms  "
          f"speedup={speedup:.3f}x")
    rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                         config={"backend": backend, "gafime": {"measure": "residency_benefit"}})
    rec["results"].update({"status": "pass", "fresh_per_call_ns": int(fresh),
                           "resident_per_call_ns": int(resident), "residency_speedup": round(speedup, 4)})
    tel.write_run(rec, mc.OUTDIR)
    print(f"artifact in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
