"""perf_01 | PERF HARDENING: resident session benefit.

Times three public shapes on the same data:
- uncached public analyze: full compile/upload per call
- cached public analyze: normal GafimeEngine.analyze() with the v1 resident
  analyze cache enabled
- explicit resident artifact: gafime.compile(...).analyze()

GPU-preferred (set backend=cuda); meaningful but smaller on core. Logged.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 perf_01_residency_session_benefit.py
"""
import os
from contextlib import contextmanager

import gafime
from gafime import CompileFlags, EngineConfig
from gafime import GafimeEngine

import _measure_common as mc

REPS = 10


@contextmanager
def analyze_cache_size(value: str | None):
    old = os.environ.get("GAFIME_V1_ANALYZE_CACHE_SIZE")
    if value is None:
        os.environ.pop("GAFIME_V1_ANALYZE_CACHE_SIZE", None)
    else:
        os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GAFIME_V1_ANALYZE_CACHE_SIZE", None)
        else:
            os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = old


def main():
    backend = os.environ.get("GAFIME_BACKEND", "core")
    X, y, names, meta, _ = mc.load_synthetic_and(n=4000, f=12)
    Xl, yl = X.tolist(), y.tolist()
    tel = mc.telemetry()
    cfg = EngineConfig(backend=backend)

    # uncached: full pipeline every call
    with analyze_cache_size("0"):
        t0 = tel.monotonic_ns()
        for _ in range(REPS):
            GafimeEngine(config=cfg).analyze(Xl, yl, feature_names=names)
        uncached = (tel.monotonic_ns() - t0) / REPS

    # normal public API: first call compiles, later calls reuse the resident
    # cache when config + fp32 feature content match.
    with analyze_cache_size(None):
        GafimeEngine(config=cfg).analyze(Xl, yl, feature_names=names)
        t_cache = tel.monotonic_ns()
        for _ in range(REPS):
            GafimeEngine(config=cfg).analyze(Xl, yl, feature_names=names)
        cached = (tel.monotonic_ns() - t_cache) / REPS

    # resident: compile once, reuse session
    compiled = gafime.compile(Xl, yl, names, config=cfg, flags=CompileFlags(plan=True))
    t1 = tel.monotonic_ns()
    for _ in range(REPS):
        compiled.analyze()
    resident = (tel.monotonic_ns() - t1) / REPS
    compiled.close()

    cache_speedup = uncached / cached if cached else float("nan")
    resident_speedup = uncached / resident if resident else float("nan")
    print(
        f"[{backend}] uncached/call={uncached/1e6:.2f}ms  "
        f"public-cached/call={cached/1e6:.2f}ms  resident/call={resident/1e6:.2f}ms  "
        f"cache_speedup={cache_speedup:.3f}x resident_speedup={resident_speedup:.3f}x"
    )
    rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                         config={"backend": backend, "gafime": {"measure": "residency_benefit"}})
    rec["results"].update({"status": "pass", "uncached_per_call_ns": int(uncached),
                           "public_cached_per_call_ns": int(cached),
                           "resident_per_call_ns": int(resident),
                           "public_cache_speedup": round(cache_speedup, 4),
                           "residency_speedup": round(resident_speedup, 4)})
    tel.write_run(rec, mc.OUTDIR)
    print(f"artifact in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
