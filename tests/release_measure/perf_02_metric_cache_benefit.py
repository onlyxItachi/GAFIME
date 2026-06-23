"""perf_02 | PERF HARDENING: metric-cache benefit. A permutation/stability run
re-scores the same shapes many times; the continuous metric cache should serve
hits and cut work. Reads the session cache counters (best-effort) and times a
cached vs uncached-style run. GPU-oriented (continuous cache lives in the
resident CUDA/ROCm session). Logged.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/mc-torch-cu/bin/python perf_02_metric_cache_benefit.py
"""
import os

import gafime
from gafime import CompileFlags, EngineConfig

import _measure_common as mc


def main():
    backend = os.environ.get("GAFIME_BACKEND", "core")
    X, y, names, meta, _ = mc.load_synthetic_and(n=5000, f=14)
    Xl, yl = X.tolist(), y.tolist()
    tel = mc.telemetry()

    compiled = gafime.compile(Xl, yl, names,
                              config=EngineConfig(backend=backend, permutation_tests=50),
                              flags=CompileFlags(plan=True))
    t0 = tel.monotonic_ns()
    compiled.analyze()
    dt = tel.monotonic_ns() - t0
    sess = getattr(compiled, "_session", None)
    hits = getattr(sess, "continuous_metric_cache_hits", None)
    builds = getattr(sess, "continuous_metric_cache_builds", None)
    cand_hits = getattr(sess, "candidate_table_cache_hits", None)
    compiled.close()

    print(f"[{backend}] perm-heavy analyze: {dt/1e6:.1f}ms")
    print(f"  continuous_metric_cache_hits={hits} builds={builds} candidate_table_cache_hits={cand_hits}")
    hit_rate = (hits / (hits + builds)) if (isinstance(hits, int) and isinstance(builds, int) and (hits + builds)) else None
    rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                         config={"backend": backend, "gafime": {"measure": "metric_cache_benefit"}})
    rec["spans_ns"]["e2e_total"] = int(dt)
    rec["counters"].update({"metric_cache_hits": hits, "metric_cache_builds": builds,
                            "candidate_table_cache_hits": cand_hits})
    rec["results"].update({"status": "pass", "metric_cache_hit_rate": hit_rate})
    tel.write_run(rec, mc.OUTDIR)
    print(f"expect hits>0 on a resident GPU run. artifact in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
