"""backend_03 | per-backend END-TO-END smoke + telemetry: run a full analyze on
each available backend, confirm it produces results, and log an artifact tagged
by backend so the release notes can show coverage. GPU rows skip if absent.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 backend_03_e2e_smoke_per_backend.py
"""
from gafime.config import EngineConfig
from gafime import GafimeEngine

import _measure_common as mc


def main():
    X, y, names, meta, _ = mc.load_synthetic_and(n=2000, f=8)
    Xl, yl = X.tolist(), y.tolist()
    tel = mc.telemetry()
    for backend in ("core", "cuda", "rocm", "metal"):
        rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                             config={"backend": backend, "gafime": {"measure": "backend_e2e_smoke"}})
        try:
            with tel.span(rec, "e2e_total"):
                report = GafimeEngine(config=EngineConfig(backend=backend)).analyze(
                    Xl, yl, feature_names=names)
            n_inter = len(list(getattr(report, "interactions", []) or []))
            rec["results"].update({"status": "pass" if n_inter else "fail",
                                   "decision_path_count": n_inter})
            print(f"[{backend:<6}] e2e ok, interactions={n_inter}")
        except Exception as exc:
            tel.mark_failed(rec, exc)
            print(f"[{backend:<6}] skipped/failed: {type(exc).__name__}: {str(exc)[:50]}")
        tel.write_run(rec, mc.OUTDIR)
    print(f"\nper-backend artifacts in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
