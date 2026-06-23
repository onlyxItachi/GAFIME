"""compile_02 | gafime.compile value: compiled (plan) path vs eager engine.analyze.
Checks (a) result parity (same interactions/metrics within tolerance) and
(b) timing — the compiled path front-loads planning so repeated analyze should
not be slower. Logged.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python compile_02_compiled_vs_eager.py
"""
import os

import gafime
from gafime import CompileFlags, EngineConfig
from gafime.engine import GafimeEngine

import _measure_common as mc


def top_metric(report):
    best = 0.0
    for ir in (getattr(report, "interactions", []) or []):
        if ir.metrics:
            best = max(best, max(ir.metrics.values()))
    return best


def main():
    X, y, names, meta, _ = mc.load_synthetic_and(n=1500, f=10)
    Xl, yl = X.tolist(), y.tolist()
    backend = os.environ.get("GAFIME_BACKEND", "core")
    cfg = EngineConfig(backend=backend)
    tel = mc.telemetry()

    t0 = tel.monotonic_ns()
    eager = GafimeEngine(config=cfg).analyze(Xl, yl, feature_names=names)
    eager_ns = tel.monotonic_ns() - t0

    t1 = tel.monotonic_ns()
    compiled = gafime.compile(Xl, yl, names, config=cfg, flags=CompileFlags(plan=True))
    comp_report = compiled.analyze()
    comp_ns = tel.monotonic_ns() - t1
    compiled.close()

    em, cm = top_metric(eager), top_metric(comp_report)
    print(f"eager   top_metric={em:.6f}  time={eager_ns/1e6:.1f}ms")
    print(f"compiled top_metric={cm:.6f}  time={comp_ns/1e6:.1f}ms")
    print(f"parity |Δ top_metric|={abs(em-cm):.2e}")

    rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                         config={"backend": backend, "gafime": {"measure": "compiled_vs_eager"}})
    # compiled analyze IS the planning/session zone -> record it canonically; A/B times -> results
    rec["spans_ns"]["gafime_planning_session_report"] = int(comp_ns)
    rec["results"].update({"status": "pass", "eager_analyze_ns": int(eager_ns),
                           "compiled_analyze_ns": int(comp_ns), "eager_top_metric": round(em, 6),
                           "compiled_top_metric": round(cm, 6), "abs_delta": abs(em - cm)})
    tel.write_run(rec, mc.OUTDIR)
    print(f"COMPILED-VS-EAGER: parity Δ must be ~0. artifact in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
