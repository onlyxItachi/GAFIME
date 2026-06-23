"""perf_03 | THE release-notes timing artifact: a single representative end-to-end
run with EVERY canonical span filled (codex MSG 65 item 4) — e2e, OpenML
load/preprocess, GAFIME planning/session/report, GAFIME C++ core, GAFIME->
framework transfer, downstream fit, and Python/GIL overhead. One clean JSON the
release notes can cite for the time-share story.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python perf_03_telemetry_e2e_spans.py
"""
import numpy as np
from sklearn.metrics import accuracy_score

import _measure_common as mc

GATE_K = 4


def main():
    tel = mc.telemetry()
    rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset(),
                         config={"backend": "core", "gafime": {"family": "decision_path",
                                 "measure": "e2e_spans", "feature_variant": "gated_soft"}})
    t_e2e = tel.monotonic_ns()
    with tel.span(rec, "openml_load_preprocess"):
        X, y, names, meta, _ = mc.load_openml(37, "diabetes")
    rec["dataset"] = tel._default_dataset() | meta | {"split_policy": "0.3 holdout"}
    Xtr, Xte, ytr, yte = mc.split(X, y)

    with tel.span(rec, "gafime_planning_session_report"):
        cands, _ = mc.mine_candidates(Xtr, ytr, names)
    gated = cands[:GATE_K]

    # isolated native C++ core cost (representative gafime_cpp_core span)
    from gafime import gafime_core as gc
    Xb = gc.NativeMatrixBuffer(Xtr.astype(np.float32).tolist())
    yb = gc.NativeVectorBuffer(ytr.astype(np.float32).tolist())
    with tel.span(rec, "gafime_cpp_core"):
        gc.find_decision_path_candidates(Xb, yb, None, 3, 16, 0, 8, 20, 0.3)

    with tel.span(rec, "gafime_to_downstream_transfer"):
        Ptr, Pte = mc.cols_soft(Xtr, gated), mc.cols_soft(Xte, gated)

    base = accuracy_score(yte, mc.make_model("MLPClassifier").fit(Xtr, ytr).predict(Xte))
    with tel.span(rec, "downstream_fit"):
        asst_model = mc.make_model("MLPClassifier").fit(np.hstack([Xtr, Ptr]), ytr)
    asst = accuracy_score(yte, asst_model.predict(np.hstack([Xte, Pte])))

    e2e = tel.monotonic_ns() - t_e2e
    rec["spans_ns"]["e2e_total"] = e2e
    planning = rec["spans_ns"].get("gafime_planning_session_report") or 0
    accounted = sum(v for k, v in rec["spans_ns"].items() if k != "e2e_total" and isinstance(v, int))
    rec["spans_ns"]["python_orchestration_gil"] = max(0, e2e - accounted)
    rec["results"].update({"status": "pass", "baseline_score": round(base, 6),
                           "gafime_score": round(asst, 6), "predictive_lift": round(asst - base, 6),
                           "gafime_time_share": round(planning / e2e, 4) if e2e else None,
                           "decision_path_count": len(cands)})
    json_path, _ = tel.write_run(rec, mc.OUTDIR)
    print("e2e span breakdown (ms):")
    for k, v in rec["spans_ns"].items():
        if isinstance(v, int):
            print(f"  {k:<34}{v/1e6:>10.2f}")
    print(f"lift={asst-base:+.3f}  gafime_time_share={rec['results']['gafime_time_share']}")
    print(f"artifact: {json_path}")


if __name__ == "__main__":
    main()
