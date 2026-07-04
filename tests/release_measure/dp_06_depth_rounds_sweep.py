"""dp_06 | CAPACITY KNOBS: sweep decision_path_max_depth {1,2,3} x rounds
{1,5,20}. Reports MLP gated_soft lift, native split-find time, and #paths so we
can pick release defaults with cost in view. Logged per cell.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 dp_06_depth_rounds_sweep.py
"""
import numpy as np
from sklearn.metrics import accuracy_score

import _measure_common as mc

DEPTHS, ROUNDS, GATE_K = [1, 2, 3], [1, 5, 20], 4


def run(name, loader):
    X, y, names, meta, _ = loader()
    Xtr, Xte, ytr, yte = mc.split(X, y)
    base = accuracy_score(yte, mc.make_model("MLPClassifier").fit(Xtr, ytr).predict(Xte))
    tel = mc.telemetry()
    grid = []
    for d in DEPTHS:
        for r in ROUNDS:
            t0 = tel.monotonic_ns()
            cands, _ = mc.mine_candidates(Xtr, ytr, names,
                                          {"decision_path_max_depth": d, "decision_path_rounds": r})
            gen_ns = tel.monotonic_ns() - t0
            gated = cands[:GATE_K]
            Ptr, Pte = mc.cols_soft(Xtr, gated), mc.cols_soft(Xte, gated)
            asst = accuracy_score(yte, mc.make_model("MLPClassifier")
                                  .fit(np.hstack([Xtr, Ptr]), ytr).predict(np.hstack([Xte, Pte])))
            grid.append((d, r, len(cands), gen_ns / 1e6, asst - base))
            rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                                 config={"backend": "core", "gafime": {"family": "decision_path",
                                         "decision_path_max_depth": d, "decision_path_rounds": r}})
            rec["spans_ns"]["gafime_planning_session_report"] = int(gen_ns)
            rec["results"].update({"status": "pass", "baseline_score": round(base, 6),
                                   "predictive_lift": round(asst - base, 6),
                                   "decision_path_count": len(cands)})
            tel.write_run(rec, mc.OUTDIR)
    return name, base, grid


def main():
    for name in ("diabetes", "friedman1"):
        try:
            _, base, grid = run(name, mc.dataset_loader(name))
            print(f"\n== {name} (MLP base {base:.3f}) ==")
            print(f"{'depth':>6}{'rounds':>8}{'paths':>7}{'gen_ms':>9}{'lift':>8}")
            for d, r, n, ms, lift in grid:
                print(f"{d:>6}{r:>8}{n:>7}{ms:>9.1f}{lift:>+8.3f}")
        except Exception as exc:
            print(f"{name}: {type(exc).__name__}: {str(exc)[:40]}")
    print(f"\nartifacts in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
