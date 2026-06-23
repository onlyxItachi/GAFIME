"""dp_02 | RELEASE-NOTE EVIDENCE: logged OpenML tour. Baseline vs decision_path-
assisted (gated-soft) on LogReg + MLP across the dataset registry. Leakage-safe
(mine on TRAIN, materialize train+test). One telemetry artifact per (dataset,
model). This is the tour whose artifacts back the release notes.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python dp_02_openml_tour_logged.py
"""
import numpy as np
from sklearn.metrics import accuracy_score

import _measure_common as mc

GATE_K = 4


def run(name, loader):
    X, y, names, meta, _ = loader()
    Xtr, Xte, ytr, yte = mc.split(X, y)

    tel = mc.telemetry()
    rec = tel.new_record(worktree=mc.WORKTREE,
                         dataset=tel._default_dataset() | meta | {"split_policy": "0.3 holdout"},
                         config={"backend": "core", "gafime": {"family": "decision_path",
                                 "feature_variant": "gated_soft", "gated_k": GATE_K}})
    t0 = tel.monotonic_ns()
    with tel.span(rec, "gafime_planning_session_report"):
        cands, _ = mc.mine_candidates(Xtr, ytr, names)
    gen_ns = tel.monotonic_ns() - t0
    gated = cands[:GATE_K]
    with tel.span(rec, "gafime_to_downstream_transfer"):
        Ptr, Pte = mc.cols_soft(Xtr, gated), mc.cols_soft(Xte, gated)

    results = {}
    for mk in ("LogisticRegression", "MLPClassifier"):
        base = mc.make_model(mk).fit(Xtr, ytr)
        b = accuracy_score(yte, base.predict(Xte))
        with tel.span(rec, "downstream_fit"):
            asst = mc.make_model(mk).fit(np.hstack([Xtr, Ptr]), ytr)
        a = accuracy_score(yte, asst.predict(np.hstack([Xte, Pte])))
        results[mk] = (b, a)

    rec["results"].update({
        "status": "pass" if cands else "fail",
        "decision_path_count": len(cands),
        "gafime_feature_generation_ns": int(gen_ns),
        "per_model": {mk: {"baseline_score": round(b, 6), "gafime_score": round(a, 6),
                           "predictive_lift": round(a - b, 6)} for mk, (b, a) in results.items()},
        "top_candidates": [{"features": list(c.features),
                            "thresholds": [round(float(t), 4) for t in c.thresholds],
                            "signs": list(c.signs)} for c in gated],
    })
    tel.write_run(rec, mc.OUTDIR)
    return name, results


def main():
    print(f"{'dataset':<20}{'model':<20}{'base':>7}{'gafime':>8}{'lift':>8}")
    for name in ("synthetic_and", "diabetes", "credit-g", "blood-transfusion", "phoneme", "banknote"):
        try:
            _, res = run(name, mc.dataset_loader(name))
            for mk, (b, a) in res.items():
                print(f"{name:<20}{mk:<20}{b:>7.3f}{a:>8.3f}{a-b:>+8.3f}")
        except Exception as exc:  # log nothing extra; tour continues
            print(f"{name:<20}{'<error>':<20}{type(exc).__name__}: {str(exc)[:40]}")
    print(f"\nartifacts + index.csv in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
