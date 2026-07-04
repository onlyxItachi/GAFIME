"""dp_03 | METHOD EFFECT: isolate the indicator method. For each dataset x model,
compare baseline vs {all_hard, gated_hard, gated_soft}. Confirms the MSG-57
finding: gating reduces harm on real data; soft helps the MLP where structure
exists. Logged per (dataset, model, variant).

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 dp_03_method_effect_gated_soft.py
"""
import numpy as np
from sklearn.metrics import accuracy_score

import _measure_common as mc

GATE_K = 4


def run(name, loader):
    X, y, names, meta, _ = loader()
    Xtr, Xte, ytr, yte = mc.split(X, y)
    cands, _ = mc.mine_candidates(Xtr, ytr, names)
    gated = cands[:GATE_K]
    variants = {
        "all_hard": (mc.cols_hard(Xtr, cands), mc.cols_hard(Xte, cands)),
        "gated_hard": (mc.cols_hard(Xtr, gated), mc.cols_hard(Xte, gated)),
        "gated_soft": (mc.cols_soft(Xtr, gated), mc.cols_soft(Xte, gated)),
    }
    out = {}
    for mk in ("LogisticRegression", "MLPClassifier"):
        base = accuracy_score(yte, mc.make_model(mk).fit(Xtr, ytr).predict(Xte))
        out[(mk, "baseline")] = base
        for v, (Ptr, Pte) in variants.items():
            m = mc.make_model(mk).fit(np.hstack([Xtr, Ptr]), ytr)
            sc = accuracy_score(yte, m.predict(np.hstack([Xte, Pte])))
            out[(mk, v)] = sc
            tel = mc.telemetry()
            rec = tel.new_record(worktree=mc.WORKTREE,
                                 dataset=tel._default_dataset() | meta,
                                 config={"backend": "core", "gafime": {"family": "decision_path",
                                         "feature_variant": v, "gated_k": GATE_K},
                                         "downstream_model": {"name": mk}})
            rec["results"].update({"status": "pass", "baseline_score": round(base, 6),
                                   "gafime_score": round(sc, 6), "predictive_lift": round(sc - base, 6)})
            tel.write_run(rec, mc.OUTDIR)
    return name, out


def main():
    print(f"{'dataset':<18}{'model':<20}{'base':>7}{'all_hard':>10}{'gated_hard':>12}{'gated_soft':>12}")
    for name in ("synthetic_and", "diabetes", "blood-transfusion", "credit-g", "friedman1"):
        try:
            _, r = run(name, mc.dataset_loader(name))
            for mk in ("LogisticRegression", "MLPClassifier"):
                b = r[(mk, "baseline")]
                print(f"{name:<18}{mk:<20}{b:>7.3f}"
                      f"{r[(mk,'all_hard')]-b:>+10.3f}{r[(mk,'gated_hard')]-b:>+12.3f}"
                      f"{r[(mk,'gated_soft')]-b:>+12.3f}")
        except Exception as exc:
            print(f"{name:<18}{type(exc).__name__}: {str(exc)[:40]}")
    print(f"\n(values after base are LIFT; artifacts in {mc.OUTDIR})")


if __name__ == "__main__":
    main()
