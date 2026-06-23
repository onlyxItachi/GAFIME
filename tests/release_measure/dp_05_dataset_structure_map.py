"""dp_05 | WHERE IT PAYS: map decision_path lift against dataset structure
(rich / mixed / poor threshold-conjunction structure). Charts the honest claim
"helps where structure exists, not universal". MLP + gated_soft (the recipe that
helps). Logged per dataset with the structure tag.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python dp_05_dataset_structure_map.py
"""
import numpy as np
from sklearn.metrics import accuracy_score

import _measure_common as mc

GATE_K = 4


def run(name, structure, loader):
    X, y, names, meta, _ = loader()
    Xtr, Xte, ytr, yte = mc.split(X, y)
    cands, _ = mc.mine_candidates(Xtr, ytr, names)
    gated = cands[:GATE_K]
    Ptr, Pte = mc.cols_soft(Xtr, gated), mc.cols_soft(Xte, gated)
    base = accuracy_score(yte, mc.make_model("MLPClassifier").fit(Xtr, ytr).predict(Xte))
    asst = accuracy_score(yte, mc.make_model("MLPClassifier")
                          .fit(np.hstack([Xtr, Ptr]), ytr).predict(np.hstack([Xte, Pte])))
    tel = mc.telemetry()
    rec = tel.new_record(worktree=mc.WORKTREE,
                         dataset=tel._default_dataset() | meta | {"structure_tag": structure},
                         config={"backend": "core", "gafime": {"family": "decision_path",
                                 "feature_variant": "gated_soft", "downstream_model": "MLP"}})
    rec["results"].update({"status": "pass", "baseline_score": round(base, 6),
                           "gafime_score": round(asst, 6), "predictive_lift": round(asst - base, 6),
                           "structure_tag": structure})
    tel.write_run(rec, mc.OUTDIR)
    return name, structure, base, asst - base


def main():
    items = [("synthetic_and", "rich"), ("banknote", "rich"), ("phoneme", "rich"),
             ("diabetes", "rich"), ("ilpd", "mixed"), ("credit-g", "mixed"),
             ("blood-transfusion", "poor")]
    out = []
    for name, tag in items:
        try:
            out.append(run(name, tag, mc.dataset_loader(name)))
        except Exception as exc:
            print(f"{name:<18}{type(exc).__name__}: {str(exc)[:40]}")
    print(f"{'dataset':<18}{'structure':<10}{'MLP base':>9}{'lift(gated_soft)':>18}")
    for name, tag, base, lift in sorted(out, key=lambda r: -r[3]):
        print(f"{name:<18}{tag:<10}{base:>9.3f}{lift:>+18.3f}")
    print(f"\nexpect: lift concentrates on 'rich'; ~0/negative on 'poor'. artifacts in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
