"""dp_08 | METHODOLOGY GUARD: leakage safety. The honest protocol mines specs on
TRAIN only and materializes them on TEST. This compares honest (mine-on-train)
vs cheating (mine-on-train+test) test accuracy; the cheating number should be
>= honest, and the GAP quantifies the leakage we avoid. Protects every other
script's numbers.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python dp_08_leakage_safety.py
"""
import numpy as np
from sklearn.metrics import accuracy_score

import _measure_common as mc

GATE_K = 4


def assisted_acc(Xtr, ytr, Xte, yte, cands):
    gated = cands[:GATE_K]
    Ptr, Pte = mc.cols_soft(Xtr, gated), mc.cols_soft(Xte, gated)
    m = mc.make_model("MLPClassifier").fit(np.hstack([Xtr, Ptr]), ytr)
    return accuracy_score(yte, m.predict(np.hstack([Xte, Pte])))


def main():
    print(f"{'dataset':<18}{'honest(train-mined)':>20}{'cheat(all-mined)':>18}{'leakage_gap':>13}")
    for name in ("diabetes", "credit-g", "phoneme"):
        try:
            X, y, names, meta, _ = mc.dataset_loader(name)()
            Xtr, Xte, ytr, yte = mc.split(X, y)
            honest_cands, _ = mc.mine_candidates(Xtr, ytr, names)          # TRAIN only (correct)
            cheat_cands, _ = mc.mine_candidates(np.vstack([Xtr, Xte]),
                                                np.concatenate([ytr, yte]), names)  # leaks test
            honest = assisted_acc(Xtr, ytr, Xte, yte, honest_cands)
            cheat = assisted_acc(Xtr, ytr, Xte, yte, cheat_cands)
            print(f"{name:<18}{honest:>20.3f}{cheat:>18.3f}{cheat-honest:>+13.3f}")
        except Exception as exc:
            print(f"{name:<18}{type(exc).__name__}: {str(exc)[:40]}")
    print("\nrelease numbers MUST use the honest (train-mined) column.")


if __name__ == "__main__":
    main()
