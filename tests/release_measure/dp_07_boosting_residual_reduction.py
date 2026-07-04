"""dp_07 | CORRECTNESS: residual boosting adds signal. As rounds increase, the
native miner should surface more distinct paths and a downstream linear model on
the indicators should fit TRAIN better (diminishing returns). Confirms the
boosting loop works as intended (not just round-1 repeated).

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 dp_07_boosting_residual_reduction.py
"""
import numpy as np
from sklearn.linear_model import LinearRegression

import _measure_common as mc


def main():
    X, y, names, meta, _ = mc.load_friedman1()
    rng = np.random.default_rng(0)
    print(f"{'rounds':>7}{'unique_paths':>14}{'train_R2(indicators->y)':>26}")
    prev_paths = -1
    for r in (1, 2, 3, 5, 10, 20):
        cands, _ = mc.mine_candidates(X, y, names,
                                      {"decision_path_rounds": r, "decision_path_max_paths": 64})
        P = mc.cols_soft(X, cands[:32])
        if P.shape[1] == 0:
            print(f"{r:>7}{0:>14}{'n/a':>26}")
            continue
        r2 = LinearRegression().fit(P, y).score(P, y)
        print(f"{r:>7}{len(cands):>14}{r2:>26.4f}")
        assert len(cands) >= prev_paths, "more rounds should not reduce the candidate pool"
        prev_paths = len(cands)
    print("\nexpect: unique_paths non-decreasing, train R2 rising then plateauing (boosting works)")


if __name__ == "__main__":
    main()
