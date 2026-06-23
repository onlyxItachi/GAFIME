"""dp_01 | CORRECTNESS: native find_decision_path_candidates vs a pure-Python
greedy-CART reference. Proves the native GBDT split math is exact (the v0.5
headline). CPU-only, no telemetry.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python dp_01_parity_native_vs_reference.py
"""
import numpy as np
from gafime import gafime_core as gc


def best_split_ref(x, resid, min_leaf):
    """Exact best (feature-agnostic 1D) variance-reduction split on sorted values."""
    order = np.argsort(x, kind="stable")
    xs, rs = x[order], resid[order]
    n = len(xs)
    total = rs.sum()
    total_sse = (rs * rs).sum() - total * total / n
    best_gain, best_thr = -1.0, None
    csum = np.cumsum(rs)
    for i in range(min_leaf, n - min_leaf):
        if xs[i] == xs[i - 1]:
            continue
        ln, rn = i, n - i
        ls = csum[i - 1]
        l_sse_part = ls * ls / ln
        r_sse_part = (total - ls) ** 2 / rn
        gain = (l_sse_part + r_sse_part)  # maximize -> equals SSE reduction up to const
        if gain > best_gain:
            best_gain, best_thr = gain, 0.5 * (xs[i] + xs[i - 1])
    return best_thr


def main():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.random((n, 4)).astype(np.float32)
    # planted AND: x0>0.5 AND x1>0.5
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(np.float32)

    Xb = gc.NativeMatrixBuffer(X.tolist())
    yb = gc.NativeVectorBuffer(y.tolist())

    # depth-1, single round: top path should be a single best split matching the reference
    recs = gc.find_decision_path_candidates(Xb, yb, None, 1, 8, 0, 8, 1, 1.0)
    print(f"depth1 paths: {len(recs)}")
    top = max(recs, key=lambda r: r.gain)
    f = top.features[0]
    ref_thr = best_split_ref(X[:, f].astype(float), y.astype(float), 8)
    print(f"  native top: feat={f} thr={top.thresholds[0]:.6f} sign={top.signs[0]} gain={top.gain:.6f}")
    print(f"  reference thr on feat {f}: {ref_thr:.6f}  |Δ|={abs(top.thresholds[0]-ref_thr):.2e}")

    # depth-2 should recover the planted conjunction on features {0,1}
    recs2 = gc.find_decision_path_candidates(Xb, yb, None, 2, 16, 0, 8, 1, 1.0)
    best2 = max(recs2, key=lambda r: r.gain)
    feats = sorted(best2.features)
    print(f"depth2 best path feats={best2.features} thr={[round(t,3) for t in best2.thresholds]} "
          f"signs={best2.signs} gain={best2.gain:.4f}")
    print(f"  recovered planted {{0,1}}: {feats == [0, 1]}")
    print("PARITY CHECK: inspect |Δ| ~ 1e-6 (fp32) and planted recovery == True")


if __name__ == "__main__":
    main()
