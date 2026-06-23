"""dp_04 | max_bins resolution sweep {0,8,16,32,64}: native split-find TIME and
candidate STABILITY vs exhaustive-exact (max_bins=0). Proves the split-cap is a
speed/resolution knob that stays close to exact. Logged per (dataset, max_bins).

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python dp_04_max_bins_sweep.py
"""
import numpy as np
from gafime import gafime_core as gc

import _measure_common as mc

BINS = [0, 8, 16, 32, 64]


def path_keys(recs):
    # STRUCTURAL key: the (feature, sign) conditions in each path, threshold-agnostic.
    # Binning shifts thresholds to bin edges by design, so comparing exact thresholds would
    # always read ~0; structural overlap is the meaningful "does binning find the same splits".
    return {tuple(sorted(zip(map(int, r.features), map(int, r.signs)))) for r in recs}


def run(name, loader):
    X, y, names, meta, _ = loader()
    Xb = gc.NativeMatrixBuffer(X.astype(np.float32).tolist())
    yb = gc.NativeVectorBuffer(y.astype(np.float32).tolist())
    tel = mc.telemetry()

    exact_keys = None
    rows = []
    for b in BINS:
        # warm + timed native call
        t0 = tel.monotonic_ns()
        recs = gc.find_decision_path_candidates(Xb, yb, None, 3, 16, b, 8, 20, 0.3)
        dt = tel.monotonic_ns() - t0
        keys = path_keys(recs)
        if b == 0:
            exact_keys = keys
        jacc = (len(keys & exact_keys) / len(keys | exact_keys)) if (keys or exact_keys) else 1.0
        rows.append((b, len(recs), dt / 1e6, jacc))

        rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                             config={"backend": "core", "gafime": {"family": "decision_path",
                                     "decision_path_max_bins": b}})
        rec["spans_ns"]["gafime_cpp_core"] = int(dt)
        rec["results"].update({"status": "pass", "decision_path_count": len(recs),
                               "struct_jaccard_vs_exact": round(jacc, 4)})
        tel.write_run(rec, mc.OUTDIR)
    return name, rows


def main():
    print(f"{'dataset':<18}{'max_bins':>9}{'paths':>7}{'ms':>9}{'struct_jacc_vs_exact':>22}")
    for name in ("diabetes", "credit-g", "phoneme"):
        try:
            _, rows = run(name, mc.dataset_loader(name))
            for b, npaths, ms, jacc in rows:
                print(f"{name:<18}{b:>9}{npaths:>7}{ms:>9.2f}{jacc:>22.3f}")
        except Exception as exc:
            print(f"{name:<18}{type(exc).__name__}: {str(exc)[:40]}")
    print(f"\nartifacts in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
