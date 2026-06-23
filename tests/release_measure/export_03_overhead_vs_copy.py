"""export_03 | FRAMEWORK EXPORT value: zero-copy vs copying out. Times handing the
feature matrix to a framework three ways at increasing size:
  (a) zero-copy  np.from_dlpack(buf)
  (b) buffer-protocol np.asarray(buf)
  (c) Python-copy  np.asarray(buf.rows())   <- the old/wrong "export"
Proves (a)/(b) are ~O(1) vs (c) growing with data. Logged.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python export_03_overhead_vs_copy.py
"""
import numpy as np
from gafime import gafime_core as gc

import _measure_common as mc

SIZES = [(1000, 16), (10000, 32), (50000, 64)]
REPS = 20


def time_ns(fn, reps=REPS):
    tel = mc.telemetry()
    best = None
    for _ in range(reps):
        t0 = tel.monotonic_ns()
        fn()
        dt = tel.monotonic_ns() - t0
        best = dt if best is None else min(best, dt)
    return best


def main():
    if not hasattr(gc.NativeMatrixBuffer([[1.0]]), "__dlpack__"):
        print("FAIL: __dlpack__ missing -> export commit not merged/rebuilt")
        return
    tel = mc.telemetry()
    print(f"{'rows':>7}{'feat':>6}{'dlpack_us':>12}{'asarray_us':>12}{'pycopy_us':>12}")
    for n, f in SIZES:
        X = np.random.default_rng(0).random((n, f)).astype(np.float32).tolist()
        buf = gc.NativeMatrixBuffer(X)
        dl = time_ns(lambda: np.from_dlpack(buf))
        aa = time_ns(lambda: np.asarray(buf))
        cp = time_ns(lambda: np.asarray(buf.rows()))
        print(f"{n:>7}{f:>6}{dl/1e3:>12.2f}{aa/1e3:>12.2f}{cp/1e3:>12.2f}")
        rec = tel.new_record(worktree=mc.WORKTREE,
                             dataset=tel._default_dataset() | {"source": "synthetic",
                                     "name": "export_overhead", "rows": n, "features": f},
                             config={"backend": "core", "gafime": {"measure": "export_overhead"}})
        rec["spans_ns"].update({"export_dlpack_ns": int(dl), "export_asarray_ns": int(aa),
                                "export_pycopy_ns": int(cp)})
        rec["results"]["status"] = "pass"
        tel.write_run(rec, mc.OUTDIR)
    print(f"\nexpect: dlpack/asarray ~flat with size; pycopy grows. artifacts in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
