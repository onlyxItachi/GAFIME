"""export_02 | FRAMEWORK EXPORT lifetime safety (the framework-grade gate). A
borrowed tensor must keep GAFIME's memory alive after the owning buffer is
dropped, and unconsumed capsules must free cleanly. Includes a stress loop to
shake out leaks/use-after-free. Requires export commit merged + rebuilt.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python export_02_lifetime_safety.py
"""
import gc as _gc

import numpy as np
from gafime import gafime_core as gc


def main():
    if not hasattr(gc.NativeVectorBuffer([1.0]), "__dlpack__"):
        print("FAIL: __dlpack__ missing -> export commit not merged/rebuilt")
        return

    # 1) owner dropped while tensor borrows it
    vals = [11.0, 22.0, 33.0]
    buf = gc.NativeVectorBuffer(vals)
    t = np.from_dlpack(buf)
    del buf
    _gc.collect()
    print(f"owner-dropped tensor intact: {t.tolist() == vals}")
    del t
    _gc.collect()
    print("borrow released cleanly (no crash)")

    # 2) unconsumed capsule frees cleanly
    b = gc.NativeMatrixBuffer([[1.0, 2.0], [3.0, 4.0]])
    cap = b.__dlpack__()
    del cap, b
    _gc.collect()
    print("unconsumed capsule GC'd cleanly")

    # 3) stress loop: create/borrow/drop many times; survive without crash/leak blowup
    keep = []
    for i in range(5000):
        m = gc.NativeMatrixBuffer([[float(i), float(i + 1)], [float(i + 2), float(i + 3)]])
        arr = np.from_dlpack(m)
        if i % 1000 == 0:
            keep.append(arr)          # hold a few borrows past their owner
        del m
    _gc.collect()
    ok = all(k.shape == (2, 2) for k in keep)
    print(f"stress 5000x create/borrow/drop -> held borrows valid: {ok}")
    print("EXPORT LIFETIME SAFETY: all checks must pass with no crash")


if __name__ == "__main__":
    main()
