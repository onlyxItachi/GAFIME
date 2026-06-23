"""export_01 | FRAMEWORK EXPORT correctness: the native buffers hand frameworks a
zero-copy view of GAFIME memory. Verifies __dlpack__ (torch + numpy from_dlpack)
and __array_interface__ share the SAME pointer and matching values; matrix=2D,
vector=1D. Requires the export commit (lab 46482f7) merged + gafime_core rebuilt.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python export_01_zero_copy_parity.py
"""
import numpy as np
from gafime import gafime_core as gc


def main():
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    buf = gc.NativeMatrixBuffer(X)

    if not hasattr(buf, "__dlpack__"):
        print("FAIL: native buffer has no __dlpack__ -> export commit not merged/rebuilt")
        return

    ai = buf.__array_interface__
    d = np.from_dlpack(buf)
    a = np.asarray(buf)
    print(f"array_interface: shape={ai['shape']} typestr={ai['typestr']}")
    print(f"from_dlpack: shape={d.shape} dtype={d.dtype}")
    print(f"same pointer (dlpack==interface): {d.ctypes.data == ai['data'][0]}")
    print(f"same pointer (asarray==interface): {a.ctypes.data == ai['data'][0]}")
    print(f"dlpack 2D matches input: {d.shape == (3, 3) and d[0].tolist() == [1.0, 2.0, 3.0]}")

    vec = gc.NativeVectorBuffer([10.0, 20.0, 30.0, 40.0])
    dv = np.from_dlpack(vec)
    print(f"vector 1D zero-copy: shape={dv.shape} values={dv.tolist()} "
          f"same_ptr={dv.ctypes.data == vec.__array_interface__['data'][0]}")
    print(f"dlpack_device (kDLCPU,0): {tuple(buf.__dlpack_device__())}")

    try:
        import torch
        t = torch.from_dlpack(gc.NativeMatrixBuffer(X))
        print(f"torch.from_dlpack: shape={tuple(t.shape)} dtype={t.dtype} ok={t[2,2].item()==9.0}")
    except ImportError:
        print("torch not installed; numpy DLPack path validated")
    print("EXPORT ZERO-COPY PARITY: pointers must match across all paths")


if __name__ == "__main__":
    main()
