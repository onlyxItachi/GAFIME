"""Framework-grade zero-copy export protocols on the native buffers.

GAFIME's Export track is native-pointer / framework-integration plumbing (NOT a
user file dump). These tests prove the CPU path: NativeMatrixBuffer /
NativeVectorBuffer expose __array_interface__ (numpy/cupy) and __dlpack__ /
__dlpack_device__ (torch / JAX / numpy from_dlpack), all sharing GAFIME's native
memory with no copy, and lifetime-safe (a borrowed tensor outlives its owner).

numpy-gated; torch is exercised only if installed. CPU-safe (no GPU).
"""
import gc as _gc
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is required for these protocols
    np = None

from gafime import gafime_core as gc


def _numpy_or_skip(testcase: unittest.TestCase):
    if np is None:
        testcase.skipTest("numpy not available")
    return np


class NativeExportProtocolTests(unittest.TestCase):
    def test_matrix_dlpack_is_2d_zero_copy(self):
        numpy = _numpy_or_skip(self)
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        buf = gc.NativeMatrixBuffer(X)
        arr = numpy.from_dlpack(buf)
        self.assertEqual(arr.shape, (3, 3))
        self.assertEqual(arr.dtype, numpy.float32)
        self.assertEqual(arr.ctypes.data, buf.__array_interface__["data"][0])
        self.assertEqual(arr[0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(arr[2, 2], 9.0)

    def test_vector_dlpack_is_1d_zero_copy(self):
        numpy = _numpy_or_skip(self)
        vals = [10.0, 20.0, 30.0, 40.0]
        buf = gc.NativeVectorBuffer(vals)
        arr = numpy.from_dlpack(buf)
        self.assertEqual(arr.shape, (4,))
        self.assertEqual(arr.tolist(), vals)
        self.assertEqual(arr.ctypes.data, buf.__array_interface__["data"][0])

    def test_array_interface_wellformed_2d(self):
        numpy = _numpy_or_skip(self)
        buf = gc.NativeMatrixBuffer([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ai = buf.__array_interface__
        self.assertEqual(ai["version"], 3)
        self.assertEqual(ai["shape"], (3, 2))
        self.assertIn(ai["typestr"], ("<f4", "<f8"))
        self.assertIsInstance(ai["data"][0], int)
        self.assertIs(ai["data"][1], False)
        proxy = type("P", (), {"__array_interface__": ai})()
        self.assertEqual(numpy.asarray(proxy).shape, (3, 2))

    def test_asarray_shares_memory(self):
        numpy = _numpy_or_skip(self)
        buf = gc.NativeMatrixBuffer([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arr = numpy.asarray(buf)
        self.assertFalse(arr.flags["OWNDATA"])
        self.assertEqual(arr.ctypes.data, buf.__array_interface__["data"][0])
        self.assertEqual(arr.size, 6)

    def test_dlpack_device_is_cpu(self):
        buf = gc.NativeMatrixBuffer([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(tuple(buf.__dlpack_device__()), (1, 0))

    def test_dlpack_lifetime_owner_dropped(self):
        numpy = _numpy_or_skip(self)
        vals = [11.0, 22.0, 33.0]
        buf = gc.NativeVectorBuffer(vals)
        tensor = numpy.from_dlpack(buf)
        del buf
        _gc.collect()
        self.assertEqual(tensor.tolist(), vals)
        del tensor
        _gc.collect()

    def test_unconsumed_capsule_frees_cleanly(self):
        buf = gc.NativeMatrixBuffer([[1.0, 2.0], [3.0, 4.0]])
        cap = buf.__dlpack__()
        del cap, buf
        _gc.collect()

    def test_torch_from_dlpack_if_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        tensor = torch.from_dlpack(gc.NativeMatrixBuffer(X))
        self.assertEqual(tuple(tensor.shape), (3, 3))
        self.assertEqual(tensor[0].tolist(), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
