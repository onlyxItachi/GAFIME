"""CUDA Graph track tests (v0.5 compile, Part 6).

Skipped automatically when the native CUDA payload is not present, so the suite
stays green on CPU-only machines while still validating capture/replay on a real
device when ``gafime-cuda`` is installed/built.
"""
from __future__ import annotations

import random
import unittest

from gafime.compile.flags import CompileFlags
from gafime.metrics import MetricSuite
from gafime.native_data import coerce_inputs


def _cuda_backend():
    try:
        from gafime.backends.native_cuda_backend import NativeCudaBackend

        return NativeCudaBackend(device_id=0)
    except Exception:
        return None


def _max_metric_diff(a, b):
    diff = 0.0
    for key in a:
        for metric, value in a[key].items():
            diff = max(diff, abs(value - b[key][metric]))
    return diff


@unittest.skipIf(_cuda_backend() is None, "native CUDA payload unavailable")
class CudaGraphTrackTests(unittest.TestCase):
    def setUp(self):
        self.backend = _cuda_backend()
        assert self.backend is not None  # guaranteed by skipIf
        rng = random.Random(7)
        n, f = 8000, 8
        self.X = [[rng.random() for _ in range(f)] for _ in range(n)]
        self.y1 = [self.X[i][0] * self.X[i][1] + 0.2 * self.X[i][2] for i in range(n)]
        self.y2 = [self.X[i][3] - self.X[i][4] for i in range(n)]
        self.Xn, self.y1n, _ = coerce_inputs(self.X, self.y1, None)
        _, self.y2n, _ = coerce_inputs(self.X, self.y2, None)
        self.suite = MetricSuite(("pearson", "r2"))
        self.combos = [(i,) for i in range(f)] + [(i, j) for i in range(f) for j in range(i + 1, f)]

    def test_graph_session_reports_captured(self):
        session = self.backend.compile_session(
            self.Xn, self.y1n, None, self.suite, CompileFlags(graph=True)
        )
        try:
            self.assertEqual(session.graph_status, "captured")
            self.assertFalse(session.warnings)
        finally:
            session.close()

    def test_graph_matches_non_graph_and_replays(self):
        reference = self.backend.score_combos(self.Xn, self.y1n, self.combos, self.suite)
        session = self.backend.compile_session(
            self.Xn, self.y1n, None, self.suite, CompileFlags(graph=True)
        )
        try:
            captured = session.score_combos(self.Xn, self.y1n, self.combos, self.suite)
            replayed = session.score_combos(self.Xn, self.y1n, self.combos, self.suite)
            # fp32 + cross-block atomic ordering -> tolerate small numeric noise
            self.assertLess(_max_metric_diff(captured, reference), 1e-3)
            self.assertLess(_max_metric_diff(captured, replayed), 1e-3)
            self.assertEqual(session.graph_captured_shapes, {(1, 8), (2, 28)})
        finally:
            session.close()

    def test_resident_target_update_enables_correct_replay(self):
        from gafime.backends.native_cuda_backend import (
            _continuous_scheduler_batches,
            _stats_metric_names,
            _stats_to_metrics,
        )

        session = self.backend.compile_session(
            self.Xn, self.y1n, None, self.suite, CompileFlags(graph=True)
        )
        try:
            session.score_combos(self.Xn, self.y1n, self.combos, self.suite)  # capture
            self.backend.update_resident_target(session._matrix, self.y2n)    # swap y in place
            names = _stats_metric_names(self.suite.metric_names)
            replayed = {}
            for batch in _continuous_scheduler_batches(self.combos):
                _k, idx, _o, _i, _t, arity, batch_size = batch
                stats = self.backend._launch_global_continuous_batch_graph(
                    session._matrix, idx, int(arity), int(batch_size)
                )
                for row_idx, row in enumerate(stats):
                    combo = tuple(int(idx[row_idx * int(arity) + c]) for c in range(int(arity)))
                    replayed[combo] = _stats_to_metrics(row, names)
            reference = self.backend.score_combos(self.Xn, self.y2n, self.combos, self.suite)
            self.assertLess(_max_metric_diff(replayed, reference), 1e-3)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
