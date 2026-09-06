from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[2]


def test_maturin_python_source_exposes_complete_v1_adapter_surface():
    code = r"""
import os
import sys
import types


class FakeReport:
    def __len__(self):
        return 2

    def combo(self, index):
        return [index]

    def metric_values(self, index):
        return [1.0 if index == 0 else -1.0, 1.0]

    def candidate_id(self, index):
        return index

    def ranked_indices(self, *, metric_index=None, descending=True, limit=None):
        values = [0, 1]
        return values if limit is None else values[:limit]


class FakeArtifact:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __init__(self):
        self.closed = False

    def analyze(self):
        return FakeReport()

    def close(self):
        self.closed = True


boundary = types.ModuleType("_pkg_fake_boundary")
boundary.BOUNDARY_NAME = "pkg-boundary"
boundary.calls = []


def compile_continuous(config, features, target, *, rows, cols):
    boundary.calls.append(
        {
            "config": config,
            "features": features,
            "target": target,
            "rows": rows,
            "cols": cols,
        }
    )
    return FakeArtifact()


boundary.compile_continuous = compile_continuous
sys.modules[boundary.__name__] = boundary
os.environ["GAFIME_V1_BOUNDARY_MODULE"] = boundary.__name__

import gafime

# The source-only compatibility path intentionally has only the fake legacy
# boundary above.  `gafime.semantic` must therefore be discoverable without
# eagerly importing its real PyO3 extension.
assert "semantic" not in gafime.__all__
assert "semantic" in dir(gafime)
assert "semantic" not in vars(gafime)
legacy_star = {}
exec("from gafime import *", legacy_star)
assert "semantic" not in legacy_star
assert legacy_star["GafimeEngine"] is gafime.GafimeEngine


class FrameLike:
    columns = ["a", "b"]

    def to_dicts(self):
        return [{"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 4.0}]


cfg = gafime.EngineConfig(
    backend="core",
    metric_names=("pearson", "r2"),
    stability_std_threshold=0.12,
    permutation_p_threshold=0.03,
    time_series_lags=(2, 5),
    time_series_windows=(8,),
    decision_path_max_depth=3,
    decision_path_rounds=2,
    decision_path_max_paths=9,
    decision_path_max_bins=7,
    decision_path_min_leaf=4,
    decision_path_learning_rate=0.25,
    decision_path_top_k_features=6,
    budget=gafime.ComputeBudget(
        max_comb_size=1,
        max_combinations_per_k=8,
        keep_in_vram=False,
        max_feature_candidate=11,
    ),
    permutation_tests=0,
    num_repeats=1,
)
compiled = gafime.GafimeEngine(cfg).compile(FrameLike(), [1.0, 2.0])
assert compiled.boundary_name == "pkg-boundary"
report = compiled.analyze()
payload = boundary.calls[0]["config"]
assert boundary.calls[0]["features"] == [1.0, 3.0, 2.0, 4.0]
assert payload["stability_std_threshold"] == 0.12
assert payload["permutation_p_threshold"] == 0.03
assert payload["time_series_lags"] == [2, 5]
assert payload["time_series_windows"] == [8]
assert payload["decision_path_max_depth"] == 3
assert payload["decision_path_rounds"] == 2
assert payload["decision_path_max_paths"] == 9
assert payload["decision_path_max_bins"] == 7
assert payload["decision_path_min_leaf"] == 4
assert payload["decision_path_learning_rate"] == 0.25
assert payload["decision_path_top_k_features"] == 6
assert payload["budget"]["keep_in_vram"] is False
assert payload["budget"]["max_feature_candidate"] == 11
assert report.interactions.is_native_backed
assert report.interactions.top_k(1)[0].combo == (0,)
compiled.close()
try:
    compiled.analyze()
except RuntimeError:
    pass
else:
    raise AssertionError("closed native compile artifact must reject analyze")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    env.pop("GAFIME_V1_BOUNDARY_MODULE", None)
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT / "python",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
