"""Shared helpers for the GAFIME v1 release measurement suite.

Logged scripts emit telemetry artifacts into ~/gafime_telemetry so OpenML tour
and performance numbers become release-note evidence.

Run convention (each script documents its own line):
  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 <script>.py

DO NOT hand-edit numbers into release notes; only logged artifacts count.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

WORKTREE = "/home/hamza-usta/GAFIME"
OUTDIR = os.path.expanduser("~/gafime_telemetry")
DEFAULT_SEED = 7

# decision_path engine config knobs shared across scripts (override per-script).
DP_CONFIG = dict(
    enable_decision_path_functions=True,
    decision_path_max_depth=3,
    decision_path_rounds=20,
    decision_path_max_paths=16,
    decision_path_min_leaf=8,
    decision_path_learning_rate=0.3,
    backend="core",
    permutation_tests=0,
)

# OpenML datasets used across the tour, tagged by whether we expect
# threshold-conjunction structure decision_path can exploit.
DATASET_REGISTRY = {
    "diabetes": dict(data_id=37, structure="rich"),            # glucose/BMI/age thresholds
    "credit-g": dict(data_id=31, structure="mixed"),
    "blood-transfusion": dict(data_id=1464, structure="poor"),  # near-linear recency/frequency
    "phoneme": dict(data_id=1489, structure="rich"),
    "banknote": dict(data_id=1462, structure="rich"),
    "ilpd": dict(data_id=1480, structure="mixed"),
}


def telemetry():
    """Import the canonical telemetry helper from the v1 Python package."""
    import gafime.telemetry as tel
    return tel


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
def load_synthetic_and(seed: int = DEFAULT_SEED, n: int = 2000, f: int = 6):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.random((n, f))
    truth = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(float)
    y = (truth + (rng.random(n) < 0.05)).clip(0, 1).round().astype(int)
    meta = {"source": "synthetic", "name": "planted_and_x0x1", "openml_id": None,
            "rows": n, "features": f, "target_type": "binary"}
    return X, y, [f"x{i}" for i in range(f)], meta, truth


def load_friedman1(seed: int = DEFAULT_SEED, n: int = 2000):
    """Friedman1 regression (nonlinear + interaction) -> binarized for classifiers."""
    import numpy as np
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=seed)
    yb = (y > np.median(y)).astype(int)
    meta = {"source": "synthetic", "name": "friedman1", "openml_id": None,
            "rows": n, "features": 10, "target_type": "binary"}
    return X, yb, [f"x{i}" for i in range(10)], meta, None


def load_openml(data_id: int, name: str, seed: int = DEFAULT_SEED):
    import numpy as np
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=data_id, as_frame=True)
    Xdf = ds.data.select_dtypes("number")
    X = np.nan_to_num(Xdf.to_numpy(dtype=float))
    classes = sorted(set(ds.target.tolist()))
    y = np.array([classes.index(v) for v in ds.target.tolist()], dtype=int)
    meta = {"source": "openml", "name": name, "openml_id": data_id,
            "rows": int(X.shape[0]), "features": int(X.shape[1]), "target_type": "binary"}
    return X, y, list(Xdf.columns), meta, None


def dataset_loader(name: str) -> Callable:
    if name == "synthetic_and":
        return load_synthetic_and
    if name == "friedman1":
        return load_friedman1
    info = DATASET_REGISTRY[name]
    return lambda seed=DEFAULT_SEED: load_openml(info["data_id"], name, seed)


_PATH_TERM = re.compile(r"^(?P<name>.+?)(?P<op><=|>)(?P<threshold>-?\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class DecisionPathCandidate:
    features: Tuple[int, ...]
    thresholds: Tuple[float, ...]
    signs: Tuple[int, ...]


def _candidate_from_path_label(label: str, names: List[str]) -> DecisionPathCandidate:
    if not label.startswith("path[") or not label.endswith("]"):
        raise ValueError(f"not a decision_path label: {label!r}")
    body = label[len("path[") : -1]
    features: List[int] = []
    thresholds: List[float] = []
    signs: List[int] = []
    for raw_term in body.split(" & "):
        match = _PATH_TERM.match(raw_term)
        if match is None:
            raise ValueError(f"cannot parse decision_path term {raw_term!r} from {label!r}")
        features.append(names.index(match.group("name")))
        thresholds.append(float(match.group("threshold")))
        signs.append(-1 if match.group("op") == "<=" else 1)
    return DecisionPathCandidate(tuple(features), tuple(thresholds), tuple(signs))


# --------------------------------------------------------------------------
# decision_path candidate materialization (leakage-safe: fit specs on train,
# materialize on any split)
# --------------------------------------------------------------------------
def mine_candidates(Xtr, ytr, names, config_overrides: Dict[str, Any] | None = None):
    """Run the engine on TRAIN, return decision_path candidates sorted by score."""
    from gafime.config import EngineConfig
    from gafime import GafimeEngine
    cfg = dict(DP_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)
    report = GafimeEngine(config=EngineConfig(**cfg)).analyze(
        Xtr.tolist(), ytr.tolist(), feature_names=list(names))
    dp = [ir for ir in (getattr(report, "interactions", []) or [])
          if getattr(ir, "family", None) == "decision_path"]
    dp.sort(key=lambda r: -(max(r.metrics.values()) if r.metrics else 0.0))
    candidates = []
    for item in dp:
        label = item.feature_names[0] if item.feature_names else item.expression
        candidates.append(_candidate_from_path_label(str(label), list(names)))
    return candidates, report


def cols_hard(Xnp, cands):
    import numpy as np
    if not cands:
        return np.zeros((len(Xnp), 0))
    out = []
    for c in cands:
        col = np.ones(len(Xnp))
        for f, t, s in zip(c.features, c.thresholds, c.signs):
            cond = (Xnp[:, int(f)] <= float(t)) if int(s) < 0 else (Xnp[:, int(f)] > float(t))
            col = col * cond.astype(float)
        out.append(col)
    return np.column_stack(out)


def cols_soft(Xnp, cands, sharpness: float = 8.0):
    import numpy as np
    if not cands:
        return np.zeros((len(Xnp), 0))
    out = []
    for c in cands:
        col = np.ones(len(Xnp))
        for f, t, s in zip(c.features, c.thresholds, c.signs):
            z = np.clip(-sharpness * int(s) * (Xnp[:, int(f)] - float(t)), -30.0, 30.0)
            col = col * (1.0 / (1.0 + np.exp(z)))
        out.append(col)
    return np.column_stack(out)


# --------------------------------------------------------------------------
# downstream models (scaled so the baseline is FAIR; unscaled MLP underfits and
# fakes lift -- the confound we caught in MSG 57)
# --------------------------------------------------------------------------
def make_model(kind: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if kind == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                      early_stopping=True, random_state=0))


def split(X, y, seed: int = DEFAULT_SEED, test_size: float = 0.3):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def new_log(dataset_meta: Dict[str, Any], config: Dict[str, Any]):
    """Open a telemetry record pre-filled with worktree + dataset + config."""
    tel = telemetry()
    rec = tel.new_record(worktree=WORKTREE,
                         dataset=tel._default_dataset() | dataset_meta,
                         config=config)
    return tel, rec
