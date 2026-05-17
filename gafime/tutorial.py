"""Build the comprehensive GAFIME tutorial notebook as nbformat JSON.

Produces a single .ipynb covering every public API with real DS usecases:
  - GafimeEngine / EngineConfig / ComputeBudget
  - Discrete function search (v0.4.0)
  - DiagnosticReport (interactions, stability, permutations, decision, backend)
  - Backend selection, inspection, benchmarking
  - GafimeStreamer (large-file Polars streaming)
  - GafimeSelector (sklearn pipeline integration)
  - CLI (`gafime --check`, `--init`)
  - Feature engineering workflow with a real tabular dataset
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": [ln if ln.endswith("\n") else ln + "\n" for ln in lines][:-1]
        + [lines[-1]],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln if ln.endswith("\n") else ln + "\n" for ln in lines][:-1]
        + [lines[-1]],
    }


def build() -> dict:
    cells: List[dict] = []

    cells.append(md(
        "# GAFIME — Full API Tour 🚀\n",
        "\n",
        "**GPU-Accelerated Feature Interaction Mining Engine**\n",
        "\n",
        "This notebook is the canonical end-to-end reference for every public API\n",
        "surface in `gafime`. Each section is self-contained — run them top to\n",
        "bottom, or jump directly to the one that matches your usecase.\n",
        "\n",
        "## Table of contents\n",
        "1. Installation & backend check\n",
        "2. 60-second quickstart — planted interaction signal\n",
        "3. `EngineConfig` & `ComputeBudget` — tuning the search\n",
        "4. Discrete function search — thresholds, intervals, rectangles\n",
        "5. Interpreting `DiagnosticReport` (interactions, stability, permutations, decision)\n",
        "6. Real dataset — California housing regression\n",
        "7. Classification — binary target with engineered interactions\n",
        "8. `GafimeSelector` — scikit-learn pipeline drop-in\n",
        "9. `GafimeStreamer` — VRAM-aware streaming over CSV/Parquet\n",
        "10. CLI cheatsheet — `gafime --check`, `gafime --init`\n",
        "11. Production tips\n",
        "\n",
        "> **Hardware-aware.** GAFIME auto-selects the fastest available backend\n",
        "> (CUDA → Metal → OpenMP/C++ core → NumPy). A CUDA-capable GPU will\n",
        "> accelerate Pearson pair scoring up to **40×** over CPU at 200K×100.\n"
    ))

    # ---- 1. Install / check --------------------------------------------
    cells.append(md("## 1. Installation & backend check\n"))
    cells.append(md(
        "Install the latest release from PyPI:\n",
        "\n",
        "```bash\n",
        "pip install gafime                # core + CUDA wheel (linux x86_64, macOS arm64, windows)\n",
        "pip install 'gafime[sklearn]'     # + scikit-learn for GafimeSelector\n",
        "pip install 'gafime[streaming]'   # + polars for GafimeStreamer\n",
        "```\n",
        "\n",
        "Then ask GAFIME which backend it would pick on this machine:\n"
    ))
    cells.append(code(
        "import gafime\n",
        "print('gafime version:', gafime.__version__)\n",
        "\n",
        "# Equivalent to `gafime --check` on the command line\n",
        "import numpy as np\n",
        "from gafime.config import EngineConfig\n",
        "from gafime.backends import resolve_backend\n",
        "\n",
        "cfg = EngineConfig(backend='auto')\n",
        "backend, warnings = resolve_backend(cfg, np.zeros((10, 2)), np.zeros(10))\n",
        "info = backend.info()\n",
        "print(f'Selected backend: {info.name}')\n",
        "print(f'Device         : {info.device}')\n",
        "print(f'Is GPU         : {info.is_gpu}')\n",
        "for w in warnings:\n",
        "    print('warn:', w)\n"
    ))

    # ---- 2. Quickstart --------------------------------------------------
    cells.append(md(
        "## 2. 60-second quickstart\n",
        "\n",
        "Plant an `X[:,0] * X[:,1]` interaction in a synthetic target and see if\n",
        "GAFIME recovers it. This is the 'hello world' for feature interaction\n",
        "mining.\n"
    ))
    cells.append(code(
        "import numpy as np\n",
        "from gafime import GafimeEngine, EngineConfig\n",
        "\n",
        "rng = np.random.default_rng(42)\n",
        "n_samples, n_features = 20_000, 20\n",
        "X = rng.standard_normal((n_samples, n_features))\n",
        "y = X[:, 3] * X[:, 7] + 0.3 * rng.standard_normal(n_samples)\n",
        "feature_names = [f'f{i}' for i in range(n_features)]\n",
        "\n",
        "# Pearson-only is the fast path on the CUDA backend\n",
        "engine = GafimeEngine(EngineConfig(metric_names=('pearson',)))\n",
        "report = engine.analyze(X, y, feature_names=feature_names)\n",
        "\n",
        "print('backend :', report.backend.name, '/', report.backend.device)\n",
        "print('decision:', report.decision.message)\n",
        "\n",
        "# Top 5 strongest interactions\n",
        "top = sorted(report.interactions,\n",
        "             key=lambda r: max(abs(v) for v in r.metrics.values()),\n",
        "             reverse=True)[:5]\n",
        "for r in top:\n",
        "    combo = ' × '.join(r.feature_names)\n",
        "    score = max(abs(v) for v in r.metrics.values())\n",
        "    print(f'  {combo:<20}  |pearson|={score:.3f}')\n"
    ))
    cells.append(md(
        "You should see `f3 × f7` at the top — that's the planted signal.\n"
    ))

    # ---- 3. EngineConfig / ComputeBudget -------------------------------
    cells.append(md(
        "## 3. `EngineConfig` & `ComputeBudget`\n",
        "\n",
        "Every knob that shapes the search is in these two frozen dataclasses.\n",
        "Defaults are tuned for a 'small laptop' workload (≤50K × ≤50).\n",
        "\n",
        "### `EngineConfig`\n",
        "| field | default | meaning |\n",
        "|---|---|---|\n",
        "| `budget` | `ComputeBudget()` | combinatorial search limits (below) |\n",
        "| `metric_names` | `('pearson','spearman','mutual_info','r2')` | metrics to score each combo with |\n",
        "| `num_repeats` | `3` | bootstrap repeats for stability analysis |\n",
        "| `permutation_tests` | `25` | y-permutations for null distribution |\n",
        "| `random_seed` | `7` | determinism |\n",
        "| `stability_std_threshold` | `0.10` | σ above which a metric is called unstable |\n",
        "| `permutation_p_threshold` | `0.05` | significance cutoff |\n",
        "| `mi_bins` | `16` | histogram bins for mutual-information |\n",
        "| `backend` | `'auto'` | `'auto' | 'cuda' | 'metal' | 'cpu' | 'numpy'` |\n",
        "| `device_id` | `0` | CUDA device index |\n",
        "| `enable_discrete_functions` | `False` | include threshold/interval/rectangle candidates |\n",
        "| `discrete_mode` | `'soft'` | `'soft'` everywhere; `'hard'` only on CPU/NumPy |\n",
        "| `discrete_ranking` | `'split_aware'` | rank discrete candidates by split/impurity/residual scores; `'metric'` follows report metrics |\n",
        "| `discrete_threshold_source` | `'quantile'` | v0.4.0 uses quantile thresholds only |\n",
        "| `discrete_gate_sharpness` | `12.0` | soft gate steepness |\n",
        "\n",
        "### `ComputeBudget`\n",
        "| field | default | meaning |\n",
        "|---|---|---|\n",
        "| `max_comb_size` | `2` | up to k-way interactions (2 = pairs) |\n",
        "| `max_combinations_per_k` | `5000` | cap per arity |\n",
        "| `top_features_for_higher_k` | `50` | restrict k≥3 to top-N unary features |\n",
        "| `max_generated_features` | `0` | extra derived columns to attempt |\n",
        "| `keep_in_vram` | `True` | prefer GPU when available |\n",
        "| `vram_budget_mb` | `6144` | VRAM ceiling (RTX 4060 8 GB leaves headroom) |\n",
        "| `max_discrete_candidates` | `100000` | cap for discrete candidates |\n",
        "| `max_thresholds_per_feature` | `9` | max quantile thresholds per selected feature |\n",
        "| `max_intervals_per_feature` | `12` | max intervals built from thresholds |\n",
        "| `max_feature_pairs_for_rectangles` | `500` | rectangle pair cap |\n",
        "| `top_k_features_for_discrete` | `50` | feature shortlist for discrete planning |\n",
        "\n",
        "#### Recipes\n"
    ))
    cells.append(code(
        "from gafime import EngineConfig, ComputeBudget\n",
        "\n",
        "# A) Fast triage on a wide table: Pearson only, pairs only\n",
        "fast = EngineConfig(\n",
        "    metric_names=('pearson',),\n",
        "    budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=20_000),\n",
        "    permutation_tests=10,\n",
        ")\n",
        "\n",
        "# B) Deep search: 3-way interactions, strict significance\n",
        "deep = EngineConfig(\n",
        "    metric_names=('pearson', 'spearman', 'mutual_info'),\n",
        "    budget=ComputeBudget(max_comb_size=3, top_features_for_higher_k=30),\n",
        "    permutation_tests=100,\n",
        "    permutation_p_threshold=0.01,\n",
        ")\n",
        "\n",
        "# C) Force CPU for determinism/debugging\n",
        "cpu_only = EngineConfig(backend='cpu', metric_names=('pearson', 'r2'))\n",
        "\n",
        "for name, cfg in [('fast', fast), ('deep', deep), ('cpu_only', cpu_only)]:\n",
        "    print(name, '->', cfg)\n"
    ))

    cells.append(md(
        "## 4. Discrete function search\n",
        "\n",
        "v0.4.0 can search threshold, interval, and rectangle-style feature\n",
        "representations inside `GafimeEngine`. Discrete candidates use the same\n",
        "`metric_names` as ordinary interactions; there is no separate discrete\n",
        "metric selector. The default report ordering uses split-aware selection\n",
        "scores instead of Pearson-only ranking; set `discrete_ranking='metric'`\n",
        "if you explicitly want report-metric ordering.\n",
        "\n",
        "CUDA and Metal support soft/vectorized discrete functions only. Hard mode\n",
        "is CPU/NumPy-only and raises on GPU backends.\n"
    ))
    cells.append(code(
        "from gafime import ComputeBudget, EngineConfig, GafimeEngine\n",
        "\n",
        "rng = np.random.default_rng(123)\n",
        "X_step = rng.normal(size=(5000, 8)).astype('float32')\n",
        "y_step = (X_step[:, 0] > 0.25).astype('float32')\n",
        "y_step += 0.05 * rng.normal(size=X_step.shape[0]).astype('float32')\n",
        "\n",
        "cfg = EngineConfig(\n",
        "    backend='auto',\n",
        "    metric_names=('pearson',),\n",
        "    enable_discrete_functions=True,\n",
        "    discrete_mode='soft',\n",
        "    discrete_ranking='split_aware',\n",
        "    budget=ComputeBudget(\n",
        "        max_discrete_candidates=1000,\n",
        "        top_k_features_for_discrete=6,\n",
        "        max_feature_pairs_for_rectangles=12,\n",
        "    ),\n",
        "    permutation_tests=0,\n",
        "    num_repeats=1,\n",
        ")\n",
        "report_step = GafimeEngine(cfg).analyze(X_step, y_step)\n",
        "top_discrete = [item for item in report_step.interactions if item.family == 'discrete_function'][:5]\n",
        "for item in top_discrete:\n",
        "    print(item.expression, item.metrics)\n"
    ))

    # ---- 4. DiagnosticReport deep-dive ---------------------------------
    cells.append(md(
        "## 5. Interpreting `DiagnosticReport`\n",
        "\n",
        "`engine.analyze()` returns a single `DiagnosticReport` with:\n",
        "\n",
        "| attribute | type | what it is |\n",
        "|---|---|---|\n",
        "| `interactions` | `List[InteractionResult]` | combo + per-metric score |\n",
        "| `stability` | `List[StabilityResult]` | mean/std of each metric over bootstrap repeats |\n",
        "| `permutations` | `List[PermutationResult]` | empirical p-value per metric vs. y-permuted null |\n",
        "| `decision` | `Decision` | `signal_detected` + human-readable message |\n",
        "| `warnings` | `List[str]` | budget / memory warnings raised during the run |\n",
        "| `backend` | `BackendInfo` | which backend actually ran |\n",
        "| `feature_names` | `List[str]` | normalised feature names (auto-generated if not provided) |\n",
        "| `config` | `EngineConfig` | echoed config for reproducibility |\n",
        "| `.to_dict()` | method | JSON-serialisable snapshot |\n"
    ))
    cells.append(code(
        "# Build a richer report and walk through every piece\n",
        "cfg = EngineConfig(metric_names=('pearson', 'spearman'),\n",
        "                   budget=ComputeBudget(max_comb_size=2),\n",
        "                   permutation_tests=20)\n",
        "report = GafimeEngine(cfg).analyze(X, y, feature_names=feature_names)\n",
        "\n",
        "print('=== warnings ===')\n",
        "for w in report.warnings:\n",
        "    print(' -', w)\n",
        "\n",
        "print()\n",
        "print('=== decision ===')\n",
        "print(' signal_detected:', report.decision.signal_detected)\n",
        "print(' message        :', report.decision.message)\n",
        "\n",
        "print()\n",
        "print('=== backend ===')\n",
        "print(vars(report.backend))\n"
    ))
    cells.append(code(
        "# Merge interactions + stability + permutations into one pandas DataFrame\n",
        "import pandas as pd\n",
        "\n",
        "stab = {r.combo: r for r in report.stability}\n",
        "perm = {r.combo: r for r in report.permutations}\n",
        "\n",
        "rows = []\n",
        "for r in report.interactions:\n",
        "    row = {'combo': ' × '.join(r.feature_names), 'size': len(r.combo)}\n",
        "    for m, v in r.metrics.items():\n",
        "        row[f'{m}'] = v\n",
        "        row[f'{m}_std'] = stab.get(r.combo, None) and stab[r.combo].metrics_std.get(m)\n",
        "        row[f'{m}_p'] = perm.get(r.combo, None) and perm[r.combo].p_values.get(m)\n",
        "    rows.append(row)\n",
        "\n",
        "df = pd.DataFrame(rows)\n",
        "# keep only significant + stable, sort by |pearson|\n",
        "df['|pearson|'] = df['pearson'].abs()\n",
        "keep = (df.get('pearson_p', 1.0) <= 0.05) & (df.get('pearson_std', 0.0) <= 0.10)\n",
        "df.loc[keep].sort_values('|pearson|', ascending=False).head(10)\n"
    ))

    # ---- 6. California housing -----------------------------------------
    cells.append(md(
        "## 6. Real dataset — California housing regression\n",
        "\n",
        "A classic tabular regression benchmark. We'll look for non-linear pair\n",
        "interactions between the 8 numeric features that predict median house\n",
        "value.\n"
    ))
    cells.append(code(
        "from sklearn.datasets import fetch_california_housing\n",
        "data = fetch_california_housing(as_frame=True)\n",
        "X_df = data.data\n",
        "y_arr = data.target.values\n",
        "print(X_df.head())\n",
        "print('shape:', X_df.shape)\n"
    ))
    cells.append(code(
        "cfg = EngineConfig(\n",
        "    metric_names=('pearson', 'spearman'),\n",
        "    budget=ComputeBudget(max_comb_size=2),\n",
        "    permutation_tests=30,\n",
        ")\n",
        "report = GafimeEngine(cfg).analyze(\n",
        "    X_df.values, y_arr, feature_names=list(X_df.columns)\n",
        ")\n",
        "\n",
        "pairs = [(r, max(abs(v) for v in r.metrics.values()))\n",
        "         for r in report.interactions if len(r.combo) == 2]\n",
        "pairs.sort(key=lambda t: t[1], reverse=True)\n",
        "print('Top 5 pair interactions:')\n",
        "for r, s in pairs[:5]:\n",
        "    name = ' × '.join(r.feature_names)\n",
        "    pear = r.metrics.get('pearson', float('nan'))\n",
        "    spear = r.metrics.get('spearman', float('nan'))\n",
        "    print(f'  {name:<25}  pearson={pear:+.3f}  spearman={spear:+.3f}')\n"
    ))

    # ---- 7. Classification ---------------------------------------------
    cells.append(md(
        "## 7. Classification with engineered interactions\n",
        "\n",
        "Interactions surfaced by GAFIME plug straight into downstream models.\n",
        "We'll binarise the California target, mine pairs, craft an interaction\n",
        "matrix, and fit a logistic regression to compare against the raw\n",
        "features.\n"
    ))
    cells.append(code(
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import roc_auc_score\n",
        "\n",
        "y_bin = (y_arr > np.median(y_arr)).astype(int)\n",
        "Xtr, Xte, ytr, yte = train_test_split(\n",
        "    X_df.values, y_bin, test_size=0.25, random_state=0, stratify=y_bin\n",
        ")\n",
        "\n",
        "# Baseline: raw features\n",
        "scaler = StandardScaler().fit(Xtr)\n",
        "base = LogisticRegression(max_iter=500).fit(scaler.transform(Xtr), ytr)\n",
        "auc_base = roc_auc_score(yte, base.predict_proba(scaler.transform(Xte))[:, 1])\n",
        "\n",
        "# Engineered: append top-5 pair products discovered by GAFIME\n",
        "report = GafimeEngine(EngineConfig(metric_names=('pearson',),\n",
        "                                   budget=ComputeBudget(max_comb_size=2))\n",
        "                     ).analyze(Xtr, ytr, feature_names=list(X_df.columns))\n",
        "top_pairs = [r.combo for r in sorted(\n",
        "    (r for r in report.interactions if len(r.combo) == 2),\n",
        "    key=lambda r: abs(r.metrics.get('pearson', 0.0)),\n",
        "    reverse=True,\n",
        ")[:5]]\n",
        "\n",
        "def augment(A):\n",
        "    extra = np.column_stack([A[:, i] * A[:, j] for i, j in top_pairs])\n",
        "    return np.hstack([A, extra])\n",
        "\n",
        "scaler2 = StandardScaler().fit(augment(Xtr))\n",
        "lift = LogisticRegression(max_iter=500).fit(scaler2.transform(augment(Xtr)), ytr)\n",
        "auc_lift = roc_auc_score(yte, lift.predict_proba(scaler2.transform(augment(Xte)))[:, 1])\n",
        "\n",
        "print(f'AUC baseline          : {auc_base:.4f}')\n",
        "print(f'AUC + GAFIME pairs    : {auc_lift:.4f}')\n",
        "print(f'Δ                     : {auc_lift - auc_base:+.4f}')\n"
    ))

    # ---- 8. GafimeSelector in sklearn ----------------------------------
    cells.append(md(
        "## 8. `GafimeSelector` — sklearn pipeline drop-in\n",
        "\n",
        "`GafimeSelector` is a `BaseEstimator + TransformerMixin` that evaluates\n",
        "all pairwise interactions during `fit`, keeps the top-`k`, and appends\n",
        "them as new columns during `transform`. It composes with\n",
        "`Pipeline`, `GridSearchCV`, `cross_val_score`, etc.\n",
        "\n",
        "**Constructor**\n",
        "```python\n",
        "GafimeSelector(\n",
        "    k=10,                 # how many interactions to retain\n",
        "    backend='auto',       # passthrough to the engine\n",
        "    metric='pearson',     # 'pearson' | 'spearman' | 'r2'\n",
        "    operator='multiply',  # how to combine the pair -> 'multiply' | 'add' | 'subtract' | 'divide'\n",
        "    n_jobs=-1,\n",
        "    verbose=False,\n",
        ")\n",
        "```\n",
        "Fitted attributes: `top_interactions_` (list of `(i, j)`), `n_features_in_`.\n"
    ))
    cells.append(code(
        "from gafime.sklearn import GafimeSelector\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.ensemble import GradientBoostingClassifier\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "pipe = Pipeline([\n",
        "    ('gafime', GafimeSelector(k=8, metric='pearson', operator='multiply')),\n",
        "    ('clf', GradientBoostingClassifier(random_state=0)),\n",
        "])\n",
        "scores = cross_val_score(pipe, X_df.values, y_bin, cv=3, scoring='roc_auc')\n",
        "print('CV ROC-AUC:', scores, '| mean:', scores.mean().round(4))\n",
        "\n",
        "# Inspect what was selected on the full training set\n",
        "pipe.fit(X_df.values, y_bin)\n",
        "names = list(X_df.columns)\n",
        "print('\\nTop interactions kept by the selector:')\n",
        "for i, j in pipe.named_steps['gafime'].top_interactions_:\n",
        "    print(f'  {names[i]} × {names[j]}')\n"
    ))

    # ---- 9. Streaming --------------------------------------------------
    cells.append(md(
        "## 9. `GafimeStreamer` — VRAM-aware streaming\n",
        "\n",
        "When your dataset exceeds RAM, stream chunks directly from CSV/Parquet\n",
        "in sizes that fit your VRAM budget. Polars lazy frames do the disk I/O;\n",
        "GAFIME sanitises each chunk to contiguous `float32` for `cudaMemcpy`.\n",
        "\n",
        "```python\n",
        "GafimeStreamer(file_path, target_cols=None, y_col=None)\n",
        "    .total_rows                                   # cached row count\n",
        "    .estimate_optimal_batch_size(vram_budget_gb)  # rows that fit your GPU\n",
        "    .stream(batch_size=None)                      # yields X chunks\n",
        "    .stream_with_target(batch_size=None)          # yields (X, y) chunks\n",
        "```\n"
    ))
    cells.append(code(
        "# Demo: write a synthetic parquet/csv and stream it\n",
        "import tempfile, os\n",
        "try:\n",
        "    import polars as pl\n",
        "    HAS_POLARS = True\n",
        "except ImportError:\n",
        "    HAS_POLARS = False\n",
        "\n",
        "if not HAS_POLARS:\n",
        "    print(\"polars not installed — `pip install 'gafime[streaming]'` to run this section\")\n",
        "else:\n",
        "    tmp = tempfile.mkdtemp()\n",
        "    path = os.path.join(tmp, 'demo.parquet')\n",
        "    cols = {f'f{i}': rng.standard_normal(50_000).astype('float32') for i in range(8)}\n",
        "    cols['y'] = (cols['f0'] * cols['f1'] + 0.5 * rng.standard_normal(50_000)).astype('float32')\n",
        "    pl.DataFrame(cols).write_parquet(path)\n",
        "\n",
        "    from gafime import GafimeStreamer\n",
        "    streamer = GafimeStreamer(path, y_col='y')\n",
        "    print('rows   :', streamer.total_rows)\n",
        "    print('features:', streamer.n_features)\n",
        "    print('optimal batch (6 GB VRAM):', streamer.estimate_optimal_batch_size(vram_budget_gb=6.0))\n",
        "\n",
        "    for bi, (X_chunk, y_chunk) in enumerate(streamer.stream_with_target(batch_size=10_000)):\n",
        "        print(f'  batch {bi}: X={X_chunk.shape} {X_chunk.dtype}, y={y_chunk.shape}')\n",
        "        if bi == 2:\n",
        "            break\n"
    ))

    # ---- 10. CLI --------------------------------------------------------
    cells.append(md(
        "## 10. CLI cheatsheet\n",
        "\n",
        "After `pip install gafime`, the `gafime` executable is on `$PATH`:\n",
        "\n",
        "```bash\n",
        "gafime --version         # print installed version\n",
        "gafime --check           # enumerate available backends and the one that would be picked\n",
        "gafime --init            # write a starter notebook (`gafime_tutorial.ipynb`) to the cwd\n",
        "gafime --init -o path.ipynb   # custom path\n",
        "```\n",
        "\n",
        "You can also invoke programmatically:\n"
    ))
    cells.append(code(
        "from gafime import generate_tutorial\n",
        "# generate_tutorial('my_starter.ipynb')   # uncomment to write file\n",
        "print(generate_tutorial.__doc__)\n"
    ))

    # ---- 11. Production tips -------------------------------------------
    cells.append(md(
        "## 11. Production tips\n",
        "\n",
        "- **Pin a seed.** `EngineConfig(random_seed=…)` → deterministic bootstrap,\n",
        "  permutations, and combo sampling across runs.\n",
        "- **Keep it pair-only for speed.** `max_comb_size=2` + `metric_names=('pearson',)`\n",
        "  activates the CUDA bucket fast path (≈40× CPU on ≥100K rows).\n",
        "- **Gate on both p-value *and* stability.** A combo with low `permutations`\n",
        "  p-value but large `stability.metrics_std` is a bootstrap artefact.\n",
        "- **Reuse the engine.** `GafimeEngine` is stateless between calls but its\n",
        "  CUDA backend caches device buffers keyed on `id(X)` — if you call\n",
        "  `analyze` multiple times with the **same** `X` array, permutation passes\n",
        "  skip upload + host-side centering.\n",
        "- **Save the report.** `report.to_dict()` is JSON-serialisable after you\n",
        "  convert the small handful of dataclasses with\n",
        "  `dataclasses.asdict`, making it a drop-in artefact for MLflow /\n",
        "  experiment tracking.\n",
        "- **Pair with GPU-aware storage.** `GafimeStreamer` on Parquet → batched\n",
        "  into `engine.analyze` lets you scan datasets larger than RAM without\n",
        "  manual chunking.\n",
        "\n",
        "### Reference summary (every public symbol)\n",
        "| symbol | kind | lives in |\n",
        "|---|---|---|\n",
        "| `GafimeEngine(config=None).analyze(X, y, feature_names=None)` | class | `gafime.engine` |\n",
        "| `EngineConfig(...)` | dataclass | `gafime.config` |\n",
        "| `ComputeBudget(...)` | dataclass | `gafime.config` |\n",
        "| `GafimeStreamer(file_path, target_cols=None, y_col=None)` | class | `gafime.io` |\n",
        "| `GafimeSelector(k, backend, metric, operator, n_jobs, verbose)` | class | `gafime.sklearn` |\n",
        "| `generate_tutorial(output_path)` | function | `gafime.tutorial` |\n",
        "| `DiagnosticReport`, `InteractionResult`, `StabilityResult`, `PermutationResult`, `Decision` | dataclasses | `gafime.reporting` |\n",
        "| `resolve_backend(config, X, y)` | function | `gafime.backends` |\n",
        "| `__version__` | str | `gafime` |\n",
        "\n",
        "Happy mining! 🧠⚡\n"
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    import sys
    nb = build()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gafime_tutorial.ipynb")
    out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({len(nb['cells'])} cells)")


def generate_tutorial(output_path: str = "gafime_tutorial.ipynb") -> str:
    """Write a comprehensive GAFIME starter notebook covering every public API.

    The generated notebook is a guided tour of `GafimeEngine`, `EngineConfig`,
    `ComputeBudget`, `DiagnosticReport`, `GafimeSelector`, `GafimeStreamer`,
    and the CLI — with real data-science usecases (planted signal,
    California housing regression, classification with engineered
    interactions, sklearn pipeline, streaming).

    Args:
        output_path: Destination .ipynb file path.

    Returns:
        The absolute path that was written.
    """
    out = Path(output_path)
    out.write_text(json.dumps(build(), indent=2), encoding="utf-8")
    abs_path = str(out.resolve())
    print(f"✨ Wrote GAFIME tutorial notebook ({len(build()['cells'])} cells) to: {abs_path}")
    print(f"Run `jupyter notebook {output_path}` or open it in VS Code to get started!")
    return abs_path
