import json
import os


def generate_tutorial(output_path: str = "gafime_tutorial.ipynb"):
    """
    Generates a starter Jupyter Notebook that demonstrates the real GAFIME API
    with a planted signal so users see meaningful results immediately.

    Args:
        output_path: The filename/path where the notebook should be saved.
    """
    notebook = {
        "cells": [
            # ── 1. Welcome ──────────────────────────────────────────────
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": [
                    "# GAFIME Quickstart Tutorial 🚀\n",
                    "\n",
                    "Welcome to **GAFIME** (GPU-Accelerated Feature Interaction Mining Engine)!\n",
                    "\n",
                    "This notebook walks you through:\n",
                    "1. Creating synthetic data with a **planted interaction signal**\n",
                    "2. Running the GAFIME engine to detect that signal\n",
                    "3. Inspecting the top feature interactions\n",
                    "4. Using `GafimeSelector` inside a scikit-learn pipeline\n",
                    "\n",
                    "GAFIME automatically picks the fastest backend available "
                    "(CUDA → Metal → OpenMP → Rust → Python)."
                ]
            },
            # ── 2. Generate synthetic data ──────────────────────────────
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "setup_data",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "\n",
                    "np.random.seed(42)\n",
                    "n_samples, n_features = 10_000, 20\n",
                    "X = np.random.randn(n_samples, n_features).astype(np.float64)\n",
                    "\n",
                    "# Plant a signal: y correlates with X[:,0] * X[:,1]\n",
                    "y = (X[:, 0] * X[:, 1] + 0.5 * np.random.randn(n_samples)).astype(np.float64)\n",
                    "feature_names = [f\"feature_{i}\" for i in range(n_features)]\n",
                    "\n",
                    "print(f\"Created dataset with {n_samples:,} samples and {n_features} features.\")\n",
                    "print(f\"Planted signal: y ≈ feature_0 × feature_1 + noise\")"
                ]
            },
            # ── 3. Initialize Engine (markdown) ────────────────────────
            {
                "cell_type": "markdown",
                "id": "engine_markdown",
                "metadata": {},
                "source": [
                    "## Initializing the Engine\n",
                    "\n",
                    "Create an `EngineConfig` (optionally with a `ComputeBudget` to control search space), "
                    "then pass it to `GafimeEngine`."
                ]
            },
            # ── 4. Create engine & analyze ─────────────────────────────
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "engine_init",
                "metadata": {},
                "outputs": [],
                "source": [
                    "from gafime import GafimeEngine, EngineConfig, ComputeBudget\n",
                    "\n",
                    "config = EngineConfig(\n",
                    "    budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=5000),\n",
                    "    metric_names=(\"pearson\", \"spearman\", \"mutual_info\", \"r2\"),\n",
                    "    backend=\"auto\",\n",
                    ")\n",
                    "engine = GafimeEngine(config=config)\n",
                    "report = engine.analyze(X, y, feature_names=feature_names)\n",
                    "\n",
                    "print(f\"Backend : {report.backend.name} ({report.backend.device})\")\n",
                    "print(f\"Signal detected: {report.decision.signal_detected}\")\n",
                    "print(f\"Message : {report.decision.message}\")"
                ]
            },
            # ── 5. View top interactions (markdown) ────────────────────
            {
                "cell_type": "markdown",
                "id": "results_markdown",
                "metadata": {},
                "source": [
                    "## Top Feature Interactions\n",
                    "\n",
                    "The `report.interactions` list contains every evaluated combination "
                    "together with its metric scores. Let's sort by the strongest metric "
                    "and print the top 10."
                ]
            },
            # ── 6. Display interactions ─────────────────────────────────
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "show_interactions",
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Top Feature Interactions (sorted by metric strength):\")\n",
                    "sorted_interactions = sorted(\n",
                    "    report.interactions,\n",
                    "    key=lambda x: max(abs(v) for v in x.metrics.values()),\n",
                    "    reverse=True,\n",
                    ")\n",
                    "for rank, interaction in enumerate(sorted_interactions[:10], 1):\n",
                    "    print(f\"  #{rank} {' × '.join(interaction.feature_names)}\")\n",
                    "    for metric, value in interaction.metrics.items():\n",
                    "        print(f\"       {metric}: {value:.4f}\")"
                ]
            },
            # ── 7. Sklearn integration (markdown) ──────────────────────
            {
                "cell_type": "markdown",
                "id": "sklearn_markdown",
                "metadata": {},
                "source": [
                    "## Scikit-learn Integration\n",
                    "\n",
                    "`GafimeSelector` is a drop-in sklearn transformer that mines the best "
                    "feature interactions and appends them to your feature matrix.\n",
                    "\n",
                    "> **Note:** This requires `scikit-learn` (`pip install gafime[sklearn]`)."
                ]
            },
            # ── 8. GafimeSelector pipeline ─────────────────────────────
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "sklearn_pipeline",
                "metadata": {},
                "outputs": [],
                "source": [
                    "try:\n",
                    "    from gafime.sklearn import GafimeSelector\n",
                    "    from sklearn.pipeline import Pipeline\n",
                    "    from sklearn.linear_model import LogisticRegression\n",
                    "    from sklearn.model_selection import train_test_split\n",
                    "\n",
                    "    # Create a binary target for classification\n",
                    "    y_cls = (y > np.median(y)).astype(int)\n",
                    "    X_train, X_test, y_train, y_test = train_test_split(\n",
                    "        X, y_cls, test_size=0.2, random_state=42,\n",
                    "    )\n",
                    "\n",
                    "    selector = GafimeSelector(k=5, backend='auto', metric='pearson', operator='multiply')\n",
                    "    pipe = Pipeline([\n",
                    "        ('miner', selector),\n",
                    "        ('clf', LogisticRegression(max_iter=200)),\n",
                    "    ])\n",
                    "    pipe.fit(X_train, y_train)\n",
                    "    accuracy = pipe.score(X_test, y_test)\n",
                    "    print(f\"Pipeline accuracy: {accuracy:.4f}\")\n",
                    "\n",
                    "except ImportError:\n",
                    "    print(\"Install scikit-learn for sklearn integration: pip install gafime[sklearn]\")"
                ]
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)
        print(f"✨ Successfully generated interactive GAFIME tutorial at: {os.path.abspath(output_path)}")
        print(f"Run `jupyter notebook {output_path}` or open it in VSCode to get started!")
    except Exception as e:
        print(f"Failed to write tutorial notebook to {output_path}: {e}")
