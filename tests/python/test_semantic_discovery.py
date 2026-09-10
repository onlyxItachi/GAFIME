"""Installed tabular consumer contracts, not general feature-quality evidence."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))

from gafime import EngineConfig, semantic

np = pytest.importorskip("numpy")


def _session(data, *, precision="mixed", **limits):
    return semantic.TabularSession(
        data,
        config=EngineConfig(backend="core", precision=precision),
        feature_names=[f"x{i}" for i in range(data.shape[1])],
        row_keys=list(range(data.shape[0])),
        row_domain="consumer-training",
        provenance="training-only fixture",
        **limits,
    )


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_pareto_preserves_tradeoffs_and_primary_only_orders_frontier(precision):
    dtype = np.float64 if precision == "fp64" else np.float32
    a = np.array([-1, -1, 1, 1], dtype=dtype)
    b = np.array([-1, 1, -1, 1], dtype=dtype)
    with _session(
        np.column_stack((a, b, a + b, a * b, a + b)), precision=precision
    ) as session:
        session.begin_round()
        candidates = session.propose(["source"])
        axis_a = semantic.Evidence.reference("axis-a", candidates[0])
        axis_b = semantic.Evidence.reference("axis-b", candidates[1])
        report = session.evaluate(candidates, [axis_a, axis_b])
        frontier = session.select(
            report,
            semantic.SelectionPolicy(
                axis_a,
                limit=5,
                pareto=[(axis_a, "maximize"), (axis_b, "maximize")],
            ),
        )
        assert list(frontier) == [candidates[i] for i in (0, 2, 4, 1)]
        opposed = session.select(
            report,
            semantic.SelectionPolicy(
                axis_a,
                limit=5,
                pareto=[(axis_a, "maximize"), (axis_b, "minimize")],
            ),
        )
        assert list(opposed) == [candidates[0]]


def test_pareto_missing_objective_is_not_an_optional_constraint():
    with _session(np.arange(12, dtype=np.float32).reshape(4, 3)) as session:
        session.begin_round()
        candidates = session.propose(["source"])
        observed = semantic.Evidence.reference("observed", candidates[0])
        absent = semantic.Evidence.labels("not-supplied")
        report = session.evaluate(candidates, [observed, absent])
        optional = semantic.Constraint(absent, minimum=0, missing="ignore")
        assert (
            len(
                session.select(
                    report, semantic.SelectionPolicy(observed, constraints=[optional])
                )
            )
            == 3
        )
        policy = semantic.SelectionPolicy(
            observed,
            constraints=[optional],
            pareto=[(observed, "maximize"), (absent, "maximize")],
        )
        assert len(session.select(report, policy)) == 0
        with pytest.raises(ValueError, match="unavailable"):
            session.select(
                report,
                semantic.SelectionPolicy(
                    observed,
                    missing="error",
                    constraints=[optional],
                    pareto=[(observed, "maximize"), (absent, "maximize")],
                ),
            )
        for objectives in (
            [(observed, "maximize")],
            [(observed, "maximize"), (observed, "minimize")],
            [(observed, "maximize"), (absent, "sideways")],
        ):
            with pytest.raises(ValueError):
                semantic.SelectionPolicy(observed, pareto=objectives)


def test_pareto_work_failure_does_not_retain_or_accept_partial_results():
    with _session(
        np.arange(32, dtype=np.float32).reshape(4, 8), max_work=100
    ) as session:
        session.begin_round()
        candidates = session.propose(["source"])
        a = semantic.Evidence.reference("a", candidates[0])
        b = semantic.Evidence.reference("b", candidates[1])
        report = session.evaluate(candidates, [a, b])
        retained = session.retained_bytes
        with pytest.raises(ValueError, match="Pareto comparison work limit"):
            session.select(
                report,
                semantic.SelectionPolicy(a, pareto=[(a, "maximize"), (b, "maximize")]),
            )
        assert session.retained_bytes == retained
        assert len(session.select(report, semantic.SelectionPolicy(a))) == 8


def _arrow_values(table):
    pa = pytest.importorskip("pyarrow")
    record = pa.array(table)
    return np.column_stack(
        [record.field(name).to_numpy() for name in table.feature_names]
    )


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_fitting_context_is_separate_from_identity_and_acceptance_snapshot(precision):
    dtype = np.float64 if precision == "fp64" else np.float32
    data = np.array([[-2, -3], [-1, -1], [1, 1], [2, 3]], dtype=dtype)
    with _session(data, precision=precision) as session:
        session.begin_round()
        fitted = session.propose_centered_interactions()
        assert len(fitted) == 1
        assert session.diagnostics["fitted_mean_columns"] == 2
        assert session.diagnostics["fitted_mean_rows"] == 8
        assert session.describe(fitted[0])["means"] == [0.0, 0.0]
        original = session.fitting_origins(fitted[0])
        assert len(original) == 1
        assert original[0]["provenance"] == "training-only fixture"
        target = session.frame.labels(
            row_keys=list(range(4)),
            values=[6.0, 1.0, 1.0, 6.0],
            provenance="train labels",
        )
        channel = semantic.Evidence.labels("training", target)
        report = session.evaluate(fitted, [channel])
        accepted = session.select(report, semantic.SelectionPolicy(channel))
        assert report.fitting_origins(fitted[0]) == original
        assert accepted.fitting_origins(0) == original

        second_frame = session.snapshot(
            data * 2,
            feature_names=["x0", "x1"],
            row_keys=list(range(4)),
            row_domain="other-training",
            provenance="independently fitted equal means",
            role="discovery",
        )
        again = session.propose_centered_interactions(frame=second_frame)
        assert again[0] == fitted[0]
        assert len(session.fitting_origins(fitted[0])) == 2
        assert report.fitting_origins(fitted[0]) == original
        assert accepted.fitting_origins(0) == original

        inference = session.snapshot(
            np.array([[10, 11], [12, 13]], dtype=dtype),
            feature_names=["x0", "x1"],
            row_keys=[80, 81],
            row_domain="unlabeled-inference",
            provenance="never fitted",
        )
        with pytest.raises(ValueError, match="discovery"):
            session.propose_centered_interactions(frame=inference)
        assert len(session.fitting_origins(fitted[0])) == 2
        result = session.transform(accepted, inference)
        np.testing.assert_array_equal(_arrow_values(result)[:, 0], [110, 156])
        session.close()
        assert accepted.fitting_origins(0) == original
        np.testing.assert_array_equal(_arrow_values(result)[:, 0], [110, 156])


def test_new_fitted_and_predicate_atom_controls_require_acceptance_before_reuse():
    with _session(np.arange(12, dtype=np.float32).reshape(4, 3)) as session:
        with pytest.raises(ValueError):
            session.propose_centered_interactions()
        assert session.diagnostics["fitted_mean_rows"] == 0
        session.begin_round()
        raw = session.source("x0")
        new_derived = session.softsign(raw)
        before = session.diagnostics
        with pytest.raises(ValueError):
            session.propose_centered_interactions(atoms=[raw, new_derived])
        with pytest.raises(ValueError):
            session.predicate(new_derived, relation="gt", threshold=0)
        assert session.diagnostics == before
        channel = semantic.Evidence.reference("association", raw)
        report = session.evaluate([new_derived], [channel])
        accepted = session.select(report, semantic.SelectionPolicy(channel))
        session.begin_round(accepted)
        predicate = session.predicate(accepted[0], relation="gt", threshold=0.5)
        assert session.describe(predicate)["logical_arity"] == 1
        assert session.describe(predicate)["source_arity"] == 1
        fitted = session.propose_centered_interactions(atoms=[raw, accepted[0]])
        assert len(fitted) == 1
        assert session.describe(fitted[0])["logical_arity"] == 2
        assert session.describe(fitted[0])["source_arity"] == 1


def test_fitted_reference_is_evaluation_origin_not_candidate_fitted_state():
    data = np.array([[-2, -3, 6], [-1, -1, 1], [1, 1, 1], [2, 3, 6]], dtype=np.float32)
    with _session(data) as session:
        session.begin_round()
        reference = session.propose_centered_interactions(
            atoms=[session.source("x0"), session.source("x1")]
        )[0]
        raw = session.source("x2")
        channel = semantic.Evidence.reference("fitted-reference", reference)
        report = session.evaluate([raw], [channel])
        accepted = session.select(report, semantic.SelectionPolicy(channel))
        origin = session.fitting_origins(reference)
        assert len(origin) == 1
        assert report.evaluation_origins(raw) == origin
        assert accepted.evaluation_origins(0) == origin
        assert report.fitting_origins(raw) == accepted.fitting_origins(0) == []

        other = session.snapshot(
            data,
            feature_names=["x0", "x1", "x2"],
            row_keys=list(range(4)),
            row_domain="other-fit",
            provenance="later equal constants",
            role="discovery",
        )
        again = session.propose_centered_interactions(
            atoms=[session.source("x0"), session.source("x1")], frame=other
        )
        assert again[0] == reference
        assert len(session.fitting_origins(reference)) == 2
        assert (
            report.evaluation_origins(raw) == accepted.evaluation_origins(0) == origin
        )
        session.begin_round(accepted)
        derived = session.softsign(accepted[0])
        assert session.fitting_origins(derived) == []
        session.close()
        assert (
            report.evaluation_origins(raw) == accepted.evaluation_origins(0) == origin
        )


def test_overlapping_accepted_batches_use_unique_node_authority():
    with _session(np.arange(4, dtype=np.float32).reshape(4, 1), max_nodes=2) as session:
        session.begin_round()
        raw = session.source("x0")
        channel = semantic.Evidence.reference("self", raw)
        accepted = session.select(
            session.evaluate([raw], [channel]), semantic.SelectionPolicy(channel)
        )
        session.begin_round([accepted, accepted, accepted])
        assert session.softsign(accepted[0]) is not None


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_region_logical_arity_is_distinct_atoms_and_inference_boundaries_are_exact(
    precision,
):
    dtype = np.float64 if precision == "fp64" else np.float32
    data = np.array([[-1], [0], [1], [2]], dtype=dtype)
    with _session(
        data, precision=precision, max_logical_arity=1, max_source_arity=1
    ) as session:
        session.begin_round()
        atom = session.source("x0")
        lower = session.predicate(atom, relation="gt", threshold=0)
        upper = session.predicate(atom, relation="le", threshold=1)
        region = session.decision_region([upper, lower])
        assert session.decision_region([lower, upper]) == region
        assert session.describe(region)["logical_arity"] == 1
        assert session.describe(region)["region_term_count"] == 2
        assert session.describe(region)["source_arity"] == 1
        assert session.describe(lower)["relation"] == "gt"
        assert session.describe(upper)["threshold"] == 1.0
        labels = session.frame.labels(
            row_keys=list(range(4)),
            values=[0.0, 0.0, 1.0, 0.0],
            provenance="region fixture",
        )
        channel = semantic.Evidence.labels("outcome", labels)
        report = session.evaluate([region], [channel])
        accepted = session.select(report, semantic.SelectionPolicy(channel))
        values = session.transform(accepted, session.frame)
        np.testing.assert_array_equal(_arrow_values(values)[:, 0], [0, 0, 1, 0])
        with pytest.raises(ValueError):
            session.decision_region([lower, lower])
        with pytest.raises(ValueError):
            session.decision_region(
                [lower, session.predicate(atom, relation="le", threshold=0)]
            )
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                session.predicate(atom, relation="gt", threshold=bad)
        with pytest.raises(ValueError, match="relation"):
            session.predicate(atom, relation="approximately", threshold=0)


@pytest.mark.parametrize("paradigm", ["supervised", "target_free", "paired", "hybrid"])
def test_fold_local_discovery_delivers_composable_unlabeled_learner_inputs(paradigm):
    """Synthetic interaction fixture tests delivery, not a comparative ML claim."""
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(421)
    train = rng.normal(size=(128, 3)).astype(np.float32)
    test = rng.normal(size=(32, 3)).astype(np.float32)
    target = (train[:, 0] * train[:, 1]).astype(np.float64)
    with _session(train) as session:
        session.begin_round()
        proposals = session.propose_centered_interactions(arities=[2], limit=3)
        reference = semantic.Evidence.reference("redundancy", session.source("x2"))
        view = session.snapshot(
            train * np.float32(1.01),
            feature_names=["x0", "x1", "x2"],
            row_keys=list(range(128)),
            row_domain="consumer-training",
            provenance="declared paired augmentation; no holdout claim",
            role="discovery",
        )
        paired = semantic.Evidence.paired("consistency", view)
        labels = session.frame.labels(
            row_keys=list(range(0, 128, 2)),
            values=target[::2].tolist(),
            provenance="partial labels from training partition only",
        )
        labeled = semantic.Evidence.labels("labeled", labels)
        if paradigm == "supervised":
            channels = [labeled]
            policy = semantic.SelectionPolicy(labeled, limit=2)
        elif paradigm == "target_free":
            channels = [reference]
            policy = semantic.SelectionPolicy(reference, direction="minimize", limit=2)
        elif paradigm == "paired":
            channels = [paired]
            policy = semantic.SelectionPolicy(paired, limit=2)
        else:
            channels = [reference, paired, labeled]
            policy = semantic.SelectionPolicy(
                labeled,
                limit=2,
                constraints=[semantic.Constraint(paired, minimum=0.5)],
                pareto=[(labeled, "maximize"), (reference, "minimize")],
            )
        report = session.evaluate(proposals, channels)
        first = session.select(report, policy)
        assert len(first) > 0
        session.begin_round(first)
        composite = session.softsign(first[0])
        second_report = session.evaluate([composite], channels)
        second = session.select(second_report, policy)
        assert len(second) == 1
        assert session.describe(composite)["logical_arity"] == 1
        assert session.describe(composite)["source_arity"] == 2
        assert second.fitting_origins(0) == first.fitting_origins(0)
        train_features = _arrow_values(session.transform(second, session.frame))
        inference = session.snapshot(
            test,
            feature_names=["x0", "x1", "x2"],
            row_keys=list(range(500, 532)),
            row_domain="holdout-unlabeled",
            provenance="never used in proposal, evidence or fitting",
        )
        before = session.diagnostics["evidence_kernel_calls"]
        output = session.transform(second, inference)
        assert session.diagnostics["evidence_kernel_calls"] == before
        assert output.row_keys == list(range(500, 532))
        assert (
            output.feature_names
            == session.transform(second, session.frame).feature_names
        )
        test_features = _arrow_values(output)
        assert train_features.dtype == test_features.dtype == np.float32
        model = linear_model.LinearRegression().fit(train_features, target)
        prediction = model.predict(test_features)
        assert prediction.shape == (32,)
        assert np.isfinite(prediction).all()
        session.close()
        np.testing.assert_array_equal(_arrow_values(output), test_features)
