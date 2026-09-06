from __future__ import annotations

import os
import sys
import threading
from array import array
from pathlib import Path

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime
from gafime import EngineConfig, semantic

SEMANTIC_PUBLIC_EXPORTS = (
    "AcceptedSet",
    "Candidate",
    "CandidateSet",
    "Constraint",
    "Evidence",
    "EvidenceReport",
    "FeatureTable",
    "Graph",
    "Labels",
    "SelectionPolicy",
    "Snapshot",
    "TabularSession",
)

_ROWS = (
    (0.0, 10.0),
    (1.0, 11.0),
    (2.0, 12.0),
    (3.0, 13.0),
)
_NAMES = ("left", "right")
_KEYS = (101, 102, 103, 104)


def _matrix(
    rows: tuple[tuple[float, ...], ...] | list[tuple[float, ...]],
    *,
    typecode: str = "f",
) -> tuple[array, memoryview]:
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise AssertionError("semantic fixture must be a nonempty rectangular matrix")
    storage = array(typecode, (value for row in rows for value in row))
    view = memoryview(storage).cast("B").cast(typecode, shape=(len(rows), len(rows[0])))
    return storage, view


def _session(
    rows: tuple[tuple[float, ...], ...] | list[tuple[float, ...]] = _ROWS,
    *,
    row_keys: tuple[int, ...] | list[int] = _KEYS,
    row_domain: str = "public-boundary",
    provenance: str = "public-boundary-input",
    config: EngineConfig | None = None,
    role: str = "discovery",
    typecode: str = "f",
    **limits: int,
) -> tuple[semantic.TabularSession, array]:
    storage, matrix = _matrix(rows, typecode=typecode)
    session = semantic.TabularSession(
        matrix,
        feature_names=list(_NAMES),
        row_keys=list(row_keys),
        row_domain=row_domain,
        provenance=provenance,
        config=config,
        role=role,
        **limits,
    )
    return session, storage


def _labels(session: semantic.TabularSession) -> semantic.Labels:
    return session.frame.labels(
        row_keys=list(_KEYS),
        values=[0.0, 1.0, 2.0, 3.0],
        provenance="public-boundary-labels",
    )


def _select_left(
    session: semantic.TabularSession,
) -> tuple[
    semantic.Candidate, semantic.Evidence, semantic.EvidenceReport, semantic.AcceptedSet
]:
    assert session.begin_round() == 1
    left = session.source("left")
    channel = semantic.Evidence.labels("outcome", _labels(session))
    report = session.evaluate([left], [channel])
    accepted = session.select(
        report,
        semantic.SelectionPolicy(channel, direction="maximize", limit=1),
    )
    assert len(accepted) == 1
    return left, channel, report, accepted


def test_semantic_namespace_is_module_scoped_and_handles_are_opaque() -> None:
    assert gafime.semantic is semantic
    assert "semantic" in gafime.__all__
    assert tuple(semantic.__all__) == SEMANTIC_PUBLIC_EXPORTS
    assert set(semantic.__all__) == set(SEMANTIC_PUBLIC_EXPORTS)
    assert not any(name in gafime.__all__ for name in SEMANTIC_PUBLIC_EXPORTS)

    for name in SEMANTIC_PUBLIC_EXPORTS:
        value = getattr(semantic, name)
        assert value.__module__ == "gafime.semantic"

    with pytest.raises(TypeError):
        semantic.Candidate()
    with pytest.raises(TypeError):
        semantic.CandidateSet()
    with pytest.raises(TypeError):
        semantic.AcceptedSet()


def test_session_copies_input_and_reports_operation_specific_capabilities() -> None:
    session, storage = _session()
    try:
        assert session.configured_backend == "auto"
        assert session.selected_backend == "core"
        assert session.precision == "mixed"
        assert session.retained_bytes == 0
        assert session.frame.feature_names == list(_NAMES)
        assert session.frame.row_keys == list(_KEYS)
        assert session.frame.row_domain == "public-boundary"
        assert session.frame.role == "discovery"

        capabilities = session.capabilities
        assert capabilities == {
            "configured_backend": "auto",
            "selected_backend": "core",
            "precision": "mixed",
            "configured_device_id": 0,
            "selected_device_id": None,
            "programs": [
                "source",
                "absolute_difference",
                "softsign",
                "centered_product",
            ],
            "statistics": ["pearson", "spearman", "fixed_nmi", "graph_energy"],
            "contexts": ["reference", "paired_view", "labels", "graph"],
            "selection_reason": (
                "Core supports the complete tabular semantic vocabulary; supervised "
                "GPU route support alone is insufficient"
            ),
            "source": "static",
        }
        assert session.diagnostics == {
            "materialized_nodes": 0,
            "retained_hits": 0,
            "source_shares": 0,
            "output_allocations": 0,
            "output_bytes": 0,
            "evidence_kernel_calls": 0,
            "retained_bytes": 0,
        }

        assert session.begin_round() == 1
        left = session.source("left")
        description = session.describe(left)
        assert description["operation"] == "source"
        assert description["sources"] == ["left"]
        assert description["logical_arity"] == 1
        assert description["source_arity"] == 1
        assert description["depth"] == 0
        assert description["precision"] == "mixed"
        assert len(description["operands"]) == 0
        assert "id" not in description

        storage[0] = 100.0
        channel = semantic.Evidence.labels("outcome", _labels(session))
        report = session.evaluate([left], [channel])
        assert report.context == {
            "row_domain": "public-boundary",
            "provenance": "public-boundary-input",
            "rows": 4,
            "role": "discovery",
            "channels": [
                {
                    "name": "outcome",
                    "semantics": "absolute-pearson/labeled-subset/v1",
                    "bins": None,
                    "kind": "labels",
                    "provenance": "public-boundary-labels",
                }
            ],
        }
        value = report.value(left, channel)
        assert value == {
            "state": "measured",
            "value": pytest.approx(1.0),
            "support": 4,
            "reason": None,
        }
        assert session.diagnostics["evidence_kernel_calls"] == 1
        assert session.diagnostics["source_shares"] >= 1
    finally:
        session.close()


def test_semantic_constructor_rejects_implicit_legacy_controls_and_invalid_ingest() -> (
    None
):
    storage, matrix = _matrix(_ROWS)
    constructor = {
        "feature_names": list(_NAMES),
        "row_keys": list(_KEYS),
        "row_domain": "validation",
        "provenance": "validation-input",
    }

    with pytest.raises(ValueError, match="not a tabular semantic control"):
        semantic.TabularSession(
            matrix,
            config=EngineConfig(metric_names=("pearson",)),
            **constructor,
        )
    with pytest.raises(NotImplementedError, match="not supported"):
        semantic.TabularSession(
            matrix,
            config=EngineConfig(backend="cuda"),
            **constructor,
        )
    for alias in ("cpu", " RUST ", "v1-rust-cpu"):
        alias_session, alias_storage = _session(config=EngineConfig(backend=alias))
        try:
            assert alias_session.configured_backend == "core"
            assert alias_session.capabilities["configured_backend"] == "core"
        finally:
            alias_session.close()
            _ = alias_storage
    with pytest.raises(NotImplementedError, match="not supported"):
        semantic.TabularSession(
            matrix,
            config=EngineConfig(backend=" HIP "),
            **constructor,
        )

    configured_session, configured_storage = _session(config=EngineConfig(device_id=7))
    try:
        assert configured_session.capabilities["configured_device_id"] == 7
        assert configured_session.capabilities["selected_device_id"] is None
    finally:
        configured_session.close()
        _ = configured_storage
    with pytest.raises(ValueError, match="feature_names are required"):
        semantic.TabularSession(
            matrix,
            row_keys=list(_KEYS),
            row_domain="validation",
            provenance="validation-input",
        )
    with pytest.raises(ValueError):
        semantic.TabularSession(
            matrix,
            feature_names=list(_NAMES),
            row_keys=[101, 101, 103, 104],
            row_domain="validation",
            provenance="validation-input",
        )
    with pytest.raises(ValueError):
        one_storage, one_row = _matrix([(1.0, 2.0)])
        _ = one_storage
        semantic.TabularSession(
            one_row,
            feature_names=list(_NAMES),
            row_keys=[1],
            row_domain="validation",
            provenance="validation-input",
        )
    with pytest.raises(ValueError):
        nonfinite_storage, nonfinite = _matrix([(0.0, 1.0), (float("nan"), 2.0)])
        _ = nonfinite_storage
        semantic.TabularSession(
            nonfinite,
            feature_names=list(_NAMES),
            row_keys=[1, 2],
            row_domain="validation",
            provenance="validation-input",
        )
    with pytest.raises((BufferError, TypeError, ValueError)):
        semantic.TabularSession(
            matrix,
            config=EngineConfig(precision="fp64"),
            **constructor,
        )

    fp64_session, fp64_storage = _session(
        config=EngineConfig(precision="fp64"), typecode="d"
    )
    try:
        assert fp64_session.precision == "fp64"
        assert fp64_session.frame.precision == "fp64"
    finally:
        fp64_session.close()
        _ = fp64_storage
        _ = storage


def test_contextual_evidence_keeps_channels_distinct_and_supports_hybrid_policy() -> (
    None
):
    session, _ = _session()
    try:
        assert session.begin_round() == 1
        left = session.source("left")
        right = session.source("right")
        paired_storage, paired_matrix = _matrix(
            ((0.5, 10.5), (1.5, 11.5), (2.5, 12.5), (3.5, 13.5))
        )
        paired = session.snapshot(
            paired_matrix,
            feature_names=list(_NAMES),
            row_keys=list(_KEYS),
            row_domain="public-boundary",
            provenance="public-boundary-paired",
            role="discovery",
        )
        reference = semantic.Evidence.reference("redundancy", right)
        consistency = semantic.Evidence.paired("view-consistency", paired)
        labels = semantic.Evidence.labels("outcome", _labels(session))
        graph = session.frame.graph(
            left_keys=[101, 102, 103],
            right_keys=[102, 103, 104],
            weights=[1.0, 1.0, 1.0],
            provenance="public-boundary-graph",
        )
        smoothness = semantic.Evidence.graph("smoothness", graph)

        report = session.evaluate([left], [reference, consistency, labels, smoothness])
        assert report.backend == "core"
        assert report.precision == "mixed"
        assert report.provenance == "public-boundary-input"
        assert len(report.candidates) == 1
        assert report.candidates[0] == left
        assert report.value(left, reference)["state"] == "measured"
        assert report.value(left, consistency)["state"] == "measured"
        assert report.value(left, labels)["state"] == "measured"
        assert report.value(left, smoothness)["state"] == "measured"
        assert report.value(left, smoothness)["support"] == 3
        assert reference.semantics == "absolute-pearson/reference/v1"
        assert consistency.semantics == "signed-pearson/aligned-view/v1"
        assert labels.semantics == "absolute-pearson/labeled-subset/v1"
        assert smoothness.semantics == "uncentered-edge-energy-ratio/v1"

        rebound_reference = reference.rebind_reference(left)
        rebound_paired = consistency.rebind_paired(paired)
        rebound_labels = labels.rebind_labels(_labels(session))
        rebound_graph = smoothness.rebind_graph(graph)
        for original, rebound in (
            (reference, rebound_reference),
            (consistency, rebound_paired),
            (labels, rebound_labels),
            (smoothness, rebound_graph),
        ):
            assert rebound.name == original.name
            assert rebound.semantics == original.semantics

        accepted = session.select(
            report,
            semantic.SelectionPolicy(
                labels,
                direction="maximize",
                limit=1,
                constraints=[semantic.Constraint(reference, minimum=0.99)],
            ),
        )
        assert len(accepted) == 1
        assert len(report.__arrow_c_array__()) == 2
        _ = paired_storage
    finally:
        session.close()


def test_missing_labels_and_invalid_context_inputs_fail_closed() -> None:
    session, _ = _session()
    try:
        assert session.begin_round() == 1
        left = session.source("left")
        missing = semantic.Evidence.labels("missing-labels")
        report = session.evaluate([left], [missing])
        assert report.value(left, missing) == {
            "state": "unavailable",
            "value": None,
            "support": 0,
            "reason": "missing_labels",
        }
        assert (
            len(session.select(report, semantic.SelectionPolicy(missing, limit=1))) == 0
        )
        with pytest.raises(ValueError, match="required selection evidence"):
            session.select(
                report,
                semantic.SelectionPolicy(missing, limit=1, missing="error"),
            )
        with pytest.raises(ValueError, match="ignore"):
            semantic.SelectionPolicy(missing, missing="ignore")
        with pytest.raises(ValueError, match="finite ordered"):
            semantic.Constraint(missing, minimum=1.0, maximum=0.0)
        with pytest.raises(ValueError, match="exact supported bin"):
            semantic.Evidence.reference("bad-nmi", left, statistic="fixed_nmi", bins=3)
        with pytest.raises(ValueError, match="statistic must"):
            semantic.Evidence.reference("bad-bins", left, statistic="pearson", bins=2)
        with pytest.raises(ValueError, match="duplicate labeled row"):
            session.frame.labels(
                row_keys=[101, 101],
                values=[1.0, 2.0],
                provenance="duplicate-labels",
            )
        with pytest.raises(ValueError, match="neighbor graph"):
            session.frame.graph(
                left_keys=[101],
                right_keys=[101],
                weights=[1.0],
                provenance="self-loop",
            )

        other_storage, other_matrix = _matrix(_ROWS)
        other = session.snapshot(
            other_matrix,
            feature_names=list(_NAMES),
            row_keys=list(_KEYS),
            row_domain="public-boundary",
            provenance="other-snapshot",
            role="discovery",
        )
        foreign_labels = other.labels(
            row_keys=list(_KEYS),
            values=[0.0, 1.0, 2.0, 3.0],
            provenance="other-labels",
        )
        with pytest.raises(ValueError, match="another input context"):
            session.evaluate(
                [left], [semantic.Evidence.labels("foreign", foreign_labels)]
            )
        _ = other_storage
    finally:
        session.close()


def test_proposal_rounds_and_accepted_reuse_are_deterministic_and_owner_bound() -> None:
    session, _ = _session()
    foreign_session, _ = _session(provenance="foreign-input")
    try:
        with pytest.raises(ValueError, match="no active discovery round"):
            session.propose(["source"], limit=1)
        assert session.begin_round() == 1
        left = session.source("left")
        right = session.source("right")
        requested_operators = ["absolute_difference", "source", "softsign"]
        proposed = session.propose(requested_operators, limit=8)
        repeated = session.propose(requested_operators, limit=8)
        assert 1 <= len(proposed) <= 8
        assert [proposed[index] for index in range(len(proposed))] == [
            repeated[index] for index in range(len(repeated))
        ]
        operation_rank = {
            "absolute_difference": 0,
            "source": 1,
            "softsign": 2,
        }
        assert [
            operation_rank[session.describe(proposed[index])["operation"]]
            for index in range(len(proposed))
        ] == sorted(
            operation_rank[session.describe(proposed[index])["operation"]]
            for index in range(len(proposed))
        )
        prefix = session.propose(requested_operators, limit=2)
        assert [session.describe(prefix[index])["operation"] for index in range(2)] == [
            "absolute_difference",
            "source",
        ]
        restricted = session.propose(["softsign"], atoms=[left], limit=1)
        assert len(restricted) == 1
        assert session.describe(restricted[0])["sources"] == ["left"]
        with pytest.raises(ValueError, match="unique"):
            session.propose(["source", "source"], limit=1)
        with pytest.raises(ValueError, match="between one"):
            session.propose(["source"], limit=0)
        assert session.absolute_difference(left, right) == session.absolute_difference(
            right, left
        )
        centered = session.centered_product([left, right], [1.0, 11.0])
        centered_description = session.describe(centered)
        assert centered_description["operation"] == "centered_product"
        assert centered_description["means"] == [1.0, 11.0]
        assert centered_description["sources"] == ["left", "right"]
        with pytest.raises(ValueError, match="length"):
            session.centered_product([left, right], [1.0])

        derived = session.softsign(left)
        channel = semantic.Evidence.labels("outcome", _labels(session))
        report = session.evaluate([derived], [channel])
        accepted = session.select(report, semantic.SelectionPolicy(channel, limit=1))
        assert len(accepted) == 1

        assert session.begin_round() == 2
        with pytest.raises(ValueError, match="previous discovery round"):
            session.select(report, semantic.SelectionPolicy(channel, limit=1))
        with pytest.raises(ValueError, match="eligible atom"):
            session.softsign(accepted[0])
        assert session.begin_round(accepted) == 3
        assert (
            session.describe(session.softsign(accepted[0]))["operation"] == "softsign"
        )

        assert foreign_session.begin_round() == 1
        with pytest.raises(ValueError, match="different owner"):
            foreign_session.softsign(left)
        with pytest.raises(ValueError, match="different owner"):
            foreign_session.begin_round(accepted)
    finally:
        session.close()
        foreign_session.close()


def test_later_round_can_union_independently_accepted_batches() -> None:
    session, _ = _session()
    foreign_session, _ = _session(provenance="foreign-union-input")
    try:
        assert session.begin_round() == 1
        left = session.source("left")
        softened_left = session.softsign(left)
        channel = semantic.Evidence.labels("outcome", _labels(session))
        report = session.evaluate([left, softened_left], [channel])

        strongest = session.select(
            report, semantic.SelectionPolicy(channel, direction="maximize", limit=1)
        )
        weakest = session.select(
            report, semantic.SelectionPolicy(channel, direction="minimize", limit=1)
        )
        assert len(strongest) == len(weakest) == 1
        assert strongest[0] != weakest[0]

        assert session.begin_round([strongest, weakest]) == 2
        composite = session.absolute_difference(strongest[0], weakest[0])
        assert session.describe(composite)["operation"] == "absolute_difference"

        with pytest.raises(ValueError, match="different owner"):
            foreign_session.begin_round([strongest, weakest])
    finally:
        session.close()
        foreign_session.close()


def test_single_row_inference_transform_preserves_arrow_output_after_close() -> None:
    session, _ = _session()
    try:
        _, channel, _, accepted = _select_left(session)
        incompatible_storage, incompatible_matrix = _matrix([(9.0, 19.0)])
        incompatible = session.snapshot(
            incompatible_matrix,
            feature_names=["other-left", "other-right"],
            row_keys=[9001],
            row_domain="inference-boundary",
            provenance="incompatible-inference",
        )
        with pytest.raises(ValueError, match="schema mismatch"):
            session.transform(accepted, incompatible)

        inference_storage, inference_matrix = _matrix([(9.0, 19.0)])
        inference = session.snapshot(
            inference_matrix,
            feature_names=list(_NAMES),
            row_keys=[9001],
            row_domain="inference-boundary",
            provenance="single-row-inference",
        )
        assert inference.role == "inference"
        assert inference.rows == 1
        with pytest.raises(ValueError, match="inference frames cannot be used"):
            session.evaluate([accepted[0]], [channel], frame=inference)
        table = session.transform(accepted, inference)
        assert table.feature_names == ["feature_0"]
        assert table.row_keys == [9001]
        assert table.rows == 1
        assert table.precision == "mixed"
        assert len(table.__arrow_c_array__()) == 2
        with pytest.raises(ValueError, match="schema casting"):
            table.__arrow_c_array__(object())

        session.close()
        assert table.row_keys == [9001]
        assert len(table.__arrow_c_array__()) == 2
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.frame
        _ = incompatible_storage
        _ = inference_storage
    finally:
        session.close()


def test_resource_bounds_and_thread_affinity_fail_before_unbounded_work() -> None:
    limited, _ = _session(max_work=1)
    try:
        assert limited.begin_round() == 1
        left = limited.source("left")
        channel = semantic.Evidence.labels("outcome", _labels(limited))
        with pytest.raises(ValueError, match="work limit"):
            limited.evaluate([left], [channel])
    finally:
        limited.close()

    session, _ = _session(max_rounds=1)
    try:
        assert session.begin_round() == 1
        with pytest.raises(ValueError, match="round resource limit"):
            session.begin_round()

        errors: list[BaseException] = []

        def read_from_other_thread() -> None:
            try:
                _ = session.frame
            except RuntimeError as exc:  # PyO3 reports thread affinity at call time.
                errors.append(exc)

        worker = threading.Thread(target=read_from_other_thread)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
    finally:
        session.close()

    bounded, _ = _session()
    try:
        assert bounded.begin_round() == 1
        left = bounded.source("left")

        class MisreportedSequence:
            def __init__(self) -> None:
                self.indexed: list[int] = []
                self.iterated = False

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int):
                self.indexed.append(index)
                if index == 0:
                    return left
                raise IndexError(index)

            def __iter__(self):
                self.iterated = True
                raise AssertionError("candidate atom iterators must not be consumed")

        atoms = MisreportedSequence()
        proposed = bounded.propose(["softsign"], atoms=atoms, limit=1)
        assert len(proposed) == 1
        assert atoms.indexed == [0]
        assert not atoms.iterated

        class IterableOnly:
            def __len__(self) -> int:
                return 1

            def __iter__(self):
                raise AssertionError(
                    "iterable-only atoms must be rejected without iteration"
                )

        with pytest.raises(TypeError):
            bounded.propose(["softsign"], atoms=IterableOnly(), limit=1)
    finally:
        bounded.close()
