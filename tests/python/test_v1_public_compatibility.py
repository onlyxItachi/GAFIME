from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import types

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
_ROOT = _PYTHON_SRC.parent
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(_PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime  # noqa: E402
from gafime import (  # noqa: E402
    CompiledGafime,
    ComputeBudget,
    DecisionPathCandidate,
    EngineConfig,
    GafimeEngine,
    GafimeSelector,
    GafimeStreamer,
    generate_tutorial,
)
from gafime.sklearn import GafimeSelector as ModuleGafimeSelector  # noqa: E402


def test_history_backed_symbols_are_exported_at_top_level():
    expected = {
        "DecisionPathCandidate",
        "FamilySignificanceSupport",
        "GafimeSelector",
        "GafimeStreamer",
        "generate_tutorial",
    }

    assert expected <= set(gafime.__all__)
    assert GafimeSelector is ModuleGafimeSelector


def test_subfunctions_import_is_lazy_and_public():
    proxy = importlib.reload(importlib.import_module("gafime.subfunctions"))

    assert gafime.subfunctions is proxy
    assert "subfunctions" in gafime.__all__
    assert proxy._rust_helpers is None


def test_subfunctions_dir_and_known_helper_forward_to_current_native_module():
    proxy = importlib.reload(importlib.import_module("gafime.subfunctions"))
    native = pytest.importorskip("gafime.gafime_py")

    assert "native_version" in dir(proxy)
    assert proxy._rust_helpers is native
    assert proxy.native_version is native.native_version
    assert proxy.native_version() == native.native_version()


def test_native_cpu_convenience_boundary_routes_all_precision_profiles():
    native = pytest.importorskip("gafime.gafime_py")
    function = native.analyze_continuous_cpu
    signature = inspect.signature(function)

    assert signature.parameters["precision"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["precision"].default == "mixed"

    delta = 2.0**-30
    features = [[1.0 + index * delta] for index in range(16)]
    target = [1.0 + index * delta for index in range(16)]
    scores = {}
    for precision in ("fp32", "mixed", "fp64"):
        report = function(
            features,
            target,
            1,
            1,
            [1],
            precision=precision,
        )
        assert report.precision == precision
        assert report.storage_dtype == ("float64" if precision == "fp64" else "float32")
        assert report.result_dtype == ("float32" if precision == "fp32" else "float64")
        values = report.metric_values(0)
        expected_typecode = "f" if precision == "fp32" else "d"
        assert values.typecode == expected_typecode
        assert report.record(0).metrics.typecode == expected_typecode
        scores[precision] = float(values[0])

    assert scores["fp32"] == 0.0
    assert scores["mixed"] == 0.0
    assert abs(scores["fp64"] - 1.0) < 1.0e-12

    # The five historical positional arguments remain accepted; only the new
    # canonical profile is keyword-only.
    assert function(features, target, 1, 1, [1]).precision == "mixed"
    with pytest.raises(TypeError):
        function(features, target, 1, 1, [1], "fp64")

    class CoercionTrap:
        def __float__(self):
            raise AssertionError("precision validation ran after input coercion")

    with pytest.raises(ValueError, match="fp32, mixed, fp64"):
        function([[CoercionTrap()]], [CoercionTrap()], precision="binary128")


def test_subfunctions_falls_back_to_legacy_helper_names(monkeypatch):
    proxy = importlib.reload(importlib.import_module("gafime.subfunctions"))
    legacy = types.SimpleNamespace(known_legacy_helper=lambda: "legacy")
    attempted = []

    def import_helper(module_name):
        attempted.append(module_name)
        if module_name == "gafime_cpu":
            return legacy
        raise ImportError(f"missing {module_name}")

    monkeypatch.setattr(proxy, "import_module", import_helper)

    assert "known_legacy_helper" in dir(proxy)
    assert proxy.known_legacy_helper() == "legacy"
    assert attempted == ["gafime.gafime_py", "gafime.gafime_cpu", "gafime_cpu"]


def test_subfunctions_exports_and_runs_published_v047_helpers():
    proxy = importlib.reload(importlib.import_module("gafime.subfunctions"))
    pytest.importorskip("gafime.gafime_py")

    scheduler = proxy.BatchScheduler(max_blocks=96)
    assert scheduler.max_blocks() == 96
    assert scheduler.optimal_batch_size() == 96

    cache = proxy.CacheAwareScheduler(
        4, window_size=2, ops=[0, 1], interaction_types=[0], arity=2
    )
    assert cache.window_size() >= 1
    assert cache.total_interactions() >= 1

    encoder = proxy.OTSEncoder(prior=0.25, n_permutations=1)
    encoded = encoder.fit_transform([0, 0, 1], [1.0, 0.0, 1.0])
    assert len(encoded) == 3
    assert encoder.n_categories() == 2

    quality = proxy.DataQualityAnalyzer()
    assert quality.check_alignment([[1.0, 2.0], [3.0, 4.0]]) == 2

    smart = proxy.SmartScheduler(3, 2, 2)
    batch = smart.generate_batch(4)
    assert len(batch) == 5
    assert len(batch[0]) == 4


def test_decision_path_candidate_preserves_v05_data_contract():
    candidate = DecisionPathCandidate(
        features=(2, 0, 2),
        thresholds=(1.5, -0.25, 3.0),
        signs=(-1, 1, 1),
        gain=0.75,
        support=2,
        round_id=3,
        native_candidate_id=11,
        candidate_id="decision_path:11",
    )

    assert candidate.combo == (0, 2)
    assert candidate.params() == {
        "kind": "decision_path",
        "features": (2, 0, 2),
        "thresholds": (1.5, -0.25, 3.0),
        "signs": (-1, 1, 1),
        "gain": 0.75,
        "support": 2,
        "round_id": 3,
        "native_candidate_id": 11,
        "candidate_id": "decision_path:11",
    }


def test_v05_native_report_builder_remains_public_and_python_backed():
    import gafime.reporting as reporting
    from gafime.reporting import NativeReportBuilder

    assert "NativeReportBuilder" in reporting.__all__
    builder = NativeReportBuilder("interaction")
    assert builder.is_native_backed is False
    assert builder.native_handle is None

    builder.append_interaction(
        combo=(1,),
        feature_names=("b",),
        metrics={"pearson": -0.9},
        candidate_id="interaction:b",
    )
    builder.append_interaction(
        combo=(0,),
        feature_names=("a",),
        metrics={"pearson": 0.5},
        candidate_id="interaction:a",
    )
    sequence = builder.sequence()

    assert sequence.kind == "interaction"
    assert sequence.is_native_backed is False
    assert sequence.native_handle is None
    assert sequence.native_indices is None
    assert [item.combo for item in sequence] == [(1,), (0,)]
    assert sequence.top_k(1)[0].candidate_id == "interaction:b"
    assert sequence.top_k(1, metric_name="pearson")[0].candidate_id == "interaction:a"

    builder.append_interaction(
        combo=(2,), feature_names=("c",), metrics={"pearson": 1.0}
    )
    assert len(sequence) == 2


def test_streamer_preserves_csv_batch_and_target_contract(tmp_path):
    pytest.importorskip("polars")
    path = tmp_path / "samples.csv"
    path.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n", encoding="utf-8")
    streamer = GafimeStreamer(path, y_col="target")

    assert streamer.total_rows == 3
    assert streamer.n_features == 2
    assert list(streamer.stream(batch_size=2)) == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0]],
    ]
    assert list(streamer.stream_with_target(batch_size=2)) == [
        ([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0]),
        ([[5.0, 6.0]], [0.0]),
    ]


def test_generate_tutorial_uses_current_public_api(tmp_path):
    path = tmp_path / "tutorial.ipynb"

    assert generate_tutorial(str(path)) == str(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert notebook["nbformat"] == 4
    assert "GafimeSelector" in source
    assert "CompileFlags(export=True)" in source
    assert "enable_decision_path_functions=True, permutation_tests=25" in source
    assert "rediscovers paths for every" in source
    assert "variability conditional on selection" in source
    assert "not out-of-sample generalization" in source
    assert "enable_discrete_functions" not in source


def test_decision_path_docs_disclose_per_target_permutation_rediscovery():
    usage = (_ROOT / "USAGE.md").read_text(encoding="utf-8")
    decision_usage = usage.split("## Decision Paths and Time-Series Candidates", 1)[1]
    capabilities = (_ROOT / "docs" / "capabilities.md").read_text(encoding="utf-8")

    assert "enable_decision_path_functions=True" in decision_usage
    assert "permuted target" in decision_usage
    assert "Significance support" in capabilities
    assert "per-target path rediscovery" in capabilities
    assert "conditional on selection" in capabilities
    assert "does not correct selection bias" in capabilities


def test_compile_artifact_module_path_is_import_compatible():
    artifact_module = importlib.import_module("gafime.compile.artifact")
    compile_package = importlib.import_module("gafime.compile")

    assert artifact_module.CompiledGafime is gafime.CompiledGafime
    assert artifact_module.NativeCompiledGafime is gafime.NativeCompiledGafime
    assert callable(compile_package)


def test_v05_compiled_classmethod_and_scenario_plan_contract():
    engine = GafimeEngine(
        EngineConfig(
            backend="cpu",
            metric_names=("pearson",),
            permutation_tests=0,
            num_repeats=1,
            budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
        )
    )
    artifact = CompiledGafime.from_engine(
        engine,
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]],
        [0.0, 1.0, 2.0],
        ["a", "b"],
    )
    try:
        plan = artifact.scenario_plan
        assert plan.n_features == 2
        assert plan.rows == 3
        assert plan.cols == 2
        assert plan.max_arity == 2
        assert plan.metric_ids == (1,)
        assert plan.feature_candidate_count == 2
        assert plan.planned_count == 3
        assert [descriptor.arity for descriptor in plan.continuous] == [1, 2]
        assert isinstance(plan.warnings, tuple)
        assert len(artifact.analyze().interactions) == plan.planned_count
    finally:
        artifact.close()


def test_max_feature_candidate_bounds_the_native_candidate_family():
    features = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
    ]
    target = [0.0, 1.0, 2.0]

    def analyze(limit):
        return GafimeEngine(
            EngineConfig(
                backend="cpu",
                metric_names=("pearson",),
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(
                    max_comb_size=1,
                    max_combinations_per_k=10,
                    keep_in_vram=False,
                    max_feature_candidate=limit,
                ),
            )
        ).analyze(features, target)

    assert [item.combo for item in analyze(2).interactions] == [(0,), (1,)]
    assert list(analyze(0).interactions) == []
    assert len(analyze(1 << 32).interactions) == 5
    with pytest.raises(ValueError, match="max_feature_candidate"):
        analyze(-2)


def test_v05_decision_path_helpers_remain_importable_and_executable():
    from gafime.decision_path import (
        decision_path_candidate_from_record,
        decision_path_candidate_from_result,
        evaluate_decision_path_candidate,
        score_decision_path_candidates,
    )

    record = types.SimpleNamespace(
        candidate_id=7,
        features=(0, 1),
        thresholds=(0.5, 1.5),
        signs=(1, -1),
        gain=0.75,
        support=2,
        round_id=2,
    )
    candidate = decision_path_candidate_from_record(record)
    values = evaluate_decision_path_candidate(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 2.0]], candidate
    )
    assert values == [0.0, 1.0, 0.0]

    result = types.SimpleNamespace(
        family="decision_path",
        candidate_id="decision_path:7",
        params=candidate.params(),
    )
    assert decision_path_candidate_from_result(result) == candidate

    suite = types.SimpleNamespace(
        score=lambda feature, target: {"sum": sum(feature), "rows": len(target)}
    )
    assert score_decision_path_candidates(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 2.0]],
        [0.0, 1.0, 2.0],
        [candidate],
        suite,
    )[candidate] == {"sum": 1.0, "rows": 3}


def test_native_decision_path_result_without_params_decodes_membership_feature():
    from gafime.decision_path import decision_path_candidate_from_result
    from gafime.reporting import InteractionResult

    result = InteractionResult(
        combo=(2,),
        feature_names=("path[f0>0.5095 & f1>0.5095]",),
        metrics={"pearson": 1.0},
        family="decision_path",
        expression="path[f0>0.5095 & f1>0.5095]",
        params={},
        candidate_id="decision_path:2",
    )

    candidate = decision_path_candidate_from_result(result)

    assert candidate.features == (2,)
    assert candidate.thresholds == (0.5,)
    assert candidate.signs == (1,)
    assert candidate.native_candidate_id == 2
    assert candidate.candidate_id == "decision_path:2"


def test_native_decision_path_result_without_params_rejects_mixed_combo():
    from gafime.decision_path import decision_path_candidate_from_result
    from gafime.reporting import InteractionResult

    result = InteractionResult(
        combo=(0, 2),
        feature_names=("f0", "path[f0>0.5095 & f1>0.5095]"),
        metrics={"pearson": 1.0},
        family="decision_path",
        expression="f0*path[f0>0.5095 & f1>0.5095]",
        params={},
        candidate_id="decision_path:2",
    )

    with pytest.raises(ValueError, match="exactly one generated path-membership"):
        decision_path_candidate_from_result(result)


def test_native_decision_path_result_with_params_rejects_mixed_combo():
    from gafime.decision_path import decision_path_candidate_from_result
    from gafime.reporting import InteractionResult

    result = InteractionResult(
        combo=(0, 2),
        feature_names=("f0", "path[f0>0.5 & f1>0.5]"),
        metrics={"pearson": 1.0},
        family="decision_path",
        params={"features": (0, 1), "thresholds": (0.5, 0.5), "signs": (1, 1)},
        candidate_id="decision_path:2",
    )

    with pytest.raises(ValueError, match="not a standalone DecisionPathCandidate"):
        decision_path_candidate_from_result(result)
