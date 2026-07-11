"""compile_02 | gafime.compile value: compiled (plan) path vs eager engine.analyze.
Blocks on result parity (same candidate identities and per-metric values).
Eager and compile-plus-analyze timings are logged as context, not a performance
assertion.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 compile_02_compiled_vs_eager.py
"""
import math
import os

import gafime
from gafime import CompileFlags, EngineConfig
from gafime import GafimeEngine

import _measure_common as mc

ABS_TOL = 1e-7


def freeze_identity(value):
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), freeze_identity(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_identity(item) for item in value)
    return value


def result_map(report):
    results = {}
    for item in (getattr(report, "interactions", []) or []):
        candidate_id = str(getattr(item, "candidate_id", "") or "")
        family = str(getattr(item, "family", "") or "")
        if not candidate_id or not family:
            raise AssertionError(
                f"interaction lacks stable candidate identity: "
                f"family={family!r} candidate_id={candidate_id!r}"
            )
        key = (family, candidate_id)
        if key in results:
            raise AssertionError(f"duplicate candidate identity: {key}")

        metrics = {str(name): float(value) for name, value in item.metrics.items()}
        if not metrics:
            raise AssertionError(f"candidate {key} has no metrics")
        for metric, value in metrics.items():
            if not math.isfinite(value):
                raise AssertionError(
                    f"candidate {key} has non-finite {metric}: {value}"
                )

        results[key] = {
            "combo": tuple(int(value) for value in item.combo),
            "feature_names": tuple(str(value) for value in item.feature_names),
            "expression": str(getattr(item, "expression", "") or ""),
            "params": freeze_identity(getattr(item, "params", {}) or {}),
            "metrics": metrics,
        }
    if not results:
        raise AssertionError("analysis produced no candidate results")
    return results


def significance_map(report, collection_name):
    results = {}
    for item in (getattr(report, collection_name, []) or []):
        candidate_id = str(getattr(item, "candidate_id", "") or "")
        family = str(getattr(item, "family", "") or "")
        if not candidate_id or not family:
            raise AssertionError(
                f"{collection_name} row lacks stable candidate identity: "
                f"family={family!r} candidate_id={candidate_id!r}"
            )
        key = (family, candidate_id)
        if key in results:
            raise AssertionError(f"duplicate {collection_name} identity: {key}")

        value_fields = ("p_values",) if collection_name == "permutations" else (
            "metrics_mean",
            "metrics_std",
        )
        values = {}
        for field in value_fields:
            metrics = {
                str(name): float(value)
                for name, value in getattr(item, field).items()
            }
            if not metrics:
                raise AssertionError(f"{collection_name} row {key} has no {field}")
            for metric, value in metrics.items():
                if not math.isfinite(value):
                    raise AssertionError(
                        f"{collection_name} row {key} has non-finite "
                        f"{field}.{metric}: {value}"
                    )
            values[field] = metrics

        results[key] = {
            "combo": tuple(int(value) for value in item.combo),
            "expression": str(getattr(item, "expression", "") or ""),
            "params": freeze_identity(getattr(item, "params", {}) or {}),
            "values": values,
        }
    return results


def assert_report_parity(eager_report, compiled_report):
    eager = result_map(eager_report)
    compiled = result_map(compiled_report)
    eager_ids = set(eager)
    compiled_ids = set(compiled)
    missing = sorted(eager_ids - compiled_ids)
    extra = sorted(compiled_ids - eager_ids)
    failures = []
    if missing or extra:
        failures.append(
            f"candidate identity mismatch: eager_only={missing[:5]} "
            f"compiled_only={extra[:5]}"
        )

    max_delta = 0.0
    compared_metrics = 0
    for key in sorted(eager_ids & compiled_ids):
        eager_item = eager[key]
        compiled_item = compiled[key]
        for field in ("combo", "feature_names", "expression", "params"):
            if eager_item[field] != compiled_item[field]:
                failures.append(
                    f"{key} {field} differs: eager={eager_item[field]!r} "
                    f"compiled={compiled_item[field]!r}"
                )

        eager_metrics = eager_item["metrics"]
        compiled_metrics = compiled_item["metrics"]
        if eager_metrics.keys() != compiled_metrics.keys():
            failures.append(
                f"{key} metric identities differ: "
                f"eager={sorted(eager_metrics)} compiled={sorted(compiled_metrics)}"
            )
        for metric in sorted(eager_metrics.keys() & compiled_metrics.keys()):
            eager_value = eager_metrics[metric]
            compiled_value = compiled_metrics[metric]
            delta = abs(eager_value - compiled_value)
            max_delta = max(max_delta, delta)
            compared_metrics += 1
            if delta > ABS_TOL:
                failures.append(
                    f"{key} {metric} delta={delta:.6g} exceeds {ABS_TOL}: "
                    f"eager={eager_value:.9g} compiled={compiled_value:.9g}"
                )

    for collection_name in ("permutations", "stability"):
        eager_significance = significance_map(eager_report, collection_name)
        compiled_significance = significance_map(compiled_report, collection_name)
        eager_significance_ids = set(eager_significance)
        compiled_significance_ids = set(compiled_significance)
        missing = sorted(eager_significance_ids - compiled_significance_ids)
        extra = sorted(compiled_significance_ids - eager_significance_ids)
        if missing or extra:
            failures.append(
                f"{collection_name} identity mismatch: eager_only={missing[:5]} "
                f"compiled_only={extra[:5]}"
            )
        for key in sorted(eager_significance_ids & compiled_significance_ids):
            eager_item = eager_significance[key]
            compiled_item = compiled_significance[key]
            for field in ("combo", "expression", "params"):
                if eager_item[field] != compiled_item[field]:
                    failures.append(
                        f"{collection_name} {key} {field} differs: "
                        f"eager={eager_item[field]!r} compiled={compiled_item[field]!r}"
                    )
            if eager_item["values"].keys() != compiled_item["values"].keys():
                failures.append(f"{collection_name} {key} value fields differ")
            for value_field in eager_item["values"].keys() & compiled_item["values"].keys():
                eager_values = eager_item["values"][value_field]
                compiled_values = compiled_item["values"][value_field]
                if eager_values.keys() != compiled_values.keys():
                    failures.append(
                        f"{collection_name} {key} {value_field} metric identities differ"
                    )
                for metric in eager_values.keys() & compiled_values.keys():
                    delta = abs(eager_values[metric] - compiled_values[metric])
                    max_delta = max(max_delta, delta)
                    if delta > ABS_TOL:
                        failures.append(
                            f"{collection_name} {key} {value_field}.{metric} "
                            f"delta={delta:.6g} exceeds {ABS_TOL}"
                        )

    eager_decision = getattr(eager_report, "decision", None)
    compiled_decision = getattr(compiled_report, "decision", None)
    eager_decision_value = None if eager_decision is None else (
        bool(eager_decision.signal_detected),
        str(eager_decision.message),
    )
    compiled_decision_value = None if compiled_decision is None else (
        bool(compiled_decision.signal_detected),
        str(compiled_decision.message),
    )
    if eager_decision_value != compiled_decision_value:
        failures.append(
            f"decision differs: eager={eager_decision_value!r} "
            f"compiled={compiled_decision_value!r}"
        )

    if compared_metrics == 0:
        failures.append("no common candidate metrics were compared")
    if failures:
        detail = "; ".join(failures[:12])
        if len(failures) > 12:
            detail += f"; ... {len(failures) - 12} more"
        raise AssertionError(f"compiled-vs-eager parity failed: {detail}")
    return len(eager), compared_metrics, max_delta


def main():
    X, y, names, meta, _ = mc.load_synthetic_and(n=1500, f=10)
    Xl, yl = X.tolist(), y.tolist()
    backend = os.environ.get("GAFIME_BACKEND", "core")
    cfg = EngineConfig(backend=backend)
    tel = mc.telemetry()

    t0 = tel.monotonic_ns()
    eager = GafimeEngine(config=cfg).analyze(Xl, yl, feature_names=names)
    eager_ns = tel.monotonic_ns() - t0

    t1 = tel.monotonic_ns()
    compiled = gafime.compile(
        Xl, yl, names, config=cfg, flags=CompileFlags(plan=True)
    )
    try:
        comp_report = compiled.analyze()
        comp_ns = tel.monotonic_ns() - t1
    finally:
        compiled.close()

    if cfg.permutation_tests > 0 and (
        not eager.permutations or not comp_report.permutations
    ):
        raise AssertionError("requested eager/compiled permutation rows are missing")
    if cfg.num_repeats > 1 and (not eager.stability or not comp_report.stability):
        raise AssertionError("requested eager/compiled stability rows are missing")
    if eager.decision is None or comp_report.decision is None:
        raise AssertionError("requested eager/compiled significance decision is missing")

    candidates, compared_metrics, max_delta = assert_report_parity(eager, comp_report)
    print(f"eager   candidates={candidates}  time={eager_ns/1e6:.1f}ms")
    print(f"compiled candidates={candidates}  time={comp_ns/1e6:.1f}ms")
    print(
        f"parity metrics={compared_metrics} max|delta|={max_delta:.2e} "
        f"(tol {ABS_TOL})"
    )

    rec = tel.new_record(worktree=mc.WORKTREE, dataset=tel._default_dataset() | meta,
                         config={"backend": backend, "gafime": {"measure": "compiled_vs_eager"}})
    # compiled analyze IS the planning/session zone -> record it canonically; A/B times -> results
    rec["spans_ns"]["gafime_planning_session_report"] = int(comp_ns)
    rec["results"].update({"status": "pass", "eager_analyze_ns": int(eager_ns),
                           "compiled_analyze_ns": int(comp_ns),
                           "candidate_count": candidates,
                           "compared_metric_count": compared_metrics,
                           "max_abs_delta": max_delta})
    tel.write_run(rec, mc.OUTDIR)
    print(f"COMPILED-VS-EAGER: PASS. artifact in {mc.OUTDIR}")


if __name__ == "__main__":
    main()
