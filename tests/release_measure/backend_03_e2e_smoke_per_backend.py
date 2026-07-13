"""backend_03 | Per-backend end-to-end smoke and telemetry.

Unconfigured optional payloads skip only at their expected missing-payload
boundary. Configured failures and selections that complete no backend fail.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 backend_03_e2e_smoke_per_backend.py
"""
from gafime import ComputeBudget, EngineConfig, GafimeEngine

import _measure_common as mc
from _backend_contract import (
    assert_resolved_backend,
    is_unconfigured_payload_error,
    payload_is_configured,
    selected_backends,
)


def main():
    X, y, names, meta, _ = mc.load_synthetic_and(n=2000, f=8)
    Xl, yl = X.tolist(), y.tolist()
    tel = mc.telemetry()
    selected = selected_backends(("core", "cuda", "rocm", "metal"))
    executed = 0
    skipped = []
    failures = []
    for backend in selected:
        rec = tel.new_record(
            worktree=mc.WORKTREE,
            dataset=tel._default_dataset() | meta,
            config={
                "backend": backend,
                "gafime": {"measure": "backend_e2e_smoke"},
            },
        )
        try:
            with tel.span(rec, "e2e_total"):
                report = GafimeEngine(
                    config=EngineConfig(
                        backend=backend,
                        permutation_tests=0,
                        num_repeats=1,
                        budget=ComputeBudget(
                            max_comb_size=2,
                            max_combinations_per_k=64,
                        ),
                    )
                ).analyze(Xl, yl, feature_names=names)
            assert_resolved_backend(backend, report.backend)
            n_inter = len(list(getattr(report, "interactions", []) or []))
            if n_inter == 0:
                raise AssertionError(f"{backend} end-to-end smoke produced no interactions")
            executed += 1
            rec["results"].update({"status": "pass", "interaction_count": n_inter})
            print(
                f"[{backend:<6}] PASS resolved={report.backend.name} "
                f"interactions={n_inter}"
            )
        except Exception as exc:
            if is_unconfigured_payload_error(backend, exc):
                skipped.append(backend)
                rec["results"].update(
                    {
                        "status": "skip",
                        "reason": "payload_unconfigured",
                        "error_type": type(exc).__name__,
                    }
                )
                print(
                    f"[{backend:<6}] SKIP unconfigured payload: "
                    f"{type(exc).__name__}: {str(exc)[:80]}"
                )
            else:
                tel.mark_failed(rec, exc)
                configured = payload_is_configured(backend)
                failures.append(f"{backend}: {type(exc).__name__}: {exc}")
                state = "configured" if configured else "unexpected"
                print(
                    f"[{backend:<6}] FAIL ({state}): "
                    f"{type(exc).__name__}: {str(exc)[:80]}"
                )
        tel.write_run(rec, mc.OUTDIR)
    print(
        f"\nbackend e2e: selected={len(selected)} executed={executed} "
        f"skipped={len(skipped)} failed={len(failures)}"
    )
    print(f"per-backend artifacts in {mc.OUTDIR}")
    if executed == 0:
        failures.append(
            "selected backend suite executed nothing: " + ",".join(selected)
        )
    if failures:
        raise AssertionError("backend e2e failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
