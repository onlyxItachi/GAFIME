"""backend_01 | Backend availability through the public v1 API.

Unconfigured optional payloads are reported as skips only when execution reaches
the expected missing-payload boundary. Configured payload failures, fallback to
another backend, and a selection that completes no backend are release failures.
"""
import numpy as np

from gafime import ComputeBudget, EngineConfig, GafimeEngine

from _backend_contract import (
    assert_resolved_backend,
    is_unconfigured_payload_error,
    payload_is_configured,
    selected_backends,
)


def main():
    Xraw = np.random.default_rng(0).random((64, 6))
    yraw = (Xraw[:, 0] > 0.5).astype(float)
    X, y = Xraw.tolist(), yraw.tolist()
    selected = selected_backends(("core", "cuda", "rocm", "metal", "auto"))
    executed = 0
    skipped = []
    failures = []
    print(f"{'requested':<10}{'resolved':<18}{'native?':<9}{'notes'}")
    for name in selected:
        try:
            cfg = EngineConfig(
                backend=name,
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=16),
            )
            report = GafimeEngine(cfg).analyze(X, y)
            info = report.backend
            assert_resolved_backend(name, info)
            if not list(getattr(report, "interactions", []) or []):
                raise AssertionError(f"{name} availability smoke produced no interactions")
            executed += 1
            print(f"{name:<10}{info.name:<18}{str(bool(info)):<9}PASS")
        except Exception as exc:
            if is_unconfigured_payload_error(name, exc):
                skipped.append(name)
                print(
                    f"{name:<10}{'<unconfigured>':<18}{'False':<9}"
                    f"SKIP: {type(exc).__name__}: {str(exc)[:80]}"
                )
                continue
            configured = payload_is_configured(name)
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            state = "configured" if configured else "unexpected"
            print(
                f"{name:<10}{'<failed>':<18}{'False':<9}"
                f"FAIL ({state}): {type(exc).__name__}: {str(exc)[:80]}"
            )

    print(
        f"\nbackend availability: selected={len(selected)} executed={executed} "
        f"skipped={len(skipped)} failed={len(failures)}"
    )
    if executed == 0:
        failures.append(
            "selected backend suite executed nothing: " + ",".join(selected)
        )
    if failures:
        raise AssertionError("backend availability failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
