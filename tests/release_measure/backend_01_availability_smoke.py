"""backend_01 | BACKEND coverage through the public v1 API."""
import numpy as np

from gafime import ComputeBudget, EngineConfig, GafimeEngine


def main():
    Xraw = np.random.default_rng(0).random((64, 6))
    yraw = (Xraw[:, 0] > 0.5).astype(float)
    X, y = Xraw.tolist(), yraw.tolist()
    print(f"{'requested':<10}{'resolved':<18}{'native?':<9}{'notes'}")
    for name in ("core", "cuda", "rocm", "metal", "auto"):
        try:
            cfg = EngineConfig(
                backend=name,
                permutation_tests=0,
                num_repeats=1,
                budget=ComputeBudget(max_comb_size=1, max_combinations_per_k=16),
            )
            report = GafimeEngine(cfg).analyze(X, y)
            info = report.backend
            print(f"{name:<10}{getattr(info, 'name', '?'):<18}{str(bool(info)):<9}ok")
        except Exception as exc:
            print(f"{name:<10}{'<unavailable>':<18}{'False':<9}{type(exc).__name__}: {str(exc)[:50]}")
    print("\nrecord which backends resolve native on THIS host (4060 sm_89 / gfx1150).")


if __name__ == "__main__":
    main()
