# CPU Covariance Finite-Pass Evidence

This record evaluates issue #43 through the public compiled Core API. It is a
bounded local observation, not a universal performance claim.

## Provenance

- base source: `c53a9ac8a13050c6a7184692fc3fe539eeb83f1c`
- candidate source: `adcaab09a938e865e908c620eca58f597dd10b5b`
- host: AMD Ryzen AI 9 HX 370, 24 logical CPUs, Linux x86-64
- pinned logical CPU: `4`
- Python: `3.14.3`
- Rust: `1.89.0`, LLVM `20.1.7`

The base and candidate Python boundaries were built from their exact source
states with isolated Cargo target directories:

```bash
CARGO_TARGET_DIR=build/issue43-candidate-rust \
  cargo +1.89.0 build --release --locked -p gafime-py
```

The measured binaries are bound by:

```text
base PyO3       e0f52a83055bdc1b5a23b9f08b9eb250ccf2b5e1452f2138242747845d49e1ac
candidate PyO3  acad4fbc68f575b58e501f2e678acc49e962413c5236680c8504ad9b04b2ec8d
```

## Workload

`perf_10_cpu_covariance_finite_pass.py` compiles one unary candidate with
1,048,576 deterministic fp32 rows, keeps the compiled Core artifact resident,
and times `artifact.analyze()` after 20 warmups. Each reported distribution has
101 samples. It also checks the expected source-nonfinite diagnostic, finite
metric output, and `r2 == pearson**2` when both metrics are requested.

Each process was pinned to one logical CPU:

```bash
taskset -c 4 env PYTHONPATH=python \
  python tests/release_measure/perf_10_cpu_covariance_finite_pass.py \
  --rows 1048576 --metrics pearson \
  --nonfinite-position none --warmups 20 --repetitions 101
```

The candidate was run before and after the base binary to expose ordinary
run-order variation. The same commands also used `--metrics pearson,r2` and
`--nonfinite-position first|last`.

## Observations

| Workload | Base median | Candidate medians | Base / candidate |
|---|---:|---:|---:|
| finite, Pearson | 1.226 ms | 0.651 ms, 0.665 ms | 1.84x to 1.88x |
| finite, Pearson + R2 | 2.409 ms | 0.662 ms, 0.661 ms | 3.64x to 3.65x |
| first-row NaN, Pearson | 2.019 ms | 2.019 ms, 2.047 ms | 0.99x to 1.00x |
| last-row NaN, Pearson | 2.692 ms | 2.590 ms, 2.608 ms | 1.03x to 1.04x |

Finite outputs were identical across binaries: Pearson
`0.9677826762199402` and R2 `0.936603307723999`. The first- and last-row NaN
workloads also produced identical outputs for each position.

The finite Pearson result reflects removal of one full scalar finiteness pass:
each SIMD implementation tests the values already loaded during the required
sum pass. The Pearson-plus-R2 result additionally reflects exact per-candidate
reuse of the Pearson result instead of computing covariance twice. A fixed
16-row prefix probe retains cheap rejection for early nonfinite input without
restoring an input-sized third pass.

## Boundaries

- Measurements cover the AVX-512 dispatch on this one host and workload.
- SSE4.2, AVX2, and AArch64 NEON implementations are parity-tested and
  compile-validated, but no performance result is claimed for them here.
- The benchmark uses one resident unary candidate. Candidate generation,
  higher-arity interaction materialization, significance, and thread-pool
  scaling are outside the timed region.
- Dynamic clocks and ordinary host contention were not controlled.
- The nonfinite path deliberately retains the scalar filtering oracle. The
  evidence does not claim that every nonfinite position is faster.
- The observed ratios are regression evidence for this implementation and
  machine, not a release-wide throughput guarantee.
