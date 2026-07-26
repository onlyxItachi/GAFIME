# Interaction Overflow Diagnostic Evidence

This record measures the safe finite-input overhead of issue #37. It is a
bounded local observation, not a universal performance claim.

## Provenance

- base source: `887d5954e4a74fba64b33cf6038d9be3a2905490`
- candidate source: `664e5c815ea6c664fcc7699e4cbdf887db549d90`
- host: AMD Ryzen AI 9 HX 370, 24 logical CPUs, Linux x86-64
- CUDA device: NVIDIA GeForce RTX 4060 Laptop GPU, driver `610.43.02`
- ROCm device: AMD Radeon Graphics `gfx1150`
- Python: `3.12.13`
- Rust: `1.89.0`, LLVM `20.1.7`
- CUDA: `13.3.73`
- HIP: `7.1.52801-9999`, Clang `21.1.8`
- CMake: `4.3.2`

Both Python boundaries were rebuilt with the same command and isolated Cargo
target directories:

```bash
CARGO_TARGET_DIR=build/issue37-perf-rust \
  cargo +1.89.0 build --release --locked -p gafime-py
```

The base and candidate CUDA/ROCm payloads were separate explicit Release native
builds. CUDA targeted `sm_89`; HIP targeted `gfx1150`; both used the default
fp32 MI accumulator. The measured binaries are bound by:

```text
candidate CUDA  3b97ab8619e52d2b22ae198bd64cbb0d7ee36e961cf924fa974d3d3c3058ac37
candidate ROCm  aec241d2faab91397c3cb3f1bed5f23bb3820da725419ec82d140c03859ccb0d
base CUDA       68f365daf399045f4e48bd49177769f33905f8fa1bb3022f89ba4e8b39741d81
base ROCm       b3220f77d0640adee25ffa96fb06d586020d7f8087f94eadad21bc94626aba71
candidate PyO3  e0f52a83055bdc1b5a23b9f08b9eb250ccf2b5e1452f2138242747845d49e1ac
base PyO3       61a523047514f95706b705eb268e685db78563d651e765ad0b43ac45ddd4703d
```

## Workload

`perf_09_interaction_diagnostics_overhead.py` uses the public API with a
deterministic finite `2048 x 12` matrix, Pearson scoring, arities one through
five, and 1,585 surfaced candidates. It verifies candidate count, diagnostic
availability, and zero false-positive overflow or source-nonfinite flags.

Each timing distribution below used 20 warmups and 51 samples:

```bash
python tests/release_measure/perf_09_interaction_diagnostics_overhead.py \
  --backend BACKEND \
  --expect-diagnostics yes \
  --warmups 20 \
  --repetitions 51
```

The same command used `--expect-diagnostics no` in the detached base worktree.
CUDA and ROCm runs set the matching `GAFIME_*_V1_LIB` to the hash-bound payload.

## Observations

The final paired trial reported:

| Backend | One-shot base | One-shot candidate | First resident base | First resident candidate | Steady resident base | Steady resident candidate |
|---|---:|---:|---:|---:|---:|---:|
| Core | 5.555 ms | 1.820 ms | 1.015 ms | 0.868 ms | 0.838 ms | 0.856 ms |
| CUDA | 5.307 ms | 1.615 ms | 0.259 ms | 0.206 ms | 0.155 ms | 0.141 ms |
| ROCm | 6.678 ms | 2.630 ms | 1.349 ms | 1.091 ms | 0.698 ms | 0.785 ms |

Across post-cache paired trials, candidate-minus-base steady resident median
deltas were:

| Backend | Observed paired deltas |
|---|---|
| Core | +0.018 ms, -0.103 ms |
| CUDA | -0.014 ms, +0.002 ms |
| ROCm | +0.019 ms, -0.191 ms, +0.087 ms |

The integrated ROCm device showed the widest run-to-run variance. The numbers
do not justify attributing the lower one-shot medians to this change, and no
speedup is claimed. They show no pathological safe-path overhead in this local
workload. Exact-shape compiled execution reuses the diagnostic result; target
or planning changes invalidate that state before reuse.

## Boundaries

- Core, CUDA `sm_89` runtime, and ROCm `gfx1150` runtime were exercised locally.
- The CUDA payload contained native `sm_89` code; no PTX-JIT comparison was
  measured.
- Metal was not compiled or timed locally. Its source and public behavior
  require the hosted Apple Silicon workflow before merge.
- Safe products use upload/transpose metadata and a prefix proof. Only an
  unproven surfaced combination receives an exact row scan.
- Overflow diagnostics are observational. They do not recover a lost product
  or change scores, ranking, significance, graph state, or candidate identity.
- Dynamic clocks and ordinary host/device contention were not controlled.
  These results are regression evidence for this machine and workload only.
