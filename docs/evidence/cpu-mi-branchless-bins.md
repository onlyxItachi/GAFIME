# CPU MI Branchless Bin Evidence

This record evaluates issue #44 through both the public compiled Core API and a
separate internal kernel/helper microbenchmark. It is a bounded local
observation, not a universal performance claim.

## Provenance

- branch base: `680c95814923aa5be5fe7db7f2412978f2cb5258`
- baseline benchmark harness: `a53342a846906a549440389b6009dcb246a0d0a1`
- candidate implementation: `30a3684eb4c78aae44d41296197e478f48b2c3b0`
- host: AMD Ryzen AI 9 HX 370, 24 logical CPUs, Linux x86-64
- pinned logical CPU: `4`
- Python: `3.14.3`
- Rust: `1.89.0`, LLVM `20.1.7`

The retained base PyO3 binary was built at the parent branch's compiled-source
state `adcaab09a938e865e908c620eca58f597dd10b5b`; the later parent commit changed
only documentation and release gates. Base and candidate PyO3 builds used the
same locked Release profile and isolated Cargo target directories:

```bash
CARGO_TARGET_DIR=build/issue44-candidate-rust \
  cargo +1.89.0 build --release --locked -p gafime-py
```

The measured binaries are bound by:

```text
base PyO3             acad4fbc68f575b58e501f2e678acc49e962413c5236680c8504ad9b04b2ec8d
candidate PyO3        ff76a3d8f39e0a6bad755d24be821291e6c8f7f56e88b6ac3b9916952e20a6fe
base Rust test        ed530e980de659188fa90be338e7f771475591c31d283caf82c6ba2e41e54c18
candidate Rust test   de3a275dfef7475959fd6939eb8f6bb86f6e0b17677f53396be7ed225f78f2a5
```

## Public MI Workload

`perf_11_cpu_mi_histogram.py` compiles one unary fixed-bin MI candidate, keeps
the Core artifact resident, and times `artifact.analyze()`. It rejects a shape
unless `rows >= 8*bins^2`, so the named requested bin specialization remains
effective instead of being silently lowered by adaptive quantization.

The one-million-row distributions used 20 warmups and 101 samples. Small and
medium distributions used 50/201 and 30/151 warmup/sample counts,
respectively. Every process was pinned to one logical CPU:

```bash
taskset -c 4 env PYTHONPATH=python \
  python tests/release_measure/perf_11_cpu_mi_histogram.py \
  --rows 1048576 --bins 96 --warmups 20 --repetitions 101
```

| Rows | Bins | Base median | Candidate median | Base / candidate |
|---:|---:|---:|---:|---:|
| 4,096 | 16 | 0.0361 ms | 0.0286 ms | 1.26x |
| 65,536 | 64 | 0.5109 ms | 0.3892 ms | 1.31x |
| 1,048,576 | 24 | 7.332-7.498 ms | 5.349-5.384 ms | 1.36x-1.40x |
| 1,048,576 | 96 | 8.504-8.665 ms | 6.542-6.549 ms | 1.30x-1.32x |

MI values were bit-identical across the base and candidate binaries for every
row above. The one-million-row values were `0.3842184543609619` at 24 bins and
`0.3164912164211273` at 96 bins.

## Internal Mechanism Workload

The ignored `fixed_bin_release_benchmark` test times the direct 96-bin
histogram and the allocating index helper on 1,048,576 rows after 20 warmups,
using 101 samples. The candidate also times the new caller-owned output form.
It is invoked separately from the public benchmark:

```bash
taskset -c 4 cargo +1.89.0 test --release -p gafime-cpu \
  fixed_bin_release_benchmark --locked -- --ignored --nocapture
```

| Operation | Base median | Candidate median | Base / candidate |
|---|---:|---:|---:|
| direct 96-bin 2D histogram | 3.994-4.130 ms | 2.360-2.372 ms | 1.68x-1.75x |
| allocating index helper | 1.270-1.271 ms | 0.186-0.190 ms | 6.70x-6.84x |
| caller-owned index helper | not available | 0.127-0.128 ms | not comparable |

The larger helper ratios are not public MI throughput claims. Production MI
uses the fused histogram, not a materialized vector of bin indices.

## Implementation and Correctness Boundaries

- AVX2 performs truncation, lower/upper clamping, and positive-infinity
  recovery in registers. Only converted integer lanes are stored for the
  unavoidable scalar scatter.
- Histogram increments remain bounds-checked. No unchecked scatter was added.
- The allocating helper remains compatible and delegates to
  `fixed_bin_indices_into`, which accepts caller-owned output storage.
- Bin domains above signed 32-bit conversion capacity use the scalar oracle.
  Production adaptive bins are at most 96.
- Exact tests cover NaN, positive/negative infinity, signed zero, subnormals,
  finite extremes, exact bin boundaries, vector tails, every adaptive bin
  specialization, and the oversized scalar fallback.
- No per-candidate bin-index vector was introduced into the production MI path.
- Measurements cover the AVX2 rung on this one AVX-512-capable host. No
  throughput claim is made for scalar fallback or AArch64.
- Dynamic clocks and ordinary host contention were not controlled. These
  results are regression evidence for this implementation and machine only.
