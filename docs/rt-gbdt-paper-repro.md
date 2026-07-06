# RT GBDT Paper Reproducibility Notes

This document records how to rebuild the RT/GBDT disclosure PDF and how to
rerun the benchmark evidence cited by the paper. The TeX source is the
maintained artifact; the PDF is committed as a convenient review artifact.

## Artifacts

- Source: `docs/rt-gbdt-hardware-ray-tracing-paper.tex`
- Review PDF: `docs/rt-gbdt-hardware-ray-tracing-paper.pdf`
- RT implementation notes: `docs/rt-gbdt-cuda.md`
- Release-measure gate: `tests/release_measure/perf_05_cuda_rt_firsthit_scale.py`

## Rebuild the PDF

The checked-in PDF was built from the TeX source with Tectonic 0.16.9:

```bash
tectonic -o docs -p docs/rt-gbdt-hardware-ray-tracing-paper.tex
```

Expected PDF metadata from `pdfinfo`:

```text
Title:    Accelerating Gradient Boosting Decision Trees via Hardware Ray Tracing Cores and On-Device Fused Reduction
Author:   GAFIME Project
Creator:  LaTeX with hyperref
Producer: xdvipdfmx
Pages:    10
```

After rebuilding, run:

```bash
git diff --check
pdfinfo docs/rt-gbdt-hardware-ray-tracing-paper.pdf | head -20
pdftotext docs/rt-gbdt-hardware-ray-tracing-paper.pdf - | sed -n '1,120p'
```

The text extraction check is intentional. It catches macro-spacing regressions
such as `CUDAowns` or `OptiXprovides` that can pass a TeX compile but make the
paper look unprofessional.

## Benchmark Evidence

The paper's strongest correctness-backed RT result is the partitioned first-hit
score case. It is wired through:

```bash
PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
GAFIME_CUDA_RT_SCALE_BENCH=/tmp/cuda_rt_membership_scale_bench_new \
GAFIME_CUDA_V1_LIB=/tmp/libgafime_cuda_v1.so \
GAFIME_CUDA_RT_FIRSTHIT_CASE=262144x8192 \
GAFIME_CUDA_RT_FIRSTHIT_MIN_GEVALS=1000 \
GAFIME_CUDA_RT_FIRSTHIT_MAX_ABS=1e-4 \
python3 tests/release_measure/perf_05_cuda_rt_firsthit_scale.py
```

The benchmark script expands this to the native benchmark command:

```bash
/tmp/cuda_rt_membership_scale_bench_new \
  --score-only \
  --partitioned-grid \
  --overlap-axis-pairs=8 \
  --firsthit-score \
  --rt-only \
  --repeats=3 \
  262144x8192
```

The gate parses `gpu_rt_score` and `rt_max_abs`, then fails if throughput is
below `GAFIME_CUDA_RT_FIRSTHIT_MIN_GEVALS` or if parity exceeds
`GAFIME_CUDA_RT_FIRSTHIT_MAX_ABS`.

The larger `262144x1048576` first-hit result cited in the paper is
throughput-only evidence. It must stay paired with smaller parity-covered runs
and must not be represented as independent CPU-parity validation.

## Current RT Boundary

The RT path is good enough to pause if the immediate goal is broader backend
hardening:

- Source ownership is explicit: `rt_kernels.cu` and `rt_launcher.cu` own RT
  execution, while generic CUDA files keep only the public C ABI bridge.
- First-hit mode is fail-closed: overlap or unsupported geometry returns
  unsupported instead of silently routing to the SM comparator.
- The v1 architecture gate checks RT symbols, tests, docs, and release-measure
  wiring.
- The paper records the remaining research gaps: real trained ensembles,
  deterministic buffered first-hit statistics, MI/Spearman parity, architecture
  sweep, graph replay, and deeper hardware-counter automation.

## Next Backend Work

The next broad backend pass should be separate from this RT disclosure work.
Recommended scope:

- CUDA and HIP continuous kernels: parity, memory layout, launch shape,
  occupancy, and metric completeness.
- Cross-backend metric support: Pearson, R2, mutual information, and Spearman
  should have explicit support or explicit unsupported behavior per backend.
- Reproducible benchmark telemetry: record candidate-row throughput, memory
  footprint, upload/reuse cost, DRAM behavior where available, and CPU vector
  reference throughput.
- PR gating: keep code changes off `main` until the relevant numerical,
  architecture, and performance gates pass.
