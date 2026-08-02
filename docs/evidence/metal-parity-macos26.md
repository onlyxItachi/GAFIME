# Metal CPU-Oracle Parity Evidence

This record distinguishes the preserved ABI 1.0 compatibility surface from the
ABI 1.1 genuine-fp32 precision surface. It is correctness evidence, not a
throughput result.

## Provenance

- source commit: `23c54bf2f00c7f16a66006e4c29c7fa7cc531664`
- GitHub Actions run: `30207767348`
- job: `89808713080`, `Metal shader, payload, and v1 API validation`
- runner: GitHub-hosted Apple Silicon, `macos-26-arm64`
- runner image: `macos-26-arm64/20260720.0258`
- operating system: macOS 26.4
- provisional measurement tolerance: `0.002`

The workflow compiled the Metal shader and payload from the source commit,
loaded that payload through the legacy ABI 1.0 C surface, and compared it with
the legacy CPU oracle.
The test covered all 31 arity-1 through arity-5 candidates over a 160-row,
5-feature high-dynamic-range matrix. It ran once with finite inputs and once
with injected NaN and infinity values.

## Maximum Absolute Deltas

| Dataset | Pearson | R2 | Fixed-bin MI | Spearman |
|---|---:|---:|---:|---:|
| finite | `4.045665264e-6` | `9.536743164e-7` | `3.911554813e-8` | `5.960464478e-8` |
| NaN/Inf injected | `1.490116119e-6` | `2.622604370e-6` | `3.119930625e-8` | `1.303851604e-7` |

The worst observation was Pearson at `4.045665264e-6`.

## ABI 1.0 Compatibility Bound

`GAFIME_METAL_PARITY_TOLERANCE=0.00005` remains the approved absolute tolerance
for this legacy compatibility gate. It is approximately `12.36x` the observed
maximum and `40x` tighter than the former provisional `0.002` guard.

This margin accounts for ordinary Apple-runner and reduction-order variation
without allowing the former two-orders-of-magnitude blind spot.

## ABI 1.1 Genuine-fp32 Correction

The additive ABI 1.1 route performs fp32 ingest, interaction centering,
reductions, ranking, and public output. Run `30767042591` first exposed a
deterministic reduction-order difference at commit
`c67f886c1feeccf1033fbd574c35096c90b5ee1f`: the shared `0.00005` guard stopped
at a Core value of `0.2004999`, a Metal value of `0.2005889`, and an absolute
delta of about `8.899e-5`.

The correction does not widen the bound. For ABI 1.1 fp32 inputs of up to 256
rows, lane 0 computes the paired finite correlation means in Core row order,
then all 64 lanes retain the parallel fp32 covariance pass. The profile travels
in existing launch-info padding, so the host/MSL structure remains 24 bytes.
ABI 1.0 keeps its established parallel route, and larger ABI 1.1 workloads
remain fully parallel. No f64 accumulation, software-emulated double, final
downcast, or Core fallback is admitted.

The typed gate reports per-metric maxima before enforcing the same approved
`GAFIME_METAL_PARITY_TOLERANCE=0.00005` bound. Increasing that bound requires
new Apple-hardware evidence and explicit maintainer approval. The broader
end-to-end fp32 profile gate separately uses `2e-4` absolute and `2e-5` relative
cross-backend bounds for larger fully parallel workloads; it does not replace
or relax this direct gate.

## Boundaries

- The ABI 1.0 measurements cover the exact dataset, metrics, candidate family,
  payload, compiler environment, and hosted Apple hardware above.
- Final-head ABI 1.1 Apple-hardware results remain a separate required CI
  artifact; the c67 run documents the reduction-order defect corrected here.
- It does not prove bit equality, universal error bounds for arbitrary input
  distributions, or performance.
- Metal remains fp32-only because Metal Shading Language does not provide fp64
  arithmetic for this kernel path.
- Existing exact-count, candidate-identity, top-k, non-finite, and public API
  gates remain separate requirements.
