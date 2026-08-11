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

## Corrected ABI 1.1 Apple-Hardware Evidence

Implementation commit `86ef2f875a20bdb84d6d90e1dce13053be032b55` was
validated by V1 Contract run `30770770860`, job `91557414422`, on the hosted
`macos-26` Apple Silicon runner. The offline shader compile passed with
`-fno-fast-math`; the exact integer MI histogram remained parallel and its fp32
probability/logarithm/final-score pass used deterministic row-major bin order.
The approved `0.00005` short-vector bound was not changed.

Across the typed ABI 1.1 cases at 160, 255, and 256 rows, including both finite
and injected NaN/Inf inputs, the worst absolute deltas were:

| Pearson | R2 | Fixed-bin MI | Spearman |
|---:|---:|---:|---:|
| `4.633888602e-5` | `8.523464203e-6` | `5.960464478e-8` | `1.788139343e-7` |

The separate 257-row fully parallel case remained inside its existing `2e-4`
bound: its worst Pearson delta was `9.143166244e-5`. Native Platform run
`30770770854`, job `91557414412`, also passed the installed-wheel, top-level
profile, legacy-payload compatibility, and adversarial MI-boundary checks. In
particular, the earlier Metal `0.0` versus Core `0.015706634148955345` MI
failure no longer occurs.

## Boundaries

- The ABI 1.0 measurements cover the exact dataset, metrics, candidate family,
  payload, compiler environment, and hosted Apple hardware above.
- Required checks still rerun on the final reviewed PR head; this record binds
  the numerical correction to the implementation commit and named physical
  Apple jobs above.
- It does not prove bit equality, universal error bounds for arbitrary input
  distributions, or performance.
- Metal remains fp32-only because Metal Shading Language does not provide fp64
  arithmetic for this kernel path.
- Existing exact-count, candidate-identity, top-k, non-finite, and public API
  gates remain separate requirements.
