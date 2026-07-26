# Metal CPU-Oracle Parity Evidence

This record approves the absolute fp32 release tolerance used by the Metal
continuous-metric gate. It is correctness evidence, not a throughput result.

## Provenance

- source commit: `23c54bf2f00c7f16a66006e4c29c7fa7cc531664`
- GitHub Actions run: `30207767348`
- job: `89808713080`, `Metal shader, payload, and v1 API validation`
- runner: GitHub-hosted Apple Silicon, `macos-26-arm64`
- runner image: `macos-26-arm64/20260720.0258`
- operating system: macOS 26.4
- provisional measurement tolerance: `0.002`

The workflow compiled the Metal shader and payload from the source commit,
loaded that payload through the v1 C ABI, and compared it with the CPU oracle.
The test covered all 31 arity-1 through arity-5 candidates over a 160-row,
5-feature high-dynamic-range matrix. It ran once with finite inputs and once
with injected NaN and infinity values.

## Maximum Absolute Deltas

| Dataset | Pearson | R2 | Fixed-bin MI | Spearman |
|---|---:|---:|---:|---:|
| finite | `4.045665264e-6` | `9.536743164e-7` | `3.911554813e-8` | `5.960464478e-8` |
| NaN/Inf injected | `1.490116119e-6` | `2.622604370e-6` | `3.119930625e-8` | `1.303851604e-7` |

The worst observation was Pearson at `4.045665264e-6`.

## Approved Bound

`GAFIME_METAL_PARITY_TOLERANCE=0.00005` is the approved absolute tolerance for
this release gate. It is approximately `12.36x` the observed maximum and `40x`
tighter than the former provisional `0.002` guard.

This margin accounts for ordinary Apple-runner and reduction-order variation
without allowing the former two-orders-of-magnitude blind spot. Increasing the
bound requires new Apple-hardware evidence and explicit maintainer approval.

## Boundaries

- This evidence covers the exact dataset, metrics, candidate family, payload,
  compiler environment, and hosted Apple hardware above.
- It does not prove bit equality, universal error bounds for arbitrary input
  distributions, or performance.
- Metal remains fp32-only because Metal Shading Language does not provide fp64
  arithmetic for this kernel path.
- Existing exact-count, candidate-identity, top-k, non-finite, and public API
  gates remain separate requirements.
