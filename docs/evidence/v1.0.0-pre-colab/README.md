# GAFIME v1.0.0 pre-Colab checkpoint

This evidence branch records the official release candidate and its qualification results. Use the source commit below to reproduce the candidate or run its measurement suite.

| Identity | Value |
| --- | --- |
| Source | [`06d82a4082e96f0821abe0e323ea91f4c7ee6744`](https://github.com/onlyxItachi/GAFIME/commit/06d82a4082e96f0821abe0e323ea91f4c7ee6744) |
| Source tree | `1fd446e65ff6b2303e3d8b10e34e76c75ce6d216` |
| Exact-tree remediation review | [`2bc0f144b425dc8db0a0ec464c84169fc189c72d`](https://github.com/onlyxItachi/GAFIME/pull/126#pullrequestreview-6076996802) |
| Official wheel run | [45857407051](https://github.com/onlyxItachi/GAFIME/actions/runs/45857407051) · push to `release/v1.0.0` |
| Official ZIP | [Artifact 15136914727](https://github.com/onlyxItachi/GAFIME/actions/runs/45857407051/artifacts/15136914727) · expires 2026-12-26T16:46:10Z |
| ZIP SHA-256 | `bcb6882f3e1a8c89feb7f956cee636d84faad7229d6cc87519e2e2072c0e8ada` |
| Release manifest SHA-256 | `cb884fff37d2aaa7f23001fe9cdaae4fb40de31fdfd4af607ebf6fb434b04f58` |
| `SHA256SUMS` SHA-256 | `6029a66850ff3c70f0c0193ed306678ede476c9fea7d0b904530b1e85a816eea` |
| Provenance JSON SHA-256 | `90f5d75527fbeb3c54b8704b1326f441b63d419731c991a83971f061e1373a9f` |

The bundle contains 40 wheels and three source distributions. [Checksums](SHA256SUMS) and [provenance](release-bundle-provenance.json) are retained verbatim. Keep the original ZIP with the session evidence so the exact bytes survive Actions artifact expiration.

```bash
gh api repos/onlyxItachi/GAFIME/actions/artifacts/15136914727/zip > official-release-bundle.zip
printf '%s  %s\n' bcb6882f3e1a8c89feb7f956cee636d84faad7229d6cc87519e2e2072c0e8ada official-release-bundle.zip | sha256sum --check --strict
```

## Verified qualification

- [V1 Contract Validation](https://github.com/onlyxItachi/GAFIME/actions/runs/45857407105) passed all five jobs, including the macOS Metal regression, installed Core correctness, native CUDA lifecycle checks and hosted [public ingestion evidence](https://github.com/onlyxItachi/GAFIME/actions/runs/45857407105/artifacts/15136549366).
- [Native Platform Validation](https://github.com/onlyxItachi/GAFIME/actions/runs/45857406975) passed all three jobs. [Build and Validate Wheels](https://github.com/onlyxItachi/GAFIME/actions/runs/45857407051) passed 50 jobs. Metal executed from the macOS wheel inputs to the freeze job. CUDA and ROCm hosted jobs verified distribution and runtime linkage.
- [Windows ARM64](https://github.com/onlyxItachi/GAFIME/actions/runs/45857407051/job/140039290097) and [Windows x64](https://github.com/onlyxItachi/GAFIME/actions/runs/45857407051/job/140039290145) passed receipt and filesystem validation. Both checkout newline configurations produced the same fixture receipt. The runner-temp volumes reported 8.3 name creation enabled on ARM64 and disabled on x64.
- Local verification passed all 44 checksums, exact run/repository/commit provenance, version identity, the full composition check and release policy for the 43 packages. PyPI collision preflight returned 38 new distributable filenames and zero collisions. The three stable versions are absent. The beta.1 resolver-safety check passed.
- The frozen Linux CPython 3.14 Core and CUDA wheels passed all three precision profiles on the RTX 4060 Laptop GPU and 18 Core/CUDA comparisons. Static inspection of the exact CUDA library also found `sm_120`.
- The frozen Linux CPython 3.14 Core and ROCm wheels passed all three profiles on a physical Radeon 890M with the pinned ROCm 7.2.3 userspace and 18 Core/ROCm comparisons.
- Frozen Core ingestion passed its reviewed latency and memory budget across 18 reference cells. The default miss/hit control also passed. This result covers the public ingestion workload on the qualification host.
- The exact frozen Core wheel passed 46 DL Method CPU consumer tests. One compatibility test failed and two compiled-path tests fell back. Consumer remediation and repeat validation are required before DL CUDA timing.
- GitHub Private Vulnerability Reporting is enabled. The source tree matches Astra’s fix approval and carries all 12 closed dispositions. The detailed evidence remains in the maintainer’s private security record.

## Remaining Stable gates

1. Run the [Colab qualification](colab.md) with the official wheel pair. Retain physical CUDA execution, Core/CUDA numerical parity, performance and resource evidence on the actual assigned GPU, including the planned RTX PRO 6000 Blackwell. Keep Ada evidence bound to its own device.
2. Complete the DL Method CUDA consumer and performance evidence on the approved v1 consumer revision. Its exact private source pin, local results and experiment commands are in the maintainer’s private DL Method receipt.
3. Complete the tracked Stable resident Core comparison on an eligible runner and exact-head PR. The repository currently reports zero self-hosted runners. Retain the frozen-wheel public end-to-end evidence separately from resident throughput.
4. Resolve the distinct exact-source Stable security qualification gate under [release policy](https://github.com/onlyxItachi/GAFIME/blob/06d82a4082e96f0821abe0e323ea91f4c7ee6744/docs/releases/STATUS.md). The earlier whole-source scan used a different tree, and the later exact-tree review verified the scoped fixes.
5. Confirm the three account-side PyPI Trusted Publisher bindings for `publish_release.yml` and environment `pypi`, then refresh registry and immutable-ref checks when publication is authorized. GitHub’s environment protects deployment through reviewer rules and the `v*` tag policy. The account-side settings require a signed-in PyPI session. Exact-ref lock, unchanged admission to `main`, tag creation and publication remain approval gates.

Machine-readable source, CI, artifact and local result bindings are in [verification.json](verification.json).
