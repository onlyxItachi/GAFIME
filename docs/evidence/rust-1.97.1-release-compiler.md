# Rust 1.97.1 Release Compiler Evidence

This record evaluates issue #65 as a compiler-only migration. Rust `1.89`
remains the minimum supported Rust version (MSRV), while official distributed
builds move to the exact Rust `1.97.1` stable compiler. It is a bounded
same-host observation, not a universal performance claim.

## Decision and provenance

- product source snapshot: `946ebf288d95a8e8a3b953f082dbbe26b127981a`
- current-main parent: `fabbcc09c2051920a8e9a66ee5393ec7e7039ffd`
- baseline: Rust `1.89.0` (`29483883e`, LLVM `20.1.7`)
- candidate: Rust `1.97.1` (`8bab26f4f`, LLVM `22.1.6`)
- host: AMD Ryzen AI 9 HX 370, 24 logical CPUs, Linux x86-64
- Python/wheel tag: CPython `3.14.3`, `cp314-cp314-linux_x86_64`
- measurement date: 2026-08-02

The Rust, Python, C/C++, Cargo manifest, and locked dependency source was
identical between the two builds. The adoption patch changes compiler pins,
validation policy, contract assertions, and documentation only. `Cargo.toml`
continues to declare `rust-version = "1.89"`, `clippy.toml` continues to declare
`msrv = "1.89"`, and `Cargo.lock` is unchanged.

The compiler choice is therefore:

```text
MSRV:             1.89
release compiler: exact 1.97.1
latest stable CI: exact 1.97.1
nightly release:  forbidden
```

No Edition migration, source modernization, dependency update, public API or C
ABI change, numerical change, backend change, packaging redesign, or new
nightly feature is part of this uplift.

## Build and link comparison

Each clean Release build used the locked workspace, fat LTO, one codegen unit,
and a separate Cargo target directory. The order was reversed for the second
pair to expose ordinary cache and run-order effects:

```bash
env CARGO_TARGET_DIR=/tmp/gafime-rust-189-release cargo +1.89.0 \
  build --workspace --release --locked
env CARGO_TARGET_DIR=/tmp/gafime-rust-197-release cargo +1.97.1 \
  build --workspace --release --locked
```

| Clean-build order | Rust 1.89.0 | Rust 1.97.1 | Candidate change |
|---|---:|---:|---:|
| baseline then candidate, wall time | 40.18 s | 36.11 s | -10.13% |
| candidate then baseline, wall time | 39.81 s | 36.97 s | -7.13% |
| baseline then candidate, peak RSS | 658,820 KiB | 618,224 KiB | -6.16% |
| candidate then baseline, peak RSS | 656,320 KiB | 620,256 KiB | -5.50% |

The mean of the two clean observations is 39.995 s versus 36.540 s
(-8.64%) and 657,570 KiB versus 619,240 KiB peak RSS (-5.83%). An unchanged
incremental Release rebuild completed in 0.05 s with 1.89.0 and 0.06 s with
1.97.1; this 10 ms difference is below a useful resolution for this host.

The standalone workspace extension grew, but the distributable wheel and its
wheel extension became smaller. `file` identifies both wheel extensions as not
stripped; the result is therefore a build/link/package-size observation, not a
stripping effect:

| Artifact | Rust 1.89.0 | Rust 1.97.1 | Candidate change |
|---|---:|---:|---:|
| workspace `libgafime_py.so` | 2,705,232 B | 2,857,328 B | +5.62% |
| wheel extension | 2,490,120 B | 2,442,736 B | -1.90% |
| complete wheel, 34 files | 1,066,409 B | 1,018,005 B | -4.54% |

Artifact bindings:

```text
Rust 1.89.0 unstripped  6a9bf4bc0d92bf9b858f9bfa569d5a0710fc20cc0eaffb88734e5e010ef9e83c
Rust 1.97.1 unstripped  d049ba52fd9d2ea10a972e079b771750b644dae556020385eff146d78448a223
Rust 1.89.0 wheel       c79b55d7986b8795bbd23a02242e7bce901d9057744ae51eb8a4c86f963b7819
Rust 1.97.1 wheel       651c44fcb735024a9abb57a34280ff2d862b49062262fa94a901e9970cc3be86
Rust 1.89.0 wheel .so   ae9240b27bfa20bfd384c9f1d518dfa2001cd30fa7542c3e601e0003ea5c7cb9
Rust 1.97.1 wheel .so   24a6ecd3ff8734324d0530f46bedb6ce0be55782db47a8114aec1d0f608a579a
Rust 1.97.1 GNU-ld wheel 02803165d1cad5bd7653b0f6c484460566b0c38609011456a9ca9774fa7ae601
```

### LLD control

Rust 1.97.1 selected bundled LLD 22.1.6 by default on this Linux target. The
same source was also built with `RUSTFLAGS="-C linker-features=-lld"`, which
selected GNU ld 2.46:

```bash
env RUSTFLAGS="-C linker-features=-lld" cargo +1.97.1 \
  build --workspace --release --locked
```

| Rust 1.97.1 link surface | Default LLD | GNU ld control | LLD change |
|---|---:|---:|---:|
| clean wall time | 36.11-36.97 s | 37.20 s | 0.6-2.9% faster |
| peak RSS | 618,224-620,256 KiB | 622,116 KiB | 0.3-0.6% lower |
| workspace extension | 2,857,328 B | 2,652,432 B | +7.73% |
| wheel extension | 2,442,736 B | 2,451,280 B | -0.35% |
| complete wheel | 1,018,005 B | 1,019,419 B | -0.14% |

Despite the larger standalone workspace output, the actual LLD wheel is
slightly smaller without a stripping step. The GNU-ld control wheel
independently passed `installed_wheel_smoke.py`, the installed Python suite
(312 passed, 13 hardware/platform skips), bit parity for all 6 NumPy-reference
candidates, and compiled/eager parity across 120 metrics with maximum absolute
delta `0.00e+00`. Exported symbols and dynamic dependencies are also unchanged,
so no LLD-specific correctness, ABI, runtime, or distribution anomaly was
observed.

## Correctness and numerical behavior

Both toolchains passed the same required local gates:

```text
cargo +VERSION fmt --all -- --check
cargo +VERSION check --workspace --all-targets --locked
cargo +VERSION clippy --workspace --all-targets --locked -- -D warnings
cargo +VERSION test --workspace --locked -- --test-threads=1
```

Each Cargo test run passed 231 tests with one intentional ignored manual
benchmark. The suite covers non-finite input, overflow diagnostics,
zero-variance behavior, histogram boundaries, SIMD lane tails, stable tie
ordering, decision paths, and C ABI layout checks. Strict Clippy produced zero
findings under both compilers while reviewing against the repository's 1.89
MSRV.

Each freshly built wheel was installed into a separate clean CPython 3.14
environment. Both installations produced the same result:

```text
installed_wheel_smoke.py: PASS
tests/python: 312 passed, 13 skipped
```

The 13 skips are the same hardware/platform-conditional tests in both
environments. They are covered by the hosted CUDA, ROCm, Metal, and native
platform jobs rather than treated as local passes.

The installed-wheel release measurements passed under both compilers:

- NumPy reference: bit parity for all 6 top-level candidates.
- Continuous, time-series, decision-path, and dataload references: pass with
  identical generated order and labels.
- Compiled plan: 36 interactions under both compilers.
- Compiled/resident/eager comparison: 120 metrics with maximum absolute delta
  `0.00e+00` under both compilers.

The pinned single-CPU measurements used four alternating-order processes per
compiler. Every process used the public resident Core API; each reported value
was identical between compilers.

| Workload | Rust 1.89.0 median | Rust 1.97.1 median | Candidate change | Exact value |
|---|---:|---:|---:|---:|
| 1,048,576-row Pearson + R2 | 0.408346 ms | 0.407102 ms | -0.30% | Pearson `0.9677826762199402`; R2 `0.936603307723999` |
| 1,048,576-row 96-bin MI | 4.146589 ms | 3.004815 ms | -27.54% | MI `0.3164912164211273` |
| 128-row Python-boundary proxy | 0.004825 ms | 0.004845 ms | +0.41% | Pearson `0.9999977946281433` |

These local numbers cover the detected x86 SIMD paths on one dynamic-clock
host. The 0.41% boundary difference and 0.30% covariance difference are treated
as noise, not product claims. The repeatable MI improvement is retained as an
observation, not a cross-platform guarantee.

## ABI, symbols, and profiler readability

`nm -D --defined-only` reports exactly one dynamically exported symbol from
each wheel: `PyInit_gafime_py`. The two wheels have the same DT_NEEDED set
(`libgcc_s.so.1`, `libm.so.6`, `libc.so.6`, and
`ld-linux-x86-64.so.2`). Workspace tests under both compilers pass
`gpu_abi_header_and_rust_layouts_stay_in_lockstep`, so the pinned GPU C ABI
header and Rust layouts remain aligned.

The non-dynamic local symbol tables from both compilers use Rust v0 mangling,
and `nm -C` demangles project frames such as `gafime_cpu::diagnostics` and
`gafime_orchestrator::plan` under both versions. The wheel extensions are not
stripped and retain demangleable local/internal symbols, while the dynamic
export table exposes only the Python initializer. Profiler and exported-symbol
readability therefore does not regress.

## Distribution and scope boundary

An earlier hosted compiler-pin experiment at source `06eb184` ran exact Rust
1.97.1 through the full 48-job Build and Validate Wheels workflow. That includes
CPython 3.10-3.14 Core wheels on Linux x86-64, Windows AMD64, macOS ARM64, Linux
ARM64, and Windows ARM64, plus the configured CUDA/ROCm pairing validators,
macOS Metal execution, and frozen-bundle gate. The accompanying 5-job v1
contract and 3-job native-platform workflows were green regression context but
still selected Rust 1.89.0 at that old head; they are not candidate-compiler
evidence. The adoption PR's final-head CI is the first hosted execution of the
newly repinned v1 and native-platform lanes and must pass before merge. The old
48-job experiment is supporting evidence, not a substitute for that merge gate.

The release manifest, CPython-minor-specific tags, platform set, payload
ownership, system CUDA/ROCm runtime policy, Metal embedding policy, RT/OptiX
exclusions, frozen-bundle contract, and publication workflow are unchanged.
No `#![feature]`, portable SIMD, Polonius dependency, nightly toolchain, or
nightly-built release artifact was introduced.

## Conclusion

The same source remains valid on the declared Rust 1.89 MSRV and passes the
full local candidate suite on Rust 1.97.1. The candidate has no observed
correctness, ordering, ABI, Clippy, wheel-size, linker, or Python-boundary
regression. Exact Rust 1.97.1 is accepted as the release compiler while MSRV
remains 1.89; later source modernization or an MSRV increase requires a
separate reviewed change.
