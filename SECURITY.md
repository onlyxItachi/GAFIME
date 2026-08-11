# Security Policy

## Supported Versions

GAFIME provides security fixes for the active `1.0` release line and its current
beta or release candidate. Superseded prereleases and older release lines do
not receive backports unless the maintainer explicitly designates otherwise.

| Version | Supported |
| --- | --- |
| Latest `1.0.x` and the active `1.0` beta or RC | Yes |
| Superseded `1.0` prereleases | No |
| `0.x` and older | No |

Reports affecting the current default-branch code are welcome, but an
unreleased commit is not itself a supported distribution. This policy does not
promise a response or remediation deadline.

## Reporting a Vulnerability

Do not disclose suspected vulnerabilities in a public GitHub issue, discussion,
pull request, or commit.

Use [GitHub Private Vulnerability Reporting](https://github.com/onlyxItachi/GAFIME/security/advisories/new)
to submit a private report. Include, when available:

- the affected GAFIME version, distribution, backend, and precision profile;
- operating system, CPU or GPU, driver, and runtime versions;
- the entry point and attacker-controlled input;
- minimal reproduction steps and observed security impact;
- logs or a proof of concept that can be shared safely; and
- whether the issue or any details have already been disclosed elsewhere.

Do not include unrelated secrets or personal data. The maintainer will assess
the report and coordinate disclosure when practical, without a guaranteed
response timetable. The general contact address in the README is not designated
as the private vulnerability-reporting channel.

## System and Scope

GAFIME is an in-process native feature-interaction engine. Python exposes the
public API; PyO3 and Rust validate inputs, plan work, schedule Core execution,
select backends, and own public memory policy. CUDA, ROCm/HIP, and Metal payloads
execute through the published Rust-facing C ABI and their vendor runtimes.

This policy covers:

- the Python, PyO3, Rust, SIMD, C, C++, CUDA, HIP, Objective-C++, and Metal code;
- the frozen ABI 1.0 surface, generic typed ABI 1.1 route, and additive
  forward-compatibility behavior;
- package and payload discovery, resident caches, and native resource lifetime;
- wheels, source distributions, build workflows, frozen release bundles,
  checksums, provenance, and publication tooling; and
- the experimental local CUDA RT/OptiX code to the extent needed to preserve
  its isolation from standard distributions and execution paths.

GAFIME is not a network service, authentication system, multi-tenant execution
service, or sandbox for untrusted native code.

## Threat Model and Trust Boundaries

Applications may pass attacker-influenced numeric data, Arrow metadata,
feature names, shapes, strides, counts, enum values, configuration, backend and
device selections, and workload sizes through supported APIs. External ABI
consumers may also control ABI-visible structure fields and buffer metadata.

Important trust boundaries are:

- Python and Arrow inputs entering the PyO3/Rust boundary;
- safe Rust entering scoped `unsafe`, SIMD intrinsics, or native FFI;
- Rust crossing the C ABI into CUDA, ROCm/HIP, or Metal launchers;
- host memory crossing into device buffers, streams, queues, and encoders;
- process environment and filesystem discovery selecting native payloads;
- content entering resident caches or backend-local device state; and
- repository source becoming wheels, sdists, frozen bundles, and published
  artifacts.

The process owner, installed GAFIME packages, explicitly selected native
payloads, compiler toolchains, GPU drivers, and vendor runtimes are trusted
dependencies. A direct C ABI caller must provide live, suitably aligned,
caller-owned pointer spans; GAFIME must still validate all representable
metadata before using them.

## Security Invariants

GAFIME must preserve these properties:

- Safe Python and Rust APIs must validate shape, stride, count, enum, dtype,
  precision, structure-size, and capability metadata before native access.
- Element, byte-count, allocation, indexing, grid, and offset arithmetic must
  reject overflow and inconsistent lengths before pointer construction,
  allocation, copy, launch, or dereference.
- ABI 1.0 remains byte-compatible. ABI 1.1 validates stable prefixes, required
  flags, reserved fields, routes, typed buffer lengths, strides, alignment, and
  ownership. Unknown required semantics fail closed; compatible additive tails
  cannot overwrite known layouts.
- `unsafe` Rust remains narrowly scoped behind safe APIs with documented
  invariants. Ownership must not transfer implicitly across FFI.
- CUDA, ROCm/HIP, and Metal operations must bind the requested device and keep
  host and device pointers, streams or queues, encoders, and cached state
  device-consistent.
- Resources and caller-owned buffers must remain live until asynchronous native
  work is complete. Update, teardown, eviction, and error paths must not race
  in-flight execution or double-free resources.
- Explicit backend requests and unsupported precision or capability requests
  fail closed. They must not silently execute another backend or reinterpret a
  buffer under another numeric route.
- Dynamic payload discovery must reject ambiguous, mismatched, or unintended
  package identities. Environment-selected library paths are an explicit
  native-code trust decision, not a sandboxed plugin mechanism.
- Resident analysis, library, descriptor, graph, and device caches must include
  the required content, precision, backend, device, ABI, and payload identity
  and must not reuse state across incompatible lifetimes.
- Release artifacts must remain bound to their reviewed source and immutable
  frozen-bundle provenance. Publication must not rebuild, retag, rename, or
  replace package files after freezing.
- Standard wheels, sdists, workflow artifacts, and releases must exclude local
  RT/OptiX sources and outputs. The experiment must not activate through the
  standard payload path.
- Errors crossing FFI must be explicit and must not expose uninitialized
  outputs, stale handles, or partially accepted capability state.

## Reportable Findings and Severity Context

A reportable vulnerability requires a realistic path from an input or trust
boundary to security impact. Examples include:

- host or device out-of-bounds access, use-after-free, double-free, or other
  memory corruption;
- arbitrary native code execution or unintended native payload loading;
- integer overflow or malformed ABI metadata bypassing buffer validation;
- meaningful unauthorized file access or sensitive host/device data exposure;
- device confusion or asynchronous lifetime misuse exposing or corrupting
  another resource;
- package, frozen-bundle, checksum, provenance, or publication-integrity bypass;
- path, environment, cache-key, or artifact-identity confusion that crosses the
  documented trust boundary; or
- fail-open capability handling that reaches an unsafe or incompatible native
  operation.

Severity depends on realistic reachability and impact:

- **Critical:** practical arbitrary code execution or release-artifact
  substitution affecting downstream users.
- **High:** reachable memory corruption, meaningful arbitrary read/write,
  unintended payload execution, or sensitive cross-resource disclosure.
- **Medium:** a constrained but realistic integrity, lifetime, file-access, or
  disclosure failure with security impact.
- **Low:** defense-in-depth weakness without a demonstrated release-blocking
  attack path.

The presence of `unsafe`, FFI, or a crash alone does not establish severity.

## Non-Reportable Examples

The following are not security vulnerabilities unless they produce a concrete
security impact such as memory corruption, unauthorized access, disclosure,
integrity bypass, or code execution:

- unsupported hardware, operating systems, drivers, toolchains, or runtime
  combinations;
- an explicit fail-closed unsupported-backend or unsupported-profile error;
- invalid local development or compiler configuration;
- documented floating-point variance or an ordinary numerical-correctness bug;
- performance regressions or incomplete benchmark evidence;
- expected out-of-memory or resource-exhaustion behavior from a workload chosen
  by the same trusted process owner;
- an ordinary crash that cannot cross a trust boundary or violate memory safety;
- a direct ABI caller supplying a dangling or inaccessible pointer contrary to
  the pointer-ownership contract, unless a supported safe API can create that
  state or declared metadata should have rejected it;
- deliberate loading of an untrusted library through an explicitly configured
  `GAFIME_*_V1_LIB` path by the trusted process owner; or
- an upstream driver or dependency vulnerability with no reachable
  GAFIME-specific path or misuse.

## Accepted Limitations

GAFIME runs native code inside the caller's process and does not isolate a
malicious caller, extension module, native payload, driver, or compiler.
Applications that allow less-trusted users to influence process environment or
library search paths must sanitize those inputs before starting GAFIME.

Direct C ABI pointer validity cannot be proven from an address alone. The ABI
validates layouts and lengths, while direct callers remain responsible for the
actual allocation and lifetime. Safe Python and Rust APIs must not transfer
that responsibility to untrusted data.

Vendor GPU runtimes and drivers remain responsible for their own security.
GAFIME's checksums and frozen provenance protect its release workflow but are
not a general code-signing or sandbox guarantee.

Some immutable native libraries and backend state may remain resident for a
thread or process lifetime. This is acceptable only while identity,
device-isolation, synchronization, and bounded cleanup invariants hold.

The CUDA RT/OptiX path is an opt-in local experiment, not a supported
distribution surface. Unsupported RT hardware or an explicit local experiment
failure is not automatically a security issue; unintended inclusion or
activation in a standard artifact is reportable.
