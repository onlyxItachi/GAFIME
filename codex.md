# GAFIME Codex Context

Read this first after compaction. This file is the durable Codex handoff log
when the owner asks for one. `AGENT_COMMS.md` remains the ignored scratch file
for live Claude/Codex coordination.

## Operating Rules

- Work from the local GAFIME repo, not the pip-published package.
- Do not web search for repo-local build/workflow/package tasks unless the user
  explicitly asks.
- Do not tag, publish GitHub releases, publish PyPI, or re-enable release jobs
  without explicit owner approval.
- Heavy GitHub workflows should be inert on ordinary commits. They should run
  only by owner dispatch or release tag.
- Keep future roadmap private unless the user explicitly asks to publish it.

## Release Rules

Owner-approved rules after the v0.4.7 distribution hesitation:

- Canonical source lives in `gafime/`, `gafime_core/`, `src/`, `tests/`,
  `docs/`, and `.github/`. Do not commit temporary package roots such as
  `packaging/gafime-rocm/`, top-level `gafime_cuda/`, or top-level
  `gafime_rocm/` unless the owner explicitly makes them first-class packages.
- Backend/kernel implementation, distribution packaging, documentation, release
  notes, and publication are separate phases. Do not jump to release packaging
  while backend behavior is still being debated.
- Supported release scenarios only. If a scenario is outside the release
  contract, do not design, document, or test a special policy for it.
- For v0.4.7, supported install modes are base `gafime`, CUDA payload mode, and
  ROCm payload mode. Ignore combined CUDA+ROCm payload environments.
- Normal commits must not trigger expensive workflows. Wheel builds, Metal
  stress, platform validation, release jobs, and PyPI publishing require owner
  dispatch, release tags, and/or explicit owner approval.
- Docs follow proven behavior. Do not claim backend maturity before tests or
  benchmarks prove it.
- Stop if a fix requires weird committed package scaffolds, hardcoded paths,
  fake public packages, or public docs explaining an internal workaround.
- Ask the owner before version bumps, tags, GitHub releases, PyPI publishing,
  public install command changes, backend deletion/deprecation, or major package
  layout changes.

## Release Timeline

### v0.4.0 - Discrete Function Engine

- Added the discrete function representation family inside the existing
  `GafimeEngine` / `EngineConfig` / `ComputeBudget` style.
- Added soft/vectorized GPU-safe discrete candidates and CPU-compatible hard
  mode guard behavior.
- Added discrete candidate planning, scoring integration, docs, tests, and
  release materials.
- Kept user metric preference for report metrics; no separate public discrete
  metric was finalized for this release.
- Tagged commit: `a9a43bc docs: prepare v0.4.0 release materials`

### v0.4.1 - Math Correction Release

- Hardened discrete ranking math after fixed-bin MI issues were found.
- Added adaptive target-bin template logic and better GPU selector consistency.
- Corrected small-sample support behavior so true low-support regions are not
  suppressed too aggressively.
- Verified ranking behavior against stronger sklearn/reference comparisons.
- Tagged commit: `2a4fc91 Release GAFIME v0.4.1 math corrections`

### v0.4.5 - Native Spine / CPU-CUDA Refactor

- Removed production NumPy backend/fallback; C++ Core became the CPU execution
  spine.
- Added native fp32 memory ownership, `real_t` policy, Core buffer scoring, and
  Rust cache-local batching/orchestration.
- Reworked CUDA broad continuous scoring around global matrix batch ABI,
  homogeneous arity templates up to 5, and no global `--use_fast_math`.
- Added explicit time-series candidate family and hardened GPU hard-discrete
  guard behavior.
- Added x86 SIMD dispatch paths and ARM CPU distribution support in the base
  package.
- Tagged commit: `92dc41c Clarify v0.4.5 CUDA benchmark context`

### v0.4.6 - Metal / Apple Silicon Release

- Updated Metal backend toward full native capability on Apple Silicon.
- Hardened ARM Core backend behavior and platform-aware backend contracts.
- Added Linux ARM and Windows ARM CPU wheel/distribution coverage.
- Validated through GitHub macOS runners because the owner does not have local
  Apple Silicon hardware.
- First release with the expanded 5-platform x 5-Python wheel matrix:
  macOS arm64, Linux x86_64, Windows x86_64, Linux arm64, Windows arm64 for
  CPython 3.10-3.14.
- JSON reporting is deprecated; C++ pointer flow holds the native report/data
  path.
- Tagged commit: `c486681 Release v0.4.6 Metal validation and native reports`

## v0.4.7 ROCm / Distribution State

- ROCm kernels have been written and should be represented as HIP source:
  `src/rocm/kernels.hip`.
- ROCm platform handling must not classify GPUs by product name or by parsing
  ROCm offload-target suffixes. ROCm target strings are exact build/diagnostic
  metadata only. Runtime policy should use HIP capability flags (`integrated`,
  managed memory, host registration, memory pools, large BAR,
  CU/L2/wavefront data) and should let ROCm own target support/detection.
- ROCm shared-memory integrated GPUs use an explicit UMA launch mode for broad
  matrix scans and local bucket/time-series scans. Python selects
  `GAFIME_ROCM_MEMORY_UMA_HOST_MAPPED` from HIP capability flags; native HIP
  uses page-aware `hipHostRegister(..., hipHostRegisterMapped)` and falls back
  per buffer if a registration is rejected. Local iGPU smoke verified both
  continuous arity 2-5 and bucket/time-series paths reporting
  `uma_host_mapped`.
- CUDA and ROCm payloads must stay separated from the base/core `gafime`
  distribution:
  - base `gafime`: Python API, Core/CPU, Metal, resolver/orchestration;
    no CUDA/ROCm native payload binaries.
  - `gafime-cuda`: NVIDIA CUDA native payload package.
  - `gafime-rocm`: AMD ROCm/HIP native payload package.
  - convenience installs should resolve through extras such as
    `gafime[cuda]` and `gafime[rocm]` once metadata is correct.
- PyPI treats `gafime`, `gafime-cuda`, and `gafime-rocm` as separate projects.
  Publishing must use three independent lanes from `.github/workflows/build_wheels.yml`
  with the same GitHub environment `pypi`:
  - base/core lane selects only `gafime-*` distributions.
  - CUDA lane selects only `gafime_cuda-*` distributions.
  - ROCm lane selects only `gafime_rocm-*` distributions.
  Each PyPI project needs its own Trusted Publisher entry. A failure in one
  payload project must not block the others.
- v0.4.7 publishing fix completed on 2026-06-15:
  - Commit `90b85b3 Fix PyPI payload publishing lanes` is pushed to `main`.
  - Workflow run `27572832122` completed successfully.
  - `gafime==0.4.7`, `gafime-cuda==0.4.7`, and `gafime-rocm==0.4.7` are visible
    on PyPI.
  - Core `gafime` wheels were republished with wheel build tag `1`
    (`gafime-0.4.7-1-...whl`) because PyPI permanently blocks reusing deleted
    filenames.
  - Core sdist publishing was disabled for the retry because
    `gafime-0.4.7.tar.gz` had already been used/deleted.
  - ROCm Linux payload wheels now use a PyPI-compatible manylinux platform tag
    instead of the invalid `linux_x86_64` tag.
  - CUDA and ROCm payload wheels are thin binary packages and declare
    `Requires-Dist: gafime==0.4.7`.
  - Verified from PyPI on local Python 3.14.3: core wheel installs and imports,
    Core engine smoke runs, CUDA and ROCm payload wheels download and pass ZIP
    integrity checks.
- GitHub Actions policy for v0.4.7:
  - ordinary commits to `main` should not automatically run heavy workflows.
  - heavy tests/benchmarks/wheels should be disabled/inert by default and only
    run when the owner explicitly dispatches them or when a release tag is
    created.
  - do not delete test workflows; make them non-running by default.
  - release wheel workflow should stay owner-controlled and not surprise-run
    heavy jobs on normal commits.

## v0.5.0 Prework - v0.4.7 Distribution and Backend Findings

Recorded on 2026-06-19 after testing the published v0.4.7 PyPI wheels in clean
temporary virtualenvs under `/tmp/gafime-v047-wheelcheck.mxIq7O`.

- GitHub/release reference state:
  - Remote repo is `git@github.com:onlyxItachi/GAFIME.git`.
  - Remote `main` is `90b85b384584cebe3e9c88256f9dbab92666437d`
    (`Fix PyPI payload publishing lanes` locally).
  - Annotated tag `v0.4.7` resolves to commit
    `428626822413604d59f4b2e3e1595a58cc4e456a`.
- PyPI split-package install/download checks passed:
  - `pip install --only-binary=:all: gafime==0.4.7` installed the republished
    core wheel `gafime-0.4.7-1-cp314-cp314-manylinux_2_28_x86_64.whl`.
  - `pip install --only-binary=:all: "gafime[cuda]==0.4.7"` installed
    `gafime==0.4.7` plus `gafime-cuda==0.4.7`.
  - `pip install --only-binary=:all: "gafime[rocm]==0.4.7"` installed
    `gafime==0.4.7` plus `gafime-rocm==0.4.7`.
  - `pip index versions` shows `gafime-cuda` and `gafime-rocm` at `0.4.7`
    only, while core `gafime` latest is also `0.4.7`.
- Wheel metadata/content shape is correct for the v0.4.7 package split:
  - Core `gafime` declares extras that depend on same-version
    `gafime-cuda==0.4.7` and `gafime-rocm==0.4.7`.
  - Core `gafime` carries `_native`, `gafime_core`, and `gafime_cpu` native
    modules; it does not carry CUDA or ROCm payload libraries. The only
    CUDA/ROCm hits in the core wheel are Python backend wrapper modules.
  - `gafime-cuda` declares `Requires-Dist: gafime==0.4.7` and carries only
    `gafime_cuda/_native...so` plus `gafime_cuda/libgafime_cuda.so`.
  - `gafime-rocm` declares `Requires-Dist: gafime==0.4.7` and carries only
    `gafime_rocm/_native...so` plus `gafime_rocm/libgafime_rocm.so`.
- Owner clarification on package/repo shape:
  - `gafime`, `gafime-cuda`, and `gafime-rocm` are three distribution targets
    from the same repository, not separate repos and not separate product
    lines.
  - CUDA/ROCm payload packages can be expanded more aggressively because they
    do not add runtime overhead or binary size to the base CPU/Core execution
    package unless the shared code/contracts change.
  - Changes to shared CPU Vector ISA + GPU contracts, descriptor formats,
    native result tables, C++ headers, or orchestration APIs affect the core
    package too because those are common engine interfaces.
  - The realistic framing is not "payload changes cannot poison core"; it is
    "payload-specific binaries do not burden core installs, but shared native
    contracts must remain disciplined."
- Runtime smoke checks from clean installs:
  - Core-only env: `python -m gafime --check` reports Auto -> C++ Core.
  - CUDA env: Auto -> `cuda-native`, NVIDIA GeForce RTX 4060 Laptop GPU
    `sm_89`; tiny end-to-end engine smoke passed.
  - ROCm env: Auto -> `rocm-native`, AMD Radeon Graphics HIP 11.5 target
    `gfx1150`, shared system memory/UMA host-mapped path; tiny end-to-end
    engine smoke passed.
- Continuous backend timing notes, using 2026-06-19 local hardware:
  - For direct pair scoring at `n=32768`, `p=64`, `2016` pairs, report metrics
    `("pearson", "r2")`: Core was roughly `0.04-0.07s`, CUDA was roughly
    `0.016-0.020s`, and ROCm was roughly `0.036s` on the working serial path.
  - Engine-level continuous analysis for the same broad pair workload with no
    validation/permutation work: Core was about `0.080s`, CUDA about `0.039s`,
    and ROCm serial about `0.072s`.
  - Python list -> `NativeMatrix` coercion at `n=32768`, `p=64` cost about
    `0.013-0.015s`, large enough to matter for short GPU scans.
- Full report metrics expose the main Python/backend squeeze point. Here,
  "fallback" means report-metric completion after the GPU selector/stat path,
  not backend fallback and not the adaptive histogram selector:
  - Continuous full report metrics
    `("pearson", "spearman", "mutual_info", "r2")` at `n=32768`, `p=64`,
    `2016` pairs took about `0.513s` on Core, `0.538s` on CUDA, and `0.546s`
    on ROCm.
  - CUDA/ROCm currently compute fast sufficient-stat report metrics on GPU,
    then use Core completion for missing continuous report metrics such as
    Spearman and continuous mutual information. This means GPU speedups largely
    disappear when public report metrics request full continuous metrics.
  - Discrete adaptive selector MI is GPU-side: CUDA/ROCm run the soft-mask
    histogram/bin selector kernels on device. The slower path is only when the
    public report asks for metrics outside the GPU stat/report set.
  - Discrete/time-series report completion is more expensive because those
    missing report metrics complete through Python candidate evaluation instead
    of a C++ Core batch-completion API.
  - Discrete examples: CUDA scored `192` soft-threshold candidates in about
    `0.0006s` for `("pearson", "r2")` but about `0.845s` with full report
    metrics; ROCm scored `512` candidates in about `0.0023s` for stats metrics
    but about `5.0s` with full report metrics.
- Backend language capability read:
  - C++ Core is the strongest CPU metric engine today: it owns fp32 buffers,
    releases the GIL, uses OpenMP/SIMD dispatch, and already completes full
    continuous metric reports.
  - CUDA and ROCm are strongest for broad sufficient-stat scans and split-aware
    discrete selector work. They need either native/GPU Spearman/MI paths or an
    explicit selector/report separation so full report requests do not hide the
    GPU gains.
  - Rust is currently valuable for scheduling, ordering, data quality helpers,
    and cache-aware orchestration. It is not yet the main float-kernel backend.
    v0.5 should use Rust to reduce Python object churn in candidate planning,
    descriptor packing, and result-table construction before moving numeric
    kernels there.
  - Python is the control-plane/API layer, but it should not be in hot loops for
    report completion, candidate materialization, or per-candidate metric
    dictionaries on large scans.
- v0.5.0 performance priorities from these checks:
  - Keep selector metrics and report metrics separate. The hot path should avoid
    full report metrics unless the user explicitly needs them.
  - Add C++ Core batch report completion for discrete and time-series candidate
    families before adding new Python-visible scoring families.
  - Add a backend-resident analysis/session object so unary, pair/higher-order,
    discrete, time-series, stability, and permutation phases can reuse the same
    native/device matrix and planning state instead of re-uploading/rebuilding.
  - Replace large Python candidate/result object loops with packed native
    descriptors and native report/result tables.
  - Add native/GPU Spearman and adaptive MI only if v0.5 selector/report
    semantics need those metrics in the hot path; otherwise route them to Core
    intentionally as a diagnostic/report refinement mode.
  - Add release CI smoke that installs `gafime[cuda]` and `gafime[rocm]` in
    clean envs, runs `python -m gafime --check`, and executes a tiny E2E
    analysis for each payload package.
  - Headline target is end-to-end orchestration time, not only kernel speed:
    Python should become a thin API layer while native code owns planning,
    feature pointers, candidate descriptors, backend sessions, report
    completion, top-k, and result-table construction.
  - Native/C++ exports should be designed so external frameworks can consume
    GAFIME feature pointers/results as first-class citizens. XGBoost-style
    integrations are an explicit target because avoiding Python-side feature
    materialization/copying can greatly reduce orchestration overhead.
  - `gafime-dl-method` / Torch C++ extension style is useful architectural
    precedent for framework-level integration, but it is not the current
    objective unless the owner explicitly resumes that path.
  - Owner has a v0.5 concept named `gafime.compile` with three compile flags
    still to be finalized. Treat it as a native execution-plan compiler:
    scenario templates, compact orchestration artifacts, backend session
    ownership, optional CUDA/HIP/Metal graph capture, and framework-exportable
    native handles. Do not reduce this idea to source-code JIT only.
  - Current v0.4.5+ code already fixed the old CUDA giant candidate-matrix
    problem for broad continuous scans: the input matrix stays resident inside
    `gafime_cuda_matrix_compute_batch` / ROCm / Metal matrix handles and
    homogeneous arity batches stream global feature-index descriptors. Similar
    but different orchestration issues remain: Python materializes combination
    lists and result dicts, Rust scheduler clones/returns descriptor batches to
    Python, GPU matrix handles are allocated/uploaded/freed per scoring phase,
    batch size is still capped around 1024, and validation/permutation/report
    completion can re-enter the same orchestration path repeatedly.
  - One transient parallel ROCm stress run returned native error `-4`
    (`GAFIME_ERROR_KERNEL_FAILED`), but serial reproductions across `p=16..64`
    passed. If this recurs, inspect HIP kernel error details and concurrent
    process/device pressure before treating it as a deterministic backend bug.

## v0.5.0 Architecture Memory

Do not implement this unless the user explicitly resumes v0.5.0 planning or
implementation.

- Strategic theme: template-centered, scenario-aware, mathematically safer,
  performance-stable feature-combination search.
- Separate report metrics from internal candidate selector metrics:
  `metric_names` controls report output, while future
  `candidate_selector="recommended"` may use family-aware ranking.
- Family-aware selectors should treat dense/general MI and discrete soft-binary
  MI separately. Discrete selector direction: soft-binary MI, variance /
  impurity reduction, residual gain, and possibly CV gain.
- Adaptive MI/binning direction: GPU-friendly target-bin templates around
  `4..96`; exact or near-exact information-theoretic scoring belongs in CPU /
  diagnostic / refinement mode, not the hot GPU selector.
- Dataset-aware template planning should use feature-count/combination-count
  analysis, not only fixed arity caps. Narrow/tiny datasets may allow much
  deeper search under budget.
- Power-user mode idea: `max_candidate_budget=-1` means uncapped by safety
  policy, not infinite; still bounded by technical/index limits and explicit
  user settings.
- v0.5 needs careful Rust/C++/CUDA/Metal/ROCm orchestration inspection before
  implementation.

## v0.5.0 Compile Artifact Implementation Log

Recorded on 2026-06-22 from branch `work/v0.5-compile-artifact`.

### Branch / Communication State

- Current branch is `work/v0.5-compile-artifact`.
- `AGENT_COMMS.md` is a local ignored coordination scratch file between Claude
  and Codex. Do not rely on it for durable project history.
- `codex.md` is now the durable local handoff log for this workstream when the
  owner asks to preserve the state.
- No tags, pushes, releases, GitHub releases, or PyPI work have been approved
  or done for v0.5 from this branch.
- `build/libgafime_rocm.so` remains stale/deferred; this is a payload rebuild
  item, not a blocker for the v0.5 compile perf branch.

### Checkpoint Range

The v0.5 compile branch starts at:

- `05eb14b checkpoint: compile api skeleton`

The latest committed checkpoint at the time of this log is:

- `52f8675 checkpoint: native ridge baseline guards`

Important checkpoint sequence:

- `05eb14b checkpoint: compile api skeleton`
- `1e079ad checkpoint: native scenario plan`
- `24eebfe checkpoint: compiled continuous session`
- `54503e8 checkpoint: compiled discrete session`
- `463175b checkpoint: compiled time-series session`
- `17bde72 checkpoint: native report tables`
- `754145d checkpoint: export handles`
- `cbf662d checkpoint: cuda graph capture`
- `4cccec2 checkpoint: hip graph capture`
- `0ddf08c checkpoint: metal graph evaluation`
- `9d3f241 checkpoint: rust compile plan`
- `b36eca2 checkpoint: rust continuous combo expansion`
- `4ca1f62 checkpoint: compiled continuous combo cache`
- `17a724c checkpoint: rust time-series candidate expansion`
- `c21e23d checkpoint: rust discrete candidate expansion`
- `bdd55ff checkpoint: compiled planner cache`
- `c0241a7 checkpoint: resident session input guard`
- `398ec4c checkpoint: native report ranking`
- `2bd0a82 checkpoint: compiled all-family parity`
- `3b993dc checkpoint: cuda graph native capture and replay`
- `1408c82 checkpoint: hip graph native capture and replay`
- `879bacb checkpoint: metal graph feasibility and graph track docs`
- `a87d480 chore: ignore AGENT_COMMS.md local agent comms file`
- `ce1b379 checkpoint: backend resident-target-update predicate`
- `9e397e4 checkpoint: resident target swap`
- `e7ee543 checkpoint: resident permutation contract`
- `b3512f4 checkpoint: backend TS-on-resident-matrix (Option B)`
- `ae13df2 checkpoint: resident time-series target swap`
- `dfd9d62 checkpoint: native split-aware ridge baseline (87x)`
- `52f8675 checkpoint: native ridge baseline guards`

### What Landed

- Public compile API:
  - `gafime.compile(...)`
  - `GafimeEngine.compile(...)`
  - `CompileFlags`
  - `CompiledGafime`
  - temporary legacy fallback through `GAFIME_USE_LEGACY_ENGINE=1`
- Rust-first planning:
  - Rust compile sources live under `gafime/compile/RustCompileSrc/`.
  - PyO3 crate exposes planning classes through
    `src/cpu/gafime_cpu/src/lib.rs`.
  - Python in `gafime/compile/scenario.py` is a thin adapter through
    `gafime.subfunctions`.
  - Continuous, discrete, and time-series descriptor/candidate expansion have
    Rust-backed paths and Python fallbacks.
- Plan/session track:
  - Compiled backend sessions are now the engine execution surface.
  - Continuous resident matrix sessions reuse backend matrix handles.
  - Planner/candidate descriptor caches reduce repeated Python materialization.
  - Changed-X stability still falls back because bootstrap samples create new
    matrices.
  - Same-X changed-y permutation can update resident targets in place when the
    backend predicate allows it.
- Native reports:
  - Native report tables and native-backed ranking/top-k views landed.
  - `to_dict()` remains materializing/deprecated.
- Export track:
  - `CompileFlags(export=True)` exposes internal export handles tied to the
    compiled artifact lifetime.
- Graph track:
  - CUDA and HIP graph capture/replay surfaces landed.
  - Metal graph remains documented unsupported/neutral for v0.5.
  - Graph correctness is present, but measured speedups came from residency,
    not graph capture.
- Backend/perf track:
  - CUDA/ROCm target-update predicate and `update_resident_target(...)` landed.
  - Continuous permutations use resident target swaps.
  - Time-series uses Option B: resident continuous matrix handle plus
    `score_time_series_candidates_resident(...)`.
  - Native split-aware ridge baseline landed in Core as
    `gafime_core.ridge_baseline_prediction(...)`.

### Measured Perf Scorecard

Measured by Claude on RTX 4060 with parity/count contracts, no wall-clock CI
thresholds:

- Continuous resident permutation:
  - `1.6x` end-to-end
  - `1.76x` permutation phase
  - full matrix uploads reduced from `30 -> 3`
  - target swaps: `25`
- Time-series resident permutation:
  - `2.3x` end-to-end
  - `3.6x` time-series permutation phase
  - bucket calls reduced from `29 -> 3`
  - resident TS scorings: `1 actual + 25 permutations`
- Native discrete split-aware ridge baseline:
  - baseline prediction `2012 ms -> 23 ms`, about `87x`
  - discrete end-to-end `3.2x`
  - parity max absolute/relative about `6.9e-8`
- CUDA/HIP graph capture:
  - correct, but experimental/neutral for measured v0.5 performance
  - residency is the proven speedup path

### Tests / Verification State

Codex-side verification after the last committed checkpoint:

- `python3 -m py_compile tests/test_v045_native_spine.py gafime/engine.py`
- `python3 -m pytest tests/test_v045_native_spine.py`
  - `12 passed`
- `python3 -m pytest tests/test_compile_api.py tests/test_compile_scenario.py tests/test_compile_graph_flags.py tests/test_v045_native_spine.py tests/test_compile_graph_cuda.py`
  - `41 passed, 3 skipped` on the Codex host because CUDA graph payload tests
    skip here

Claude-side verification with local CUDA payload:

- Same targeted compile/native graph suite: `44 passed`
- CUDA payload present there, so the three CUDA graph tests ran instead of
  skipping.

### Codex / Claude Work Split

Codex-owned work:

- Compile API/artifact/session architecture.
- Rust-first planner placement and Python adapter behavior.
- Native report table/ranking tests.
- Session target-swap routing and count-based tests.
- Time-series resident-session routing after Claude landed backend surface.
- Native ridge guard test.
- Coordination rules and final checkpoint summaries.

Claude-owned work:

- C++ Core native ridge baseline.
- CUDA/HIP graph native capture/replay.
- CUDA/HIP resident target update predicates and backend hooks.
- CUDA/HIP time-series resident matrix ABI and backend wiring.
- RTX 4060 measurement passes and backend parity checks.

### Current Open Handoff: Residency v2

Claude measured the remaining default-metric permutation tax after v0.5 compile
perf was banked:

- Scenario: continuous-only, `permutation_tests=25`, `num_repeats=3`,
  `n=32768`, `f=24`, `300` combos, CUDA RTX 4060.
- Compiled default metrics
  `("pearson", "spearman", "mutual_info", "r2")`:
  - total `4141 ms`
  - permutation `3502 ms`
  - native Core spearman/MI completion `3881 ms` across `30` score calls
- Compiled pearson/r2 control:
  - total `167 ms`
  - permutation `123 ms`
- Legacy default metrics:
  - total `4269 ms`
  - permutation `3663 ms`
  - completion `3934 ms`

Conclusion:

- Residency v1 removed uploads/allocation, but default metrics still recompute
  X-invariant work per permutation.
- The repeated work is native Core, not Python interpreter loops:
  interaction vectors, `sum_x/sum_x2`, X ranks for Spearman, and X bins for MI.
- The measured cache memory estimate at this size is about:
  - vectors: `39 MB`
  - ranks: `39 MB`
  - bins: `39 MB`
  - total: about `118 MB`
- At default combination caps this can become GB-scale, so an explicit budget
  gate is mandatory.

Codex approved opening **residency v2** with a narrow first checkpoint:

- First target only: continuous permutation report-metric completion for
  default metrics.
- Do not mix in time-series, discrete, vector ISA cleanup, native permute, or
  scheduler-cache cleanup.
- Claude will start on the C++ Core side.
- Codex will wire the Python/session side after the native API lands.

Requested native Core API shape:

```python
cache = core.build_continuous_metric_cache(
    X.buffer,
    combos,
    metric_names,
    mi_bins,
    max_bytes,
)

scores = core.score_continuous_metric_cache(
    cache,
    y.buffer,
    metric_names,
    mi_bins,
)
```

Cache contract:

- Return a cache object/capsule or `None`/clear unsupported result when the
  memory budget is exceeded.
- Cache owns only X-invariant state:
  - engineered vectors when needed
  - `sum_x`, `sum_x2`
  - X ranks / centered rank stats for Spearman
  - X bins / bin counts for MI
  - combo order and metric order
  - memory estimate / actual bytes
- Re-score recomputes only y-dependent state:
  - y sums / variance
  - y ranks / centered y ranks
  - y bins
  - dot/reduction or histogram join against cached candidate state
- Must match current `CoreBackend().score_combos(...)` within tolerance for
  actual and permuted y values.
- Do not cache anything tied to a permuted y.
- Cache lifetime belongs to the compiled artifact/session, not global state.
- Session key should include:
  `(X buffer identity, combo tuple, metric_names, mi_bins, max_bytes)`.
- If cache is unavailable or budget-exceeded, session must fall back to
  residency v1 behavior.

Expected tests after native API lands:

- Native parity: cached score vs current Core score for actual y and multiple
  permuted y values.
- Budget fallback: tiny `max_bytes` disables cache and falls back.
- Session count contract: cache builds once and cached score is called for
  each same-X permutation.
- Changed-X stability remains full/v1 fallback.
- No wall-clock CI assertions.

### Residency v2 Session Wiring

Implemented after Claude landed `926fe2d checkpoint: residency v2 core -
continuous metric cache`.

- `ResidentContinuousMatrixSession` now owns a session-lifetime continuous metric
  cache table keyed by:
  `(X buffer identity, combo tuple, metric_names, mi_bins, max_bytes)`.
- The cache is used only after resident-input preparation succeeds, so changed-X
  stability still falls back to the v1/full backend path.
- Resident pearson/r2 stats remain on the backend path; missing continuous
  report metrics such as Spearman and MI are completed through
  `gafime_core.score_continuous_metric_cache(...)` when available.
- A conservative internal cache budget constant is used:
  `256 * 1024 * 1024` bytes. Budget miss or unavailable API falls back to the
  existing report-completion path.
- Actual same-X scoring can use the cache, so the first call builds and scores;
  same-X permutations reuse the cache and only rescore y-dependent state.

Tests added:

- Native Core cache parity vs `score_combos_buffer(...)` for actual and permuted
  y, plus tiny-budget `None` behavior.
- Resident-session unit tests for missing-metric cache completion, cache reuse
  after target swap, changed-X fallback, and tiny-budget fallback.
- Engine-level count test proving one cache build, cached score calls for
  actual plus permutations, no report-completion fallback on same-X
  permutations, and full fallback only for stability repeats.

Verification on the Codex host:

- `python3 -m py_compile gafime/compile/sessions.py tests/test_compile_api.py tests/test_v045_native_spine.py`
- `python3 -m unittest tests.test_compile_api tests.test_v045_native_spine`
  - `35 tests` passed.
- `python3 -m unittest discover` is blocked by an existing package import issue:
  `gafime.preprocessors` imports missing `TimeSeriesConfig`.
- `python3 -m unittest discover tests` ran `55 tests`; one ROCm policy test
  failed because this host resolved auto priority as `['core']` while the test
  expected `['rocm', 'core']`.

### Deferred / Non-Blocking Items

- Rebuild `build/libgafime_rocm.so` in a proper payload/device build
  environment.
- Whole-sweep single-graph capture.
- Metal ICB exploration.
- Time-series metric-cache v2 after continuous v2 is proven.
- Discrete residency remains deferred; native ridge baseline already removed
  the measured discrete bottleneck.
- Vector ISA cleanup for `build_interaction`, `rankdata`, MI binning, and ridge
  XtX remains possible headroom but should follow measurements, not hunches.
