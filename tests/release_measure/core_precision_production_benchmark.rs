//! Standalone production-executor benchmark for Core precision profiles.
//!
//! This tracked common harness is deliberately separate from the supplemental
//! direct-kernel diagnostic in `core_precision_native_benchmark.rs`.  Every
//! invocation creates the real Core path:
//!
//! ```text
//! planner/protocol -> CpuPrecisionMatrix -> PrecisionComputeBackend
//!                  -> ranked typed result table -> deterministic digest
//! ```
//!
//! It is compiled only by `run_core_precision_production_benchmark.py` against
//! explicitly named product rlibs.  The runner starts one fresh process for
//! every profile/metric/workload/input-policy/worker cell so neither a loaded
//! runtime nor a Rayon pool can silently leak between variants or cells.

use std::{
    collections::BTreeSet,
    env,
    fmt::Write as _,
    fs,
    hint::black_box,
    io::Write as _,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    time::Instant,
};

use gafime_cpu::{
    precision::CpuPrecisionMatrix,
    result::{OwnedResultTable, OwnedResultTableF64},
    CpuBackend,
};
use gafime_orchestrator::{
    config::EngineConfig, prepare_ranked_continuous_execution_for_feature_orders,
};
use gafime_types::{
    GafimeRankSpec, PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_METRIC_MUTUAL_INFO,
    GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
};
use rayon::{ThreadPool, ThreadPoolBuilder};

const MIN_RELEASE_WARMUPS: usize = 10;
const MIN_RELEASE_REPETITIONS: usize = 30;
const TARGET_REGION_NS: u128 = 100_000_000;
const CALIBRATION_TARGET_REGION_NS: u128 = 200_000_000;
const MAX_LOOP_COUNT: usize = 4_096;

const COMPILED_HARNESS_SOURCE_SHA256: Option<&str> =
    option_env!("GAFIME_COMPILED_HARNESS_SOURCE_SHA256");
const COMPILED_HARNESS_SOURCE_GIT_BLOB: Option<&str> =
    option_env!("GAFIME_COMPILED_HARNESS_SOURCE_GIT_BLOB");
const COMPILED_HARNESS_SOURCE_RELATIVE_PATH: Option<&str> =
    option_env!("GAFIME_COMPILED_HARNESS_SOURCE_RELATIVE_PATH");
const COMPILED_HARNESS_RUNNER_SHA256: Option<&str> =
    option_env!("GAFIME_COMPILED_HARNESS_RUNNER_SHA256");
const COMPILED_HARNESS_RUNNER_GIT_BLOB: Option<&str> =
    option_env!("GAFIME_COMPILED_HARNESS_RUNNER_GIT_BLOB");
const COMPILED_HARNESS_RUNNER_RELATIVE_PATH: Option<&str> =
    option_env!("GAFIME_COMPILED_HARNESS_RUNNER_RELATIVE_PATH");
const COMPILED_PRODUCT_RLIB_SHA256: Option<&str> =
    option_env!("GAFIME_COMPILED_PRODUCT_RLIB_SHA256");
const COMPILED_ORCHESTRATOR_RLIB_SHA256: Option<&str> =
    option_env!("GAFIME_COMPILED_ORCHESTRATOR_RLIB_SHA256");
const COMPILED_TYPES_RLIB_SHA256: Option<&str> = option_env!("GAFIME_COMPILED_TYPES_RLIB_SHA256");
const COMPILED_RAYON_RLIB_SHA256: Option<&str> = option_env!("GAFIME_COMPILED_RAYON_RLIB_SHA256");
const COMPILED_COMMAND_SHA256: Option<&str> = option_env!("GAFIME_COMPILED_COMMAND_SHA256");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Profile {
    Fp32,
    Mixed,
    Fp64,
}

impl Profile {
    fn parse(value: &str) -> Self {
        match value {
            "fp32" => Self::Fp32,
            "mixed" => Self::Mixed,
            "fp64" => Self::Fp64,
            _ => panic!("unsupported GAFIME_PRODUCTION_PROFILE: {value}"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Mixed => "mixed",
            Self::Fp64 => "fp64",
        }
    }

    const fn native(self) -> PrecisionProfile {
        match self {
            Self::Fp32 => PrecisionProfile::Fp32,
            Self::Mixed => PrecisionProfile::Mixed,
            Self::Fp64 => PrecisionProfile::Fp64,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Metric {
    Pearson,
    Spearman,
    MutualInfo,
    R2,
}

impl Metric {
    fn parse(value: &str) -> Self {
        match value {
            "pearson" => Self::Pearson,
            "spearman" => Self::Spearman,
            "mutual_info" => Self::MutualInfo,
            "r2" => Self::R2,
            _ => panic!("unsupported GAFIME_PRODUCTION_METRIC: {value}"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Pearson => "pearson",
            Self::Spearman => "spearman",
            Self::MutualInfo => "mutual_info",
            Self::R2 => "r2",
        }
    }

    const fn id(self) -> u32 {
        match self {
            Self::Pearson => GAFIME_METRIC_PEARSON,
            Self::Spearman => GAFIME_METRIC_SPEARMAN,
            Self::MutualInfo => GAFIME_METRIC_MUTUAL_INFO,
            Self::R2 => GAFIME_METRIC_R2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputPolicy {
    CommonF64,
    Native,
}

impl InputPolicy {
    fn parse(value: &str) -> Self {
        match value {
            "common-f64" => Self::CommonF64,
            "native" => Self::Native,
            _ => panic!("unsupported GAFIME_PRODUCTION_INPUT_POLICY: {value}"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::CommonF64 => "common-f64",
            Self::Native => "native",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Workload {
    Latency,
    Medium,
    Kernel,
}

#[derive(Clone, Copy, Debug)]
struct WorkloadSpec {
    name: &'static str,
    rows: usize,
    features: usize,
    candidates: usize,
    top_k: usize,
    mi_bins: u32,
}

impl Workload {
    fn parse(value: &str) -> Self {
        match value {
            "latency" => Self::Latency,
            "medium" => Self::Medium,
            "kernel" => Self::Kernel,
            _ => panic!("unsupported GAFIME_PRODUCTION_WORKLOAD: {value}"),
        }
    }

    const fn spec(self) -> WorkloadSpec {
        match self {
            // A latency-oriented public-sized plan, while still retaining
            // enough independent candidates to exercise worker scheduling.
            Self::Latency => WorkloadSpec {
                name: "latency",
                rows: 2_048,
                features: 48,
                candidates: 48,
                top_k: 16,
                mi_bins: 32,
            },
            // Mixed planner/executor overhead: candidate count is deliberately
            // well above a typical four-worker pool.
            Self::Medium => WorkloadSpec {
                name: "medium",
                rows: 8_192,
                features: 128,
                candidates: 128,
                top_k: 32,
                mi_bins: 48,
            },
            // Kernel-dominant resident execution. The profile-specific
            // production executor, not a leaf kernel, remains the timed unit.
            Self::Kernel => WorkloadSpec {
                name: "kernel",
                rows: 32_768,
                features: 256,
                candidates: 256,
                top_k: 64,
                mi_bins: 64,
            },
        }
    }
}

#[derive(Clone, Debug)]
enum MatrixInput {
    F32 {
        features: Vec<f32>,
        target: Vec<f32>,
    },
    F64 {
        features: Vec<f64>,
        target: Vec<f64>,
    },
}

#[derive(Clone, Debug)]
struct InputSet {
    matrix: MatrixInput,
    f32_feature_digest: String,
    f32_target_digest: String,
    f64_feature_digest: String,
    f64_target_digest: String,
}

struct PreparedCase {
    profile: Profile,
    metric: Metric,
    spec: WorkloadSpec,
    rank: GafimeRankSpec,
    prepared: gafime_orchestrator::PreparedContinuousExecution,
    matrix: CpuPrecisionMatrix,
    planner_protocol_ns: u128,
    resident_matrix_ns: u128,
    input: InputSet,
}

struct ProductionMeasurement {
    spec: WorkloadSpec,
    input: InputSet,
    planner_protocol_ns: u128,
    resident_matrix_ns: u128,
    loops: usize,
    raw_samples: Vec<u128>,
    normalized_samples: Vec<f64>,
    final_digest: ExecutionDigest,
    result_snapshot: ResultSnapshot,
}

#[derive(Clone, Copy, Debug)]
struct ExecutionDigest {
    rows_written: u64,
    visible_score_bits: u64,
    candidate_digest: u64,
}

#[derive(Clone, Debug)]
struct ResultSnapshot {
    result_dtype: &'static str,
    row_count: usize,
    max_arity: usize,
    metric_count: usize,
    result_flags: u32,
    metric_ids: Vec<u32>,
    combo_indices: Vec<u32>,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
    metric_value_bits: Vec<u64>,
    metric_value_text: Vec<String>,
    metric_value_classes: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct WorkerStartRecord {
    worker_id: usize,
    os_tid: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct WorkerCpuTickRecord {
    worker_id: usize,
    os_tid: Option<u64>,
    ticks_before: Option<u64>,
    ticks_after: Option<u64>,
    work_ticks: Option<u64>,
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn parse_positive_env(name: &str) -> usize {
    let parsed = required_env(name)
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{name} must be a positive usize"));
    assert!(parsed > 0, "{name} must be positive");
    parsed
}

fn parse_usize_env(name: &str) -> usize {
    required_env(name)
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{name} must be a non-negative usize"))
}

fn parse_worker_mode() -> (String, Option<usize>) {
    let mode = required_env("GAFIME_PRODUCTION_RAYON_WORKERS");
    if mode == "default" {
        (mode, None)
    } else {
        let count = mode
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("GAFIME_PRODUCTION_RAYON_WORKERS must be default or usize"));
        assert!(
            count > 0,
            "GAFIME_PRODUCTION_RAYON_WORKERS must be positive"
        );
        (mode, Some(count))
    }
}

/// Setup-only evidence for the dedicated Rayon pool.  This proves only that
/// the pool created worker threads; it deliberately does *not* claim that a
/// particular worker scored a candidate.  The repository's cfg(test)
/// production-executor topology test owns that stronger dynamic assertion.
struct BenchmarkPool {
    pool: ThreadPool,
    started_workers: Arc<Mutex<Vec<WorkerStartRecord>>>,
}

#[cfg(target_os = "linux")]
fn current_os_tid() -> Option<u64> {
    fs::read_to_string("/proc/thread-self/stat")
        .ok()?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
}

#[cfg(not(target_os = "linux"))]
fn current_os_tid() -> Option<u64> {
    None
}

fn make_pool(requested: Option<usize>, allowed_parallelism: usize) -> BenchmarkPool {
    let started_workers = Arc::new(Mutex::new(Vec::new()));
    let start_handler_workers = Arc::clone(&started_workers);
    // Do not let an inherited global Rayon setting make the default worker
    // mode differ from the process's actual allowed CPU set.  The Python
    // runner removes RAYON_NUM_THREADS, and this dedicated pool is always
    // explicitly bounded here as a second fail-closed layer.
    let worker_count = requested.unwrap_or(allowed_parallelism);
    assert!(
        worker_count <= allowed_parallelism,
        "requested Rayon workers must not exceed std::thread::available_parallelism"
    );
    let builder = ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .start_handler(move |worker_id| {
            start_handler_workers
                .lock()
                .expect("record dedicated Rayon worker start")
                .push(WorkerStartRecord {
                    worker_id,
                    os_tid: current_os_tid(),
                });
        });
    BenchmarkPool {
        pool: builder
            .build()
            .unwrap_or_else(|error| panic!("create dedicated Rayon benchmark pool: {error}")),
        started_workers,
    }
}

fn pool_start_workers(pool: &BenchmarkPool) -> Vec<WorkerStartRecord> {
    // `broadcast` runs once on every worker and is intentionally outside every
    // measured region.  It ensures the setup-only start-handler evidence is
    // complete before the per-thread CPU tick baseline is sampled.
    pool.pool.broadcast(|_| ());
    let mut workers = pool
        .started_workers
        .lock()
        .expect("read dedicated Rayon worker start evidence")
        .clone();
    workers.sort_unstable_by_key(|record| record.worker_id);
    workers.dedup_by_key(|record| record.worker_id);
    workers
}

#[cfg(target_os = "linux")]
fn linux_thread_cpu_ticks(tid: u64) -> Option<u64> {
    let stat = fs::read_to_string(format!("/proc/self/task/{tid}/stat")).ok()?;
    // The second field is a parenthesized comm that may contain spaces.  The
    // suffix begins with field 3 (state), so utime/stime fields 14/15 are
    // suffix indexes 11/12.
    let suffix = stat.get(stat.rfind(')')?.checked_add(2)?..)?;
    let fields = suffix.split_whitespace().collect::<Vec<_>>();
    let user = fields.get(11)?.parse::<u64>().ok()?;
    let system = fields.get(12)?.parse::<u64>().ok()?;
    user.checked_add(system)
}

#[cfg(not(target_os = "linux"))]
fn linux_thread_cpu_ticks(_tid: u64) -> Option<u64> {
    None
}

fn worker_tick_snapshot(workers: &[WorkerStartRecord]) -> Vec<Option<u64>> {
    workers
        .iter()
        .map(|record| record.os_tid.and_then(linux_thread_cpu_ticks))
        .collect()
}

fn worker_cpu_tick_records(
    workers: &[WorkerStartRecord],
    before: &[Option<u64>],
    after: &[Option<u64>],
) -> Vec<WorkerCpuTickRecord> {
    workers
        .iter()
        .zip(before)
        .zip(after)
        .map(|((worker, before), after)| WorkerCpuTickRecord {
            worker_id: worker.worker_id,
            os_tid: worker.os_tid,
            ticks_before: *before,
            ticks_after: *after,
            work_ticks: (*before)
                .zip(*after)
                .and_then(|(start, end)| end.checked_sub(start)),
        })
        .collect()
}

fn pseudo_random(index: usize, feature: usize) -> f64 {
    let a = ((index.wrapping_mul(37) + feature.wrapping_mul(97) + index / 13) % 65_521) as f64
        / 65_521.0;
    let b = ((index.wrapping_mul(17) + feature.wrapping_mul(11) + 3) % 997) as f64 / 997.0;
    a + b * 0.031_25 + ((index + feature * 31) as f64 * 0.000_013_7).sin() * 1.0e-6
}

fn build_inputs(policy: InputPolicy, spec: WorkloadSpec, profile: Profile) -> InputSet {
    let element_count = spec
        .rows
        .checked_mul(spec.features)
        .expect("workload matrix size fits usize");
    let mut f64_features = Vec::with_capacity(element_count);
    let mut f64_target = Vec::with_capacity(spec.rows);
    for row in 0..spec.rows {
        let mut target = 0.0f64;
        for feature in 0..spec.features {
            let value =
                pseudo_random(row, feature) + (feature as f64 * 0.000_000_73).cos() * 1.0e-7;
            target += value * (1.0 + (feature % 7) as f64 * 0.01);
            f64_features.push(value);
        }
        f64_target.push((target * 0.017 + (row as f64 * 0.000_071).cos()).sin());
    }
    let f32_features: Vec<f32> = match policy {
        InputPolicy::CommonF64 => f64_features
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect(),
        InputPolicy::Native => (0..spec.rows)
            .flat_map(|row| {
                (0..spec.features).map(move |feature| {
                    let row32 = row as f32;
                    let feature32 = feature as f32;
                    let a = ((row.wrapping_mul(37) + feature.wrapping_mul(97) + row / 13) % 65_521)
                        as f32
                        / 65_521.0f32;
                    let b = ((row.wrapping_mul(17) + feature.wrapping_mul(11) + 3) % 997) as f32
                        / 997.0f32;
                    a + b * 0.031_25f32
                        + ((row32 + feature32 * 31.0f32) * 0.000_013_7f32).sin() * 1.0e-6f32
                        + (feature32 * 0.000_000_73f32).cos() * 1.0e-7f32
                })
            })
            .collect(),
    };
    let f32_target: Vec<f32> = match policy {
        InputPolicy::CommonF64 => f64_target
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect(),
        InputPolicy::Native => (0..spec.rows)
            .map(|row| {
                let target = (0..spec.features)
                    .map(|feature| {
                        let row32 = row as f32;
                        let feature32 = feature as f32;
                        let a = ((row.wrapping_mul(37) + feature.wrapping_mul(97) + row / 13)
                            % 65_521) as f32
                            / 65_521.0f32;
                        let b = ((row.wrapping_mul(17) + feature.wrapping_mul(11) + 3) % 997)
                            as f32
                            / 997.0f32;
                        (a + b * 0.031_25f32
                            + ((row32 + feature32 * 31.0f32) * 0.000_013_7f32).sin() * 1.0e-6f32
                            + (feature32 * 0.000_000_73f32).cos() * 1.0e-7f32)
                            * (1.0f32 + (feature % 7) as f32 * 0.01f32)
                    })
                    .sum::<f32>();
                (target * 0.017f32 + (row as f32 * 0.000_071f32).cos()).sin()
            })
            .collect(),
    };
    assert!(
        f64_features
            .iter()
            .zip(&f32_features)
            .any(|(&wide, &narrow)| wide != f64::from(narrow)),
        "f64 source must retain values that do not round-trip through f32"
    );
    let matrix = if profile == Profile::Fp64 {
        MatrixInput::F64 {
            features: f64_features.clone(),
            target: f64_target.clone(),
        }
    } else {
        MatrixInput::F32 {
            features: f32_features.clone(),
            target: f32_target.clone(),
        }
    };
    InputSet {
        matrix,
        f32_feature_digest: sha256_bytes(&f32_bytes(&f32_features)),
        f32_target_digest: sha256_bytes(&f32_bytes(&f32_target)),
        f64_feature_digest: sha256_bytes(&f64_bytes(&f64_features)),
        f64_target_digest: sha256_bytes(&f64_bytes(&f64_target)),
    }
}

fn prepare_case(
    profile: Profile,
    metric: Metric,
    workload: Workload,
    policy: InputPolicy,
) -> PreparedCase {
    let spec = workload.spec();
    let input = build_inputs(policy, spec, profile);
    let matrix_start = Instant::now();
    let matrix = match &input.matrix {
        MatrixInput::F32 { features, target } => CpuPrecisionMatrix::from_row_major_f32(
            profile.native(),
            spec.rows as u64,
            spec.features as u32,
            features.clone(),
            target.clone(),
        ),
        MatrixInput::F64 { features, target } => CpuPrecisionMatrix::from_row_major_f64(
            profile.native(),
            spec.rows as u64,
            spec.features as u32,
            features.clone(),
            target.clone(),
        ),
    }
    .unwrap_or_else(|error| panic!("construct typed CpuPrecisionMatrix: {error:?}"));
    let resident_matrix_ns = matrix_start.elapsed().as_nanos();

    let mut config = EngineConfig::default();
    config.precision = profile.native();
    config.backend_kind = GAFIME_BACKEND_CPU;
    config.metric_ids = vec![metric.id()];
    config.mi_bins = spec.mi_bins;
    // The production CPU precision executor receives the same explicit fixed
    // MI path used by GPU parity/precision validation. This is a real user
    // configuration, recorded below rather than disguised as a leaf helper.
    config.mi_approximate = true;
    config.budget.max_comb_size = 1;
    config.budget.max_combinations_per_k = spec.candidates as u64;
    config.budget.max_feature_candidate = spec.candidates as i64;
    let rank = GafimeRankSpec {
        top_k: spec.top_k as u32,
        primary_metric: metric.id(),
        descending: 1,
        include_ties: 0,
        reserved: [0; 4],
    };
    let unary_features = (0..spec.candidates as u32).collect::<Vec<_>>();
    let planning_start = Instant::now();
    let prepared = prepare_ranked_continuous_execution_for_feature_orders(
        &config,
        spec.rows as u64,
        spec.features as u32,
        &unary_features,
        &[],
        true,
        false,
        rank,
    )
    .unwrap_or_else(|error| panic!("build production Core planner/protocol: {error:?}"));
    let planner_protocol_ns = planning_start.elapsed().as_nanos();
    PreparedCase {
        profile,
        metric,
        spec,
        rank,
        prepared,
        matrix,
        planner_protocol_ns,
        resident_matrix_ns,
        input,
    }
}

fn f32_digest(table: &OwnedResultTable, metric_ids: &[u32]) -> ExecutionDigest {
    let candidate_digest = structural_digest(
        32,
        table.row_count(),
        table.max_arity(),
        table.metric_count(),
        table.raw().flags,
        metric_ids,
        table.combo_indices(),
        table.ranks(),
        table.families(),
        table.candidate_ids(),
        table.row_flags(),
    );
    let visible_value_count = table
        .row_count()
        .checked_mul(table.metric_count())
        .expect("fp32 visible result width fits usize");
    let mut visible_score_bits = 0u64;
    for value in table.metric_values().iter().take(visible_value_count) {
        visible_score_bits = visible_score_bits.rotate_left(11) ^ u64::from(value.to_bits());
    }
    ExecutionDigest {
        rows_written: table.row_count() as u64,
        visible_score_bits,
        candidate_digest,
    }
}

fn f64_digest(table: &OwnedResultTableF64, metric_ids: &[u32]) -> ExecutionDigest {
    let candidate_digest = structural_digest(
        64,
        table.row_count(),
        table.max_arity(),
        table.metric_count(),
        table.raw().flags,
        metric_ids,
        table.combo_indices(),
        table.ranks(),
        table.families(),
        table.candidate_ids(),
        table.row_flags(),
    );
    let visible_value_count = table
        .row_count()
        .checked_mul(table.metric_count())
        .expect("f64 visible result width fits usize");
    let mut visible_score_bits = 0u64;
    for value in table.metric_values().iter().take(visible_value_count) {
        visible_score_bits = visible_score_bits.rotate_left(11) ^ value.to_bits();
    }
    ExecutionDigest {
        rows_written: table.row_count() as u64,
        visible_score_bits,
        candidate_digest,
    }
}

fn structural_digest(
    result_dtype_tag: u64,
    row_count: usize,
    max_arity: usize,
    metric_count: usize,
    result_flags: u32,
    metric_ids: &[u32],
    combo_indices: &[u32],
    ranks: &[u32],
    families: &[u32],
    candidate_ids: &[u64],
    row_flags: &[u32],
) -> u64 {
    let mut digest = (row_count as u64).rotate_left(17)
        ^ (max_arity as u64).rotate_left(29)
        ^ (metric_count as u64).rotate_left(41)
        ^ u64::from(result_flags).rotate_left(53)
        ^ result_dtype_tag;
    for &metric_id in metric_ids {
        digest = digest.rotate_left(5) ^ u64::from(metric_id);
    }
    for row in 0..row_count {
        let combo_start = row
            .checked_mul(max_arity)
            .expect("visible result combo offset fits usize");
        for &feature in &combo_indices[combo_start..combo_start + max_arity] {
            digest = digest.rotate_left(5) ^ u64::from(feature);
        }
        digest = digest.rotate_left(3) ^ u64::from(ranks[row]);
        digest = digest.rotate_left(3) ^ u64::from(families[row]);
        digest = digest.rotate_left(7) ^ candidate_ids[row];
        digest = digest.rotate_left(3) ^ u64::from(row_flags[row]);
    }
    digest
}

enum OwnedPrecisionResult {
    F32(OwnedResultTable),
    F64(OwnedResultTableF64),
}

fn execute_result(case: &PreparedCase) -> OwnedPrecisionResult {
    let mut backend = CpuBackend;
    let matrix = case.matrix.handle();
    let capacity = case
        .prepared
        .ranked_result_capacity(case.rank)
        .unwrap_or_else(|error| panic!("query ranked result capacity: {error:?}"));
    match case.profile {
        Profile::Fp32 => {
            let mut result = OwnedResultTable::new(
                capacity,
                case.prepared.result_max_arity(),
                case.prepared.result_metric_count(),
            );
            case.prepared
                .execute_precision_ranked_fp32(case.rank, &mut backend, &matrix, result.raw_mut())
                .unwrap_or_else(|error| panic!("execute production fp32 Core path: {error:?}"));
            OwnedPrecisionResult::F32(result)
        }
        Profile::Mixed | Profile::Fp64 => {
            let mut result = OwnedResultTableF64::new(
                capacity,
                case.prepared.result_max_arity(),
                case.prepared.result_metric_count(),
            );
            case.prepared
                .execute_precision_ranked_f64(case.rank, &mut backend, &matrix, result.raw_mut())
                .unwrap_or_else(|error| panic!("execute production f64 Core path: {error:?}"));
            OwnedPrecisionResult::F64(result)
        }
    }
}

fn execute_one(case: &PreparedCase) -> ExecutionDigest {
    let metric_ids = [case.metric.id()];
    match execute_result(case) {
        OwnedPrecisionResult::F32(result) => f32_digest(&result, &metric_ids),
        OwnedPrecisionResult::F64(result) => f64_digest(&result, &metric_ids),
    }
}

fn value_class(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else if value == f64::INFINITY {
        "positive_infinity".to_owned()
    } else if value == f64::NEG_INFINITY {
        "negative_infinity".to_owned()
    } else {
        "finite".to_owned()
    }
}

fn result_snapshot(case: &PreparedCase) -> ResultSnapshot {
    match execute_result(case) {
        OwnedPrecisionResult::F32(result) => {
            let row_count = result.row_count();
            let max_arity = result.max_arity();
            let metric_count = result.metric_count();
            let combo_count = row_count
                .checked_mul(max_arity)
                .expect("fp32 snapshot combo width fits usize");
            let value_count = row_count
                .checked_mul(metric_count)
                .expect("fp32 snapshot width fits usize");
            let values = &result.metric_values()[..value_count];
            ResultSnapshot {
                result_dtype: "f32",
                row_count,
                max_arity,
                metric_count,
                result_flags: result.raw().flags,
                metric_ids: vec![case.metric.id()],
                combo_indices: result.combo_indices()[..combo_count].to_vec(),
                ranks: result.ranks()[..row_count].to_vec(),
                families: result.families()[..row_count].to_vec(),
                candidate_ids: result.candidate_ids()[..row_count].to_vec(),
                row_flags: result.row_flags()[..row_count].to_vec(),
                metric_value_bits: values
                    .iter()
                    .map(|value| u64::from(value.to_bits()))
                    .collect(),
                metric_value_text: values.iter().map(|value| format!("{value:.9e}")).collect(),
                metric_value_classes: values
                    .iter()
                    .map(|value| value_class(f64::from(*value)))
                    .collect(),
            }
        }
        OwnedPrecisionResult::F64(result) => {
            let row_count = result.row_count();
            let max_arity = result.max_arity();
            let metric_count = result.metric_count();
            let combo_count = row_count
                .checked_mul(max_arity)
                .expect("f64 snapshot combo width fits usize");
            let value_count = row_count
                .checked_mul(metric_count)
                .expect("f64 snapshot width fits usize");
            let values = &result.metric_values()[..value_count];
            ResultSnapshot {
                result_dtype: "f64",
                row_count,
                max_arity,
                metric_count,
                result_flags: result.raw().flags,
                metric_ids: vec![case.metric.id()],
                combo_indices: result.combo_indices()[..combo_count].to_vec(),
                ranks: result.ranks()[..row_count].to_vec(),
                families: result.families()[..row_count].to_vec(),
                candidate_ids: result.candidate_ids()[..row_count].to_vec(),
                row_flags: result.row_flags()[..row_count].to_vec(),
                metric_value_bits: values.iter().map(|value| value.to_bits()).collect(),
                metric_value_text: values.iter().map(|value| format!("{value:.17e}")).collect(),
                metric_value_classes: values.iter().map(|value| value_class(*value)).collect(),
            }
        }
    }
}

fn measured_region(case: &PreparedCase, loops: usize) -> (u128, ExecutionDigest) {
    let start = Instant::now();
    let mut digest = ExecutionDigest {
        rows_written: 0,
        visible_score_bits: 0,
        candidate_digest: 0,
    };
    for _ in 0..loops {
        digest = execute_one(case);
        black_box(digest);
    }
    (start.elapsed().as_nanos(), digest)
}

fn median_u128(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn calibrate(case: &PreparedCase, warmups: usize) -> usize {
    for _ in 0..warmups {
        black_box(measured_region(case, 1));
    }
    let probe = median_u128(
        (0..5)
            .map(|_| measured_region(case, 1).0)
            .collect::<Vec<_>>(),
    )
    .max(1);
    usize::try_from(CALIBRATION_TARGET_REGION_NS.div_ceil(probe))
        .unwrap_or(MAX_LOOP_COUNT)
        .clamp(1, MAX_LOOP_COUNT)
}

/// Construct the ABI-bound plan and resident matrix inside the dedicated Rayon
/// pool, then retain both for all warmups and samples.  The ABI protocol owns
/// raw pointers and is intentionally not Sync; keeping it pool-local avoids
/// inventing unsafe sharing merely to benchmark the real production executor.
fn measure_case_in_pool(
    profile: Profile,
    metric: Metric,
    workload: Workload,
    policy: InputPolicy,
    pool: &ThreadPool,
    warmups: usize,
    repetitions: usize,
) -> ProductionMeasurement {
    pool.install(move || {
        let case = prepare_case(profile, metric, workload, policy);
        let loops = calibrate(&case, warmups);
        for _ in 0..warmups {
            black_box(measured_region(&case, loops));
        }
        let mut raw_samples = Vec::with_capacity(repetitions);
        let mut normalized_samples = Vec::with_capacity(repetitions);
        let mut final_digest = ExecutionDigest {
            rows_written: 0,
            visible_score_bits: 0,
            candidate_digest: 0,
        };
        for _ in 0..repetitions {
            let (duration, digest) = measured_region(&case, loops);
            raw_samples.push(duration);
            normalized_samples.push(duration as f64 / loops as f64);
            final_digest = digest;
        }
        // This complete ordered snapshot is deliberately outside every timed
        // region.  It authenticates visible result identity/parity without
        // charging string conversion or comparison evidence to product time.
        let result_snapshot = result_snapshot(&case);
        ProductionMeasurement {
            spec: case.spec,
            input: case.input,
            planner_protocol_ns: case.planner_protocol_ns,
            resident_matrix_ns: case.resident_matrix_ns,
            loops,
            raw_samples,
            normalized_samples,
            final_digest,
            result_snapshot,
        }
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>()
}

fn digest_command(program: &str, args: &[&str], bytes: &[u8]) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(bytes).ok()?;
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let digest = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    (digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(digest)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    digest_command("sha256sum", &[], bytes)
        .or_else(|| digest_command("shasum", &["-a", "256"], bytes))
        .expect("sha256sum or shasum must be available for benchmark provenance")
}

fn json_string(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len() + 2);
    encoded.push('"');
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if character <= '\u{1f}' => {
                write!(encoded, "\\u{:04x}", character as u32).expect("write JSON escape");
            }
            character => encoded.push(character),
        }
    }
    encoded.push('"');
    encoded
}

fn json_strings(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_u64s(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn json_u32s(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn json_worker_cpu_ticks(records: &[WorkerCpuTickRecord]) -> String {
    records
        .iter()
        .map(|record| {
            let optional = |value: Option<u64>| {
                value
                    .map(|number| number.to_string())
                    .unwrap_or_else(|| "null".to_owned())
            };
            format!(
                "{{\"worker_id\":{},\"os_tid\":{},\"cpu_ticks_before\":{},\"cpu_ticks_after\":{},\"work_ticks\":{}}}",
                record.worker_id,
                optional(record.os_tid),
                optional(record.ticks_before),
                optional(record.ticks_after),
                optional(record.work_ticks),
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn process_affinity() -> String {
    if cfg!(target_os = "linux") {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            if let Some(line) = status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
            {
                return line
                    .split_once(':')
                    .map(|(_, value)| value.trim().to_owned())
                    .unwrap_or_default();
            }
        }
    }
    command_output("taskset", &["-pc", &std::process::id().to_string()])
}

fn linux_affinity_cardinality(value: &str) -> Option<usize> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let mut count = 0usize;
    for part in value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start.parse::<usize>().ok()?;
            let end = end.parse::<usize>().ok()?;
            count = count.checked_add(end.checked_sub(start)?.checked_add(1)?)?;
        } else {
            part.parse::<usize>().ok()?;
            count = count.checked_add(1)?;
        }
    }
    (count > 0).then_some(count)
}

fn allowed_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

fn cpu_logical_count() -> usize {
    if cfg!(target_os = "linux") {
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            let count = cpuinfo
                .lines()
                .filter(|line| line.trim_start().starts_with("processor"))
                .count();
            if count > 0 {
                return count;
            }
        }
    }
    command_output("sysctl", &["-n", "hw.logicalcpu"])
        .parse::<usize>()
        .ok()
        .filter(|count| *count > 0)
        .unwrap_or_else(allowed_parallelism)
}

fn cpu_physical_count() -> Option<usize> {
    if cfg!(target_os = "linux") {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
        let mut cores = BTreeSet::new();
        for paragraph in cpuinfo.split("\n\n") {
            let mut package = None;
            let mut core = None;
            for line in paragraph.lines() {
                let Some((name, value)) = line.split_once(':') else {
                    continue;
                };
                match name.trim() {
                    "physical id" => package = Some(value.trim().to_owned()),
                    "core id" => core = Some(value.trim().to_owned()),
                    _ => {}
                }
            }
            if let (Some(package), Some(core)) = (package, core) {
                cores.insert((package, core));
            }
        }
        return (!cores.is_empty()).then_some(cores.len());
    }
    let output = command_output("sysctl", &["-n", "hw.physicalcpu"]);
    output.parse::<usize>().ok().filter(|count| *count > 0)
}

fn cpu_governors() -> Vec<String> {
    let Ok(policies) = fs::read_dir("/sys/devices/system/cpu/cpufreq") else {
        return vec!["unobservable".to_owned()];
    };
    let mut values = policies
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("policy"))
        .filter_map(|entry| fs::read_to_string(entry.path().join("scaling_governor")).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    if values.is_empty() {
        values.push("unobservable".to_owned());
    }
    values
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Capture observable CPU governor, clock, and platform-power state immediately
/// before/after the timed regions.  The benchmark never changes these controls.
fn cpu_clock_power_snapshot(governors: &[String]) -> String {
    let mut policy_records = Vec::new();
    if let Ok(policies) = fs::read_dir("/sys/devices/system/cpu/cpufreq") {
        let mut paths = policies
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("policy"))
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let policy = path
                .file_name()
                .map(|value| value.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_owned());
            let field = |name: &str| {
                read_trimmed(&path.join(name)).unwrap_or_else(|| "unobservable".to_owned())
            };
            policy_records.push(format!(
                "{{\"policy\":{},\"scaling_cur_freq_khz\":{},\"scaling_min_freq_khz\":{},\"scaling_max_freq_khz\":{},\"cpuinfo_min_freq_khz\":{},\"cpuinfo_max_freq_khz\":{},\"energy_performance_preference\":{}}}",
                json_string(&policy),
                json_string(&field("scaling_cur_freq")),
                json_string(&field("scaling_min_freq")),
                json_string(&field("scaling_max_freq")),
                json_string(&field("cpuinfo_min_freq")),
                json_string(&field("cpuinfo_max_freq")),
                json_string(&field("energy_performance_preference")),
            ));
        }
    }
    let linux_platform_profile = read_trimmed(Path::new("/sys/firmware/acpi/platform_profile"));
    let macos_pmset = if env::consts::OS == "macos" {
        let value = command_output("pmset", &["-g", "custom"]);
        (!value.is_empty()).then_some(value)
    } else {
        None
    };
    let power_interface = if linux_platform_profile.is_some() {
        "linux_acpi_platform_profile"
    } else if macos_pmset.is_some() {
        "macos_pmset_custom"
    } else {
        "unobservable"
    };
    format!(
        "{{\"cpu_governor\":{},\"policy_clock_state\":[{}],\"platform_power_profile\":{},\"macos_pmset_custom\":{},\"power_interface\":{}}}",
        json_strings(governors),
        policy_records.join(","),
        json_string(linux_platform_profile.as_deref().unwrap_or("unobservable")),
        json_string(macos_pmset.as_deref().unwrap_or("unobservable")),
        json_string(power_interface),
    )
}

fn cpu_identity() -> String {
    if cfg!(target_os = "linux") {
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            if let Some(model) = cpuinfo
                .lines()
                .find_map(|line| line.strip_prefix("model name\t: "))
            {
                return model.to_owned();
            }
        }
    }
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
}

fn json_environment() -> String {
    const KEYS: [&str; 18] = [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "GOMP_CPU_AFFINITY",
        "KMP_AFFINITY",
        "KMP_HW_SUBSET",
        "MALLOC_ARENA_MAX",
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "RUSTFLAGS",
    ];
    let mut entries = Vec::new();
    for key in KEYS {
        if let Ok(value) = env::var(key) {
            entries.push(format!("{}:{}", json_string(key), json_string(&value)));
        }
    }
    format!("{{{}}}", entries.join(","))
}

fn main() {
    let profile = Profile::parse(&required_env("GAFIME_PRODUCTION_PROFILE"));
    let metric = Metric::parse(&required_env("GAFIME_PRODUCTION_METRIC"));
    let workload = Workload::parse(&required_env("GAFIME_PRODUCTION_WORKLOAD"));
    let policy = InputPolicy::parse(&required_env("GAFIME_PRODUCTION_INPUT_POLICY"));
    let warmups = parse_positive_env("GAFIME_PRODUCTION_WARMUPS");
    let repetitions = parse_positive_env("GAFIME_PRODUCTION_REPETITIONS");
    let (worker_mode, requested_workers) = parse_worker_mode();
    let governor_before = cpu_governors();
    let clock_power_before = cpu_clock_power_snapshot(&governor_before);
    let logical_cpu_count = cpu_logical_count();
    let physical_cpu_count = cpu_physical_count();
    let allowed_parallelism = allowed_parallelism();
    let process_affinity = process_affinity();
    let affinity_cardinality = linux_affinity_cardinality(&process_affinity);
    let affinity_matches_allowed = affinity_cardinality == Some(allowed_parallelism);
    let pool = make_pool(requested_workers, allowed_parallelism);
    let effective_workers = pool.pool.current_num_threads();
    assert_eq!(
        effective_workers,
        requested_workers.unwrap_or(allowed_parallelism),
        "dedicated Rayon pool must honor the requested/default worker count"
    );
    let pool_start_workers = pool_start_workers(&pool);
    assert_eq!(
        pool_start_workers.len(),
        effective_workers,
        "all dedicated Rayon workers must be observed by the pool start handler"
    );
    let worker_ticks_before = worker_tick_snapshot(&pool_start_workers);
    let measurement = measure_case_in_pool(
        profile,
        metric,
        workload,
        policy,
        &pool.pool,
        warmups,
        repetitions,
    );
    let worker_ticks_after = worker_tick_snapshot(&pool_start_workers);
    let worker_cpu_ticks = worker_cpu_tick_records(
        &pool_start_workers,
        &worker_ticks_before,
        &worker_ticks_after,
    );
    let worker_ticks_observable = cfg!(target_os = "linux")
        && worker_cpu_ticks.iter().all(|record| {
            record.os_tid.is_some()
                && record.ticks_before.is_some()
                && record.ticks_after.is_some()
                && record.work_ticks.is_some()
        });
    let every_worker_has_positive_work_ticks = worker_ticks_observable
        && worker_cpu_ticks
            .iter()
            .all(|record| record.work_ticks.is_some_and(|ticks| ticks > 0));
    let worker_tick_status = if every_worker_has_positive_work_ticks {
        "all_effective_workers_positive"
    } else if worker_ticks_observable {
        "observable_but_one_or_more_workers_zero"
    } else {
        "portable_unobservable"
    };
    let raw_minimum = measurement.raw_samples.iter().copied().min().unwrap_or(0);
    let governor_after = cpu_governors();
    let clock_power_after = cpu_clock_power_snapshot(&governor_after);
    let claim_ready = warmups >= MIN_RELEASE_WARMUPS
        && repetitions >= MIN_RELEASE_REPETITIONS
        && raw_minimum >= TARGET_REGION_NS
        && affinity_matches_allowed
        && every_worker_has_positive_work_ticks;
    let source_root = required_env("GAFIME_PRODUCTION_PRODUCT_SOURCE_ROOT");
    let harness_root = required_env("GAFIME_PRODUCTION_HARNESS_SOURCE_ROOT");
    let source_commit = required_env("GAFIME_PRODUCTION_PRODUCT_COMMIT");
    let source_tree = required_env("GAFIME_PRODUCTION_PRODUCT_TREE");
    let harness_commit = required_env("GAFIME_PRODUCTION_HARNESS_COMMIT");
    let harness_tree = required_env("GAFIME_PRODUCTION_HARNESS_TREE");
    let output = PathBuf::from(required_env("GAFIME_PRODUCTION_BENCH_OUTPUT"));
    let binary = required_env("GAFIME_PRODUCTION_BENCH_BINARY");
    let wheel = required_env("GAFIME_PRODUCTION_BENCH_WHEEL");
    let product_rlib = required_env("GAFIME_PRODUCTION_PRODUCT_RLIB");
    let orchestrator_rlib = required_env("GAFIME_PRODUCTION_ORCHESTRATOR_RLIB");
    let types_rlib = required_env("GAFIME_PRODUCTION_TYPES_RLIB");
    let rayon_rlib = required_env("GAFIME_PRODUCTION_RAYON_RLIB");
    let raw_json = measurement
        .raw_samples
        .iter()
        .map(u128::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let normalized_json = measurement
        .normalized_samples
        .iter()
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let command_line = json_strings(&env::args().collect::<Vec<_>>());
    let variant = required_env("GAFIME_PRODUCTION_VARIANT");
    let measurement_mode = required_env("GAFIME_PRODUCTION_MODE");
    let runner_pid = parse_positive_env("GAFIME_PRODUCTION_RUNNER_PID");
    assert_ne!(
        runner_pid,
        std::process::id() as usize,
        "Python runner and fresh benchmark child must have distinct PIDs"
    );
    let variant_sequence = required_env("GAFIME_PRODUCTION_VARIANT_SEQUENCE")
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(
        !variant_sequence.is_empty(),
        "GAFIME_PRODUCTION_VARIANT_SEQUENCE must contain at least one variant"
    );
    let ab_block = required_env("GAFIME_PRODUCTION_AB_BLOCK")
        .parse::<u32>()
        .expect("GAFIME_PRODUCTION_AB_BLOCK must be a u32");
    let schedule_index = parse_usize_env("GAFIME_PRODUCTION_SCHEDULE_INDEX");
    let schedule_seed = required_env("GAFIME_PRODUCTION_SCHEDULE_SEED")
        .parse::<u64>()
        .expect("GAFIME_PRODUCTION_SCHEDULE_SEED must be a u64");
    let schedule_sha256 = required_env("GAFIME_PRODUCTION_SCHEDULE_SHA256");
    assert!(
        schedule_sha256.len() == 64 && schedule_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "GAFIME_PRODUCTION_SCHEDULE_SHA256 must be a SHA-256"
    );
    let profile_order = required_env("GAFIME_PRODUCTION_PROFILE_ORDER")
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(
        profile_order.iter().any(|entry| entry == profile.name()),
        "cell profile must belong to its scheduled profile order"
    );
    let pool_start_worker_ids = pool_start_workers
        .iter()
        .map(|record| record.worker_id)
        .collect::<Vec<_>>();
    let affinity_cardinality_json = affinity_cardinality
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_owned());
    let report = format!(
        "{{\"schema\":\"gafime.core-production-executor.child.v1\",\"status\":\"{}\",\"claim_ready\":{},\"backend\":\"core\",\"profile\":{},\"metric\":{},\"operation\":\"production_executor_metric\",\"execution_surface\":\"planner_protocol_resident_precision_compute_backend_ranked_result\",\"measurement_scope\":\"production_precision_compute_backend\",\"candidate_family_scope\":\"ranked_unary_candidates_only\",\"measurement_mode\":{},\"input_policy\":{},\"workload\":{{\"name\":{},\"class\":\"production_core_executor\",\"rows\":{},\"features\":{},\"candidates\":{},\"arity\":1,\"top_k\":{},\"mi_bins\":{},\"mi_approximate\":true,\"metric_set\":[{}]}},\"warmups\":{},\"repetitions\":{},\"loop_count_per_sample\":{},\"target_region_ns\":{},\"calibration_target_region_ns\":{},\"sample_region_min_observed_ns\":{},\"sample_region_target_met\":{},\"samples_ns\":[{}],\"raw_samples_ns\":[{}],\"setup\":{{\"planner_protocol_ns\":{},\"resident_matrix_ns\":{},\"timed_region\":\"real ranked PrecisionComputeBackend execution plus typed result table allocation/materialization; input generation, resident matrix construction, planner/protocol setup, and the complete result snapshot are outside the timed region\",\"untimed_post_measurement_snapshot\":true}},\"execution_topology\":{{\"candidate_parallelism\":\"rayon_candidate_level\",\"worker_mode\":{},\"measurement_role\":{},\"requested_rayon_workers\":{},\"effective_rayon_workers\":{},\"allowed_parallelism\":{},\"allowed_parallelism_source\":\"std::thread::available_parallelism\",\"process_affinity\":{},\"process_affinity_cardinality\":{},\"affinity_matches_allowed_parallelism\":{},\"pool_start_worker_ids\":[{}],\"pool_start_worker_count\":{},\"pool_start_evidence_scope\":\"dedicated_pool_construction_only_not_candidate_work_participation\",\"worker_os_cpu_ticks\":[{}],\"worker_cpu_tick_status\":{},\"worker_cpu_ticks_observable\":{},\"every_effective_worker_positive_work_ticks\":{},\"worker_participation_evidence_scope\":\"per_worker_linux_os_tid_cpu_ticks_around_real_production_measurement\",\"semantic_candidate_participation_guard\":\"cfg_test_precision_executor_parallelism_contract\"}},\"result\":{{\"rows_written\":{},\"visible_score_bits\":{},\"candidate_digest\":{},\"digest_scope\":\"all_visible_result_metadata_structural_arrays_and_metric_bits\",\"untimed_snapshot\":{{\"result_dtype\":{},\"row_count\":{},\"max_arity\":{},\"metric_count\":{},\"result_flags\":{},\"metric_ids\":[{}],\"combo_indices\":[{}],\"ranks\":[{}],\"families\":[{}],\"candidate_ids\":[{}],\"row_flags\":[{}],\"metric_value_bits\":[{}],\"metric_value_text\":{},\"metric_value_classes\":{}}}}},\"variant\":{},\"ab_block\":{},\"variant_sequence\":{},\"cell_schedule\":{{\"index\":{},\"seed\":{},\"sha256\":{},\"profile_order\":{}}},\"runner_pid\":{},\"process_id\":{},\"runner_invocation_id\":{},\"process_isolation\":\"fresh_helper_process_per_variant_trial\",\"source_commit\":{},\"product_source_commit\":{},\"product_source_tree\":{},\"product_source_tree_state\":{{\"status\":\"clean\"}},\"harness_source_commit\":{},\"harness_source_tree\":{},\"harness_source_tree_state\":{{\"status\":\"clean\"}},\"harness_source_blob\":{{\"relative_path\":{},\"source_sha256\":{},\"current_git_blob\":{},\"head_git_blob\":{}}},\"harness_runner_blob\":{{\"relative_path\":{},\"source_sha256\":{},\"current_git_blob\":{},\"head_git_blob\":{}}},\"compiled_harness\":{{\"product_rlib_sha256\":{},\"orchestrator_rlib_sha256\":{},\"types_rlib_sha256\":{},\"rayon_rlib_sha256\":{},\"command_sha256\":{}}},\"input_identity\":{{\"generator\":\"gafime-core-production-executor-v1\",\"byte_order\":\"native\",\"fp32_matrix_sha256\":{},\"fp32_target_sha256\":{},\"fp64_matrix_sha256\":{},\"fp64_target_sha256\":{}}},\"provenance\":{{\"product_source_root\":{},\"harness_source_root\":{},\"product_rlib\":{},\"orchestrator_rlib\":{},\"types_rlib\":{},\"rayon_rlib\":{},\"benchmark_binary\":{},\"wheel\":{}}},\"device\":{{\"kind\":\"cpu\",\"identity\":{},\"logical_cpu_count\":{},\"physical_cpu_count\":{}}},\"process_affinity\":{},\"environment\":{},\"command_line\":{},\"clock\":\"std::time::Instant monotonic clock\",\"clock_and_power_capture_point\":\"before and after all timed benchmark regions\",\"clock_and_power_state\":{{\"before\":{},\"after\":{}}}}}",
        if claim_ready { "pass" } else { "informational" },
        claim_ready,
        json_string(profile.name()),
        json_string(metric.name()),
        json_string(&measurement_mode),
        json_string(policy.name()),
        json_string(measurement.spec.name),
        measurement.spec.rows,
        measurement.spec.features,
        measurement.spec.candidates,
        measurement.spec.top_k,
        measurement.spec.mi_bins,
        json_string(metric.name()),
        warmups,
        repetitions,
        measurement.loops,
        TARGET_REGION_NS,
        CALIBRATION_TARGET_REGION_NS,
        raw_minimum,
        raw_minimum >= TARGET_REGION_NS,
        normalized_json,
        raw_json,
        measurement.planner_protocol_ns,
        measurement.resident_matrix_ns,
        json_string(&worker_mode),
        json_string(if worker_mode == "default" {
            "primary_default_worker_production_result"
        } else {
            "thread_scaling_diagnostic"
        }),
        requested_workers
            .map(|workers| workers.to_string())
            .unwrap_or_else(|| "null".to_owned()),
        effective_workers,
        allowed_parallelism,
        json_string(&process_affinity),
        affinity_cardinality_json,
        affinity_matches_allowed,
        pool_start_worker_ids
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
        pool_start_worker_ids.len(),
        json_worker_cpu_ticks(&worker_cpu_ticks),
        json_string(worker_tick_status),
        worker_ticks_observable,
        every_worker_has_positive_work_ticks,
        measurement.final_digest.rows_written,
        measurement.final_digest.visible_score_bits,
        measurement.final_digest.candidate_digest,
        json_string(measurement.result_snapshot.result_dtype),
        measurement.result_snapshot.row_count,
        measurement.result_snapshot.max_arity,
        measurement.result_snapshot.metric_count,
        measurement.result_snapshot.result_flags,
        json_u32s(&measurement.result_snapshot.metric_ids),
        json_u32s(&measurement.result_snapshot.combo_indices),
        json_u32s(&measurement.result_snapshot.ranks),
        json_u32s(&measurement.result_snapshot.families),
        json_u64s(&measurement.result_snapshot.candidate_ids),
        json_u32s(&measurement.result_snapshot.row_flags),
        json_u64s(&measurement.result_snapshot.metric_value_bits),
        json_strings(&measurement.result_snapshot.metric_value_text),
        json_strings(&measurement.result_snapshot.metric_value_classes),
        json_string(&variant),
        ab_block,
        json_strings(&variant_sequence),
        schedule_index,
        schedule_seed,
        json_string(&schedule_sha256),
        json_strings(&profile_order),
        runner_pid,
        std::process::id(),
        json_string(&required_env("GAFIME_PRODUCTION_RUNNER_INVOCATION_ID")),
        json_string(&source_commit),
        json_string(&source_commit),
        json_string(&source_tree),
        json_string(&harness_commit),
        json_string(&harness_tree),
        json_string(COMPILED_HARNESS_SOURCE_RELATIVE_PATH.expect("compiled source path")),
        json_string(COMPILED_HARNESS_SOURCE_SHA256.expect("compiled source hash")),
        json_string(COMPILED_HARNESS_SOURCE_GIT_BLOB.expect("compiled source blob")),
        json_string(COMPILED_HARNESS_SOURCE_GIT_BLOB.expect("compiled source blob")),
        json_string(COMPILED_HARNESS_RUNNER_RELATIVE_PATH.expect("compiled runner path")),
        json_string(COMPILED_HARNESS_RUNNER_SHA256.expect("compiled runner hash")),
        json_string(COMPILED_HARNESS_RUNNER_GIT_BLOB.expect("compiled runner blob")),
        json_string(COMPILED_HARNESS_RUNNER_GIT_BLOB.expect("compiled runner blob")),
        json_string(COMPILED_PRODUCT_RLIB_SHA256.expect("compiled product rlib hash")),
        json_string(COMPILED_ORCHESTRATOR_RLIB_SHA256.expect("compiled orchestrator rlib hash")),
        json_string(COMPILED_TYPES_RLIB_SHA256.expect("compiled types rlib hash")),
        json_string(COMPILED_RAYON_RLIB_SHA256.expect("compiled rayon rlib hash")),
        json_string(COMPILED_COMMAND_SHA256.expect("compiled command hash")),
        json_string(&measurement.input.f32_feature_digest),
        json_string(&measurement.input.f32_target_digest),
        json_string(&measurement.input.f64_feature_digest),
        json_string(&measurement.input.f64_target_digest),
        json_string(&source_root),
        json_string(&harness_root),
        json_string(&product_rlib),
        json_string(&orchestrator_rlib),
        json_string(&types_rlib),
        json_string(&rayon_rlib),
        json_string(&binary),
        json_string(&wheel),
        json_string(&cpu_identity()),
        logical_cpu_count,
        physical_cpu_count
            .map(|count| count.to_string())
            .unwrap_or_else(|| "null".to_owned()),
        json_string(&process_affinity),
        json_environment(),
        command_line,
        clock_power_before,
        clock_power_after,
    );
    fs::write(&output, &report).expect("write production Core child artifact");
    println!("GAFIME_CORE_PRODUCTION_BENCH {report}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workload_matrix_has_parallel_candidate_headroom() {
        for workload in [Workload::Latency, Workload::Medium, Workload::Kernel] {
            let spec = workload.spec();
            assert!(spec.candidates >= 4);
            assert!(spec.candidates <= spec.features);
            assert!(spec.top_k > 0 && spec.top_k <= spec.candidates);
        }
    }

    #[test]
    fn precision_input_construction_preserves_fp64_without_f32_staging() {
        let spec = Workload::Latency.spec();
        let fp64 = build_inputs(InputPolicy::CommonF64, spec, Profile::Fp64);
        assert!(matches!(fp64.matrix, MatrixInput::F64 { .. }));
        let fp32 = build_inputs(InputPolicy::Native, spec, Profile::Fp32);
        assert!(matches!(fp32.matrix, MatrixInput::F32 { .. }));
    }

    #[test]
    fn production_scope_never_aliases_leaf_kernel_scope() {
        assert_ne!(
            "production_precision_compute_backend",
            "supplemental_single_core_leaf_kernel_diagnostic"
        );
    }
}
