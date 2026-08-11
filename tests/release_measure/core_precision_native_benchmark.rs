//! Supplemental single-core leaf-kernel benchmark for Core precision arithmetic.
//!
//! This file deliberately has no benchmark target of its own. The release
//! runner compiles this exact tracked blob once per product variant with
//! `rustc` and an explicit
//! `--extern gafime_cpu=/exact/product/libgafime_cpu-....rlib`. Cargo includes
//! it only to run the `cfg(test)` methodology adversaries; benchmark `main` is
//! never invoked there. Consequently a benchmark source present in a baseline
//! product checkout cannot affect the A/B harness.
//!
//! This benchmark intentionally calls individual metric kernels directly. It
//! neither constructs a planner/protocol nor invokes `PrecisionComputeBackend`,
//! candidate-level Rayon scheduling, result materialization, or ranking. Its
//! evidence is therefore a supplemental *single-core leaf-kernel diagnostic*,
//! never Core production-throughput evidence. The production executor harness
//! lives in `core_precision_production_benchmark.rs`.

use std::{
    env,
    fmt::Write as _,
    fs,
    hint::black_box,
    io::Write as _,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

#[cfg(not(test))]
use gafime_cpu::kernels::precision::{
    mutual_info_fixed_f32, mutual_info_fixed_f64, mutual_info_fixed_mixed, pearson_f32,
    pearson_f64, pearson_mixed, spearman_f32, spearman_f64, spearman_mixed,
};

#[cfg(test)]
mod test_kernel_stubs {
    pub fn pearson_f32(_: &[f32], _: &[f32]) -> f32 {
        0.0
    }
    pub fn pearson_mixed(_: &[f32], _: &[f32]) -> f64 {
        0.0
    }
    pub fn pearson_f64(_: &[f64], _: &[f64]) -> f64 {
        0.0
    }
    pub fn spearman_f32(_: &[f32], _: &[f32]) -> f32 {
        0.0
    }
    pub fn spearman_mixed(_: &[f32], _: &[f32]) -> f64 {
        0.0
    }
    pub fn spearman_f64(_: &[f64], _: &[f64]) -> f64 {
        0.0
    }
    pub fn mutual_info_fixed_f32(_: &[f32], _: &[f32], _: usize) -> f32 {
        0.0
    }
    pub fn mutual_info_fixed_mixed(_: &[f32], _: &[f32], _: usize) -> f64 {
        0.0
    }
    pub fn mutual_info_fixed_f64(_: &[f64], _: &[f64], _: usize) -> f64 {
        0.0
    }
}

#[cfg(test)]
use test_kernel_stubs::{
    mutual_info_fixed_f32, mutual_info_fixed_f64, mutual_info_fixed_mixed, pearson_f32,
    pearson_f64, pearson_mixed, spearman_f32, spearman_f64, spearman_mixed,
};

const ROWS: usize = 131_072;
const WARMUPS: usize = 10;
const PER_SAMPLE_UNTIMED_SAME_CELL_PRECONDITIONS: usize = 10;
const PER_SAMPLE_UNTIMED_PRECONDITION_MIN_NS: u128 = 100_000_000;
const BALANCED_SCHEDULE_CYCLES: usize = 20;
const PROFILE_ORDER_COUNT: usize = 6;
const METRIC_ROTATION_COUNT: usize = 4;
const BLOCKS_PER_BALANCED_CYCLE: usize = PROFILE_ORDER_COUNT * METRIC_ROTATION_COUNT;
const REPETITIONS: usize = BALANCED_SCHEDULE_CYCLES * BLOCKS_PER_BALANCED_CYCLE;
const TARGET_REGION_NS: u128 = 100_000_000;
const CALIBRATION_TARGET_REGION_NS: u128 = 200_000_000;
const MAX_LOOP_COUNT: usize = 4_096;
const BOOTSTRAP_RESAMPLES: usize = 2_000;
const ORDER_BOOTSTRAP_RESAMPLES: usize = 200_000;
const ORDER_CONTAMINATION_LIMIT_PERCENT: f64 = 1.0;
const ORDER_FAMILYWISE_ALPHA: f64 = 0.05;
const ORDER_MULTIPLE_COMPARISON_CELLS: usize = 12;
const ORDER_POSITION_PAIR_CONTRASTS: usize = 3;
const ORDER_MULTIPLE_COMPARISON_TESTS: usize =
    ORDER_MULTIPLE_COMPARISON_CELLS * ORDER_POSITION_PAIR_CONTRASTS;
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
const COMPILED_COMMAND_SHA256: Option<&str> = option_env!("GAFIME_COMPILED_COMMAND_SHA256");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Profile {
    Fp32,
    Mixed,
    Fp64,
}

impl Profile {
    const ALL: [Self; 3] = [Self::Fp32, Self::Mixed, Self::Fp64];

    const fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Mixed => "mixed",
            Self::Fp64 => "fp64",
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
    const ALL: [Self; 4] = [Self::Pearson, Self::Spearman, Self::MutualInfo, Self::R2];

    const fn name(self) -> &'static str {
        match self {
            Self::Pearson => "pearson",
            Self::Spearman => "spearman",
            Self::MutualInfo => "mutual_info",
            Self::R2 => "r2",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputPolicy {
    CommonF64,
    Native,
}

impl InputPolicy {
    fn from_environment() -> Self {
        match required_env("GAFIME_NATIVE_INPUT_POLICY").as_str() {
            "common-f64" => Self::CommonF64,
            "native" => Self::Native,
            other => panic!("unsupported GAFIME_NATIVE_INPUT_POLICY: {other}"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::CommonF64 => "common-f64",
            Self::Native => "native",
        }
    }
}

struct Inputs {
    x_f32: Vec<f32>,
    y_f32: Vec<f32>,
    x_f64: Vec<f64>,
    y_f64: Vec<f64>,
}

impl Inputs {
    fn new(policy: InputPolicy) -> Self {
        let mut x_f32 = Vec::with_capacity(ROWS);
        let mut y_f32 = Vec::with_capacity(ROWS);
        let mut x_f64 = Vec::with_capacity(ROWS);
        let mut y_f64 = Vec::with_capacity(ROWS);
        for row in 0..ROWS {
            let row_f64 = row as f64;
            let x64 = ((row * 17 + row / 29) % 65_521) as f64 / 65_521.0
                + (row_f64 * 0.000_031_7).sin() * 1.0e-9
                + ((row * 11 + 3) % 97) as f64 * 1.0e-12;
            let y64 = (0.71f64 * x64
                + ((row * 13) % 997) as f64 / 2_991.0
                + (row_f64 * 0.000_017_3).cos() * 1.0e-8)
                .sin();
            let (x32, y32) = match policy {
                InputPolicy::CommonF64 => (x64 as f32, y64 as f32),
                InputPolicy::Native => {
                    let row_f32 = row as f32;
                    let native_x = ((row * 17 + row / 29) % 65_521) as f32 / 65_521.0f32
                        + (row_f32 * 0.000_031_7f32).sin() * 1.0e-9f32
                        + ((row * 11 + 3) % 97) as f32 * 1.0e-12f32;
                    let native_y = (0.71f32 * native_x
                        + ((row * 13) % 997) as f32 / 2_991.0f32
                        + (row_f32 * 0.000_017_3f32).cos() * 1.0e-8f32)
                        .sin();
                    (native_x, native_y)
                }
            };
            x_f32.push(x32);
            y_f32.push(y32);
            x_f64.push(x64);
            y_f64.push(y64);
        }
        assert!(
            x_f64
                .iter()
                .zip(&x_f32)
                .any(|(&wide, &narrow)| wide != f64::from(narrow)),
            "f64 benchmark input must not be a f32 widening"
        );
        Self {
            x_f32,
            y_f32,
            x_f64,
            y_f64,
        }
    }
}

fn evaluate(profile: Profile, metric: Metric, data: &Inputs) -> f64 {
    match (profile, metric) {
        (Profile::Fp32, Metric::Pearson) => f64::from(pearson_f32(&data.x_f32, &data.y_f32)),
        (Profile::Fp32, Metric::Spearman) => f64::from(spearman_f32(&data.x_f32, &data.y_f32)),
        (Profile::Fp32, Metric::MutualInfo) => {
            f64::from(mutual_info_fixed_f32(&data.x_f32, &data.y_f32, 32))
        }
        (Profile::Fp32, Metric::R2) => {
            let value = pearson_f32(&data.x_f32, &data.y_f32);
            f64::from(value * value)
        }
        (Profile::Mixed, Metric::Pearson) => pearson_mixed(&data.x_f32, &data.y_f32),
        (Profile::Mixed, Metric::Spearman) => spearman_mixed(&data.x_f32, &data.y_f32),
        (Profile::Mixed, Metric::MutualInfo) => {
            mutual_info_fixed_mixed(&data.x_f32, &data.y_f32, 32)
        }
        (Profile::Mixed, Metric::R2) => {
            let value = pearson_mixed(&data.x_f32, &data.y_f32);
            value * value
        }
        (Profile::Fp64, Metric::Pearson) => pearson_f64(&data.x_f64, &data.y_f64),
        (Profile::Fp64, Metric::Spearman) => spearman_f64(&data.x_f64, &data.y_f64),
        (Profile::Fp64, Metric::MutualInfo) => mutual_info_fixed_f64(&data.x_f64, &data.y_f64, 32),
        (Profile::Fp64, Metric::R2) => {
            let value = pearson_f64(&data.x_f64, &data.y_f64);
            value * value
        }
    }
}

fn timed_region(profile: Profile, metric: Metric, data: &Inputs, loops: usize) -> u128 {
    let start = Instant::now();
    for _ in 0..loops {
        black_box(evaluate(profile, metric, black_box(data)));
    }
    start.elapsed().as_nanos()
}

fn precondition_cell(profile: Profile, metric: Metric, data: &Inputs) -> (usize, u128) {
    let start = Instant::now();
    let mut iterations = 0usize;
    loop {
        black_box(evaluate(profile, metric, black_box(data)));
        iterations += 1;
        let elapsed_ns = start.elapsed().as_nanos();
        if iterations >= PER_SAMPLE_UNTIMED_SAME_CELL_PRECONDITIONS
            && elapsed_ns >= PER_SAMPLE_UNTIMED_PRECONDITION_MIN_NS
        {
            return (iterations, elapsed_ns);
        }
    }
}

fn calibrated_loop_count(profile: Profile, metric: Metric, data: &Inputs) -> usize {
    for _ in 0..WARMUPS {
        black_box(timed_region(profile, metric, data, 1));
    }
    let probe = median_u128(
        (0..5)
            .map(|_| timed_region(profile, metric, data, 1))
            .collect(),
    )
    .max(1);
    let loops = CALIBRATION_TARGET_REGION_NS.div_ceil(probe);
    usize::try_from(loops)
        .unwrap_or(MAX_LOOP_COUNT)
        .clamp(1, MAX_LOOP_COUNT)
}

fn median_u128(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_f64(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    assert!(!values.is_empty());
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let position = (ordered.len() - 1) as f64 * fraction;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        ordered[lower]
    } else {
        ordered[lower] * (upper as f64 - position) + ordered[upper] * (position - lower as f64)
    }
}

fn mad(values: &[f64], center: f64) -> f64 {
    median_f64(values.iter().map(|value| (value - center).abs()).collect())
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn bootstrap_ci(values: &[f64], seed: u64) -> [f64; 2] {
    let mut state = seed;
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let sample = (0..values.len())
            .map(|_| values[(next_u64(&mut state) as usize) % values.len()])
            .collect();
        medians.push(median_f64(sample));
    }
    [percentile(&medians, 0.025), percentile(&medians, 0.975)]
}

const fn profile_orders() -> [[Profile; 3]; PROFILE_ORDER_COUNT] {
    [
        [Profile::Fp32, Profile::Mixed, Profile::Fp64],
        [Profile::Fp32, Profile::Fp64, Profile::Mixed],
        [Profile::Mixed, Profile::Fp32, Profile::Fp64],
        [Profile::Mixed, Profile::Fp64, Profile::Fp32],
        [Profile::Fp64, Profile::Fp32, Profile::Mixed],
        [Profile::Fp64, Profile::Mixed, Profile::Fp32],
    ]
}

#[derive(Clone, Copy, Debug)]
struct MeasurementBlock {
    balanced_cycle: usize,
    order_index: usize,
    profile_order: [Profile; 3],
    metric_rotation: usize,
}

fn balanced_measurement_schedule(seed: u64) -> Vec<MeasurementBlock> {
    let mut state = seed;
    let mut measured = Vec::with_capacity(REPETITIONS);
    for balanced_cycle in 0..BALANCED_SCHEDULE_CYCLES {
        let mut cycle = Vec::with_capacity(BLOCKS_PER_BALANCED_CYCLE);
        for (order_index, profile_order) in profile_orders().into_iter().enumerate() {
            for metric_rotation in 0..METRIC_ROTATION_COUNT {
                cycle.push(MeasurementBlock {
                    balanced_cycle,
                    order_index,
                    profile_order,
                    metric_rotation,
                });
            }
        }
        for index in (1..cycle.len()).rev() {
            let swap = (next_u64(&mut state) as usize) % (index + 1);
            cycle.swap(index, swap);
        }
        measured.extend(cycle);
    }
    assert_eq!(measured.len(), REPETITIONS);
    measured
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

fn git_output(root: &Path, args: &[&str]) -> String {
    let mut command = Command::new("git");
    command.arg("-C").arg(root).args(args);
    command
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
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

fn benchmark_environment_json() -> String {
    const KEYS: [&str; 10] = [
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "SHELL",
        "TERM",
        "RUST_BACKTRACE",
    ];
    let mut entries = Vec::new();
    for key in KEYS {
        if let Ok(value) = env::var(key) {
            entries.push(format!("{}:{}", json_string(key), json_string(&value)));
        }
    }
    format!("{{{}}}", entries.join(","))
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn required_canonical_path(name: &str) -> PathBuf {
    PathBuf::from(required_env(name))
        .canonicalize()
        .unwrap_or_else(|error| panic!("{name} must name an existing path: {error}"))
}

fn repository_commit(root: &Path) -> String {
    let commit = git_output(root, &["rev-parse", "HEAD"]);
    assert_full_hex(&commit, 40, "repository commit");
    commit
}

fn repository_tree(root: &Path) -> String {
    let tree = git_output(root, &["rev-parse", "HEAD^{tree}"]);
    assert_full_hex(&tree, 40, "repository tree");
    tree
}

fn assert_full_hex(value: &str, length: usize, label: &str) {
    assert!(
        value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} must be a {length}-character hexadecimal identity"
    );
}

fn source_tree_is_clean(root: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain=v1", "--untracked-files=all"])
        .output()
        .is_ok_and(|output| output.status.success() && output.stdout.is_empty())
}

struct TrackedSourceIdentity {
    path: PathBuf,
    relative_path: String,
    git_blob: String,
    sha256: String,
}

fn tracked_source_identity(root: &Path, source: &Path) -> TrackedSourceIdentity {
    let relative = source
        .strip_prefix(root)
        .expect("benchmark source must be inside the harness source root");
    assert!(
        !relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir)),
        "benchmark source must be repository relative"
    );
    let relative_text = relative.to_string_lossy().into_owned();
    assert!(source.is_file(), "benchmark source must exist");
    assert!(
        !git_output(root, &["ls-files", "--error-unmatch", &relative_text]).is_empty(),
        "benchmark source must be tracked by the harness repository"
    );
    let worktree_blob = git_output(root, &["hash-object", &relative_text]);
    let head_spec = format!("HEAD:{relative_text}");
    let head_blob = git_output(root, &["rev-parse", &head_spec]);
    assert!(
        !worktree_blob.is_empty() && worktree_blob == head_blob,
        "benchmark source must exactly match the harness HEAD blob"
    );
    let digest = sha256_file(source);
    TrackedSourceIdentity {
        path: source.to_path_buf(),
        relative_path: relative_text,
        git_blob: head_blob,
        sha256: digest,
    }
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
        .expect("sha256sum or shasum -a 256 must be available")
}

fn sha256_file(path: &Path) -> String {
    sha256_bytes(&fs::read(path).expect("read file for SHA-256"))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn file_identity(path: &Path) -> String {
    let size = fs::metadata(path).expect("stat provenance file").len();
    format!(
        "{{\"path\":{},\"size_bytes\":{},\"sha256\":{}}}",
        json_string(&path.display().to_string()),
        size,
        json_string(&sha256_file(path))
    )
}

fn assert_expected_file_hash(path: &Path, environment_key: &str) {
    assert_eq!(
        sha256_file(path),
        required_env(environment_key).to_ascii_lowercase(),
        "{environment_key} does not match {}",
        path.display()
    );
}

fn observed_python_executable() -> PathBuf {
    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        let unix = Path::new(&virtual_env).join("bin/python");
        if unix.is_file() {
            return unix.canonicalize().expect("canonical Python path");
        }
        let windows = Path::new(&virtual_env).join("Scripts/python.exe");
        if windows.is_file() {
            return windows.canonicalize().expect("canonical Python path");
        }
    }
    let observed = command_output("sh", &["-c", "command -v python3 || command -v python"]);
    PathBuf::from(observed)
        .canonicalize()
        .expect("a Python executable is required for performance evidence")
}

fn cpu_identity() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        if let Some(identity) = cpuinfo.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim() == "model name").then(|| value.trim().to_owned())
        }) {
            return identity;
        }
    }
    let sysctl = command_output("sysctl", &["-n", "machdep.cpu.brand_string"]);
    if !sysctl.is_empty() {
        return sysctl;
    }
    format!("{}-{}", env::consts::ARCH, env::consts::OS)
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

fn process_affinity() -> String {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        if let Some(value) = status.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name == "Cpus_allowed_list").then(|| value.trim().to_owned())
        }) {
            return value;
        }
    }
    env::var("GAFIME_NATIVE_AFFINITY").unwrap_or_else(|_| "unobservable".to_owned())
}

struct Provenance {
    product_root: PathBuf,
    product_commit: String,
    product_tree: String,
    harness_root: PathBuf,
    harness_commit: String,
    harness_tree: String,
    harness_source: TrackedSourceIdentity,
    harness_runner: TrackedSourceIdentity,
    product_rlib: PathBuf,
    wheel: PathBuf,
    binary: PathBuf,
    python: PathBuf,
    rustc: String,
    linker: String,
    compiler_command_json: String,
}

fn provenance() -> Provenance {
    let product_root = required_canonical_path("GAFIME_NATIVE_PRODUCT_SOURCE_ROOT");
    let harness_root = required_canonical_path("GAFIME_NATIVE_HARNESS_SOURCE_ROOT");
    assert!(
        source_tree_is_clean(&product_root),
        "product source tree must be clean"
    );
    assert!(
        source_tree_is_clean(&harness_root),
        "benchmark harness source tree must be clean"
    );
    let product_commit = repository_commit(&product_root);
    let product_tree = repository_tree(&product_root);
    let harness_commit = repository_commit(&harness_root);
    let harness_tree = repository_tree(&harness_root);
    assert_eq!(
        product_commit,
        required_env("GAFIME_NATIVE_EXPECTED_PRODUCT_COMMIT"),
        "product HEAD changed after runner validation"
    );
    assert_eq!(
        product_tree,
        required_env("GAFIME_NATIVE_EXPECTED_PRODUCT_TREE"),
        "product tree changed after runner validation"
    );
    assert_eq!(
        harness_commit,
        required_env("GAFIME_NATIVE_EXPECTED_HARNESS_COMMIT"),
        "harness HEAD changed after runner validation"
    );
    assert_eq!(
        harness_tree,
        required_env("GAFIME_NATIVE_EXPECTED_HARNESS_TREE"),
        "harness tree changed after runner validation"
    );
    let source = required_canonical_path("GAFIME_NATIVE_HARNESS_SOURCE");
    let harness_source = tracked_source_identity(&harness_root, &source);
    let runner = required_canonical_path("GAFIME_NATIVE_HARNESS_RUNNER");
    let harness_runner = tracked_source_identity(&harness_root, &runner);
    assert_eq!(
        harness_source.sha256,
        required_env("GAFIME_NATIVE_HARNESS_SOURCE_SHA256"),
        "compiled harness source SHA-256 changed after runner validation"
    );
    assert_eq!(
        harness_source.git_blob,
        required_env("GAFIME_NATIVE_HARNESS_SOURCE_GIT_BLOB"),
        "compiled harness source Git blob changed after runner validation"
    );
    let product_rlib = required_canonical_path("GAFIME_NATIVE_PRODUCT_RLIB");
    let wheel = required_canonical_path("GAFIME_NATIVE_BENCH_WHEEL");
    let binary = env::current_exe()
        .expect("benchmark executable path")
        .canonicalize()
        .expect("canonical benchmark executable path");
    assert_expected_file_hash(&product_rlib, "GAFIME_NATIVE_PRODUCT_RLIB_SHA256");
    assert_expected_file_hash(&wheel, "GAFIME_NATIVE_BENCH_WHEEL_SHA256");
    assert_expected_file_hash(&binary, "GAFIME_NATIVE_BENCH_BINARY_SHA256");
    assert_eq!(
        Some(harness_source.sha256.as_str()),
        COMPILED_HARNESS_SOURCE_SHA256,
        "executable was not compiled from the declared harness source SHA-256"
    );
    assert_eq!(
        Some(harness_source.git_blob.as_str()),
        COMPILED_HARNESS_SOURCE_GIT_BLOB,
        "executable was not compiled from the declared harness Git blob"
    );
    assert_eq!(
        Some(harness_source.relative_path.as_str()),
        COMPILED_HARNESS_SOURCE_RELATIVE_PATH,
        "executable was not compiled from the declared harness relative path"
    );
    assert_eq!(
        harness_runner.sha256,
        required_env("GAFIME_NATIVE_HARNESS_RUNNER_SHA256"),
        "harness runner SHA-256 changed after compile"
    );
    assert_eq!(
        harness_runner.git_blob,
        required_env("GAFIME_NATIVE_HARNESS_RUNNER_GIT_BLOB"),
        "harness runner Git blob changed after compile"
    );
    assert_eq!(
        Some(harness_runner.sha256.as_str()),
        COMPILED_HARNESS_RUNNER_SHA256,
        "executable does not embed the declared harness runner SHA-256"
    );
    assert_eq!(
        Some(harness_runner.git_blob.as_str()),
        COMPILED_HARNESS_RUNNER_GIT_BLOB,
        "executable does not embed the declared harness runner Git blob"
    );
    assert_eq!(
        Some(harness_runner.relative_path.as_str()),
        COMPILED_HARNESS_RUNNER_RELATIVE_PATH,
        "executable does not embed the declared harness runner relative path"
    );
    let runtime_compiler_command = required_env("GAFIME_NATIVE_COMPILER_COMMAND_JSON");
    let runtime_compiler_command_sha256 = sha256_bytes(runtime_compiler_command.as_bytes());
    assert_eq!(
        COMPILED_COMMAND_SHA256,
        Some(runtime_compiler_command_sha256.as_str()),
        "runtime compiler command differs from the fixed-width identity embedded in the executable"
    );
    let product_rlib_sha256 = sha256_file(&product_rlib);
    assert_eq!(
        Some(product_rlib_sha256.as_str()),
        COMPILED_PRODUCT_RLIB_SHA256,
        "executable was not linked against the declared product rlib SHA-256"
    );
    let python = observed_python_executable();
    Provenance {
        product_root,
        product_commit,
        product_tree,
        harness_root,
        harness_commit,
        harness_tree,
        harness_source,
        harness_runner,
        product_rlib,
        wheel,
        binary,
        python,
        rustc: required_env("GAFIME_NATIVE_RUSTC_VERSION").replace('"', "'"),
        linker: required_env("GAFIME_NATIVE_LINKER_VERSION").replace('"', "'"),
        compiler_command_json: runtime_compiler_command,
    }
}

fn profile_index(profile: Profile) -> usize {
    match profile {
        Profile::Fp32 => 0,
        Profile::Mixed => 1,
        Profile::Fp64 => 2,
    }
}

#[derive(Clone)]
struct RawObservation {
    profile: Profile,
    metric: Metric,
    order_index: usize,
    block_index: usize,
    balanced_cycle: usize,
    metric_rotation: usize,
    position: usize,
    profile_order: [Profile; 3],
    precondition_iterations: usize,
    precondition_duration_ns: u128,
    duration_ns: u128,
}

fn json_order(order: &[Profile; 3]) -> String {
    format!(
        "[\"{}\",\"{}\",\"{}\"]",
        order[0].name(),
        order[1].name(),
        order[2].name()
    )
}

fn json_raw(values: &[u128]) -> String {
    values
        .iter()
        .map(u128::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn json_f64(values: &[f64]) -> String {
    values
        .iter()
        // `f64::to_string` uses the shortest representation that round-trips
        // to the same binary64 value.  A fixed number of digits after the
        // decimal point is not sufficient for very small values.
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn samples_for(raw: &[RawObservation], profile: Profile, metric: Metric) -> Vec<u128> {
    raw.iter()
        .filter(|item| item.profile == profile && item.metric == metric)
        .map(|item| item.duration_ns)
        .collect()
}

fn position_medians(
    raw: &[RawObservation],
    profile: Profile,
    metric: Metric,
    loops: usize,
) -> [f64; 3] {
    std::array::from_fn(|position| {
        median_f64(
            raw.iter()
                .filter(|item| {
                    item.profile == profile && item.metric == metric && item.position == position
                })
                .map(|item| item.duration_ns as f64 / loops as f64)
                .collect(),
        )
    })
}

fn order_spread_percent(position_medians: &[f64; 3], overall_median: f64) -> f64 {
    let minimum = position_medians
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .expect("position medians");
    let maximum = position_medians
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .expect("position medians");
    (maximum - minimum) / overall_median.max(f64::MIN_POSITIVE) * 100.0
}

const POSITION_CONTRASTS: [[usize; 2]; ORDER_POSITION_PAIR_CONTRASTS] = [[0, 1], [0, 2], [1, 2]];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrderClassification {
    Clean,
    Inconclusive,
    Contaminated,
}

impl OrderClassification {
    const fn name(self) -> &'static str {
        match self {
            Self::Clean => {
                "no_order_effect_above_one_percent_with_95_percent_familywise_confidence"
            }
            Self::Inconclusive => "inconclusive_order_effect_requires_rerun",
            Self::Contaminated => "confirmed_order_contamination_above_one_percent",
        }
    }
}

struct PositionContrastAssessment {
    positions: [usize; 2],
    observed_signed_percent: f64,
    corrected_bootstrap_ci_percent: [f64; 2],
    classification: OrderClassification,
}

struct OrderAssessment {
    position_medians: [f64; 3],
    observed_spread_percent: f64,
    contrasts: [PositionContrastAssessment; ORDER_POSITION_PAIR_CONTRASTS],
    corrected_per_contrast_confidence_level: f64,
    balanced_cycle_cluster_count: usize,
    observations_per_position: usize,
    classification: OrderClassification,
}

fn order_assessment(
    raw: &[RawObservation],
    profile: Profile,
    metric: Metric,
    loops: usize,
    seed: u64,
) -> OrderAssessment {
    let position_medians = position_medians(raw, profile, metric, loops);
    let comparable_samples = raw
        .iter()
        .filter(|item| item.profile == profile && item.metric == metric)
        .map(|item| item.duration_ns as f64 / loops as f64)
        .collect::<Vec<_>>();
    assert_eq!(comparable_samples.len(), REPETITIONS);
    let overall_median = median_f64(comparable_samples);
    let observed_spread_percent = order_spread_percent(&position_medians, overall_median);
    // Each balanced cycle contains the complete 6 profile-order x 4 metric-
    // rotation cross product. A given profile therefore contributes eight
    // observations at each position per cycle. Resampling whole 24-block
    // cycles preserves their order/rotation and temporal/thermal correlation;
    // rotation is never independently resampled or aliased with position.
    let mut balanced_cycle_clusters = Vec::<[Vec<f64>; 3]>::with_capacity(BALANCED_SCHEDULE_CYCLES);
    for balanced_cycle in 0..BALANCED_SCHEDULE_CYCLES {
        let position_samples = std::array::from_fn(|position| {
            raw.iter()
                .filter(|item| {
                    item.profile == profile
                        && item.metric == metric
                        && item.balanced_cycle == balanced_cycle
                        && item.position == position
                })
                .map(|item| item.duration_ns as f64 / loops as f64)
                .collect::<Vec<_>>()
        });
        assert!(position_samples
            .iter()
            .all(|values| values.len() == 2 * METRIC_ROTATION_COUNT));
        balanced_cycle_clusters.push(position_samples);
    }

    let observations_per_position = raw
        .iter()
        .filter(|item| item.profile == profile && item.metric == metric && item.position == 0)
        .count();
    assert_eq!(observations_per_position, REPETITIONS / 3);

    let mut bootstrap_state = seed;
    let mut bootstrap_effects: [Vec<f64>; ORDER_POSITION_PAIR_CONTRASTS] =
        std::array::from_fn(|_| Vec::with_capacity(ORDER_BOOTSTRAP_RESAMPLES));
    for _ in 0..ORDER_BOOTSTRAP_RESAMPLES {
        let mut position_samples: [Vec<f64>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(observations_per_position));
        for _ in 0..BALANCED_SCHEDULE_CYCLES {
            let sampled_cycle =
                (next_u64(&mut bootstrap_state) as usize) % BALANCED_SCHEDULE_CYCLES;
            let cluster = &balanced_cycle_clusters[sampled_cycle];
            for position in 0..3 {
                position_samples[position].extend(cluster[position].iter().copied());
            }
        }
        let bootstrap_medians: [f64; 3] =
            std::array::from_fn(|position| median_f64(position_samples[position].clone()));
        let bootstrap_center =
            median_f64(position_samples.into_iter().flatten().collect::<Vec<_>>());
        for (contrast_index, [left, right]) in POSITION_CONTRASTS.into_iter().enumerate() {
            bootstrap_effects[contrast_index].push(
                (bootstrap_medians[right] - bootstrap_medians[left])
                    / bootstrap_center.max(f64::MIN_POSITIVE)
                    * 100.0,
            );
        }
    }

    // Twelve profile x metric cells and all three position-pair contrasts are
    // inspected. A two-sided Bonferroni interval with alpha 0.05/(12*3)
    // accounts for selecting the observed fastest/slowest pair and keeps
    // family-wise coverage at least 95%.
    let per_contrast_alpha = ORDER_FAMILYWISE_ALPHA / ORDER_MULTIPLE_COMPARISON_TESTS as f64;
    let corrected_per_contrast_confidence_level = 1.0 - per_contrast_alpha;
    let contrasts = std::array::from_fn(|contrast_index| {
        let [left, right] = POSITION_CONTRASTS[contrast_index];
        let corrected_bootstrap_ci_percent = [
            percentile(&bootstrap_effects[contrast_index], per_contrast_alpha * 0.5),
            percentile(
                &bootstrap_effects[contrast_index],
                1.0 - per_contrast_alpha * 0.5,
            ),
        ];
        let classification = if corrected_bootstrap_ci_percent[0]
            > ORDER_CONTAMINATION_LIMIT_PERCENT
            || corrected_bootstrap_ci_percent[1] < -ORDER_CONTAMINATION_LIMIT_PERCENT
        {
            OrderClassification::Contaminated
        } else if corrected_bootstrap_ci_percent[0] >= -ORDER_CONTAMINATION_LIMIT_PERCENT
            && corrected_bootstrap_ci_percent[1] <= ORDER_CONTAMINATION_LIMIT_PERCENT
        {
            OrderClassification::Clean
        } else {
            OrderClassification::Inconclusive
        };
        PositionContrastAssessment {
            positions: [left, right],
            observed_signed_percent: (position_medians[right] - position_medians[left])
                / overall_median.max(f64::MIN_POSITIVE)
                * 100.0,
            corrected_bootstrap_ci_percent,
            classification,
        }
    });
    let classification = if contrasts
        .iter()
        .any(|contrast| contrast.classification == OrderClassification::Contaminated)
    {
        OrderClassification::Contaminated
    } else if contrasts
        .iter()
        .all(|contrast| contrast.classification == OrderClassification::Clean)
    {
        OrderClassification::Clean
    } else {
        OrderClassification::Inconclusive
    };
    OrderAssessment {
        position_medians,
        observed_spread_percent,
        contrasts,
        corrected_per_contrast_confidence_level,
        balanced_cycle_cluster_count: balanced_cycle_clusters.len(),
        observations_per_position,
        classification,
    }
}

fn main() {
    let benchmark_provenance = provenance();
    let policy = InputPolicy::from_environment();
    let seed = required_env("GAFIME_NATIVE_BENCH_SEED")
        .parse::<u64>()
        .expect("GAFIME_NATIVE_BENCH_SEED must be an unsigned integer");
    let governor_before = cpu_governors();
    let clock_power_before = cpu_clock_power_snapshot(&governor_before);
    let data = Inputs::new(policy);
    let measured_schedule = balanced_measurement_schedule(seed);
    let mut loop_counts = [[1usize; 3]; 4];
    let mut observations = Vec::with_capacity(REPETITIONS * Metric::ALL.len() * 3);

    for (metric_index, metric) in Metric::ALL.into_iter().enumerate() {
        for profile in Profile::ALL {
            let loops = calibrated_loop_count(profile, metric, &data);
            loop_counts[metric_index][profile_index(profile)] = loops;
            for _ in 0..WARMUPS {
                black_box(timed_region(profile, metric, &data, loops));
            }
        }
    }

    for (block_index, block) in measured_schedule.iter().copied().enumerate() {
        let order = block.profile_order;
        let metric_rotation = block.metric_rotation;
        for metric_offset in 0..Metric::ALL.len() {
            let metric_index = (metric_rotation + metric_offset) % Metric::ALL.len();
            let metric = Metric::ALL[metric_index];
            for (position, profile) in order.into_iter().enumerate() {
                // Normalize code, input-cache, allocator, CPU-frequency, and
                // thermal state immediately before every measured cell. A
                // fixed call count is not a fixed stabilization window: ten
                // MI calls can finish in a few milliseconds while ten
                // Spearman calls occupy tens of milliseconds. Require both
                // the public warmup floor and a common minimum elapsed region,
                // and retain the actual untimed work in the raw evidence.
                let (precondition_iterations, precondition_duration_ns) =
                    precondition_cell(profile, metric, &data);
                let duration_ns = timed_region(
                    profile,
                    metric,
                    &data,
                    loop_counts[metric_index][profile_index(profile)],
                );
                observations.push(RawObservation {
                    profile,
                    metric,
                    order_index: block.order_index,
                    block_index,
                    balanced_cycle: block.balanced_cycle,
                    metric_rotation,
                    position,
                    profile_order: order,
                    precondition_iterations,
                    precondition_duration_ns,
                    duration_ns,
                });
            }
        }
    }

    let mut records = String::new();
    let mut sensitivity_cells = String::new();
    let mut maximum_order_spread_percent = 0.0f64;
    let mut confirmed_order_contamination_cells = 0usize;
    let mut inconclusive_order_sensitivity_cells = 0usize;
    let mut under_target_region_cells = 0usize;
    let mut record_count = 0usize;
    for profile in Profile::ALL {
        for (metric_index, metric) in Metric::ALL.into_iter().enumerate() {
            let samples = samples_for(&observations, profile, metric);
            assert_eq!(samples.len(), REPETITIONS);
            let loops = loop_counts[metric_index][profile_index(profile)];
            let per_call = samples
                .iter()
                .map(|value| *value as f64 / loops as f64)
                .collect::<Vec<_>>();
            let center = median_f64(per_call.clone());
            let ci = bootstrap_ci(
                &per_call,
                seed ^ (metric_index as u64) ^ (profile_index(profile) as u64),
            );
            let assessment = order_assessment(
                &observations,
                profile,
                metric,
                loops,
                seed ^ ((metric_index as u64) << 32) ^ profile_index(profile) as u64,
            );
            let positions = assessment.position_medians;
            let spread = assessment.observed_spread_percent;
            maximum_order_spread_percent = maximum_order_spread_percent.max(spread);
            match assessment.classification {
                OrderClassification::Clean => {}
                OrderClassification::Inconclusive => {
                    inconclusive_order_sensitivity_cells += 1;
                }
                OrderClassification::Contaminated => {
                    confirmed_order_contamination_cells += 1;
                }
            }
            let sensitivity_status = assessment.classification.name();
            if record_count != 0 {
                records.push(',');
                sensitivity_cells.push(',');
            }
            let raw_minimum = samples.iter().copied().min().unwrap_or(0);
            if raw_minimum < TARGET_REGION_NS {
                under_target_region_cells += 1;
            }
            let contrast_json = assessment
                .contrasts
                .iter()
                .map(|contrast| {
                    format!(
                        "{{\"positions\":[{},{}],\"observed_signed_percent\":{},\"corrected_bootstrap_ci_percent\":[{},{}],\"status\":\"{}\"}}",
                        contrast.positions[0],
                        contrast.positions[1],
                        contrast.observed_signed_percent,
                        contrast.corrected_bootstrap_ci_percent[0],
                        contrast.corrected_bootstrap_ci_percent[1],
                        contrast.classification.name(),
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            records.push_str(&format!(
                "{{\"profile\":\"{}\",\"operation\":\"metric_kernel\",\"metric\":\"{}\",\"samples_ns\":[{}],\"raw_samples_ns\":[{}],\"median_ns_per_call\":{},\"mad_ns_per_call\":{},\"p05_ns_per_call\":{},\"p95_ns_per_call\":{},\"bootstrap_median_95_ci_ns_per_call\":[{},{}],\"loop_count_per_sample\":{},\"sample_region_target_ns\":{},\"sample_region_min_observed_ns\":{},\"sample_region_target_met\":{},\"order_position_median_ns\":[{},{},{}],\"max_order_position_spread_percent\":{},\"position_contrasts\":[{}],\"corrected_per_contrast_confidence_level\":{},\"order_balanced_cycle_cluster_count\":{},\"order_observations_per_position\":{},\"order_sensitivity_status\":\"{}\"}}",
                profile.name(),
                metric.name(),
                json_f64(&per_call),
                json_raw(&samples),
                center,
                mad(&per_call, center),
                percentile(&per_call, 0.05),
                percentile(&per_call, 0.95),
                ci[0],
                ci[1],
                loops,
                TARGET_REGION_NS,
                raw_minimum,
                raw_minimum >= TARGET_REGION_NS,
                positions[0],
                positions[1],
                positions[2],
                spread,
                contrast_json,
                assessment.corrected_per_contrast_confidence_level,
                assessment.balanced_cycle_cluster_count,
                assessment.observations_per_position,
                sensitivity_status,
            ));
            sensitivity_cells.push_str(&format!(
                "{{\"profile\":\"{}\",\"metric\":\"{}\",\"order_position_median_ns\":[{},{},{}],\"max_order_position_spread_percent\":{},\"position_contrasts\":[{}],\"corrected_per_contrast_confidence_level\":{},\"balanced_cycle_cluster_count\":{},\"observations_per_position\":{},\"status\":\"{}\"}}",
                profile.name(),
                metric.name(),
                positions[0],
                positions[1],
                positions[2],
                spread,
                contrast_json,
                assessment.corrected_per_contrast_confidence_level,
                assessment.balanced_cycle_cluster_count,
                assessment.observations_per_position,
                sensitivity_status,
            ));
            record_count += 1;
        }
    }

    let order_gate_passed =
        confirmed_order_contamination_cells == 0 && inconclusive_order_sensitivity_cells == 0;
    let sample_region_gate_passed = under_target_region_cells == 0;
    let status = if order_gate_passed && sample_region_gate_passed {
        "pass"
    } else {
        "investigate"
    };
    let order_sensitivity_status = if confirmed_order_contamination_cells != 0 {
        OrderClassification::Contaminated.name()
    } else if inconclusive_order_sensitivity_cells != 0 {
        OrderClassification::Inconclusive.name()
    } else {
        OrderClassification::Clean.name()
    };
    let sample_region_status = if sample_region_gate_passed {
        "all_raw_regions_meet_minimum"
    } else {
        "under_target_raw_region_requires_recalibration"
    };
    let raw_order = observations
        .iter()
        .map(|item| {
            format!(
                "{{\"profile\":\"{}\",\"metric\":\"{}\",\"block_index\":{},\"balanced_cycle\":{},\"order_index\":{},\"metric_rotation\":{},\"position\":{},\"profile_order\":{},\"precondition_iterations\":{},\"precondition_duration_ns\":{},\"duration_ns\":{}}}",
                item.profile.name(),
                item.metric.name(),
                item.block_index,
                item.balanced_cycle,
                item.order_index,
                item.metric_rotation,
                item.position,
                json_order(&item.profile_order),
                item.precondition_iterations,
                item.precondition_duration_ns,
                item.duration_ns,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let measured_order_json = measured_schedule
        .iter()
        .map(|block| json_order(&block.profile_order))
        .collect::<Vec<_>>()
        .join(",");
    let measured_metric_rotation_json = measured_schedule
        .iter()
        .map(|block| block.metric_rotation.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let measured_schedule_json = measured_schedule
        .iter()
        .enumerate()
        .map(|(block_index, block)| {
            format!(
                "{{\"block_index\":{},\"balanced_cycle\":{},\"order_index\":{},\"profile_order\":{},\"metric_rotation\":{}}}",
                block_index,
                block.balanced_cycle,
                block.order_index,
                json_order(&block.profile_order),
                block.metric_rotation,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let governor_after = cpu_governors();
    let clock_power_after = cpu_clock_power_snapshot(&governor_after);
    let input_identity = format!(
        "{{\"generator\":\"gafime-core-leaf-kernel-diagnostic-v1\",\"source_policy\":{},\"native_byte_order\":{},\"rows\":{},\"mi_bins\":32,\"fp32_matrix_sha256\":{},\"fp32_target_sha256\":{},\"fp64_matrix_sha256\":{},\"fp64_target_sha256\":{}}}",
        json_string(policy.name()),
        json_string(if cfg!(target_endian = "little") { "little" } else { "big" }),
        ROWS,
        json_string(&sha256_bytes(&f32_bytes(&data.x_f32))),
        json_string(&sha256_bytes(&f32_bytes(&data.y_f32))),
        json_string(&sha256_bytes(&f64_bytes(&data.x_f64))),
        json_string(&sha256_bytes(&f64_bytes(&data.y_f64))),
    );
    let environment = benchmark_environment_json();
    let affinity = process_affinity();
    let command_line = json_strings(&env::args().collect::<Vec<_>>());
    let cpu = cpu_identity();
    let target = format!("{}-{}", env::consts::ARCH, env::consts::OS);
    let source_identity = file_identity(&benchmark_provenance.harness_source.path);
    let runner_identity = file_identity(&benchmark_provenance.harness_runner.path);
    let report = format!(
        "{{\"schema\":\"gafime.core-leaf-kernel-diagnostic.v1\",\"status\":\"{}\",\"backend\":\"core\",\"profiles\":[\"fp32\",\"mixed\",\"fp64\"],\"source_commit\":{},\"product_source_commit\":{},\"product_source_tree\":{},\"product_source_tree_state\":{{\"status\":\"clean\"}},\"harness_source_commit\":{},\"harness_source_tree\":{},\"harness_source_tree_state\":{{\"status\":\"clean\"}},\"harness_source_blob\":{{\"relative_path\":{},\"source_sha256\":{},\"current_git_blob\":{},\"head_git_blob\":{}}},\"harness_runner_blob\":{{\"relative_path\":{},\"source_sha256\":{},\"current_git_blob\":{},\"head_git_blob\":{}}},\"compiled_harness_source\":{{\"relative_path\":{},\"source_sha256\":{},\"git_blob\":{},\"product_rlib_sha256\":{}}},\"source_tree_state\":{{\"status\":\"clean\"}},\"input_policy\":{},\"input_identity\":{},\"workload\":{{\"name\":\"single-core-leaf-metric-kernel\",\"rows\":{},\"features\":1,\"candidates\":1,\"arity\":1,\"mi_bins\":32,\"input_bytes_fp32\":{},\"input_bytes_fp64\":{}}},\"rows\":{},\"warmups\":{},\"repeats\":{},\"order_seed\":{},\"profile_orders\":[[\"fp32\",\"mixed\",\"fp64\"],[\"fp32\",\"fp64\",\"mixed\"],[\"mixed\",\"fp32\",\"fp64\"],[\"mixed\",\"fp64\",\"fp32\"],[\"fp64\",\"fp32\",\"mixed\"],[\"fp64\",\"mixed\",\"fp32\"]],\"metric_rotations\":[0,1,2,3],\"balanced_schedule_cycles\":{},\"profile_order_metric_rotation_pair_repetitions\":{},\"measured_profile_orders\":[{}],\"measured_metric_rotations\":[{}],\"measured_schedule\":[{}],\"all_six_profile_orders_covered\":true,\"all_profile_order_metric_rotation_pairs_covered\":true,\"order_sensitivity\":{{\"threshold_percent\":{},\"maximum_spread_percent\":{},\"confirmed_contamination_cells\":{},\"inconclusive_cells\":{},\"bootstrap_resamples\":{},\"familywise_confidence_level\":{},\"multiple_comparison_correction\":\"bonferroni_two_sided_across_profile_metric_cells_and_position_pair_contrasts\",\"comparison_cells\":{},\"position_pair_contrasts_per_cell\":{},\"total_comparisons\":{},\"corrected_per_contrast_confidence_level\":{},\"bootstrap_stratification\":\"whole_balanced_cycle_cluster\",\"decision_rule\":\"contaminated when any simultaneous contrast interval lies wholly beyond plus or minus one percent; clean only when all three simultaneous contrast intervals lie wholly inside plus or minus one percent; otherwise inconclusive\",\"status\":{},\"cells\":[{}]}},\"target_region_ns\":{},\"calibration_target_region_ns\":{},\"calibration_policy\":\"fixed_loop_count_per_cell_no_per_sample_rescaling\",\"sample_region_gate\":{{\"minimum_required_ns\":{},\"under_target_cells\":{},\"status\":{}}},\"measurement_scope\":\"supplemental_single_core_leaf_kernel_diagnostic\",\"claim_boundary\":{{\"eligible_for_core_production_throughput\":false,\"reason\":\"direct metric-kernel calls do not exercise planner/protocol, PrecisionComputeBackend, candidate-level Rayon scheduling, result materialization, or ranking\"}},\"execution_topology\":{{\"candidate_parallelism\":\"not_exercised\",\"rayon_worker_count\":null,\"direct_leaf_kernel_calls\":true}},\"preemption_observation\":{{\"status\":\"not_used_for_sample_filtering\",\"reason\":\"the standalone cross-platform Rust harness has no portable reliable per-region involuntary-context-switch counter; fixed >=100 ms wall-clock regions and corrected whole-cycle cluster confidence intervals retain scheduler effects without selective sample deletion\"}},\"decomposition_boundaries\":{{\"ingest_conversion\":\"not measured; input vectors are prepared before timing\",\"candidate_materialization\":\"not present in metric timer; unary numeric vectors materialized before timing\",\"report_construction\":\"not measured by this native arithmetic benchmark\"}},\"compiler\":{{\"rustc\":{},\"linker\":{},\"target\":{},\"rustc_flags\":[\"--edition=2021\",\"-Copt-level=3\",\"-Ccodegen-units=1\",\"-Clto=fat\",\"-Cembed-bitcode=yes\"],\"command_argv\":{}}},\"device\":{{\"kind\":\"cpu\",\"identity\":{}}},\"process_affinity\":{},\"command_line\":{},\"clock\":\"std::time::Instant monotonic clock\",\"clock_and_power_state\":{{\"before\":{},\"after\":{}}},\"environment\":{},\"provenance\":{{\"source_root\":{},\"source_tree_state\":{{\"status\":\"clean\"}},\"product_source_root\":{},\"product_source_commit\":{},\"product_source_tree\":{},\"product_source_tree_state\":{{\"status\":\"clean\"}},\"harness_source_root\":{},\"harness_source_commit\":{},\"harness_source_tree\":{},\"harness_source_tree_state\":{{\"status\":\"clean\"}},\"harness_source\":{},\"harness_runner\":{},\"product_rlib\":{},\"benchmark_source\":{},\"benchmark_binary\":{},\"wheel\":{},\"python_executable\":{}}},\"records\":[{}],\"raw_order\":[{}]}}",
        status,
        json_string(&benchmark_provenance.product_commit),
        json_string(&benchmark_provenance.product_commit),
        json_string(&benchmark_provenance.product_tree),
        json_string(&benchmark_provenance.harness_commit),
        json_string(&benchmark_provenance.harness_tree),
        json_string(&benchmark_provenance.harness_source.relative_path),
        json_string(&benchmark_provenance.harness_source.sha256),
        json_string(&benchmark_provenance.harness_source.git_blob),
        json_string(&benchmark_provenance.harness_source.git_blob),
        json_string(&benchmark_provenance.harness_runner.relative_path),
        json_string(&benchmark_provenance.harness_runner.sha256),
        json_string(&benchmark_provenance.harness_runner.git_blob),
        json_string(&benchmark_provenance.harness_runner.git_blob),
        json_string(&benchmark_provenance.harness_source.relative_path),
        json_string(&benchmark_provenance.harness_source.sha256),
        json_string(&benchmark_provenance.harness_source.git_blob),
        json_string(
            COMPILED_PRODUCT_RLIB_SHA256.expect("compiled product rlib identity was validated")
        ),
        json_string(policy.name()),
        input_identity,
        ROWS,
        ROWS * 2 * std::mem::size_of::<f32>(),
        ROWS * 2 * std::mem::size_of::<f64>(),
        ROWS,
        WARMUPS,
        REPETITIONS,
        seed,
        BALANCED_SCHEDULE_CYCLES,
        BALANCED_SCHEDULE_CYCLES,
        measured_order_json,
        measured_metric_rotation_json,
        measured_schedule_json,
        ORDER_CONTAMINATION_LIMIT_PERCENT,
        maximum_order_spread_percent,
        confirmed_order_contamination_cells,
        inconclusive_order_sensitivity_cells,
        ORDER_BOOTSTRAP_RESAMPLES,
        1.0 - ORDER_FAMILYWISE_ALPHA,
        ORDER_MULTIPLE_COMPARISON_CELLS,
        ORDER_POSITION_PAIR_CONTRASTS,
        ORDER_MULTIPLE_COMPARISON_TESTS,
        1.0 - ORDER_FAMILYWISE_ALPHA / ORDER_MULTIPLE_COMPARISON_TESTS as f64,
        json_string(order_sensitivity_status),
        sensitivity_cells,
        TARGET_REGION_NS,
        CALIBRATION_TARGET_REGION_NS,
        TARGET_REGION_NS,
        under_target_region_cells,
        json_string(sample_region_status),
        json_string(&benchmark_provenance.rustc),
        json_string(&benchmark_provenance.linker),
        json_string(&target),
        benchmark_provenance.compiler_command_json,
        json_string(&cpu),
        json_string(&affinity),
        command_line,
        clock_power_before,
        clock_power_after,
        environment,
        json_string(&benchmark_provenance.product_root.display().to_string()),
        json_string(&benchmark_provenance.product_root.display().to_string()),
        json_string(&benchmark_provenance.product_commit),
        json_string(&benchmark_provenance.product_tree),
        json_string(&benchmark_provenance.harness_root.display().to_string()),
        json_string(&benchmark_provenance.harness_commit),
        json_string(&benchmark_provenance.harness_tree),
        source_identity,
        runner_identity,
        file_identity(&benchmark_provenance.product_rlib),
        file_identity(&benchmark_provenance.harness_source.path),
        file_identity(&benchmark_provenance.binary),
        file_identity(&benchmark_provenance.wheel),
        file_identity(&benchmark_provenance.python),
        records,
        raw_order,
    );
    let report = report.replacen(
        "\"repeats\":",
        &format!(
            "\"per_sample_untimed_same_cell_preconditions\":{},\"per_sample_untimed_precondition_min_ns\":{},\"repeats\":",
            PER_SAMPLE_UNTIMED_SAME_CELL_PRECONDITIONS,
            PER_SAMPLE_UNTIMED_PRECONDITION_MIN_NS
        ),
        1,
    );
    let output = PathBuf::from(required_env("GAFIME_NATIVE_BENCH_OUTPUT"));
    fs::write(&output, &report).expect("write native benchmark artifact");
    println!("GAFIME_NATIVE_BENCH {report}");
    if !order_gate_passed || !sample_region_gate_passed {
        eprintln!(
            "Core native benchmark is not claim-ready: confirmed order cells={}, inconclusive order cells={}, under-target raw-region cells={}, observed max spread={:.4}%, threshold={:.4}%",
            confirmed_order_contamination_cells,
            inconclusive_order_sensitivity_cells,
            under_target_region_cells,
            maximum_order_spread_percent,
            ORDER_CONTAMINATION_LIMIT_PERCENT
        );
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_observations(
        mut duration: impl FnMut(usize, &MeasurementBlock, usize) -> u128,
    ) -> Vec<RawObservation> {
        balanced_measurement_schedule(0x51a7_2026_0810)
            .iter()
            .copied()
            .enumerate()
            .map(|(block_index, block)| {
                let profile = Profile::Fp32;
                let position = block
                    .profile_order
                    .iter()
                    .position(|candidate| *candidate == profile)
                    .expect("fp32 must be in every profile order");
                RawObservation {
                    profile,
                    metric: Metric::Pearson,
                    order_index: block.order_index,
                    block_index,
                    balanced_cycle: block.balanced_cycle,
                    metric_rotation: block.metric_rotation,
                    position,
                    profile_order: block.profile_order,
                    precondition_iterations: WARMUPS,
                    precondition_duration_ns: PER_SAMPLE_UNTIMED_PRECONDITION_MIN_NS,
                    duration_ns: duration(block_index, &block, position),
                }
            })
            .collect()
    }

    #[test]
    fn schedule_crosses_every_profile_order_and_metric_rotation_in_twenty_cycles() {
        let schedule = balanced_measurement_schedule(0x51a7_2026_0810);
        assert_eq!(schedule.len(), 480);
        let mut totals = [[0usize; METRIC_ROTATION_COUNT]; PROFILE_ORDER_COUNT];
        let mut by_cycle =
            [[[0usize; METRIC_ROTATION_COUNT]; PROFILE_ORDER_COUNT]; BALANCED_SCHEDULE_CYCLES];
        for block in schedule {
            totals[block.order_index][block.metric_rotation] += 1;
            by_cycle[block.balanced_cycle][block.order_index][block.metric_rotation] += 1;
        }
        assert!(totals
            .into_iter()
            .flatten()
            .all(|count| count == BALANCED_SCHEDULE_CYCLES));
        assert!(by_cycle
            .into_iter()
            .flatten()
            .flatten()
            .all(|count| count == 1));
    }

    #[test]
    fn benchmark_regions_and_preconditions_meet_release_floors() {
        let observed = black_box((
            TARGET_REGION_NS,
            CALIBRATION_TARGET_REGION_NS,
            PER_SAMPLE_UNTIMED_SAME_CELL_PRECONDITIONS,
            PER_SAMPLE_UNTIMED_PRECONDITION_MIN_NS,
            ORDER_MULTIPLE_COMPARISON_TESTS,
            ORDER_BOOTSTRAP_RESAMPLES,
            BALANCED_SCHEDULE_CYCLES,
        ));
        assert!(observed.0 >= 100_000_000);
        assert!(observed.1 >= 2 * observed.0);
        assert!(observed.2 >= 10);
        assert!(observed.3 >= 100_000_000);
        assert_eq!(observed.4, 36);
        assert!(observed.5 >= 100_000);
        assert!(observed.6 >= 20);
    }

    #[test]
    fn json_f64_round_trips_adversarial_binary64_values() {
        let values = [
            1.0e-20,
            0.1,
            123_456_789.123_456_79,
            f64::from_bits(0x0010_0000_0000_0001),
            9_007_199_254_740_991.0,
        ];
        let encoded = json_f64(&values);
        let decoded = encoded
            .split(',')
            .map(|value| value.parse::<f64>().expect("serialized f64 must parse"))
            .collect::<Vec<_>>();
        assert_eq!(decoded, values);
    }

    #[test]
    fn heterogeneous_cycle_effect_is_inconclusive() {
        let observations = synthetic_observations(|block_index, block, position| {
            // A direction-only heuristic called this contaminated when half
            // the cycles have the same >1% direction and half do not. Whole-
            // cycle uncertainty correctly retains both temporal regimes.
            let centered_jitter = ((block_index * 37 + block.metric_rotation * 11) % 7) as i128 - 3;
            let position_offset = if block.balanced_cycle < BALANCED_SCHEDULE_CYCLES / 2 {
                [-1_500i128, 0, 1_500][position]
            } else {
                0
            };
            let value = 100_000i128 + centered_jitter * 10 + position_offset;
            value.max(1) as u128
        });
        let assessment = order_assessment(
            &observations,
            Profile::Fp32,
            Metric::Pearson,
            1,
            0xdead_beef,
        );
        assert!(assessment.observed_spread_percent > 1.0);
        assert_eq!(assessment.classification, OrderClassification::Inconclusive);
    }

    #[test]
    fn stable_repeated_position_effect_is_confirmed() {
        let observations = synthetic_observations(|block_index, _, position| {
            100_000 + position as u128 * 3_000 + (block_index % 5) as u128 * 5
        });
        let assessment = order_assessment(
            &observations,
            Profile::Fp32,
            Metric::Pearson,
            1,
            0xcafe_f00d,
        );
        assert_eq!(assessment.classification, OrderClassification::Contaminated);
        assert!(assessment.contrasts.iter().any(|contrast| {
            contrast.corrected_bootstrap_ci_percent[0] > 1.0
                || contrast.corrected_bootstrap_ci_percent[1] < -1.0
        }));
    }

    #[test]
    fn metric_ordinal_drift_does_not_alias_into_profile_position() {
        let observations = synthetic_observations(|block_index, block, _| {
            100_000
                + block.metric_rotation as u128 * 4_000
                + block.balanced_cycle as u128 * 200
                + (block_index % 3) as u128
        });
        let assessment = order_assessment(
            &observations,
            Profile::Fp32,
            Metric::Pearson,
            1,
            0x0dd1_7a11,
        );
        assert_eq!(assessment.classification, OrderClassification::Clean);
        assert!(assessment.observed_spread_percent < 1.0);
    }

    #[test]
    fn tight_sub_one_percent_effect_is_clean() {
        let observations = synthetic_observations(|block_index, _, position| {
            100_000 + position as u128 * 150 + (block_index % 3) as u128 * 3
        });
        let assessment = order_assessment(
            &observations,
            Profile::Fp32,
            Metric::Pearson,
            1,
            0x1234_5678,
        );
        assert_eq!(assessment.classification, OrderClassification::Clean);
        assert!(assessment.contrasts.iter().all(|contrast| {
            contrast.corrected_bootstrap_ci_percent[0] >= -1.0
                && contrast.corrected_bootstrap_ci_percent[1] <= 1.0
        }));
    }
}
