//! Manual release benchmark for the profile-specialized Core arithmetic lanes.
//!
//! Run with an optimized build and pin the process to an isolated CPU, for
//! example:
//!
//! ```text
//! GAFIME_NATIVE_BENCH_WHEEL=/artifacts/gafime.whl \
//! taskset -c 4 cargo +1.89.0 test -p gafime-cpu --release \
//!   precision_profiles_native_release_benchmark -- --ignored --nocapture
//! ```
//!
//! Every reported cell receives ten untimed warmups and thirty measured
//! regions.  The six profile orders are each used five times, so a fixed
//! fp32 -> mixed -> fp64 warmup advantage cannot become release evidence.

use std::{
    env,
    fmt::Write as _,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use gafime_cpu::kernels::precision::{
    mutual_info_fixed_f32, mutual_info_fixed_f64, mutual_info_fixed_mixed, pearson_f32,
    pearson_f64, pearson_mixed, spearman_f32, spearman_f64, spearman_mixed,
};

const ROWS: usize = 131_072;
const WARMUPS: usize = 10;
const REPETITIONS: usize = 30;
const TARGET_REGION_NS: u128 = 5_000_000;
const CALIBRATION_TARGET_REGION_NS: u128 = TARGET_REGION_NS * 2;
const MAX_LOOP_COUNT: usize = 256;
const BOOTSTRAP_RESAMPLES: usize = 2_000;

#[derive(Clone, Copy)]
enum Profile {
    Fp32,
    Mixed,
    Fp64,
}

impl Profile {
    const fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Mixed => "mixed",
            Self::Fp64 => "fp64",
        }
    }
}

#[derive(Clone, Copy)]
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

struct Inputs {
    x_f32: Vec<f32>,
    y_f32: Vec<f32>,
    x_f64: Vec<f64>,
    y_f64: Vec<f64>,
}

fn inputs() -> Inputs {
    let mut x_f32 = Vec::with_capacity(ROWS);
    let mut y_f32 = Vec::with_capacity(ROWS);
    let mut x_f64 = Vec::with_capacity(ROWS);
    let mut y_f64 = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        // Deterministic finite values with ties and non-power-of-two periods
        // exercise vector tails, ranking, and histogram arithmetic.
        let row_f64 = row as f64;
        let x = ((row * 17 + row / 29) % 65_521) as f64 / 65_521.0
            + (row_f64 * 0.000_031_7).sin() * 1.0e-9
            + ((row * 11 + 3) % 97) as f64 * 1.0e-12;
        let y = (0.71f64 * x
            + ((row * 13) % 997) as f64 / 2_991.0
            + (row_f64 * 0.000_017_3).cos() * 1.0e-8)
            .sin();
        x_f64.push(x);
        y_f64.push(y);
        x_f32.push(x as f32);
        y_f32.push(y as f32);
    }
    assert!(
        x_f64
            .iter()
            .zip(&x_f32)
            .any(|(&wide, &narrow)| wide != f64::from(narrow)),
        "f64 benchmark input must not be a f32 widening"
    );
    Inputs {
        x_f32,
        y_f32,
        x_f64,
        y_f64,
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

fn calibrated_loop_count(profile: Profile, metric: Metric, data: &Inputs) -> usize {
    // Calibrate from a warm median rather than a single cold call; otherwise
    // cache/page-fault cost can choose a loop count whose measured region is
    // shorter than the declared stability target.
    for _ in 0..WARMUPS {
        black_box(timed_region(profile, metric, data, 1));
    }
    let probe = median(
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

fn median(mut values: Vec<u128>) -> u128 {
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

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
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
        return Vec::new();
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
    values
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

fn sha256(path: &Path) -> String {
    command_output("sha256sum", &[path.to_string_lossy().as_ref()])
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned()
}

fn source_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn observed_python_executable() -> PathBuf {
    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        let candidate = Path::new(&virtual_env).join("bin/python");
        if candidate.is_file() {
            return candidate;
        }
    }
    let observed = command_output("sh", &["-c", "command -v python3 || command -v python"]);
    PathBuf::from(observed)
}

fn provenance() -> (String, String, String, u64) {
    let root = source_root();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/precision_native_benchmark.rs");
    let commit = env::var("GAFIME_NATIVE_SOURCE_COMMIT")
        .ok()
        .filter(|value| value.len() == 40)
        .unwrap_or_else(|| {
            command_output(
                "git",
                &["-C", root.to_string_lossy().as_ref(), "rev-parse", "HEAD"],
            )
        });
    // A release runner may invoke Cargo through a pinned `+toolchain` while
    // the unqualified `rustc` on PATH names a different default toolchain.
    // Allow the build orchestration to bind the actual compiler identity
    // instead of recording that unrelated default.
    let rustc = env::var("GAFIME_NATIVE_RUSTC_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| command_output("rustc", &["--version"]))
        .replace('"', "'");
    let linker = env::var("GAFIME_NATIVE_LINKER_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            command_output("ld", &["--version"])
                .lines()
                .next()
                .unwrap_or("")
                .to_owned()
        })
        .replace('"', "'");
    let seed = env::var("GAFIME_NATIVE_BENCH_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0x51a7_2026_0809);
    let _source_identity = format!("{}:{}", source.display(), sha256(&source));
    (commit, rustc, linker, seed)
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
    profile_order: [Profile; 3],
    duration_ns: u128,
}

fn json_identity(path: &Path) -> String {
    let size = fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    format!(
        "{{\"path\":\"{}\",\"size_bytes\":{},\"sha256\":\"{}\"}}",
        path.display(),
        size,
        sha256(path)
    )
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
        .map(|value| format!("{value:.17}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn source_tree_state(root: &Path) -> &'static str {
    match Command::new("git")
        .args([
            "-C",
            root.to_string_lossy().as_ref(),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ])
        .output()
    {
        Ok(output) if output.status.success() && output.stdout.is_empty() => "clean",
        Ok(output) if output.status.success() => "dirty",
        _ => "unavailable",
    }
}

fn metric_profile_samples(raw: &[RawObservation], profile: Profile, metric: Metric) -> Vec<u128> {
    raw.iter()
        .filter(|observation| observation.profile.name() == profile.name())
        .filter(|observation| observation.metric.name() == metric.name())
        .map(|observation| observation.duration_ns)
        .collect()
}

#[test]
#[ignore = "manual release benchmark; optimized build and pinned CPU required"]
fn precision_profiles_native_release_benchmark() {
    let governor_before = cpu_governors();
    let data = inputs();
    let profile_orders = [
        [Profile::Fp32, Profile::Mixed, Profile::Fp64],
        [Profile::Fp32, Profile::Fp64, Profile::Mixed],
        [Profile::Mixed, Profile::Fp32, Profile::Fp64],
        [Profile::Mixed, Profile::Fp64, Profile::Fp32],
        [Profile::Fp64, Profile::Fp32, Profile::Mixed],
        [Profile::Fp64, Profile::Mixed, Profile::Fp32],
    ];
    let mut loop_counts = [[1usize; 3]; 4];
    let mut raw = std::array::from_fn::<_, 4, _>(|_| {
        std::array::from_fn::<_, 3, _>(|_| Vec::<u128>::with_capacity(REPETITIONS))
    });
    let mut observations = Vec::with_capacity(REPETITIONS * Metric::ALL.len() * 3);

    for (metric_index, metric) in Metric::ALL.into_iter().enumerate() {
        for (profile_index, profile) in [Profile::Fp32, Profile::Mixed, Profile::Fp64]
            .into_iter()
            .enumerate()
        {
            let loops = calibrated_loop_count(profile, metric, &data);
            loop_counts[metric_index][profile_index] = loops;
            for _ in 0..WARMUPS {
                black_box(timed_region(profile, metric, &data, loops));
            }
        }
    }

    for block in 0..REPETITIONS {
        let order = profile_orders[block % profile_orders.len()];
        let metric_rotation = block % Metric::ALL.len();
        for metric_offset in 0..Metric::ALL.len() {
            let metric_index = (metric_rotation + metric_offset) % Metric::ALL.len();
            let metric = Metric::ALL[metric_index];
            for profile in order {
                let profile_index = profile_index(profile);
                let duration = timed_region(
                    profile,
                    metric,
                    &data,
                    loop_counts[metric_index][profile_index],
                );
                raw[metric_index][profile_index].push(duration);
                observations.push(RawObservation {
                    profile,
                    metric,
                    order_index: block % profile_orders.len(),
                    profile_order: order,
                    duration_ns: duration,
                });
            }
        }
    }

    let profiles = [Profile::Fp32, Profile::Mixed, Profile::Fp64];
    let binary = std::env::current_exe().expect("benchmark executable path");
    let root = source_root();
    let source = root.join("crates/gafime-cpu/tests/precision_native_benchmark.rs");
    let wheel = env::var("GAFIME_NATIVE_BENCH_WHEEL")
        .expect("GAFIME_NATIVE_BENCH_WHEEL must name the exact Core wheel under test");
    let wheel = PathBuf::from(wheel)
        .canonicalize()
        .expect("GAFIME_NATIVE_BENCH_WHEEL must be a readable file");
    assert!(
        wheel.is_file(),
        "Core benchmark wheel identity must be a file"
    );
    let (source_commit, rustc, linker, seed) = provenance();
    assert_eq!(
        source_commit.len(),
        40,
        "native evidence requires a full source commit"
    );
    let affinity = process_affinity();
    let mut records = String::new();
    let mut record_count = 0usize;
    for profile in profiles {
        for (metric_index, metric) in Metric::ALL.into_iter().enumerate() {
            let samples = metric_profile_samples(&observations, profile, metric);
            assert_eq!(samples.len(), REPETITIONS);
            let loops = loop_counts[metric_index][profile_index(profile)];
            let per_call: Vec<f64> = samples
                .iter()
                .map(|value| *value as f64 / loops as f64)
                .collect();
            let center = median_f64(per_call.clone());
            let ci = bootstrap_ci(
                &per_call,
                seed ^ (metric_index as u64) ^ (profile_index(profile) as u64),
            );
            if record_count != 0 {
                records.push(',');
            }
            let raw_minimum = samples.iter().copied().min().unwrap_or(0);
            records.push_str(&format!(
                "{{\"profile\":\"{}\",\"operation\":\"metric_kernel\",\"metric\":\"{}\",\"samples_ns\":[{}],\"raw_samples_ns\":[{}],\"median_ns_per_call\":{},\"mad_ns_per_call\":{},\"p05_ns_per_call\":{},\"p95_ns_per_call\":{},\"bootstrap_median_95_ci_ns_per_call\":[{},{}],\"loop_count_per_sample\":{},\"sample_region_target_ns\":{},\"sample_region_min_observed_ns\":{},\"sample_region_target_met\":{}}}",
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
            ));
            record_count += 1;
        }
    }

    let raw_order = observations
        .iter()
        .map(|observation| {
            format!(
                "{{\"profile\":\"{}\",\"metric\":\"{}\",\"order_index\":{},\"profile_order\":{},\"duration_ns\":{}}}",
                observation.profile.name(),
                observation.metric.name(),
                observation.order_index,
                json_order(&observation.profile_order),
                observation.duration_ns,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let mut report = String::from(
        "{\"schema\":\"gafime.core-native-arithmetic.v2\",\"status\":\"pass\",\"backend\":\"core\",\"profiles\":[\"fp32\",\"mixed\",\"fp64\"]",
    );
    report.push_str(&format!(
        ",\"source_commit\":\"{}\",\"source_tree_state\":{{\"status\":\"{}\"}},\"input_policy\":\"common-f64\",\"input_identity\":{{\"generator\":\"gafime-core-native-v1\",\"source_dtype\":\"float64\",\"derived_fp32_by_cast\":true,\"rows\":{},\"mi_bins\":32}},\"workload\":{{\"name\":\"metric-specific-core\",\"rows\":{},\"features\":1,\"candidates\":1,\"arity\":1,\"mi_bins\":32,\"input_bytes_f64\":{}}},\"rows\":{},\"warmups\":{},\"repeats\":{}",
        source_commit,
        source_tree_state(&root),
        ROWS,
        ROWS,
        ROWS * 2 * std::mem::size_of::<f64>(),
        ROWS,
        WARMUPS,
        REPETITIONS
    ));
    report.push_str(
        ",\"profile_orders\":[[\"fp32\",\"mixed\",\"fp64\"],[\"fp32\",\"fp64\",\"mixed\"],[\"mixed\",\"fp32\",\"fp64\"],[\"mixed\",\"fp64\",\"fp32\"],[\"fp64\",\"fp32\",\"mixed\"],[\"fp64\",\"mixed\",\"fp32\"]]",
    );
    report.push_str(&format!(
        ",\"target_region_ns\":{},\"calibration_target_region_ns\":{},\"measurement_scope\":\"native_arithmetic_only\",\"decomposition_boundaries\":{{\"candidate_materialization\":\"included in each metric evaluation\",\"report_construction\":\"not measured by this native arithmetic benchmark\"}}",
        TARGET_REGION_NS,
        CALIBRATION_TARGET_REGION_NS,
    ));
    let target = format!("{}-{}", env::consts::ARCH, env::consts::OS);
    let governor_after = cpu_governors();
    let cpu = cpu_identity();
    let environment = benchmark_environment_json();
    let python_executable = observed_python_executable();
    report.push_str(&format!(
        ",\"compiler\":{{\"rustc\":\"{}\",\"linker\":\"{}\",\"target\":\"{}\"}},\"device\":{{\"kind\":\"cpu\",\"identity\":{}}},\"process_affinity\":\"{}\",\"clock\":\"std::time::Instant monotonic clock\",\"clock_and_power_state\":{{\"before\":{{\"cpu_governor\":{}}},\"after\":{{\"cpu_governor\":{}}}}},\"environment\":{},\"provenance\":{{\"source_root\":\"{}\",\"source_tree_state\":{{\"status\":\"{}\"}},\"benchmark_source\":{},\"benchmark_binary\":{},\"wheel\":{},\"python_executable\":{}}},\"records\":[{}],\"raw_order\":[{}]}}",
        rustc,
        linker,
        target,
        json_string(&cpu),
        affinity,
        json_strings(&governor_before),
        json_strings(&governor_after),
        environment,
        root.display(),
        source_tree_state(&root),
        json_identity(&source),
        json_identity(&binary),
        json_identity(&wheel),
        json_identity(&python_executable),
        records,
        raw_order,
    ));
    if let Ok(path) = env::var("GAFIME_NATIVE_BENCH_OUTPUT") {
        fs::write(&path, &report).expect("write native benchmark artifact");
        eprintln!("wrote Core native benchmark artifact to {path}");
    }
    println!("GAFIME_NATIVE_BENCH {report}");
}
