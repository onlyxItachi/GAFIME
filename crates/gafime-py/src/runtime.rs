use gafime_cpu::simd::{finite_dispatch_isa, IsaLevel};
use gafime_gpu_sys::{GpuArchitectureClass, GpuBackend, GpuDeviceProfile, GpuSysError};
use gafime_orchestrator::config::EngineConfig;
use gafime_types::{
    GafimeGpuDeviceInfo, GafimeGpuGraphCapability, GafimePrecisionCapabilities, PrecisionProfile,
    GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM,
    GAFIME_DTYPE_MASK_F32, GAFIME_DTYPE_MASK_F64, GAFIME_GRAPH_HOST_REPLAY,
    GAFIME_GRAPH_STREAM_CAPTURE, GAFIME_GRAPH_UNSUPPORTED, GAFIME_METRIC_MUTUAL_INFO,
    GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};

use crate::common::{public_package_version, validate_metric_ids, PyBoundaryError, BOUNDARY_NAME};

#[cfg(feature = "local-cmake-experiment")]
mod local_cmake_experiment;

pub(crate) fn parse_engine_config(config: &Bound<'_, PyDict>) -> PyResult<EngineConfig> {
    validate_family_flags(
        get_bool(config, "enable_time_series_functions", false)?,
        get_bool(config, "enable_decision_path_functions", false)?,
    )
    .map_err(PyErr::from)?;
    let precision = validate_precision_request(&get_string(config, "precision", "mixed")?)
        .map_err(PyErr::from)?;

    let mut out = EngineConfig::default();
    let backend_name = get_string(config, "backend", "auto")?;
    out.device_id = get_u32(config, "device_id", 0)?;
    out.precision = precision;
    out.backend_kind = backend_kind_from_name(&backend_name, out.device_id, precision)?;
    out.metric_ids = metric_ids_from_names(get_vec_string(config, "metric_names")?)?;
    out.num_repeats = get_u32(config, "num_repeats", 3)?;
    out.permutation_tests = get_u32(config, "permutation_tests", 25)?;
    out.significance_top_n = get_u32(config, "significance_top_n", 50)?;
    let (random_seed, planning_seed_words) = get_python_integer_seed(config, "random_seed")?;
    out.random_seed = random_seed;
    out.planning_seed_words = planning_seed_words;
    out.mi_bins = get_u32(config, "mi_bins", 96)?;
    out.mi_approximate = get_bool(config, "mi_approximate", false)?;
    if let Some(flags) = get_optional_dict(config, "compile_flags")? {
        out.graph_requested = get_bool(&flags, "graph", false)?;
    }

    if let Some(budget) = get_optional_dict(config, "budget")? {
        out.budget.max_comb_size = get_u32(&budget, "max_comb_size", 2)?;
        out.budget.max_combinations_per_k = get_u64(&budget, "max_combinations_per_k", 5_000)?;
        out.budget.top_features_for_higher_k = get_u32(&budget, "top_features_for_higher_k", 50)?;
        out.budget.max_generated_features = get_u32(&budget, "max_generated_features", 0)?;
        out.budget.max_time_series_candidates =
            get_u64(&budget, "max_time_series_candidates", 100_000)?;
        out.budget.top_k_features_for_time_series =
            get_u32(&budget, "top_k_features_for_time_series", 50)?;
        out.budget.max_feature_candidate = match get_optional_i64(&budget, "max_feature_candidate")?
        {
            None => -2,
            Some(value) if value >= -1 => value,
            Some(_) => {
                return Err(PyValueError::new_err(
                    "budget.max_feature_candidate must be >= 0 or -1 for power-user mode",
                ))
            }
        };
        out.budget.vram_budget_mb = get_u64(&budget, "vram_budget_mb", 6_144)?;
    }

    if out.budget.max_comb_size == 0 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "budget.max_comb_size must be greater than zero".to_string(),
        )));
    }
    if out.budget.max_combinations_per_k == 0 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "budget.max_combinations_per_k must be greater than zero".to_string(),
        )));
    }
    if out.significance_top_n == 0 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "significance_top_n must be greater than zero".to_string(),
        )));
    }
    if out.mi_bins < 2 {
        return Err(PyErr::from(PyBoundaryError::InvalidInput(
            "mi_bins must be at least 2".to_string(),
        )));
    }
    Ok(out)
}

fn validate_precision_request(precision: &str) -> Result<PrecisionProfile, PyBoundaryError> {
    match precision.trim().to_ascii_lowercase().as_str() {
        "fp32" => Ok(PrecisionProfile::Fp32),
        "mixed" => Ok(PrecisionProfile::Mixed),
        "fp64" => Ok(PrecisionProfile::Fp64),
        _ => Err(PyBoundaryError::InvalidInput(
            "precision must be one of: fp32, mixed, fp64".to_string(),
        )),
    }
}

fn precision_profile_name(precision: PrecisionProfile) -> &'static str {
    match precision {
        PrecisionProfile::Fp32 => "fp32",
        PrecisionProfile::Mixed => "mixed",
        PrecisionProfile::Fp64 => "fp64",
    }
}

fn validate_family_flags(
    enable_time_series: bool,
    enable_decision_path: bool,
) -> Result<(), PyBoundaryError> {
    if enable_time_series {
        return Err(PyBoundaryError::UnsupportedFeature(
            "time-series families must use v1 Rust family descriptors; device kernels are not wired to this Python boundary yet"
                .to_string(),
        ));
    }
    if enable_decision_path {
        return Err(PyBoundaryError::UnsupportedFeature(
            "decision-path families must use v1 Rust family descriptors; device kernels are not wired to this Python boundary yet"
                .to_string(),
        ));
    }
    Ok(())
}

fn backend_kind_from_name(
    name: &str,
    device_id: u32,
    precision: PrecisionProfile,
) -> PyResult<u32> {
    backend_kind_from_name_result(name, device_id, precision).map_err(PyErr::from)
}

fn backend_kind_from_name_result(
    name: &str,
    device_id: u32,
    precision: PrecisionProfile,
) -> Result<u32, PyBoundaryError> {
    match name {
        "auto" => Ok(resolve_auto_backend(device_id, precision)),
        "cpu" | "core" | "rust" | "v1-rust-cpu" => Ok(GAFIME_BACKEND_CPU),
        "cuda" => Ok(GAFIME_BACKEND_CUDA),
        "gpu" => Err(PyBoundaryError::UnsupportedFeature(
            "backend \"gpu\" is ambiguous in v1; request backend \"cuda\", \"rocm\", or \"metal\" explicitly"
                .to_string(),
        )),
        "rocm" | "hip" => Ok(GAFIME_BACKEND_ROCM),
        "metal" if precision == PrecisionProfile::Fp32 => Ok(GAFIME_BACKEND_METAL),
        "metal" => Err(PyBoundaryError::UnsupportedFeature(
            "Metal supports precision=\"fp32\" only; mixed and fp64 require native double arithmetic and do not silently fall back to Core"
                .to_string(),
        )),
        other => Err(PyBoundaryError::InvalidInput(format!(
            "unknown backend {other:?}"
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AutoBackendCandidate {
    kind: u32,
    score: i64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RuntimeCacheCounters {
    pub(crate) metric_hits: u64,
    pub(crate) metric_builds: u64,
    pub(crate) candidate_table_hits: u64,
}

// Capability probing decides eligibility. The score only provides a stable
// preference among already-compatible payloads; it is neither capability proof
// nor a performance guarantee. The fixed tie-breaker keeps equal scores
// deterministic.
fn resolve_auto_backend(device_id: u32, precision: PrecisionProfile) -> u32 {
    let metal = if precision == PrecisionProfile::Fp32 {
        probe_gpu_candidate(GAFIME_BACKEND_METAL, device_id, precision)
    } else {
        None
    };
    [
        probe_gpu_candidate(GAFIME_BACKEND_CUDA, device_id, precision),
        probe_gpu_candidate(GAFIME_BACKEND_ROCM, device_id, precision),
        metal,
    ]
    .into_iter()
    .flatten()
    .max_by_key(|candidate| (candidate.score, backend_tie_breaker(candidate.kind)))
    .map(|candidate| candidate.kind)
    .unwrap_or_else(|| {
        let _ = cpu_isa_rank(finite_dispatch_isa());
        GAFIME_BACKEND_CPU
    })
}

#[derive(Clone, Debug)]
struct GpuRuntimeProbe {
    kind: u32,
    info: GafimeGpuDeviceInfo,
    graph: GafimeGpuGraphCapability,
    precision: GafimePrecisionCapabilities,
    permutation_profile_mask: u32,
    supports_interaction_diagnostics: bool,
    #[cfg(feature = "local-cmake-experiment")]
    local_cmake_experiment: local_cmake_experiment::RuntimeProbe,
    library_path: Option<String>,
}

fn probe_gpu_runtime(kind: u32, device_id: u32) -> Result<GpuRuntimeProbe, GpuSysError> {
    let backend = match kind {
        GAFIME_BACKEND_CUDA => GpuBackend::cuda_from_env(device_id),
        GAFIME_BACKEND_ROCM => GpuBackend::rocm_from_env(device_id),
        GAFIME_BACKEND_METAL => GpuBackend::metal_from_env(device_id),
        _ => return Err(GpuSysError::InvalidInput("unsupported GPU backend kind")),
    }?;
    let library_path = backend
        .loaded_library_path()
        .map(|path| path.display().to_string());
    let info = backend.device_info()?;
    let graph = backend.graph_capability()?;
    let precision = backend.precision_capabilities()?;
    Ok(GpuRuntimeProbe {
        kind,
        info,
        graph,
        precision,
        permutation_profile_mask: backend.precision_permutation_profile_mask(),
        supports_interaction_diagnostics: backend.supports_interaction_diagnostics(),
        #[cfg(feature = "local-cmake-experiment")]
        local_cmake_experiment: local_cmake_experiment::probe(&backend),
        library_path,
    })
}

fn probe_gpu_candidate(
    kind: u32,
    device_id: u32,
    precision: PrecisionProfile,
) -> Option<AutoBackendCandidate> {
    // A payload is eligible for automatic selection only when its required
    // identity/capability query succeeds. Allocation success is not a substitute:
    // older or mismatched payloads can allocate while exposing an incompatible ABI.
    let probe = probe_gpu_runtime(kind, device_id).ok()?;
    if !precision_capabilities_support(&probe.precision, precision) {
        return None;
    }
    Some(AutoBackendCandidate {
        kind,
        score: gpu_device_score(&probe.info),
    })
}

fn precision_capabilities_support(
    capabilities: &GafimePrecisionCapabilities,
    precision: PrecisionProfile,
) -> bool {
    let storage_mask = if precision == PrecisionProfile::Fp64 {
        GAFIME_DTYPE_MASK_F64
    } else {
        GAFIME_DTYPE_MASK_F32
    };
    let result_mask = if precision == PrecisionProfile::Fp32 {
        GAFIME_DTYPE_MASK_F32
    } else {
        GAFIME_DTYPE_MASK_F64
    };
    (capabilities.profile_mask & precision.capability_mask()) != 0
        && (capabilities.storage_dtype_mask & storage_mask) != 0
        && (capabilities.result_dtype_mask & result_mask) != 0
}

fn gpu_device_score(info: &GafimeGpuDeviceInfo) -> i64 {
    let profile = GpuDeviceProfile::from_info(info);
    let mut score = 1_000_000i64;
    score += match profile.architecture {
        GpuArchitectureClass::NvidiaBlackwell => 80_000,
        GpuArchitectureClass::NvidiaHopper => 75_000,
        GpuArchitectureClass::NvidiaAda => 70_000,
        GpuArchitectureClass::NvidiaAmpere => 62_000,
        GpuArchitectureClass::NvidiaTuring => 52_000,
        GpuArchitectureClass::AmdCdna => 68_000,
        GpuArchitectureClass::AmdRdna => 58_000,
        GpuArchitectureClass::Apple => 54_000,
        GpuArchitectureClass::VendorSpecific(value) => 30_000 + (value.min(20_000) as i64),
        GpuArchitectureClass::Unknown => 20_000,
    };
    if profile.discrete {
        score += 12_000;
    }
    if profile.high_bandwidth {
        score += 8_000;
    }
    if profile.unified_memory {
        score += 4_000;
    }
    if profile.integrated {
        score += 2_000;
    }
    if profile.managed_memory {
        score += 1_000;
    }
    score += (info.total_global_mem_bytes / (1024 * 1024 * 512)).min(256) as i64;
    score += (info.multiprocessor_count as i64) * 64;
    score += (info.compute_major as i64) * 128 + (info.compute_minor as i64);
    score
}

fn backend_tie_breaker(kind: u32) -> i64 {
    match kind {
        GAFIME_BACKEND_CUDA => 30,
        GAFIME_BACKEND_ROCM => 20,
        GAFIME_BACKEND_METAL => 10,
        _ => 0,
    }
}

fn cpu_isa_rank(isa: IsaLevel) -> i64 {
    match isa {
        IsaLevel::Avx512 => 50_000,
        IsaLevel::Avx2 => 40_000,
        IsaLevel::Sse42 | IsaLevel::Neon => 30_000,
        IsaLevel::Scalar => 10_000,
    }
}

fn normalize_runtime_backend(name: &str) -> Result<&'static str, PyBoundaryError> {
    match name {
        "auto" => Ok("auto"),
        "cpu" | "core" | "rust" | "v1-rust-cpu" => Ok("core"),
        "cuda" => Ok("cuda"),
        "rocm" | "hip" => Ok("rocm"),
        "metal" => Ok("metal"),
        "gpu" => Err(PyBoundaryError::UnsupportedFeature(
            "backend \"gpu\" is ambiguous in v1; request backend \"cuda\", \"rocm\", or \"metal\" explicitly"
                .to_string(),
        )),
        other => Err(PyBoundaryError::InvalidInput(format!(
            "unknown backend {other:?}"
        ))),
    }
}

fn backend_kind_for_runtime_name(name: &str) -> u32 {
    match name {
        "cuda" => GAFIME_BACKEND_CUDA,
        "rocm" => GAFIME_BACKEND_ROCM,
        "metal" => GAFIME_BACKEND_METAL,
        _ => GAFIME_BACKEND_CPU,
    }
}

fn device_name(info: &GafimeGpuDeviceInfo) -> String {
    let length = info
        .name
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(info.name.len());
    String::from_utf8_lossy(&info.name[..length]).into_owned()
}

fn graph_mode_name(mode: u32) -> &'static str {
    match mode {
        GAFIME_GRAPH_UNSUPPORTED => "unsupported",
        GAFIME_GRAPH_STREAM_CAPTURE => "stream_capture",
        GAFIME_GRAPH_HOST_REPLAY => "host_replay",
        _ => "vendor_specific",
    }
}

fn runtime_probe_to_python<'py>(
    py: Python<'py>,
    probe: &GpuRuntimeProbe,
) -> PyResult<Bound<'py, PyDict>> {
    let profile = GpuDeviceProfile::from_info(&probe.info);
    let device = PyDict::new(py);
    let name = device_name(&probe.info);
    if name.is_empty() {
        device.set_item("name", py.None())?;
    } else {
        device.set_item("name", name)?;
    }
    device.set_item("device_id", probe.info.device_id)?;
    device.set_item("flags", probe.info.flags)?;
    device.set_item("architecture_class", probe.info.reserved[0])?;
    device.set_item("total_global_mem_bytes", probe.info.total_global_mem_bytes)?;
    device.set_item("multiprocessor_count", probe.info.multiprocessor_count)?;
    device.set_item("warp_size", probe.info.warp_size)?;
    device.set_item("compute_major", probe.info.compute_major)?;
    device.set_item("compute_minor", probe.info.compute_minor)?;
    device.set_item("driver_version", probe.info.driver_version)?;
    device.set_item("runtime_version", probe.info.runtime_version)?;
    device.set_item("unified_memory", profile.unified_memory)?;
    device.set_item("integrated", profile.integrated)?;
    device.set_item("discrete", profile.discrete)?;
    device.set_item("managed_memory", profile.managed_memory)?;
    device.set_item("high_bandwidth", profile.high_bandwidth)?;

    let graph = PyDict::new(py);
    graph.set_item(
        "supported",
        probe.graph.graph_mode != GAFIME_GRAPH_UNSUPPORTED,
    )?;
    graph.set_item("mode", graph_mode_name(probe.graph.graph_mode))?;
    graph.set_item("flags", probe.graph.flags)?;
    graph.set_item(
        "supports_memcpy_nodes",
        probe.graph.supports_memcpy_nodes != 0,
    )?;
    graph.set_item(
        "supports_kernel_param_update",
        probe.graph.supports_kernel_param_update != 0,
    )?;
    graph.set_item(
        "supports_device_ranking",
        probe.graph.supports_device_ranking != 0,
    )?;
    graph.set_item("max_captured_nodes", probe.graph.max_captured_nodes)?;
    graph.set_item("stable_pointer_flags", probe.graph.stable_pointer_flags)?;

    let significance = PyDict::new(py);
    let permutation_profiles = [
        ("fp32", PrecisionProfile::Fp32),
        ("mixed", PrecisionProfile::Mixed),
        ("fp64", PrecisionProfile::Fp64),
    ]
    .into_iter()
    .filter_map(|(name, profile)| {
        ((probe.permutation_profile_mask & profile.capability_mask()) != 0).then_some(name)
    })
    .collect::<Vec<_>>();
    significance.set_item("permutation_pvalues_abi", !permutation_profiles.is_empty())?;
    significance.set_item("permutation_profiles", permutation_profiles)?;

    let precision = PyDict::new(py);
    let mut profiles = Vec::new();
    for (name, profile_kind) in [
        ("fp32", PrecisionProfile::Fp32),
        ("mixed", PrecisionProfile::Mixed),
        ("fp64", PrecisionProfile::Fp64),
    ] {
        if (probe.precision.profile_mask & profile_kind.capability_mask()) != 0 {
            profiles.push(name);
        }
    }
    let mut storage_dtypes = Vec::new();
    if (probe.precision.storage_dtype_mask & GAFIME_DTYPE_MASK_F32) != 0 {
        storage_dtypes.push("float32");
    }
    if (probe.precision.storage_dtype_mask & GAFIME_DTYPE_MASK_F64) != 0 {
        storage_dtypes.push("float64");
    }
    let mut result_dtypes = Vec::new();
    if (probe.precision.result_dtype_mask & GAFIME_DTYPE_MASK_F32) != 0 {
        result_dtypes.push("float32");
    }
    if (probe.precision.result_dtype_mask & GAFIME_DTYPE_MASK_F64) != 0 {
        result_dtypes.push("float64");
    }
    let profile_domains = PyDict::new(py);
    for profile_name in &profiles {
        let domains = PyDict::new(py);
        let storage_dtype = if *profile_name == "fp64" {
            "float64"
        } else {
            "float32"
        };
        let result_dtype = if *profile_name == "fp32" {
            "float32"
        } else {
            "float64"
        };
        domains.set_item("storage_dtype", storage_dtype)?;
        domains.set_item("interaction_arithmetic", storage_dtype)?;
        domains.set_item("reduction_dtype", result_dtype)?;
        domains.set_item("result_dtype", result_dtype)?;
        let accumulators = PyDict::new(py);
        for metric in ["pearson", "r2", "spearman", "mutual_info"] {
            accumulators.set_item(metric, result_dtype)?;
        }
        domains.set_item("accumulators", accumulators)?;
        profile_domains.set_item(profile_name, domains)?;
    }
    precision.set_item("profiles", profiles)?;
    precision.set_item("storage_dtypes", storage_dtypes)?;
    precision.set_item("result_dtypes", result_dtypes)?;
    precision.set_item("profile_domains", profile_domains)?;
    precision.set_item("scale_normalization", "adaptive_high_dynamic")?;
    precision.set_item("compensated_summation", false)?;
    precision.set_item(
        "interaction_overflow_diagnostics",
        probe.supports_interaction_diagnostics,
    )?;

    let runtime = PyDict::new(py);
    runtime.set_item("backend", backend_capability_name_for_kind(probe.kind))?;
    runtime.set_item("device", device)?;
    runtime.set_item("graph", graph)?;
    runtime.set_item("significance", significance)?;
    #[cfg(feature = "local-cmake-experiment")]
    local_cmake_experiment::add_runtime_capabilities(
        py,
        &runtime,
        &profile,
        &probe.local_cmake_experiment,
    )?;
    runtime.set_item("precision", precision)?;
    match &probe.library_path {
        Some(path) => runtime.set_item("library_path", path)?,
        None => runtime.set_item("library_path", py.None())?,
    }
    Ok(runtime)
}

fn runtime_probe_error_to_python<'py>(
    py: Python<'py>,
    error: &GpuSysError,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("status", "unavailable")?;
    result.set_item("detail", error.to_string())?;
    Ok(result)
}

/// Runtime-only facts for public Python capability reporting. The function uses
/// the same `GpuBackend::*_from_env` loader seam as normal execution; payload
/// discovery can evolve behind that seam without changing the public shape.
#[pyfunction]
#[pyo3(signature = (backend="auto", device_id=0, probe=false, *, precision="mixed"))]
pub(crate) fn runtime_capabilities(
    py: Python<'_>,
    backend: &str,
    device_id: u32,
    probe: bool,
    precision: &str,
) -> PyResult<Py<PyDict>> {
    let requested_precision = validate_precision_request(precision).map_err(PyErr::from)?;
    let backend = normalize_runtime_backend(backend).map_err(PyErr::from)?;
    if backend == "metal" && requested_precision != PrecisionProfile::Fp32 {
        return Err(PyErr::from(PyBoundaryError::UnsupportedFeature(
            "Metal supports precision=\"fp32\" only; capability probing for mixed/fp64 is rejected before payload discovery"
                .to_string(),
        )));
    }
    let result = PyDict::new(py);
    let candidates = PyDict::new(py);
    result.set_item("configured_backend", backend)?;
    result.set_item("probe_performed", probe)?;
    result.set_item("native_version", public_package_version())?;
    result.set_item("boundary_name", BOUNDARY_NAME)?;
    result.set_item(
        "requested_precision",
        precision_profile_name(requested_precision),
    )?;
    result.set_item("candidates", &candidates)?;
    result.set_item("runtime", py.None())?;

    if backend == "core" {
        result.set_item("status", "available")?;
        result.set_item("selected_backend", "core")?;
        result.set_item("detail", "Core is built into the native boundary.")?;
        result.set_item(
            "effective_precision",
            precision_profile_name(requested_precision),
        )?;
        return Ok(result.unbind());
    }

    if !probe {
        result.set_item("status", "not_probed")?;
        result.set_item("selected_backend", py.None())?;
        if backend == "auto" {
            result.set_item(
                "detail",
                "automatic selection was not probed; no backend was selected",
            )?;
        } else {
            result.set_item(
                "detail",
                format!("{backend} was configured but runtime payload probing is disabled"),
            )?;
        }
        return Ok(result.unbind());
    }

    if backend != "auto" {
        let kind = backend_kind_for_runtime_name(backend);
        match probe_gpu_runtime(kind, device_id) {
            Ok(probe_result)
                if precision_capabilities_support(&probe_result.precision, requested_precision) =>
            {
                let candidate = PyDict::new(py);
                candidate.set_item("status", "available")?;
                candidate.set_item("runtime", runtime_probe_to_python(py, &probe_result)?)?;
                candidates.set_item(backend, candidate)?;
                result.set_item("status", "available")?;
                result.set_item("selected_backend", backend)?;
                result.set_item("detail", "explicit backend passed the runtime ABI probe")?;
                result.set_item(
                    "effective_precision",
                    precision_profile_name(requested_precision),
                )?;
                result.set_item("runtime", runtime_probe_to_python(py, &probe_result)?)?;
            }
            Ok(_) => {
                result.set_item("status", "unavailable")?;
                result.set_item("selected_backend", py.None())?;
                result.set_item(
                    "detail",
                    format!(
                        "{backend} payload does not advertise precision={}",
                        precision_profile_name(requested_precision)
                    ),
                )?;
            }
            Err(error) => {
                candidates.set_item(backend, runtime_probe_error_to_python(py, &error)?)?;
                result.set_item("status", "unavailable")?;
                result.set_item("selected_backend", py.None())?;
                result.set_item("detail", error.to_string())?;
            }
        }
        return Ok(result.unbind());
    }

    let mut probes = Vec::new();
    for kind in [
        GAFIME_BACKEND_CUDA,
        GAFIME_BACKEND_ROCM,
        GAFIME_BACKEND_METAL,
    ] {
        if kind == GAFIME_BACKEND_METAL && requested_precision != PrecisionProfile::Fp32 {
            continue;
        }
        let name = backend_capability_name_for_kind(kind);
        match probe_gpu_runtime(kind, device_id) {
            Ok(probe_result)
                if precision_capabilities_support(&probe_result.precision, requested_precision) =>
            {
                let candidate = PyDict::new(py);
                candidate.set_item("status", "available")?;
                candidate.set_item("runtime", runtime_probe_to_python(py, &probe_result)?)?;
                candidates.set_item(name, candidate)?;
                probes.push(probe_result);
            }
            Ok(_) => {
                let unavailable = PyDict::new(py);
                unavailable.set_item("status", "unavailable")?;
                unavailable.set_item(
                    "detail",
                    format!(
                        "payload does not advertise precision={}",
                        precision_profile_name(requested_precision)
                    ),
                )?;
                candidates.set_item(name, unavailable)?;
            }
            Err(error) => {
                candidates.set_item(name, runtime_probe_error_to_python(py, &error)?)?;
            }
        }
    }
    let selected = probes.iter().max_by_key(|candidate| {
        (
            gpu_device_score(&candidate.info),
            backend_tie_breaker(candidate.kind),
        )
    });
    match selected {
        Some(probe_result) => {
            let name = backend_capability_name_for_kind(probe_result.kind);
            result.set_item("status", "available")?;
            result.set_item("selected_backend", name)?;
            result.set_item(
                "detail",
                format!("auto selected {name} after runtime ABI probes"),
            )?;
            result.set_item(
                "effective_precision",
                precision_profile_name(requested_precision),
            )?;
            result.set_item("runtime", runtime_probe_to_python(py, probe_result)?)?;
        }
        None => {
            result.set_item("status", "available")?;
            result.set_item("selected_backend", "core")?;
            result.set_item(
                "detail",
                "auto selected core because no GPU payload passed the runtime ABI probe",
            )?;
            result.set_item(
                "effective_precision",
                precision_profile_name(requested_precision),
            )?;
        }
    }
    Ok(result.unbind())
}

fn metric_ids_from_names(names: Vec<String>) -> PyResult<Vec<u32>> {
    metric_ids_from_names_result(names).map_err(PyErr::from)
}

fn metric_ids_from_names_result(names: Vec<String>) -> Result<Vec<u32>, PyBoundaryError> {
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        ids.push(match name.as_str() {
            "pearson" => GAFIME_METRIC_PEARSON,
            "spearman" => GAFIME_METRIC_SPEARMAN,
            "mutual_info" => GAFIME_METRIC_MUTUAL_INFO,
            "r2" => GAFIME_METRIC_R2,
            other => {
                return Err(PyBoundaryError::InvalidInput(format!(
                    "unsupported metric {other:?}"
                )))
            }
        });
    }
    validate_metric_ids(ids)
}

fn get_optional_dict<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.cast_into::<PyDict>()?)),
        _ => Ok(None),
    }
}

fn get_string(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<String>(),
        _ => Ok(default.to_string()),
    }
}

fn get_bool(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<bool>(),
        _ => Ok(default),
    }
}

pub(crate) fn get_u32(dict: &Bound<'_, PyDict>, key: &str, default: u32) -> PyResult<u32> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<u32>(),
        _ => Ok(default),
    }
}

fn get_u64(dict: &Bound<'_, PyDict>, key: &str, default: u64) -> PyResult<u64> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<u64>(),
        _ => Ok(default),
    }
}

fn get_optional_i64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<i64>().map(Some),
        _ => Ok(None),
    }
}

fn get_python_integer_seed(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<(u64, Vec<u32>)> {
    let Some(value) = dict.get_item(key)? else {
        return Ok((0, vec![0]));
    };
    if value.is_none() {
        return Ok((0, vec![0]));
    }
    parse_python_integer_seed(&value)
}

pub(crate) fn parse_python_integer_seed(value: &Bound<'_, PyAny>) -> PyResult<(u64, Vec<u32>)> {
    let absolute = value.call_method0("__abs__")?;
    let bit_length = absolute.call_method0("bit_length")?.extract::<usize>()?;
    let byte_length = bit_length.div_ceil(8).max(1);
    let bytes = absolute
        .call_method1("to_bytes", (byte_length, "little"))?
        .extract::<Vec<u8>>()?;
    let mut words = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(word));
    }
    let significance_seed = significance_seed_from_words(&words);
    Ok((significance_seed, words))
}

fn significance_seed_from_words(words: &[u32]) -> u64 {
    let low = u64::from(words.first().copied().unwrap_or(0))
        | (u64::from(words.get(1).copied().unwrap_or(0)) << 32);
    if words.len() <= 2 {
        return low;
    }

    // Preserve the historical u64 stream for ordinary seeds, but fold every
    // higher Python integer word into significance scheduling. FNV-1a is used
    // only as a stable identity compressor; planning still consumes all words.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash ^ (words.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn get_vec_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<Vec<String>>(),
        _ => Ok(vec![
            "pearson".to_string(),
            "spearman".to_string(),
            "mutual_info".to_string(),
            "r2".to_string(),
        ]),
    }
}

pub(crate) fn backend_name_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CUDA => "v1-cuda-cabi",
        GAFIME_BACKEND_ROCM => "v1-rocm-cabi",
        GAFIME_BACKEND_METAL => "v1-metal-cabi",
        _ => "v1-rust-cpu",
    }
}

pub(crate) fn backend_capability_name_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CUDA => "cuda",
        GAFIME_BACKEND_ROCM => "rocm",
        GAFIME_BACKEND_METAL => "metal",
        _ => "core",
    }
}

pub(crate) fn execution_placement_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CPU => "gafime_cpu",
        _ => backend_capability_name_for_kind(backend_kind),
    }
}

pub(crate) fn backend_device_for_kind(backend_kind: u32) -> &'static str {
    match backend_kind {
        GAFIME_BACKEND_CUDA => "cuda",
        GAFIME_BACKEND_ROCM => "rocm",
        GAFIME_BACKEND_METAL => "metal",
        _ => "cpu",
    }
}

pub(crate) fn backend_is_gpu(backend_kind: u32) -> bool {
    matches!(
        backend_kind,
        GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_cpu::simd::IsaLevel;

    #[test]
    fn rust_config_boundary_rejects_unknown_metric() {
        let error = metric_ids_from_names_result(vec!["pearson".to_string(), "bogus".to_string()])
            .unwrap_err();

        assert!(error.to_string().contains("unsupported metric"));
    }

    #[test]
    fn rust_config_boundary_accepts_explicit_cuda() {
        assert_eq!(
            backend_kind_from_name_result("cuda", 0, PrecisionProfile::Mixed).unwrap(),
            GAFIME_BACKEND_CUDA
        );
    }

    #[test]
    fn rust_config_boundary_accepts_explicit_metal() {
        assert_eq!(
            backend_kind_from_name_result("metal", 0, PrecisionProfile::Fp32).unwrap(),
            GAFIME_BACKEND_METAL
        );
    }

    #[test]
    fn rust_config_boundary_rejects_ambiguous_gpu_without_python_fallback() {
        let error = backend_kind_from_name_result("gpu", 0, PrecisionProfile::Mixed).unwrap_err();

        assert!(error.to_string().contains("ambiguous"));
    }

    #[test]
    fn rust_config_boundary_enforces_the_precision_contract() {
        assert_eq!(
            validate_precision_request("fp32").unwrap(),
            PrecisionProfile::Fp32
        );
        assert_eq!(
            validate_precision_request("mixed").unwrap(),
            PrecisionProfile::Mixed
        );
        assert_eq!(
            validate_precision_request("fp64").unwrap(),
            PrecisionProfile::Fp64
        );
        let unknown_error = validate_precision_request("binary128").unwrap_err();
        assert!(unknown_error.to_string().contains("precision"));
        for alias in ["f32", "float32", "f64", "float64"] {
            assert!(validate_precision_request(alias).is_err());
        }
        let metal_error =
            backend_kind_from_name_result("metal", 0, PrecisionProfile::Fp64).unwrap_err();
        assert!(metal_error.to_string().contains("fp32"));
    }

    #[test]
    fn auto_backend_resolver_returns_supported_backend_kind() {
        assert!(matches!(
            backend_kind_from_name_result("auto", 0, PrecisionProfile::Mixed).unwrap(),
            GAFIME_BACKEND_CPU | GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ));
    }

    #[test]
    fn auto_backend_prefers_configured_usable_gpu_payload() {
        let has_gpu_payload = std::env::var_os(gafime_gpu_sys::CUDA_LIBRARY_ENV).is_some()
            || std::env::var_os(gafime_gpu_sys::ROCM_LIBRARY_ENV).is_some()
            || std::env::var_os(gafime_gpu_sys::METAL_LIBRARY_ENV).is_some();
        if !has_gpu_payload {
            return;
        }

        assert_ne!(
            backend_kind_from_name_result("auto", 0, PrecisionProfile::Fp32).unwrap(),
            GAFIME_BACKEND_CPU
        );
    }

    #[test]
    fn auto_rank_places_gpu_above_cpu_vector_isa() {
        let mut info = GafimeGpuDeviceInfo {
            backend_kind: GAFIME_BACKEND_METAL,
            flags: gafime_types::GAFIME_GPU_DEVICE_FLAG_INTEGRATED
                | gafime_types::GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY
                | gafime_types::GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY,
            total_global_mem_bytes: 8 * 1024 * 1024 * 1024,
            multiprocessor_count: 8,
            compute_major: 1,
            reserved: [0; 8],
            ..Default::default()
        };
        info.reserved[0] = gafime_types::GAFIME_GPU_ARCH_APPLE;

        assert!(gpu_device_score(&info) > cpu_isa_rank(IsaLevel::Avx512));
        assert!(
            cpu_isa_rank(IsaLevel::Avx512) > cpu_isa_rank(IsaLevel::Avx2)
                && cpu_isa_rank(IsaLevel::Avx2) > cpu_isa_rank(IsaLevel::Sse42)
                && cpu_isa_rank(IsaLevel::Sse42) > cpu_isa_rank(IsaLevel::Scalar)
        );
    }

    #[test]
    fn auto_rank_distinguishes_devices_within_same_gpu_architecture() {
        let mut laptop_ada = GafimeGpuDeviceInfo {
            backend_kind: GAFIME_BACKEND_CUDA,
            flags: gafime_types::GAFIME_GPU_DEVICE_FLAG_DISCRETE,
            total_global_mem_bytes: 8 * 1024 * 1024 * 1024,
            multiprocessor_count: 24,
            compute_major: 8,
            compute_minor: 9,
            reserved: [0; 8],
            ..Default::default()
        };
        laptop_ada.reserved[0] = gafime_types::GAFIME_GPU_ARCH_NVIDIA_ADA;
        let mut desktop_ada = laptop_ada;
        desktop_ada.flags |= gafime_types::GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH;
        desktop_ada.total_global_mem_bytes = 24 * 1024 * 1024 * 1024;
        desktop_ada.multiprocessor_count = 128;

        assert!(gpu_device_score(&desktop_ada) > gpu_device_score(&laptop_ada));
    }

    #[test]
    fn rust_config_boundary_rejects_unwired_families() {
        let error = validate_family_flags(true, false).unwrap_err();

        assert!(error.to_string().contains("time-series families"));
    }

    #[test]
    fn significance_seed_uses_python_integer_words_above_u64() {
        assert_eq!(significance_seed_from_words(&[7]), 7);
        assert_eq!(significance_seed_from_words(&[7, 0]), 7);
        assert_ne!(
            significance_seed_from_words(&[7]),
            significance_seed_from_words(&[7, 0, 1])
        );
    }
}
