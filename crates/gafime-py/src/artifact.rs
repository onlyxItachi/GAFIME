use std::cell::RefCell;

use gafime_orchestrator::config::EngineConfig;
use gafime_types::PrecisionProfile;
#[cfg(test)]
use gafime_types::GAFIME_BACKEND_CPU;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};

use crate::common::{
    decode_f32_le, decode_f64_le, validate_shape, ContinuousReport, DecisionPathResultParams,
    OwnedNumericInput, PyBoundaryError,
};
use crate::continuous::{
    build_continuous_state, execute_continuous_state, prepare_screened_continuous_execution,
    CompiledContinuousBackend, ContinuousRunState,
};
#[cfg(feature = "local-cmake-experiment")]
use crate::generated::local_cmake_experiment::CompactDecisionPathState;
use crate::generated::{PrecisionDecisionPathCompiledState, PrecisionDecisionPathRebuild};
use crate::py_api::PyContinuousReport;
use crate::runtime::{
    backend_capability_name_for_kind, backend_device_for_kind, backend_is_gpu,
    backend_name_for_kind, execution_placement_for_kind, parse_python_integer_seed,
    RuntimeCacheCounters,
};

#[cfg(test)]
fn compile_continuous_cpu_rows(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<PyCompiledContinuousArtifact, PyBoundaryError> {
    if config.backend_kind != GAFIME_BACKEND_CPU {
        return Err(PyBoundaryError::InvalidInput(
            "compile_continuous_cpu_rows requires a CPU backend config".to_string(),
        ));
    }
    compile_continuous_rows(config, rows, cols, features, target)
}

#[cfg(test)]
pub(crate) fn compile_continuous_rows(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<PyCompiledContinuousArtifact, PyBoundaryError> {
    let input = OwnedNumericInput::from_f32(config.precision, features, target)?;
    compile_continuous_input(config, rows, cols, input)
}

pub(crate) fn compile_continuous_input(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    input: OwnedNumericInput,
) -> Result<PyCompiledContinuousArtifact, PyBoundaryError> {
    validate_shape(rows, cols, input.feature_len(), input.target_len())?;
    let state = build_continuous_state(&config, rows, cols, input)?;
    let max_arity = state.result_max_arity;
    Ok(PyCompiledContinuousArtifact {
        config: config.clone(),
        rows,
        cols,
        max_arity,
        metric_ids: config.metric_ids.clone(),
        significance_top_n: config.significance_top_n,
        state: Some(state),
        #[cfg(feature = "local-cmake-experiment")]
        local_cmake_experiment_state: None,
        runtime_cache_counters: RefCell::new(RuntimeCacheCounters::default()),
        decision_path_params: Vec::new(),
        decision_path_state: None,
        feature_names: Vec::new(),
        target_updates_supported: true,
        closed: false,
    })
}
pub(crate) fn execute_compiled_artifact(
    artifact: &mut PyCompiledContinuousArtifact,
) -> Result<ContinuousReport, PyBoundaryError> {
    if artifact.closed {
        return Err(PyBoundaryError::InvalidInput(
            "compiled artifact is closed".to_string(),
        ));
    }
    let execution_config = artifact.decision_path_state.as_ref().map_or_else(
        || artifact.config.clone(),
        |state| state.execution_config(&artifact.config),
    );
    let result = (|| {
        #[cfg(feature = "local-cmake-experiment")]
        if let Some(report) = crate::generated::local_cmake_experiment::execute_compiled(artifact)?
        {
            return Ok(report);
        }
        let state = artifact.state.as_mut().ok_or_else(|| {
            PyBoundaryError::InvalidInput("compiled artifact is closed".to_string())
        })?;
        execute_continuous_state(
            &execution_config,
            artifact.rows,
            artifact.cols,
            &artifact.metric_ids,
            artifact.significance_top_n,
            state,
            &artifact.runtime_cache_counters,
        )
    })()
    .and_then(|mut report| {
        if let Some(state) = artifact.decision_path_state.as_ref() {
            state.apply_significance(&artifact.config, &mut report)?;
        }
        Ok(report)
    });
    if result.is_err() && backend_is_gpu(artifact.backend_kind()) {
        artifact.state = None;
        #[cfg(feature = "local-cmake-experiment")]
        {
            artifact.local_cmake_experiment_state = None;
        }
        artifact.closed = true;
    }
    result
}
#[pyclass(name = "CompiledContinuousArtifact", unsendable)]
pub(crate) struct PyCompiledContinuousArtifact {
    pub(crate) config: EngineConfig,
    #[pyo3(get)]
    pub(crate) rows: u64,
    #[pyo3(get)]
    pub(crate) cols: u32,
    #[pyo3(get)]
    pub(crate) max_arity: u32,
    #[pyo3(get)]
    pub(crate) metric_ids: Vec<u32>,
    pub(crate) significance_top_n: u32,
    pub(crate) state: Option<ContinuousRunState>,
    #[cfg(feature = "local-cmake-experiment")]
    pub(crate) local_cmake_experiment_state: Option<CompactDecisionPathState>,
    pub(crate) runtime_cache_counters: RefCell<RuntimeCacheCounters>,
    pub(crate) decision_path_params: Vec<DecisionPathResultParams>,
    pub(crate) decision_path_state: Option<PrecisionDecisionPathCompiledState>,
    pub(crate) feature_names: Vec<String>,
    pub(crate) target_updates_supported: bool,
    pub(crate) closed: bool,
}

impl PyCompiledContinuousArtifact {
    fn backend_kind(&self) -> u32 {
        self.config.backend_kind
    }

    fn uses_fp64_mi_accumulation(&self) -> bool {
        self.config.precision != PrecisionProfile::Fp32
    }

    fn replace_target(&mut self, target: PrecisionTarget) -> PyResult<()> {
        if self.closed {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        }
        if !self.target_updates_supported {
            return Err(PyValueError::new_err(
                "update_target is unsupported for compiled target-derived decision-path features; recompile with the new target",
            ));
        }
        if target.len() as u64 != self.rows {
            return Err(PyValueError::new_err(
                "target length must match the compiled matrix rows",
            ));
        }
        if self.decision_path_state.is_some() {
            return self.rebuild_decision_path_target(target);
        }
        let backend_kind = self.backend_kind();
        let update_result = {
            let Some(state) = self.state.as_mut() else {
                return Err(PyValueError::new_err("compiled artifact is closed"));
            };
            match (&mut state.backend, &target) {
                (CompiledContinuousBackend::Cpu { matrix }, PrecisionTarget::F32(target)) => matrix
                    .replace_target_f32(target.clone())
                    .map_err(PyBoundaryError::from),
                (CompiledContinuousBackend::Cpu { matrix }, PrecisionTarget::F64(target)) => matrix
                    .replace_target_f64(target.clone())
                    .map_err(PyBoundaryError::from),
                (
                    CompiledContinuousBackend::Cuda { matrix, .. }
                    | CompiledContinuousBackend::Rocm { matrix, .. }
                    | CompiledContinuousBackend::Metal { matrix, .. },
                    PrecisionTarget::F32(target),
                ) => matrix
                    .update_target_f32_v2(target)
                    .map_err(PyBoundaryError::from),
                (
                    CompiledContinuousBackend::Cuda { matrix, .. }
                    | CompiledContinuousBackend::Rocm { matrix, .. }
                    | CompiledContinuousBackend::Metal { matrix, .. },
                    PrecisionTarget::F64(target),
                ) => matrix
                    .update_target_f64_v2(target)
                    .map_err(PyBoundaryError::from),
            }
        };
        if let Err(error) = update_result {
            if backend_is_gpu(backend_kind) {
                self.state = None;
                self.closed = true;
            }
            return Err(PyErr::from(error));
        }
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("compiled artifact is closed"))?;
        if let Some(significance_matrix) = &mut state.significance_matrix {
            let replacement = match target {
                PrecisionTarget::F32(target) => significance_matrix.replace_target_f32(target),
                PrecisionTarget::F64(target) => significance_matrix.replace_target_f64(target),
            };
            if let Err(error) = replacement {
                if backend_is_gpu(backend_kind) {
                    self.state = None;
                    self.closed = true;
                }
                return Err(PyErr::from(PyBoundaryError::from(error)));
            }
        }
        let run = match prepare_screened_continuous_execution(
            &self.config,
            self.rows,
            self.cols,
            &state.backend,
        ) {
            Ok(value) => value,
            Err(error) => {
                self.state = None;
                self.closed = true;
                return Err(PyErr::from(error));
            }
        };
        self.max_arity = run.result_max_arity;
        state.replace_plans(run);
        Ok(())
    }

    fn rebuild_decision_path_target(&mut self, target: PrecisionTarget) -> PyResult<()> {
        let mut decision_state = self
            .decision_path_state
            .take()
            .ok_or_else(|| PyValueError::new_err("decision-path rebuild state is missing"))?;
        let current_config = self.config.clone();
        let rebuild = match target {
            PrecisionTarget::F32(target) => {
                decision_state.rebuild_target_f32(&current_config, target)
            }
            PrecisionTarget::F64(target) => {
                decision_state.rebuild_target_f64(&current_config, target)
            }
        };
        match rebuild {
            Ok(rebuild) => self
                .apply_decision_path_rebuild(rebuild, decision_state)
                .map_err(PyErr::from),
            Err(error) => {
                self.decision_path_state = Some(decision_state);
                Err(PyErr::from(error))
            }
        }
    }

    fn apply_decision_path_rebuild(
        &mut self,
        mut rebuild: PrecisionDecisionPathRebuild,
        decision_state: PrecisionDecisionPathCompiledState,
    ) -> Result<(), PyBoundaryError> {
        let rebuilt = &mut rebuild.artifact;
        let decision_path_params = rebuild
            .paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                DecisionPathResultParams::from_precision_path(
                    (decision_state.base_candidate_cols() + index) as u32,
                    rebuilt.config.precision,
                    path,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.config = rebuilt.config.clone();
        self.rows = rebuilt.rows;
        self.cols = rebuilt.cols;
        self.max_arity = rebuilt.max_arity;
        self.metric_ids.clone_from(&rebuilt.metric_ids);
        self.significance_top_n = rebuilt.significance_top_n;
        self.state = rebuilt.state.take();
        #[cfg(feature = "local-cmake-experiment")]
        {
            self.local_cmake_experiment_state = rebuilt.local_cmake_experiment_state.take();
        }
        self.decision_path_params = decision_path_params;
        self.feature_names = rebuild.feature_names;
        self.decision_path_state = Some(decision_state);
        self.target_updates_supported = true;
        self.closed = false;
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum PrecisionTarget {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl PrecisionTarget {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    fn extract(precision: PrecisionProfile, target: &Bound<'_, PyAny>) -> PyResult<Self> {
        match precision {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                Ok(Self::F32(target.extract::<Vec<f32>>()?))
            }
            PrecisionProfile::Fp64 => Ok(Self::F64(target.extract::<Vec<f64>>()?)),
        }
    }

    fn decode(precision: PrecisionProfile, bytes: &[u8]) -> PyResult<Self> {
        match precision {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                Ok(Self::F32(decode_f32_le(bytes, "target")?))
            }
            PrecisionProfile::Fp64 => Ok(Self::F64(decode_f64_le(bytes, "target")?)),
        }
    }
}

#[pymethods]
impl PyCompiledContinuousArtifact {
    #[getter]
    fn backend_name(&self) -> &'static str {
        backend_name_for_kind(self.backend_kind())
    }

    #[getter]
    fn device(&self) -> &'static str {
        backend_device_for_kind(self.backend_kind())
    }

    #[getter]
    fn is_gpu(&self) -> bool {
        backend_is_gpu(self.backend_kind())
    }

    #[getter]
    fn selected_backend(&self) -> &'static str {
        backend_capability_name_for_kind(self.backend_kind())
    }

    #[getter]
    fn execution_placement(&self) -> &'static str {
        execution_placement_for_kind(self.backend_kind())
    }

    #[getter]
    fn precision(&self) -> &'static str {
        precision_name(self.config.precision)
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.feature_names.clone()
    }

    #[getter]
    fn generated_feature_start(&self) -> Option<usize> {
        self.decision_path_state
            .as_ref()
            .map(PrecisionDecisionPathCompiledState::base_candidate_cols)
    }

    #[getter]
    fn storage_dtype(&self) -> &'static str {
        if self.config.precision == PrecisionProfile::Fp64 {
            "float64"
        } else {
            "float32"
        }
    }

    #[getter]
    fn interaction_arithmetic(&self) -> &'static str {
        self.storage_dtype()
    }

    #[getter]
    fn result_dtype(&self) -> &'static str {
        if self.config.precision == PrecisionProfile::Fp32 {
            "float32"
        } else {
            "float64"
        }
    }

    #[getter]
    fn mi_accumulation_dtype(&self) -> &'static str {
        if self.uses_fp64_mi_accumulation() {
            "float64"
        } else {
            "float32"
        }
    }

    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    #[getter]
    fn graph_requested(&self) -> bool {
        self.state
            .as_ref()
            .and_then(|state| state.primary.as_ref())
            .is_some_and(|prepared| prepared.schedule().decision().graph_requested)
    }

    fn analyze(&mut self) -> PyResult<PyContinuousReport> {
        let report = execute_compiled_artifact(self).map_err(PyErr::from)?;
        let mut report = PyContinuousReport::from(report);
        report
            .decision_path_params
            .clone_from(&self.decision_path_params);
        Ok(report)
    }

    #[getter]
    fn continuous_metric_cache_hits(&self) -> u64 {
        self.runtime_cache_counters.borrow().metric_hits
    }

    #[getter]
    fn continuous_metric_cache_builds(&self) -> u64 {
        self.runtime_cache_counters.borrow().metric_builds
    }

    #[getter]
    fn candidate_table_cache_hits(&self) -> u64 {
        self.runtime_cache_counters.borrow().candidate_table_hits
    }

    /// Refresh stochastic planning and significance streams without uploading
    /// the resident feature matrix again. The Python wrapper uses this for
    /// `random_seed=None`, matching the legacy fresh-entropy-per-analysis rule.
    fn reseed(&mut self, seed: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.closed {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        }
        let has_state = self.state.is_some();
        #[cfg(feature = "local-cmake-experiment")]
        let has_state = has_state || self.local_cmake_experiment_state.is_some();
        if !has_state {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        }
        let (random_seed, planning_seed_words) = parse_python_integer_seed(seed)?;
        let mut config = self.config.clone();
        config.random_seed = random_seed;
        config.planning_seed_words = planning_seed_words;
        if let Some(mut decision_state) = self.decision_path_state.take() {
            let rebuild = decision_state.rebuild_current(&config);
            return match rebuild {
                Ok(rebuild) => self
                    .apply_decision_path_rebuild(rebuild, decision_state)
                    .map_err(PyErr::from),
                Err(error) => {
                    self.decision_path_state = Some(decision_state);
                    Err(PyErr::from(error))
                }
            };
        }
        #[cfg(feature = "local-cmake-experiment")]
        if self.local_cmake_experiment_state.is_some() {
            // The compact route is admitted only for the complete unary plan,
            // so reseeding cannot alter its candidate order or score rows.
            self.config = config;
            return Ok(());
        }
        let run_result = match self.state.as_ref() {
            Some(state) => {
                prepare_screened_continuous_execution(&config, self.rows, self.cols, &state.backend)
            }
            None => return Err(PyValueError::new_err("compiled artifact is closed")),
        };
        let run = match run_result {
            Ok(run) => run,
            Err(error) => {
                if backend_is_gpu(self.backend_kind()) {
                    self.state = None;
                    self.closed = true;
                }
                return Err(PyErr::from(error));
            }
        };
        self.max_arity = run.result_max_arity;
        self.config = config;
        let Some(state) = self.state.as_mut() else {
            return Err(PyValueError::new_err("compiled artifact is closed"));
        };
        state.replace_plans(run);
        Ok(())
    }

    /// Resident-session reuse: swap the target in place and re-analyze without
    /// re-uploading the features. On GPU the resident device matrix keeps its
    /// feature buffers and only `y` is refreshed (the permutation/repeat pattern);
    /// on CPU the held matrix's target is replaced. The host significance matrix
    /// (GPU runs) is updated too so a subsequent analyze scores against the new y.
    fn update_target(&mut self, target: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = PrecisionTarget::extract(self.config.precision, target)?;
        self.replace_target(target)
    }

    fn update_target_buffer(&mut self, target: &Bound<'_, PyBytes>) -> PyResult<()> {
        let target = PrecisionTarget::decode(self.config.precision, target.as_bytes())?;
        self.replace_target(target)
    }

    fn close(&mut self) {
        self.state = None;
        self.decision_path_state = None;
        #[cfg(feature = "local-cmake-experiment")]
        {
            self.local_cmake_experiment_state = None;
        }
        self.closed = true;
    }
}

fn precision_name(precision: PrecisionProfile) -> &'static str {
    match precision {
        PrecisionProfile::Fp32 => "fp32",
        PrecisionProfile::Mixed => "mixed",
        PrecisionProfile::Fp64 => "fp64",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{
        GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    };

    use crate::continuous::continuous_config_for_cpu;

    #[test]
    fn explicit_cuda_requires_configured_cabi_payload() {
        if std::env::var_os(gafime_gpu_sys::CUDA_LIBRARY_ENV).is_some() {
            return;
        }
        let mut config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();
        config.backend_kind = GAFIME_BACKEND_CUDA;

        let error = match compile_continuous_rows(
            config,
            4,
            2,
            vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ) {
            Ok(_) => panic!("CUDA compile unexpectedly succeeded without a configured payload"),
            Err(error) => error,
        };

        assert!(error.to_string().contains(gafime_gpu_sys::CUDA_LIBRARY_ENV));
    }

    #[test]
    fn explicit_metal_requires_configured_cabi_payload() {
        if std::env::var_os(gafime_gpu_sys::METAL_LIBRARY_ENV).is_some() {
            return;
        }
        let mut config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();
        config.backend_kind = GAFIME_BACKEND_METAL;
        config.precision = PrecisionProfile::Fp32;

        let error = match compile_continuous_rows(
            config,
            4,
            2,
            vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ) {
            Ok(_) => panic!("Metal compile unexpectedly succeeded without a configured payload"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains(gafime_gpu_sys::METAL_LIBRARY_ENV));
    }

    #[test]
    fn explicit_cuda_executes_when_cabi_payload_is_available() {
        if std::env::var_os(gafime_gpu_sys::CUDA_LIBRARY_ENV).is_none() {
            return;
        }
        let mut config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();
        config.backend_kind = GAFIME_BACKEND_CUDA;
        config.permutation_tests = 0;

        let mut artifact = match compile_continuous_rows(
            config,
            4,
            2,
            vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ) {
            Ok(artifact) => artifact,
            Err(error) => panic!("CUDA compile failed despite configured payload: {error}"),
        };

        assert_eq!(artifact.backend_name(), "v1-cuda-cabi");
        assert_eq!(artifact.device(), "cuda");
        assert!(artifact.is_gpu());
        let report = execute_compiled_artifact(&mut artifact).unwrap();
        assert_eq!(report.len(), 2);
        assert_eq!(report.combo(0).unwrap(), vec![0]);
        assert!((report.metric_values(0).unwrap().as_f64().unwrap()[0] - 1.0).abs() < 1.0e-5);
        assert!((report.metric_values(1).unwrap().as_f64().unwrap()[0] + 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn rust_input_boundary_validates_flat_row_major_dimensions() {
        let config =
            continuous_config_for_cpu(1, 10, vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2])
                .unwrap();

        let error = match compile_continuous_cpu_rows(config, 4, 2, vec![1.0; 7], vec![1.0; 4]) {
            Ok(_) => panic!("invalid shape unexpectedly compiled"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("feature buffer length does not match rows*cols"));
    }

    #[test]
    fn v047_unranked_power_user_plan_above_one_million_rows_is_admitted() {
        const COLS: u32 = 1_450;
        const PAIR_ROWS: u64 = 1_050_525;
        const EXPECTED_ROWS: u64 = 1_051_975;

        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..Default::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = PAIR_ROWS;
        config.budget.top_features_for_higher_k = COLS;
        let features = (0..4)
            .flat_map(|row| std::iter::repeat_n(row as f32, COLS as usize))
            .collect();

        let artifact =
            compile_continuous_cpu_rows(config, 4, COLS, features, vec![0.0, 1.0, 2.0, 3.0])
                .unwrap();
        let state = artifact.state.as_ref().unwrap();

        assert_eq!(PAIR_ROWS, u64::from(COLS) * u64::from(COLS - 1) / 2);
        assert_eq!(state.result_capacity, EXPECTED_ROWS);
        assert_eq!(state.result_max_arity, 2);
    }

    #[test]
    fn pathological_unranked_plan_still_fails_storage_admission() {
        const COLS: u32 = 20_000;
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..Default::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 100_000_000;
        config.budget.top_features_for_higher_k = COLS;
        let features = (0..2)
            .flat_map(|row| std::iter::repeat_n(row as f32, COLS as usize))
            .collect();

        let error = compile_continuous_cpu_rows(config, 2, COLS, features, vec![0.0, 1.0])
            .err()
            .expect("pathological unranked plan must fail storage admission");

        assert!(error
            .to_string()
            .contains("unranked continuous candidate storage exceeds the host-memory budget"));
    }

    #[test]
    fn close_drops_compiled_native_state_immediately() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..Default::default()
        };
        config.budget.max_comb_size = 1;
        let mut artifact =
            compile_continuous_rows(config, 3, 1, vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0])
                .unwrap();

        assert!(artifact.state.is_some());
        artifact.close();

        assert!(artifact.state.is_none());
        assert!(execute_compiled_artifact(&mut artifact).is_err());
    }
}
