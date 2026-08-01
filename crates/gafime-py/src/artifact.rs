use std::cell::RefCell;

use gafime_orchestrator::config::EngineConfig;
use gafime_types::GAFIME_BACKEND_CPU;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};

use crate::common::{
    decode_f32_le, validate_shape, ContinuousReport, DecisionPathResultParams, PyBoundaryError,
};
use crate::continuous::{
    build_continuous_state, execute_continuous_state, prepare_screened_continuous_execution,
    CompiledContinuousBackend, ContinuousRunState,
};
#[cfg(feature = "local-cmake-experiment")]
use crate::generated::local_cmake_experiment::CompactDecisionPathState;
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

pub(crate) fn compile_continuous_rows(
    config: EngineConfig,
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
) -> Result<PyCompiledContinuousArtifact, PyBoundaryError> {
    validate_shape(rows, cols, features.len(), target.len())?;
    let state = build_continuous_state(&config, rows, cols, features, target)?;
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
            &artifact.config,
            artifact.rows,
            artifact.cols,
            &artifact.metric_ids,
            artifact.significance_top_n,
            state,
            &artifact.runtime_cache_counters,
        )
    })();
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
    pub(crate) target_updates_supported: bool,
    pub(crate) closed: bool,
}

impl PyCompiledContinuousArtifact {
    fn backend_kind(&self) -> u32 {
        self.config.backend_kind
    }

    fn uses_fp64_mi_accumulation(&self) -> bool {
        let accumulation = self
            .state
            .as_ref()
            .map(|state| state.backend.uses_fp64_mi_accumulation());
        #[cfg(feature = "local-cmake-experiment")]
        let accumulation = accumulation.or_else(|| {
            self.local_cmake_experiment_state
                .as_ref()
                .map(CompactDecisionPathState::uses_fp64_mi_accumulation)
        });
        accumulation.unwrap_or(self.backend_kind() == GAFIME_BACKEND_CPU)
    }

    fn replace_target(&mut self, target: Vec<f32>) -> PyResult<()> {
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
        let backend_kind = self.backend_kind();
        let update_result = {
            let Some(state) = self.state.as_mut() else {
                return Err(PyValueError::new_err("compiled artifact is closed"));
            };
            match &mut state.backend {
                CompiledContinuousBackend::Cpu { matrix } => matrix
                    .set_target(target.clone())
                    .map_err(PyBoundaryError::from),
                CompiledContinuousBackend::Cuda { matrix, .. }
                | CompiledContinuousBackend::Rocm { matrix, .. }
                | CompiledContinuousBackend::Metal { matrix, .. } => {
                    matrix.update_target(&target).map_err(PyBoundaryError::from)
                }
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
            if let Err(error) = significance_matrix.set_target(target) {
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
    fn storage_dtype(&self) -> &'static str {
        "float32"
    }

    #[getter]
    fn compute_policy(&self) -> &'static str {
        "stable"
    }

    #[getter]
    fn interaction_arithmetic(&self) -> &'static str {
        "float32"
    }

    #[getter]
    fn result_dtype(&self) -> &'static str {
        "float32"
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
    fn update_target(&mut self, target: Vec<f32>) -> PyResult<()> {
        self.replace_target(target)
    }

    fn update_target_buffer(&mut self, target: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.replace_target(decode_f32_le(target.as_bytes(), "target")?)
    }

    fn close(&mut self) {
        self.state = None;
        #[cfg(feature = "local-cmake-experiment")]
        {
            self.local_cmake_experiment_state = None;
        }
        self.closed = true;
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
        assert!((report.metric_values(0).unwrap()[0] - 1.0).abs() < 1.0e-5);
        assert!((report.metric_values(1).unwrap()[0] + 1.0).abs() < 1.0e-5);
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
