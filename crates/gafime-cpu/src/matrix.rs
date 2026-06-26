use core::ffi::c_void;

use gafime_orchestrator::{MatrixHandle, OrchestratorError, OrchestratorResult};
use gafime_types::GAFIME_BACKEND_CPU;

#[derive(Clone, Debug, PartialEq)]
pub struct CpuMatrix {
    rows: u64,
    cols: u32,
    features: Vec<f32>,
    target: Vec<f32>,
    column_means: Vec<f32>,
}

impl CpuMatrix {
    pub fn from_row_major(
        rows: u64,
        cols: u32,
        features: Vec<f32>,
        target: Vec<f32>,
    ) -> OrchestratorResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(OrchestratorError::InvalidPlan(
                "CPU matrix requires non-empty shape",
            ));
        }
        if features.len() != rows as usize * cols as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU matrix feature buffer has invalid length",
            ));
        }
        if target.len() != rows as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU matrix target buffer has invalid length",
            ));
        }
        let column_means = compute_column_means(rows, cols, &features);
        Ok(Self {
            rows,
            cols,
            features,
            target,
            column_means,
        })
    }

    pub fn handle(&self) -> MatrixHandle {
        MatrixHandle::native(
            GAFIME_BACKEND_CPU,
            self as *const Self as *mut c_void,
            self.rows,
            self.cols,
        )
    }

    pub unsafe fn from_handle<'a>(handle: &MatrixHandle) -> OrchestratorResult<&'a CpuMatrix> {
        if handle.raw().is_null() {
            return Err(OrchestratorError::InvalidPlan(
                "CPU matrix handle has null pointer",
            ));
        }
        let matrix = &*(handle.raw() as *const CpuMatrix);
        if matrix.rows != handle.rows() || matrix.cols != handle.cols() {
            return Err(OrchestratorError::InvalidPlan(
                "CPU matrix handle shape mismatch",
            ));
        }
        Ok(matrix)
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }

    pub fn target(&self) -> &[f32] {
        &self.target
    }

    pub fn value(&self, row: usize, col: usize) -> f32 {
        self.features[row * self.cols as usize + col]
    }

    pub fn column_mean(&self, col: usize) -> f32 {
        self.column_means[col]
    }
}

fn compute_column_means(rows: u64, cols: u32, features: &[f32]) -> Vec<f32> {
    let mut means = vec![0.0f64; cols as usize];
    for row in 0..rows as usize {
        for col in 0..cols as usize {
            means[col] += features[row * cols as usize + col] as f64;
        }
    }
    means
        .into_iter()
        .map(|sum| (sum / rows as f64) as f32)
        .collect()
}
