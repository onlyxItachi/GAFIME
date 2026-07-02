use core::ffi::c_void;

use gafime_orchestrator::{MatrixHandle, OrchestratorError, OrchestratorResult};
use gafime_types::GAFIME_BACKEND_CPU;

#[derive(Clone, Debug, PartialEq)]
pub struct CpuMatrix {
    rows: u64,
    cols: u32,
    columns: Vec<f32>,
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
        let (columns, column_means) = transpose_row_major(rows, cols, &features);
        Ok(Self {
            rows,
            cols,
            columns,
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

    /// Replace the target vector in place (resident-session reuse): the feature
    /// columns and their means are kept, only `y` changes. Length must match rows.
    pub fn set_target(&mut self, target: Vec<f32>) -> OrchestratorResult<()> {
        if target.len() != self.rows as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU matrix target update has invalid length",
            ));
        }
        self.target = target;
        Ok(())
    }

    pub fn value(&self, row: usize, col: usize) -> f32 {
        self.columns[col * self.rows as usize + row]
    }

    pub fn column(&self, col: usize) -> &[f32] {
        let rows = self.rows as usize;
        let start = col * rows;
        &self.columns[start..start + rows]
    }

    pub fn column_mean(&self, col: usize) -> f32 {
        self.column_means[col]
    }
}

fn transpose_row_major(rows: u64, cols: u32, features: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let rows = rows as usize;
    let cols = cols as usize;
    let mut columns = vec![0.0f32; rows * cols];
    let mut means = vec![0.0f64; cols as usize];
    for row in 0..rows {
        for col in 0..cols {
            let value = features[row * cols + col];
            columns[col * rows + row] = value;
            means[col] += value as f64;
        }
    }
    let means = means
        .into_iter()
        .map(|sum| (sum / rows as f64) as f32)
        .collect();
    (columns, means)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_input_is_transposed_to_column_major_storage() {
        let matrix = CpuMatrix::from_row_major(
            3,
            2,
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();

        assert_eq!(matrix.column(0), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.column(1), &[10.0, 20.0, 30.0]);
        assert_eq!(matrix.value(2, 1), 30.0);
        assert_eq!(matrix.column_mean(0), 2.0);
        assert_eq!(matrix.column_mean(1), 20.0);
    }
}
