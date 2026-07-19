use core::{ffi::c_void, marker::PhantomData, ops::Deref};

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

/// Borrowed native handle whose lifetime is tied to the owning CPU matrix.
///
/// The inner pointer is consumed only by the CPU backend interconnect. Exposing
/// it through `Deref` preserves the common backend API without allowing safe
/// code to retain a handle after the matrix has been dropped.
pub struct CpuMatrixHandle<'a> {
    handle: MatrixHandle,
    _owner: PhantomData<&'a CpuMatrix>,
}

impl Deref for CpuMatrixHandle<'_> {
    type Target = MatrixHandle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
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
        let rows_usize = usize::try_from(rows).map_err(|_| {
            OrchestratorError::InvalidPlan("CPU matrix rows exceed host address space")
        })?;
        let cols_usize = usize::try_from(cols).map_err(|_| {
            OrchestratorError::InvalidPlan("CPU matrix columns exceed host address space")
        })?;
        let feature_len =
            rows_usize
                .checked_mul(cols_usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "CPU matrix shape exceeds host address space",
                ))?;
        if features.len() != feature_len {
            return Err(OrchestratorError::InvalidPlan(
                "CPU matrix feature buffer has invalid length",
            ));
        }
        if target.len() != rows_usize {
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

    pub fn handle(&self) -> CpuMatrixHandle<'_> {
        // SAFETY: the guard borrows `self`, so the raw pointer cannot outlive the
        // matrix. Rows and columns come directly from the same allocation.
        let handle = unsafe {
            MatrixHandle::native(
                GAFIME_BACKEND_CPU,
                self as *const Self as *mut c_void,
                self.rows,
                self.cols,
            )
        };
        CpuMatrixHandle {
            handle,
            _owner: PhantomData,
        }
    }

    /// # Safety
    ///
    /// `handle` must have been created by `CpuMatrix::handle` and the borrowed
    /// matrix must remain alive for the returned reference.
    pub unsafe fn from_handle(handle: &MatrixHandle) -> OrchestratorResult<&CpuMatrix> {
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
    let mut means = vec![0.0f64; cols];
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

    #[test]
    fn row_major_shape_overflow_is_rejected() {
        let error = CpuMatrix::from_row_major(1u64 << 63, 2, Vec::new(), Vec::new()).unwrap_err();
        assert_eq!(
            error,
            OrchestratorError::InvalidPlan("CPU matrix shape exceeds host address space")
        );
    }
}
