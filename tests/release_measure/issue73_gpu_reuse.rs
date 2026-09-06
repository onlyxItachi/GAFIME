//! Narrow test-only reuse of the mixed continuous scorer for issue #73.
//!
//! A caller has already materialized candidates as a row-major `f32` matrix.
//! This wrapper sends those columns through the existing unary Pearson plan.
//! For an unlabeled anchor lane, `target` is mechanically the current ABI's
//! required row-aligned reference buffer; it does not make the CUDA ABI a
//! target-free evidence protocol.  Sparse labels require a separate compact
//! matrix/target pair before calling `prepare`.
//!
//! `prepare` owns loader/setup, plan construction, allocation, upload, and the
//! typed result container.  `execute_resident_in_place` is the separately
//! measurable resident region.  `output_values` is intentionally separate so
//! result decoding is not included in a kernel timing interval.
//!
//! The contained `unsafe` call follows the established prepared-execution
//! boundary: this type owns the matrix, immutable protocol, and correctly sized
//! mixed (`f64`) result table through the synchronous call.  Raw native zero for
//! a constant input remains raw output, not an issue-73 quality decision; the
//! fixture retains its own unavailable/constant mask.  Paired-view and
//! edge-based graph evidence deliberately do not route through this wrapper.

use std::collections::HashSet;

use gafime_cpu::{precision::CpuPrecisionMatrix, result::PrecisionOwnedResultTable, CpuBackend};
use gafime_gpu_sys::{GpuBackend, OwnedGpuMatrix};
use gafime_orchestrator::{
    config::EngineConfig, prepare_continuous_execution,
    prepare_continuous_execution_for_feature_orders, PreparedContinuousExecution,
};
use gafime_types::{
    PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_METRIC_PEARSON,
};

/// Payload identity for the experiment record.  The harness, rather than this
/// module, hashes `library_path` and records its own environment evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendIdentity {
    pub backend: String,
    pub device_id: Option<u32>,
    pub library_path: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentExecution {
    pub launched_chunks: u64,
    pub graph_replays: u64,
    pub rows_written: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendFlavor {
    Core,
    Cuda,
}

impl BackendFlavor {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "core" => Ok(Self::Core),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "unsupported issue73 backend {other:?}; expected core or cuda"
            )),
        }
    }

    fn native_kind(self) -> u32 {
        match self {
            Self::Core => GAFIME_BACKEND_CPU,
            Self::Cuda => GAFIME_BACKEND_CUDA,
        }
    }
}

enum ResidentMatrix {
    Core(CpuPrecisionMatrix),
    Cuda {
        backend: Box<GpuBackend>,
        matrix: OwnedGpuMatrix,
    },
}

/// Safe owner for a fully resident existing mixed/Pearson plan.
pub struct Issue73UnaryScorer {
    prepared: PreparedContinuousExecution,
    result: PrecisionOwnedResultTable,
    resident: ResidentMatrix,
    expected_combinations: Vec<Vec<u32>>,
    identity: BackendIdentity,
    last_execution: Option<ResidentExecution>,
}

impl Issue73UnaryScorer {
    /// Prepare the existing unary continuous Pearson scorer over columns `0..cols`.
    /// Output from [`Self::execute_resident`] is exactly this column order.
    pub fn prepare(
        backend: &str,
        rows: usize,
        cols: usize,
        features: &[f32],
        target: &[f32],
    ) -> Result<Self, String> {
        let flavor = BackendFlavor::parse(backend)?;
        let (rows, cols) = validate_matrix_shape(rows, cols, features, target)?;
        let config = unary_config(flavor, cols);
        let prepared = prepare_continuous_execution(&config, rows, cols)
            .map_err(|error| format!("prepare unary continuous plan: {error:?}"))?;
        let expected_combinations = (0..cols).map(|column| vec![column]).collect();
        Self::from_prepared(
            flavor,
            rows,
            cols,
            features,
            target,
            prepared,
            expected_combinations,
        )
    }

    /// Prepare the existing arity-two centered-product path over all pairs of
    /// `source_columns`, in the supplied-list combination order.  This is only
    /// for bounded direct-versus-materialized comparisons; it is not a general
    /// candidate-program adapter.
    pub fn prepare_existing_centered_products(
        backend: &str,
        rows: usize,
        cols: usize,
        features: &[f32],
        target: &[f32],
        source_columns: &[u32],
    ) -> Result<Self, String> {
        let flavor = BackendFlavor::parse(backend)?;
        let (rows, cols) = validate_matrix_shape(rows, cols, features, target)?;
        validate_product_columns(source_columns, cols)?;
        let expected_combinations = pair_combinations(source_columns);
        let mut config = base_config(flavor);
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = u64::try_from(expected_combinations.len())
            .map_err(|_| "centered-product count exceeds u64")?;
        let prepared = prepare_continuous_execution_for_feature_orders(
            &config,
            rows,
            cols,
            &[],
            source_columns,
            false,
            false,
        )
        .map_err(|error| format!("prepare centered-product plan: {error:?}"))?;
        Self::from_prepared(
            flavor,
            rows,
            cols,
            features,
            target,
            prepared,
            expected_combinations,
        )
    }

    fn from_prepared(
        flavor: BackendFlavor,
        rows: u64,
        cols: u32,
        features: &[f32],
        target: &[f32],
        prepared: PreparedContinuousExecution,
        expected_combinations: Vec<Vec<u32>>,
    ) -> Result<Self, String> {
        let expected_rows = u64::try_from(expected_combinations.len())
            .map_err(|_| "expected result count exceeds u64")?;
        if prepared.result_capacity() != expected_rows {
            return Err(format!(
                "prepared plan has {} result rows, expected {expected_rows}",
                prepared.result_capacity()
            ));
        }
        let result = PrecisionOwnedResultTable::new(
            PrecisionProfile::Mixed,
            prepared.result_capacity(),
            prepared.result_max_arity(),
            prepared.result_metric_count(),
        );
        let (resident, identity) = match flavor {
            BackendFlavor::Core => (
                ResidentMatrix::Core(
                    CpuPrecisionMatrix::from_row_major_f32(
                        PrecisionProfile::Mixed,
                        rows,
                        cols,
                        features.to_vec(),
                        target.to_vec(),
                    )
                    .map_err(|error| format!("allocate Core mixed matrix: {error:?}"))?,
                ),
                BackendIdentity {
                    backend: "core".to_owned(),
                    device_id: None,
                    library_path: None,
                },
            ),
            BackendFlavor::Cuda => {
                let gpu = GpuBackend::cuda_from_env(0)
                    .map_err(|error| format!("load CUDA payload: {error}"))?;
                if !gpu
                    .supports_precision(PrecisionProfile::Mixed)
                    .map_err(|error| format!("query CUDA mixed route: {error}"))?
                {
                    return Err(
                        "configured CUDA payload does not support mixed precision".to_owned()
                    );
                }
                let library_path = gpu
                    .loaded_library_path()
                    .map(|path| path.display().to_string());
                let matrix = gpu
                    .alloc_matrix_for_profile(PrecisionProfile::Mixed, rows, cols)
                    .map_err(|error| format!("allocate CUDA mixed matrix: {error}"))?;
                matrix
                    .upload_f32_v2(features, target)
                    .map_err(|error| format!("upload CUDA mixed matrix: {error}"))?;
                (
                    ResidentMatrix::Cuda {
                        backend: Box::new(gpu),
                        matrix,
                    },
                    BackendIdentity {
                        backend: "cuda".to_owned(),
                        device_id: Some(0),
                        library_path,
                    },
                )
            }
        };
        Ok(Self {
            prepared,
            result,
            resident,
            expected_combinations,
            identity,
            last_execution: None,
        })
    }

    pub fn backend_identity(&self) -> &BackendIdentity {
        &self.identity
    }

    pub fn last_execution(&self) -> Option<ResidentExecution> {
        self.last_execution
    }

    /// Execute only the resident plan and preallocated result owner.
    pub fn execute_resident_in_place(&mut self) -> Result<(), String> {
        let (prepared, result, resident) = (&self.prepared, &mut self.result, &mut self.resident);
        let raw = result
            .f64_mut()
            .ok_or_else(|| "mixed scorer has no f64 result table".to_owned())?;
        raw.row_count = 0;
        raw.flags = 0;
        let stats = match resident {
            ResidentMatrix::Core(matrix) => {
                let mut backend = CpuBackend;
                let handle = matrix.handle();
                // SAFETY: `prepared` owns the protocol spans; `handle` borrows
                // the live matrix; and `raw` is a re-bound owner sized from the
                // prepared result capacity for this synchronous call.
                unsafe { prepared.execute_precision_f64(&mut backend, &handle, raw) }
            }
            ResidentMatrix::Cuda { backend, matrix } => {
                // SAFETY: `prepared` owns protocol spans, `matrix` owns the
                // ABI-1.1 allocation, and `raw` owns typed output storage while
                // `backend` is uniquely borrowed for this synchronous call.
                unsafe { prepared.execute_precision_f64(backend.as_mut(), matrix.handle(), raw) }
            }
        }
        .map_err(|error| format!("execute resident mixed continuous plan: {error:?}"))?;
        self.last_execution = Some(ResidentExecution {
            launched_chunks: stats.launched_chunks,
            graph_replays: stats.graph_replays,
            rows_written: stats.rows_written,
        });
        Ok(())
    }

    /// Decode the prior result in the requested column/combination order.
    pub fn output_values(&self) -> Result<Vec<f64>, String> {
        if self.last_execution.is_none() {
            return Err("resident scorer has not executed".to_owned());
        }
        let table = self
            .result
            .as_f64()
            .ok_or_else(|| "mixed scorer lost its f64 result table".to_owned())?;
        if table.row_count() != self.expected_combinations.len() || table.metric_count() != 1 {
            return Err(format!(
                "native result shape is rows={} metrics={}, expected rows={} metrics=1",
                table.row_count(),
                table.metric_count(),
                self.expected_combinations.len()
            ));
        }
        let arity = table.max_arity();
        let combos = table.combo_indices();
        for (row, expected) in self.expected_combinations.iter().enumerate() {
            let start = row
                .checked_mul(arity)
                .ok_or_else(|| "native result combo offset overflows usize".to_owned())?;
            let actual = combos[start..start + expected.len()].to_vec();
            if actual != *expected
                || combos[start + expected.len()..start + arity]
                    .iter()
                    .any(|&value| value != u32::MAX)
            {
                return Err(format!(
                    "native result combination at row {row} differs from requested order"
                ));
            }
        }
        Ok(table.metric_values()[..table.row_count()].to_vec())
    }

    /// Convenience form for correctness checks outside a timed resident region.
    pub fn execute_resident(&mut self) -> Result<Vec<f64>, String> {
        self.execute_resident_in_place()?;
        self.output_values()
    }
}

fn base_config(flavor: BackendFlavor) -> EngineConfig {
    // Command capture is not data-neighbor graph evidence, and a later target
    // replacement would invalidate CUDA capture anyway.
    EngineConfig {
        precision: PrecisionProfile::Mixed,
        backend_kind: flavor.native_kind(),
        device_id: 0,
        metric_ids: vec![GAFIME_METRIC_PEARSON],
        permutation_tests: 0,
        num_repeats: 1,
        graph_requested: false,
        ..EngineConfig::default()
    }
}

fn unary_config(flavor: BackendFlavor, cols: u32) -> EngineConfig {
    let mut config = base_config(flavor);
    config.budget.max_comb_size = 1;
    config.budget.max_combinations_per_k = u64::from(cols);
    config.budget.max_feature_candidate = i64::from(cols);
    config
}

fn validate_matrix_shape(
    rows: usize,
    cols: usize,
    features: &[f32],
    target: &[f32],
) -> Result<(u64, u32), String> {
    if rows == 0 || cols == 0 {
        return Err("resident matrix requires non-zero rows and columns".to_owned());
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| "resident feature shape overflows usize".to_owned())?;
    if features.len() != expected || target.len() != rows {
        return Err(format!(
            "resident shape requires {expected} features and {rows} target values, got {} and {}",
            features.len(),
            target.len()
        ));
    }
    Ok((
        u64::try_from(rows).map_err(|_| "row count exceeds u64")?,
        u32::try_from(cols).map_err(|_| "column count exceeds u32")?,
    ))
}

fn validate_product_columns(source_columns: &[u32], cols: u32) -> Result<(), String> {
    if source_columns.len() < 2 {
        return Err("centered-product comparison needs at least two source columns".to_owned());
    }
    let mut seen = HashSet::with_capacity(source_columns.len());
    for &column in source_columns {
        if column >= cols || !seen.insert(column) {
            return Err(format!("invalid centered-product source column {column}"));
        }
    }
    Ok(())
}

fn pair_combinations(source_columns: &[u32]) -> Vec<Vec<u32>> {
    let mut pairs = Vec::with_capacity(source_columns.len() * (source_columns.len() - 1) / 2);
    for (left_index, &left) in source_columns.iter().enumerate() {
        for &right in &source_columns[left_index + 1..] {
            pairs.push(vec![left, right]);
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_shape_before_any_backend_load() {
        assert!(Issue73UnaryScorer::prepare("cuda", 2, 2, &[1.0], &[0.0, 1.0]).is_err());
    }
}
