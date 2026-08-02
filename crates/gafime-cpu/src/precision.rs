//! Typed Core precision storage and profile contracts.
//!
//! The legacy v1 CPU ABI is `f32`-only.  This module is deliberately separate
//! from that compatibility surface: callers select one [`PrecisionProfile`]
//! once when they construct a resident matrix, after which the storage enum
//! makes an accidental `f64 -> f32 -> f64` round trip impossible for the
//! `fp64` lane.  Structural values (rows, columns, ranks, and candidate ids)
//! remain integers everywhere in this module.

use core::{ffi::c_void, marker::PhantomData, ops::Deref};

use gafime_orchestrator::{MatrixHandle, OrchestratorError, OrchestratorResult};
pub use gafime_types::PrecisionProfile;
use gafime_types::GAFIME_PRECISION_ABI_VERSION;

/// Floating representation used by an individual numeric execution domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuDtype {
    F32,
    F64,
}

/// The four precision decisions which make up the public profile contract.
///
/// Keeping these fields together prevents the old independently configurable
/// storage/compute-policy surface from reappearing inside the CPU backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuPrecisionContract {
    pub ingest_and_storage: CpuDtype,
    pub pointwise_and_interaction: CpuDtype,
    pub reduction_and_statistics: CpuDtype,
    pub ranking_and_public_result: CpuDtype,
}

/// CPU-specific facts derived from the shared public profile enum.
pub trait PrecisionProfileExt {
    /// Return the complete four-domain contract for this profile.
    fn cpu_contract(self) -> CpuPrecisionContract;

    /// A stable profile component for artifact/cache/descriptor keys.
    fn profile_identity(self) -> u32;

    /// `true` when the lane owns `f32` resident columns and targets.
    fn uses_f32_storage(self) -> bool;
}

impl PrecisionProfileExt for PrecisionProfile {
    fn cpu_contract(self) -> CpuPrecisionContract {
        match self {
            PrecisionProfile::Fp32 => CpuPrecisionContract {
                ingest_and_storage: CpuDtype::F32,
                pointwise_and_interaction: CpuDtype::F32,
                reduction_and_statistics: CpuDtype::F32,
                ranking_and_public_result: CpuDtype::F32,
            },
            PrecisionProfile::Mixed => CpuPrecisionContract {
                ingest_and_storage: CpuDtype::F32,
                pointwise_and_interaction: CpuDtype::F32,
                reduction_and_statistics: CpuDtype::F64,
                ranking_and_public_result: CpuDtype::F64,
            },
            PrecisionProfile::Fp64 => CpuPrecisionContract {
                ingest_and_storage: CpuDtype::F64,
                pointwise_and_interaction: CpuDtype::F64,
                reduction_and_statistics: CpuDtype::F64,
                ranking_and_public_result: CpuDtype::F64,
            },
        }
    }

    fn profile_identity(self) -> u32 {
        self as u32
    }

    fn uses_f32_storage(self) -> bool {
        matches!(self, PrecisionProfile::Fp32 | PrecisionProfile::Mixed)
    }
}

/// Owned typed numeric values used by CPU precision-aware APIs.
#[derive(Clone, Debug, PartialEq)]
pub enum CpuPrecisionValues {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// One profile-typed public numerical result.  It is intentionally not
/// `Into<f64>`: widening an fp32 public result is a presentation choice for a
/// caller, not a hidden CPU execution step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CpuPrecisionScalar {
    F32(f32),
    F64(f64),
}

impl CpuPrecisionScalar {
    pub fn dtype(self) -> CpuDtype {
        match self {
            Self::F32(_) => CpuDtype::F32,
            Self::F64(_) => CpuDtype::F64,
        }
    }

    pub fn f32(self) -> Option<f32> {
        match self {
            Self::F32(value) => Some(value),
            Self::F64(_) => None,
        }
    }

    pub fn f64(self) -> Option<f64> {
        match self {
            Self::F32(_) => None,
            Self::F64(value) => Some(value),
        }
    }
}

impl CpuPrecisionValues {
    pub fn dtype(&self) -> CpuDtype {
        match self {
            Self::F32(_) => CpuDtype::F32,
            Self::F64(_) => CpuDtype::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::F64(_) => None,
        }
    }

    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::F32(_) => None,
            Self::F64(values) => Some(values),
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::F64(_) => None,
        }
    }

    pub fn as_f64_mut(&mut self) -> Option<&mut [f64]> {
        match self {
            Self::F32(_) => None,
            Self::F64(values) => Some(values),
        }
    }
}

/// Borrowed typed values used to pass a resident matrix into specialized CPU
/// kernels without allocating or coercing it.
#[derive(Clone, Copy, Debug)]
pub enum CpuPrecisionSlice<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl<'a> CpuPrecisionSlice<'a> {
    pub fn dtype(self) -> CpuDtype {
        match self {
            Self::F32(_) => CpuDtype::F32,
            Self::F64(_) => CpuDtype::F64,
        }
    }

    pub fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(self) -> Option<&'a [f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::F64(_) => None,
        }
    }

    pub fn as_f64(self) -> Option<&'a [f64]> {
        match self {
            Self::F32(_) => None,
            Self::F64(values) => Some(values),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum CpuMatrixStorage {
    F32 {
        columns: Vec<f32>,
        target: Vec<f32>,
        column_means: Vec<f32>,
        column_centered_abs_max: Vec<f32>,
        column_has_nonfinite: Vec<bool>,
        target_has_nonfinite: bool,
    },
    F64 {
        columns: Vec<f64>,
        target: Vec<f64>,
        column_means: Vec<f64>,
        column_centered_abs_max: Vec<f64>,
        column_has_nonfinite: Vec<bool>,
        target_has_nonfinite: bool,
    },
}

/// Resident Core matrix whose storage is fixed by a single public precision
/// profile.  It has no global dtype state and carries its profile identity with
/// the resident allocation so callers can include it in cache/artifact keys.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuPrecisionMatrix {
    profile: PrecisionProfile,
    rows: u64,
    cols: u32,
    storage: CpuMatrixStorage,
}

/// Borrowed native handle for the ABI 1.1 precision path. The profile is
/// embedded in [`MatrixHandle`], so a compiled/resident artifact cannot borrow
/// the same Core allocation under a different profile identity.
pub struct CpuPrecisionMatrixHandle<'a> {
    handle: MatrixHandle,
    _owner: PhantomData<&'a CpuPrecisionMatrix>,
}

impl Deref for CpuPrecisionMatrixHandle<'_> {
    type Target = MatrixHandle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl CpuPrecisionMatrix {
    /// Construct an `fp32` or `mixed` matrix from native `f32` input.
    ///
    /// `fp64` rejects this constructor to make the no-intermediate-fp32
    /// guarantee explicit at the CPU boundary.
    pub fn from_row_major_f32(
        profile: PrecisionProfile,
        rows: u64,
        cols: u32,
        features: Vec<f32>,
        target: Vec<f32>,
    ) -> OrchestratorResult<Self> {
        if !profile.uses_f32_storage() {
            return Err(OrchestratorError::InvalidPlan(
                "fp64 CPU matrix requires f64 input; f32 input would lose precision",
            ));
        }
        let (rows_usize, cols_usize) = validate_shape(rows, cols, features.len(), target.len())?;
        let (columns, column_means, column_centered_abs_max, column_has_nonfinite) = match profile {
            PrecisionProfile::Fp32 => transpose_f32_fp32(rows_usize, cols_usize, &features),
            PrecisionProfile::Mixed => transpose_f32_mixed(rows_usize, cols_usize, &features),
            PrecisionProfile::Fp64 => unreachable!("fp64 was rejected before storage construction"),
        };
        let target_has_nonfinite = target.iter().any(|value| !value.is_finite());
        Ok(Self {
            profile,
            rows,
            cols,
            storage: CpuMatrixStorage::F32 {
                columns,
                target,
                column_means,
                column_centered_abs_max,
                column_has_nonfinite,
                target_has_nonfinite,
            },
        })
    }

    /// Construct a precision matrix from native `f64` input.
    ///
    /// For the intentionally f32-storage profiles this performs the one allowed
    /// ingest conversion.  `fp64` transposes and owns the original binary64
    /// values directly and never materializes an f32 staging buffer.
    pub fn from_row_major_f64(
        profile: PrecisionProfile,
        rows: u64,
        cols: u32,
        features: Vec<f64>,
        target: Vec<f64>,
    ) -> OrchestratorResult<Self> {
        let (rows_usize, cols_usize) = validate_shape(rows, cols, features.len(), target.len())?;
        if profile.uses_f32_storage() {
            let features = features.into_iter().map(|value| value as f32).collect();
            let target = target.into_iter().map(|value| value as f32).collect();
            return Self::from_row_major_f32(profile, rows, cols, features, target);
        }
        let (columns, column_means, column_centered_abs_max, column_has_nonfinite) =
            transpose_f64(rows_usize, cols_usize, &features);
        let target_has_nonfinite = target.iter().any(|value| !value.is_finite());
        Ok(Self {
            profile,
            rows,
            cols,
            storage: CpuMatrixStorage::F64 {
                columns,
                target,
                column_means,
                column_centered_abs_max,
                column_has_nonfinite,
                target_has_nonfinite,
            },
        })
    }

    pub fn profile(&self) -> PrecisionProfile {
        self.profile
    }

    pub fn handle(&self) -> CpuPrecisionMatrixHandle<'_> {
        // SAFETY: the guard borrows `self`, so the pointer cannot outlive the
        // CPU allocation. Shape and profile come from that same allocation.
        let handle = unsafe {
            MatrixHandle::native_with_precision(
                gafime_types::GAFIME_BACKEND_CPU,
                self.profile,
                self as *const Self as *mut c_void,
                self.rows,
                self.cols,
            )
        };
        CpuPrecisionMatrixHandle {
            handle,
            _owner: PhantomData,
        }
    }

    /// # Safety
    ///
    /// A non-null handle advertising ABI 1.1 must have been created by
    /// [`Self::handle`], and the owning precision matrix must remain alive for
    /// the returned borrow. Handles from other ABI generations are accepted as
    /// inputs only so they can be rejected before their pointer is cast.
    pub unsafe fn from_handle(handle: &MatrixHandle) -> OrchestratorResult<&Self> {
        let pointer = Self::checked_handle_ptr(handle)?;
        // SAFETY: checked_handle_ptr rejected null and non-ABI-1.1 handles. The
        // caller upholds the concrete CpuPrecisionMatrix ownership/lifetime
        // invariant documented above.
        let matrix = unsafe { &*pointer };
        if matrix.rows != handle.rows()
            || matrix.cols != handle.cols()
            || matrix.profile != handle.precision()
        {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision matrix handle identity mismatch",
            ));
        }
        Ok(matrix)
    }

    fn checked_handle_ptr(handle: &MatrixHandle) -> OrchestratorResult<*const Self> {
        // Keep host-only handles on their existing null rejection while making
        // cross-trait legacy handles fail before their pointer can be cast.
        if handle.raw().is_null() {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision matrix handle has null pointer",
            ));
        }
        if handle.native_abi_version() != Some(GAFIME_PRECISION_ABI_VERSION) {
            return Err(OrchestratorError::InvalidPlan(
                "precision CPU operation requires an ABI 1.1 matrix handle",
            ));
        }
        Ok(handle.raw().cast::<Self>().cast_const())
    }

    pub fn profile_identity(&self) -> u32 {
        self.profile.profile_identity()
    }

    pub fn contract(&self) -> CpuPrecisionContract {
        self.profile.cpu_contract()
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }

    pub fn storage_dtype(&self) -> CpuDtype {
        match self.storage {
            CpuMatrixStorage::F32 { .. } => CpuDtype::F32,
            CpuMatrixStorage::F64 { .. } => CpuDtype::F64,
        }
    }

    pub fn target(&self) -> CpuPrecisionSlice<'_> {
        match &self.storage {
            CpuMatrixStorage::F32 { target, .. } => CpuPrecisionSlice::F32(target),
            CpuMatrixStorage::F64 { target, .. } => CpuPrecisionSlice::F64(target),
        }
    }

    pub fn column(&self, col: usize) -> CpuPrecisionSlice<'_> {
        let rows = self.rows as usize;
        let start = col * rows;
        match &self.storage {
            CpuMatrixStorage::F32 { columns, .. } => {
                CpuPrecisionSlice::F32(&columns[start..start + rows])
            }
            CpuMatrixStorage::F64 { columns, .. } => {
                CpuPrecisionSlice::F64(&columns[start..start + rows])
            }
        }
    }

    pub fn column_f32(&self, col: usize) -> Option<&[f32]> {
        let rows = self.rows as usize;
        let start = col.checked_mul(rows)?;
        match &self.storage {
            CpuMatrixStorage::F32 { columns, .. } => columns.get(start..start + rows),
            CpuMatrixStorage::F64 { .. } => None,
        }
    }

    pub fn column_f64(&self, col: usize) -> Option<&[f64]> {
        let rows = self.rows as usize;
        let start = col.checked_mul(rows)?;
        match &self.storage {
            CpuMatrixStorage::F32 { .. } => None,
            CpuMatrixStorage::F64 { columns, .. } => columns.get(start..start + rows),
        }
    }

    pub fn target_f32(&self) -> Option<&[f32]> {
        match &self.storage {
            CpuMatrixStorage::F32 { target, .. } => Some(target),
            CpuMatrixStorage::F64 { .. } => None,
        }
    }

    pub fn target_f64(&self) -> Option<&[f64]> {
        match &self.storage {
            CpuMatrixStorage::F32 { .. } => None,
            CpuMatrixStorage::F64 { target, .. } => Some(target),
        }
    }

    pub fn column_mean_f32(&self, col: usize) -> Option<f32> {
        match &self.storage {
            CpuMatrixStorage::F32 { column_means, .. } => column_means.get(col).copied(),
            CpuMatrixStorage::F64 { .. } => None,
        }
    }

    pub fn column_mean_f64(&self, col: usize) -> Option<f64> {
        match &self.storage {
            CpuMatrixStorage::F32 { .. } => None,
            CpuMatrixStorage::F64 { column_means, .. } => column_means.get(col).copied(),
        }
    }

    pub fn column_centered_abs_max_f32(&self, col: usize) -> Option<f32> {
        match &self.storage {
            CpuMatrixStorage::F32 {
                column_centered_abs_max,
                ..
            } => column_centered_abs_max.get(col).copied(),
            CpuMatrixStorage::F64 { .. } => None,
        }
    }

    pub fn column_centered_abs_max_f64(&self, col: usize) -> Option<f64> {
        match &self.storage {
            CpuMatrixStorage::F32 { .. } => None,
            CpuMatrixStorage::F64 {
                column_centered_abs_max,
                ..
            } => column_centered_abs_max.get(col).copied(),
        }
    }

    pub fn column_has_nonfinite(&self, col: usize) -> bool {
        match &self.storage {
            CpuMatrixStorage::F32 {
                column_has_nonfinite,
                ..
            }
            | CpuMatrixStorage::F64 {
                column_has_nonfinite,
                ..
            } => column_has_nonfinite.get(col).copied().unwrap_or(true),
        }
    }

    pub fn target_has_nonfinite(&self) -> bool {
        match &self.storage {
            CpuMatrixStorage::F32 {
                target_has_nonfinite,
                ..
            }
            | CpuMatrixStorage::F64 {
                target_has_nonfinite,
                ..
            } => *target_has_nonfinite,
        }
    }

    /// Replace a resident f32 target.  The operation is intentionally typed so
    /// an fp64 target update cannot pass through a f32 pointer by accident.
    pub fn replace_target_f32(&mut self, target: Vec<f32>) -> OrchestratorResult<()> {
        if target.len() != self.rows as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision target update has invalid length",
            ));
        }
        match &mut self.storage {
            CpuMatrixStorage::F32 {
                target: current,
                target_has_nonfinite,
                ..
            } => {
                *target_has_nonfinite = target.iter().any(|value| !value.is_finite());
                *current = target;
                Ok(())
            }
            CpuMatrixStorage::F64 { .. } => Err(OrchestratorError::InvalidPlan(
                "fp64 CPU matrix requires an f64 target update",
            )),
        }
    }

    /// Replace a resident f64 target.  `fp64` stores it unchanged; the f32
    /// lanes use the explicit ingest-compatible narrowing conversion.
    pub fn replace_target_f64(&mut self, target: Vec<f64>) -> OrchestratorResult<()> {
        if target.len() != self.rows as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision target update has invalid length",
            ));
        }
        match &mut self.storage {
            CpuMatrixStorage::F32 { .. } => {
                self.replace_target_f32(target.into_iter().map(|value| value as f32).collect())
            }
            CpuMatrixStorage::F64 {
                target: current,
                target_has_nonfinite,
                ..
            } => {
                *target_has_nonfinite = target.iter().any(|value| !value.is_finite());
                *current = target;
                Ok(())
            }
        }
    }
}

fn validate_shape(
    rows: u64,
    cols: u32,
    feature_len: usize,
    target_len: usize,
) -> OrchestratorResult<(usize, usize)> {
    if rows == 0 || cols == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision matrix requires non-empty shape",
        ));
    }
    let rows_usize = usize::try_from(rows).map_err(|_| {
        OrchestratorError::InvalidPlan("CPU precision matrix rows exceed host address space")
    })?;
    let cols_usize = usize::try_from(cols).map_err(|_| {
        OrchestratorError::InvalidPlan("CPU precision matrix columns exceed host address space")
    })?;
    let expected_features =
        rows_usize
            .checked_mul(cols_usize)
            .ok_or(OrchestratorError::InvalidPlan(
                "CPU precision matrix shape exceeds host address space",
            ))?;
    if feature_len != expected_features {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision matrix feature buffer has invalid length",
        ));
    }
    if target_len != rows_usize {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision matrix target buffer has invalid length",
        ));
    }
    Ok((rows_usize, cols_usize))
}

fn transpose_f32_fp32(
    rows: usize,
    cols: usize,
    features: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
    let mut columns = vec![0.0f32; rows * cols];
    let mut means = vec![0.0f32; cols];
    let mut minimums = vec![f32::INFINITY; cols];
    let mut maximums = vec![f32::NEG_INFINITY; cols];
    let mut has_nonfinite = vec![false; cols];
    for row in 0..rows {
        for col in 0..cols {
            let value = features[row * cols + col];
            columns[col * rows + row] = value;
            means[col] += value;
            has_nonfinite[col] |= !value.is_finite();
            if value.is_finite() {
                minimums[col] = minimums[col].min(value);
                maximums[col] = maximums[col].max(value);
            }
        }
    }
    for mean in &mut means {
        *mean /= rows as f32;
    }
    let centered_abs_max = (0..cols)
        .map(|col| {
            if has_nonfinite[col] {
                f32::INFINITY
            } else {
                (minimums[col] - means[col])
                    .abs()
                    .max((maximums[col] - means[col]).abs())
            }
        })
        .collect();
    (columns, means, centered_abs_max, has_nonfinite)
}

fn transpose_f32_mixed(
    rows: usize,
    cols: usize,
    features: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
    let mut columns = vec![0.0f32; rows * cols];
    let mut sums = vec![0.0f64; cols];
    let mut minimums = vec![f32::INFINITY; cols];
    let mut maximums = vec![f32::NEG_INFINITY; cols];
    let mut has_nonfinite = vec![false; cols];
    for row in 0..rows {
        for col in 0..cols {
            let value = features[row * cols + col];
            columns[col * rows + row] = value;
            sums[col] += value as f64;
            has_nonfinite[col] |= !value.is_finite();
            if value.is_finite() {
                minimums[col] = minimums[col].min(value);
                maximums[col] = maximums[col].max(value);
            }
        }
    }
    let means: Vec<f32> = sums
        .into_iter()
        .map(|sum| (sum / rows as f64) as f32)
        .collect();
    let centered_abs_max = (0..cols)
        .map(|col| {
            if has_nonfinite[col] {
                f32::INFINITY
            } else {
                (minimums[col] - means[col])
                    .abs()
                    .max((maximums[col] - means[col]).abs())
            }
        })
        .collect();
    (columns, means, centered_abs_max, has_nonfinite)
}

fn transpose_f64(
    rows: usize,
    cols: usize,
    features: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut columns = vec![0.0f64; rows * cols];
    let mut means = vec![0.0f64; cols];
    let mut minimums = vec![f64::INFINITY; cols];
    let mut maximums = vec![f64::NEG_INFINITY; cols];
    let mut has_nonfinite = vec![false; cols];
    for row in 0..rows {
        for col in 0..cols {
            let value = features[row * cols + col];
            columns[col * rows + row] = value;
            means[col] += value;
            has_nonfinite[col] |= !value.is_finite();
            if value.is_finite() {
                minimums[col] = minimums[col].min(value);
                maximums[col] = maximums[col].max(value);
            }
        }
    }
    for mean in &mut means {
        *mean /= rows as f64;
    }
    let centered_abs_max = (0..cols)
        .map(|col| {
            if has_nonfinite[col] {
                f64::INFINITY
            } else {
                (minimums[col] - means[col])
                    .abs()
                    .max((maximums[col] - means[col]).abs())
            }
        })
        .collect();
    (columns, means, centered_abs_max, has_nonfinite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::CpuMatrix;

    #[test]
    fn profile_contracts_are_closed_and_cover_all_four_domains() {
        assert_eq!(
            PrecisionProfile::Fp32.cpu_contract(),
            CpuPrecisionContract {
                ingest_and_storage: CpuDtype::F32,
                pointwise_and_interaction: CpuDtype::F32,
                reduction_and_statistics: CpuDtype::F32,
                ranking_and_public_result: CpuDtype::F32,
            }
        );
        assert_eq!(
            PrecisionProfile::Mixed.cpu_contract(),
            CpuPrecisionContract {
                ingest_and_storage: CpuDtype::F32,
                pointwise_and_interaction: CpuDtype::F32,
                reduction_and_statistics: CpuDtype::F64,
                ranking_and_public_result: CpuDtype::F64,
            }
        );
        assert_eq!(
            PrecisionProfile::Fp64.cpu_contract(),
            CpuPrecisionContract {
                ingest_and_storage: CpuDtype::F64,
                pointwise_and_interaction: CpuDtype::F64,
                reduction_and_statistics: CpuDtype::F64,
                ranking_and_public_result: CpuDtype::F64,
            }
        );
    }

    #[test]
    fn fp64_matrix_preserves_adjacent_binary64_values_without_f32_staging() {
        let base = 1.0f64;
        let next = f64::from_bits(base.to_bits() + 1);
        assert_eq!(
            base as f32, next as f32,
            "oracle input must collapse in fp32"
        );

        let matrix = CpuPrecisionMatrix::from_row_major_f64(
            PrecisionProfile::Fp64,
            2,
            1,
            vec![base, next],
            vec![0.0, 1.0],
        )
        .unwrap();
        let column = matrix.column_f64(0).unwrap();
        assert_eq!(column[0].to_bits(), base.to_bits());
        assert_eq!(column[1].to_bits(), next.to_bits());
        assert_ne!(column[0].to_bits(), column[1].to_bits());
        assert!(matrix.column_f32(0).is_none());
    }

    #[test]
    fn fp64_rejects_f32_resident_input_before_allocation() {
        let error = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Fp64,
            1,
            1,
            vec![1.0],
            vec![1.0],
        )
        .unwrap_err();
        assert_eq!(
            error,
            OrchestratorError::InvalidPlan(
                "fp64 CPU matrix requires f64 input; f32 input would lose precision"
            )
        );
    }

    #[test]
    fn target_updates_are_typed_and_keep_fp64_bits() {
        let mut matrix = CpuPrecisionMatrix::from_row_major_f64(
            PrecisionProfile::Fp64,
            2,
            1,
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        )
        .unwrap();
        let exact = f64::from_bits(0x3ff0_0000_0000_0001);
        matrix.replace_target_f64(vec![exact, 2.0]).unwrap();
        assert_eq!(matrix.target_f64().unwrap()[0].to_bits(), exact.to_bits());
        assert!(matrix.replace_target_f32(vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn fp32_mean_never_uses_hidden_f64_reduction() {
        let values = vec![16_777_216.0f32, 1.0, -16_777_216.0];
        let fp32 = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Fp32,
            3,
            1,
            values.clone(),
            vec![0.0; 3],
        )
        .unwrap();
        let mixed = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Mixed,
            3,
            1,
            values,
            vec![0.0; 3],
        )
        .unwrap();
        assert_eq!(fp32.column_mean_f32(0), Some(0.0));
        assert_eq!(mixed.column_mean_f32(0), Some(1.0 / 3.0));
    }

    #[test]
    fn precision_handle_guard_rejects_legacy_generation_before_cast() {
        let legacy_matrix =
            CpuMatrix::from_row_major(2, 1, vec![1.0, 2.0], vec![1.0, 2.0]).unwrap();
        let legacy_handle = legacy_matrix.handle();

        assert_eq!(
            legacy_handle.native_abi_version(),
            Some(gafime_types::GAFIME_ABI_VERSION)
        );
        assert_eq!(
            CpuPrecisionMatrix::checked_handle_ptr(&legacy_handle).unwrap_err(),
            OrchestratorError::InvalidPlan(
                "precision CPU operation requires an ABI 1.1 matrix handle"
            )
        );
    }

    #[test]
    fn precision_handle_guard_preserves_native_and_host_null_behavior() {
        let matrix = CpuPrecisionMatrix::from_row_major_f64(
            PrecisionProfile::Fp64,
            2,
            1,
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        )
        .unwrap();
        let handle = matrix.handle();
        assert_eq!(
            handle.native_abi_version(),
            Some(GAFIME_PRECISION_ABI_VERSION)
        );
        assert_eq!(
            CpuPrecisionMatrix::checked_handle_ptr(&handle).unwrap(),
            &matrix as *const CpuPrecisionMatrix
        );

        let host = MatrixHandle::host_with_precision(
            gafime_types::GAFIME_BACKEND_CPU,
            PrecisionProfile::Fp64,
            2,
            1,
        );
        assert_eq!(host.native_abi_version(), None);
        assert_eq!(
            CpuPrecisionMatrix::checked_handle_ptr(&host).unwrap_err(),
            OrchestratorError::InvalidPlan("CPU precision matrix handle has null pointer")
        );
    }
}
