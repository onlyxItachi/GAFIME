//! Immutable, profile-compatible column storage for semantic contexts.
//!
//! `Arc<Vec<_>>` deliberately retains the vector allocation, so a source
//! materialization can share it instead of copying through `Arc<[T]>`.

use std::sync::Arc;

use gafime_types::PrecisionProfile;

use super::{SemanticError, SemanticResult};

/// Immutable numeric storage in the only two source representations used by
/// the bounded Core semantic slice.
#[derive(Clone, Debug)]
pub enum NumericColumn {
    F32(Arc<Vec<f32>>),
    F64(Arc<Vec<f64>>),
}

impl From<Vec<f32>> for NumericColumn {
    fn from(values: Vec<f32>) -> Self {
        Self::F32(Arc::new(values))
    }
}

impl From<Vec<f64>> for NumericColumn {
    fn from(values: Vec<f64>) -> Self {
        Self::F64(Arc::new(values))
    }
}

impl NumericColumn {
    /// Clone only the immutable ownership handle, never the numeric payload.
    pub fn shared_clone(&self) -> Self {
        self.clone()
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

    /// Saturation makes an impossible address-space overflow fail a caller's
    /// bounded-byte comparison rather than wrapping to a small allocation.
    pub fn bytes(&self) -> usize {
        let width = match self {
            Self::F32(_) => std::mem::size_of::<f32>(),
            Self::F64(_) => std::mem::size_of::<f64>(),
        };
        self.len().saturating_mul(width)
    }

    pub fn finite(&self) -> bool {
        match self {
            Self::F32(values) => values.iter().all(|value| value.is_finite()),
            Self::F64(values) => values.iter().all(|value| value.is_finite()),
        }
    }

    /// `Fp32` and `Mixed` retain f32 pointwise storage; `Fp64` does not
    /// permit a narrowing storage adapter.
    pub const fn supports_profile(&self, profile: PrecisionProfile) -> bool {
        matches!(
            (self, profile),
            (
                Self::F32(_),
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed
            ) | (Self::F64(_), PrecisionProfile::Fp64)
        )
    }

    pub fn as_f32(&self) -> SemanticResult<&[f32]> {
        match self {
            Self::F32(values) => Ok(values.as_slice()),
            Self::F64(_) => Err(SemanticError::Invalid("numeric column is not f32 storage")),
        }
    }

    pub fn as_f64(&self) -> SemanticResult<&[f64]> {
        match self {
            Self::F32(_) => Err(SemanticError::Invalid("numeric column is not f64 storage")),
            Self::F64(values) => Ok(values.as_slice()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloning_a_column_shares_the_payload_and_preserves_profile_rules() {
        let column = NumericColumn::from(vec![1.0f32, 2.0]);
        let clone = column.shared_clone();
        match (&column, &clone) {
            (NumericColumn::F32(left), NumericColumn::F32(right)) => {
                assert!(Arc::ptr_eq(left, right));
            }
            _ => panic!("f32 storage changed representation"),
        }
        assert!(column.supports_profile(PrecisionProfile::Fp32));
        assert!(column.supports_profile(PrecisionProfile::Mixed));
        assert!(!column.supports_profile(PrecisionProfile::Fp64));
        assert_eq!(column.bytes(), 8);
        assert!(column.finite());
    }

    #[test]
    fn typed_views_fail_closed_without_narrowing() {
        let column = NumericColumn::from(vec![1.0f64, 2.0]);
        assert_eq!(column.as_f64().unwrap(), &[1.0, 2.0]);
        assert!(column.as_f32().is_err());
    }
}
