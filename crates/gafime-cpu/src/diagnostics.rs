use gafime_orchestrator::{OrchestratorError, OrchestratorResult};

use crate::matrix::CpuMatrix;

pub const MAX_INTERACTION_ARITY: usize = 5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InteractionDiagnostic {
    pub overflow_row_count: u64,
    pub source_nonfinite: bool,
}

pub fn interaction_diagnostics(
    matrix: &CpuMatrix,
    combo_indices: &[u32],
    max_arity: usize,
    row_count: usize,
) -> OrchestratorResult<Vec<InteractionDiagnostic>> {
    if max_arity == 0 || max_arity > MAX_INTERACTION_ARITY {
        return Err(OrchestratorError::InvalidPlan(
            "interaction diagnostics require max_arity in 1..=5",
        ));
    }
    let expected = row_count
        .checked_mul(max_arity)
        .ok_or(OrchestratorError::InvalidPlan(
            "interaction diagnostic combo shape overflows",
        ))?;
    if combo_indices.len() != expected {
        return Err(OrchestratorError::InvalidPlan(
            "interaction diagnostic combo buffer has invalid length",
        ));
    }

    combo_indices
        .chunks_exact(max_arity)
        .map(|padded_combo| diagnose_combo(matrix, padded_combo))
        .collect()
}

fn diagnose_combo(
    matrix: &CpuMatrix,
    padded_combo: &[u32],
) -> OrchestratorResult<InteractionDiagnostic> {
    let arity = padded_combo
        .iter()
        .position(|&feature| feature == u32::MAX)
        .unwrap_or(padded_combo.len());
    if arity == 0
        || arity > MAX_INTERACTION_ARITY
        || padded_combo[arity..]
            .iter()
            .any(|&feature| feature != u32::MAX)
    {
        return Err(OrchestratorError::InvalidPlan(
            "interaction diagnostic combo padding is malformed",
        ));
    }
    let combo = &padded_combo[..arity];
    if combo.iter().any(|&feature| feature >= matrix.cols()) {
        return Err(OrchestratorError::InvalidPlan(
            "interaction diagnostic feature index is out of bounds",
        ));
    }

    let source_nonfinite = matrix.target_has_nonfinite()
        || combo
            .iter()
            .any(|&feature| matrix.column_has_nonfinite(feature as usize));
    if arity == 1
        || combo
            .iter()
            .any(|&feature| matrix.column_has_nonfinite(feature as usize))
        || prefix_product_is_f32_bounded(matrix, combo)
    {
        return Ok(InteractionDiagnostic {
            overflow_row_count: 0,
            source_nonfinite,
        });
    }

    let mut overflow_row_count = 0u64;
    for row in 0..matrix.rows() as usize {
        let mut product = 1.0f32;
        for &feature in combo {
            let feature = feature as usize;
            let centered = matrix.value(row, feature) - matrix.column_mean(feature);
            product *= centered;
            if !product.is_finite() {
                overflow_row_count += 1;
                break;
            }
        }
    }
    Ok(InteractionDiagnostic {
        overflow_row_count,
        source_nonfinite,
    })
}

fn prefix_product_is_f32_bounded(matrix: &CpuMatrix, combo: &[u32]) -> bool {
    let mut bound = 1.0f64;
    for &feature in combo {
        bound *= f64::from(matrix.column_centered_abs_max(feature as usize));
        if !bound.is_finite() || bound > f64::from(f32::MAX) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_from_columns(columns: &[Vec<f32>], target: Vec<f32>) -> CpuMatrix {
        let rows = target.len();
        let cols = columns.len();
        let features = (0..rows)
            .flat_map(|row| columns.iter().map(move |column| column[row]))
            .collect();
        CpuMatrix::from_row_major(rows as u64, cols as u32, features, target).unwrap()
    }

    fn padded_combo(arity: usize) -> Vec<u32> {
        (0..arity as u32)
            .chain(std::iter::repeat_n(u32::MAX, MAX_INTERACTION_ARITY - arity))
            .collect()
    }

    #[test]
    fn safe_arity_one_through_five_prove_zero_overflow() {
        let values = vec![-10_000.0, -100.0, 100.0, 10_000.0];
        let matrix = matrix_from_columns(
            &std::iter::repeat_n(values, MAX_INTERACTION_ARITY).collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0],
        );
        for arity in 1..=MAX_INTERACTION_ARITY {
            let actual = interaction_diagnostics(&matrix, &padded_combo(arity), 5, 1).unwrap()[0];
            assert_eq!(actual, InteractionDiagnostic::default(), "arity={arity}");
        }
    }

    #[test]
    fn risky_arity_two_through_five_count_partial_and_all_overflow_exactly() {
        for (arity, scale) in [(2, 1.0e20), (3, 1.0e13), (4, 1.0e10), (5, 1.0e8)] {
            let partial = vec![-scale, -1.0, 1.0, scale];
            let matrix = matrix_from_columns(
                &std::iter::repeat_n(partial, arity).collect::<Vec<_>>(),
                vec![0.0; 4],
            );
            assert_eq!(
                interaction_diagnostics(&matrix, &padded_combo(arity), 5, 1).unwrap()[0],
                InteractionDiagnostic {
                    overflow_row_count: 2,
                    source_nonfinite: false,
                },
                "partial overflow arity={arity}"
            );

            let all = vec![-scale, scale];
            let matrix = matrix_from_columns(
                &std::iter::repeat_n(all, arity).collect::<Vec<_>>(),
                vec![0.0; 2],
            );
            assert_eq!(
                interaction_diagnostics(&matrix, &padded_combo(arity), 5, 1).unwrap()[0]
                    .overflow_row_count,
                2,
                "all overflow arity={arity}"
            );
        }
    }

    #[test]
    fn zero_prefix_does_not_hide_later_centered_subtraction_overflow() {
        let matrix = matrix_from_columns(
            &[vec![0.0, 0.0, 0.0], vec![f32::MAX, -f32::MAX, -f32::MAX]],
            vec![0.0; 3],
        );
        assert_eq!(
            interaction_diagnostics(&matrix, &[0, 1], 2, 1).unwrap()[0],
            InteractionDiagnostic {
                overflow_row_count: 1,
                source_nonfinite: false,
            }
        );
    }

    #[test]
    fn source_nonfinite_is_separate_from_finite_input_overflow() {
        let mut source = vec![-1.0e8, 1.0e8];
        source[0] = f32::NAN;
        let matrix = matrix_from_columns(
            &std::iter::repeat_n(source, MAX_INTERACTION_ARITY).collect::<Vec<_>>(),
            vec![0.0, 1.0],
        );
        assert_eq!(
            interaction_diagnostics(&matrix, &padded_combo(5), 5, 1).unwrap()[0],
            InteractionDiagnostic {
                overflow_row_count: 0,
                source_nonfinite: true,
            }
        );

        let finite = vec![-1.0e8, 1.0e8];
        let matrix = matrix_from_columns(
            &std::iter::repeat_n(finite, MAX_INTERACTION_ARITY).collect::<Vec<_>>(),
            vec![f32::NAN, 1.0],
        );
        assert_eq!(
            interaction_diagnostics(&matrix, &padded_combo(5), 5, 1).unwrap()[0],
            InteractionDiagnostic {
                overflow_row_count: 2,
                source_nonfinite: true,
            }
        );
    }
}
