use gafime_orchestrator::{OrchestratorError, OrchestratorResult};
use gafime_types::GafimeResultTable;

pub fn top_k_indices(values: &[f32], k: usize, descending: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
        let left_value = values[left];
        let right_value = values[right];
        let ordering = left_value
            .partial_cmp(&right_value)
            .unwrap_or(core::cmp::Ordering::Equal);
        if descending {
            ordering.reverse().then(left.cmp(&right))
        } else {
            ordering.then(left.cmp(&right))
        }
    });
    indices.truncate(k.min(indices.len()));
    indices
}

pub unsafe fn compact_result_table_top_k(
    result: &mut GafimeResultTable,
    metric_index: usize,
    descending: bool,
    top_k: usize,
) -> OrchestratorResult<u64> {
    let row_count = result.row_count as usize;
    if top_k == 0 || row_count == 0 {
        return Ok(result.row_count);
    }
    let metric_count = result.metric_count as usize;
    let max_arity = result.max_arity as usize;
    if metric_index >= metric_count {
        return Err(OrchestratorError::InvalidPlan(
            "rank metric index exceeds result metric count",
        ));
    }
    if result.combo_indices.is_null()
        || result.metric_values.is_null()
        || result.ranks.is_null()
        || result.families.is_null()
        || result.candidate_ids.is_null()
        || result.row_flags.is_null()
    {
        return Err(OrchestratorError::InvalidPlan(
            "cannot rank result table with null buffers",
        ));
    }

    let metric_values =
        core::slice::from_raw_parts(result.metric_values, row_count.saturating_mul(metric_count));
    let rank_values: Vec<f32> = (0..row_count)
        .map(|row| metric_values[row * metric_count + metric_index])
        .collect();
    let selected = top_k_indices(&rank_values, top_k.min(row_count), descending);
    let mut rows = Vec::with_capacity(selected.len());
    for &source_row in &selected {
        rows.push(copy_result_row(result, source_row, max_arity, metric_count));
    }
    for (rank, row) in rows.iter().enumerate() {
        write_result_row(result, rank, row, max_arity, metric_count, rank as u32);
    }
    Ok(selected.len() as u64)
}

#[derive(Clone, Debug, PartialEq)]
struct ResultRow {
    combo: Vec<u32>,
    metrics: Vec<f32>,
    family: u32,
    candidate_id: u64,
    flags: u32,
}

unsafe fn copy_result_row(
    result: &GafimeResultTable,
    row: usize,
    max_arity: usize,
    metric_count: usize,
) -> ResultRow {
    let combo_base = row * max_arity;
    let metric_base = row * metric_count;
    ResultRow {
        combo: core::slice::from_raw_parts(result.combo_indices.add(combo_base), max_arity)
            .to_vec(),
        metrics: core::slice::from_raw_parts(result.metric_values.add(metric_base), metric_count)
            .to_vec(),
        family: *result.families.add(row),
        candidate_id: *result.candidate_ids.add(row),
        flags: *result.row_flags.add(row),
    }
}

unsafe fn write_result_row(
    result: &mut GafimeResultTable,
    row: usize,
    values: &ResultRow,
    max_arity: usize,
    metric_count: usize,
    rank: u32,
) {
    let combo_base = row * max_arity;
    for (slot, value) in values.combo.iter().enumerate() {
        *result.combo_indices.add(combo_base + slot) = *value;
    }
    let metric_base = row * metric_count;
    for (slot, value) in values.metrics.iter().enumerate() {
        *result.metric_values.add(metric_base + slot) = *value;
    }
    *result.ranks.add(row) = rank;
    *result.families.add(row) = values.family;
    *result.candidate_ids.add(row) = values.candidate_id;
    *result.row_flags.add(row) = values.flags;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_stable_on_ties() {
        assert_eq!(top_k_indices(&[0.5, 0.7, 0.7, 0.1], 2, true), vec![1, 2]);
    }

    #[test]
    fn compact_result_table_moves_top_rows_to_front() {
        let mut table = crate::result::OwnedResultTable::new(4, 2, 1);
        {
            let raw = table.raw_mut();
            raw.row_count = 4;
            unsafe {
                for row in 0..4usize {
                    *raw.combo_indices.add(row * 2) = row as u32;
                    *raw.combo_indices.add(row * 2 + 1) = u32::MAX;
                    *raw.metric_values.add(row) = [0.1, 0.9, 0.4, 0.8][row];
                    *raw.families.add(row) = 1;
                    *raw.candidate_ids.add(row) = row as u64;
                    *raw.row_flags.add(row) = 0;
                }
                let written = compact_result_table_top_k(raw, 0, true, 2).unwrap();
                raw.row_count = written;
            }
        }

        assert_eq!(table.raw().row_count, 2);
        assert_eq!(&table.combo_indices()[..4], &[1, u32::MAX, 3, u32::MAX]);
        assert_eq!(&table.metric_values()[..2], &[0.9, 0.8]);
    }
}
