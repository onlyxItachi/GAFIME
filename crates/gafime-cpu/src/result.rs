use gafime_types::{GafimeResultTable, GAFIME_ABI_VERSION};

#[derive(Debug)]
pub struct OwnedResultTable {
    raw: GafimeResultTable,
    combo_indices: Vec<u32>,
    metric_values: Vec<f32>,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
}

impl OwnedResultTable {
    pub fn new(capacity: u64, max_arity: u32, metric_count: u32) -> Self {
        let mut table = Self {
            raw: GafimeResultTable {
                abi_version: GAFIME_ABI_VERSION,
                max_arity,
                metric_count,
                flags: 0,
                capacity,
                row_count: 0,
                combo_indices: core::ptr::null_mut(),
                metric_values: core::ptr::null_mut(),
                ranks: core::ptr::null_mut(),
                families: core::ptr::null_mut(),
                candidate_ids: core::ptr::null_mut(),
                row_flags: core::ptr::null_mut(),
                backend_private: core::ptr::null_mut(),
                reserved: [0; 8],
            },
            combo_indices: vec![u32::MAX; capacity as usize * max_arity as usize],
            metric_values: vec![0.0; capacity as usize * metric_count as usize],
            ranks: vec![0; capacity as usize],
            families: vec![0; capacity as usize],
            candidate_ids: vec![0; capacity as usize],
            row_flags: vec![0; capacity as usize],
        };
        table.rebind();
        table
    }

    pub fn raw(&self) -> &GafimeResultTable {
        &self.raw
    }

    pub fn raw_mut(&mut self) -> &mut GafimeResultTable {
        self.rebind();
        &mut self.raw
    }

    pub fn metric_values(&self) -> &[f32] {
        &self.metric_values
    }

    pub fn combo_indices(&self) -> &[u32] {
        &self.combo_indices
    }

    pub fn ranks(&self) -> &[u32] {
        &self.ranks
    }

    pub fn families(&self) -> &[u32] {
        &self.families
    }

    pub fn candidate_ids(&self) -> &[u64] {
        &self.candidate_ids
    }

    pub fn row_flags(&self) -> &[u32] {
        &self.row_flags
    }

    pub fn row_count(&self) -> usize {
        self.raw.row_count as usize
    }

    pub fn max_arity(&self) -> usize {
        self.raw.max_arity as usize
    }

    pub fn metric_count(&self) -> usize {
        self.raw.metric_count as usize
    }

    pub fn append_rows_from(
        &mut self,
        source: &Self,
        candidate_id_offset: u64,
    ) -> Result<(), &'static str> {
        if source.max_arity() > self.max_arity() {
            return Err("source result arity exceeds destination arity");
        }
        if source.metric_count() != self.metric_count() {
            return Err("source and destination metric widths differ");
        }
        let destination_max_arity = self.max_arity();
        let source_max_arity = source.max_arity();
        let metric_count = self.metric_count();
        let start = self.row_count();
        let end = start
            .checked_add(source.row_count())
            .ok_or("result row count overflows")?;
        if end > self.raw.capacity as usize {
            return Err("source rows exceed destination capacity");
        }

        for source_row in 0..source.row_count() {
            let destination_row = start + source_row;
            let destination_combo = destination_row * destination_max_arity;
            self.combo_indices[destination_combo..destination_combo + destination_max_arity]
                .fill(u32::MAX);
            let source_combo = source_row * source_max_arity;
            self.combo_indices[destination_combo..destination_combo + source_max_arity]
                .copy_from_slice(
                    &source.combo_indices[source_combo..source_combo + source_max_arity],
                );

            let destination_metrics = destination_row * metric_count;
            let source_metrics = source_row * metric_count;
            self.metric_values[destination_metrics..destination_metrics + metric_count]
                .copy_from_slice(
                    &source.metric_values[source_metrics..source_metrics + metric_count],
                );
            self.ranks[destination_row] = destination_row
                .try_into()
                .map_err(|_| "result rank exceeds u32")?;
            self.families[destination_row] = source.families[source_row];
            self.candidate_ids[destination_row] = source.candidate_ids[source_row]
                .checked_add(candidate_id_offset)
                .ok_or("candidate id overflows")?;
            self.row_flags[destination_row] = source.row_flags[source_row];
        }
        self.raw.flags |= source.raw.flags;
        self.raw.row_count = end as u64;
        Ok(())
    }

    fn rebind(&mut self) {
        self.raw.combo_indices = self.combo_indices.as_mut_ptr();
        self.raw.metric_values = self.metric_values.as_mut_ptr();
        self.raw.ranks = self.ranks.as_mut_ptr();
        self.raw.families = self.families.as_mut_ptr();
        self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
        self.raw.row_flags = self.row_flags.as_mut_ptr();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appending_rows_pads_combos_and_offsets_candidate_ids() {
        let mut source = OwnedResultTable::new(2, 1, 1);
        source.combo_indices[..2].copy_from_slice(&[4, 2]);
        source.metric_values[..2].copy_from_slice(&[0.5, 0.25]);
        source.families[..2].fill(1);
        source.candidate_ids[..2].copy_from_slice(&[0, 1]);
        source.raw.row_count = 2;

        let mut destination = OwnedResultTable::new(3, 2, 1);
        destination.append_rows_from(&source, 7).unwrap();

        assert_eq!(destination.row_count(), 2);
        assert_eq!(&destination.combo_indices[..4], &[4, u32::MAX, 2, u32::MAX]);
        assert_eq!(&destination.metric_values[..2], &[0.5, 0.25]);
        assert_eq!(&destination.candidate_ids[..2], &[7, 8]);
        assert_eq!(&destination.ranks[..2], &[0, 1]);
    }
}
