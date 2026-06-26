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

    fn rebind(&mut self) {
        self.raw.combo_indices = self.combo_indices.as_mut_ptr();
        self.raw.metric_values = self.metric_values.as_mut_ptr();
        self.raw.ranks = self.ranks.as_mut_ptr();
        self.raw.families = self.families.as_mut_ptr();
        self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
        self.raw.row_flags = self.row_flags.as_mut_ptr();
    }
}
