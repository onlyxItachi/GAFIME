use gafime_types::GafimeResultTable;

#[derive(Debug)]
pub struct CompactResultTableView<'a> {
    raw: &'a GafimeResultTable,
}

impl<'a> CompactResultTableView<'a> {
    pub fn new(raw: &'a GafimeResultTable) -> Self {
        Self { raw }
    }

    pub fn row_count(&self) -> u64 {
        self.raw.row_count
    }

    pub fn metric_count(&self) -> u32 {
        self.raw.metric_count
    }
}
