use gafime_types::GafimeResultTable;

use crate::{plan::CompiledPlan, OrchestratorError, OrchestratorResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactResultTablePlan {
    planned_rows: u64,
    capacity: u64,
    max_arity: u32,
    metric_count: u32,
    rank_top_k: u32,
}

impl CompactResultTablePlan {
    pub fn for_plan(plan: &CompiledPlan) -> OrchestratorResult<Self> {
        let planned_rows = plan.planned_row_count();
        let rank = plan.rank();
        if rank.top_k > 0 && !plan.metric_ids().contains(&rank.primary_metric) {
            return Err(OrchestratorError::InvalidPlan(
                "rank primary metric is not in the plan metric set",
            ));
        }
        let capacity = if rank.top_k == 0 {
            planned_rows
        } else {
            planned_rows.min(rank.top_k as u64)
        };
        Ok(Self {
            planned_rows,
            capacity,
            max_arity: plan.max_arity(),
            metric_count: plan.metric_count(),
            rank_top_k: rank.top_k,
        })
    }

    pub fn planned_rows(&self) -> u64 {
        self.planned_rows
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn max_arity(&self) -> u32 {
        self.max_arity
    }

    pub fn metric_count(&self) -> u32 {
        self.metric_count
    }

    pub fn is_rank_compacted(&self) -> bool {
        self.rank_top_k > 0 && self.capacity < self.planned_rows
    }
}

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

    pub fn capacity(&self) -> u64 {
        self.raw.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{
        GafimeArityChunk, GafimeRankSpec, GafimeShapeHint, GAFIME_BACKEND_CPU,
        GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_R2,
    };

    #[test]
    fn result_plan_caps_ranked_output_capacity() {
        let plan = crate::CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            32,
            3,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 2,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });

        let table_plan = CompactResultTablePlan::for_plan(&plan).unwrap();

        assert_eq!(table_plan.planned_rows(), 3);
        assert_eq!(table_plan.capacity(), 2);
        assert!(table_plan.is_rank_compacted());
    }

    #[test]
    fn result_plan_bounds_ten_million_candidate_metadata_without_records() {
        let plan = crate::CompiledPlan::from_parts(
            GAFIME_BACKEND_CPU,
            64,
            10_000_000,
            1,
            Vec::new(),
            vec![GAFIME_METRIC_R2],
            vec![GafimeArityChunk {
                arity: 1,
                family: GAFIME_FAMILY_CONTINUOUS,
                metric_mask: 0,
                shape_hint_index: 0,
                combo_row_offset: 0,
                combo_count: 10_000_000,
                local_chunk_id: 0,
                flags: 0,
                descriptor_offset: 0,
                descriptor_count: 10_000_000,
            }],
            vec![GafimeShapeHint::default()],
            GafimeRankSpec {
                top_k: 64,
                primary_metric: GAFIME_METRIC_R2,
                descending: 1,
                include_ties: 0,
                reserved: [0; 4],
            },
            Default::default(),
        );

        let table_plan = CompactResultTablePlan::for_plan(&plan).unwrap();

        assert_eq!(table_plan.planned_rows(), 10_000_000);
        assert_eq!(table_plan.capacity(), 64);
        assert_eq!(table_plan.max_arity(), 1);
        assert_eq!(table_plan.metric_count(), 1);
    }
}
