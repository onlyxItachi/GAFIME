use gafime_types::{BackendKind, GAFIME_LAUNCH_FLAG_GRAPH};

use crate::{plan::CompiledPlan, reduce::CompactResultTablePlan, OrchestratorResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleDecision {
    pub backend_kind: BackendKind,
    pub matrix_resident: bool,
    pub graph_requested: bool,
}

impl ScheduleDecision {
    pub fn resident_graph(backend_kind: BackendKind) -> Self {
        Self {
            backend_kind,
            matrix_resident: true,
            graph_requested: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContinuousSchedule {
    decision: ScheduleDecision,
    result_table: CompactResultTablePlan,
}

impl ContinuousSchedule {
    pub fn for_plan(plan: &CompiledPlan) -> OrchestratorResult<Self> {
        Ok(Self {
            decision: ScheduleDecision {
                backend_kind: plan.protocol().backend_kind,
                matrix_resident: true,
                graph_requested: (plan.protocol().flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0,
            },
            result_table: CompactResultTablePlan::for_plan(plan)?,
        })
    }

    pub fn decision(&self) -> ScheduleDecision {
        self.decision
    }

    pub fn result_table(&self) -> CompactResultTablePlan {
        self.result_table
    }
}
